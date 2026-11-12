"""HTTP API over main.service. Paths follow the FermentIQ mockup (VITE_FERMENTIQ_API_URL); run `python -m main serve`.

    t (hours, optional): only data up to t, as if the batch were still running (Replay)
    GET  /batches?t=               list (mockup BatchSummary rows)
    GET  /batches/{id}?t=          summary + per-time-point series
    GET  /batches/{id}/shap?t=&k=  yield contributions top k (default 5, k=0 for all); empty before the first cutoff
    GET  /batches/{id}/contrib?t=  R(t) contribution per variable at t (the per-time explanation)
    POST /simulate                 {"batch_id": 92, "changes": {"Fpaa_tot": 0.05}} -> run_whatif
    GET  /batches/{id}/events?all=&t= excursions per variable (>= 1 h) during alarms; all=true for every one
    GET  /compare?ids=6,92&vars=DO,pH   batches side by side (one id: a reference batch is added)
    GET  /batches/{id}/forecast?t= penicillin concentration t .. t + 8 h with an 80% range
    GET  /batches/{id}/pca?t=      T² and SPE time series with 95% control limits (needs `python -m main pca`)
    POST /batches/{id}/optimize    {"max_change": 0.2} (optional) -> best slider settings (reference only)
    GET  /model                    model, target, repeated-CV metrics, alarm rule, demo batches
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import service

app = FastAPI(title="CAVA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _known(batch_id):
    if batch_id not in service._state()["table"].index:
        raise HTTPException(404, f"unknown batch {batch_id}")


def _call(f, *args):
    try:
        return f(*args)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/batches")
def batches(t: float | None = None):
    return [_call(service.summary, b, t) for b in service._state()["table"].index]


@app.get("/batches/{batch_id}")
def batch(batch_id: int, t: float | None = None):
    _known(batch_id)
    return _call(service.detail, batch_id, t)


@app.get("/batches/{batch_id}/shap")
def shap(batch_id: int, t: float | None = None, k: int = 5):
    _known(batch_id)
    return service.shap_top(batch_id, k or None, t)


@app.get("/batches/{batch_id}/contrib")
def contrib(batch_id: int, t: float):
    _known(batch_id)
    return _call(service.contrib, batch_id, t)


@app.get("/batches/{batch_id}/events")
def events(batch_id: int, all: bool = False, t: float | None = None):
    _known(batch_id)
    return _call(service.events, batch_id, all, t)


@app.get("/compare")
def compare(ids: str, vars: str | None = None):
    try:
        bids = [int(b) for b in ids.split(",") if b]
    except ValueError:
        raise HTTPException(422, "ids: comma-separated batch numbers")
    if not bids:
        raise HTTPException(422, "ids: at least one batch")
    for b in bids:
        _known(b)
    try:
        return service.compare(bids, vars.split(",") if vars else None)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/batches/{batch_id}/forecast")
def forecast(batch_id: int, t: float):
    _known(batch_id)
    return _call(service.forecast, batch_id, t)


class Optimize(BaseModel):
    max_change: float | None = None


@app.post("/batches/{batch_id}/optimize")
def optimize(batch_id: int, req: Optimize | None = None):
    _known(batch_id)
    return _call(service.optimize, batch_id, req.max_change if req else None)


@app.get("/model")
def model():
    return service.model_info()


class Simulate(BaseModel):
    batch_id: int
    changes: dict[str, float]


@app.post("/simulate")
def simulate(req: Simulate):
    _known(req.batch_id)
    try:
        return service.run_whatif(req.batch_id, req.changes)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/batches/{batch_id}/pca")
def pca_anomaly(batch_id: int, t: float | None = None):
    _known(batch_id)
    return _call(service.pca_anomaly, batch_id, t)


@app.get("/batches/{batch_id}/curve")
def curve(batch_id: int, feature: str):
    _known(batch_id)
    s = service._state()
    if feature not in s["bounds"]:
        raise HTTPException(422, f"not a slider feature: {feature}")
    model, _, _ = service._route(batch_id)
    from main.whatif import response_curve
    return response_curve(model, s["X"].loc[batch_id], feature, s["bounds"])


@app.get("/batches/{batch_id}/grid")
def grid(batch_id: int, feature1: str, feature2: str):
    _known(batch_id)
    s = service._state()
    for f in (feature1, feature2):
        if f not in s["bounds"]:
            raise HTTPException(422, f"not a slider feature: {f}")
    if feature1 == feature2:
        raise HTTPException(422, "feature1 and feature2 must differ")
    model, _, _ = service._route(batch_id)
    from main.whatif import grid_search
    return grid_search(model, s["X"].loc[batch_id], feature1, feature2, s["bounds"])
