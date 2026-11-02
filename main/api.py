"""HTTP API over main.service. Paths follow the FermentIQ mockup (VITE_FERMENTIQ_API_URL); run `python -m main serve`.

    t (hours, optional): only data up to t, as if the batch were still running (Replay)
    GET  /batches?t=               list (mockup BatchSummary rows)
    GET  /batches/{id}?t=          summary + per-time-point series
    GET  /batches/{id}/shap?t=&k=  yield SHAP top k (default 5, k=0 for all); empty before t_cut
    GET  /batches/{id}/contrib?t=  R(t) contribution per variable at t (the per-time explanation)
    POST /simulate                 {"batch_id": 92, "changes": {"Fpaa_tot": 0.05}} -> run_whatif
    GET  /batches/{id}/events?all=&t= excursions per variable (>= 1 h) during alarms; all=true for every one
    GET  /compare?ids=6,92&vars=DO,pH   batches side by side (one id: a reference batch is added)
    GET  /model                    model, target, validation metrics, alarm rule, demo batches
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
