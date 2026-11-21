"""HTTP API over main.service. Paths follow the FermentIQ mockup (VITE_FERMENTIQ_API_URL); run `python -m main serve`.

    t (hours, optional): only data up to t, as if the batch were still running (Replay)
    GET  /batches?t=               list (mockup BatchSummary rows)
    GET  /batches/{id}?t=          summary + per-time-point series
    GET  /batches/{id}/view?t=     everything for one batch (service.batch_view): normal bands, safe ranges, slider ranges
    GET  /batches/{id}/shap?t=&k=  yield contributions top k (default 5, k=0 for all); empty before the first cutoff
    GET  /batches/{id}/contrib?t=  R(t) contribution per variable at t (the per-time explanation)
    POST /simulate                 {"batch_id": 92, "changes": {"Fpaa_tot": 0.05}} -> run_whatif
    GET  /batches/{id}/events?all=&t= excursions per variable (>= 1 h) during alarms; all=true for every one
    GET  /compare?ids=6,92&vars=DO,pH   batches side by side (one id: a reference batch is added)
    GET  /batches/{id}/forecast?t= penicillin concentration t .. t + 8 h with an 80% range
    GET  /batches/{id}/pca?t=      T²/SPE 시계열 + 95% 관리 한계선 (사전에 `python -m main pca` 실행 필요)
    GET  /batches/{id}/curve?feature=&t=            one slider's yield curve (422 while t < t_cut, as in Replay)
    GET  /batches/{id}/grid?feature1=&feature2=&t=  two sliders, n x n predicted yield (422 while t < t_cut)
    POST /batches/{id}/optimize    {"max_change": 0.2} (optional) -> best slider settings (reference only)
    POST /runs/predict?control=&t=  body: one run as CSV text (main.predict.normalize_run) -> predict()
    POST /runs/predict-batches      body: CSV with a batch column (several runs) -> {runs: [{batch, result | error}]}
    GET  /runs/columns              columns a run needs (t + inputs), for manual entry
    POST /runs/sliders|whatif|curve|grid|optimize  {x: whatif_x of an uploaded run, ...}: what-if for that run
    GET  /model                    model, target, repeated-CV metrics, alarm rule, demo batches
"""
import io

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import service

app = FastAPI(title="CAVA")
# Front ends allowed to call this API from a browser: the Lovable app (and its previews) and local dev.
ALLOWED_ORIGINS = r"https://([a-z0-9-]+\.)*(lovable\.app|lovableproject\.com)|http://(localhost|127\.0\.0\.1)(:\d+)?"
app.add_middleware(CORSMiddleware, allow_origin_regex=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


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


@app.get("/batches/{batch_id}/view")
def view(batch_id: int, t: float | None = None):
    _known(batch_id)
    return _call(service.batch_view, batch_id, t)


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
def curve(batch_id: int, feature: str, t: float | None = None):
    _known(batch_id)
    return _call(service.curve, batch_id, feature, t)


@app.get("/batches/{batch_id}/grid")
def grid(batch_id: int, feature1: str, feature2: str, t: float | None = None):
    _known(batch_id)
    return _call(service.grid, batch_id, feature1, feature2, t)


MAX_CSV_BYTES = 20_000_000  # one run at 0.2 h for 230 h is ~0.3 MB


async def _csv(request):
    body = await request.body()
    if not body or len(body) > MAX_CSV_BYTES:
        raise HTTPException(413 if body else 422, "send CSV text (at most 20 MB)")
    try:
        import pandas as pd
        return pd.read_csv(io.StringIO(body.decode("utf-8-sig")))
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(422, f"not a readable CSV: {e}")


@app.post("/runs/predict")
async def run_predict(request: Request, control: str = "recipe", t: float | None = None):
    """A new run uploaded as CSV text (headers: short names or raw IndPenSim headers): alarm, yield, forecast, series."""
    from main.predict import predict
    df = await _csv(request)
    try:
        return predict(df, control, t)
    except ValueError as e:
        raise HTTPException(422, str(e))


class RunWhatIf(BaseModel):
    """An uploaded run's features at t_cut (POST /runs/predict -> whatif_x) plus what to compute."""
    x: dict[str, float | None]
    changes: dict[str, float] = {}
    feature: str | None = None
    feature1: str | None = None
    feature2: str | None = None
    max_change: float | None = None


@app.post("/runs/sliders")
def run_sliders(req: RunWhatIf):
    return _call(service.run_sliders, req.x)


@app.post("/runs/whatif")
def run_whatif(req: RunWhatIf):
    return _call(service.run_whatif_x, req.x, req.changes)


@app.post("/runs/curve")
def run_curve(req: RunWhatIf):
    if not req.feature:
        raise HTTPException(422, "feature is required")
    return _call(service.curve_x, req.x, req.feature)


@app.post("/runs/grid")
def run_grid(req: RunWhatIf):
    if not (req.feature1 and req.feature2):
        raise HTTPException(422, "feature1 and feature2 are required")
    return _call(service.grid_x, req.x, req.feature1, req.feature2)


@app.post("/runs/optimize")
def run_optimize(req: RunWhatIf):
    return _call(service.optimize_x, req.x, req.max_change)


MAX_RUNS = 50


@app.post("/runs/predict-batches")
async def run_predict_batches(request: Request, control: str = "recipe", t: float | None = None):
    """Several runs in one CSV, split by its batch column ("Batch ref" or batch): one result or error per run."""
    from main.predict import predict, split_runs
    runs = split_runs(await _csv(request))
    if len(runs) > MAX_RUNS:
        raise HTTPException(413, f"{len(runs)} batches in the file; at most {MAX_RUNS} per upload")
    out = []
    for batch, rows in runs:
        try:
            out.append({"batch": batch, "result": predict(rows, control, t)})
        except ValueError as e:
            out.append({"batch": batch, "error": str(e)})
    return {"runs": out}


@app.get("/runs/columns")
def run_columns():
    """Columns one run needs (t + every input), for the manual entry form."""
    from main.predict import input_columns
    return input_columns()
