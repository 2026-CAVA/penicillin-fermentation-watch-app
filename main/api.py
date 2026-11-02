"""HTTP API over main.service. Paths follow the FermentIQ mockup (VITE_FERMENTIQ_API_URL); run `python -m main serve`.

    GET  /batches                  list (mockup BatchSummary rows)
    GET  /batches/{id}             summary + per-time-point series
    GET  /batches/{id}/shap?t=     yield SHAP top 5 (t is accepted but not used yet: one SHAP per batch)
    POST /simulate                 {"batch_id": 92, "changes": {"Fpaa_tot": 0.05}} -> run_whatif
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


@app.get("/batches")
def batches():
    return [service.summary(b) for b in service._state()["table"].index]


@app.get("/batches/{batch_id}")
def batch(batch_id: int):
    _known(batch_id)
    return service.detail(batch_id)


@app.get("/batches/{batch_id}/shap")
def shap(batch_id: int, t: float | None = None):
    _known(batch_id)
    return service.shap_top(batch_id)


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
