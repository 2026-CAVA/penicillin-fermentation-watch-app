"""(5) The only entry point for the front end. Reads data/cache/ and artifacts/; returns JSON-safe dicts.

    list_batches() -> list[dict]
    batch_view(batch_id) -> dict
    run_whatif(batch_id, changes) -> dict     changes = {"Fpaa_tot": 0.05, ...}  (relative, +5%)
    summary(batch_id), detail(batch_id), shap_top(batch_id) -> FermentIQ mockup shapes (served by main.api)
"""
from functools import cache

import numpy as np
import pandas as pd

from main.data import ART, DATA, DT, cfg, load_split, load_ts, offline_available, read_json
from main.model import load_xgb, shap_values, top_k_drivers
from main.warning import GRADES, band_at, deviation, load_bands, risk_contrib
from main.whatif import response_curve, rule_whatif, rule_whatif_by_var, whatif_predict

TAGS = {"train": "Train batch - predicted by the model trained without this batch",
        "val": "Validation batch - used for model selection and alarm thresholds",
        "test": "Test batch - not used for any model or threshold decision"}
NOTES = {"RPM": "Fixed value (100 rpm)", "S": "Display only (not measurable online in a real plant)"}


@cache
def _state():
    table, q, split = pd.read_parquet(DATA / "batch_table.parquet"), read_json(ART / "quality.json"), load_split()
    X = table[q["features"]]
    return {"ts": load_ts(), "risk": pd.read_parquet(DATA / "risk_ts.parquet"), "table": table, "X": X,
            "th": read_json(ART / "thresholds.json"), "split": split, "bands": load_bands(ART / "bands.npz"),
            "final": load_xgb(ART / "xgb.json"),
            "folds": {k: load_xgb(ART / f"xgb_fold{k}.json") for k in range(cfg()["split"]["n_folds"])},
            "rules": read_json(ART / "rules.json"), "bounds": slider_bounds(X.loc[split["train"]], cfg()["sliders"])}


def slider_bounds(X_train, sliders):
    """Train [min, max] per slider; a slider whose train values are all equal is dropped (nothing to move)."""
    b = {f: (float(X_train[f].min()), float(X_train[f].max())) for f in sliders}
    return {f: (lo, hi) for f, (lo, hi) in b.items() if lo < hi}


def _route(batch_id):
    """Model, band and pred_source: a train batch uses the fold model/band that excluded it."""
    s = _state()
    row = s["table"].loc[batch_id]
    if row.split == "train":
        return s["folds"][int(row.fold)], s["bands"][f"fold{int(row.fold)}"], "oof"
    return s["final"], s["bands"]["full"], "final"


def _lst(a):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(a, float)]


def list_batches():
    return [{"batch_id": int(b), "group": r.group, "is_fault": bool(r.is_fault), "split": r.split, "tag": TAGS[r.split]}
            for b, r in _state()["table"].iterrows()]


def batch_view(batch_id):
    s = _state()
    row, (model, band, source) = s["table"].loc[batch_id], _route(batch_id)
    ranges, cuts, persist = s["th"]["ranges"], s["th"]["cuts"], s["th"]["persist"]
    ts = offline_available(s["ts"][s["ts"].batch == batch_id]).reset_index(drop=True)
    rk = s["risk"][s["risk"].batch == batch_id].reset_index(drop=True)
    x = s["X"].loc[batch_id]
    xf = x.to_frame().T.astype(float)

    mu, sd = band_at(ts, band)
    w, bvars = float(band["width"]), band["vars"].tolist()
    series = {}
    for v in dict.fromkeys([*bvars, *ranges, *NOTES]):
        e = {"values": _lst(ts[v]), "note": NOTES.get(v)}
        if v in bvars:
            j = bvars.index(v)
            e |= {"band_mean": _lst(mu[:, j]), "band_lo": _lst(mu[:, j] - w * sd[:, j]),
                  "band_hi": _lst(mu[:, j] + w * sd[:, j]), "band_flag": rk[f"band_{v}"].astype(int).tolist()}
        if v in ranges:
            e |= {"safe_lo": ranges[v]["lo"], "safe_hi": ranges[v]["hi"], "safe_source": ranges[v]["source"]}
        series[v] = e

    contrib = risk_contrib(deviation(ts, ranges), ranges)[(rk.grade != "STABLE").to_numpy()].mean()
    causes = [{"var": v, "share_pct": float(p)} for v, p in contrib.sort_values(ascending=False).items() if p > 0][:3]
    leaf = s["rules"]["leaf"][str(batch_id)]
    return {
        "batch_id": int(batch_id), "group": row.group, "is_fault": bool(row.is_fault), "split": row.split,
        "tag": TAGS[row.split], "pred_source": source, "t_cut": s["split"]["t_cut"],
        "light_final": rk.grade.iloc[-1], "light_worst": max(rk.grade, key=list(GRADES).index),
        "y_pred_kg": float(model.predict(xf)[0]), "y_true_kg": float(row.y),
        "t": ts.t.tolist(), "series": series,
        "risk": {"R": _lst(rk.R), "grade": rk.grade.tolist(), "band_alarm": rk.band_alarm.astype(bool).tolist(),
                 "cuts": cuts, "alarm_top3": causes},
        "shap_top3": top_k_drivers(shap_values(model, xf).iloc[0], x),
        "rule": {**next(r for r in s["rules"]["rules"] if r["leaf"] == leaf), "tree_r2": s["rules"]["r2"],
                 "reference_only": s["rules"]["reference_only"]},
        "whatif": {
            "note": "Computed from patterns the model learned, not a causal effect. "
                    "Only first-half (0 to t_cut) operation is changed. "
                    "Acid, base, cooling and heating water flows are set by the pH and temperature controllers.",
            "sliders": [{"feature": f, "label": cfg()["sliders"][f], "value": float(x[f]), "min": lo, "max": hi,
                         "curve": response_curve(model, x, f, s["bounds"])}
                        for f, (lo, hi) in s["bounds"].items()],
            "rule_by_var": rule_whatif_by_var(ts, ranges, cuts, persist).to_dict("records"),
            "rule_all": rule_whatif(ts, ranges, cuts, persist=persist),
            "rule_note": "Per-variable reductions do not add up: a time point with several excursions "
                         "may remain critical when only one variable is fixed.",
        },
    }


def run_whatif(batch_id, changes):
    unknown = set(changes) - set(_state()["bounds"])
    if unknown:
        raise ValueError(f"not a slider feature: {sorted(unknown)}")
    model, _, source = _route(batch_id)
    return {"pred_source": source, **whatif_predict(model, _state()["X"].loc[batch_id], changes, _state()["bounds"])}


# --- FermentIQ mockup contract (GET /batches, /batches/{id}, /batches/{id}/shap) ------------------
# Values stay in this process's units: yield kg, score = R(t) 0-100 with `scoreCuts`, temperature K.

STATE = {"STABLE": "normal", "WARNING": "caution", "CRITICAL": "anomaly"}
SERIES = {"pH": "pH", "DO": "DO", "temperature": "T", "feedRate": "Fs", "penicillin": "P_off"}  # mock key -> column


@cache
def _y_pred(batch_id):
    model, _, _ = _route(batch_id)
    return float(model.predict(_state()["X"].loc[[batch_id]].astype(float))[0])


def summary(batch_id):
    """One row of the batch list. A finished batch: elapsedH = plannedEndH = last time point."""
    s = _state()
    rk = s["risk"][s["risk"].batch == batch_id]
    state, t = rk.grade.map(STATE), rk.t.to_numpy()
    first = np.flatnonzero(state.ne(state.shift()).to_numpy())  # start of each run of one state
    ends = [*t[first[1:]], t[-1] + DT]  # a segment ends where the next one starts
    anomaly = t[(state == "anomaly").to_numpy()]
    return {"id": int(batch_id), "runStatus": "completed", "elapsedH": float(t[-1]), "plannedEndH": float(t[-1]),
            "state": STATE[max(rk.grade, key=list(GRADES).index)], "predictedFinal": _y_pred(batch_id),
            "latestScore": float(rk.R.iloc[-1]), "scoreCuts": s["th"]["cuts"],
            "finalMeasured": float(s["table"].y[batch_id]),
            "firstAnomalyH": float(anomaly[0]) if len(anomaly) else None, "anomalyDurationH": len(anomaly) * DT,
            "segments": [{"start": float(t[i]), "end": float(e), "state": state.iloc[i]} for i, e in zip(first, ends)]}


def detail(batch_id):
    """summary + one row per time point. penicillin = offline P as known in real time (4 h delay, held);
    predictedFinal exists from t_cut on (the model predicts once, at t_cut)."""
    s = _state()
    ts = offline_available(s["ts"][s["ts"].batch == batch_id]).reset_index(drop=True)
    rk = s["risk"][s["risk"].batch == batch_id].reset_index(drop=True)
    rows = pd.DataFrame({"t": ts.t, **{k: ts[c] for k, c in SERIES.items()}, "score": rk.R,
                         "state": rk.grade.map(STATE),
                         "predictedFinal": np.where(ts.t >= s["split"]["t_cut"], _y_pred(batch_id), np.nan)})
    rows = rows.astype(object).where(rows.notna(), None)
    return {**summary(batch_id), "intervalH": DT, "series": rows.to_dict("records")}


def shap_top(batch_id, k=5):
    """Yield SHAP (one per batch, not per time point), largest |value| first."""
    model, _, _ = _route(batch_id)
    sh = shap_values(model, _state()["X"].loc[[batch_id]].astype(float)).iloc[0].drop("_base")
    return [{"feature": f, "label": f, "value": float(sh[f])} for f in sh.abs().sort_values(ascending=False).index[:k]]
