"""(5) The only entry point for the front end. Reads data/cache/ and artifacts/; returns JSON-safe dicts.

    list_batches() -> list[dict]
    batch_view(batch_id) -> dict
    run_whatif(batch_id, changes) -> dict     changes = {"Fs_tot": 0.05, ...}  (relative, +5%)
"""
from functools import cache

import numpy as np
import pandas as pd

from main.data import ART, DATA, cfg, load_split, load_ts, offline_available, read_json
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
            "rules": read_json(ART / "rules.json"),
            "bounds": {f: (float(X.loc[split["train"], f].min()), float(X.loc[split["train"], f].max()))
                       for f in cfg()["sliders"]}}


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
        "rule": {**next(r for r in s["rules"]["rules"] if r["leaf"] == leaf), "tree_r2": s["rules"]["r2"]},
        "whatif": {
            "note": "Computed from patterns the model learned, not a causal effect. "
                    "Only first-half (0 to t_cut) operation is changed. "
                    "Added sugar feed also raises the end-of-half volume (V_last) and weight (Wt_last).",
            "sliders": [{"feature": f, "label": lab, "value": float(x[f]), "min": s["bounds"][f][0],
                         "max": s["bounds"][f][1], "curve": response_curve(model, x, f, s["bounds"])}
                        for f, lab in cfg()["sliders"].items()],
            "rule_by_var": rule_whatif_by_var(ts, ranges, cuts, persist).to_dict("records"),
            "rule_all": rule_whatif(ts, ranges, cuts, persist=persist),
            "rule_note": "Per-variable reductions do not add up: a time point with several excursions "
                         "may remain critical when only one variable is fixed.",
        },
    }


def run_whatif(batch_id, changes):
    unknown = set(changes) - set(cfg()["sliders"])
    if unknown:
        raise ValueError(f"not a slider feature: {sorted(unknown)}")
    model, _, source = _route(batch_id)
    return {"pred_source": source, **whatif_predict(model, _state()["X"].loc[batch_id], changes, _state()["bounds"])}
