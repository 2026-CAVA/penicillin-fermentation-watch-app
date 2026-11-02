"""(5) The only entry point for the front end. Reads data/cache/ and artifacts/; returns JSON-safe dicts.

    list_batches() -> list[dict]
    batch_view(batch_id) -> dict
    run_whatif(batch_id, changes) -> dict     changes = {"Fpaa_tot": 0.05, ...}  (relative, +5%)
    summary(batch_id), detail(batch_id), shap_top(batch_id) -> FermentIQ mockup shapes (served by main.api)
    model_info() -> dict                      model, target, val metrics, alarm rule, demo batches
    events(batch_id), explanation(batch_id)   excursion list, first-alarm sentence
"""
from functools import cache

import numpy as np
import pandas as pd

from main.data import ART, DATA, DT, cfg, col_meta, feature_label, load_split, load_ts, offline_available, read_json
from main.model import load_xgb, shap_values, top_k_drivers
from main.warning import GRADES, band_at, deviation, load_bands, range_sign, risk_contrib, runs
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


def target():
    t_cut = _state()["split"]["t_cut"]
    return {"name": "Total penicillin yield", "unit": "kg", "predicted_at_h": t_cut,
            "text": f"Total penicillin yield at batch end (kg), predicted from data up to {t_cut:g} h"}


def _meta(v):
    """label, unit, kind of a series; offline values are null until delay_h after the first sample."""
    m = col_meta(v)
    return m | ({"delay_h": cfg()["offline_delay_h"]} if m["kind"] == "offline" else {})


def _drivers(shap_row, x_row, k=3):
    return [{**d, "label": feature_label(d["feature"])} for d in top_k_drivers(shap_row, x_row, k)]


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
    sh = shap_values(model, xf).iloc[0]

    mu, sd = band_at(ts, band)
    w, bvars = float(band["width"]), band["vars"].tolist()
    series = {}
    for v in dict.fromkeys([*bvars, *ranges, *NOTES]):
        e = {"values": _lst(ts[v]), "note": NOTES.get(v), **_meta(v)}
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
        "tag": TAGS[row.split], "pred_source": source, "t_cut": s["split"]["t_cut"], "target": target(),
        "light_final": rk.grade.iloc[-1], "light_worst": max(rk.grade, key=list(GRADES).index),
        "y_pred_kg": float(model.predict(xf)[0]), "y_true_kg": float(row.y),
        "t": ts.t.tolist(), "series": series,
        "risk": {"R": _lst(rk.R), "grade": rk.grade.tolist(), "band_alarm": rk.band_alarm.astype(bool).tolist(),
                 "reason": alarm_reason(rk), "cuts": cuts, "alarm_top3": causes},
        "events": events(batch_id), "explanation": explanation(batch_id),
        "shap_top3": _drivers(sh, x), "shap_base": float(sh["_base"]),
        "shap_all": _drivers(sh, x, k=len(x)),  # every feature, largest |shap| first; base + sum = y_pred_kg
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
    r = whatif_predict(model, _state()["X"].loc[batch_id], changes, _state()["bounds"])
    return {"pred_source": source, **r, "top3": [{**d, "label": feature_label(d["feature"])} for d in r["top3"]]}


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
    meta = {k: {"column": c, **_meta(c)} for k, c in SERIES.items()}
    return {**summary(batch_id), "intervalH": DT, "shapBase": _shap(batch_id)["_base"],
            "explanation": explanation(batch_id)["text"], "seriesMeta": meta,
            "series": rows.to_dict("records")}


@cache
def _shap(batch_id):
    model, _, _ = _route(batch_id)
    return shap_values(model, _state()["X"].loc[[batch_id]].astype(float)).iloc[0].astype(float).to_dict()


def shap_top(batch_id, k=5):
    """Yield SHAP (one per batch, not per time point), largest |value| first; k=None for every feature."""
    sh = pd.Series(_shap(batch_id)).drop("_base")
    return [{"feature": f, "label": feature_label(f), "value": float(sh[f])}
            for f in sh.abs().sort_values(ascending=False).index[:k]]


def model_info():
    """Header line and the numbers behind it (validation set only; the test set is evaluated once, separately)."""
    s = _state()
    q, th, r = read_json(ART / "quality.json"), s["th"], s["rules"]
    v = q["val"]["xgboost"]
    return {"title": f"Plan A · XGBoost · val R² {v['r2']:.2f}", "plan": "Plan A", "model": "XGBoost",
            "target": target(), "n_features": len(q["features"]), "n_train": len(s["split"]["train"]),
            "n_val": len(s["split"]["val"]), "val": q["val"], "val_ci95": q["val_ci95"], "val_pass": q["val_pass"],
            "val_r2_by_group": q["val_r2_by_group"], "val_r2_by_cutoff": q["val_r2_by_cutoff"],
            "alarm": {"cuts": th["cuts"], "persist_points": th["persist"], "persist_h": th["persist"] * DT,
                      "band_rule": th["band_rule"], "val_false_alarms": th["val_far"]},
            "rule_tree": {"r2": r["r2"], "reference_only": r["reference_only"]},
            "cross_check": {**(cc := read_json(ART / "cross_check.json")),
                            "top": [{**t, "label": feature_label(t["feature"])} for t in cc["top"]],
                            "note": f"Validation set ({len(s['split']['val'])} batches): reference only. "
                                    "Directions: +1 = higher value, higher yield; 0 = constant feature."},
            "sliders": list(s["bounds"]), "demo_batches": cfg()["demo_batches"]}


# --- events and alarm explanation (P3) -------------------------------------------------------------

def _batch(batch_id):
    s = _state()
    return (offline_available(s["ts"][s["ts"].batch == batch_id]).reset_index(drop=True),
            s["risk"][s["risk"].batch == batch_id].reset_index(drop=True))


@cache
def events(batch_id, all=False):
    """Excursions per variable that last >= the alarm persistence (1 h): outside the normal band (band
    variables) or outside the safety range (R(t) variables). Sorted by start; end is exclusive.
    all=False keeps only excursions that overlap an alarm (grade above STABLE): normal batches have many
    short, harmless excursions (median 11 per val normal batch)."""
    s = _state()
    ts, rk = _batch(batch_id)
    th, rank = s["th"], list(GRADES).index
    t, n = ts.t.to_numpy(), th["persist"]
    signs = [("band", v, rk[f"band_{v}"].to_numpy()) for v in s["bands"]["full"]["vars"].tolist()]
    signs += [("safe_range", v, sg.to_numpy()) for v, sg in range_sign(ts, th["ranges"]).items()]
    out = []
    for source, v, sign in signs:
        m = col_meta(v)
        for a, b, d in runs(sign, n):
            x = ts[v].iloc[a:b]
            peak = float(x.max() if d > 0 else x.min())
            where = "normal band" if source == "band" else "safety range"
            grade = max(rk.grade.iloc[a:b], key=rank)
            if not all and grade == "STABLE":
                continue
            out.append({"var": v, "label": m["label"], "unit": m["unit"], "source": source,
                        "direction": "high" if d > 0 else "low", "start": round(float(t[a]), 4),
                        "end": round(float(t[b - 1]) + DT, 4), "duration_h": round((b - a) * DT, 4), "peak": peak,
                        "grade": grade,
                        "message": f"{m['label']} {'above' if d > 0 else 'below'} the {where} for {(b - a) * DT:.1f} h "
                                   f"({'max' if d > 0 else 'min'} {peak:.4g} {m['unit']})"})
    return sorted(out, key=lambda e: (e["start"], e["var"]))


def alarm_reason(rk):
    """Per time point: 'risk_score' when R(t) sets the grade, 'band' when only the band rule raised it, else None."""
    return np.where(rk.grade_r != "STABLE", "risk_score", np.where(rk.band_alarm, "band", None)).tolist()


@cache
def explanation(batch_id):
    """One sentence on the first alarm (time, grade, why, main variables) and the worst grade."""
    s = _state()
    ts, rk = _batch(batch_id)
    th = s["th"]
    on = np.flatnonzero((rk.grade != "STABLE").to_numpy())
    h = th["persist"] * DT
    if not len(on):
        return {"text": f"No alarm: the risk score never stayed at or above {th['cuts'][0]} for {h:g} h "
                        f"and the band rule never fired.", "first_alarm_h": None, "reason": None, "causes": []}
    i = on[0]
    g, reason = rk.grade.iloc[i], alarm_reason(rk)[i]
    if reason == "risk_score":
        share = risk_contrib(deviation(ts.iloc[[i]], th["ranges"]), th["ranges"]).iloc[0]
        causes = [v for v, p in share.sort_values(ascending=False).items() if p > 0][:3]
        why = (f"risk score held at or above {th['cuts'][g == 'CRITICAL']} for {h:g} h; main contributors: "
               + ", ".join(f"{col_meta(v)['label']} {share[v]:.0f}%" for v in causes))
    else:
        causes = [v.removeprefix("band_") for v in rk.filter(like="band_").columns
                  if v != "band_n" and v != "band_alarm" and rk[v].iloc[i] != 0]
        br = th["band_rule"]
        why = (f"{br['n_vars']} or more variables outside the normal band for {br['consec'] * DT * 60:.0f} min: "
               + ", ".join(col_meta(v)["label"] for v in causes))
    text = f"First alarm at {rk.t.iloc[i]:g} h ({g}): {why}."
    crit = rk.t[rk.grade == "CRITICAL"]
    if len(crit) and g != "CRITICAL":
        text += f" CRITICAL from {crit.iloc[0]:g} h, {len(crit) * DT:.1f} h in total."
    return {"text": text, "first_alarm_h": float(rk.t.iloc[i]), "reason": reason, "causes": causes}
