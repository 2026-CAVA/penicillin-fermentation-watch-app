"""(5) The only entry point for the front end. Reads data/cache/ and artifacts/; returns JSON-safe dicts.

    list_batches() -> list[dict]
    batch_view(batch_id, t=None) -> dict       t: data up to t only (a running batch)
    run_whatif(batch_id, changes) -> dict     changes = {"Fpaa_tot": 0.05, ...}  (relative, +5%)
    summary(batch_id, t), detail(batch_id, t), shap_top(batch_id, k, t) -> FermentIQ mockup shapes (main.api)
    contrib(batch_id, t) -> dict              R(t) contribution per variable at time t
    pred_trend(batch_id, t) -> list           yield prediction of each cutoff model reached by t
    forecast(batch_id, t) -> dict             penicillin concentration t .. t + 8 h with an 80% range
    optimize(batch_id, max_change) -> dict    slider settings with the best predicted yield (reference only)
    model_info() -> dict                      model, target, repeated-CV metrics, alarm rule, demo batches
    events(batch_id), explanation(batch_id)   excursion list, first-alarm sentence
    compare(ids, vars), reference_batch(id)  series of several batches side by side
"""
from functools import cache

import numpy as np
import pandas as pd

from main import forecast as fc
from main.data import ART, DATA, DT, cfg, col_meta, feature_label, load_split, load_ts, offline_available, read_json
from main.model import load_xgb, shap_values, top_k_drivers
from main.warning import GRADES, band_at, deviation, load_bands, range_sign, risk_contrib, runs
from main.whatif import optimize as optimize_sliders
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
            "rules": read_json(ART / "rules.json"), "bounds": slider_bounds(X.loc[split["train"]], cfg()["sliders"]),
            "trend": _provided(pd.read_parquet(DATA / "cutoff_preds.parquet"), read_json(ART / "cutoffs.json")),
            "early": (_provided(pd.read_parquet(DATA / "cutoff_contrib.parquet"), read_json(ART / "cutoffs.json")),
                      pd.read_parquet(DATA / "cutoff_features.parquet"))}


def _provided(preds, cutoffs):
    """Cutoff predictions of the cutoffs that beat the mean in repeated CV, sorted by t_cut."""
    ok = [c["t_cut"] for c in cutoffs.values() if c["provided"]]
    return preds[preds.t_cut.isin(ok)].sort_values("t_cut")


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


def batch_view(batch_id, t=None):
    """Everything for one batch. t: only data up to t, as if the batch were still running; the yield
    prediction, SHAP, rule and sliders (all built from 0..t_cut) appear once t >= t_cut."""
    s = _state()
    row, (model, band, source) = s["table"].loc[batch_id], _route(batch_id)
    ranges, cuts, persist = s["th"]["ranges"], s["th"]["cuts"], s["th"]["persist"]
    ts, rk = _batch(batch_id, t)
    st = _status(batch_id, float(ts.t.iloc[-1]))
    ready = ts.t.iloc[-1] >= s["split"]["t_cut"]  # the explained model (SHAP, rule, sliders) predicts at t_cut
    x = s["X"].loc[batch_id]
    xf = x.to_frame().T.astype(float)
    sh, xs = _yield_contribs(batch_id, float(ts.t.iloc[-1]))

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
        "running": st["running"], "t_now": float(ts.t.iloc[-1]),
        "light_final": rk.grade.iloc[-1], "light_worst": max(rk.grade, key=list(GRADES).index),
        "y_pred_kg": float(model.predict(xf)[0]) if ready else None, "y_true_kg": st["measured"],
        "pred_trend": pred_trend(batch_id, t), "pred_latest_kg": st["pred"],
        "t": ts.t.tolist(), "series": series,
        "risk": {"R": _lst(rk.R), "grade": rk.grade.tolist(), "band_alarm": rk.band_alarm.astype(bool).tolist(),
                 "reason": alarm_reason(rk), "cuts": cuts, "alarm_top3": causes},
        "events": events(batch_id, t=t), "explanation": explanation(batch_id, t),
        # yield contributions (see _yield_contribs): from t_cut TreeSHAP, base + sum = y_pred_kg; before t_cut
        # the early clipped-ridge terms, base + sum = pred_latest_kg; every feature in shap_all, largest |value| first
        "shap_top3": _drivers(sh, xs) if sh is not None else [],
        "shap_base": float(sh["_base"]) if sh is not None else None,
        "shap_all": _drivers(sh, xs, k=len(xs)) if sh is not None else [],
        "rule": {**next(r for r in s["rules"]["rules"] if r["leaf"] == leaf), "tree_r2": s["rules"]["r2"],
                 "reference_only": s["rules"]["reference_only"]} if ready else None,
        "whatif": {
            "note": "Computed from patterns the model learned, not a causal effect. "
                    "Only first-half (0 to t_cut) operation is changed. "
                    "Acid, base, cooling and heating water flows are set by the pH and temperature controllers.",
            "sliders": [{"feature": f, "label": cfg()["sliders"][f], "value": float(x[f]), "min": lo, "max": hi,
                         "curve": response_curve(model, x, f, s["bounds"])}
                        for f, (lo, hi) in s["bounds"].items()] if ready else [],
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


def _risk(batch_id, t=None):
    """risk_ts rows of a batch up to t (all rows when t is None). Every column there is causal."""
    rk = _state()["risk"]
    rk = rk[rk.batch == batch_id]
    rk = rk if t is None else rk[rk.t <= t + 1e-9]
    if rk.empty:
        raise ValueError(f"t = {t} is before the first time point")
    return rk.reset_index(drop=True)


def _batch(batch_id, t=None):
    """Real-time view (offline values delayed and held) and risk rows, both cut at t."""
    rk = _risk(batch_id, t)
    ts = offline_available(_state()["ts"][_state()["ts"].batch == batch_id]).reset_index(drop=True)
    return ts.iloc[:len(rk)], rk


def pred_trend(batch_id, t=None):
    """Yield prediction of every provided cutoff model reached by t (OOF for train), with two ranges:
    y_pred_lo/hi = +- the 80% quantile of train OOF |error| (covers ~80-90% on val; show this one) and
    spread_lo/hi = min/max of the 5 fold models ("model spread": covers far less, not an error range).
    model: "ridge_clip" (early estimate, before t_cut) or "xgboost"."""
    tr = _state()["trend"]
    tr = tr[(tr.index == batch_id) & ((tr.t_cut <= t + 1e-9) if t is not None else True)]
    cols = ["t_cut", "y_pred", "y_pred_lo", "y_pred_hi", "spread_lo", "spread_hi"]
    return [{k: float(v) for k, v in zip(cols, r)} | {"model": m} for r, m in zip(tr[cols].itertuples(index=False), tr.model)]


def _status(batch_id, now):
    """running while now < the batch end; the end and the measured yield are not known yet. pred = the latest
    cutoff prediction reached by now (None before the first provided cutoff)."""
    s = _state()
    end = float(s["risk"].t[s["risk"].batch == batch_id].max())
    running = now < end - 1e-9
    trend = pred_trend(batch_id, now)
    return {"running": running, "end": None if running else end, "pred": trend[-1]["y_pred"] if trend else None,
            "range": [trend[-1]["y_pred_lo"], trend[-1]["y_pred_hi"]] if trend else None,
            "measured": None if running else float(s["table"].y[batch_id])}


def summary(batch_id, t=None):
    """One row of the batch list as of t (None = the whole batch, completed)."""
    s = _state()
    rk = _risk(batch_id, t)
    state, tt = rk.grade.map(STATE), rk.t.to_numpy()
    st = _status(batch_id, tt[-1])
    first = np.flatnonzero(state.ne(state.shift()).to_numpy())  # start of each run of one state
    ends = [*tt[first[1:]], tt[-1] + DT]  # a segment ends where the next one starts
    anomaly = tt[(state == "anomaly").to_numpy()]
    return {"id": int(batch_id), "runStatus": "running" if st["running"] else "completed",
            "elapsedH": float(tt[-1]), "plannedEndH": st["end"],
            "state": STATE[max(rk.grade, key=list(GRADES).index)], "predictedFinal": st["pred"],
            "predictedRange": st["range"],
            "latestScore": float(rk.R.iloc[-1]), "scoreCuts": s["th"]["cuts"], "finalMeasured": st["measured"],
            "firstAnomalyH": float(anomaly[0]) if len(anomaly) else None,
            "anomalyDurationH": round(len(anomaly) * DT, 4),
            "segments": [{"start": float(tt[i]), "end": round(float(e), 4), "state": state.iloc[i]}
                         for i, e in zip(first, ends)]}


def detail(batch_id, t=None):
    """summary + one row per time point up to t. penicillin = offline P as known in real time (4 h delay,
    held); predictedFinal = the latest cutoff model reached at that time point (null before the first)."""
    s = _state()
    ts, rk = _batch(batch_id, t)
    trend = pred_trend(batch_id, t)
    hs, ys = np.array([p["t_cut"] for p in trend]), np.array([p["y_pred"] for p in trend] + [np.nan])
    i = np.searchsorted(hs, ts.t.to_numpy() + 1e-9, side="right") - 1  # -1 before the first cutoff -> nan
    sh, _ = _yield_contribs(batch_id, float(ts.t.iloc[-1]))
    rows = pd.DataFrame({"t": ts.t, **{k: ts[c] for k, c in SERIES.items()}, "score": rk.R,
                         "state": rk.grade.map(STATE), "predictedFinal": ys[i]})
    rows = rows.astype(object).where(rows.notna(), None)
    meta = {k: {"column": c, **_meta(c)} for k, c in SERIES.items()}
    return {**summary(batch_id, t), "intervalH": DT,
            "shapBase": float(sh["_base"]) if sh is not None else None,
            "explanation": explanation(batch_id, t)["text"], "seriesMeta": meta,
            "series": rows.to_dict("records")}


@cache
def _shap(batch_id):
    model, _, _ = _route(batch_id)
    return shap_values(model, _state()["X"].loc[[batch_id]].astype(float)).iloc[0].astype(float).to_dict()


def _yield_contribs(batch_id, t=None):
    """(contributions incl. '_base', feature values) behind the yield prediction at t: TreeSHAP of the t_cut
    model from t_cut on; before it the clipped-ridge terms of the latest provided early cutoff reached by t
    (base + sum = that early prediction); (None, None) before the first provided cutoff."""
    s = _state()
    if t is None or t >= s["split"]["t_cut"]:
        return pd.Series(_shap(batch_id)), s["X"].loc[batch_id]
    c, x = s["early"]
    c = c[(c.index == batch_id) & (c.t_cut <= t + 1e-9)]
    if c.empty:
        return None, None
    h = c.t_cut.max()
    x = x[(x.index == batch_id) & (x.t_cut == h)]
    return c[c.t_cut == h].iloc[0].drop("t_cut").astype(float), x.iloc[0].drop("t_cut").astype(float)


def shap_top(batch_id, k=5, t=None):
    """Yield contributions (see _yield_contribs; one set per cutoff, not per time point), largest |value|
    first; k=None for every feature. Empty before the first provided cutoff: no yield prediction yet."""
    sh, _ = _yield_contribs(batch_id, t)
    if sh is None:
        return []
    sh = sh.drop("_base")
    return [{"feature": f, "label": feature_label(f), "value": float(sh[f])}
            for f in sh.abs().sort_values(ascending=False).index[:k]]


def contrib(batch_id, t):
    """Why R(t) is what it is at time t: each variable's share of the deviation (the per-time explanation;
    yield SHAP does not change over time)."""
    ts, rk = _batch(batch_id, t)
    share = risk_contrib(deviation(ts.iloc[[-1]], _state()["th"]["ranges"]), _state()["th"]["ranges"]).iloc[0]
    return {"t": float(rk.t.iloc[-1]), "R": float(rk.R.iloc[-1]), "grade": rk.grade.iloc[-1],
            "reason": alarm_reason(rk.iloc[[-1]])[0],
            "contrib": [{"var": v, "label": col_meta(v)["label"], "share_pct": float(p)}
                        for v, p in share.sort_values(ascending=False).items() if p > 0]}


def model_info():
    """Header line and the numbers behind it. Performance = repeated CV over every normal batch (`cv` step:
    the whole pipeline refit without the scored batches); the final models are trained on all of them."""
    s = _state()
    q, th, r, cv = read_json(ART / "quality.json"), s["th"], s["rules"], read_json(ART / "cv.json")
    cut = read_json(ART / "cutoffs.json")
    t_cut = s["split"]["t_cut"]
    used = {f"{c['t_cut']:g}": c["model"] for c in cut.values()}  # model per cutoff, as in `cutoffs`
    r2 = cv["yield"][f"normal {t_cut:g} h xgboost"]["r2"]
    return {"title": f"Plan A · ridge + XGBoost · CV R² {r2['mean']:.2f} at {t_cut:g} h", "plan": "Plan A",
            "model": {"before_t_cut": "ridge (clipped inputs)", "from_t_cut": "XGBoost"},
            "target": target(), "n_features": len(q["features"]), "n_train": len(s["split"]["train"]),
            "cv": {"setup": cv["config"], "n_normal": cv["n_normal"], "n_fault": cv["n_fault"], "alarms": cv["alarms"],
                   "yield": {h: cv["yield"][f"normal {h} h {m}"] for h, m in used.items()}, "forecast": cv["forecast"],
                   "note": "mean / min / max over repeats; each batch scored by a pipeline that never saw it"},
            "cutoffs": cut,
            "alarm": {"cuts": th["cuts"], "persist_points": th["persist"], "persist_h": th["persist"] * DT,
                      "band_rule": th["band_rule"] if th["band_rule"]["n_vars"] <= len(s["bands"]["full"]["vars"]) else None,
                      "tuned_on": th["tune_on"], "train_false_alarms": th["train_far"]},
            "rule_tree": {"r2": r["r2"], "reference_only": r["reference_only"]},
            "cross_check": {**(cc := read_json(ART / "cross_check.json")),
                            "top": [{**t, "label": feature_label(t["feature"])} for t in cc["top"]],
                            "note": "Each fold model on its own held-out fold. "
                                    "Directions: +1 = higher value, higher yield; 0 = constant feature."},
            "forecast": {k: v for k, v in read_json(ART / "forecast_meta.json").items() if k != "features"},
            "sliders": list(s["bounds"]), "demo_batches": cfg()["demo_batches"]}


# --- events and alarm explanation (P3) -------------------------------------------------------------

@cache
def events(batch_id, all=False, t=None):
    """Excursions per variable that last >= the alarm persistence (1 h): outside the normal band (band
    variables) or outside the safety range (R(t) variables). Sorted by start; end is exclusive.
    all=False keeps only excursions that overlap an alarm (grade above STABLE): normal batches have many
    short, harmless excursions (median 11 per val normal batch). t: only data up to t (an excursion still
    going on at t ends at t)."""
    s = _state()
    ts, rk = _batch(batch_id, t)
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
def explanation(batch_id, t=None):
    """One sentence on the first alarm (time, grade, why, main variables) and the worst grade, up to t."""
    s = _state()
    ts, rk = _batch(batch_id, t)
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


# --- batch comparison (P3) ---------------------------------------------------------------------------

def reference_batch(batch_id):
    """Default comparison partner: a normal batch of the same control group with the closest length."""
    s = _state()
    m = s["table"].assign(t_end=s["ts"].groupby("batch").t.max())
    c = m[(m.group == m.group[batch_id]) & ~m.is_fault & (m.index != batch_id)]
    return int(min(c.index, key=lambda b: (abs(c.t_end[b] - m.t_end[batch_id]), b)))


def compare(ids, vars=None):
    """Series of several batches side by side; one id gets reference_batch() added. vars: column names or 'R'
    (default: the mockup's five series)."""
    s = _state()
    vars = list(vars or SERIES.values())
    unknown = set(vars) - set(s["ts"].columns) - {"R"}
    if unknown:
        raise ValueError(f"unknown variable: {sorted(unknown)}")
    ids = [*ids, reference_batch(ids[0])] if len(ids) == 1 else list(ids)
    out = []
    for b in ids:
        ts, rk = _batch(b)
        vals = {v: _lst(rk.R if v == "R" else ts[v]) for v in vars}
        out.append({**{k: v for k, v in summary(b).items() if k != "segments"}, "group": s["table"].group[b],
                    "t": ts.t.tolist(), "values": vals})
    meta = {v: ({"label": "Risk score R(t)", "unit": "0-100", "kind": "online"} if v == "R" else _meta(v)) for v in vars}
    return {"vars": meta, "batches": out,
            "reference_rule": "One batch given: the closest-length normal batch of the same control group is added."}


# --- P(t + h) forecast (P7) ------------------------------------------------------------------------

@cache
def _fc_models():
    meta = read_json(ART / "forecast_meta.json")
    return meta, load_xgb(ART / "forecast.json"), {k: load_xgb(ART / f"forecast_fold{k}.json")
                                                   for k in range(cfg()["split"]["n_folds"])}


def forecast(batch_id, t):
    """Penicillin concentration path from t to t + 8 h (every 0.5 h) with an 80% range, from data up to t
    only. A train batch is forecast by the fold model that left it out."""
    s = _state()
    meta, final, folds = _fc_models()
    ts = s["ts"][(s["ts"].batch == batch_id) & (s["ts"].t <= t + 1e-9)]
    if ts.empty:
        raise ValueError(f"t = {t} is before the first time point")
    row = fc.frame(ts).iloc[[-1]]
    R = pd.concat([row.assign(h=h) for h in meta["horizons"]], ignore_index=True)
    r = s["table"].loc[batch_id]
    model = folds[int(r.fold)] if r.split == "train" else final
    p = fc.predict(model, R, meta["features"])
    now, half = float(row.t.iloc[0]), meta["halfwidth"]
    return {"t": now, "unit": "g/L", "label": "Penicillin concentration",
            "last_measured": {"value": None if pd.isna(row.P_off.iloc[0]) else float(row.P_off.iloc[0]),
                              "age_h": None if pd.isna(row.P_off_age_h.iloc[0]) else float(row.P_off_age_h.iloc[0])},
            "path": [{"t": round(now + h, 4), "h": h, "p": float(v), "lo": float(v - half[f"{h:g}"]),
                      "hi": float(v + half[f"{h:g}"])} for h, v in zip(meta["horizons"], p)],
            "note": "80% range from out-of-fold errors on train batches; offline P arrives 4 h after sampling."}


# --- optimisation (P8) -----------------------------------------------------------------------------

def optimize(batch_id, max_change=None):
    """Slider settings with the highest predicted yield near this batch's own (train range, +-max_change),
    scored as the fold models' mean minus their spread. Reference only: patterns, not causes."""
    s, o = _state(), cfg()["optimize"]
    mc = o["max_change"] if max_change is None else max_change
    if not 0 < mc <= 1:
        raise ValueError("max_change must be in (0, 1]")
    r = optimize_sliders(list(s["folds"].values()), s["X"].loc[batch_id], s["bounds"], mc, o["penalty"],
                         o["n_samples"], cfg()["seed"])
    half = read_json(ART / "cutoffs.json")[f"{s['split']['t_cut']:g}"]["interval_halfwidth"]
    return {**r, "max_change": mc, "labels": {f: cfg()["sliders"][f] for f in r["best"]},
            "note": f"Reference only. Computed from patterns the fold models learned, not a causal effect; "
                    f"{r['models_agree']}/{r['n_models']} models predict a gain. Acid, base, cooling and heating "
                    f"water flows are set by the pH and temperature controllers. The yield prediction itself is "
                    f"uncertain by about +-{half:.0f} kg (80% range)."}
