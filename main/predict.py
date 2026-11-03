"""(8) Predict for a new run that is not in the data (`python -m main predict run.csv`).

Input: one run's time series from t = 0 up to now (any length), columns named as in config.yaml `columns`
(t and every `input` column; offline columns blank where not sampled). Other columns (P, fault_ref, ...)
are ignored, so nothing the models never see can leak in. Uses only frozen artifacts: no training.
Output (dict, JSON-safe): alarm grade and R(t) now, the yield prediction of the latest cutoff reached with
its 80% range and contributions, the prediction trend, and the P(t + h) forecast path.
"""
import joblib
import numpy as np
import pandas as pd

from main import forecast as fc
from main.data import ART, DT, build_batch_table, cfg, feature_label, names, offline_available, read_json
from main.model import linear_contribs, load_xgb, shap_values
from main.warning import (band_alarm, band_flags, combine, deviation, load_bands, risk_contrib, risk_grade, risk_score,
                          sustained)

GROUPS = ("recipe", "operator", "apc")


def read_run(path_or_df, t=None):
    """The input columns of one run, sorted by t, cut at t; batch = 0, k = time step."""
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df
    need = ["t", *names("input")]
    missing = [c for c in need if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    ts = df[need].sort_values("t", ignore_index=True)
    ts = ts[ts.t <= t + 1e-9] if t is not None else ts
    if ts.empty:
        raise ValueError("no rows up to t")
    if ts.drop(columns=names("input", "offline")).isna().any().any():
        raise ValueError("online columns must have no blanks (only offline columns may be blank)")
    return ts.assign(batch=0, k=np.rint(ts.t / DT).astype(int), fault_ref=0).reset_index(drop=True)


def alarm(ts, rt, th, band):
    """R(t), band flags and grades with the frozen bands, safety ranges and rule (as warning.fit_warning)."""
    ranges, br = th["ranges"], th["band_rule"]
    flags = band_flags(ts, band)
    risk = ts[["batch", "k", "t", "fault_ref"]].join(flags)
    risk["R"] = risk_score(deviation(rt, ranges), ranges)
    risk["band_n"] = (flags != 0).sum(axis=1)
    risk["band_alarm"] = band_alarm(risk.band_n, risk.batch, br["n_vars"], br["consec"])
    risk["grade_r"] = risk_grade(sustained(risk.R, risk.batch, th["persist"]), th["cuts"])
    risk["grade"] = combine(risk.grade_r, risk.band_alarm)
    return risk


def forecast_path(ts, model, meta):
    """P(t + h) for every horizon from the last row of ts, with the 80% range (shared with service.forecast)."""
    row = fc.frame(ts).iloc[[-1]]
    R = pd.concat([row.assign(h=h) for h in meta["horizons"]], ignore_index=True)
    p, now, half = fc.predict(model, R, meta["features"]), float(row.t.iloc[0]), meta["halfwidth"]
    return row, [{"t": round(now + h, 4), "h": h, "p": float(v), "lo": float(v - half[f"{h:g}"]),
                  "hi": float(v + half[f"{h:g}"])} for h, v in zip(meta["horizons"], p)]


def predict(run, control="recipe", t=None, k=5):
    """run: CSV path or DataFrame of one run; control: recipe | operator | apc (how the batch is controlled)."""
    if control not in GROUPS:
        raise ValueError(f"control must be one of {GROUPS}")
    c, th = cfg(), read_json(ART / "thresholds.json")
    ts = read_run(run, t)
    rt = offline_available(ts)
    risk = alarm(ts, rt, th, load_bands(ART / "bands.npz")["full"])
    now = float(ts.t.iloc[-1])

    last = risk.iloc[-1]
    share = risk_contrib(deviation(rt.iloc[[-1]], th["ranges"]), th["ranges"]).iloc[0]
    on = risk.t[risk.grade != "STABLE"]
    out = {"t_now": now, "control": control, "n_points": len(ts),
           "alarm": {"grade": last.grade, "R": float(last.R), "cuts": th["cuts"],
                     "worst": max(risk.grade, key=["STABLE", "WARNING", "CRITICAL"].index),
                     "first_alarm_h": float(on.iloc[0]) if len(on) else None,
                     "R_contrib_now": [{"var": v, "share_pct": float(p)}
                                       for v, p in share.sort_values(ascending=False).items() if p > 0][:3]}}

    meta = pd.DataFrame({"group": [control]}, index=[0])
    feats = read_json(ART / "quality.json")["features"]
    trend, latest = [], None
    for cut in read_json(ART / "cutoffs.json").values():
        h = cut["t_cut"]
        if not cut["provided"] or h > now + 1e-9:
            continue
        X = build_batch_table(rt, risk, meta, h, c["features"])[feats].astype(float)
        if cut["model"] == "ridge_clip" and X.isna().any().any():
            raise ValueError(f"missing feature values at {h:g} h: {list(X.columns[X.isna().any()])}")
        model = joblib.load(ART / f"cutoff_{h:g}.joblib")
        y = float(model.predict(X)[0])
        q = cut["interval_halfwidth"]
        trend.append({"t_cut": h, "model": cut["model"], "y_pred": y, "y_pred_lo": y - q, "y_pred_hi": y + q})
        latest = (h, cut["model"], model, X)
    out["yield_trend"] = trend
    if latest:
        h, name, model, X = latest
        sh = (linear_contribs(model, X) if name == "ridge_clip" else shap_values(model, X)).iloc[0]
        terms = sh.drop("_base")
        out["yield"] = {**trend[-1], "unit": "kg", "base": float(sh["_base"]),
                        "contrib": [{"feature": f, "label": feature_label(f, h), "value": float(terms[f]),
                                     "x": float(X[f].iloc[0])}
                                    for f in terms.abs().sort_values(ascending=False).index[:k]]}
    else:
        out["yield"] = None  # before the first provided cutoff

    fmeta = read_json(ART / "forecast_meta.json")
    row, path = forecast_path(ts, load_xgb(ART / "forecast.json"), fmeta)
    out["forecast"] = {"unit": "g/L", "last_measured": None if pd.isna(row.P_off.iloc[0]) else float(row.P_off.iloc[0]),
                       "path": path}
    return out


def report(r):
    a, y = r["alarm"], r["yield"]
    lines = [f"t = {r['t_now']:g} h ({r['n_points']} points, control {r['control']})",
             f"alarm: {a['grade']} now (R = {a['R']:.0f}, cuts {a['cuts']}), worst {a['worst']}"
             + (f", first alarm at {a['first_alarm_h']:g} h" if a["first_alarm_h"] is not None else ""),
             ]
    if a["R_contrib_now"]:
        lines.append("  R(t) now from: " + ", ".join(f"{d['var']} {d['share_pct']:.0f}%" for d in a["R_contrib_now"]))
    if y is None:
        lines.append(f"yield: not yet (first prediction at {min(read_json(ART / 'cutoffs.json').values(), key=lambda c: c['t_cut'])['t_cut']:g} h)")
    else:
        early = " (early estimate)" if y["model"] == "ridge_clip" else ""
        lines.append(f"yield: {y['y_pred']:.0f} kg (80% range {y['y_pred_lo']:.0f}-{y['y_pred_hi']:.0f}), "
                     f"{y['model']} at {y['t_cut']:g} h{early}")
        lines += [f"  {d['value']:+.0f} kg  {d['label']} = {d['x']:.4g}" for d in y["contrib"]]
    p = r["forecast"]["path"]
    lines.append(f"penicillin: {p[0]['p']:.2f} g/L at +{p[0]['h']:g} h -> {p[-1]['p']:.2f} g/L at +{p[-1]['h']:g} h "
                 f"(80% range {p[-1]['lo']:.2f}-{p[-1]['hi']:.2f})")
    return "\n".join(lines)
