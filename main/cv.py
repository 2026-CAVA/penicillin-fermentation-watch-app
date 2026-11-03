"""(7) Repeated cross-validation over every normal batch (`python -m main cv`).

The normal batches are split into n_splits parts (control groups balanced); for each part the whole pipeline
(bands, safety ranges, alarm cuts, yield models per cutoff, 80% ranges, P(t + h) forecast) is rebuilt from
the other parts and scored on that part and on every fault batch. Repeating with new splits gives a mean
and a range for each number, and every normal batch is scored by a pipeline that never saw it.
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from main import forecast as fc
from main.data import (ART, DATA, DT, batch_meta, batch_yield_kg, build_batch_table, cfg, load_ts, offline_available,
                       write_json)
from main.model import baselines, fit_ridge_clip, make_xgb, metrics
from main.warning import fit_warning, warning_metrics


def _parts(ids, groups, n, seed):
    skf = StratifiedKFold(n, shuffle=True, random_state=seed)
    return [(ids[a], ids[b]) for a, b in skf.split(ids, groups)]


def fit_spline(X, y, s):
    """Additive spline ridge on the top_k |corr| features of (X, y) plus the slider inputs (kept so What-if
    still moves); each feature is flat outside its training range."""
    sliders = list(cfg()["sliders"])
    top = X.drop(columns=sliders).corrwith(y).abs().nlargest(s["top_k"]).index
    return make_pipeline(ColumnTransformer([("x", "passthrough", [*top, *sliders])]),
                         SplineTransformer(n_knots=s["n_knots"], degree=s["degree"]),
                         RidgeCV(alphas=np.logspace(-2, 4, 40))).fit(X, y)


def _yield(tables, y, train, ev, inner, params, seed, q_level, spline):
    """Per cutoff: XGBoost, spline ridge, clipped ridge and baselines fit on train, predictions for ev, 80% range
    (inner OOF |error|) and model spread (inner fold models) for the first three."""
    fits = {"xgboost": lambda X, y: make_xgb(params, seed).fit(X, y), "spline": lambda X, y: fit_spline(X, y, spline),
            "ridge_clip": fit_ridge_clip}
    rows = []
    for h, X in tables.items():
        Xt, yt = X.loc[train], y.loc[train]
        for name, fit in fits.items():
            oof, spread = pd.Series(np.nan, index=train), []
            for k in sorted(set(inner.values())):
                out = np.array([inner[b] == k for b in train])
                m = fit(Xt[~out], yt[~out])
                oof[out] = m.predict(Xt[out])
                spread.append(m.predict(X.loc[ev]))
            q = float((yt - oof).abs().quantile(q_level))
            p = fit(Xt, yt).predict(X.loc[ev])
            spread = np.array(spread)
            rows.append(pd.DataFrame({"batch": ev, "t_cut": h, "model": name, "pred": p, "lo": p - q, "hi": p + q,
                                      "spread_lo": spread.min(axis=0), "spread_hi": spread.max(axis=0)}))
        for name, m in baselines(Xt, yt).items():
            rows.append(pd.DataFrame({"batch": ev, "t_cut": h, "model": f"baseline_{name}",
                                      "pred": m.predict(X.loc[ev])}))
    return pd.concat(rows, ignore_index=True)


def _forecast(R, train, ev_parts, inner, f, seed):
    """Fit on train rows, inner OOF |error| quantile per h, then squared errors and coverage per part and h."""
    cols, tr = R.attrs["cols"], R[R.batch.isin(train)]
    k_of = tr.batch.map(inner)
    oof = pd.Series(np.nan, index=tr.index)
    for k in sorted(set(inner.values())):
        out = k_of == k
        oof[out] = fc.predict(fc.fit(tr[~out], cols, f["params"], seed), tr[out], cols)
    q = (tr.p_true - oof).abs().groupby(tr.h).quantile(f["interval"])
    model = fc.fit(tr, cols, f["params"], seed)
    out = []
    for part, ids in ev_parts.items():
        d = R[R.batch.isin(ids)]
        e = d.p_true - fc.predict(model, d, cols)
        out.append(pd.DataFrame({"part": part, "h": d.h, "se": e ** 2, "se_persistence": (d.p_true - d.base) ** 2,
                                 "se_linear": (d.p_true - fc.linear(d)) ** 2,
                                 "covered": (e.abs() <= d.h.map(q)).astype(float)}))
    return pd.concat(out, ignore_index=True)


def run_cv():
    c, v = cfg(), cfg()["cv"]
    t0 = time.time()
    ts = load_ts()
    rt, meta, y = offline_available(ts), batch_meta(ts), batch_yield_kg(ts)
    normal = meta.index[~meta.is_fault].to_numpy()
    faults = [int(b) for b in meta.index[meta.is_fault]]
    t_end = meta.t_end[normal].median()
    hours = [round(fr * t_end / DT) * DT for fr in c["cutoff_fracs"]]
    params = c["xgb"]["fixed"]
    F = fc.frame(ts)
    R = fc.rows(ts, F, fc.horizons(), c["forecast"]["every_h"])
    R.attrs["cols"] = fc.feature_cols(F)

    alarms, preds, fcast = [], [], []
    for r in range(v["repeats"]):
        for j, (train, ev) in enumerate(_parts(normal, meta.group[normal], v["n_splits"], c["seed"] + r)):
            train, ev = [int(b) for b in train], [int(b) for b in ev]
            inner = {int(b): k for k, (_, part) in enumerate(_parts(np.array(train), meta.group[train],
                                                                     c["split"]["n_folds"], c["seed"]))
                     for b in part}
            _, _, risk, rule = fit_warning(ts, rt, train, inner, train, c)
            for name, a in {"warning+": risk.grade != "STABLE", "critical": risk.grade == "CRITICAL"}.items():
                m = warning_metrics(risk, a, faults, ev)
                alarms.append({"repeat": r, "k": j, "signal": name, "cuts": rule["cuts"],
                               "normal_alarmed": m["far_batches"] * len(ev), "n_normal": len(ev),
                               "fault_detected": m["batch_detection"] * len(faults), "n_fault": len(faults),
                               "delay_h_median": m["delay_h_median"]})
            tables = {h: build_batch_table(rt, risk, meta, h, c["features"]) for h in hours}
            preds.append(_yield(tables, y, train, ev + faults, inner, params, c["seed"], c["pred_interval"], v["spline"])
                         .assign(repeat=r, k=j))
            if r < v["forecast_repeats"]:
                fcast.append(_forecast(R, train, {"normal": ev, "fault": faults}, inner, c["forecast"], c["seed"])
                             .assign(repeat=r, k=j))
            print(f"repeat {r + 1}/{v['repeats']} part {j + 1}/{v['n_splits']}: cuts {rule['cuts']} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(DATA / "cv_preds.parquet")
    fcast = pd.concat(fcast, ignore_index=True) if fcast else pd.DataFrame(columns=["part", "h"])
    summary = summarize(pd.DataFrame(alarms), preds, fcast, y, set(faults))
    write_json(ART / "cv.json", {"config": v, "n_normal": len(normal), "n_fault": len(faults), **summary})
    report(summary)


def _spread(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def summarize(alarms, preds, fcast, y, faults):
    """Per repeat over all its parts, then mean / min / max over repeats."""
    out = {"alarms": {}, "yield": {}, "forecast": {}}
    for sig, d in alarms.groupby("signal"):
        g = d.groupby("repeat")
        out["alarms"][sig] = {
            "normal_alarmed_share": _spread(g.normal_alarmed.sum() / g.n_normal.sum()),
            "fault_detected_share": _spread(g.fault_detected.sum() / g.n_fault.sum()),
            "delay_h_median": _spread(g.delay_h_median.median()),
            "cuts_range": [int(min(c[0] for c in d.cuts)), int(max(c[0] for c in d.cuts))]}
    for (part_name, keep) in (("normal", lambda b: b not in faults), ("fault", lambda b: b in faults)):
        p = preds[preds.batch.map(keep)]
        for (h, model), d in p.groupby(["t_cut", "model"]):
            per = []
            for _, e in d.groupby("repeat"):
                # fault batches are scored once per part: average their predictions within the repeat
                e = e.groupby("batch")[["pred", "lo", "hi", "spread_lo", "spread_hi"]].mean()
                yy = y.loc[e.index]
                row = metrics(yy, e.pred)
                if e.lo.notna().all():  # models with an 80% range
                    row |= {"coverage_range": float(yy.between(e.lo, e.hi).mean()),
                            "coverage_spread": float(yy.between(e.spread_lo, e.spread_hi).mean())}
                per.append(row)
            per = pd.DataFrame(per)
            out["yield"][f"{part_name} {h:g} h {model}"] = {k: _spread(per[k]) for k in per.columns}
    for (part, h), d in fcast.groupby(["part", "h"]):
        g = d.groupby("repeat")
        out["forecast"][f"{part} {h:g}"] = {k: _spread(np.sqrt(g[f"se{s}"].mean())) for k, s in
                                            (("rmse", ""), ("rmse_persistence", "_persistence"),
                                             ("rmse_linear", "_linear"))} | {"coverage": _spread(g.covered.mean())}
    return out


def report(s):
    def mm(x, fmt):
        return f"{fmt.format(x['mean'])} ({fmt.format(x['min'])}-{fmt.format(x['max'])})"

    print("\nmean over repeats (min-max)")
    for sig, a in s["alarms"].items():
        print(f"alarm {sig:<9} normal batches alarmed {mm(a['normal_alarmed_share'], '{:.0%}')}, "
              f"faults detected {mm(a['fault_detected_share'], '{:.0%}')}, "
              f"delay after onset {mm(a['delay_h_median'], '{:.1f}')} h, cuts {a['cuts_range']}")
    print(f"{'yield (normal)':<34}{'rmse':>22}{'r2':>22}{'range cover':>20}")
    for key, m in s["yield"].items():
        if key.startswith("normal"):
            cover = mm(m["coverage_range"], "{:.0%}") if "coverage_range" in m else ""
            print(f"{key[7:]:<34}{mm(m['rmse'], '{:.0f}'):>22}{mm(m['r2'], '{:.2f}'):>22}{cover:>20}")
    print(f"{'forecast (g/L)':<16}{'rmse':>8}{'persist':>9}{'linear':>8}{'cover':>7}")
    for key, m in s["forecast"].items():
        if key.split()[1] in ("0.5", "4", "8"):
            print(f"{key + ' h':<16}{m['rmse']['mean']:8.2f}{m['rmse_persistence']['mean']:9.2f}"
                  f"{m['rmse_linear']['mean']:8.2f}{m['coverage']['mean']:7.0%}")
