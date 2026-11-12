"""Pipeline CLI: python -m main {prepare,check,warning,quality,cutoffs,forecast,explain,test,cv,serve,all} ...

all = prepare warning quality cutoffs forecast explain cv. `check` runs only on request; `test` (the one-time
evaluation of the old fixed split) is retired.
serve = HTTP API (main.api) on http://127.0.0.1:8000, reads the files the steps wrote.
cv = repeated cross-validation of the whole pipeline over every normal batch (main.cv, several minutes).
predict = python -m main predict run.csv [--control recipe|operator|apc] [--t H] [--out result.json]:
          a new run (not in the data) through the frozen models (main.predict).
Each step reads the previous step's files in data/cache/ and artifacts/ and prints its results.
"""
import argparse
import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from main import forecast as fc
from main.cv import run_cv
from main.data import (ART, DATA, DT, assemble, batch_meta, batch_yield_kg, cfg, feature_cols, load_raw,
                       load_split, load_ts, make_split, names, offline_available, raw_path, read_json, write_json)
from main.model import (baselines, bootstrap_ci, cross_check, dummy_check, export_rules, fit_ridge_clip, fit_rule_tree,
                        global_importance, linear_contribs, load_xgb, make_xgb, metrics, oof_predict, repeated_cv, search_params,
                        select_rule_tree, shap_values)
from main.warning import fit_warning, save_bands, warning_metrics


def _md(df, fmt="{:.3g}"):
    df = df.reset_index()
    rows = [[fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r] for r in df.to_numpy()]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns),
                      *["| " + " | ".join(r) + " |" for r in rows]])


def run_prepare():
    """raw CSV -> data/cache/timeseries.parquet, split.json (incl. the fixed cutoff t_cut)."""
    c = cfg()
    DATA.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)
    ts = load_raw(raw_path("raw_ts"))
    ts.to_parquet(DATA / "timeseries.parquet")
    meta = batch_meta(ts)
    split = make_split(meta, c["split"]["per_group"], c["split"]["n_folds"], c["seed"])
    # one absolute cutoff for every batch: total length is unknown while a batch runs
    split["t_cut"] = round(c["cutoff_frac"] * meta.t_end[split["train"]].median() / DT) * DT
    write_json(DATA / "split.json", split)
    print(f"timeseries {ts.shape}; train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}; "
          f"t_cut = {split['t_cut']} h")


def run_check():
    """Data checks on the prepared time series and the Statistics file."""
    ts = load_ts()
    m = batch_meta(ts)
    normal, fault_window = ts[~ts.batch.map(m.is_fault)], ts[ts.fault_ref == 1]
    y = batch_yield_kg(ts)
    stats = (pd.read_csv(raw_path("raw_stats"))
             .set_axis(["batch", "during", "end", "total", "fault"], axis=1)
             .set_index("batch"))

    def out_rate(d, v, r):
        x = d[v].dropna()
        return float(((x < r.get("lo", -np.inf)) | (x > r.get("hi", np.inf))).mean())

    # 1-9: one value or table per check
    lengths = m.groupby(["group", "is_fault"]).t_end.describe()[["count", "min", "50%", "max"]]
    intervals = sorted({float(v) for v in ts.groupby("batch").t.diff().dropna().round(4).unique()})
    duplicates = int(ts.duplicated(["batch", "t"]).sum())
    faults = fault_window.groupby("batch").agg(onset_h=("t", "min"), end_h=("t", "max"), n_points=("t", "size"))
    groups = m.reset_index().groupby(["group", "is_fault"]).batch.agg(["count", "min", "max"])
    low_var = ", ".join(f"{v} std {ts[v].std():.3g}" for v in ["RPM", "pH", "T"])
    offline = {c: float(ts[ts[c].notna()].groupby("batch").t.diff().median()) for c in names("input", "offline")}
    stats_corr = np.corrcoef(stats.total, y)[0, 1]
    exceed = pd.DataFrame({
        v: {"lo": r.get("lo", ""), "hi": r.get("hi", ""),
            "normal": out_rate(normal, v, r), "fault_window": out_rate(fault_window, v, r)}
        for v, r in cfg()["safety"]["vars"].items() if r["source"] == "li2024"}).T
    yields = pd.DataFrame({"y_kg": y, "group": m.group})[~m.is_fault].groupby("group").y_kg.describe()

    print(f"""# Data checks

1. Batch length (h)
{_md(lengths)}

2. Sampling interval {intervals} h, duplicate time points {duplicates}

3. Fault windows (`fault_ref`; sum over normal batches = {int(normal.fault_ref.sum())})
{_md(faults)}

4. Batch groups
{_md(groups)}

5. Fixed / low-variance variables: {low_var}

6. Offline sampling interval (h): {offline}; used {cfg()["offline_delay_h"]} h after sampling

7. Per-batch correlation, Statistics total yield vs. time-series yield: {stats_corr:.2f}
   time-series yield = P_T * V_T + sum_t P_t * |Fremoved_t| * dt
   (Statistics is labelled kg but its magnitude is g) -> the target is the time-series yield

8. Safety-limit exceedance rate = count(x < lo or x > hi) / count(x)
   normal = time points of normal batches, fault_window = time points while a fault is active
{_md(exceed)}

9. Yield vs. batch length correlation: {y.corr(m.t_end):.2f}
{_md(yields[["mean", "std", "min", "max"]])}""")


def run_warning():
    """Bands (full + per fold), R(t), thresholds from batch-level false-alarm targets on normal batches, frozen."""
    c, w = cfg(), cfg()["warning"]
    ts, split = load_ts(), load_split()
    tune = [b for part in w["tune_on"] for b in split[part]]
    bands, ranges, risk, rule = fit_warning(ts, offline_available(ts), split["train"], split["fold"], tune, c)
    save_bands(ART / "bands.npz", bands)
    risk.to_parquet(DATA / "risk_ts.parquet")

    parts = [p for p in ("train", "val") if split[p]]
    far = {part: {name: warning_metrics(risk, a, [], split[part]) for name, a in signals(risk).items()}
           for part in parts}
    write_json(ART / "thresholds.json", {**rule, "ranges": ranges, "tune_on": w["tune_on"],
                                         "val_far": far.get("val"), "train_far": far["train"]})

    (warn, crit), n, br = rule["cuts"], rule["persist"], rule["band_rule"]
    print(f"R(t) cuts: WARNING >= {warn}, CRITICAL >= {crit}, held {n} points ({n * DT * 60:.0f} min); "
          + (f"band rule: >= {br['n_vars']} vars for {br['consec']} points ({br['consec'] * DT * 60:.0f} min); "
             if br["n_vars"] <= len(c["band"]["vars"]) else "band rule: off (no setting meets the target); ")
          + f"tuned on {' + '.join(w['tune_on'])} normal batches")
    for part in parts:
        nb, ww, cc = len(split[part]), far[part]["combined_warning+"], far[part]["combined_critical"]
        print(f"{part} normal false alarms: WARNING+ {ww['far_batches'] * nb:.0f}/{nb} batches, "
              f"CRITICAL {cc['far_batches'] * nb:.0f}/{nb} batches")


def signals(risk):
    return {"combined_warning+": risk.grade != "STABLE", "combined_critical": risk.grade == "CRITICAL",
            "r_only_warning+": risk.grade_r != "STABLE", "band_only": risk.band_alarm}


def run_quality():
    """Batch table, XGBoost (search on train CV) + fold models, baselines on val (if any; else see `cv`)."""
    c = cfg()
    ts, split = load_ts(), load_split()
    risk = pd.read_parquet(DATA / "risk_ts.parquet")
    table = assemble(ts, risk, split, split["t_cut"])
    feats = feature_cols(table)
    if table[feats].isna().any().any():
        raise ValueError("missing feature values")
    table.to_parquet(DATA / "batch_table.parquet")
    X, y, tr, va = table[feats], table.y, split["train"], split["val"]

    params, fixed_rmse, cv = search_params(X.loc[tr], y.loc[tr], c["xgb"], c["seed"])
    model = make_xgb(params, c["seed"]).fit(X.loc[tr], y.loc[tr])
    model.save_model(ART / "xgb.json")
    _, folds = oof_predict(X.loc[tr], y.loc[tr], params, table.fold.loc[tr], c["seed"])
    for k, m in folds.items():
        m.save_model(ART / f"xgb_fold{k}.json")
    out = {"features": feats, "params": params, "t_cut": split["t_cut"], "cv_rmse": {"fixed": fixed_rmse, "chosen": cv}}
    if not va:
        write_json(ART / "quality.json", out)
        print(f"{len(tr)} train batches, no val set: performance comes from `python -m main cv`")
        return
    val = {"xgboost": metrics(y.loc[va], model.predict(X.loc[va]))}
    for k, m in baselines(X.loc[tr], y.loc[tr]).items():
        val[f"baseline_{k}"] = metrics(y.loc[va], m.predict(X.loc[va]))
    passed = all(val["xgboost"]["rmse"] < m["rmse"] for k, m in val.items() if k != "xgboost")

    pv = pd.DataFrame({"group": table.group.loc[va], "y": y.loc[va], "y_pred": model.predict(X.loc[va])})
    by_group = {g: metrics(d.y, d.y_pred)["r2"] for g, d in pv.groupby("group")}
    write_json(ART / "quality.json", out | {
        "val": val, "val_pass": passed, "val_r2_by_group": by_group,
        "val_ci95": bootstrap_ci(y.loc[va], model.predict(X.loc[va]), seed=c["seed"])})  # 15 batches: wide

    print(f"val ({len(va)} batches)  " + "".join(f"{h:>7}" for h in ("rmse", "mae", "mape", "r2")))
    for name, m in val.items():
        print(f"{name:<19}{m['rmse']:7.0f}{m['mae']:7.0f}{m['mape']:7.1f}{m['r2']:7.3f}")
    print(f"pass: {'yes' if passed else 'no'}")
    print("val r2 by group: " + ", ".join(f"{g} {r:.2f}" for g, r in by_group.items()))


def run_cutoffs():
    """Yield models at several cutoffs (same features as `quality`; clipped ridge at cutoff_linear_fracs,
    XGBoost with the `quality` params elsewhere), for the prediction trend.
    Saves per batch and cutoff the routed prediction (OOF for train) and every fold model's prediction,
    and for the clipped-ridge cutoffs the routed yield contributions (cutoff_contrib.parquet)."""
    c, q = cfg(), read_json(ART / "quality.json")
    ts, split = load_ts(), load_split()
    risk = pd.read_parquet(DATA / "risk_ts.parquet")
    tr, va, feats, params = split["train"], split["val"], q["features"], q["params"]
    t_end = batch_meta(ts).t_end
    rows, tables, preds, contribs = {}, [], [], []
    for frac in c["cutoff_fracs"]:
        h = round(frac * t_end[tr].median() / DT) * DT
        if h >= t_end.min():
            raise ValueError(f"cutoff {h} h is not before the shortest batch ({t_end.min()} h)")
        table = assemble(ts, risk, split, h)
        X, y = table[feats], table.y
        name = "ridge_clip" if frac in c["cutoff_linear_fracs"] else "xgboost"
        fit = fit_ridge_clip if name == "ridge_clip" else (lambda X, y: make_xgb(params, c["seed"]).fit(X, y))
        cv = repeated_cv(X.loc[tr], y.loc[tr], params, c["seed"], fit=fit)
        final = fit(X.loc[tr], y.loc[tr])
        joblib.dump(final, ART / f"cutoff_{h:g}.joblib")  # for `predict` on runs outside the data
        oof, folds = oof_predict(X.loc[tr], y.loc[tr], params, table.fold.loc[tr], c["seed"], fit=fit)
        p = pd.DataFrame({f"fold{k}": m.predict(X) for k, m in folds.items()}, index=X.index)
        p["y_pred"] = final.predict(X)
        p.loc[tr, "y_pred"] = oof  # a train batch is predicted by the fold model that left it out
        # two ranges: spread of the fold models, and +-q with q = a quantile of the train OOF |error|
        p["spread_lo"], p["spread_hi"] = p.filter(like="fold").min(axis=1), p.filter(like="fold").max(axis=1)
        q = float((y.loc[tr] - oof).abs().quantile(c["pred_interval"]))
        p["y_pred_lo"], p["y_pred_hi"] = p.y_pred - q, p.y_pred + q
        preds.append(p.assign(t_cut=h, model=name))
        tables.append(X.assign(t_cut=h))
        if name == "ridge_clip":  # yield contributions, routed like y_pred (train batch -> its fold model)
            rest = X.index.difference(tr)
            parts = [linear_contribs(final, X.loc[rest])] if len(rest) else []
            parts += [linear_contribs(folds[int(k)], X.loc[g.index]) for k, g in table.loc[tr].groupby("fold")]
            contribs.append(pd.concat(parts).assign(t_cut=h))
        lower = cv.r2.mean() - 2 * cv.r2.std() / np.sqrt(len(cv))  # beats the mean beyond repeat noise
        rows[f"{h:g}"] = {"t_cut": h, "model": name, "cv_rmse": float(cv.rmse.mean()), "cv_rmse_sd": float(cv.rmse.std()),
                          "cv_r2": float(cv.r2.mean()), "cv_r2_lower": float(lower),
                          "provided": bool(lower > c["cutoff_min_cv_r2"]), "interval_halfwidth": q}
        if va:
            v, pv, yv = metrics(y.loc[va], final.predict(X.loc[va])), p.loc[va], y.loc[va]
            rows[f"{h:g}"] |= {"val_rmse": v["rmse"], "val_r2": v["r2"],
                               "val_coverage_interval": float(yv.between(pv.y_pred_lo, pv.y_pred_hi).mean()),
                               "spread_width_median": float((pv.spread_hi - pv.spread_lo).median()),
                               "val_coverage_spread": float(yv.between(pv.spread_lo, pv.spread_hi).mean())}
    later_ok = True  # provide from the earliest cutoff after which every cutoff passes
    for r in reversed(rows.values()):
        later_ok = r["provided"] = later_ok and r["provided"]
    pd.concat(tables).to_parquet(DATA / "cutoff_features.parquet")
    pd.concat(preds).to_parquet(DATA / "cutoff_preds.parquet")
    pd.concat(contribs).to_parquet(DATA / "cutoff_contrib.parquet")
    write_json(ART / "cutoffs.json", rows)
    print(f"{'cutoff':>7}{'cv rmse':>10}{'sd':>6}{'cv r2':>8}{'-2se':>8}{'+-range':>9}"
          + (f"{'val rmse':>10}{'val r2':>8}{'cover':>7}" if va else "") + "  provided  model")
    for k, r in rows.items():
        print(f"{k + ' h':>7}{r['cv_rmse']:10.0f}{r['cv_rmse_sd']:6.0f}{r['cv_r2']:8.3f}{r['cv_r2_lower']:8.3f}"
              f"{r['interval_halfwidth']:9.0f}"
              + (f"{r['val_rmse']:10.0f}{r['val_r2']:8.3f}{r['val_coverage_interval']:7.0%}" if va else "")
              + f"  {'yes' if r['provided'] else 'no':<10}{r['model']}")


def run_forecast():
    """P(t + h) forecast, h = 0.5..8 h: final + fold models, 80% range from train OOF |error| per h,
    val comparison with persistence (latest arrived P_off) and its linear extrapolation."""
    c, f = cfg(), cfg()["forecast"]
    ts, split = load_ts(), load_split()
    F = fc.frame(ts)
    cols, H = fc.feature_cols(F), fc.horizons()
    R = fc.rows(ts, F, H, f["every_h"])
    tr, va = R[R.batch.isin(split["train"])].copy(), R[R.batch.isin(split["val"])].copy()
    fc.fit(tr, cols, f["params"], c["seed"]).save_model(ART / "forecast.json")
    fold = tr.batch.map(split["fold"])
    tr["oof"] = np.nan
    for k in range(c["split"]["n_folds"]):
        m = fc.fit(tr[fold != k], cols, f["params"], c["seed"])
        m.save_model(ART / f"forecast_fold{k}.json")
        tr.loc[fold == k, "oof"] = fc.predict(m, tr[fold == k], cols)
    q = (tr.p_true - tr.oof).abs().groupby(tr.h).quantile(f["interval"])
    meta = {"features": cols, "horizons": H, "every_h": f["every_h"],
            "halfwidth": {f"{h:g}": float(v) for h, v in q.items()}}
    if va.empty:
        write_json(ART / "forecast_meta.json", meta)
        print(f"rows: train {len(tr)}, no val set: performance comes from `python -m main cv`")
        return
    va["pred"] = fc.predict(load_xgb(ART / "forecast.json"), va, cols)
    by_h = {}
    for h, d in va.groupby("h"):
        by_h[f"{h:g}"] = {"rmse": metrics(d.p_true, d.pred)["rmse"], "rmse_persistence": metrics(d.p_true, d.base)["rmse"],
                          "rmse_linear": metrics(d.p_true, fc.linear(d))["rmse"],
                          "halfwidth": float(q[h]), "coverage": float(((d.p_true - d.pred).abs() <= q[h]).mean())}
    passed = all(r["rmse"] < min(r["rmse_persistence"], r["rmse_linear"]) for r in by_h.values())
    write_json(ART / "forecast_meta.json", meta | {"val": by_h, "val_pass": passed})
    print(f"rows: train {len(tr)}, val {len(va)} ({f['every_h']} h grid x {len(H)} horizons)")
    print(f"{'h':>5}{'rmse':>8}{'persist':>9}{'linear':>8}{'+-80%':>8}{'cover':>7}  (val, g/L)")
    for h in (0.5, 2, 4, 6, 8):
        r = by_h[f"{h:g}"]
        print(f"{h:5g}{r['rmse']:8.2f}{r['rmse_persistence']:9.2f}{r['rmse_linear']:8.2f}{r['halfwidth']:8.2f}"
              f"{r['coverage']:7.0%}")
    print(f"beats persistence and linear extrapolation at every horizon: {'yes' if passed else 'no'}")


def run_explain():
    """Global SHAP importance (fold models on their own fold), rule tree, dummy-variable check."""
    c, split, q = cfg(), load_split(), read_json(ART / "quality.json")
    table = pd.read_parquet(DATA / "batch_table.parquet")
    X, y, tr, va = table[q["features"]], table.y, split["train"], split["val"]

    sh = pd.concat([shap_values(load_xgb(ART / f"xgb_fold{k}.json"), X.loc[g.index])
                    for k, g in table.loc[tr].groupby("fold")])
    tree_cfg, dummy_cfg = c["tree"], c["dummy"]
    depth, leaf, cv_r2 = select_rule_tree(X.loc[tr], y.loc[tr], table.fold.loc[tr], tree_cfg, c["seed"])
    tree = fit_rule_tree(X.loc[tr], y.loc[tr], depth, leaf, c["seed"])
    r2 = {s: metrics(y.loc[ids], tree.predict(X.loc[ids]))["r2"] for s, ids in (("train", tr), ("val", va)) if ids}
    r2["train_cv"] = cv_r2
    y_ref = float(y.loc[tr].mean())
    rules = export_rules(tree, q["features"], y_ref)
    reference_only = not r2.get("val", cv_r2) > 0  # rules that do not beat the mean are shown as reference only
    write_json(ART / "rules.json", {
        "rules": rules, "r2": r2, "params": {"max_depth": depth, "min_samples_leaf": leaf},
        "reference_only": reference_only,
        "leaf": {str(b): int(l) for b, l in zip(X.index, tree.apply(X))}})
    dc = dummy_check(X.loc[tr], y.loc[tr], q["params"], **dummy_cfg)
    held_out = ([(load_xgb(ART / "xgb.json"), X.loc[va], y.loc[va])] if va else  # else each fold model on its fold
                [(load_xgb(ART / f"xgb_fold{k}.json"), X.loc[g.index], y.loc[g.index])
                 for k, g in table.loc[tr].groupby("fold")])
    cc = cross_check(held_out, X.loc[tr], y.loc[tr], c["seed"])
    write_json(ART / "cross_check.json", cc)

    top = global_importance(sh).head(5)
    meaningless = set(dc.index[dc.meaningless])
    sliders = set(c["sliders"])
    print("top features: " + ", ".join(f"{f} {v:.0f}" for f, v in top.items()) + " (mean |SHAP| kg)")
    print(f"rule tree: max_depth {depth}, min_samples_leaf {leaf}; train r2 {r2['train']:.3f}, "
          f"train cv r2 {cv_r2:.3f}" + (f", val r2 {r2['val']:.3f}" if va else "")
          + (" (reference only)" if reference_only else ""))
    for r in rules:
        print(f"- {r['text']}")
    print(f"SHAP vs permutation on {'val' if va else 'held-out folds'}: top3 overlap {cc['overlap']['top3']:.0%}, top5 {cc['overlap']['top5']:.0%}; "
          f"SHAP direction = train data direction for "
          f"{sum(t['shap_direction'] == t['data_direction'] for t in cc['top'])}/{len(cc['top'])} of the SHAP top 5")
    print(f"dummy check: {len(meaningless)}/{len(dc)} features meaningless "
          f"(sliders: {len(meaningless & sliders)}/{len(sliders)})")


def run_test():
    """One-time test-set evaluation with frozen thresholds and final models, judged against the criteria
    fixed in config.yaml (test_criteria) before the run. Numbers go to artifacts/test_results.json."""
    done, c, split = ART / "test_evaluated.json", cfg(), load_split()
    if done.exists():
        sys.exit(f"{done} exists: the test set is evaluated once. Do not re-run.")
    demo = c["demo_batches"]
    if not demo or not set(demo) <= set(split["test"]):
        sys.exit("set demo_batches (test batches) in config.yaml before the test evaluation")
    th, q = read_json(ART / "thresholds.json"), read_json(ART / "quality.json")  # read only (Threshold Freeze)
    risk, table = pd.read_parquet(DATA / "risk_ts.parquet"), pd.read_parquet(DATA / "batch_table.parquet")
    X, y, tr, te = table[q["features"]], table.y, split["train"], split["test"]
    t_fault = [b for b in te if table.is_fault[b]]
    t_norm = [b for b in te if not table.is_fault[b]]
    parts = {"normal": t_norm, "fault": t_fault}

    # (1) alarms
    warn = pd.DataFrame({k: warning_metrics(risk, a, t_fault, t_norm) for k, a in signals(risk).items()}).T
    # (2) yield at t_cut: final model and baselines
    final = load_xgb(ART / "xgb.json")
    models = {"xgboost": final}
    for k, m in baselines(X.loc[tr], y.loc[tr]).items():
        models[f"baseline_{k}"] = m
    quality = pd.DataFrame({(part, name): metrics(y.loc[ids], m.predict(X.loc[ids]))
                            for part, ids in {**parts, "all": te}.items() for name, m in models.items()}).T
    ci = bootstrap_ci(y.loc[t_norm], final.predict(X.loc[t_norm]), seed=c["seed"])
    # (3) yield trend: every provided cutoff, with the 80% range and the model spread
    cp = pd.read_parquet(DATA / "cutoff_preds.parquet")
    trend = {}
    for h in [r["t_cut"] for r in read_json(ART / "cutoffs.json").values() if r["provided"]]:
        for part, ids in parts.items():
            d, yy = cp[cp.t_cut == h].loc[ids], y.loc[ids]
            trend[f"{part} {h:g} h"] = {**metrics(yy, d.y_pred),
                                        "coverage_range": float(yy.between(d.y_pred_lo, d.y_pred_hi).mean()),
                                        "coverage_spread": float(yy.between(d.spread_lo, d.spread_hi).mean())}
    # (4) P(t + h) forecast against persistence and linear extrapolation
    fm, ts = read_json(ART / "forecast_meta.json"), load_ts()
    fmodel, forecast = load_xgb(ART / "forecast.json"), {}
    for part, ids in parts.items():
        tsp = ts[ts.batch.isin(ids)]
        R = fc.rows(tsp, fc.frame(tsp), fm["horizons"], fm["every_h"])
        R["pred"] = fc.predict(fmodel, R, fm["features"])
        for h, d in R.groupby("h"):
            forecast[f"{part} {h:g}"] = {
                "rmse": metrics(d.p_true, d.pred)["rmse"], "rmse_persistence": metrics(d.p_true, d.base)["rmse"],
                "rmse_linear": metrics(d.p_true, fc.linear(d))["rmse"],
                "coverage": float(((d.p_true - d.pred).abs() <= fm["halfwidth"][f"{h:g}"]).mean())}

    # (5) verdicts on test normal batches; the criteria were fixed before the run
    k = c["test_criteria"]
    qn, wn, cn = quality.loc["normal"], warn.loc["combined_warning+"], warn.loc["combined_critical"]
    trn = [v for key, v in trend.items() if key.startswith("normal")]
    fn = [forecast[f"normal {h:g}"] for h in fm["horizons"]]
    best_base = qn.rmse.drop("xgboost").min()
    wins = [r["rmse"] < min(r["rmse_persistence"], r["rmse_linear"]) for r in fn]
    cover = [r["coverage"] for r in fn]
    verdicts = {
        "normal batches with WARNING+": (wn["far_batches"], wn["far_batches"] <= k["normal_warning_max"]),
        "normal batches with CRITICAL": (cn["far_batches"], cn["far_batches"] <= k["normal_critical_max"]),
        "yield RMSE minus best baseline (kg)": (qn.rmse["xgboost"] - best_base, qn.rmse["xgboost"] < best_base),
        "yield trend: lowest R2 over cutoffs": (min(v["r2"] for v in trn), min(v["r2"] for v in trn) > k["trend_r2_min"]),
        "yield 80% range: lowest coverage": (min(v["coverage_range"] for v in trn),
                                             min(v["coverage_range"] for v in trn) >= k["yield_coverage_min"]),
        "forecast: share of horizons beating both baselines": (sum(wins) / len(wins), all(wins)),
        "forecast 80% range: coverage min-max": ((min(cover), max(cover)), k["forecast_coverage"][0] <= min(cover)
                                                 and max(cover) <= k["forecast_coverage"][1]),
    }
    write_json(ART / "test_results.json", {
        "demo_batches": demo, "n_normal": len(t_norm), "n_fault": len(t_fault), "criteria": k,
        "alarms": warn.to_dict("index"), "yield": {f"{a} {b}": m for (a, b), m in quality.to_dict("index").items()},
        "yield_ci95_normal": ci, "yield_trend": trend, "forecast": forecast,
        "verdicts": {name: {"value": v, "pass": bool(ok)} for name, (v, ok) in verdicts.items()}})
    write_json(done, {"evaluated_at": datetime.now().isoformat(timespec="seconds"), "demo_batches": demo})

    print(f"test: {len(t_norm)} normal + {len(t_fault)} fault batches; demo batches {demo}; "
          f"R(t) cuts {th['cuts']}, band rule {th['band_rule']}")
    for name in ("combined_warning+", "combined_critical"):
        m = warn.loc[name]
        print(f"{name}: detected {m['batch_detection'] * len(t_fault):.0f}/{len(t_fault)} fault batches, "
              f"sensitivity {m['sensitivity_points']:.1%}, median delay after onset {m['delay_h_median']:.1f} h, "
              f"false alarms {m['far_points']:.1%} of points, "
              f"{m['far_batches'] * len(t_norm):.0f}/{len(t_norm)} batches")
    print(f"{'yield':<26}{'rmse':>7}{'r2':>8}")
    for (part, name), m in quality.iterrows():
        print(f"{part + ' ' + name:<26}{m['rmse']:7.0f}{m['r2']:8.3f}")
    print(f"xgboost normal bootstrap 95%: rmse {ci['rmse'][0]:.0f}-{ci['rmse'][1]:.0f} kg, "
          f"r2 {ci['r2'][0]:.3f}-{ci['r2'][1]:.3f}")
    print(f"{'yield trend':<18}{'rmse':>7}{'r2':>8}{'range':>8}{'spread':>8}")
    for key, m in trend.items():
        print(f"{key:<18}{m['rmse']:7.0f}{m['r2']:8.3f}{m['coverage_range']:8.0%}{m['coverage_spread']:8.0%}")
    print(f"{'forecast (g/L)':<16}{'rmse':>7}{'persist':>9}{'linear':>8}{'cover':>7}")
    for part in parts:
        for h in ("0.5", "4", "8"):
            r = forecast[f"{part} {h}"]
            print(f"{part + ' ' + h + ' h':<16}{r['rmse']:7.2f}{r['rmse_persistence']:9.2f}{r['rmse_linear']:8.2f}"
                  f"{r['coverage']:7.0%}")
    print("criteria (test normal batches, fixed before the run):")
    for name, (v, ok) in verdicts.items():
        shown = f"{v[0]:.2f}-{v[1]:.2f}" if isinstance(v, tuple) else f"{v:.3g}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {shown}")


def run_pca():
    """PCA T²/SPE model for the online variables: fit on train batches, save to artifacts/pca.npz.
    Run after `warning` (needs timeseries.parquet and split.json).
    Endpoint: GET /batches/{id}/pca?t="""
    from main.pca import fit_pca, save_pca
    ts, split = load_ts(), load_split()
    state = fit_pca(ts, split["train"], names("input", "online"))
    save_pca(ART / "pca.npz", state)
    nc, evr = state["n_components"], state["explained_variance_ratio"]
    print(f"{nc} PCs explain {evr:.1%} of variance; "
          f"T² limit {state['T2_limit']:.2f}, SPE limit {state['SPE_limit']:.2f} "
          f"(alpha={state['alpha']:.0%}, {len(split['train'])} train batches x {len(state['vars'])} vars)")


def run_serve():
    """HTTP API for the front end (main.api)."""
    import uvicorn
    uvicorn.run("main.api:app", host="127.0.0.1", port=8000)


STEPS = {"prepare": run_prepare, "check": run_check, "warning": run_warning, "quality": run_quality,
         "cutoffs": run_cutoffs, "forecast": run_forecast, "explain": run_explain, "test": run_test,
         "cv": run_cv, "pca": run_pca, "serve": run_serve}
ALL = ["prepare", "warning", "quality", "cutoffs", "forecast", "explain", "cv"]

def run_predict(argv):
    from main.predict import GROUPS, predict, report
    ap = argparse.ArgumentParser(prog="python -m main predict", description="Predict for a new run (no training).")
    ap.add_argument("csv", help="one run from t = 0: column names as in config.yaml (t + every input column)")
    ap.add_argument("--control", choices=GROUPS, default="recipe", help="how the run is controlled")
    ap.add_argument("--t", type=float, help="use rows up to this hour only (default: every row)")
    ap.add_argument("--out", help="also write the full result as JSON here")
    a = ap.parse_args(argv)
    try:
        r = predict(a.csv, a.control, a.t)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(report(r))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["predict"]:
        run_predict(sys.argv[2:])
        sys.exit()
    ap = argparse.ArgumentParser(prog="python -m main", description=__doc__.splitlines()[0])
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    for s in ap.parse_args().steps:
        for step in ALL if s == "all" else [s]:
            print(f"\n===== {step}")
            STEPS[step]()
