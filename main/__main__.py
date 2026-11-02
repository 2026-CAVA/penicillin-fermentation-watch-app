"""Pipeline CLI: python -m main {prepare,check,warning,quality,explain,test,all} ...

all = prepare warning quality explain. `test` (one-time test-set evaluation) and `check` run only on request.
Each step reads the previous step's files in data/cache/ and artifacts/ and prints its results.
"""
import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from main.data import (ART, DATA, DT, assemble, batch_meta, batch_yield_kg, cfg, feature_cols, load_raw,
                       load_split, load_ts, make_split, names, offline_available, raw_path, read_json, write_json)
from main.model import (baselines, bootstrap_ci, dummy_check, export_rules, fit_rule_tree, global_importance,
                        load_xgb, make_xgb, metrics, oof_predict, search_params, select_rule_tree, shap_values)
from main.warning import (band_alarm, band_flags, combine, deviation, fit_band, lowest_cut, resolve_ranges,
                          risk_grade, risk_score, save_bands, sustained, warning_metrics)


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
    """Bands (full + per fold), R(t), thresholds from false-alarm targets on val normal batches, frozen."""
    c, w = cfg(), cfg()["warning"]
    ts, split = load_ts(), load_split()
    rt = offline_available(ts)
    train, bc = split["train"], c["band"]

    def band(batches):
        return fit_band(ts, batches, bc["vars"], bc["k"], bc["min_n"], bc["std_floor_q"])

    bands = {"full": band(train)} | {f"fold{k}": band([b for b in train if split["fold"][b] != k])
                                     for k in range(c["split"]["n_folds"])}
    save_bands(ART / "bands.npz", bands)

    ranges = resolve_ranges(rt[rt.batch.isin(train)], c["safety"])
    flags = pd.concat([band_flags(g, bands[f"fold{split['fold'][b]}"] if b in split["fold"] else bands["full"])
                       for b, g in ts.groupby("batch")]).loc[ts.index]
    risk = ts[["batch", "k", "t", "fault_ref"]].join(flags)
    risk["R"] = risk_score(deviation(rt, ranges), ranges)
    risk["band_n"] = (flags != 0).sum(axis=1)

    v, n = risk.batch.isin(split["val"]), w["persist"]
    held = sustained(risk.R, risk.batch, n)  # R(t) grades count only after n consecutive points

    def batch_rate(alarm):  # share of val normal batches with at least one alarm point
        return alarm[v].groupby(risk.batch[v]).any().mean()

    peak = held[v].groupby(risk.batch[v]).max()
    warn = lowest_cut(peak, w["far_warning"])
    crit = lowest_cut(peak, w["far_critical"], start=warn + 1)
    rule = next(((nv, k) for nv in w["band_n_vars"] for k in w["band_consec"]
                 if batch_rate((held >= warn) | band_alarm(risk.band_n, risk.batch, nv, k)) <= w["far_warning"]),
                (len(bc["vars"]) + 1, 1))  # nothing fits: band never raises the signal
    risk["band_alarm"] = band_alarm(risk.band_n, risk.batch, *rule)
    risk["grade_r"] = risk_grade(held, [warn, crit])
    risk["grade"] = combine(risk.grade_r, risk.band_alarm)
    risk.to_parquet(DATA / "risk_ts.parquet")

    val = {name: warning_metrics(risk, a, [], split["val"]) for name, a in signals(risk).items()}
    write_json(ART / "thresholds.json", {
        "cuts": [warn, crit], "persist": n, "band_rule": {"n_vars": rule[0], "consec": rule[1]},
        "ranges": ranges, "val_far": val})

    n_val, ww, cc = len(split["val"]), val["combined_warning+"], val["combined_critical"]
    print(f"R(t) cuts: WARNING >= {warn}, CRITICAL >= {crit}, held {n} points ({n * DT * 60:.0f} min); "
          f"band rule: >= {rule[0]} vars for {rule[1]} points ({rule[1] * DT * 60:.0f} min)")
    print(f"val normal false alarms: "
          f"WARNING+ {ww['far_points']:.1%} of points, {ww['far_batches'] * n_val:.0f}/{n_val} batches; "
          f"CRITICAL {cc['far_points']:.1%}, {cc['far_batches'] * n_val:.0f}/{n_val} batches")


def signals(risk):
    return {"combined_warning+": risk.grade != "STABLE", "combined_critical": risk.grade == "CRITICAL",
            "r_only_warning+": risk.grade_r != "STABLE", "band_only": risk.band_alarm}


def run_quality():
    """Batch table, XGBoost (search on train CV) + fold models, baselines on val, cutoff sensitivity."""
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
    write_json(ART / "quality.json", {
        "features": feats, "params": params, "t_cut": split["t_cut"],
        "cv_rmse": {"fixed": fixed_rmse, "chosen": cv}})

    val = {"xgboost": metrics(y.loc[va], model.predict(X.loc[va]))}
    for k, m in baselines(X.loc[tr], y.loc[tr]).items():
        val[f"baseline_{k}"] = metrics(y.loc[va], m.predict(X.loc[va]))
    passed = all(val["xgboost"]["rmse"] < m["rmse"] for k, m in val.items() if k != "xgboost")

    med = batch_meta(ts).t_end[tr].median()
    curve = {}
    for frac in sorted({*c["sensitivity_fracs"], c["cutoff_frac"]}):
        t = assemble(ts, risk, split, round(frac * med / DT) * DT)
        Xt = t[feats]
        oof, _ = oof_predict(Xt.loc[tr], y.loc[tr], params, t.fold.loc[tr], c["seed"])
        val_pred = make_xgb(params, c["seed"]).fit(Xt.loc[tr], y.loc[tr]).predict(Xt.loc[va])
        curve[f"{frac * med:.0f} h"] = {"train_oof_r2": metrics(y.loc[tr], oof)["r2"],
                                                     "val_r2": metrics(y.loc[va], val_pred)["r2"]}

    pv = pd.DataFrame({"group": table.group.loc[va], "y": y.loc[va], "y_pred": model.predict(X.loc[va])})
    by_group = {g: metrics(d.y, d.y_pred)["r2"] for g, d in pv.groupby("group")}

    print(f"val ({len(va)} batches)  " + "".join(f"{h:>7}" for h in ("rmse", "mae", "mape", "r2")))
    for name, m in val.items():
        print(f"{name:<19}{m['rmse']:7.0f}{m['mae']:7.0f}{m['mape']:7.1f}{m['r2']:7.3f}")
    print(f"pass: {'yes' if passed else 'no'}")
    print("val r2 by group: " + ", ".join(f"{g} {r:.2f}" for g, r in by_group.items()))
    print("val r2 by cutoff: " + ", ".join(f"{k} {v['val_r2']:.2f}" for k, v in curve.items()))


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
    r2 = {s: metrics(y.loc[ids], tree.predict(X.loc[ids]))["r2"] for s, ids in (("train", tr), ("val", va))}
    r2["train_cv"] = cv_r2
    y_ref = float(y.loc[tr].mean())
    rules = export_rules(tree, q["features"], y_ref)
    reference_only = not r2["val"] > 0  # rules that do not beat the mean on val are shown as reference only
    write_json(ART / "rules.json", {
        "rules": rules, "r2": r2, "params": {"max_depth": depth, "min_samples_leaf": leaf},
        "reference_only": reference_only,
        "leaf": {str(b): int(l) for b, l in zip(X.index, tree.apply(X))}})
    dc = dummy_check(X.loc[tr], y.loc[tr], q["params"], **dummy_cfg)

    top = global_importance(sh).head(5)
    meaningless = set(dc.index[dc.meaningless])
    sliders = set(c["sliders"])
    print("top features: " + ", ".join(f"{f} {v:.0f}" for f, v in top.items()) + " (mean |SHAP| kg)")
    print(f"rule tree: max_depth {depth}, min_samples_leaf {leaf}; train r2 {r2['train']:.3f}, "
          f"train cv r2 {cv_r2:.3f}, val r2 {r2['val']:.3f}{' (reference only)' if reference_only else ''}")
    for r in rules:
        print(f"- {r['text']}")
    print(f"dummy check: {len(meaningless)}/{len(dc)} features meaningless "
          f"(sliders: {len(meaningless & sliders)}/{len(sliders)})")


def run_test():
    """One-time test-set evaluation with frozen thresholds and the final model."""
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

    warn = pd.DataFrame({k: warning_metrics(risk, a, t_fault, t_norm) for k, a in signals(risk).items()}).T
    final = load_xgb(ART / "xgb.json")
    models = {"xgboost": final}
    for k, m in baselines(X.loc[tr], y.loc[tr]).items():
        models[f"baseline_{k}"] = m
    parts = {"normal": t_norm, "fault": t_fault, "all": te}
    quality = pd.DataFrame({(part, name): metrics(y.loc[ids], m.predict(X.loc[ids]))
                            for part, ids in parts.items() for name, m in models.items()}).T
    ci = bootstrap_ci(y.loc[t_norm], final.predict(X.loc[t_norm]), seed=c["seed"])
    write_json(done, {"evaluated_at": datetime.now().isoformat(timespec="seconds"), "demo_batches": demo})
    print(f"test: {len(t_norm)} normal + {len(t_fault)} fault batches; demo batches {demo}; "
          f"R(t) cuts {th['cuts']}, band rule {th['band_rule']}")
    for name in ("combined_warning+", "combined_critical"):
        m = warn.loc[name]
        print(f"{name}: detected {m['batch_detection'] * len(t_fault):.0f}/{len(t_fault)} fault batches, "
              f"sensitivity {m['sensitivity_points']:.1%}, median lead {m['lead_time_h_median']:.1f} h, "
              f"false alarms {m['far_points']:.1%} of points, "
              f"{m['far_batches'] * len(t_norm):.0f}/{len(t_norm)} batches")
    print(f"{'yield':<26}{'rmse':>7}{'r2':>8}")
    for (part, name), m in quality.iterrows():
        print(f"{part + ' ' + name:<26}{m['rmse']:7.0f}{m['r2']:8.3f}")
    print(f"xgboost normal bootstrap 95%: rmse {ci['rmse'][0]:.0f}-{ci['rmse'][1]:.0f} kg, "
          f"r2 {ci['r2'][0]:.3f}-{ci['r2'][1]:.3f}")


STEPS = {"prepare": run_prepare, "check": run_check, "warning": run_warning, "quality": run_quality,
         "explain": run_explain, "test": run_test}
ALL = ["prepare", "warning", "quality", "explain"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m main", description=__doc__.splitlines()[0])
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    for s in ap.parse_args().steps:
        for step in ALL if s == "all" else [s]:
            print(f"\n===== {step}")
            STEPS[step]()
