"""(4) What-if: slider re-prediction (first-half inputs) and rule what-if on R(t)."""
import numpy as np
import pandas as pd

from main.model import shap_values, top_k_drivers
from main.warning import deviation, risk_score, sustained


# Mass balance: linked += factor x (slider change). Extra sugar feed (L) stays in the vessel, so the
# end-of-half volume (L) and weight (kg) rise with it. Wt factor = feed density, ~1 kg/L assumed.
COUPLED = {"Fs_tot": {"V_last": 1.0, "Wt_last": 1.0}}


def _frame(x_row):
    return x_row.to_frame().T.astype(float)


def _linked(x_row, feature, coupled):
    """{linked feature: factor} for a slider, e.g. Fs_tot -> V_last, Wt_last. Only features the model uses."""
    return {g: k for g, k in (coupled or {}).get(feature, {}).items() if g in x_row.index}


def whatif_predict(model, x_row, changes, bounds, coupled=COUPLED):
    """changes: {feature: relative change, 0.05 = +5%}. New values are clipped to the train [min, max].
    coupled: {feature: {linked: factor}}; linked += factor x (slider change), e.g. added sugar feed (L)
    also raises the end-of-half volume (L) and weight (kg). Linked values are not clipped."""
    x = x_row.astype(float).copy()
    linked = {}
    for f, r in changes.items():
        lo, hi = bounds[f]
        x[f] = float(np.clip(x_row[f] * (1 + r), lo, hi))
        for g, k in _linked(x_row, f, coupled).items():
            x[g] += k * (x[f] - x_row[f])
            linked[g] = f
    before, after = float(model.predict(_frame(x_row))[0]), float(model.predict(_frame(x))[0])
    inputs = {f: {"before": float(x_row[f]), "after": float(x[f])} for f in changes}
    inputs |= {g: {"before": float(x_row[g]), "after": float(x[g]), "linked_to": f} for g, f in linked.items()}
    return {"y_pred_before": before, "y_pred_after": after, "delta": after - before, "inputs": inputs,
            "top3": top_k_drivers(shap_values(model, _frame(x)).iloc[0], x)}


def response_curve(model, x_row, feature, bounds, n=25, coupled=COUPLED):
    """This batch's prediction over the feature's train range, others held (shows the step shape).
    Linked features (coupled) move with the slider, as in whatif_predict."""
    grid = np.linspace(*bounds[feature], n)
    X = pd.concat([_frame(x_row)] * n, ignore_index=True)
    X[feature] = grid
    for g, k in _linked(x_row, feature, coupled).items():
        X[g] = float(x_row[g]) + k * (grid - float(x_row[feature]))
    return {"x": grid.tolist(), "y": model.predict(X).astype(float).tolist()}


def _n_critical(ts, ranges, cuts, persist):
    r = risk_score(deviation(ts, ranges), ranges)
    return int((sustained(r, 0 * r, persist) >= cuts[1]).sum())


def rule_whatif(ts_batch, ranges, cuts, fix_vars="all", persist=1):
    """Pull out-of-range values back to the boundary, recompute R(t), count CRITICAL points (R-based,
    R held >= the cut for `persist` points)."""
    vs = list(ranges) if fix_vars == "all" else list(fix_vars)
    fixed = ts_batch.copy()
    for v in vs:
        fixed[v] = fixed[v].clip(ranges[v]["lo"], ranges[v]["hi"])
    before, after = _n_critical(ts_batch, ranges, cuts, persist), _n_critical(fixed, ranges, cuts, persist)
    return {"vars": vs, "critical_before": before, "critical_after": after,
            "reduction_pct": 100 * (before - after) / before if before else 0.0}


def rule_whatif_by_var(ts_batch, ranges, cuts, persist=1):
    """One variable at a time. Reductions do not add up: a point with several excursions may need all fixed."""
    rows = [{"var": v, **rule_whatif(ts_batch, ranges, cuts, [v], persist)} for v in ranges]
    df = pd.DataFrame(rows).drop(columns="vars")
    df["reduction"] = df.critical_before - df.critical_after
    return df.sort_values("reduction", ascending=False, ignore_index=True)


def grid_search(model, x_row, feature1, feature2, bounds, n=10, max_change=0.2, coupled=COUPLED):
    """n×n grid over feature1 x feature2 within +-max_change of current values (clipped to train bounds).
    Other features and coupled variables adjust as in whatif_predict."""
    def narrow(f):
        cur = float(x_row[f])
        lo, hi = bounds[f]
        return max(lo, cur * (1 - max_change)), min(hi, cur * (1 + max_change))

    g1 = np.linspace(*narrow(feature1), n)
    g2 = np.linspace(*narrow(feature2), n)
    rows = []
    for v1 in g1:
        for v2 in g2:
            x = x_row.astype(float).copy()
            x[feature1] = v1
            x[feature2] = v2
            for g, k in _linked(x_row, feature1, coupled).items():
                x[g] += k * (v1 - float(x_row[feature1]))
            for g, k in _linked(x_row, feature2, coupled).items():
                x[g] += k * (v2 - float(x_row[feature2]))
            rows.append(_frame(x))
    preds = model.predict(pd.concat(rows, ignore_index=True)).astype(float).tolist()
    return {"feature1": feature1, "x1": g1.tolist(),
            "feature2": feature2, "x2": g2.tolist(),
            "y": [preds[i * n:(i + 1) * n] for i in range(n)]}


def optimize(models, x_row, bounds, max_change, penalty=1.0, n=2000, seed=0):
    """Random search over the sliders inside the train range and +-max_change of the current values.
    Score = mean - penalty x std over `models` (the fold models), so conditions the models disagree on lose.
    Returns the current and best conditions, their predictions per model, and the improvement."""
    rng = np.random.default_rng(seed)
    lo = {f: max(b[0], x_row[f] * (1 - max_change)) for f, b in bounds.items()}
    hi = {f: min(b[1], x_row[f] * (1 + max_change)) for f, b in bounds.items()}
    X = pd.concat([_frame(x_row)] * (n + 1), ignore_index=True)  # row 0 = current condition
    for f in bounds:
        if lo[f] < hi[f]:
            X.loc[1:, f] = rng.uniform(lo[f], hi[f], n)
    P = np.column_stack([m.predict(X) for m in models])  # rows x models
    score = P.mean(axis=1) - penalty * P.std(axis=1)
    best = int(np.argmax(score))
    gain = P[best] - P[0]  # per model
    return {"current": {f: float(x_row[f]) for f in bounds}, "best": {f: float(X.loc[best, f]) for f in bounds},
            "change_pct": {f: 100 * (X.loc[best, f] / x_row[f] - 1) if x_row[f] else 0.0 for f in bounds},
            "y_pred_current": float(P[0].mean()), "y_pred_best": float(P[best].mean()),
            "gain_kg": float(gain.mean()), "gain_by_model_kg": gain.tolist(),
            "models_agree": int((gain > 0).sum()), "n_models": len(models),
            "spread_current_kg": float(P[0].std()), "spread_best_kg": float(P[best].std())}
