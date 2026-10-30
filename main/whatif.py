"""(4) What-if: slider re-prediction (first-half inputs) and rule what-if on R(t)."""
import numpy as np
import pandas as pd

from main.model import shap_values, top_k_drivers
from main.warning import deviation, risk_score


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


def _n_critical(ts, ranges, cuts):
    return int((risk_score(deviation(ts, ranges), ranges) >= cuts[1]).sum())


def rule_whatif(ts_batch, ranges, cuts, fix_vars="all"):
    """Pull out-of-range values back to the boundary, recompute R(t), count CRITICAL points (R-based)."""
    vs = list(ranges) if fix_vars == "all" else list(fix_vars)
    fixed = ts_batch.copy()
    for v in vs:
        fixed[v] = fixed[v].clip(ranges[v]["lo"], ranges[v]["hi"])
    before, after = _n_critical(ts_batch, ranges, cuts), _n_critical(fixed, ranges, cuts)
    return {"vars": vs, "critical_before": before, "critical_after": after,
            "reduction_pct": 100 * (before - after) / before if before else 0.0}


def rule_whatif_by_var(ts_batch, ranges, cuts):
    """One variable at a time. Reductions do not add up: a point with several excursions may need all fixed."""
    rows = [{"var": v, **rule_whatif(ts_batch, ranges, cuts, [v])} for v in ranges]
    df = pd.DataFrame(rows).drop(columns="vars")
    df["reduction"] = df.critical_before - df.critical_after
    return df.sort_values("reduction", ascending=False, ignore_index=True)
