"""(2) Yield prediction and (3) explanations: XGBoost, baselines, OOF, metrics, TreeSHAP, rule tree, dummy check."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def make_xgb(params, seed=0):
    return xgb.XGBRegressor(**params, random_state=seed, n_jobs=4)


def load_xgb(path):
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m


def metrics(y, yhat):
    """rmse = sqrt(mean((y - yhat)^2)), mae = mean(|y - yhat|), mape = mean(|y - yhat| / y) * 100,
    r2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"rmse": float(root_mean_squared_error(y, yhat)), "mae": float(mean_absolute_error(y, yhat)),
            "mape": float(np.mean(np.abs((y - yhat) / y)) * 100),
            "r2": float(r2_score(y, yhat)) if len(y) > 1 else float("nan")}


def bootstrap_ci(y, yhat, n=2000, seed=0, alpha=0.05):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([metrics(y[i], yhat[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))])
    return {m: (float(df[m].quantile(alpha / 2)), float(df[m].quantile(1 - alpha / 2))) for m in ("rmse", "r2")}


# --- training ----------------------------------------------------------------

def cv_rmse(X, y, params, seeds, n_splits=5):
    s = []
    for seed in seeds:
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
            m = make_xgb(params, seed).fit(X.iloc[tr], y.iloc[tr])
            s.append(root_mean_squared_error(y.iloc[te], m.predict(X.iloc[te])))
    return float(np.mean(s))


def search_params(X, y, cfg, seed):
    """Random search over cfg['space']; keep the fixed params unless search beats them by min_gain.
    Returns (params, fixed_cv_rmse, chosen_cv_rmse)."""
    rng = np.random.default_rng(seed)
    fixed = cv_rmse(X, y, cfg["fixed"], cfg["cv_seeds"])
    best, best_rmse = cfg["fixed"], fixed
    for _ in range(cfg["n_iter"]):
        p = {k: v[rng.integers(len(v))] for k, v in cfg["space"].items()}
        r = cv_rmse(X, y, p, cfg["cv_seeds"])
        if r < best_rmse:
            best, best_rmse = p, r
    if best_rmse >= fixed * (1 - cfg["min_gain"]):
        return cfg["fixed"], fixed, fixed
    return best, fixed, best_rmse


def baselines(X, y):
    return {
        "mean": DummyRegressor().fit(X, y),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40))).fit(X, y),
        "offline_p": make_pipeline(ColumnTransformer([("p", "passthrough", ["P_off_last"])]),
                                   LinearRegression()).fit(X, y),  # "current value extended"
    }


def oof_predict(X, y, params, fold, seed):
    """fold k model (trained on the other folds) predicts fold k. Returns predictions and the models."""
    pred, models = pd.Series(np.nan, index=X.index), {}
    for k in sorted(fold.unique()):
        te = fold == k
        models[int(k)] = make_xgb(params, seed).fit(X[~te], y[~te])
        pred[te] = models[int(k)].predict(X[te])
    return pred, models


# --- explanations ------------------------------------------------------------

def shap_values(model, X):
    """Exact TreeSHAP via xgboost pred_contribs; '_base' is the expected value."""
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(c, index=X.index, columns=[*X.columns, "_base"])


def top_k_drivers(shap_row, x_row, k=3):
    s = shap_row.drop("_base", errors="ignore")
    return [{"feature": f, "value": float(x_row[f]), "shap": float(s[f]), "direction": "+" if s[f] >= 0 else "-"}
            for f in s.abs().sort_values(ascending=False).index[:k]]


def global_importance(shap_df):
    return shap_df.drop(columns="_base", errors="ignore").abs().mean().sort_values(ascending=False)


def fit_rule_tree(X, y, max_depth=3, min_samples_leaf=5, seed=0):
    return DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed).fit(X, y)


def select_rule_tree(X, y, fold, grid, seed=0):
    """(max_depth, min_samples_leaf) with the best fold-OOF r2 on train; ties go to the fewest leaves,
    then the smallest depth."""
    scores = []
    for d in grid["max_depth"]:
        for leaf in grid["min_samples_leaf"]:
            oof = pd.Series(np.nan, index=X.index)
            for k in fold.unique():
                te = fold == k
                oof[te] = fit_rule_tree(X[~te], y[~te], d, leaf, seed).predict(X[te])
            n_leaves = fit_rule_tree(X, y, d, leaf, seed).get_n_leaves()
            scores.append((metrics(y, oof)["r2"], -n_leaves, -d, leaf))
    r2, _, d, leaf = max(scores)
    return -d, leaf, r2


def export_rules(tree, feature_names, y_ref):
    """One rule per leaf: conditions, mean yield, batch count, high/low versus y_ref."""
    t, rules = tree.tree_, []

    def walk(node, conds):
        if t.children_left[node] == -1:
            m, n = float(t.value[node].ravel()[0]), int(t.n_node_samples[node])
            rules.append({"leaf": int(node), "conditions": conds, "mean_y": m, "n": n,
                          "text": " AND ".join(conds) + f" -> {'high' if m >= y_ref else 'low'} yield"
                                                        f" (mean {m:,.0f} kg, {n} batches)"})
            return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.4g}"])

    walk(0, [])
    return rules


def dummy_check(X, y, params, n_dummy=3, n_seeds=10, flag_frac=0.7):
    """Add noise columns, retrain; share of seeds where a feature's mean |SHAP| <= the best noise column."""
    noise = [f"_noise{i}" for i in range(n_dummy)]
    below = pd.Series(0.0, index=X.columns)
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        Xd = X.assign(**{c: rng.normal(size=len(X)) for c in noise})
        imp = global_importance(shap_values(make_xgb(params, s).fit(Xd, y), Xd))
        below += imp[X.columns] <= imp[noise].max()
    frac = below / n_seeds
    return pd.DataFrame({"frac_below_noise": frac, "meaningless": frac >= flag_frac}).sort_values("frac_below_noise")
