"""PCA T²/SPE multivariate anomaly detection on the online sensor variables.

Fit on training batch time points (standardized). At each time point:
  T²  = sum_i (score_i² / λ_i)  — distance from origin in PC space (mean shift)
  SPE = ||x - x̂||²              — residual not explained by the PCs (novel pattern)
Control limits = empirical α-quantile over training data.
"""
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(ts, train_batches, vars, n_components=0.95, alpha=0.95):
    """Fit PCA on the online variables of train_batches.
    n_components: float in (0,1) = minimum variance to retain, or int = fixed count.
    alpha: quantile for T² and SPE control limits derived from training data."""
    X = ts[ts.batch.isin(train_batches)][vars].to_numpy(float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(Z)
    scores = pca.transform(Z)
    lambdas = pca.explained_variance_
    T2 = (scores ** 2 / lambdas).sum(axis=1)
    Z_hat = scores @ pca.components_
    SPE = ((Z - Z_hat) ** 2).sum(axis=1)
    return {"vars": np.array(vars), "mean": mean, "scale": scale,
            "components": pca.components_, "explained_variance": lambdas,
            "T2_limit": float(np.quantile(T2, alpha)),
            "SPE_limit": float(np.quantile(SPE, alpha)),
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "alpha": float(alpha)}


def pca_scores(state, ts_batch):
    """T² and SPE per time point (NaN where any variable is missing)."""
    X = ts_batch[state["vars"].tolist()].to_numpy(float)
    Z = (X - state["mean"]) / state["scale"]
    scores = Z @ state["components"].T
    T2 = (scores ** 2 / state["explained_variance"]).sum(axis=1)
    Z_hat = scores @ state["components"]
    SPE = ((Z - Z_hat) ** 2).sum(axis=1)
    missing = np.isnan(X).any(axis=1)
    T2[missing] = np.nan
    SPE[missing] = np.nan
    return T2, SPE


def save_pca(path, state):
    np.savez(path, **state)


def load_pca(path):
    with np.load(path) as z:
        state = {k: z[k] for k in z.files}
    state["vars"] = state["vars"].tolist()
    for k in ("T2_limit", "SPE_limit", "explained_variance_ratio", "alpha"):
        state[k] = float(state[k])
    state["n_components"] = int(state["n_components"])
    return state
