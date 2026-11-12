"""온라인 센서 변수에 대한 PCA T²/SPE 다변량 이상 탐지.

학습 배치의 각 시점 데이터를 표준화 후 PCA 훈련. 시점마다 계산:
  T²  = sum_i (score_i² / λ_i)  — PC 공간에서의 거리 (평균 이탈 감지)
  SPE = ||x - x̂||²              — PCA로 설명 안 되는 잔차 (기존에 없던 패턴 감지)
관리 한계선 = 학습 데이터의 경험적 α 분위수.
"""
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(ts, train_batches, vars, n_components=0.95, alpha=0.95):
    """학습 배치의 온라인 변수로 PCA 훈련.
    n_components: (0,1) 실수 = 보존할 최소 분산 비율, 정수 = PC 개수 고정.
    alpha: 학습 데이터 기반 T²/SPE 관리 한계선 분위수."""
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
    """시점마다 T²와 SPE 계산 (변수값이 하나라도 없으면 NaN)."""
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
