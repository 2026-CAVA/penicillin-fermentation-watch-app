"""온라인 센서 변수에 대한 PCA T²/SPE 다변량 이상 탐지.

시점 k별 평균/표준편차로 표준화 후 PCA 훈련 (시간에 따른 추세 제거).
시점마다 계산:
  T²  = sum_i (score_i² / λ_i)  — PC 공간에서의 거리 (평균 이탈 감지)
  SPE = ||x - x̂||²              — PCA로 설명 안 되는 잔차 (기존에 없던 패턴 감지)
관리 한계선 = 학습 데이터의 경험적 α 분위수.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(ts, train_batches, vars, n_components=0.95, alpha=0.95, std_floor_q=0.1, min_n=10):
    """학습 배치의 온라인 변수로 PCA 훈련 (시점 k별 표준화).
    n_components: (0,1) 실수 = 보존할 최소 분산 비율, 정수 = PC 개수 고정.
    alpha: 학습 데이터 기반 T²/SPE 관리 한계선 분위수.
    std_floor_q: 거의 상수인 변수의 std 하한선 분위수 (band 모델과 동일).
    min_n: 학습 배치가 이보다 적은 시점은 제외 (band 모델과 동일, 이 시점은 NaN)."""
    tr = ts[ts.batch.isin(train_batches)]
    g = tr.groupby("k")[vars]
    enough = g.size() >= min_n  # 늦은 시점은 배치가 1개뿐이라 std가 NaN
    mean_k = g.mean()[enough]
    std_k = g.std()[enough]
    # 거의 상수인 변수(pH, RPM 등)는 std가 0에 가까워 표준화가 불안정 → 하한선 적용
    std_k = std_k.clip(lower=std_k.where(std_k > 0).quantile(std_floor_q), axis=1)
    # 모든 시점에서 std가 0인 변수(Fg, pressure 등 레시피 고정값)는 하한도 NaN → 1 (편차를 원 단위로)
    std_k = std_k.fillna(1.0).replace(0.0, 1.0)
    k_index = mean_k.index.to_numpy()

    # 각 행의 k에 맞는 mean/std로 표준화 (학습 범위 밖 k는 제외)
    pos = pd.Index(k_index).get_indexer(tr["k"])
    ok = pos >= 0
    X_raw = tr[vars].to_numpy(float)[ok]
    Z = (X_raw - mean_k.to_numpy()[pos[ok]]) / std_k.to_numpy()[pos[ok]]

    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(Z)
    scores = pca.transform(Z)
    lambdas = pca.explained_variance_
    T2 = (scores ** 2 / lambdas).sum(axis=1)
    Z_hat = scores @ pca.components_
    SPE = ((Z - Z_hat) ** 2).sum(axis=1)

    return {"vars": np.array(vars), "k_index": k_index,
            "mean_by_k": mean_k.to_numpy(), "std_by_k": std_k.to_numpy(),
            "components": pca.components_, "explained_variance": lambdas,
            "T2_limit": float(np.quantile(T2, alpha)),
            "SPE_limit": float(np.quantile(SPE, alpha)),
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "alpha": float(alpha)}


def pca_scores(state, ts_batch):
    """시점마다 T²와 SPE 계산 (학습 데이터에 없는 시점이거나 변수값이 없으면 NaN)."""
    vars = list(state["vars"])  # fit_pca는 ndarray, load_pca는 list
    pos = pd.Index(state["k_index"]).get_indexer(ts_batch["k"])
    ok = pos >= 0

    n = len(ts_batch)
    T2, SPE = np.full(n, np.nan), np.full(n, np.nan)

    if ok.any():
        X_raw = ts_batch[vars].to_numpy(float)[ok]
        Z = (X_raw - state["mean_by_k"][pos[ok]]) / state["std_by_k"][pos[ok]]
        missing = np.isnan(Z).any(axis=1)
        Z[missing] = 0.0  # 계산은 하되 아래에서 NaN으로 덮어씀
        scores = Z @ state["components"].T
        t2 = (scores ** 2 / state["explained_variance"]).sum(axis=1)
        Z_hat = scores @ state["components"]
        spe = ((Z - Z_hat) ** 2).sum(axis=1)
        t2[missing] = np.nan
        spe[missing] = np.nan
        T2[ok] = t2
        SPE[ok] = spe

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
