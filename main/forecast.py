"""(6) Forecast: penicillin concentration P up to 8 h ahead from what is known at t.

Inputs at t: online inputs, offline values that have arrived (offline_delay_h after sampling, held), feed
totals so far, 4 h changes. P itself (online) is not measurable in a real plant and is only the target.
One model for every horizon h (h is a feature); it predicts P(t + h) minus the latest arrived P_off.
"""
import numpy as np
import pandas as pd

from main.data import DT, cfg, names, offline_available
from main.model import make_xgb

CUM = ["Fs", "Fpaa", "Foil"]               # feed totals so far (L)
DIFF = ["OUR", "CER", "CO2outgas", "DO", "V"]  # change over the last 4 h


def frame(ts):
    """Causal features at every time point of the batches in ts (raw time series). base = latest arrived P_off
    (0 before the first arrives); P_off_age_h = t - sampling time of that value; P_off_slope = slope between
    the last two arrived samples."""
    rt = offline_available(ts)
    g, lag = rt.groupby("batch"), round(4 / DT)
    F = rt[["batch", "k", "t", *names("input", "online"), *names("input", "offline")]].copy()
    for c in CUM:
        F[f"{c}_cum"] = g[c].cumsum() * DT
    for c in DIFF:
        F[f"{c}_d4h"] = rt[c] - g[c].shift(lag)
    smp = ts.loc[ts.P_off.notna(), ["batch", "t", "P_off"]]
    slope = smp.P_off.groupby(smp.batch).diff() / smp.t.groupby(smp.batch).diff()  # between the last 2 samples
    lag_off = round(cfg()["offline_delay_h"] / DT)
    for name, col in (("P_off_age_h", smp.t), ("P_off_slope", slope)):  # known from sampling + delay on
        known = col.reindex(ts.index).groupby(ts.batch).shift(lag_off).groupby(ts.batch).ffill()
        F[name] = F.t - known if name == "P_off_age_h" else known
    F["base"] = F.P_off.fillna(0.0)
    return F


def linear(R):
    """Baseline: extend the latest arrived P_off along the slope of the last two samples."""
    return R.base + R.P_off_slope.fillna(0) * (R.P_off_age_h.fillna(0) + R.h)


def feature_cols(F):
    return [c for c in F.columns if c not in ("batch", "k")] + ["h"]


def rows(ts, F, horizons, every_h):
    """Time points on an every_h grid x horizons; y = P(t + h) - base, p_true = P(t + h)."""
    grid = F[F.k % round(every_h / DT) == 0]
    out = []
    for h in horizons:
        fut = ts.P.groupby(ts.batch).shift(-round(h / DT)).loc[grid.index]
        out.append(grid.assign(h=h, y=fut - grid.base, p_true=fut)[fut.notna()])
    return pd.concat(out, ignore_index=True)


def fit(R, cols, params, seed):
    return make_xgb(params, seed).fit(R[cols], R.y)


def predict(model, R, cols):
    """P forecast for rows R (base + predicted change)."""
    return R.base.to_numpy() + model.predict(R[cols])


def horizons():
    f = cfg()["forecast"]
    return [round(i * f["step_h"], 4) for i in range(1, round(f["horizon_h"] / f["step_h"]) + 1)]
