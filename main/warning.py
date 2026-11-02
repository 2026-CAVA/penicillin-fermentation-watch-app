"""(1) Early warning: normal +-k sigma band, rule risk score R(t), grades, alarm rule, warning metrics."""
import numpy as np
import pandas as pd

GRADES = np.array(["STABLE", "WARNING", "CRITICAL"])


# --- band (fitted on train normal batches only) ------------------------------

def fit_band(ts, batches, vars, k=3.0, min_n=10, std_floor_q=0.1):
    g = ts[ts.batch.isin(batches)].groupby("k")[vars]
    mean, std = g.mean(), g.std()
    std = std.clip(lower=std.where(std > 0).quantile(std_floor_q), axis=1)  # near-constant vars (pH, T)
    return {"k": mean.index.to_numpy(), "mean": mean.to_numpy(), "std": std.to_numpy(), "n": g.size().to_numpy(),
            "vars": np.array(vars), "batches": np.array(sorted(batches)),
            "width": np.array(k), "min_n": np.array(min_n)}


def band_at(ts, band):
    """Band mean/std aligned to ts rows; NaN where the band does not apply (missing step or n < min_n)."""
    pos = pd.Index(band["k"]).get_indexer(ts["k"])
    ok = pos >= 0
    ok[ok] = band["n"][pos[ok]] >= band["min_n"]
    p = np.where(ok, pos, 0)
    mu, sd = band["mean"][p].astype(float), band["std"][p].astype(float)
    mu[~ok], sd[~ok] = np.nan, np.nan
    return mu, sd


def band_flags(ts, band):
    """Per variable: +1 above, -1 below, 0 inside (or no band)."""
    mu, sd = band_at(ts, band)
    x = ts[list(band["vars"])].to_numpy(float)
    f = np.nan_to_num(np.sign(x - mu) * (np.abs(x - mu) > band["width"] * sd))
    return pd.DataFrame(f.astype(int), index=ts.index, columns=[f"band_{v}" for v in band["vars"]])


def save_bands(path, bands):
    np.savez(path, **{f"{name}/{key}": np.asarray(v) for name, b in bands.items() for key, v in b.items()})


def load_bands(path):
    out = {}
    with np.load(path) as z:
        for key in z.files:
            name, field = key.split("/")
            out.setdefault(name, {})[field] = z[key]
    return out


# --- rule risk score ---------------------------------------------------------

def resolve_ranges(ts_train_normal, spec):
    """Fill `data` bounds from train-normal percentiles; add s_i = scale_mult x train-normal std."""
    q_lo, q_hi = spec["data_quantiles"]
    out = {}
    for v, r in spec["vars"].items():
        x = ts_train_normal[v].dropna()
        lo, hi = r.get("lo"), r.get("hi")
        if r["source"] == "data":
            lo, hi = float(x.quantile(q_lo)), float(x.quantile(q_hi))
        out[v] = {"lo": lo, "hi": hi, "s": float(spec["scale_mult"] * x.std()),
                  "w": float(r.get("w", 1.0)), "source": r["source"]}
    return out


def deviation(ts, ranges):
    """d_i(t) = clip(max(lo_i - x_i, x_i - hi_i, 0) / s_i, 0, 1); R(t) = 100 * min(1, sum_i w_i * d_i(t))."""
    d = {}
    for v, r in ranges.items():
        x = ts[v].to_numpy(float)
        lo = -np.inf if r["lo"] is None else r["lo"]
        hi = np.inf if r["hi"] is None else r["hi"]
        e = np.maximum(np.maximum(lo - x, x - hi), 0)
        d[v] = np.nan_to_num(np.clip(e / r["s"], 0, 1))  # NaN = offline value not yet available
    return pd.DataFrame(d, index=ts.index)


def risk_score(dev, ranges):
    w = np.array([ranges[v]["w"] for v in dev.columns])
    return pd.Series(100 * np.minimum(1, dev.to_numpy() @ w), index=dev.index)


def risk_grade(r, cuts):
    return pd.Series(GRADES[np.searchsorted(cuts, r.to_numpy(), side="right")], index=r.index)


def risk_contrib(dev, ranges):
    """Share (%) of each variable in the deviation at that time point."""
    wd = dev * np.array([ranges[v]["w"] for v in dev.columns])
    tot = wd.sum(axis=1)
    return wd.div(tot.where(tot > 0), axis=0).fillna(0) * 100


# --- alarm rule and thresholds -----------------------------------------------

def sustained(x, batch, n):
    """Lowest value over the last n points of the same batch (causal); 0 until n points exist.
    sustained(R) >= cut  <=>  R stayed at or above cut for n consecutive points."""
    return x.astype(float).groupby(batch).transform(lambda s: s.rolling(n, min_periods=n).min()).fillna(0)


def band_alarm(n_flagged, batch, n_vars, consec):
    """True when >= n_vars band excursions held for the last `consec` points (causal)."""
    return sustained(n_flagged >= n_vars, batch, consec).astype(bool)


def runs(sign, min_len):
    """[(start, end, sign)]: positions (end exclusive) where a nonzero sign holds for >= min_len points."""
    s = np.asarray(sign)
    edges = np.flatnonzero(np.diff(s)) + 1
    return [(a, b, int(s[a])) for a, b in zip(np.r_[0, edges], np.r_[edges, len(s)]) if s[a] != 0 and b - a >= min_len]


def range_sign(ts, ranges):
    """Per range variable: +1 above hi, -1 below lo, 0 inside or not yet available."""
    out = {}
    for v, r in ranges.items():
        x = ts[v].to_numpy(float)
        out[v] = np.where(x > (np.inf if r["hi"] is None else r["hi"]), 1,
                          np.where(x < (-np.inf if r["lo"] is None else r["lo"]), -1, 0))
    return pd.DataFrame(out, index=ts.index)


def combine(grade_r, alarm):
    """Signal = R(t) grade, raised to at least WARNING while the band alarm is on."""
    return grade_r.where(~(alarm & (grade_r == "STABLE")), "WARNING")


def lowest_cut(r, target, start=1):
    """Smallest integer cut whose alarm rate (r >= cut) is within target; 101 = never fires.
    r = one peak per batch gives a batch-level rate."""
    for c in range(start, 101):
        if (r >= c).mean() <= target:
            return c
    return 101


def warning_metrics(risk_ts, alarm, fault_batches, normal_batches):
    """alarm: bool Series aligned to risk_ts (needs batch, t, fault_ref)."""
    d = risk_ts.assign(alarm=alarm.to_numpy())
    f, n = d[d.batch.isin(fault_batches)], d[d.batch.isin(normal_batches)]
    out = {"far_points": float(n.alarm.mean()) if len(n) else np.nan,
           "far_batches": float(n.groupby("batch").alarm.any().mean()) if len(n) else np.nan}
    if len(f):
        in_fault = f[f.fault_ref == 1]
        onset = in_fault.groupby("batch").t.min()
        first = f[f.alarm & (f.t >= f.batch.map(onset))].groupby("batch").t.min()
        out |= {"batch_detection": float(f.groupby("batch").alarm.any().mean()),
                "sensitivity_points": float(in_fault.alarm.mean()),
                "lead_time_h_median": float((first - onset.reindex(first.index)).median()) if len(first) else np.nan,
                "detected_after_onset": int(len(first)), "n_fault_batches": int(f.batch.nunique())}
    return out
