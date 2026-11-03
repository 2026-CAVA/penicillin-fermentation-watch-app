"""Data: config, loading, batch meta, yield target, real-time offline view, split, batch summary table."""
import csv
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, ART = ROOT / "data" / "cache", ROOT / "artifacts"  # raw CSVs sit in data/ (config `paths`)
DT = 0.2  # h, sampling interval
META = ["y", "group", "is_fault", "split", "fold"]  # batch table columns that are never features


@cache
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def raw_path(key):
    return (ROOT / cfg()["paths"][key]).resolve()


def names(role, kind=None):
    return [c["name"] for c in cfg()["columns"] if c["role"] == role and (kind is None or c.get("kind") == kind)]


def col_meta(name):
    """{label, unit, kind} of a raw column (kind: online | offline | None)."""
    c = next(c for c in cfg()["columns"] if c["name"] == name)
    return {"label": c.get("label", name), "unit": c.get("unit"), "kind": c.get("kind")}


FIXED_LABELS = {"R_mean": "Mean risk score R(t) up to t_cut", "R_max": "Max risk score R(t) up to t_cut",
                "n_critical": "CRITICAL points up to t_cut", "n_band": "Band excursion points up to t_cut",
                "ctrl_operator": "Operator-controlled batch (0/1)", "ctrl_apc": "APC-controlled batch (0/1)"}


def feature_label(f, t_cut=None):
    """Display name with unit for a batch-table feature, e.g. DO_min -> 'First-half min Dissolved oxygen (mg/L)';
    with t_cut (the cutoff the feature was built at) the span is explicit: '0-46 h min Dissolved oxygen (mg/L)'."""
    label = _feature_label(f)
    return label if t_cut is None else label.replace("First-half", f"0-{t_cut:g} h").replace("t_cut", f"{t_cut:g} h")


def _feature_label(f):
    if f in FIXED_LABELS:
        return FIXED_LABELS[f]
    for suffix, text in (("_min_t", "Hour of minimum {} (h)"), ("_mean", "First-half mean {} ({})"),
                         ("_min", "First-half min {} ({})"), ("_last", "{} at t_cut ({})"),
                         ("_tot", "First-half total {} ({})"),
                         ("_slope", f"{{}} trend, last {cfg()['features']['slope_h']} h ({{}}/h)")):
        if f.endswith(suffix):
            m = col_meta(f[: -len(suffix)])
            if suffix == "_tot":  # flow L/h x h = L
                return text.format(m["label"].removesuffix(" rate"), m["unit"].removesuffix("/h"))
            return text.format(m["label"], m["unit"])
    return f


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# --- time series -------------------------------------------------------------

# One raw header name contains unquoted commas, so the header has 2 more fields than the data rows.
BROKEN_HEADER_FIELD = "1- No Raman spec, 1-Raman spec recorded,2-PAT control(PAT_ref:PAT ref)"


def read_header(ts_csv):
    """Raw header fields with the comma-containing name re-joined, checked against the data row width."""
    with open(ts_csv, encoding="utf-8") as f:
        line, row = f.readline().rstrip("\n"), f.readline()
    header = next(csv.reader([line.replace(BROKEN_HEADER_FIELD, f'"{BROKEN_HEADER_FIELD}"')]))
    if len(header) != len(row.split(",")):
        raise ValueError(f"header has {len(header)} fields, data rows have {len(row.split(','))}")
    return header


def load_raw(ts_csv):
    """Read the first 37 (non-Raman) columns, checked against the repaired header and renamed."""
    spec = cfg()["columns"]
    for c, h in zip(spec, read_header(ts_csv)):
        if c["src"] != h:
            raise ValueError(f"column {c['name']}: expected {c['src']!r}, got {h!r}")
    ts = pd.read_csv(ts_csv, header=0, names=[c["name"] for c in spec], usecols=range(len(spec)))
    if not (ts.batch == ts.batch_dup).all():
        raise ValueError("batch reference columns disagree")
    ts = ts.drop(columns="batch_dup").sort_values(["batch", "t"], ignore_index=True)
    ts["k"] = np.rint(ts.t / DT).astype(int)  # integer time step, used to align batches
    return ts


def load_ts():
    return pd.read_parquet(DATA / "timeseries.parquet")


def batch_meta(ts):
    m = ts.groupby("batch").agg(ctrl=("ctrl", "first"), pat=("pat", "first"),
                                is_fault=("fault_flag", "max"), t_end=("t", "max"))
    m["group"] = np.select([m.pat == 2, m.ctrl == 1], ["apc", "operator"], "recipe")
    m["is_fault"] = m.is_fault.astype(bool)
    return m[["group", "is_fault", "t_end"]]


def batch_yield_kg(ts):
    """Li et al. 2024 Eq. 13: P_T*V_T + sum_t P_t*F_dis,t*dt (g/L * L = g), in kg.
    Built from the time series because the Statistics file does not reconcile with it."""
    end = ts.groupby("batch").last()
    dumped = (ts.P * -ts.Fremoved * DT).groupby(ts.batch).sum()  # Fremoved is logged as negative
    return (end.P * end.V + dumped) / 1000


def offline_available(ts, delay_h=None):
    """Offline values as known in real time: a sample taken at t is available from t + delay, then held."""
    lag = int(round((cfg()["offline_delay_h"] if delay_h is None else delay_h) / DT))
    out = ts.copy()
    left = ts[["batch", "k"]].reset_index().sort_values("k")
    for c in names("input", "offline"):
        s = ts.loc[ts[c].notna(), ["batch", "k", c]]
        s = s.assign(k=s.k + lag).sort_values("k")
        m = pd.merge_asof(left, s, on="k", by="batch", direction="backward")
        out[c] = m.set_index("index")[c].reindex(ts.index)
    return out


# --- split -------------------------------------------------------------------

def make_split(meta, per_group, n_folds, seed):
    """Normal batches stratified by control group; all fault batches test-only."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": [], "fold": {}}
    a = per_group["train"]
    b = a + per_group["val"]
    for _, d in meta[~meta.is_fault].groupby("group"):
        if len(d) < b + per_group["test"]:
            raise ValueError(f"group too small for split: {len(d)}")
        ids = [int(i) for i in rng.permutation(d.index.to_numpy())]
        out["train"] += ids[:a]
        out["val"] += ids[a:b]
        out["test"] += ids[b:b + per_group["test"]]
        out["fold"].update({bid: i % n_folds for i, bid in enumerate(ids[:a])})  # folds balanced by group
    out["test"] += [int(i) for i in meta.index[meta.is_fault]]
    for k in ("train", "val", "test"):
        out[k].sort()
    return out


def load_split():
    s = read_json(DATA / "split.json")
    s["fold"] = {int(b): f for b, f in s["fold"].items()}
    return s


# --- batch summary table -----------------------------------------------------

def build_batch_table(ts_rt, risk_ts, meta, t_cut, spec):
    """One row per batch from 0 <= t <= t_cut. ts_rt: real-time view (offline delayed + held).
    risk_ts already uses the OOF band for train batches, so band features are out-of-fold there."""
    early = ts_rt[ts_rt.t <= t_cut]
    g = early.groupby("batch")
    w = early[early.t > t_cut - spec["slope_h"]].groupby("batch")  # trend over the last slope_h hours
    r = risk_ts[risk_ts.t <= t_cut]
    rg = r.groupby("batch")
    X = pd.concat([
        g[spec["mean"]].mean().add_suffix("_mean"),
        g[spec["min"]].min().add_suffix("_min"),
        g[spec["last"]].last().add_suffix("_last"),
        (g[spec["total"]].sum() * DT).add_suffix("_tot"),
        g[spec["offline_last"]].last().add_suffix("_last"),  # last available (non-null) value
        (w[spec["slope"]].last() - w[spec["slope"]].first()).div(w.t.last() - w.t.first(), axis=0).add_suffix("_slope"),
        pd.DataFrame({f"{v}_min_t": early.t[g[v].idxmin()].to_numpy() for v in spec["argmin_t"]},
                     index=g.size().index),  # hour of the first minimum
        pd.DataFrame({"R_mean": rg.R.mean(), "R_max": rg.R.max(),
                      "n_critical": (r.grade == "CRITICAL").groupby(r.batch).sum(),
                      "n_band": (r.band_n > 0).groupby(r.batch).sum()}),
    ], axis=1)
    X["ctrl_operator"] = (meta.group == "operator").astype(int)  # control mode is known before the batch
    X["ctrl_apc"] = (meta.group == "apc").astype(int)
    return X


def feature_cols(table):
    return [c for c in table.columns if c not in META]


def assemble(ts, risk_ts, split, t_cut):
    """Batch table with features, target y (kg) and meta columns."""
    meta = batch_meta(ts)
    X = build_batch_table(offline_available(ts), risk_ts, meta, t_cut, cfg()["features"])
    where = {b: s for s in ("train", "val", "test") for b in split[s]}
    return X.assign(y=batch_yield_kg(ts), group=meta.group, is_fault=meta.is_fault,
                    split=X.index.map(where), fold=X.index.map(lambda b: split["fold"].get(b, -1)))
