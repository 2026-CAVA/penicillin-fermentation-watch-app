"""Leakage, split, threshold-freeze and what-if checks. Run `python -m main all` first (tests read data/cache/ and artifacts/)."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from main import __main__ as pipeline
from main import service
from main.data import (ART, DATA, META, batch_meta, build_batch_table, cfg, load_split, load_ts, names,
                       offline_available, read_json)
from main.model import load_xgb, oof_predict
from main.warning import band_flags, deviation, load_bands, resolve_ranges, risk_score, sustained, warning_metrics
from main.whatif import rule_whatif, rule_whatif_by_var, whatif_predict


@pytest.fixture(scope="module")
def s():
    return {"split": load_split(), "table": pd.read_parquet(DATA / "batch_table.parquet"), "ts": load_ts(),
            "risk": pd.read_parquet(DATA / "risk_ts.parquet"), "th": read_json(ART / "thresholds.json"),
            "q": read_json(ART / "quality.json"), "bands": load_bands(ART / "bands.npz")}


def test_split_disjoint(s):
    sp = s["split"]
    sets = [set(sp[k]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(range(1, 101))
    assert set(range(91, 101)) <= sets[2]  # fault batches are test-only
    assert set(sp["fold"]) == sets[0]


def test_forbidden_columns(s):
    banned = set(names("forbidden")) | set(names("display")) | set(names("meta"))
    feats = s["q"]["features"]
    assert not set(META) & set(feats)
    for f in feats:
        if not f.startswith("ctrl_"):  # control mode is an input by decision (known before the batch)
            assert f.rsplit("_", 1)[0] not in banned, f


def test_fit_on_train_only(s):
    sp = s["split"]
    assert sorted(s["bands"]["full"]["batches"]) == sp["train"]
    rt = offline_available(s["ts"])
    again = resolve_ranges(rt[rt.batch.isin(sp["train"])], cfg()["safety"])
    for v, r in s["th"]["ranges"].items():
        assert r["s"] == pytest.approx(again[v]["s"]) and r["lo"] == pytest.approx(again[v]["lo"])


def test_oof_excludes_self(s):
    """Saved fold models equal models retrained without their fold (also checks reproducibility)."""
    sp, t = s["split"], s["table"]
    tr = sp["train"]
    X, y = t.loc[tr, s["q"]["features"]], t.y[tr]
    pred, _ = oof_predict(X, y, s["q"]["params"], t.fold[tr], cfg()["seed"])
    for k in range(cfg()["split"]["n_folds"]):
        ids = [b for b in tr if sp["fold"][b] == k]
        assert not set(ids) & set(s["bands"][f"fold{k}"]["batches"])
        np.testing.assert_allclose(load_xgb(ART / f"xgb_fold{k}.json").predict(X.loc[ids]), pred[ids], rtol=1e-5)


def test_band_features_oof(s):
    sp, ts, risk = s["split"], s["ts"], s["risk"]
    rows = ts.batch == sp["train"][0]
    expect = band_flags(ts[rows], s["bands"][f"fold{sp['fold'][sp['train'][0]]}"])
    pd.testing.assert_frame_equal(risk.loc[rows, expect.columns], expect, check_dtype=False)
    assert (band_flags(ts[rows], s["bands"]["full"]) != expect).any().any()  # the full band includes the batch


def test_risk_single_var():
    ranges = {"a": {"lo": 0, "hi": 1, "s": 2, "w": 1}, "b": {"lo": 0, "hi": 1, "s": 2, "w": 1}}
    ts = pd.DataFrame({"a": [0.5, 3.0, 2.0], "b": [0.5, 0.5, np.nan]})
    assert risk_score(deviation(ts, ranges), ranges).tolist() == [0, 100, 50]  # one var a full tolerance out -> 100


def test_sustained_ignores_short_spikes():
    r = pd.Series([0, 90, 0, 50, 60, 70, 0, 80.0])
    batch = pd.Series([1] * 6 + [2] * 2)
    assert sustained(r, batch, 3).tolist() == [0, 0, 0, 0, 0, 50, 0, 0]  # never across batches


def test_no_future_info(s):
    ts = s["ts"][s["ts"].batch.isin([1, 2])].copy()
    risk = s["risk"][s["risk"].batch.isin([1, 2])].copy()
    t_cut, spec, meta = 110.0, cfg()["features"], batch_meta(ts)  # an offline sample (108 h) is within delay of t_cut
    base = build_batch_table(offline_available(ts), risk, meta, t_cut, spec)
    late = ts.t > t_cut
    for c in names("input"):
        ts.loc[late, c] = ts.loc[late, c] * 10 + 7
    risk.loc[risk.t > t_cut, "R"] = 99
    recent = ts.P_off.notna() & (ts.t > t_cut - cfg()["offline_delay_h"]) & ~late
    assert recent.any()
    ts.loc[recent, "P_off"] = 1e6  # sampled before t_cut but its result is not available yet
    pd.testing.assert_frame_equal(base, build_batch_table(offline_available(ts), risk, meta, t_cut, spec))


def test_test_step_is_frozen():
    src = inspect.getsource(pipeline.run_test)
    assert "write_json(ART / \"thresholds" not in src and "save_model" not in src  # Threshold Freeze
    assert "baselines(" in src and "baselines(" in inspect.getsource(pipeline.run_quality)  # Baseline Comparison
    assert "fold" not in src  # test metrics come from the final model only


def test_display_source(s):
    sp = s["split"]
    for bid, src in [(sp["train"][0], "oof"), (sp["val"][0], "final"), (sp["test"][0], "final")]:
        v = service.batch_view(bid)
        assert v["pred_source"] == src and v["tag"]
        json.dumps(v)  # JSON-safe for the front end
    assert all(b["tag"] for b in service.list_batches())


def test_whatif_zero_change_and_bounds(s):
    st = service._state()
    bid = s["split"]["val"][0]
    x = st["X"].loc[bid]
    r0 = whatif_predict(st["final"], x, {f: 0.0 for f in st["bounds"]}, st["bounds"])
    assert r0["delta"] == 0 and r0["y_pred_before"] == pytest.approx(service.batch_view(bid)["y_pred_kg"])
    for f, io in whatif_predict(st["final"], x, {f: 50.0 for f in st["bounds"]}, st["bounds"])["inputs"].items():
        if "linked_to" not in io:  # linked features (V_last, Wt_last) follow the slider and are not clipped
            assert st["bounds"][f][0] <= io["after"] <= st["bounds"][f][1]


def test_whatif_sugar_raises_volume_and_weight():
    class Sum:  # stand-in model: prediction = sum of features, so linked changes show up in delta
        def predict(self, X):
            return X.sum(axis=1).to_numpy()
    import main.whatif as w
    x = pd.Series({"Fs_tot": 100.0, "V_last": 50000.0, "Wt_last": 60000.0, "Fg_mean": 30.0})
    coupled = w.COUPLED
    w_shap, w.shap_values = w.shap_values, lambda m, X: X * 0
    try:
        r = whatif_predict(Sum(), x, {"Fs_tot": 0.1}, {"Fs_tot": (0, 1000)}, coupled)
        c = w.response_curve(Sum(), x, "Fs_tot", {"Fs_tot": (0, 200)}, n=3, coupled=coupled)
    finally:
        w.shap_values = w_shap
    k = coupled["Fs_tot"]
    assert r["inputs"]["V_last"] == {"before": 50000.0, "after": 50000.0 + 10 * k["V_last"], "linked_to": "Fs_tot"}
    assert r["inputs"]["Wt_last"]["after"] == pytest.approx(60000.0 + 10 * k["Wt_last"])
    assert r["delta"] == pytest.approx(10 * (1 + k["V_last"] + k["Wt_last"]))
    assert c["y"][1] == pytest.approx(x.sum())  # grid midpoint = current value -> unchanged


def test_rule_whatif(s):
    ranges, cuts = s["th"]["ranges"], s["th"]["cuts"]
    ts = offline_available(s["ts"][s["ts"].batch == 92])
    fixed = ts.copy()
    for v, r in ranges.items():
        fixed[v] = fixed[v].clip(r["lo"], r["hi"])
    assert (deviation(fixed, ranges) == 0).all().all()
    assert rule_whatif(ts, ranges, cuts)["critical_after"] == 0
    by = rule_whatif_by_var(ts, ranges, cuts)
    assert (by.reduction >= 0).all() and set(by["var"]) == set(ranges)


def test_warning_metrics():
    rt = pd.DataFrame({"batch": [1] * 4 + [2] * 4, "t": [0, 1, 2, 3] * 2, "fault_ref": [0, 0, 1, 1, 0, 0, 0, 0]})
    m = warning_metrics(rt, pd.Series([0, 0, 0, 1, 0, 1, 0, 0], dtype=bool), [1], [2])
    assert m["batch_detection"] == 1 and m["sensitivity_points"] == 0.5 and m["lead_time_h_median"] == 1
    assert m["far_points"] == 0.25 and m["far_batches"] == 1


def test_test_step_on_synthetic_batches(s, tmp_path, monkeypatch, capsys):
    """Run the test step end to end without touching the real test set: val normal batches play the
    test normals, and copies of val batches with a fault window from 50 h play the fault batches."""
    import main.data as data_mod
    sp, risk, table = s["split"], s["risk"], s["table"].copy()  # do not touch the shared fixture
    normals, sources = sp["val"][:5], sp["val"][5:8]
    fakes = {900 + i: b for i, b in enumerate(sources, start=1)}
    for fake, src in fakes.items():
        r = risk[risk.batch == src].assign(batch=fake)
        r["fault_ref"] = (r.t >= 50).astype(int)
        risk = pd.concat([risk, r], ignore_index=True)
        table.loc[fake] = table.loc[src]
        table.loc[fake, "is_fault"] = True

    data_dir, art_dir = tmp_path / "data", tmp_path / "artifacts"
    data_dir.mkdir()
    art_dir.mkdir()
    risk.to_parquet(data_dir / "risk_ts.parquet")
    table.to_parquet(data_dir / "batch_table.parquet")
    (data_dir / "split.json").write_text(json.dumps({**sp, "test": normals + list(fakes)}), encoding="utf-8")
    for f in ("thresholds.json", "quality.json", "xgb.json"):
        (art_dir / f).write_bytes((ART / f).read_bytes())
    for mod in (data_mod, pipeline):
        monkeypatch.setattr(mod, "DATA", data_dir)
    monkeypatch.setattr(pipeline, "ART", art_dir)
    monkeypatch.setitem(cfg(), "demo_batches", [normals[0], 901])

    pipeline.run_test()
    out = capsys.readouterr().out
    print(out)  # shown with `pytest -rP`
    assert "test: 5 normal + 3 fault batches" in out and "combined_warning+: detected" in out
    assert (art_dir / "test_evaluated.json").exists() and not (ART / "test_evaluated.json").exists()
    with pytest.raises(SystemExit):  # second run refused
        pipeline.run_test()
