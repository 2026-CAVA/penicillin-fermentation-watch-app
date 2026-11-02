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
from main.model import load_xgb, oof_predict, shap_values
from main.warning import band_flags, deviation, load_bands, resolve_ranges, risk_score, sustained, warning_metrics
from main.whatif import response_curve, rule_whatif, rule_whatif_by_var, whatif_predict


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


def test_sliders_move(s):
    """Every configured slider has a train range wider than a point and moves the prediction somewhere."""
    st = service._state()
    assert service.slider_bounds(pd.DataFrame({"a": [1, 1], "b": [1, 2]}), ["a", "b"]) == {"b": (1.0, 2.0)}
    assert set(st["bounds"]) == set(cfg()["sliders"])
    for f in st["bounds"]:
        spans = [np.ptp(response_curve(st["final"], st["X"].loc[b], f, st["bounds"])["y"]) for b in s["split"]["val"]]
        assert max(spans) > 0, f


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
    sp, risk, table, ts = s["split"], s["risk"], s["table"].copy(), s["ts"]  # do not touch the shared fixture
    cp = pd.read_parquet(DATA / "cutoff_preds.parquet")
    normals, sources = sp["val"][:5], sp["val"][5:8]
    fakes = {900 + i: b for i, b in enumerate(sources, start=1)}
    for fake, src in fakes.items():
        r = risk[risk.batch == src].assign(batch=fake)
        r["fault_ref"] = (r.t >= 50).astype(int)
        risk = pd.concat([risk, r], ignore_index=True)
        ts = pd.concat([ts, ts[ts.batch == src].assign(batch=fake)], ignore_index=True)
        cp = pd.concat([cp, cp.loc[[src]].rename(index={src: fake})])
        table.loc[fake] = table.loc[src]
        table.loc[fake, "is_fault"] = True

    data_dir, art_dir = tmp_path / "data", tmp_path / "artifacts"
    data_dir.mkdir()
    art_dir.mkdir()
    risk.to_parquet(data_dir / "risk_ts.parquet")
    table.to_parquet(data_dir / "batch_table.parquet")
    ts.to_parquet(data_dir / "timeseries.parquet")
    cp.to_parquet(data_dir / "cutoff_preds.parquet")
    (data_dir / "split.json").write_text(json.dumps({**sp, "test": normals + list(fakes)}), encoding="utf-8")
    for f in ("thresholds.json", "quality.json", "xgb.json", "cutoffs.json", "forecast.json", "forecast_meta.json"):
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
    res = read_json(art_dir / "test_results.json")
    assert len(res["verdicts"]) == 7 and "normal 161 h" in res["yield_trend"] and "fault 8" in res["forecast"]
    assert not (ART / "test_results.json").exists()
    with pytest.raises(SystemExit):  # second run refused
        pipeline.run_test()


def test_rule_tree_beats_mean_or_is_marked():
    r = read_json(ART / "rules.json")
    assert r["r2"]["val"] > 0 or r["reference_only"]
    assert r["reference_only"] == (not r["r2"]["val"] > 0)


def test_api_matches_mockup_contract(s):
    """Key sets of the FermentIQ mockup (BatchSummary, series rows, SHAP items); scoreCuts is our addition."""
    from fastapi.testclient import TestClient
    from main.api import app
    c, bid = TestClient(app), s["split"]["val"][0]
    summary = {"id", "runStatus", "elapsedH", "plannedEndH", "state", "predictedFinal", "predictedRange", "latestScore",
               "finalMeasured", "firstAnomalyH", "anomalyDurationH", "segments", "scoreCuts"}
    rows = c.get("/batches").json()
    assert len(rows) == 100 and all(set(r) == summary for r in rows)
    d = c.get(f"/batches/{bid}").json()
    assert set(d) == summary | {"intervalH", "shapBase", "explanation", "seriesMeta", "series"}
    assert set(d["seriesMeta"]) == set(d["series"][0]) - {"t", "score", "state", "predictedFinal"}
    assert set(d["series"][0]) == {"t", "pH", "DO", "temperature", "feedRate", "penicillin", "score", "state",
                                   "predictedFinal"}
    assert {x["state"] for x in d["series"]} <= {"normal", "caution", "anomaly"}
    seg = d["segments"]
    assert seg[0]["start"] == d["series"][0]["t"] and all(a["end"] == b["start"] for a, b in zip(seg, seg[1:]))
    t_cut = s["split"]["t_cut"]
    assert all((x["predictedFinal"] is None) == (x["t"] < t_cut) for x in d["series"])
    assert c.get(f"/batches/{bid}/shap?t=50").json() == []  # no yield prediction before t_cut
    sh = c.get(f"/batches/{bid}/shap?t=150").json()
    assert len(sh) == 5 and set(sh[0]) == {"feature", "label", "value"}
    r = c.post("/simulate", json={"batch_id": bid, "changes": {"Fpaa_tot": 0.05}})
    assert r.status_code == 200 and "delta" in r.json()
    assert c.post("/simulate", json={"batch_id": bid, "changes": {"Fs_tot": 0.05}}).status_code == 422
    assert c.get("/batches/999").status_code == 404


def test_metadata(s):
    from fastapi.testclient import TestClient
    from main.api import app
    m = TestClient(app).get("/model").json()
    assert m["title"] == f"Plan A · XGBoost · val R² {m['val']['xgboost']['r2']:.2f}"
    assert set(m["demo_batches"]) <= set(s["split"]["test"]) and m["target"]["unit"] == "kg"
    v = service.batch_view(s["split"]["val"][0])
    assert v["series"]["DO"]["unit"] == "mg/L" and v["series"]["DO"]["kind"] == "online"
    assert v["series"]["PAA_off"]["kind"] == "offline" and v["series"]["PAA_off"]["delay_h"] == cfg()["offline_delay_h"]
    assert all(d["label"] != d["feature"] for d in v["shap_top3"])
    sh = shap_values(service._route(v["batch_id"])[0], s["table"].loc[[v["batch_id"]], s["q"]["features"]].astype(float))
    assert v["shap_base"] + sh.iloc[0].drop("_base").sum() == pytest.approx(v["y_pred_kg"], rel=1e-4)


def test_shap_all(s):
    from fastapi.testclient import TestClient
    from main.api import app
    c, bid = TestClient(app), s["split"]["val"][0]
    allv = c.get(f"/batches/{bid}/shap?k=0").json()
    assert len(allv) == len(s["q"]["features"])
    d = c.get(f"/batches/{bid}").json()
    t_cut = s["split"]["t_cut"]
    pred = next(x["predictedFinal"] for x in d["series"] if x["t"] >= t_cut)
    assert d["shapBase"] + sum(x["value"] for x in allv) == pytest.approx(pred, rel=1e-4)  # waterfall adds up
    v = service.batch_view(bid)
    assert [x["feature"] for x in v["shap_all"]] == [x["feature"] for x in allv]


def test_runs_and_events(s):
    from main.warning import runs
    assert runs([0, 1, 1, 1, -1, -1, 0, 1], 2) == [(1, 4, 1), (4, 6, -1)]
    st = service._state()
    n = st["th"]["persist"]
    for bid in (s["split"]["val"][0], 92):
        ev_all, ev = service.events(bid, all=True), service.events(bid)
        assert all(e["duration_h"] >= n * 0.2 - 1e-9 and e["start"] < e["end"] for e in ev_all)
        assert ev == [e for e in ev_all if e["grade"] != "STABLE"]
        ex = service.explanation(bid)
        rk = st["risk"][st["risk"].batch == bid]
        assert (ex["first_alarm_h"] is None) == (rk.grade == "STABLE").all()
        if ex["first_alarm_h"] is not None:
            assert ex["first_alarm_h"] == rk.t[rk.grade != "STABLE"].iloc[0] and ex["causes"]


def test_cross_check_saved():
    m = service.model_info()["cross_check"]
    assert set(m["overlap"]) == {"top3", "top5"} and all(0 <= v <= 1 for v in m["overlap"].values())
    assert [t["shap_rank"] for t in m["top"]] == [1, 2, 3, 4, 5] and all(t["label"] for t in m["top"])


def test_compare(s):
    from fastapi.testclient import TestClient
    from main.api import app
    c, t = TestClient(app), s["table"]
    r = c.get("/compare?ids=92&vars=DO,R").json()
    a, b = r["batches"]
    assert a["id"] == 92 and t.group[b["id"]] == t.group[92] and not t.is_fault[b["id"]]
    assert set(a["values"]) == {"DO", "R"} and len(a["values"]["DO"]) == len(a["t"])
    assert len(c.get("/compare?ids=6,92,62").json()["batches"]) == 3
    assert c.get("/compare?ids=6&vars=nope").status_code == 422 and c.get("/compare?ids=x").status_code == 422
    assert c.get("/compare?ids=999").status_code == 404


@pytest.mark.parametrize("t0", [100.0, 150.0])
def test_replay_has_no_future(t0, monkeypatch):
    """Responses at t0 do not change when everything after t0 is replaced with garbage."""
    def views():
        for f in (service.events, service.explanation):
            f.cache_clear()
        return json.dumps([service.detail(92, t0), service.batch_view(92, t0), service.events(92, True, t0),
                           service.contrib(92, t0), service.shap_top(92, 5, t0), service.forecast(92, t0)],
                          sort_keys=True, default=str)
    base = views()
    st = dict(service._state())
    ts, risk = st["ts"].copy(), st["risk"].copy()
    late = (ts.batch == 92) & (ts.t > t0)
    num = [c for c in names("input") + names("display") if c in ts]
    ts.loc[late, num] = ts.loc[late, num] * 10 + 7
    rl = (risk.batch == 92) & (risk.t > t0)
    risk.loc[rl, ["R", "grade", "grade_r"]] = [99.0, "CRITICAL", "CRITICAL"]
    st |= {"ts": ts, "risk": risk}
    monkeypatch.setattr(service, "_state", lambda: st)
    try:
        assert views() == base
        d = json.loads(base)[0]
        assert d["runStatus"] == "running" and d["finalMeasured"] is None and d["plannedEndH"] is None
        assert max(x["t"] for x in d["series"]) <= t0 and (d["predictedFinal"] is None) == (t0 < 115)
    finally:
        for f in (service.events, service.explanation):
            f.cache_clear()


def test_replay_api():
    from fastapi.testclient import TestClient
    from main.api import app
    c = TestClient(app)
    rows = c.get("/batches?t=100").json()
    assert {r["runStatus"] for r in rows} == {"running"} and all(r["predictedFinal"] is None for r in rows)
    assert c.get("/batches/92?t=0.1").status_code == 422 and c.get("/batches/92/contrib").status_code == 422
    r = c.get("/batches/92/contrib?t=150").json()
    assert r["t"] == 150 and abs(sum(x["share_pct"] for x in r["contrib"]) - 100) < 1e-6


@pytest.mark.parametrize("h", [69.0, 161.0])
def test_cutoff_features_no_future(s, h):
    """Stored cutoff features equal features rebuilt after every value later than the cutoff is scrambled."""
    stored = pd.read_parquet(DATA / "cutoff_features.parquet")
    stored = stored[stored.t_cut == h].drop(columns="t_cut").loc[[1, 2]]
    ts = s["ts"][s["ts"].batch.isin([1, 2])].copy()
    risk = s["risk"][s["risk"].batch.isin([1, 2])].copy()
    late = ts.t > h
    for c in names("input"):
        ts.loc[late, c] = ts.loc[late, c] * 10 + 7
    risk.loc[risk.t > h, "R"] = 99
    again = build_batch_table(offline_available(ts), risk, batch_meta(ts), h, cfg()["features"])
    pd.testing.assert_frame_equal(stored, again[stored.columns], check_dtype=False)


def test_pred_trend(s):
    cut = read_json(ART / "cutoffs.json")
    shown = [r["provided"] for r in cut.values()]
    assert shown == sorted(shown)  # once provided, every later cutoff is provided too
    bid, t_cut = s["split"]["val"][0], s["split"]["t_cut"]
    trend = service.pred_trend(bid)
    first = {p["t_cut"]: p["y_pred"] for p in trend}
    assert first[t_cut] == pytest.approx(service.batch_view(bid)["y_pred_kg"], rel=1e-5)  # same model as quality
    d = service.detail(bid)
    for row in d["series"]:
        reached = [p["y_pred"] for p in trend if p["t_cut"] <= row["t"] + 1e-9]
        assert row["predictedFinal"] == (reached[-1] if reached else None)
    assert d["predictedFinal"] == trend[-1]["y_pred"]


def test_prediction_ranges(s):
    cut = read_json(ART / "cutoffs.json")
    assert all(0 <= r["val_coverage_interval"] <= 1 and 0 <= r["val_coverage_spread"] <= 1 for r in cut.values())
    for p in service.pred_trend(s["split"]["val"][0]):
        assert p["y_pred_lo"] < p["y_pred"] < p["y_pred_hi"] and p["spread_lo"] <= p["spread_hi"]
        half = cut[f"{p['t_cut']:g}"]["interval_halfwidth"]
        assert p["y_pred_hi"] - p["y_pred"] == pytest.approx(half, rel=1e-5)


@pytest.mark.parametrize("t0", [30.0, 100.0, 180.0])
def test_forecast_features_causal(s, t0):
    """Features built from the whole batch equal features built from data up to t0, at every point <= t0."""
    from main import forecast as fc
    ts = s["ts"][s["ts"].batch.isin([1, 2])]
    full, cut = fc.frame(ts), fc.frame(ts[ts.t <= t0])
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_forecast_api(s):
    from fastapi.testclient import TestClient
    from main.api import app
    meta = read_json(ART / "forecast_meta.json")
    assert meta["val_pass"]  # beats persistence and linear extrapolation at every horizon on val
    r = TestClient(app).get(f"/batches/{s['split']['val'][0]}/forecast?t=120").json()
    assert r["t"] == 120 and len(r["path"]) == 16 and r["path"][-1]["h"] == 8
    assert all(p["lo"] < p["p"] < p["hi"] for p in r["path"])


def test_optimize(s):
    from fastapi.testclient import TestClient
    from main.api import app
    st, bid = service._state(), s["split"]["val"][0]
    r = TestClient(app).post(f"/batches/{bid}/optimize", json={}).json()
    mc = cfg()["optimize"]["max_change"]
    for f, v in r["best"].items():
        lo, hi = st["bounds"][f]
        cur = r["current"][f]
        assert max(lo, cur * (1 - mc)) - 1e-6 <= v <= min(hi, cur * (1 + mc)) + 1e-6
    assert r["y_pred_best"] - r["y_pred_current"] == pytest.approx(r["gain_kg"], abs=0.01)  # float32 predictions
    assert r["note"].startswith("Reference only") and 0 <= r["models_agree"] <= r["n_models"] == 5
    assert TestClient(app).post(f"/batches/{bid}/optimize", json={"max_change": 2}).status_code == 422
