"""Plan 009 V1-V4 / H2-H6 / C3 on synthetic fixtures (no project data, no audit answers)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nakkhu import hazard
from nakkhu.audit import core as audit_core
from nakkhu.flood import validate as V

LABELS = hazard.tier_labels()
PCOL = "glo30__breach_burn__t0p2__centroid"


# --- weighted H6 primitives -------------------------------------------------------------------


def test_h6_weights_of_one_equal_the_unweighted_fits():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 20, 400)
    y = (rng.random(400) < 1 / (1 + np.exp(-(1 - 0.4 * x)))).astype(float)
    ones = np.ones(400)
    np.testing.assert_allclose(hazard.fit_logistic(x, y, ones), hazard.fit_logistic(x, y))
    for a, b in zip(hazard.fit_isotonic(x, y, ones), hazard.fit_isotonic(x, y), strict=True):
        np.testing.assert_allclose(a, b)
    assert hazard.brier(y * 0.5, y, ones) == pytest.approx(hazard.brier(y * 0.5, y))
    assert hazard.youden_threshold(x, y, ones) == pytest.approx(hazard.youden_threshold(x, y))
    blocks = rng.integers(0, 8, 400)
    for m in V.H6_METHODS:
        np.testing.assert_allclose(hazard.calibrate_out_of_fold(x, y, blocks, m, ones),
                                   hazard.calibrate_out_of_fold(x, y, blocks, m))
    assert hazard.fit_logistic.__kwdefaults__ == {"l2": 1e-6, "max_iter": 100, "tol": 1e-10}


def test_design_weights_undo_oversampled_positives():
    """A stratified sample that oversamples affected buildings inflates an unweighted
    P(affected | HAND); the design-weighted fit recovers the population curve."""
    rng = np.random.default_rng(2)
    n = 60_000
    x = rng.uniform(0, 20, n)
    y = rng.random(n) < 1 / (1 + np.exp(-(-1.0 - 0.3 * x)))
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    take = np.r_[rng.choice(pos, 400, replace=False), rng.choice(neg, 400, replace=False)]
    w = np.where(y[take], len(pos) / 400, len(neg) / 400)
    a_w, b_w = hazard.fit_logistic(x[take], y[take].astype(float), w)
    a_u, _ = hazard.fit_logistic(x[take], y[take].astype(float))
    assert abs(a_w - -1.0) < 0.35 and abs(b_w - -0.3) < 0.08
    assert a_u > a_w + 1.0                                     # unweighted level is far too high
    p_w = hazard.predict_isotonic(hazard.fit_isotonic(x[take], y[take].astype(float), w),
                                  np.array([1.0]))[0]
    truth = float(y[x < 2].mean())
    assert abs(p_w - truth) < 0.1


def test_h6_calibration_weighted_and_primary_method():
    rng = np.random.default_rng(3)
    hand = rng.uniform(0, 20, 2000)
    y = rng.random(2000) < 1 / (1 + np.exp(-(0.5 - 0.35 * hand)))
    blocks = rng.integers(0, 12, 2000)
    tiers = hazard.tier_index(hand)
    w = np.where(y, 1.0, 3.0)
    out = V.h6_calibration(hand, y, blocks, tiers, LABELS, w)
    assert out["weighted"] and out["lower_brier_method"] in V.H6_METHODS
    for d in out["methods"].values():
        assert sum(d["class_shares_weighted"].values()) == pytest.approx(1.0)
        assert len(d["reliability"]) == 10
    lo = out["fit_all_rows"]["logistic"]["p_at_hand_m"]
    assert lo["0.0"] > lo["12.0"]
    assert V.h6_primary_method({"logistic": 0.1, "isotonic": 0.1}) == "logistic"
    assert V.h6_primary_method({"logistic": 0.11, "isotonic": 0.1}) == "isotonic"
    assert V.h6_primary_method({"logistic": float("nan"), "isotonic": 0.1}) == "isotonic"
    assert V.reliability.__defaults__ == (10,)               # pinned by prereg v5


# --- V3 pieces --------------------------------------------------------------------------------


def test_weighted_cumulative_skill_and_roc_curve():
    rng = np.random.default_rng(4)
    hand = rng.uniform(0, 20, 500)
    y = rng.random(500) < 0.3
    plain = V.cumulative_skill(hand, y)
    ones = V.cumulative_skill(hand, y, w=np.ones(500))
    pd.testing.assert_frame_equal(plain, ones)
    w = np.where(y, 1.0, 4.0)
    weighted = V.cumulative_skill(hand, y, w=w)
    assert weighted["weighted"].all() and (weighted["tp"] == plain["tp"]).all()
    expect_far = weighted["fp_w"] / (weighted["tp_w"] + weighted["fp_w"])
    np.testing.assert_allclose(weighted["FAR"], expect_far)
    some = plain["FAR"].notna()
    assert (weighted["FAR"][some] > plain["FAR"][some]).all()   # negatives count 4x
    pd.testing.assert_frame_equal(V.roc_curve(-hand, y), V.roc_curve(-hand, y, np.ones(500)))
    assert V.roc_curve(-hand, y, w)["tpr"].iloc[-1] == pytest.approx(1.0)


def test_odds_ratios_report_separation_instead_of_spurious_values():
    rng = np.random.default_rng(5)
    tiers = rng.integers(0, 6, 3000)
    p = np.array([0.0, 0.35, 0.2, 0.12, 0.08, 0.05])[tiers]
    y = (rng.random(3000) < p).astype(float)
    clusters = rng.integers(0, 30, 3000)
    ors = V.tier_odds_ratios(tiers, y, clusters, LABELS)
    first = ors.loc[ors["tier"] == "on_drainage"].iloc[0]
    assert first["separated"] and first["odds_ratio"] == 0.0 and np.isnan(first["or_lo"])
    keep = tiers != 0
    sub = V.tier_odds_ratios(tiers[keep], y[keep], clusters[keep], LABELS)
    np.testing.assert_allclose(ors.loc[~ors["separated"], "odds_ratio"].to_numpy(),
                               sub["odds_ratio"].to_numpy())
    assert V.monotone(ors) is False                         # 0 then > 0 is not non-increasing
    y_ref = y.copy()
    y_ref[tiers == 5] = 0.0
    y_ref[tiers == 0] = 1.0
    undefined = V.tier_odds_ratios(tiers, y_ref, clusters, LABELS)
    assert undefined.attrs["reference_separated"] == "none_affected"
    assert V.monotone(undefined) is None
    c3 = V.c3_outcome({"delta": 0.1, "boot_lo": 0.02, "boot_hi": 0.2}, {}, undefined)
    assert c3["pass"] is None and c3["ors_monotone"] is None
    c3 = V.c3_outcome({"delta": 0.1, "boot_lo": -0.02, "boot_hi": 0.2}, {}, undefined)
    assert c3["pass"] is False
    inf_first = pd.DataFrame({"odds_ratio": [float("inf"), 3.0, 1.0]})
    assert V.monotone(inf_first) is True


def test_config_skill_ranges_use_routing_gate_members_only():
    grid = pd.DataFrame({
        "surface": ["a__centroid", "b__centroid", "c__centroid", "a__footprint_min"],
        "auc": [0.7, 0.6, 0.95, 0.65], "pr_auc": [0.3, 0.2, 0.9, 0.25],
        "is_primary": [True, False, False, False],
        "in_sensitivity_range": [True, True, False, True],
        "statistic": ["centroid", "centroid", "centroid", "footprint_min"],
        "dem": ["glo30", "fabdem", "gedtm30", "glo30"],
        "conditioning": ["breach_burn"] * 4, "threshold": ["drop"] * 4})
    ranges = V.config_skill_ranges(grid).set_index("group")
    allrow = ranges.loc["all surfaces x statistics"]
    assert allrow["n_members"] == 3 and allrow["auc_max"] == 0.7 and allrow["auc_min"] == 0.6
    assert ranges.loc["excluded (diagnostic or routing gate)", "excluded_surfaces"] == \
        "c__centroid"
    assert ranges.loc["centroid, dem=fabdem", "auc_primary"] == 0.7


def test_jsonable_handles_numpy_and_frames():
    value = {"a": np.int64(3), 1: np.float32(0.5), "t": (np.bool_(True),),
             "f": pd.DataFrame({"x": np.arange(2)}), "arr": np.arange(3)}
    text = json.dumps(V.jsonable(value), sort_keys=True)
    assert json.loads(text)["f"] == [{"x": 0}, {"x": 1}]


# --- prerequisites ----------------------------------------------------------------------------


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prerequisites_check_sample_inventory_and_stale_estimates(project):
    text = " ".join(V.prerequisites(project, None, []))
    assert "sample.json missing" in text and "channel lines" in text
    _write_json(project.paths.root / "audit" / "flood_trace" / "sample.json",
                {"inputs": {"inventory": "google_v3"}})
    text = " ".join(V.prerequisites(project, None, []))
    assert "drawn on 'google_v3'" in text and "items.parquet missing" in text
    items = audit_core.items_path(project, "flood_trace")
    pd.DataFrame({"item_id": ["ft1"]}).to_parquet(items)
    _write_json(project.paths.artifacts / V.ESTIMATES,
                {"material": {"items": audit_core.sha256_file(items), "responses": None}})
    assert V.estimates_drift(project) == []
    audit_core.responses_path(project, "flood_trace").write_text("{}\n", encoding="utf-8")
    assert V.estimates_drift(project) == ["responses"]
    text = " ".join(V.prerequisites(project, None, ["detector_final"]))
    assert "computed from other flood_trace responses" in text
    assert "building HAND table for detector_final missing" in text


def test_file_hash_drift_is_caught(project):
    from nakkhu.evaluate.guard import sha256_file

    flood = project.paths.artifacts / "flood"
    table = flood / "change_map_buildings_detector_final.parquet"
    flood.mkdir(parents=True)
    pd.DataFrame({"building_id": ["x"]}).to_parquet(table)
    _write_json(flood / "change_map_buildings_detector_final.json", {})
    _write_json(project.paths.root / "audit" / "flood_trace" / "sample.json",
                {"inputs": {"inventory": "detector_final",
                            "change_map_file_sha256": sha256_file(table)}})
    assert V.sample_hash_drift(project) == []
    pd.DataFrame({"building_id": ["y"]}).to_parquet(table)
    assert V.sample_hash_drift(project) == ["change_map_file_sha256"]


def test_flood_trace_followups_must_be_answered(project):
    from nakkhu.audit.workflow import require_followups

    cfg = {"tasks": {"flood_trace": {"followup": "damage"}}}
    log = audit_core.responses_path(project, "flood_trace")
    log.parent.mkdir(parents=True)
    yes = {"event": "answer", "slot": "ft1", "item_id": "ft1", "question": "flood_trace",
           "answer": "yes", "ms": 1}
    log.write_text(json.dumps(yes) + "\n", encoding="utf-8")
    with pytest.raises(audit_core.AuditError, match="without the damage follow-up"):
        require_followups(project, cfg, "flood_trace")
    dmg = {**yes, "question": "damage", "answer": "no"}
    log.write_text(json.dumps(yes) + "\n" + json.dumps(dmg) + "\n", encoding="utf-8")
    require_followups(project, cfg, "flood_trace")


def test_detection_inventory_export_keeps_boxes_at_the_val_threshold(project, monkeypatch):
    import geopandas as gpd
    import shapely

    import nakkhu.ensemble as ensemble
    from nakkhu.flood.buildings import SHIFT_KIND
    from nakkhu.hydro import detections as D

    run = project.paths.runs / "infer" / "t1"
    run.mkdir(parents=True)
    boxes = [shapely.box(i, 0, i + 5, 5) for i in range(0, 50, 10)]
    gpd.GeoDataFrame({"conf": [0.1, 0.2, 0.3, 0.4, 0.5]}, geometry=boxes,
                     crs="EPSG:32645").to_file(run / "detections.gpkg", layer="obb")
    member = {"seed": 0, "train_run": "train/L5/yolo11n-obb/seed0", "infer_val": "infer/v",
              "eval_val": "eval/v", "threshold": 0.3, "infer_test": "infer/t1"}
    monkeypatch.setattr(ensemble, "final_members", lambda p, a, m, s: [member])
    entry = {"path": "data/interim/inventories/seed0.parquet",
             "detections": {"arm": "L5", "model": "yolo11n-obb", "seed": 0}}
    meta = D.export_one(project, "seed0", entry, project.paths.resolve)
    out = gpd.read_parquet(project.paths.resolve(entry["path"]))
    assert meta["n_boxes_kept"] == 3 and (out["conf"] >= 0.3).all()
    assert json.loads(project.paths.resolve(entry["path"]).with_suffix(".json").read_text(
        encoding="utf-8"))["infer_test"] == "infer/t1"
    assert D.detection_entries({"inventories": {"seed0": entry, "osm": {"path": "x"}}}) == {
        "seed0": entry}
    member["infer_test"] = "infer/missing"
    with pytest.raises(D.DetectionInventoryError, match="nakkhu georef"):
        D.export_one(project, "seed0", entry, project.paths.resolve)
    assert all(SHIFT_KIND[n] is None for n in ("seed0", "seed4", "L2_seed2", "overture"))


def test_change_map_by_inventory_promotion_merges_by_inventory():
    from nakkhu.flood.pipeline import merge_by_inventory

    old = pd.DataFrame({"inventory": ["google_v3", "microsoft", "osm"],
                        "n_buildings": [10, 20, 30]})
    one = pd.DataFrame({"inventory": ["microsoft"], "n_buildings": [21], "extra": [1.0]})
    merged = merge_by_inventory(one, old)
    assert merged["inventory"].tolist() == ["google_v3", "microsoft", "osm"]
    assert merged.set_index("inventory")["n_buildings"].to_dict() == {
        "google_v3": 10, "microsoft": 21, "osm": 30}
    added = merge_by_inventory(pd.DataFrame({"inventory": ["detector_final"],
                                             "n_buildings": [5]}), merged)
    assert added["inventory"].tolist()[-1] == "detector_final" and len(added) == 4
    assert merge_by_inventory(one, None).equals(one)


# --- end to end -------------------------------------------------------------------------------


def _synthetic(project, drop: bool, seed: int = 6):
    """A synthetic detector_final world: buildings, HAND table with three surfaces, change map,
    sitting-3 sample, answers (with Unsure and damage follow-ups) and audit estimates."""
    rng = np.random.default_rng(seed)
    n = 1500
    x, y = rng.uniform(0, 3000, n), rng.uniform(0, 3000, n)
    hand = np.clip(rng.gamma(2.0, 4.0, n) - 1.0, -0.5, None)
    truth = rng.random(n) < 1 / (1 + np.exp(-(0.5 - 0.4 * hand)))
    uid = np.array([f"detector_final:{i}" for i in range(n)])
    pred = pd.DataFrame({
        "building_uid": uid, PCOL: hand.astype(np.float32),
        "fabdem__breach_burn__t0p2__centroid": (hand + rng.normal(0, 1.5, n)).astype(np.float32),
        "gedtm30__breach_burn__t0p2__centroid": rng.uniform(0, 20, n).astype(np.float32),
        "x": x, "y": y, "dist_channel_m": hand * 25 + rng.uniform(0, 150, n),
        "elev_glo30_m": 1300 + hand + rng.normal(0, 3, n),
        "asf_hand_m": hand + rng.normal(0, 2, n),
        "block_id": [f"b{int(a // 600)}_{int(b // 600)}" for a, b in zip(x, y, strict=True)]})
    observable = rng.random(n) > 0.1
    cls = np.where(~observable, "not_observable",
                   np.where(truth ^ (rng.random(n) < 0.08), "positive", "negative"))
    change = pd.DataFrame({"building_id": uid, "class": cls, "observable": observable,
                           "class_all_strips": cls, "observable_all_strips": observable})
    obs = np.flatnonzero(observable)
    strata = np.where(cls[obs] == "positive", "positive",
                      np.where(pred["dist_channel_m"].to_numpy()[obs] <= 300, "negative_near",
                               "negative_far"))
    rows = []
    for s in ("positive", "negative_near", "negative_far"):
        members = obs[strata == s]
        pick = rng.choice(members, min(70, len(members)), replace=False)
        rows += [{"item_id": f"ft{len(rows) + k}", "building_id": uid[i], "stratum": s,
                  "frame_n": len(members), "design_n": len(pick)} for k, i in enumerate(pick)]
    items = pd.DataFrame(rows)
    task = audit_core.task_dir(project, "flood_trace")
    task.mkdir(parents=True)
    items.to_parquet(audit_core.items_path(project, "flood_trace"), index=False)
    by_uid = dict(zip(uid, truth, strict=True))
    events = []
    for item, building in zip(items["item_id"], items["building_id"], strict=True):
        ans = "unsure" if rng.random() < 0.05 else ("yes" if by_uid[building] else "no")
        events.append({"event": "answer", "slot": item, "item_id": item,
                       "question": "flood_trace", "answer": ans, "ms": 1500})
        if ans == "yes":
            events.append({"event": "answer", "slot": item, "item_id": item,
                           "question": "damage", "ms": 900,
                           "answer": "yes" if rng.random() < 0.4 else "no"})
    audit_core.responses_path(project, "flood_trace").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    flood = project.paths.artifacts / "flood"
    flood.mkdir(parents=True)
    change.to_parquet(flood / "change_map_buildings_detector_final.parquet", index=False)
    _write_json(flood / "change_map_buildings_detector_final.json", {})
    _write_json(task / "sample.json", {"inputs": {"inventory": "detector_final"}})
    lo = 0.42 if drop else 0.61
    _write_json(project.paths.artifacts / V.ESTIMATES, {
        "material": {"items": "i", "responses": "r"},
        "estimates": {"ppv": {"yes": 60, "n": 70, "estimate": 0.86, "wilson_lo": lo,
                              "wilson_hi": 0.93},
                      "npv": {"estimate": 0.9}, "drop_change_map": drop, "not_observable": 4}})
    hand_dir = project.paths.runs / "hand-attribute" / "synthetic"
    hand_dir.mkdir(parents=True)
    table = hand_dir / "building_hand_detector_final.parquet"
    pred[[c for c in pred.columns if c == "building_uid" or "__" in c]].to_parquet(table)
    pd.DataFrame({"inventory": "detector_final",
                  "cfg_id": ["glo30__breach_burn__t0p2", "fabdem__breach_burn__t0p2",
                             "gedtm30__breach_burn__t0p2"],
                  "statistic": "centroid", "dem": ["glo30", "fabdem", "gedtm30"],
                  "conditioning": "breach_burn", "threshold": "drop", "threshold_km2": 0.2,
                  "diagnostic": False, "routing_gate_pass": [True, True, False],
                  "in_sensitivity_range": [True, True, False]}).to_csv(
        hand_dir / "count_stability.csv", index=False)
    return pred, table


@pytest.fixture
def wired(project, monkeypatch):
    """Patch only the project-specific loaders (sweep config, predictors, guard state)."""
    import nakkhu.evaluate.guard as guard

    state: dict[str, Any] = {}
    monkeypatch.setattr(V, "_sweep_primary", lambda p, c: (
        project.paths.artifacts / "detector" / "inventory_final.parquet", PCOL,
        [0.0, 3.0, 6.0, 9.0, 12.0]))
    monkeypatch.setattr(V, "prerequisites", lambda *a, **k: state.get("problems", []))
    monkeypatch.setattr(V, "predictors", lambda p, inv, c: state["pred"].copy())
    monkeypatch.setattr(V, "hand_table_path", lambda p, inv: state["table"])
    monkeypatch.setattr(guard, "require_frozen", lambda p: {"prereg_sha256": "p" * 64,
                                                            "split_sha256": "s" * 64})
    return state


def _journal(project) -> list[dict]:
    path = project.paths.runs / "eval" / "test_journal.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] \
        if path.is_file() else []


def test_run_validate_end_to_end_primary_r1(project, wired):
    wired["pred"], wired["table"] = _synthetic(project, drop=False)
    run, summary = V.run_validate(project, ["detector_final"], n_boot=40)
    journal = _journal(project)
    assert len(journal) == 1 and journal[0]["run"] == "flood/validate:detector_final"
    art = project.paths.artifacts
    for key in V.VALIDATION_ARTIFACTS:
        assert (art / "validation" / f"{key}.csv").is_file(), key
    on_disk = json.loads((art / "validation" / "validation_summary.json").read_text(
        encoding="utf-8"))
    prim = on_disk["primary"]
    assert prim["reference"] == "R1" and prim["surface"] == PCOL
    assert prim["h6_method"] in V.H6_METHODS and prim["C3"]["pass"] in (True, False, None)
    refs = on_disk["inventories"]["detector_final"]["references"]
    assert set(refs) == {"R1", "R1-all-strips", "R1-aud"}
    assert refs["R1-aud"]["weighted"] and not refs["R1"]["weighted"]
    assert refs["R1"]["auc"] > 0.6                          # HAND carries the synthetic signal
    v2 = on_disk["inventories"]["detector_final"]["V2_damage"]
    assert v2["n"] > 0 and len(v2["by_tier"]) == len(LABELS)
    v1 = json.loads((art / "flood" / "change_map_audit_summary.json").read_text(
        encoding="utf-8"))
    assert v1["r1_used"] and v1["inventory"] == "detector_final"
    by_inv = pd.read_csv(art / "validation" / "by_inventory.csv")
    assert by_inv.loc[by_inv["primary_reference"], "reference"].tolist() == ["R1"]
    assert {"inventory", "reference", "auc", "pr_auc"} <= set(by_inv.columns)  # plan 010 D4
    grid = pd.read_csv(art / "validation" / "config_skill_grid.csv")
    assert grid.loc[grid["surface"].str.startswith("gedtm30"), "in_sensitivity_range"].eq(
        False).all()
    ranges = pd.read_csv(art / "validation" / "config_skill_range.csv")
    assert (ranges.loc[ranges["group"] == "all surfaces x statistics", "n_members"] == 2).all()
    comp = pd.read_csv(art / "validation" / "competitors_auc.csv")
    r1_comp = comp.loc[comp["reference"] == "R1"]
    assert r1_comp["delong_p"].notna().all()
    assert comp.loc[comp["reference"] == "R1-aud", "delong_p"].isna().all()
    cal = pd.read_csv(art / "validation" / "calibration_h6.csv")
    assert cal.groupby("reference")["lower_brier"].sum().eq(1).all()


def test_run_validate_uses_r1_aud_when_the_drop_rule_fires(project, wired):
    wired["pred"], wired["table"] = _synthetic(project, drop=True)
    _, summary = V.run_validate(project, ["detector_final"], n_boot=30)
    inv = summary["inventories"]["detector_final"]
    assert not inv["r1_usable"] and set(inv["references"]) == {"R1-aud"}
    assert summary["primary"]["reference"] == "R1-aud"
    assert summary["V1"]["drop_change_map"] and not summary["V1"]["r1_used"]


def test_run_validate_refuses_or_fails_before_journalling(project, wired):
    wired["pred"], wired["table"] = _synthetic(project, drop=False)
    wired["problems"] = ["sitting 3: flood_trace is not complete"]
    with pytest.raises(RuntimeError, match="refused"):
        V.run_validate(project, ["detector_final"], n_boot=10)
    wired["problems"] = []
    wired["pred"] = wired["pred"].iloc[:-1]                 # HAND table lost a building
    with pytest.raises(ValueError, match="different buildings"):
        V.run_validate(project, ["detector_final"], n_boot=10)
    assert _journal(project) == []


# --- competitor re-run after the CRS fix (plans/competitor_auc_bug.md) -------------------------


@pytest.fixture
def rerun(project, wired, monkeypatch):
    """A validate run made with the journal-47 bug (both raster competitors constant 0), the
    test hold released and the guard state of that run; the fixed predictors are then live."""
    import nakkhu.evaluate.guard as guard

    good, wired["table"] = _synthetic(project, drop=False)
    wired["pred"] = good.assign(elev_glo30_m=0.0, asf_hand_m=0.0)
    run, _ = V.run_validate(project, ["detector_final"], n_boot=30)
    wired["pred"] = good
    (project.paths.runs / "eval" / "test_hold.yaml").write_text("hold: false\nby: test\n",
                                                               encoding="utf-8")
    state = {"prereg_sha256": "p" * 64, "split_sha256": "s" * 64, "problems": []}
    monkeypatch.setattr(guard, "verify", lambda p: dict(state))
    wired["guard"] = state
    return run


def _sha(path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rerun_competitors_recomputes_only_the_raster_rows_once(project, wired, rerun):
    promoted = project.paths.artifacts / "validation" / "competitors_auc.csv"
    before = _sha(promoted), _sha(rerun.path / "competitors_auc.csv")
    reason = "competitor CRS fix, user approved 2026-10-17"
    dry = V.rerun_competitors(project, rerun.ref, reason=reason, dry_run=True)
    assert dry["ready"] and dry["dry_run"] and dry["source_journal_entry"] == 1
    assert dry["predictors"]["detector_final"]["asf_hand_m"]["finite"] == 1500
    assert len(_journal(project)) == 1                       # a dry run is no test access
    res = V.rerun_competitors(project, rerun.ref, reason=reason)
    journal = _journal(project)
    assert len(journal) == 2 and res["journal_entry"] == 2
    assert journal[1]["run"] == f"flood/validate-competitors:{rerun.run_id}"
    assert reason in journal[1]["note"] and res["run"] in journal[1]["note"]
    check = res["reproduction"]
    assert check["reproduced"] and check["unchanged_rows"] == 6 == check["corrected_rows"]
    new = pd.read_csv(project.paths.runs / res["run"] / "competitors_auc.csv")
    old = pd.read_csv(rerun.path / "competitors_auc.csv")
    fixed = old["competitor"].isin(V.RERUN_CORRECTED)
    np.testing.assert_allclose(old.loc[fixed, "auc_competitor"], 0.5, atol=1e-9)
    assert (abs(new.loc[fixed, "auc_competitor"] - 0.5) > 0.01).all()
    r1 = new.loc[(new["reference"] == "R1") & fixed]
    assert r1["delong_p"].notna().all() and (r1["auc_competitor"] > 0.55).all()
    assert (_sha(promoted), _sha(rerun.path / "competitors_auc.csv")) == before
    again = V.rerun_competitors(project, rerun.ref, reason=reason)
    assert not again["ready"] and any("already re-run" in p for p in again["problems"])
    assert len(_journal(project)) == 2


def test_rerun_competitors_refuses_before_any_access(project, wired, rerun):
    import os

    def problems(source=rerun.ref, reason="fix"):
        out = V.rerun_competitors(project, source, reason=reason)
        assert not out["ready"]
        return " | ".join(out["problems"])

    assert "--reason" in problems(reason=" ")
    assert "not a validate run" in problems(source="eval/whatever")
    wired["guard"]["prereg_sha256"] = "q" * 64
    assert "settings changed since" in problems()
    wired["guard"]["prereg_sha256"] = "p" * 64
    cm = project.paths.artifacts / "flood" / "change_map_buildings_detector_final.parquet"
    later = cm.stat().st_mtime + 3600
    os.utime(cm, (later, later))
    assert "change_map:detector_final" in problems()
    os.utime(cm, (later - 3600, later - 3600))
    (project.paths.runs / "eval" / "test_hold.yaml").write_text("hold: true\n", encoding="utf-8")
    assert "hold:" in problems()
    assert len(_journal(project)) == 1


def test_compare_competitor_tables_flags_any_untouched_difference(tmp_path):
    rows = [{"competitor": c, "auc_hand": 0.8, "auc_competitor": 0.5 if c in V.RERUN_CORRECTED
             else 0.7, "n": 100, "n_blocks": 9, "n_boot_valid": 30, "inventory": "inv",
             "reference": "R1"} for c in ("distance_to_channel", "glo30_elevation",
                                          "river_buffer_100m", "asf_glo30_hand")]
    old = pd.DataFrame(rows)
    old.to_csv(tmp_path / "old.csv", index=False)
    fixed = old["competitor"].isin(V.RERUN_CORRECTED)
    good = old.assign(auc_competitor=np.where(fixed, 0.66, old["auc_competitor"]))
    good.to_csv(tmp_path / "good.csv", index=False)
    assert V.compare_competitor_tables(tmp_path / "old.csv", tmp_path / "good.csv")["reproduced"]
    drift = good.copy()
    drift.loc[0, "auc_competitor"] = 0.7000001                 # an untouched row moved
    drift.loc[1, "n"] = 99                                     # a kept column moved
    drift.to_csv(tmp_path / "drift.csv", index=False)
    out = V.compare_competitor_tables(tmp_path / "old.csv", tmp_path / "drift.csv")
    assert not out["reproduced"]
    assert out["unchanged_rows_differ"] == ["inv / R1 / distance_to_channel"]
    assert out["kept_columns_differ"] == ["inv / R1 / glo30_elevation: n 100 -> 99"]
    good.iloc[::-1].to_csv(tmp_path / "order.csv", index=False)
    assert not V.compare_competitor_tables(tmp_path / "old.csv", tmp_path / "order.csv")[
        "keys_equal"]


# --- thinned ROC / PR curves (nakkhu.flood.curves) ---------------------------------------------


def test_rdp_keeps_endpoints_and_corners_and_bounds_dropped_points():
    from nakkhu.flood import curves as C

    x = np.linspace(0, 1, 201)
    xy = np.c_[x, np.where(x < 0.5, 2 * x, 1.0)]        # a corner at (0.5, 1)
    keep = C.rdp_keep(xy, 1e-6)
    assert keep[0] and keep[-1] and keep[100] and keep.sum() == 3
    rng = np.random.default_rng(0)
    noisy = np.c_[x, np.sqrt(x) + rng.normal(0, 1e-3, len(x))]
    tol = 2e-3
    keep = C.rdp_keep(noisy, tol)
    kept = np.flatnonzero(keep)
    for a, b in zip(kept[:-1], kept[1:], strict=True):   # every dropped point within tol
        p, q = noisy[a], noisy[b]
        for m in range(a + 1, b):
            d = q - p
            t = np.clip(np.dot(noisy[m] - p, d) / np.dot(d, d), 0, 1)
            assert np.hypot(*(noisy[m] - (p + t * d))) <= tol + 1e-12
    assert np.array_equal(keep, C.rdp_keep(noisy, tol))   # deterministic


def test_thin_table_caps_points_and_preserves_auc():
    from nakkhu.flood import curves as C

    rng = np.random.default_rng(1)
    y = rng.random(20_000) < 0.2
    score = rng.normal(0, 1, len(y)) + 0.8 * y
    w = rng.uniform(0.5, 2.0, len(y))
    parts = []
    for ref, weights in (("R1", None), ("R1-aud", w)):
        parts.append(V.roc_curve(score, y, weights).assign(inventory="inv", reference=ref))
    frame = pd.concat(parts, ignore_index=True)
    thin, records = C.thin_table(frame, "fpr", "tpr", tol0=1e-5, cap=300)
    assert list(thin.columns) == list(frame.columns)
    for rec, weights in zip(records, (None, w), strict=True):
        assert rec["kept"] <= 300 < rec["points"] and rec["tolerance"] > 1e-5
        assert rec["auc_full"] == pytest.approx(V.roc_auc(score, y, weights), abs=1e-9)
        assert abs(rec["auc_thin"] - rec["auc_full"]) < 1e-3
        part = thin.loc[thin["reference"] == rec["reference"]]
        full = frame.loc[frame["reference"] == rec["reference"]]
        assert part.iloc[0].equals(full.iloc[0]) and part.iloc[-1].equals(full.iloc[-1])
    pr = V.pr_curve(score, y).assign(inventory="inv", reference="R1")
    pthin, (prec,) = C.thin_table(pr, "recall", "precision", cap=500)
    assert prec["kept"] <= 500 and len(pthin) == prec["kept"]
    assert prec["ap_full"] == pytest.approx(V.average_precision(score, y), abs=1e-12)


# --- competitor raster sampling (plans/competitor_auc_bug.md) -------------------------------


def _lonlat_raster(path: Any, nodata: float | None = None) -> None:
    """EPSG:4326 float raster over the Nakkhu area whose value is a function of the cell."""
    import rasterio
    from rasterio.transform import from_origin

    res = 1 / 3600
    rows, cols = np.mgrid[0:120, 0:150]
    band = (1300.0 + rows + 0.01 * cols).astype(np.float32)
    if nodata is not None:
        band[:10, :10] = nodata
    with rasterio.open(path, "w", driver="GTiff", height=120, width=150, count=1,
                       dtype="float32", crs="EPSG:4326", nodata=nodata,
                       transform=from_origin(85.30, 27.66, res, res)) as ds:
        ds.write(band, 1)


def test_sample_raster_reprojects_points_to_the_raster_crs(tmp_path):
    import geopandas as gpd
    import rasterio
    from shapely.geometry import Point

    path = tmp_path / "dem.tif"
    _lonlat_raster(path)
    res = 1 / 3600
    lonlat = [(85.30 + (c + 0.5) * res, 27.66 - (r + 0.5) * res)
              for r, c in ((3, 4), (50, 70), (119, 149))]
    expect = [1300.0 + r + 0.01 * c for r, c in ((3, 4), (50, 70), (119, 149))]
    pts = gpd.GeoSeries([Point(p) for p in lonlat], crs="EPSG:4326").to_crs("EPSG:32645")
    x, y = pts.x.to_numpy(), pts.y.to_numpy()
    got = V._sample_raster(path, x, y, pts.crs)
    np.testing.assert_allclose(got, expect, rtol=0, atol=1e-3)
    # the pre-fix call (metres straight into a degree grid) read the fill value 0 everywhere
    with rasterio.open(path) as ds:
        old = np.array([v[0] for v in ds.sample(zip(x, y, strict=True))], dtype=float)
    assert (old == 0).all()
    # same CRS: no reprojection, same cells
    np.testing.assert_allclose(V._sample_raster(path, *np.array(lonlat).T, "EPSG:4326"),
                               expect, atol=1e-3)


def test_sample_raster_outside_and_nodata_are_nan_and_all_outside_raises(tmp_path):
    path = tmp_path / "hand.tif"
    _lonlat_raster(path, nodata=-9999.0)
    res = 1 / 3600
    x = np.array([85.30 + 2.5 * res, 85.30 + 60.5 * res, 84.0])     # nodata, valid, outside
    y = np.array([27.66 - 2.5 * res, 27.66 - 60.5 * res, 27.0])
    got = V._sample_raster(path, x, y, "EPSG:4326")
    assert np.isnan(got[0]) and np.isnan(got[2])
    assert got[1] == pytest.approx(1300.0 + 60 + 0.6, abs=1e-3)
    with pytest.raises(ValueError, match="no point lands"):
        V._sample_raster(path, np.array([330_000.0]), np.array([3_060_000.0]), "EPSG:4326")
