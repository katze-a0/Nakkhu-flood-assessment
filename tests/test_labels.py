"""Plan 004: consensus weak labels (Steps 1-7) and the OSM firewall."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import shapely.affinity

from nakkhu.labels import arms, gate, harmonise, match, obb, register, sam2_refine, selftrain

LABELS_PKG = Path(__file__).resolve().parents[1] / "src" / "nakkhu" / "labels"
FORBIDDEN = ("osm", "overture", "labelqa", "openstreetmap")


# --- Step 1: no OSM / Overture in any label-building code path ------------------------------


def test_labels_package_never_touches_osm_or_overture():
    for path in sorted(LABELS_PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                for name in names:
                    assert not any(f in name.lower() for f in FORBIDDEN), (path.name, name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.lower()
                assert "fp_osm" not in text and "fp_overture" not in text, path.name
    assert set(harmonise.LABEL_SOURCES.values()) == {"fp_google_v3", "fp_microsoft"}


def test_source_path_rejects_reference_sources(project):
    for key in ("osm", "overture"):
        with pytest.raises(harmonise.HarmoniseError):
            harmonise.source_path(project, key)


def test_explode_clean_and_thresholds():
    multi = shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(2, 0, 3, 1)])
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    frame = gpd.GeoDataFrame({"source_id": ["a", "b", "c"]},
                             geometry=[multi, bowtie, shapely.Polygon()], crs="EPSG:32645")
    out, stats = harmonise.explode_clean(frame)
    assert stats["invalid_repaired"] == 1
    assert out.geometry.is_valid.all() and not out.geometry.is_empty.any()
    assert (out["source_id"] == "a").sum() == 2
    cells = gpd.GeoDataFrame({"s2_token": ["x"], "t80": [0.7], "t85": [0.74], "t90": [0.79]},
                             geometry=[shapely.box(85, 27, 86, 28)], crs="EPSG:4326")
    google = pd.DataFrame({"latitude": [27.5, 10.0], "longitude": [85.5, 10.0]})
    joined = harmonise.attach_thresholds(google, cells, {"t90": 0.75})
    assert joined["t90"].tolist() == [0.79, 0.75]
    assert joined["threshold_source"].tolist() == ["s2_cell", "fallback"]


# --- Step 2: match classes ------------------------------------------------------------------


def _match(google, microsoft, conf=None):
    conf = np.full(len(google), 0.9) if conf is None else np.asarray(conf)
    return match.match(np.array(google, dtype=object), conf, np.full(len(google), 0.8),
                       np.array(microsoft, dtype=object), iou_min=0.3, contain_min=0.5)


def test_one_to_one_one_to_three_and_disjoint():
    one_g = [shapely.box(0, 0, 10, 10)]
    one_m = [shapely.box(1, 1, 11, 11)]
    three_g = [shapely.box(100 + 4 * i, 0, 103 + 4 * i, 10) for i in range(3)]
    block_m = [shapely.box(100, 0, 112, 10)]
    lone_g = [shapely.box(500, 500, 510, 510), shapely.box(600, 600, 610, 610)]
    lone_m = [shapely.box(900, 900, 910, 910)]
    res = _match(one_g + three_g + lone_g, one_m + block_m + lone_m,
                 conf=[0.9, 0.9, 0.9, 0.9, 0.95, 0.7])
    assert res.google["cls"].tolist() == ["agree", "split", "split", "split",
                                          "lone_google_hi", "lone_google_lo"]
    assert res.microsoft["cls"].tolist() == ["agree", "split", "lone_ms"]
    merged = _match([shapely.box(0, 0, 20, 10)], [shapely.box(0, 0, 9, 10),
                                                  shapely.box(11, 0, 20, 10)])
    assert merged.google["cls"].tolist() == ["merge"]
    assert merged.microsoft["cls"].tolist() == ["merge", "merge"]


def test_arm_rules():
    g = [shapely.box(0, 0, 10, 10), shapely.box(50, 0, 60, 10), shapely.box(80, 0, 90, 10)]
    m = [shapely.box(1, 1, 11, 11), shapely.box(51, 0, 61, 10), shapely.box(200, 0, 210, 10)]
    res = _match(g, m, conf=[0.9, 0.7, 0.7])   # agree (hi), agree (lo), lone_lo; lone_ms
    l1 = arms.select("google_t90", res)
    assert l1.index.tolist() == [0]
    l3 = arms.select("union", res)
    assert sorted(zip(l3.source, l3.index, strict=True)) == [
        ("google", 0), ("microsoft", 1), ("microsoft", 2)]
    l4 = arms.select("intersection", res)
    assert l4.index.tolist() == [0, 1]
    l5 = arms.select("consensus", res)
    assert sorted(zip(l5.source, l5.index, strict=True)) == [("google", 0), ("google", 1)]
    assert set(l5.action) == {"keep"}
    # Conflicts keep Google's split units; the Microsoft blob is not used.
    split = _match([shapely.box(4 * i, 0, 3 + 4 * i, 10) for i in range(3)],
                   [shapely.box(0, 0, 12, 10)])
    l5s = arms.select("consensus", split)
    assert list(l5s.source) == ["google"] * 3 and l5s.index.tolist() == [0, 1, 2]


def test_arm_configs_resolve():
    root = Path(__file__).resolve().parents[1]
    for arm in arms.ARMS:
        cfg = arms.arm_config(root, arm).values
        assert cfg["arm"] == arm and cfg["match"]["iou_min"] == 0.3
    assert arms.arm_config(root, "L7").values["use_gate"] is True


# --- Step 3: registration -------------------------------------------------------------------


def test_registration_recovers_synthetic_shift():
    rng = np.random.default_rng(0)
    image = np.full((640, 640, 3), 90.0)
    footprints = []
    for _ in range(40):
        x, y = rng.integers(20, 580, 2)
        w, h = rng.integers(10, 40, 2)
        image[y:y + h, x:x + w] = rng.uniform(150, 250)
        footprints.append(shapely.box(x - 5, y + 3, x + w - 5, y + h + 3))  # offset (-5, +3)
    image += rng.normal(0, 8, image.shape)
    reg = register.register_tile(image, None, np.array(footprints, dtype=object),
                                 max_shift_px=12, min_peak_ratio=1.0)
    assert reg.ok
    assert abs(reg.dx - 5) <= 1 and abs(reg.dy + 3) <= 1


def test_registration_flags_large_shift_and_empty_tiles():
    empty = register.register_tile(np.zeros((64, 64, 3)), None, np.array([], dtype=object),
                                   max_shift_px=12, min_peak_ratio=1.0)
    assert not empty.ok and empty.reason == "no_footprints" and empty.dx == 0


# --- Step 5: OBB, filters, round trip -------------------------------------------------------


def test_yolo_obb_round_trip(tmp_path):
    poly = shapely.affinity.rotate(shapely.box(100, 120, 140, 150), 27, origin="centroid")
    corners = obb.mrr_corners(poly)[None]
    path = tmp_path / "t.txt"
    obb.write_yolo_obb(path, corners, 640)
    cls, back = obb.read_yolo_obb(path, 640)
    assert cls.tolist() == [0]
    assert np.abs(back / 640 - corners / 640).max() < 1e-6
    assert shapely.Polygon(back[0]).symmetric_difference(poly).area / poly.area < 1e-4


def test_tile_filters():
    f = obb.ObbFilters()
    region = shapely.box(0, 0, 640, 640)
    polys = np.array([
        shapely.box(10, 10, 40, 40),                      # keep (225 m2)
        shapely.box(100, 100, 104, 104),                  # 4 m2 -> drop
        shapely.box(620, 10, 660, 40),                    # 50 % inside -> keep, clipped
        shapely.box(630, 100, 670, 130),                  # 25 % inside -> ignore
        shapely.Polygon([(200, 200), (260, 200), (260, 206), (206, 206), (206, 260),
                         (200, 260)]),                    # thin L -> rectangularity ignore
        shapely.box(300, 300, 340, 340),                  # rule ignore
    ], dtype=object)
    actions = np.array(["keep"] * 5 + ["ignore"], dtype=object)
    res = obb.tile_labels(polys, actions, region, 640, f)
    assert res.decisions == ["keep", "drop", "keep", "ignore", "ignore", "ignore"]
    assert res.reasons[3:] == ["outside", "rectangularity", "rule"]
    assert res.boxes.shape == (2, 4, 2) and res.boxes.max() <= 640
    mask = obb.ignore_mask(res.ignore, 640)
    assert mask[320, 320] and not mask[25, 25]
    filled = obb.fill_ignore(np.full((640, 640, 3), 7, np.uint8), mask)
    assert filled[320, 320].tolist() == [7, 7, 7]


# --- Step 6: gate ---------------------------------------------------------------------------


def test_gate_is_deterministic_and_ranks_missing_last():
    feats = pd.DataFrame({"tile_id": ["a", "b", "c", "d"], "peak_ratio": [2.0, 1.0, 1.5, np.nan],
                          "conflict_frac": [0.1, 0.5, 0.3, np.nan]})
    spec = {"peak_ratio": "higher", "conflict_frac": "lower"}
    one, two = gate.gate_scores(feats, spec), gate.gate_scores(feats, spec)
    assert np.array_equal(one["gate_score"], two["gate_score"])
    assert one.sort_values("gate_score")["tile_id"].tolist() == ["d", "b", "c", "a"]
    assert gate.admit(one, ["a", "b", "c", "d"], 0.5) == {"a", "c"}


def test_setpoint_cannot_read_test_tiles():
    curve = pd.DataFrame({"fraction": [0.5, 1.0], "f1": [0.6, 0.6],
                          "tile_ids": [["v1", "v2"], ["v1", "v2"]]})
    assert gate.choose_admit_fraction(curve, ["v1", "v2"], ["t1"]) == 1.0
    leaky = curve.assign(tile_ids=[["v1", "t1"], ["v1"]])
    with pytest.raises(gate.GateError):
        gate.choose_admit_fraction(leaky, ["v1", "v2"], ["t1"])


# --- Step 4 (no GPU) and Step 7 -------------------------------------------------------------


def test_sam2_mask_polygon_and_merge(tmp_path):
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 30:50] = True
    mask[40:42, 5:7] = True
    poly = sam2_refine.mask_polygon(mask)
    assert poly.bounds == (30.0, 10.0, 50.0, 20.0)
    prompts = pd.DataFrame({"source": ["google"], "index": [3],
                            "footprint_wkt": [shapely.box(30, 10, 50, 20).wkt]})
    rec = sam2_refine.refine_records(mask[None], prompts)[0]
    assert rec["iou"] == 1.0 and rec["mask_area_px"] == 204
    lines = [{"key": "t1", "status": "error", "error": "x"},
             {"key": "t1", "status": "ok", "results": [rec], "n_prompts": 1},
             {"key": "t2", "status": "ok", "results": [], "n_prompts": 0}]
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n{trunc", encoding="utf-8")
    results, tiles = sam2_refine.merge([path])
    assert len(results) == 1 and tiles["status"].tolist() == ["ok", "ok"]
    assert sam2_refine.read_done(path) == {"t1", "t2"}


def test_selftrain_round():
    def box(x, y):
        return np.array([[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]], float)

    labels = np.stack([box(0, 0), box(100, 100)])
    dets = np.stack([box(1, 1), box(300, 300), box(400, 400)])
    res = selftrain.self_train_tile(labels, dets, np.array([0.5, 0.95, 0.4]), add_conf=0.9,
                                    support_conf=0.1)
    assert res.dropped == 1 and res.added == 1 and len(res.boxes) == 2
    assert selftrain.conf_at_precision([0.9, 0.8, 0.7, 0.6], [1, 1, 1, 0], 0.9) == 0.7
    assert selftrain.conf_at_precision([0.9], [0], 0.9) is None


# --- labelqa (OSM reference) ----------------------------------------------------------------


def test_labelqa_matching_and_test_guard(project):
    from nakkhu import labelqa

    pred = np.array([shapely.box(0, 0, 10, 10), shapely.box(20, 0, 30, 10)], dtype=object)
    ref = np.array([shapely.box(1, 0, 11, 10), shapely.box(100, 0, 110, 10)], dtype=object)
    assert labelqa.greedy_match(pred, ref, 0.5) == (1, 1, 1)
    assert labelqa.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    with pytest.raises(labelqa.LabelQAError, match="does not exist"):
        labelqa.tile_label_quality(project, Path("x"), pd.DataFrame({"domain": []}),
                                   gpd.GeoDataFrame(geometry=[]), pd.DataFrame(), {"test"}, 640,
                                   access=("labels/x", "abc"))
    shifted = labelqa.shift_map(np.array([shapely.box(0, 0, 1, 1)], dtype=object),
                                [0.5, 0, 100.0, 0, -0.5, 200.0], 4.0, -2.0)
    assert shapely.bounds(shifted[0]).tolist() == [2.0, 1.0, 3.0, 2.0]


def test_selftrain_applies_to_an_arm_directory(tmp_path):
    l7 = tmp_path / "L7"
    corners = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], float)
    obb.write_yolo_obb(l7 / "labels" / "t1.txt", corners, 640)
    (l7 / "tiles.txt").write_text("t1\n", encoding="utf-8")
    det = pd.DataFrame({"tile_id": ["t1", "t1"], "conf": [0.5, 0.95],
                        "x1": [1, 300], "y1": [1, 300], "x2": [11, 310], "y2": [1, 300],
                        "x3": [11, 310], "y3": [11, 310], "x4": [1, 300], "y4": [11, 310]})
    stats = selftrain.apply_to_arm(l7, tmp_path / "L8", det, add_conf=0.9, support_conf=0.1,
                                   overlap_iou=0.0, tile_px=640)
    assert stats[["added", "dropped", "labels"]].iloc[0].tolist() == [1, 0, 2]
    assert len(obb.read_yolo_obb(tmp_path / "L8" / "labels" / "t1.txt", 640)[1]) == 2


def test_sam2_delta_diff_and_merge_override(tmp_path):
    cols = {"source": "google", "x0": 0.0, "y0": 0.0, "x1": 5.0, "y1": 5.0}
    old = pd.DataFrame([{"tile_id": "a", "index": 1, "footprint_wkt": "P1", **cols},
                        {"tile_id": "b", "index": 2, "footprint_wkt": "P2", **cols},
                        {"tile_id": "c", "index": 3, "footprint_wkt": "P3", **cols}])
    new = old.copy()
    new.loc[1, "footprint_wkt"] = "P2-shifted"
    new = new.loc[new["tile_id"] != "c"]
    assert sam2_refine.changed_tiles(old, new) == ["b", "c"]

    def rec(key, iou):
        return {"key": key, "status": "ok", "n_prompts": 1, "results": [
            {"source": "google", "index": 1, "iou": iou, "mask_area_px": 1, "mask_wkt": "",
             "mrr": None}]}

    base = tmp_path / "base.jsonl"
    base.write_text("\n".join(json.dumps(rec(k, 0.1)) for k in "abc") + "\n", encoding="utf-8")
    delta = tmp_path / "delta.jsonl"
    delta.write_text(json.dumps(rec("b", 0.9)) + "\n" + json.dumps(rec("a", 0.7)) + "\n",
                     encoding="utf-8")
    results, tiles = sam2_refine.merge([base], [({"b", "c"}, [delta])])
    by = dict(zip(results["tile_id"], results["iou"], strict=True))
    assert by == {"a": 0.1, "b": 0.9}             # a: base (not a delta tile); c: stale, dropped
    assert dict(zip(tiles["tile_id"], tiles["origin"], strict=True)) == {"a": "base",
                                                                          "b": "delta0"}


def test_g1_test_scoring_needs_the_plan005_freeze_and_journals_access(project, monkeypatch):
    from nakkhu import labelqa
    from nakkhu.evaluate import guard

    empty = (pd.DataFrame({"domain": []}), gpd.GeoDataFrame(geometry=[]),
             pd.DataFrame({"tile_id": []}))
    # A prereg file that exists but is not frozen (no hash artifact) is refused.
    (project.paths.root / "configs" / "prereg.yaml").write_text("split: {}\n", encoding="utf-8")
    with pytest.raises(labelqa.LabelQAError, match="not frozen"):
        labelqa.tile_label_quality(project, Path("x"), *empty, {"test"}, 640,
                                   access=("labels/x", "abc"))
    # Frozen state: still refused without a run reference, journalled with one.
    state = {"prereg_sha256": "p" * 64, "split_sha256": "s" * 64, "problems": []}
    monkeypatch.setattr(guard, "require_frozen", lambda _project: state)
    with pytest.raises(labelqa.LabelQAError, match="run reference"):
        labelqa.tile_label_quality(project, Path("x"), *empty, {"test"}, 640)
    labelqa.tile_label_quality(project, Path("x"), *empty, {"test"}, 640,
                               access=("labels/run1", "abc"))
    journal = guard.read_test_journal(project)
    assert [(e["run"], e["prereg_sha256"], e["split_sha256"]) for e in journal] == [
        ("labels/run1", "p" * 64, "s" * 64)]
    # Non-test domains never touch the guard or the journal.
    labelqa.tile_label_quality(project, Path("x"), *empty, {"train"}, 640)
    assert len(guard.read_test_journal(project)) == 1


def test_prompt_box_skips_footprints_outside_the_tile():
    inside = sam2_refine.prompt_box(shapely.box(10, 10, 30, 30), 36, 4, 640)
    assert inside is not None and inside.tolist() == [6, 6, 34, 34]
    assert sam2_refine.prompt_box(shapely.box(650, 10, 680, 40), 36, 4, 640) is None
    assert sam2_refine.prompt_box(shapely.box(10, 10, 12, 12), 36, 4, 640) is None


def test_label_quality_refuses_already_aligned_osm(project):
    from nakkhu import labelqa

    aligned = gpd.GeoDataFrame({"osm_tile": ["t"]}, geometry=[shapely.box(0, 0, 1, 1)])
    with pytest.raises(labelqa.LabelQAError, match="unshifted"):
        labelqa.tile_label_quality(project, Path("x"), pd.DataFrame({"domain": []}), aligned,
                                   pd.DataFrame({"tile_id": []}), {"train"}, 640)


def test_label_quality_shifts_osm_once_and_compares_boxes(tmp_path, project):
    from nakkhu import labelqa

    affine = [0.5, 0.0, 100.0, 0.0, -0.5, 420.0]          # tile 100..420 E, 100..420 N
    index = pd.DataFrame({"tile_id": ["t"], "domain": ["train"], "affine": [affine],
                          "bounds": [[100.0, 100.0, 420.0, 420.0]]})
    label = np.array([[[100, 100], [140, 100], [140, 140], [100, 140]]], float)  # px
    obb.write_yolo_obb(tmp_path / "labels" / "t.txt", label, 640)
    # The label in map units is E 150..170, N 350..370; OSM sits 4 px (2 m) west of it.
    osm = gpd.GeoDataFrame(geometry=[shapely.box(148, 350, 168, 370)])
    shifts = pd.DataFrame({"tile_id": ["t"], "dx": [4.0], "dy": [0.0], "reg_ok": [True]})
    q = labelqa.tile_label_quality(project, tmp_path, index, osm, shifts, {"train"}, 640)
    assert q[["tp_50", "fp_50", "fn_50"]].iloc[0].tolist() == [1, 0, 0]


def _g1_audit_case():
    tiles = pd.DataFrame({"tile_id": ["A", "B"],
                          "bounds": [[0.0, 0.0, 100.0, 100.0], [100.0, 0.0, 200.0, 100.0]]})
    obbs = {"A": np.array([shapely.box(10, 10, 20, 20)], dtype=object),
            "B": np.array([shapely.box(150, 50, 160, 60)], dtype=object)}
    rows = [("a1", "S1", 15, 15, "yes"), ("a2", "S1", 20.8, 15, "yes"),
            ("a3", "S1", 22, 15, "yes"), ("a4", "S2", 50, 50, "no"),
            ("a5", "S2", 60, 60, "unsure"), ("a6", "S2", 70, 70, "yes"),
            ("a7", "outside", 15, 15, "yes"), ("b1", "S1", 155, 55, "yes"),
            ("b2", "S2", 180, 80, "no"), ("b3", "S1", 190, 90, "no"),
            ("c1", "S1", 500, 500, "yes")]
    frame_n = {"S1": 100, "S2": 150, "outside": 0}
    items = pd.DataFrame({"item_id": [r[0] for r in rows], "stratum": [r[1] for r in rows],
                          "frame_n": [frame_n[r[1]] for r in rows],
                          "x": [float(r[2]) for r in rows], "y": [float(r[3]) for r in rows]})
    answers = {r[0]: r[4] for r in rows}
    clicked = {"a6": (12.0, 12.0)}            # near-miss Yes: its clicked roof point is scored
    return items, answers, clicked, tiles, obbs


def test_g1_audit_quality_design_weights_clicked_points_and_eligibility():
    from nakkhu import labelqa

    table, counts = labelqa.g1_audit_quality(*_g1_audit_case())
    by = table.set_index("tile_id")
    # w = N_h / n_h over answered units: S1 100 / 6, S2 150 / 3 (a5 Unsure excluded).
    w1, w2 = 100 / 6, 150 / 3
    assert by.loc["A", "answered"] == 5 and bool(by.loc["A", "eligible"])
    assert by.loc["A", "yes"] == 4 and by.loc["A", "yes_hit"] == 3      # a3 is 2 m away
    assert by.loc["A", "audit_quality"] == pytest.approx((2 * w1 + w2) / (3 * w1 + w2))
    assert by.loc["B", "answered"] == 3 and not bool(by.loc["B", "eligible"])
    assert by.loc["B", "audit_quality"] == pytest.approx(1.0)
    assert counts == {"units": 10, "answered_units": 9, "answered_units_in_tiles": 8,
                      "answered_units_outside_tiles": 1, "clicked_points_used": 1}
    # Without the clicked point a6 is scored at its frame point (no label within 1 m).
    items, answers, _, tiles, obbs = _g1_audit_case()
    plain, _ = labelqa.g1_audit_quality(items, answers, {}, tiles, obbs)
    assert plain.set_index("tile_id").loc["A", "audit_quality"] == pytest.approx(
        2 * w1 / (3 * w1 + w2))
    # A tile with no labels scores 0; the minimum answered count is a parameter.
    loose, _ = labelqa.g1_audit_quality(items, answers, {}, tiles, {"A": obbs["A"]},
                                        min_answered=3)
    assert loose.set_index("tile_id").loc["B", "audit_quality"] == 0.0
    assert bool(loose.set_index("tile_id").loc["B", "eligible"])


def test_g1_confirmatory_rho_uses_eligible_tiles_with_a_yes_unit():
    from nakkhu import labelqa

    quality = pd.DataFrame({"tile_id": ["t1", "t2", "t3", "t4", "t5"],
                            "audit_quality": [0.1, 0.5, 0.9, np.nan, 0.0],
                            "eligible": [True, True, True, True, False]})
    scores = pd.DataFrame({"tile_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
                           "gate_score": [0.2, 0.4, 0.6, 0.8, 1.0, 0.0]})
    stat = labelqa.g1_confirmatory(scores, quality)
    assert stat == {"spearman_rho": pytest.approx(1.0), "tiles": 3, "eligible_tiles": 4,
                    "eligible_tiles_without_yes": 1}
    flipped = scores.assign(gate_score=-scores["gate_score"])
    assert labelqa.g1_confirmatory(flipped, quality)["spearman_rho"] == pytest.approx(-1.0)
    with pytest.raises(labelqa.LabelQAError, match="without a gate score"):
        labelqa.g1_confirmatory(scores.loc[scores["tile_id"] != "t2"], quality)


def test_g1_confirmatory_journal_note_is_a_pending_analysis():
    from nakkhu.evaluate.guard import pending_analyses
    from nakkhu.labels_cli import G1_CONF_NOTE

    assert pending_analyses([{"run": "labels/x", "note": G1_CONF_NOTE}]) == ["labels/x"]
