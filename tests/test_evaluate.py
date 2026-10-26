"""Plan 005 Steps 2-5: matching, metrics, bootstrap, intervals, guard and test journal."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
import shapely.affinity

from nakkhu.cli import main
from nakkhu.config import load_project
from nakkhu.evaluate import bootstrap, ci, guard, metrics
from nakkhu.evaluate.match import iou_pairs, match_by_confidence
from nakkhu.evaluate.score import compare_blocks, score

# --- Step 3: matching on hand-constructed polygons ------------------------------------------


def _rot(box: shapely.Polygon, deg: float) -> shapely.Polygon:
    return shapely.affinity.rotate(box, deg, origin="centroid")


def test_rotated_matching_known_tp_fp_fn():
    refs = np.array([
        _rot(shapely.box(0, 0, 10, 6), 30),        # r0: matched exactly by p0
        shapely.box(20, 0, 30, 10),                # r1: p1 overlaps it with IoU 1/3
        shapely.box(50, 0, 60, 10),                # r2: nothing near -> FN
    ], dtype=object)
    preds = np.array([
        _rot(shapely.box(0, 0, 10, 6), 30),        # p0: IoU 1 with r0
        shapely.box(25, 0, 35, 10),                # p1: IoU 50/150 with r1
        shapely.box(100, 100, 110, 110),           # p2: far -> FP
    ], dtype=object)
    conf = np.array([0.9, 0.8, 0.7])
    pairs = iou_pairs(preds, refs)
    assert np.isclose(pairs.iou[(pairs.pred == 1) & (pairs.ref == 1)][0], 1 / 3)
    p2r, r2p = match_by_confidence(conf, len(refs), pairs, 0.5)
    assert p2r.tolist() == [0, -1, -1] and r2p.tolist() == [0, -1, -1]   # TP 1, FP 2, FN 2
    p2r, r2p = match_by_confidence(conf, len(refs), pairs, 0.3)
    assert p2r.tolist() == [0, 1, -1] and r2p.tolist() == [0, 1, -1]     # TP 2, FP 1, FN 1


def test_higher_confidence_wins_a_contested_reference():
    ref = np.array([shapely.box(0, 0, 10, 10)], dtype=object)
    preds = np.array([shapely.box(0, 0, 10, 10), shapely.box(1, 0, 11, 10)], dtype=object)
    pairs = iou_pairs(preds, ref)
    p2r, _ = match_by_confidence(np.array([0.2, 0.9]), 1, pairs, 0.5)
    assert p2r.tolist() == [-1, 0]       # the 0.9 box takes it although its IoU is lower


# --- Step 3: AP and threshold on a toy PR set -----------------------------------------------


def _reference_coco_ap(conf, tp, n_ref):
    """Literal COCOeval accumulate() logic for one category."""
    order = np.argsort(-np.asarray(conf), kind="mergesort")
    tps = np.cumsum(np.asarray(tp)[order]).astype(float)
    fps = np.cumsum(1 - np.asarray(tp)[order]).astype(float)
    rc = tps / n_ref
    pr = list(tps / (fps + tps + np.spacing(1)))
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    q = np.zeros(101)
    inds = np.searchsorted(rc, np.linspace(0, 1, 101), side="left")
    for ri, pi in enumerate(inds):
        if pi < len(pr):
            q[ri] = pr[pi]
    return q.mean()


def test_ap_matches_hand_value_and_reference():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    tp = np.array([1, 0, 1, 0])
    # recall 1/3 at precision 1, 2/3 at 2/3: 34 recall points at 1, 33 at 2/3, rest 0
    assert np.isclose(metrics.ap_coco101(conf, tp, 3), 56 / 101)
    assert np.isclose(metrics.ap_voc_all(conf, tp, 3), 1 / 3 + (1 / 3) * (2 / 3))
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.random(50)
        t = (rng.random(50) < 0.6).astype(int)
        n = int(t.sum()) + int(rng.integers(0, 10))
        assert np.isclose(metrics.ap_coco101(c, t, n), _reference_coco_ap(c, t, n), atol=1e-9)
    assert np.isclose(metrics.ap_coco101(conf, tp, 3, weights=np.ones(4)),
                      metrics.ap_coco101(conf, tp, 3))


def test_f1_max_threshold_and_ties():
    conf = np.array([0.9, 0.8, 0.8, 0.5, 0.4])
    tp = np.array([1, 1, 0, 1, 0])
    out = metrics.f1_max_threshold(conf, tp, n_ref=3)
    # cuts: 0.9 -> 2/4; 0.8 -> 4/6; 0.5 -> 6/7; 0.4 -> 6/8
    assert out == {"threshold": 0.5, "f1": pytest.approx(6 / 7), "n_pred": 4}
    tie = metrics.f1_max_threshold(np.array([0.9, 0.5]), np.array([1, 0]), n_ref=1)
    assert tie["threshold"] == 0.9         # F1 1.0 vs 2/3: higher cut


def test_count_errors_and_prf():
    assert metrics.count_errors([3, 0, 5], [2, 1, 5]) == {"mae": pytest.approx(2 / 3),
                                                          "bias": 0.0, "units": 3}
    assert metrics.prf(2, 3, 2, 4) == {"precision": pytest.approx(2 / 3), "recall": 0.5,
                                       "f1": pytest.approx(4 / 7)}


# --- Step 4: bootstrap and intervals --------------------------------------------------------


def test_bootstrap_reproducible_and_ratio_estimator():
    a = bootstrap.multiplicities(7, 2000, 0)
    b = bootstrap.multiplicities(7, 2000, 0)
    assert np.array_equal(a, b) and (a.sum(axis=1) == 7).all()
    assert not np.array_equal(a, bootstrap.multiplicities(7, 2000, 1))
    cols = {"tp": np.array([1.0, 3.0]), "n": np.array([2.0, 4.0])}
    point = bootstrap.additive(cols, np.ones((1, 2)), bootstrap.ratio_stat("tp", "n"))
    assert np.isclose(point[0], 4 / 6)   # pooled ratio, not the mean of 0.5 and 0.75


def test_wilson_textbook_values():
    # Newcombe (1998), Stat. Med. 17:857, Table I: 81/263 -> 0.2553-0.3662; 0/20 -> 0-0.1611
    lo, hi = ci.wilson(81, 263)
    assert (round(lo, 4), round(hi, 4)) == (0.2553, 0.3662)
    lo, hi = ci.wilson(0, 20)
    assert lo == 0.0 and round(hi, 4) == 0.1611
    assert ci.seed_summary([0.5, 0.7, 0.6]) == {"mean": pytest.approx(0.6),
                                                "sd": pytest.approx(0.1), "n": 3}


# --- Step 3/4: end-to-end scoring on a synthetic region -------------------------------------


def _tiles() -> pd.DataFrame:
    rows = []
    for i, (c, r) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        w, s = c * 320.0, r * 320.0
        rows.append({"tile_id": f"t{i}", "bounds": [w, s, w + 320.0, s + 320.0],
                     "block_id": "bA" if c == 0 else "bB"})
    return pd.DataFrame(rows)


def test_score_counts_and_blocks():
    tiles = _tiles()
    region = shapely.box(0, 0, 640, 640)
    refs = np.array([shapely.box(10, 10, 20, 20), shapely.box(400, 10, 410, 20),
                     shapely.box(400, 400, 410, 410), shapely.box(700, 700, 710, 710)],
                    dtype=object)
    preds = np.array([shapely.box(10, 10, 20, 20), shapely.box(402, 10, 412, 20),
                      shapely.box(100, 100, 110, 110), shapely.box(700, 700, 710, 710)],
                     dtype=object)
    conf = np.array([0.9, 0.8, 0.3, 0.9])
    out = score(preds, conf, refs, region, tiles, threshold=0.5,
                params={"n_boot": 50}, ap_ci=False)
    m = out.metrics
    # the object at (700, 700) is outside the region and ignored; the 0.3 box is cut
    assert m["n_ref"] == 3 and m["n_pred_in_region"] == 3 and m["n_pred_kept"] == 2
    assert m["iou_50"]["precision"]["value"] == pytest.approx(1.0)       # 2/2 (IoU 0.667)
    assert m["iou_50"]["recall"]["value"] == pytest.approx(2 / 3)
    pb = out.per_block.set_index("block_id")
    assert pb.loc["bA", "n_ref"] == 1 and pb.loc["bB", "n_ref"] == 2
    assert pb["tp_pred_50"].sum() == 2 and pb["tp_ref_50"].sum() == 2
    assert out.per_tile["n_ref"].sum() == 3
    assert m["tile_count"]["mae"]["value"] == pytest.approx(0.25)        # t3: 0 vs 1
    same = compare_blocks(out.per_block, out.per_block, {"n_boot": 50})
    assert same["delta_f1_50"]["value"] == 0.0 and same["delta_f1_50"]["ci_high"] == 0.0


# --- Steps 2 and 5: guard, freeze and test journal ------------------------------------------


def _frozen_project(root: Path):
    (root / "configs" / "splits").mkdir(parents=True)
    split = root / "configs" / "splits" / "split_v1.yaml"
    split.write_text("tiles: {}\n", encoding="utf-8")
    sha = hashlib.sha256(split.read_bytes()).hexdigest()
    (root / "configs" / "mirrored.yaml").write_text("a: {x: 1, y: 2}\n", encoding="utf-8")
    (root / "configs" / "prereg.yaml").write_text(
        f"version: prereg_v1\nsplit: {{file: configs/splits/split_v1.yaml, sha256: {sha}}}\n"
        "thing:\n  mirror_of: configs/mirrored.yaml#a\n  values: {x: 1}\n", encoding="utf-8")
    project = load_project(root)
    return project


def test_guard_refuses_test_until_frozen_and_after_edits(root):
    project = _frozen_project(root)
    with pytest.raises(guard.GuardError, match="not frozen"):
        guard.check_access(project, "test")
    assert guard.check_access(project, "val")["problems"]        # val allowed, state recorded
    result = guard.freeze(project)
    assert result["status"] == "frozen"
    assert (root / "artifacts" / guard.PREREG_SHA_ARTIFACT).is_file()
    assert guard.require_frozen(project)["problems"] == []
    # a mirrored threshold edited after the freeze
    (root / "configs" / "mirrored.yaml").write_text("a: {x: 5, y: 2}\n", encoding="utf-8")
    with pytest.raises(guard.GuardError, match="mirrored.yaml"):
        guard.require_frozen(project)
    (root / "configs" / "mirrored.yaml").write_text("a: {x: 1, y: 3}\n", encoding="utf-8")
    guard.require_frozen(project)                                 # unmirrored key: fine
    # the prereg itself edited after the freeze
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + "extra: 1\n", encoding="utf-8")
    with pytest.raises(guard.GuardError, match="changed after the freeze"):
        guard.require_frozen(project)
    # the split edited
    with pytest.raises(guard.GuardError):
        guard.freeze(project)                                     # needs --replace
    guard.freeze(project, replace=True)
    (root / "configs" / "splits" / "split_v1.yaml").write_text("tiles: {a: 1}\n")
    with pytest.raises(guard.GuardError, match="split file"):
        guard.require_frozen(project)


def test_cli_refuses_test_split_without_prereg(root, capsys):
    rc = main(["--root", str(root), "eval", "--run", "infer/x", "--split", "test"])
    assert rc == 1
    assert "test scoring refused" in capsys.readouterr().err
    assert not guard.journal_path(load_project(root)).exists()


def test_test_journal_flags_rescoring_with_other_config(root):
    project = _frozen_project(root)
    first = guard.record_test_access(project, "infer/r1", "cfgA", "p", "s")
    assert not first["flagged"] and first["prior_accesses"] == 0
    again = guard.record_test_access(project, "infer/r1", "cfgA", "p", "s")
    assert not again["flagged"] and again["prior_accesses"] == 1
    other = guard.record_test_access(project, "infer/r1", "cfgB", "p", "s")
    assert other["flagged"] and "different" in other["flag_reason"]
    assert len(guard.read_test_journal(project)) == 3
    guard.freeze(project)
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + "extra: 1\n", encoding="utf-8")
    with pytest.raises(guard.GuardError, match="test was accessed"):
        guard.freeze(project, replace=True)


def test_file_hash_constant_and_nested_mirrors(root):
    project = _frozen_project(root)
    data = root / "configs" / "lines.geojson"
    data.write_text("{}", encoding="utf-8")
    (root / "configs" / "nested.yaml").write_text("t: {a: {b: 1, c: 2}, d: 3}\n",
                                                  encoding="utf-8")
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + (
        f"lines:\n  sha256_of: configs/lines.geojson\n  sha256: "
        f"{hashlib.sha256(b'{}').hexdigest()}\n"
        "tags:\n  constant_of: nakkhu.evaluate.metrics:AP_IOUS\n"
        "  values: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]\n"
        "deep:\n  mirror_of: configs/nested.yaml#t\n  values: {a: {b: 1}}\n"), encoding="utf-8")
    assert guard.verify(project)["problems"] == [
        f"prereg not frozen (artifacts/{guard.PREREG_SHA_ARTIFACT} missing)"]
    data.write_text("{ }", encoding="utf-8")
    (root / "configs" / "nested.yaml").write_text("t: {a: {b: 9, c: 2}, d: 3}\n",
                                                  encoding="utf-8")
    problems = guard.mirror_problems(project, guard.load_prereg(project))
    assert any("lines.geojson sha256" in p for p in problems)
    assert any("nested.yaml#t.a.b is 9" in p for p in problems)
    assert not any("AP_IOUS" in p for p in problems)


def test_new_version_keeps_old_hash_and_history(root, tmp_path):
    project = _frozen_project(root)
    v1 = guard.freeze(project)["prereg_sha256"]
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8").replace("prereg_v1", "prereg_v2"),
                      encoding="utf-8")
    with pytest.raises(guard.GuardError, match="--replace"):
        guard.freeze(project)
    log = tmp_path / "changelog.md"
    log.write_text("# v2\n", encoding="utf-8")
    out = guard.freeze(project, replace=True, changelog=log)
    assert out["version"] == "prereg_v2" and out["kept"] == ["prereg_v1"]
    assert guard.frozen_versions(project) == {"prereg_v1": v1, "prereg_v2": out["prereg_sha256"]}
    history = guard._history(project)
    assert [h["version"] for h in history] == ["prereg_v1", "prereg_v2"]
    assert (root / "artifacts" / guard.CHANGELOG).read_text(encoding="utf-8") == "# v2\n"
    assert guard.require_frozen(project)["prereg_sha256"] == out["prereg_sha256"]


def _amend_setup(root, tmp_path):
    """v1 frozen, one test access journalled, a copy of v1 kept, live prereg edited to v2."""
    project = _frozen_project(root)
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + "main: {inventory: a}\n",
                      encoding="utf-8")
    v1 = guard.freeze(project)["prereg_sha256"]
    previous = tmp_path / "prereg_v1_frozen.yaml"
    previous.write_bytes(prereg.read_bytes())
    guard.record_test_access(project, "infer/r1", "cfgA", v1, "s")
    prereg.write_text(prereg.read_text(encoding="utf-8").replace("prereg_v1", "prereg_v2")
                      .replace("main: {inventory: a}", "main: {inventory: b, note: v2}"),
                      encoding="utf-8")
    log = tmp_path / "v2.md"
    log.write_text(f"## prereg_v1 -> prereg_v2\nprevious {v1}\n", encoding="utf-8")
    (root / "artifacts" / guard.CHANGELOG).parent.mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / guard.CHANGELOG).write_text("# old log\n", encoding="utf-8")
    return project, prereg, previous, log, v1


def test_amend_accepts_an_audited_amendment_after_test_access(root, tmp_path):
    project, prereg, previous, log, v1 = _amend_setup(root, tmp_path)
    with pytest.raises(guard.GuardError, match="no new freeze is allowed"):
        guard.freeze(project, replace=True, changelog=log)          # the old path still refuses
    out = guard.freeze(project, amend=True, previous_file=previous, allow=["version", "main"],
                       changelog=log)
    assert out["status"] == "frozen" and out["amends"] == "prereg_v1"
    assert out["changed_paths"] == ["main.inventory", "main.note", "version"]
    last = guard._history(project)[-1]
    assert last["amends"] == "prereg_v1" and last["previous_sha256"] == v1
    assert last["test_journal_entries"] == 1 and last["last_journal_utc"]
    assert last["allow"] == ["version", "main"]
    assert (root / "artifacts" / "prereg" / "prereg_v1.yaml").read_bytes() == \
        previous.read_bytes()
    text = (root / "artifacts" / guard.CHANGELOG).read_text(encoding="utf-8")
    assert text.startswith("# old log\n") and "prereg_v1 -> prereg_v2" in text   # appended
    assert guard.frozen_versions(project)["prereg_v1"] == v1
    assert guard.require_frozen(project)["prereg_sha256"] == out["prereg_sha256"]
    assert guard.freeze(project, amend=True, previous_file=previous, allow=["main"],
                        changelog=log)["status"] == "already frozen"


def test_amend_refuses_locked_section_and_paths_outside_allow(root, tmp_path):
    project, prereg, previous, log, _ = _amend_setup(root, tmp_path)
    with pytest.raises(guard.GuardError, match="outside --allow: main.inventory"):
        guard.freeze(project, amend=True, previous_file=previous, allow=["version",
                                                                         "main.note"],
                     changelog=log)
    prereg.write_text(prereg.read_text(encoding="utf-8").replace(
        "split: {", "split: {rule: other, "), encoding="utf-8")
    with pytest.raises(guard.GuardError, match="locked section.*split.rule"):
        guard.freeze(project, amend=True, previous_file=previous,
                     allow=["version", "main", "split"], changelog=log)
    assert guard.locked("falsifiers.C5.match") and not guard.locked("falsifiers.C5.outcome")
    assert guard.locked("detector_final.cut") and not guard.locked("detector_final.uses")
    assert not guard.locked("uncertainty_propagation.u3")       # not under `uncertainty`
    assert guard.locked("uncertainty.block_bootstrap.seed")


def test_amend_refuses_hash_mismatch_missing_changelog_and_pending_analysis(root, tmp_path):
    project, prereg, previous, log, _ = _amend_setup(root, tmp_path)
    args = {"amend": True, "allow": ["version", "main"]}
    with pytest.raises(guard.GuardError, match="--changelog"):
        guard.freeze(project, previous_file=previous, changelog=None, **args)
    with pytest.raises(guard.GuardError, match="--changelog"):
        guard.freeze(project, previous_file=previous, changelog=tmp_path / "none.md", **args)
    with pytest.raises(guard.GuardError, match="--previous-file"):
        guard.freeze(project, previous_file=None, changelog=log, **args)
    edited = tmp_path / "edited.yaml"
    edited.write_text(previous.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(guard.GuardError, match="sha256 .* != frozen prereg_v1"):
        guard.freeze(project, previous_file=edited, changelog=log, **args)
    vague = tmp_path / "vague.md"
    vague.write_text("changed things\n", encoding="utf-8")
    with pytest.raises(guard.GuardError, match="must name prereg_v2"):
        guard.freeze(project, previous_file=previous, changelog=vague, **args)
    for run, note in (("audit/flood_trace", ""), ("flood/validate:final", ""),
                      ("labels/x", "plan 004 G1"), ("uncertainty/run", "")):
        journal = guard.journal_path(project)
        keep = journal.read_bytes()
        guard.record_test_access(project, run, "c", "p", "s", note=note)
        with pytest.raises(guard.GuardError, match="pending analyses"):
            guard.freeze(project, previous_file=previous, changelog=log, **args)
        journal.write_bytes(keep)
    assert not guard.frozen_versions(project).get("prereg_v2")
    prereg.write_text(prereg.read_text(encoding="utf-8").replace("prereg_v2", "prereg_v1"),
                      encoding="utf-8")
    with pytest.raises(guard.GuardError, match="already frozen with another hash"):
        guard.freeze(project, previous_file=previous, changelog=log, **args)
    prereg.write_text(prereg.read_text(encoding="utf-8").replace("prereg_v1", "prereg_v0"),
                      encoding="utf-8")
    with pytest.raises(guard.GuardError, match="must be above"):
        guard.freeze(project, previous_file=previous, changelog=log, **args)


def test_dotted_constant_pins_function_defaults(root):
    project = _frozen_project(root)
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + (
        "moran:\n  constant_of: nakkhu.flood.validate:morans_i.__defaults__\n"
        "  values: [8, 999, 0]\n"
        "kw:\n  constant_of: nakkhu.hazard:fit_logistic.__kwdefaults__\n"
        "  values: {l2: 1.0e-6, max_iter: 100}\n"), encoding="utf-8")
    assert guard.mirror_problems(project, guard.load_prereg(project)) == []
    prereg.write_text(prereg.read_text(encoding="utf-8").replace("[8, 999, 0]", "[8, 99, 0]"),
                      encoding="utf-8")
    problems = guard.mirror_problems(project, guard.load_prereg(project))
    assert any("morans_i.__defaults__" in p for p in problems)


def test_duplicate_keys_refused_at_any_level(root):
    assert guard.duplicate_keys("a: 1\nb: {c: 1, d: 2}\n") == []
    assert guard.duplicate_keys("a: 1\nb: 2\na: 3\n") == ["a (lines 1, 3)"]
    assert guard.duplicate_keys("x:\n  y: 1\n  y: 2\nz: [{k: 1, k: 2}]\n") == [
        "x.y (lines 2, 3)", "z[0].k (lines 4, 4)"]
    project = _frozen_project(root)
    prereg = root / "configs" / "prereg.yaml"
    prereg.write_text(prereg.read_text(encoding="utf-8") + "thing:\n  values: {x: 2}\n",
                      encoding="utf-8")
    problems = guard.verify(project)["problems"]
    assert any("duplicate keys" in p and "thing (lines" in p for p in problems)
    with pytest.raises(guard.GuardError, match="duplicate keys"):
        guard.freeze(project)
