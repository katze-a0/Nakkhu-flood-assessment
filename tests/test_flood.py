"""Plan 009: change-map building blocks (Steps 1-5) and validation statistics (V1-V4, H6)."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
import yaml
from scipy import ndimage as ndi

from nakkhu.flood import change as C
from nakkhu.flood import cloudmask as M
from nakkhu.flood import coregister as R
from nakkhu.flood import features as F
from nakkhu.flood import normalise as N
from nakkhu.flood import validate as V

ROOT = Path(__file__).resolve().parents[1]
PURE = ("coregister", "cloudmask", "normalise", "features", "change")


def texture(seed: int, shape: tuple[int, int] = (400, 400)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (ndi.gaussian_filter(rng.normal(size=shape), 2.0) * 60 + 120).astype(np.float32)


# --- Step 5 verify: the change map cannot import HAND, DEM or river layers --------------------


def test_change_map_modules_import_no_hand_dem_or_river_code():
    code = ("import sys\n"
            + "".join(f"import nakkhu.flood.{m}\n" for m in PURE)
            + "bad = [m for m in sys.modules if m.startswith(('nakkhu.hydro', 'nakkhu.hazard',"
              " 'nakkhu.splits', 'nakkhu.aoi', 'nakkhu.audit', 'nakkhu.evaluate', 'pysheds',"
              " 'pyflwdir', 'whitebox', 'exactextract', 'rasterio'))]\n"
              "print(bad)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=ROOT)
    assert out.stdout.strip() == "[]"


def test_change_map_modules_only_import_array_libraries():
    allowed = {"__future__", "collections.abc", "dataclasses", "typing", "warnings", "numpy", "pandas",
               "scipy", "shapely"}
    for name in PURE:
        tree = ast.parse((ROOT / "src" / "nakkhu" / "flood" / f"{name}.py").read_text(
            encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                assert mod.split(".")[0] in {m.split(".")[0] for m in allowed}, (name, mod)


def test_change_config_reads_no_hand_dem_or_channel_file():
    cfg = yaml.safe_load((ROOT / "configs" / "flood" / "change.yaml").read_text(
        encoding="utf-8"))
    cfg["buildings"].pop("hand_sweep_config")   # inventory definitions only (documented)
    text = json.dumps(cfg).lower()
    for word in ("hand", "dem_", "glo30", "fabdem", "gedtm", "river", "channel", "waterway",
                 "dist"):
        assert not re.search(word, text), word
    assert set(cfg["score"]["flood"]) == {
        "ring_d_brightness", "ring_d_saturation", "ring_d_sediment", "ring_d_texture",
        "s2_d_mndwi", "s2_d_bsi", "s2_d_ndvi"}


# --- Step 1: co-registration ------------------------------------------------------------------


def test_phase_correlation_recovers_shift_and_convention():
    base = texture(0, (600, 600))
    pre = base[50:550, 50:550]
    post = base[57:557, 47:547]          # post[r, c] = pre[r + 7, c - 3]  ->  (dy, dx) = (-7, 3)
    mask = np.ones(pre.shape, dtype=bool)
    mask[100:220, 100:220] = False
    s = R.phase_correlation(pre, post, mask, np.ones(pre.shape, dtype=bool))
    assert s.dy == pytest.approx(-7, abs=0.2) and s.dx == pytest.approx(3, abs=0.2)
    moved = R.shift_image(post, s.dy, s.dx)
    assert np.abs(moved[20:-20, 20:-20] - pre[20:-20, 20:-20]).mean() < 1.0
    ok, why = R.accept(s, 40, 1.2, 0.05)
    assert ok and why == "ok"


def test_shared_mask_does_not_force_zero_shift():
    base = texture(1, (600, 600))
    pre, post = base[50:550, 50:550], base[54:554, 44:544]
    holes = np.ones(pre.shape, dtype=bool)
    for r, c in product(range(20, 480, 60), range(20, 480, 60)):
        holes[r:r + 25, c:c + 25] = False     # footprints: masked in pre only
    s = R.phase_correlation(pre, post, holes, np.ones(pre.shape, dtype=bool))
    assert s.dy == pytest.approx(-4, abs=0.3) and s.dx == pytest.approx(6, abs=0.3)


def test_fill_rejected_uses_neighbours_then_all():
    keys = [(0, 0), (0, 1), (5, 5)]
    shifts = {(0, 0): (1.0, 2.0), (0, 1): (9.0, 9.0), (5, 5): (0.0, 0.0)}
    final = R.fill_rejected(keys, shifts, {(0, 0): True, (0, 1): False, (5, 5): False})
    assert final[(0, 0)] == (1.0, 2.0, "own")
    assert final[(0, 1)] == (1.0, 2.0, "neighbours")
    assert final[(5, 5)] == (1.0, 2.0, "all blocks")
    with pytest.raises(ValueError):
        R.fill_rejected(keys, shifts, {})


def test_piecewise_shift_applies_each_blocks_shift():
    img = texture(2, (300, 300))
    full = np.concatenate([img[None], np.ones((1, 300, 300), dtype=np.float32)])

    def read(r0: int, c0: int, h: int, w: int) -> np.ndarray:
        out = np.zeros((2, h, w), dtype=np.float32)
        rs, cs = max(r0, 0), max(c0, 0)
        re_, ce = min(r0 + h, 300), min(c0 + w, 300)
        if re_ > rs and ce > cs:
            out[:, rs - r0:re_ - r0, cs - c0:ce - c0] = full[:, rs:re_, cs:ce]
        return out

    field = {(0, 0): (2.0, 0.0), (0, 1): (0.0, -3.0)}
    out = R.piecewise_shift(read, 50, 50, 100, 200, 150, lambda br, bc: field[(br, bc)], 6)
    np.testing.assert_allclose(out[0, :, :100], img[52:152, 50:150], atol=1e-4)
    np.testing.assert_allclose(out[0, :, 100:], img[50:150, 147:247], atol=1e-4)


# --- Steps 2-3: mask and normalisation --------------------------------------------------------


def test_histogram_match_recovers_monotone_transform():
    rng = np.random.default_rng(3)
    ref = rng.integers(20, 230, size=(3, 200, 200)).astype(np.float32)
    src = np.rint(0.5 * ref + 30)
    out = N.match(src, ref, np.ones((200, 200), dtype=bool))
    assert np.abs(out - ref).mean() < 1.5


def test_cloud_and_shadow_masks():
    h = w = 100
    rng = np.random.default_rng(11)
    grey = rng.uniform(40, 200, size=(h, w)).astype(np.float32)
    grey[78:95, 78:95] = 120                        # ground under the later cloud shadow
    pre = np.repeat(grey[None], 3, axis=0)
    pre[:, 70:76, 5:11] = 230                       # bright roof in both dates
    post = pre * 0.6
    post[:, 5:20, 5:20] = 235                       # cloud (15 x 15 cells = 3600 m2)
    post[:, 40:43, 60:63] = 235                     # small bright patch: not a cloud
    post[:, 70:76, 5:11] = 230
    post[:, 80:93, 80:93] = 10                      # 13 x 13 dark cells (2704 m2): shadow
    valid = np.ones((h, w), dtype=bool)
    opts = yaml.safe_load((ROOT / "configs" / "flood" / "change.yaml").read_text(
        encoding="utf-8"))["mask"]
    out = M.observability(post, pre, valid, 4.0, opts, N.match)
    assert out["cloud"][10, 10] and not out["cloud"][41, 61] and not out["cloud"][72, 7]
    assert out["shadow"][86, 86] and not out["shadow"][50, 50]
    assert out["unobservable"][10, 22]              # 12 m dilation reaches 3 cells
    assert not out["unobservable"][50, 50]


# --- Step 4: features -------------------------------------------------------------------------


def test_pixel_features_values():
    rgb = np.zeros((3, 5, 5), dtype=np.float32)
    rgb[0], rgb[1], rgb[2] = 120, 90, 60
    f = F.pixel_features(rgb)
    assert f["brightness"][2, 2] == pytest.approx(90)
    assert f["saturation"][2, 2] == pytest.approx(0.5)
    assert f["sediment"][2, 2] == pytest.approx(1 / 3)
    assert f["texture"][2, 2] == pytest.approx(0)


def test_building_masks_ring_and_observability():
    win = F.Window(x0=0.0, y0=100.0, height=200, width=200)
    poly = shapely.box(40, 40, 50, 50)
    built = np.zeros((200, 200), dtype=bool)
    built[100:120, 120:140] = True                  # neighbour roof at x 60-70, y 40-50
    m = F.building_masks(poly, win, built, (1.0, 6.0), 6.0)
    assert m is not None and not m.clipped
    r0, r1, c0, c1 = m.box
    assert m.roof.sum() == 400 and (m.ring & m.roof).sum() == 0
    xx, yy = win.centres(r0, r1, c0, c1)
    ring_pts = shapely.points(xx[m.ring], yy[m.ring])
    d = shapely.distance(ring_pts, poly)
    assert d.min() >= 1.0 and d.max() <= 6.0 + 1e-9
    diffs = {k: np.ones((200, 200), dtype=np.float32) for k in F.PIXEL_FEATURES}
    usable = np.ones((200, 200), dtype=bool)
    valid = np.ones((200, 200), dtype=bool)
    clear = np.zeros((200, 200), dtype=bool)
    out = F.strip_features(m, diffs, usable, valid, clear, 8)
    assert out["observable"] and out["ring_d_brightness"] == 1.0
    cloudy = clear.copy()
    i, j = np.argwhere(m.roof)[0]
    cloudy[r0 + i, c0 + j] = True
    assert F.strip_features(m, diffs, usable, valid, cloudy, 8)["reason"] == "cloud or shadow"
    edge = F.building_masks(shapely.box(0.5, 60, 8, 70), win, built, (1.0, 6.0), 6.0)
    assert edge is not None and edge.clipped


def test_s2_indices_and_window_mean():
    b = {k: np.full((5, 5), v) for k, v in
         {"B02": 0.1, "B03": 0.2, "B04": 0.3, "B08": 0.4, "B11": 0.2}.items()}
    idx = F.s2_indices(b)
    assert idx["mndwi"][0, 0] == pytest.approx(0.0)
    assert idx["ndvi"][0, 0] == pytest.approx(0.1 / 0.7)
    assert idx["bsi"][0, 0] == pytest.approx(0.0)
    vals = np.arange(25, dtype=float).reshape(5, 5)
    clear = np.ones((5, 5), dtype=bool)
    clear[1, 1] = False
    out = F.s2_window_mean(vals, clear, np.array([2, 0]), np.array([2, 2]), 3, 5)
    assert out[0] == pytest.approx((vals[1:4, 1:4].sum() - vals[1, 1]) / 8)
    assert np.isnan(out[1])


# --- Step 5: score and classes ----------------------------------------------------------------


def test_robust_z_and_signs():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    pop = np.ones(5, dtype=bool)
    z = C.robust_z(x, pop)
    assert z[2] == 0 and z[4] > 50
    np.testing.assert_allclose(C.signed_z(x, pop, -1), -z)
    assert C.signed_z(-x, pop, "abs")[4] == pytest.approx(z[4])
    assert np.isnan(C.robust_z(np.ones(5), pop)).all()
    with pytest.raises(ValueError):
        C.signed_z(x, pop, 2)


def test_mixture_boundary_separates_modes_and_is_deterministic():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(0, 1, 2000), rng.normal(6, 1, 200)]
    a, b = C.fit_mixture(x), C.fit_mixture(x)
    assert a == b and a.converged
    cut = C.boundary(a)
    assert 2.0 < cut < 5.0
    assert a.posterior_background(np.array([cut]))[0] == pytest.approx(0.5)
    # heavy-tailed scores: a narrow core and a wide component with a lower mean
    tail = C.fit_mixture(np.r_[rng.normal(0, 0.5, 5000), rng.normal(-1, 10, 400)])
    t_cut = C.boundary(tail)
    assert 1.0 < t_cut < 3.0 and tail.posterior_background(np.array([t_cut]))[0] ==         pytest.approx(0.5)
    assert C.classify(np.array([0.0, 7.0, np.nan]), np.array([True, True, True]),
                      cut).tolist() == ["negative", "positive", "not_observable"]


def test_score_buildings_strip_preference_and_unobservable():
    rng = np.random.default_rng(5)
    n = 200
    cols = ["ring_d_brightness", "ring_d_saturation", "ring_d_sediment", "ring_d_texture",
            "s2_d_mndwi", "s2_d_bsi", "s2_d_ndvi", "roof_d_brightness", "roof_d_texture"]

    def table(obs: np.ndarray) -> pd.DataFrame:
        t = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
        t["observable"] = obs
        return t

    obs6 = np.ones(n, dtype=bool)
    obs6[:20] = False
    obs5 = np.ones(n, dtype=bool)
    obs5[:10] = False
    per = {"0600": table(obs6), "0500": table(obs5)}
    cfg = yaml.safe_load((ROOT / "configs" / "flood" / "change.yaml").read_text(
        encoding="utf-8"))["score"]
    out, meta = C.score_buildings(per, ["0600", "0500"], cfg["flood"], cfg["damage"],
                                  cfg["required"], cfg["gmm"])
    assert (out["strip"][:10] == "").all() and (out["strip"][10:20] == "0500").all()
    assert (out["strip"][20:] == "0600").all()
    assert (out["class"][:10] == "not_observable").all()
    assert set(out["class"][10:]) <= {"positive", "negative"}
    again, _ = C.score_buildings(per, ["0600", "0500"], cfg["flood"], cfg["damage"],
                                 cfg["required"], cfg["gmm"])
    pd.testing.assert_frame_equal(out, again)
    assert np.isfinite(meta["flood"]["boundary"]) or meta["flood"]["boundary"] in (
        float("inf"), float("-inf"))


# --- V3 / V4 statistics -----------------------------------------------------------------------


def brute_auc(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(s)) if w is None else w
    pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
    num = sum(w[i] * w[j] * ((s[i] > s[j]) + 0.5 * (s[i] == s[j])) for i in pos for j in neg)
    return float(num / (w[pos].sum() * w[neg].sum()))


def test_auc_matches_brute_force_with_ties_and_weights():
    rng = np.random.default_rng(6)
    s = rng.integers(0, 5, 60).astype(float)
    y = rng.random(60) < 0.4
    w = rng.uniform(0.5, 3, 60)
    assert V.roc_auc(s, y) == pytest.approx(brute_auc(s, y))
    assert V.roc_auc(s, y, w) == pytest.approx(brute_auc(s, y, w))
    assert V.roc_auc(s, y, np.ones(60)) == pytest.approx(V.roc_auc(s, y))


def test_average_precision_simple_case():
    s = np.array([3.0, 2.0, 1.0])
    y = np.array([True, False, True])
    assert V.average_precision(s, y) == pytest.approx(0.5 * 1 + 0.5 * (2 / 3))


def test_delong_and_block_bootstrap():
    rng = np.random.default_rng(7)
    n = 600
    y = rng.random(n) < 0.3
    s1 = y * 1.0 + rng.normal(0, 1, n)
    s2 = y * 0.2 + rng.normal(0, 1, n)
    d = V.delong(y, s1, s2)
    assert d["auc1"] == pytest.approx(V.roc_auc(s1, y))
    assert d["delta"] > 0 and d["p"] < 0.01 and d["ci_lo"] > 0
    same = V.delong(y, s1, s1)
    assert same["delta"] == 0
    blocks = np.arange(n) // 20
    bb = V.block_bootstrap_delta_auc(y, s1, s2, blocks, 200, 0)
    assert bb["boot_lo"] < bb["delta"] < bb["boot_hi"] and bb["boot_lo"] > 0


def test_tier_rates_skill_and_odds_ratios():
    rng = np.random.default_rng(8)
    labels = ["on_drainage", "0-3", "3-6", "6-9", "9-12", ">12"]
    tiers = rng.integers(0, 6, 6000)
    p = np.array([0.5, 0.35, 0.2, 0.12, 0.08, 0.05])[tiers]
    y = rng.random(6000) < p
    rates = V.tier_hit_rates(tiers, y, labels)
    assert rates["n"].sum() == 6000 and rates["capture"].sum() == pytest.approx(1)
    assert (rates["hit_lo"] <= rates["hit_rate"]).all()
    hand = tiers * 3.0 - 1.0
    skill = V.cumulative_skill(hand, y)
    assert (np.diff(skill["H"]) >= 0).all() and (skill["tp"] + skill["fn"] == y.sum()).all()
    clusters = rng.integers(0, 40, 6000)
    ors = V.tier_odds_ratios(tiers, y.astype(float), clusters, labels)
    true_or = (0.5 / 0.5) / (0.05 / 0.95)
    first = ors.loc[ors["tier"] == "on_drainage"].iloc[0]
    assert first["or_lo"] < true_or < first["or_hi"]
    assert V.monotone(ors)
    trend = V.trend_test(tiers, y.astype(float), clusters)
    assert trend["slope_log_or_per_tier"] < 0 and trend["p"] < 0.001


def test_morans_i_detects_clustering():
    rng = np.random.default_rng(9)
    x, y = rng.uniform(0, 1000, 800), rng.uniform(0, 1000, 800)
    random = V.morans_i(rng.normal(size=800), x, y, n_perm=199)
    clustered = V.morans_i(np.sin(x / 100) + np.cos(y / 100), x, y, n_perm=199)
    assert abs(random["I"]) < 0.1 and clustered["I"] > 0.5 and clustered["p_perm"] < 0.01


def test_h6_calibration_and_damage_model():
    rng = np.random.default_rng(10)
    hand = rng.uniform(0, 20, 3000)
    y = rng.random(3000) < 1 / (1 + np.exp(-(1.0 - 0.4 * hand)))
    blocks = rng.integers(0, 10, 3000)
    from nakkhu import hazard

    tiers = hazard.tier_index(hand)
    out = V.h6_calibration(hand, y, blocks, tiers, hazard.tier_labels())
    for method in ("logistic", "isotonic"):
        assert 0 < out["methods"][method]["brier"] < 0.25
        assert sum(out["methods"][method]["class_counts"].values()) == 3000
    assert 0 < out["youden"]["hand_cut_m"] < 10
    few = V.damage_model(hand[:12], y[:12], blocks[:12])
    assert few["model"] == "descriptive only"
    many = V.damage_model(hand, y, blocks)
    assert many["model"] == "binary logistic on HAND" and many["or_per_m"] < 1


def test_r1_drop_rule_and_c3():
    change = pd.DataFrame({"building_id": ["a", "b", "c"], "class":
                           ["positive", "negative", "not_observable"],
                           "observable": [True, True, False]})
    est = {"estimates": {"ppv": {"wilson_lo": 0.42, "estimate": 0.6}, "npv": {},
                         "drop_change_map": True}}
    r1 = V.r1_reference(change, est)
    assert not r1["usable"] and list(r1["building_id"]) == ["a", "b"]
    assert list(r1["y"]) == [True, False]
    ors = pd.DataFrame({"odds_ratio": [4.0, 2.0, 1.0]})
    good = V.c3_outcome({"delta": 0.05, "boot_lo": 0.01, "boot_hi": 0.09}, {"p": 0.02}, ors)
    assert good["pass"]
    bad = V.c3_outcome({"delta": 0.05, "boot_lo": -0.01, "boot_hi": 0.09}, {"p": 0.2}, ors)
    assert not bad["pass"]


def test_validate_refuses_without_prerequisites(project):
    problems = V.prerequisites(project, project.paths.root / "missing.parquet", ["google_v3"])
    text = " ".join(problems)
    assert "prereg" in text and "flood_trace_estimates" in text
    assert "change map for google_v3 missing" in text and "plan 007" in text
    with pytest.raises((RuntimeError, FileNotFoundError, KeyError, ValueError)):
        V.run_validate(project, ["google_v3"])
    assert not (project.paths.runs / "eval" / "test_journal.jsonl").exists()


def test_sample_hash_drift_detects_changed_change_map(project):
    task = project.paths.root / "audit" / "flood_trace"
    task.mkdir(parents=True)
    side = project.paths.artifacts / "flood" / "change_map_buildings_detector_final.json"
    side.parent.mkdir(parents=True)
    hashes = {"change_map_sha256": "a", "inventory_sha256": "b", "buildings_sha256": "c"}
    side.write_text(json.dumps({"inventory": "detector_final", **hashes}), encoding="utf-8")
    (task / "sample.json").write_text(json.dumps(
        {"inputs": {"inventory": "detector_final", **hashes}}), encoding="utf-8")
    assert V.sample_hash_drift(project) == []
    side.write_text(json.dumps({"inventory": "detector_final", **hashes,
                                "change_map_sha256": "z"}), encoding="utf-8")
    assert V.sample_hash_drift(project) == ["change_map_sha256"]
    assert any("changed since the sitting-3 sample" in p
               for p in V.prerequisites(project, None, []))


def test_gate_scores_primary_on_passing_strips_only():
    rng = np.random.default_rng(12)
    n = 300
    cols = ["ring_d_brightness", "ring_d_saturation", "ring_d_sediment", "ring_d_texture",
            "s2_d_mndwi", "s2_d_bsi", "s2_d_ndvi", "roof_d_brightness", "roof_d_texture"]
    obs6 = np.ones(n, dtype=bool)
    obs6[:40] = False                      # 40 gap-fill buildings seen only by 0500
    per = {s: pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols).assign(observable=o)
           for s, o in (("0600", obs6), ("0500", np.ones(n, dtype=bool)))}
    cfg = yaml.safe_load((ROOT / "configs" / "flood" / "change.yaml").read_text(
        encoding="utf-8"))
    sc = cfg["score"]
    assert cfg["coregistration"]["gate"] is True
    gated, _ = C.score_buildings(per, ["0600"], sc["flood"], sc["damage"], sc["required"],
                                 sc["gmm"])
    every, _ = C.score_buildings(per, ["0600", "0500"], sc["flood"], sc["damage"],
                                 sc["required"], sc["gmm"])
    assert (gated["class"][:40] == "not_observable").all()
    assert (every["strip"][:40] == "0500").all() and every["observable"][:40].all()
    np.testing.assert_allclose(gated["score"][40:], every["score"][40:])
