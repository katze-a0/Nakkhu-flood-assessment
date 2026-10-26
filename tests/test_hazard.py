from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from nakkhu import hazard
from nakkhu.hydro.dem import GridSpec, write_raster


def test_tier_labels_and_boundaries():
    assert hazard.tier_labels() == ["on_drainage", "0-3", "3-6", "6-9", "9-12", ">12"]
    values = np.array([0.0, 1e-6, 3.0, 3.01, 6.0, 9.0, 12.0, 12.5, np.nan])
    np.testing.assert_array_equal(hazard.tier_index(values), [0, 1, 1, 2, 2, 3, 4, 5, -1])


def test_tier_counts_and_transitions():
    a = np.array([0, 1, 1, 2, -1], dtype=np.int8)
    b = np.array([0, 1, 2, 2, 3], dtype=np.int8)
    np.testing.assert_array_equal(hazard.tier_counts(a, 3), [1, 2, 1])
    m = hazard.transition_matrix(a, b, 3)
    assert m.sum() == 4 and m[1, 2] == 1 and m[2, 2] == 1


def test_weighted_kappa_known_values():
    a = np.array([0, 1, 2, 3, 4, 5] * 5, dtype=np.int8)
    assert hazard.weighted_kappa(a, a, 6) == pytest.approx(1.0)
    # 2x2 example: observed [[20, 5], [10, 15]] -> unweighted kappa 0.4 (equals linear for k=2)
    x = np.array([0] * 25 + [1] * 25, dtype=np.int8)
    y = np.array([0] * 20 + [1] * 5 + [0] * 10 + [1] * 15, dtype=np.int8)
    assert hazard.weighted_kappa(x, y, 2) == pytest.approx(0.4)


def test_spearman_monotone():
    a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    assert hazard.spearman(a, a ** 3) == pytest.approx(1.0)
    assert hazard.spearman(a, -a) == pytest.approx(-1.0)


def test_compare_surfaces():
    a = np.array([0.0, 1.0, 5.0, 20.0])
    b = np.array([0.0, 1.2, 7.0, 20.0])
    m = hazard.compare_surfaces(a, b, 0.5)
    assert m["n_cells"] == 4 and m["max_abs_diff_m"] == pytest.approx(2.0)
    assert m["share_abs_diff_gt_tol"] == pytest.approx(0.25)
    assert m["share_same_tier"] == pytest.approx(0.75)


def test_cell_groups_minimum_with_nan_and_empty():
    groups = hazard.CellGroups.from_lists([[0, 1], [], [2], [3]])
    values = np.array([5.0, 2.0, np.nan, 7.0])
    np.testing.assert_array_equal(groups.minimum(values), [2.0, np.nan, np.nan, 7.0])


def test_cell_groups_from_exactextract(tmp_path: Path):
    grid = GridSpec("EPSG:32645", Affine(30, 0, 0, 0, -30, 300), 10, 10)
    index = np.arange(100, dtype=np.int32).reshape(10, 10)
    path = write_raster(tmp_path / "idx.tif", index, grid)
    geoms = gpd.GeoSeries([box(10, 250, 50, 290), box(31, 31, 59, 59)], crs="EPSG:32645")
    groups = hazard.cell_groups(geoms, path)
    assert sorted(groups.cells[groups.offsets[0]:groups.offsets[1]].tolist()) == [0, 1, 10, 11]
    assert groups.cells[groups.offsets[1]:groups.offsets[2]].tolist() == [81]
    np.testing.assert_array_equal(hazard.centroid_cells(geoms, grid), [11, 81])  # corner -> floor


def test_logistic_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 20, 20000)
    p = 1 / (1 + np.exp(-(2.0 - 0.5 * x)))
    y = rng.uniform(size=x.size) < p
    a, b = hazard.fit_logistic(x, y)
    assert a == pytest.approx(2.0, abs=0.15) and b == pytest.approx(-0.5, abs=0.03)
    assert hazard.predict_logistic((a, b), np.array([0.0]))[0] > 0.8


def test_isotonic_is_non_increasing_pava():
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 0, 1, 0, 0, 1], dtype=float)
    knots = hazard.fit_isotonic(x, y)
    p = knots[1]
    assert np.all(np.diff(p) <= 1e-12)
    # PAVA solution: [1, .5, .5, 1/3, 1/3, 1/3]
    np.testing.assert_allclose(p, [1, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    assert hazard.predict_isotonic(knots, np.array([10.0]))[0] == pytest.approx(1 / 3)


def test_youden_brier_classes_and_cv():
    x = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])
    y = np.array([1, 1, 1, 0, 0, 0])
    t, j = hazard.youden_threshold(x, y)
    assert t == 2.0 and j == pytest.approx(1.0)
    assert hazard.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    np.testing.assert_array_equal(hazard.likelihood_class(np.array([0.05, 0.1, 0.3, 0.5])),
                                  [0, 1, 2, 3])
    blocks = np.array([0, 1, 0, 1, 0, 1])
    oof = hazard.calibrate_out_of_fold(x, y, blocks, "isotonic")
    assert oof.shape == x.shape and np.isfinite(oof).all()
    with pytest.raises(ValueError):
        hazard.youden_threshold(x, np.ones(6))


def test_confidence_cut_uses_cell_threshold(tmp_path: Path):
    csv = tmp_path / "thr.csv"
    csv.write_text(
        "s2_token,geometry,confidence_threshold_90%_precision\n"
        '39f,"POLYGON ((85 27, 86 27, 86 28, 85 28, 85 27))",0.789\n', encoding="utf-8")
    frame = gpd.GeoDataFrame({"longitude": [85.3, 85.3, 85.3], "latitude": [27.6, 27.6, 27.6],
                              "confidence": [0.80, 0.789, 0.70]})
    keep, thr = hazard.confidence_cut(frame, csv)
    assert keep.tolist() == [True, True, False] and set(thr) == {0.789}
    frame.loc[0, "longitude"] = 90.0
    with pytest.raises(ValueError):
        hazard.confidence_cut(frame, csv)


def test_range_summary_excludes_gated_and_diagnostic():
    import pandas as pd

    from nakkhu.hydro.sweep import range_summary

    labels = hazard.tier_labels()
    base = {f"n_{lab}": 0 for lab in labels}
    rows = []
    for cfg, n0, member, primary in (("p", 10, True, True), ("a", 20, True, False),
                                     ("gated", 99, False, False)):
        rows.append({**base, "inventory": "x", "cfg_id": cfg, "dem": "d", "conditioning": "c",
                     "threshold": "drop", "statistic": "centroid", "is_primary": primary,
                     "in_sensitivity_range": member, "kappa_linear_vs_primary": 1.0,
                     "n_on_drainage": n0})
    out = range_summary(pd.DataFrame(rows), labels)
    overall = out[out["group"] == "all surfaces x statistics"].iloc[0]
    assert overall["n_members"] == 2
    assert (overall["n_le_3_min"], overall["n_le_3_max"], overall["n_le_3_primary"]) == (10, 20, 10)
    assert "gated" in out[out["group"].str.startswith("excluded")].iloc[0]["excluded_cfg_ids"]
