from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from affine import Affine
from shapely.geometry import LineString, box

from nakkhu.hydro import condition as cond
from nakkhu.hydro import dem as demmod
from nakkhu.hydro import drop_analysis, route
from nakkhu.hydro import hand as handmod
from nakkhu.hydro.dem import GridSpec

REPO_SWEEP = Path(__file__).resolve().parents[1] / "configs" / "hand" / "sweep.yaml"


def small_grid(height: int = 40, width: int = 41, res: float = 30.0) -> GridSpec:
    return GridSpec(crs="EPSG:32645", transform=Affine(res, 0, 300000, 0, -res, 3000000 + height
                                                       * res), width=width, height=height)


def v_valley(grid: GridSpec, sx: float = 0.1, sy: float = 0.01) -> tuple[np.ndarray, int]:
    """V-shaped valley: channel in the centre column sloping south, sides rising at ``sx``."""
    rows, cols = np.indices(grid.shape)
    c0 = grid.width // 2
    z = 1000.0 - sy * grid.res * rows + sx * grid.res * np.abs(cols - c0)
    return z.astype(np.float64), c0


# --- Step 1: grid and DEM units --------------------------------------------------------------


def test_grid_for_bounds_snaps_outward():
    grid = demmod.grid_for_bounds((321359.3, 3047027.9, 354600.4, 3078024.3), 30, 2000,
                                  "EPSG:32645")
    left, bottom, right, top = grid.bounds
    assert left % 30 == 0 and top % 30 == 0
    assert left <= 319359.3 and bottom <= 3045027.9 and right >= 356600.4 and top >= 3080024.3
    assert (grid.width, grid.height) == (1242, 1168)


def _write_scaled(path: Path, values: np.ndarray, grid: GridSpec, scale: float) -> None:
    profile = grid.profile("float32", -9999.0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)
        dst.scales = (scale,)


def test_gedtm30_scale_tag_is_not_applied(tmp_path: Path):
    """GEDTM30 stores metres under a 0.1 scale tag (plan 002): never apply it by default."""
    grid = small_grid(8, 8)
    metres = np.full(grid.shape, 1234.5, dtype=np.float32)
    _write_scaled(tmp_path / "gedtm.tif", metres, grid, scale=0.1)
    with rasterio.open(tmp_path / "gedtm.tif") as src:
        assert src.scales == (0.1,)
    raw = demmod.prepare_dem(tmp_path / "gedtm.tif", grid)
    np.testing.assert_allclose(raw, 1234.5, atol=1e-3)
    scaled = demmod.prepare_dem(tmp_path / "gedtm.tif", grid, apply_scale=True)
    np.testing.assert_allclose(scaled, 123.45, atol=1e-3)
    domain = np.ones(grid.shape, dtype=bool)
    assert demmod.check_elevation_units(raw, metres, domain, 50) == pytest.approx(0, abs=1e-3)
    with pytest.raises(demmod.UnitsError):
        demmod.check_elevation_units(scaled, metres, domain, 50)


def test_sweep_config_keeps_gedtm30_unscaled():
    config = yaml.safe_load(REPO_SWEEP.read_text(encoding="utf-8"))
    assert config["dems"]["gedtm30"]["apply_scale"] is False
    assert not any(entry.get("apply_scale") for entry in config["dems"].values())


def test_prepare_dem_maps_nodata_to_nan(tmp_path: Path):
    grid = small_grid(6, 6)
    values = np.full(grid.shape, 900.0, dtype=np.float32)
    values[0, 0] = -9999.0
    _write_scaled(tmp_path / "d.tif", values, grid, scale=1.0)
    out = demmod.prepare_dem(tmp_path / "d.tif", grid, resampling="nearest")
    assert np.isnan(out[0, 0]) and np.nanmin(out) == pytest.approx(900.0)


# --- Step 2: burn ----------------------------------------------------------------------------


def test_burn_and_reference_round_trip():
    grid = small_grid(10, 10)
    z = np.full(grid.shape, 1000.0, dtype=np.float32)
    left, bottom, right, top = grid.bounds
    line = LineString([(left + 5 * 30 + 15, top), (left + 5 * 30 + 15, bottom)])
    import geopandas as gpd

    mask = cond.burn_mask(gpd.GeoDataFrame(geometry=[line], crs=grid.crs), grid)
    assert mask[:, 5].all() and mask.sum() >= 10
    burned = cond.burn(z, mask, 60)
    assert burned[mask].max() == pytest.approx(940.0) and burned[~mask].min() == 1000.0
    np.testing.assert_allclose(cond.reference_elevation(burned, mask, z), z)
    # a fill that partly refilled the trench (+20 m) is undone; one above the terrain is kept
    refilled = burned + 20 * mask
    np.testing.assert_allclose(cond.reference_elevation(refilled, mask, z), z)
    raised = burned + 70 * mask
    assert cond.reference_elevation(raised, mask, z)[mask].min() == pytest.approx(1010.0)
    assert cond.reference_elevation(z, None, z) is not z


def test_read_overpass_lines(tmp_path: Path):
    doc = {"elements": [
        {"type": "way", "id": 1, "tags": {"name": "A"},
         "geometry": [{"lat": 27.6, "lon": 85.3}, {"lat": 27.61, "lon": 85.31}]},
        {"type": "node", "id": 2},
    ]}
    path = tmp_path / "rivers.json"
    path.write_text(__import__("json").dumps(doc), encoding="utf-8")
    lines = cond.read_lines([path], "EPSG:32645")
    assert len(lines) == 1 and lines.iloc[0]["name"] == "A"


# --- Steps 3-5: routing and HAND -------------------------------------------------------------


def test_pysheds_numpy_shim():
    route.pysheds_grid(small_grid(3, 3))
    assert hasattr(np, "in1d")
    np.testing.assert_array_equal(np.in1d(np.array([1, 5]), [1, 2]), [True, False])


def test_downstream_index_and_catchment():
    fdir = np.array([[4, 4, 8], [4, 4, 8], [1, 1, 4]])  # all water ends at (2, 2)
    ds = route.downstream_index(fdir)
    assert ds[0] == 3 and ds[2] == 4 and ds[6] == 7 and ds[8] == -1
    acc = np.array([[1, 1, 1], [2, 4, 1], [3, 8, 9]], dtype=float)
    basin = route.catchment(ds, acc, 8)
    assert basin.all()
    assert route.catchment(ds, acc, 3).nonzero()[0].tolist() == [0, 3]


def test_hand_tilted_plane_matches_analytic():
    """Plan 008 Step 5: V-valley with one channel; HAND = 30 m x sx x |col - c0| (1e-3 m)."""
    grid = small_grid()
    z, c0 = v_valley(grid)
    fdir = route.flowdir(z, grid)
    streams = np.zeros(grid.shape, dtype=bool)
    streams[:, c0] = True
    raw = handmod.hand_pysheds(fdir, z, streams, grid)
    cols = np.indices(grid.shape)[1]
    analytic = 30 * 0.1 * np.abs(cols - c0)
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(raw[interior], analytic[interior], atol=1e-3)
    domain = np.zeros(grid.shape, dtype=bool)
    domain[interior] = True
    hand, report = handmod.check_hand(raw, domain, streams, z)
    assert report.ok, report.problems
    assert report.stream_max_abs_m == 0 and hand[domain].min() >= 0


def test_pyflwdir_and_whitebox_on_valley():
    """WhiteboxTools follows the steepest slope (lateral, like pysheds). pyflwdir ``from_dem``
    drains to the lowest neighbour (largest drop), i.e. diagonally here, reaching the channel
    d = |col - c0| rows downstream: HAND = 30 m x (sx + sy) x d. H1 measures this difference."""
    grid = small_grid()
    z, c0 = v_valley(grid)
    streams = np.zeros(grid.shape, dtype=bool)
    streams[:, c0] = True
    rows, cols = np.indices(grid.shape)
    d = np.abs(cols - c0)
    analytic = 30 * 0.1 * d
    interior = (slice(1, -1), slice(1, -1))
    flw, _ = handmod.hand_pyflwdir(z, z, streams, grid)
    reach = (rows + d <= grid.height - 2) & (cols > 0) & (cols < grid.width - 1)
    np.testing.assert_allclose(flw[reach], (30 * (0.1 + 0.01) * d)[reach], atol=1e-3)
    try:
        wbt = cond.whitebox_tools()
    except Exception as error:  # binary download blocked
        pytest.skip(f"WhiteboxTools unavailable: {error}")
    wb = handmod.hand_whitebox(z, streams, grid, wbt)
    np.testing.assert_allclose(wb[interior], analytic[interior], atol=1e-3)


def test_check_hand_flags_unit_problems():
    grid = small_grid(5, 5)
    z = np.linspace(1000, 1010, 25).reshape(5, 5)
    streams = np.zeros(grid.shape, dtype=bool)
    streams[2, 2] = True
    domain = np.ones(grid.shape, dtype=bool)
    raw = np.full(grid.shape, 2.0)
    raw[2, 2] = 0.0
    assert handmod.check_hand(raw, domain, streams, z)[1].ok
    _, report = handmod.check_hand(raw * 1000, domain, streams, z)  # the old 0-31633 problem
    assert not report.ok and "relief" in report.problems[0]
    raw[0, 0] = np.nan
    assert "non-finite" in handmod.check_hand(raw, domain, streams, z)[1].problems[0]
    raw[0, 0] = -1.0
    hand, report = handmod.check_hand(raw, domain, streams, z)
    assert report.ok and hand[0, 0] == 0 and report.negative_min_m == -1.0


def test_stream_mask_uses_contributing_area():
    grid = small_grid(2, 2)
    acc = np.array([[1, 111], [112, 2000]], dtype=float)  # 0.1 km2 = 111.1 cells of 900 m2
    np.testing.assert_array_equal(handmod.stream_mask(acc, grid, 0.1),
                                  [[False, False], [True, True]])


# --- Step 4: drop analysis -------------------------------------------------------------------


def test_strahler_streams_y_network():
    #   0   1        two first-order heads (0 and 1) join at 3, which flows to 4 then out.
    #     3
    #     4
    n = 5
    ds = np.array([3, 3, -1, 4, -1])
    stream = np.array([True, True, False, True, True])
    acc = np.array([1, 1, 1, 3, 4], dtype=float)
    s = drop_analysis.strahler_streams(ds, stream, acc)
    assert sorted(s.order.tolist()) == [1, 1, 2]
    second = int(np.flatnonzero(s.order == 2)[0])
    assert s.start[second] == 3 and s.junction[second] == -1  # leaves the network
    firsts = np.flatnonzero(s.order == 1)
    assert set(s.junction[firsts].tolist()) == {3}
    z = np.array([10.0, 12.0, 0.0, 5.0, 4.0])
    test = drop_analysis.drop_test(s, z, 1.0, n)
    assert test.n_first == 2 and test.n_higher == 0  # the order-2 stream is incomplete
    assert test.mean_drop_first_m == pytest.approx(6.0)


def test_drop_analysis_selects_smallest_passing_threshold():
    # Two identical Y networks feeding a trunk; drops are constant so t is undefined -> none.
    ds = np.array([2, 2, 3, -1])
    stream = np.ones(4, dtype=bool)
    tests, selected = drop_analysis.drop_analysis(
        ds, np.array([1.0, 1.0, 3.0, 4.0]), np.array([10.0, 9.0, 5.0, 1.0]), stream,
        [0.5, 2.0])
    assert [t.threshold_km2 for t in tests] == [0.5, 2.0]
    assert selected is None  # fewer than two streams per class


def test_snap_outlet_picks_max_accumulation_near_line():
    grid = small_grid(10, 10)
    acc = np.arange(100, dtype=float).reshape(10, 10)
    left, bottom, right, top = grid.bounds
    line = LineString([(left + 75, top), (left + 75, bottom)])  # column 2 centres
    point = box(left, bottom, right, top).centroid
    idx = route.snap_outlet(acc, grid, line, point, line_buffer_m=10, max_dist_m=200)
    assert idx is not None and idx % 10 == 2


def test_fill_keeps_flat_resolution_at_high_elevation():
    """Regression: float32 at ~1300 m erases the 1e-5 m resolve_flats increments."""
    grid = small_grid(30, 30)
    rows, cols = np.indices(grid.shape)
    z = 1300.0 + 0.5 * np.maximum(np.abs(rows - 15), np.abs(cols - 15)).astype(np.float64)
    z[10:21, 10:21] = 1290.0  # a broad depression that filling turns into a flat
    z[15, 0] = 1200.0  # outlet on the west edge
    filled = cond.fill(z, grid)
    assert filled.dtype == np.float64
    fdir = route.flowdir(filled, grid)
    assert not np.isin(fdir[1:-1, 1:-1], (-1, -2)).any()


def test_cli_registers_hand_commands(capsys):
    from nakkhu.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["hand", "--help"])
    out = capsys.readouterr().out
    for action in ("run", "sweep", "attribute"):
        assert action in out
    args = build_parser().parse_args(["hand", "attribute", "--inventory", "all"])
    assert args.hand_config == "configs/hand/sweep.yaml" and args.sweep is None


def test_check_hand_tolerates_cells_draining_off_grid():
    grid = small_grid(3, 3)
    z = np.arange(9, dtype=float).reshape(3, 3)
    streams = np.zeros(grid.shape, dtype=bool)
    streams[1, 1] = True
    raw = np.zeros(grid.shape)
    raw[0, 0] = np.nan
    domain = np.ones(grid.shape, dtype=bool)
    fdir = np.full(grid.shape, 4)
    fdir[0, 0] = 16  # west: leaves the grid without meeting the stream
    _, report = handmod.check_hand(raw, domain, streams, z, fdir)
    assert report.ok and report.n_nan_exit_grid == 1
    # (1, 0) drains east into an interior pit at (1, 1) that is not a stream: a real failure.
    raw[0, 0], raw[1, 0] = 0.0, np.nan
    fdir[1, 0], fdir[1, 1] = 1, -2
    streams[1, 1] = False
    _, report = handmod.check_hand(raw, domain, streams, z, fdir)
    assert not report.ok and report.n_nan_exit_grid == 0


def test_pyflwdir_hand_on_given_d8_matches_pysheds():
    """H1 isolation: pyflwdir's HAND step on pysheds (and Whitebox) D8 = pysheds HAND."""
    grid = small_grid()
    z, c0 = v_valley(grid)
    streams = np.zeros(grid.shape, dtype=bool)
    streams[:, c0] = True
    fdir = route.flowdir(z, grid)
    ref = handmod.hand_pysheds(fdir, z, streams, grid)
    iso = handmod.hand_pyflwdir_on_d8(fdir, z, streams, grid)
    ok = np.isfinite(ref)
    np.testing.assert_allclose(iso[ok], ref[ok], atol=1e-6)
    try:
        wbt = cond.whitebox_tools()
    except Exception as error:  # binary download blocked
        pytest.skip(f"WhiteboxTools unavailable: {error}")
    d8 = handmod.whitebox_d8(z, grid, wbt)
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_array_equal(d8[interior], fdir[interior])
