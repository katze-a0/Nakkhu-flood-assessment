"""Grid, mosaic and tiling on a small synthetic project (plan 003, Steps 1 and 3)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import Affine
from shapely.geometry import box

from nakkhu import tiling
from nakkhu.config import load_project

NATIVE = 0.30517578125  # Maxar visual grid
CRS = "EPSG:32645"
WEST, NORTH = 330000.0, 3060000.0


def _geojson(path: Path, geom) -> None:
    from geopandas import GeoDataFrame

    GeoDataFrame(geometry=[geom], crs=CRS).to_file(path, driver="GeoJSON")


def _scene_tif(path: Path, col: int, row: int, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 256, size=(3, height, width), dtype="uint8")
    mask = np.full((height, width), 255, dtype="uint8")
    mask[:, :5] = 0  # an invalid strip on the west edge
    transform = Affine(NATIVE, 0, WEST + col * NATIVE, 0, -NATIVE, NORTH - row * NATIVE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8",
        crs=CRS, transform=transform, tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(data)
        dst.write_mask(mask)


@pytest.fixture
def synth(tmp_path: Path):
    root = tmp_path / "nakkhu"
    (root / "configs" / "aoi").mkdir(parents=True)
    (root / "configs" / "data").mkdir(parents=True)
    # training domain 300 x 500 m; extent a 100 x 150 m box inside it
    training = box(WEST + 10, NORTH - 490, WEST + 290, NORTH - 10)
    extent = box(WEST + 100, NORTH - 300, WEST + 200, NORTH - 150)
    _geojson(root / "configs" / "aoi" / "training.geojson", training)
    _geojson(root / "configs" / "aoi" / "extent.geojson", extent)
    project = {
        "schema_version": 1, "crs": CRS,
        "paths": {"data_root": "data", "manifest": "configs/data/manifest.yaml"},
        "aoi": {"analysis_extent": "configs/aoi/extent.geojson",
                "routing_domain": "configs/aoi/extent.geojson",
                "training_domain": "configs/aoi/training.geojson"},
    }
    (root / "configs" / "project.yaml").write_text(yaml.safe_dump(project), encoding="utf-8")
    entries = []
    # scene A: two overlapping native tiles covering the whole training domain
    _scene_tif(root / "data/raw/scene_a/visual/a_1.tif", 0, 0, 1000, 1700, seed=1)
    _scene_tif(root / "data/raw/scene_a/visual/a_2.tif", 900, 0, 200, 1700, seed=2)
    # scene B: only around the extent
    _scene_tif(root / "data/raw/scene_b/visual/b_1.tif", 250, 400, 500, 700, seed=3)
    for sid in ("scene_a", "scene_b"):
        entries.append({"id": sid, "local_path": f"data/raw/{sid}", "sha256": sid * 2})
    (root / "configs" / "data" / "manifest.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "entries": entries}), encoding="utf-8")
    text = Path(tiling.__file__).parents[2].joinpath("configs", "tiling.yaml").read_text(
        encoding="utf-8")
    scenes = (
        "\nscenes:\n"
        "  scene_a: {catalog: AAAA, purposes: [train, infer]}\n"
        "  scene_b: {catalog: BBBB, purposes: [infer]}\n"
    )
    text = text[: text.index("\nscenes:")] + scenes + text[text.index("\ngeoref:"):]
    for pattern, line in {
        r"^  origin: .*$": "  origin: {west: null, north: null}",
        r"^  width_px: .*$": "  width_px: null",
        r"^  height_px: .*$": "  height_px: null",
        r"^  tile_px: .*$": "  tile_px: 64",  # 32 m tiles
        r"^  train: 640$": "  train: 64",
        r"^  infer: 512$": "  infer: 48",
        r"^test_buffer_m: .*$": "test_buffer_m: 40.0",
        r"^  block_px: .*$": "  block_px: 128",
        r"^  workers: .*$": "  workers: 2",
    }.items():
        text, n = re.subn(pattern, line, text, count=1, flags=re.MULTILINE)
        assert n == 1, pattern
    (root / "configs" / "tiling.yaml").write_text(text, encoding="utf-8")
    return load_project(root), training, extent


def test_compute_grid_rounds_outward_to_tile_multiples():
    grid = tiling.compute_grid(box(331204.04, 3053372.21, 338223.27, 3074612.49), CRS, 0.5, 640)
    assert (grid.west, grid.north) == (331200.0, 3074880.0)
    assert grid.width_px * 0.5 % 320 == 0 and grid.height_px * 0.5 % 320 == 0
    west, south, east, north = grid.bounds
    assert west <= 331204.04 and south <= 3053372.21 and east >= 338223.27 and north >= 3074612.49


def test_frozen_grid_requires_write_and_detects_drift(synth):
    project, _, _ = synth
    cfg = tiling.load_tiling(project).values
    with pytest.raises(tiling.TilingError, match="write-grid"):
        tiling.frozen_grid(project, cfg)
    grid = tiling.write_grid(project)
    assert tiling.frozen_grid(project, tiling.load_tiling(project).values) == grid
    assert grid.west % 32 == 0 and grid.north % 32 == 0
    cfg = tiling.load_tiling(project).values
    cfg["grid"]["origin"]["west"] += 32
    with pytest.raises(tiling.TilingError, match="differs"):
        tiling.frozen_grid(project, cfg)


def test_tiles_share_grid_ids_and_cover_the_extent(synth):
    project, training, extent = synth
    tiling.write_grid(project)
    a_train = tiling.run_tile(project, "scene_a", "train")
    a_infer = tiling.run_tile(project, "scene_a", "infer")
    b_infer = tiling.run_tile(project, "scene_b", "infer")
    with pytest.raises(tiling.TilingError, match="not configured"):
        tiling.run_tile(project, "scene_b", "train")
    cfg = tiling.load_tiling(project).values
    grid = tiling.frozen_grid(project, cfg)

    # Step 1: mosaics share transform and shape exactly
    mosaics = sorted((project.paths.data_root / "interim" / "mosaic").glob("*.tif"))
    assert [m.name for m in mosaics] == ["AAAA_0p5m.tif", "BBBB_0p5m.tif"]
    profiles = []
    for path in mosaics:
        with rasterio.open(path) as src:
            profiles.append((src.transform, src.width, src.height, src.crs.to_string()))
    assert profiles[0] == profiles[1] == (grid.transform, grid.width_px, grid.height_px, CRS)

    frames = {}
    for scene, purpose in (("scene_a", "train"), ("scene_a", "infer"), ("scene_b", "infer")):
        frame, meta = tiling.read_index(tiling.tiles_dir(project, scene, purpose) / "index.parquet")
        frames[scene, purpose] = frame
        assert meta["grid_sha256"] == tiling.grid_sha256(grid, "training_domain")
        assert tiling.lattice_ok(frame, grid, meta["stride_px"])
        assert tiling.coverage_of(frame, extent) == pytest.approx(1.0, abs=1e-12)
        for _, row in frame[frame["image"].notna()].iterrows():
            assert (tiling.tiles_dir(project, scene, purpose) / row["image"]).is_file()
    assert tiling.coverage_of(frames["scene_a", "train"], training) == pytest.approx(1.0)

    # Step 3: same window -> same tile id and bounds in both scenes
    a = frames["scene_a", "infer"].set_index("tile_id")
    b = frames["scene_b", "infer"].set_index("tile_id")
    assert set(a.index) == set(b.index)
    for tid in a.index:
        assert list(a.loc[tid, "bounds"]) == list(b.loc[tid, "bounds"])
        assert list(a.loc[tid, "affine"]) == list(b.loc[tid, "affine"])

    train = frames["scene_a", "train"]
    assert set(train["domain"]) == {"train", "buffer", "test"}
    test = train[train["domain"] == "test"]
    assert (test["extent_frac"] > 0).all()
    assert ((train["domain"] == "train") <= (train["dist_extent_m"] >= 40.0)).all()
    assert a_train["tiles"] == len(train)
    assert a_infer["tiles"] == b_infer["tiles"] == len(a)
    # radiometry stored in the run manifest
    run_dir = project.paths.runs / a_train["run"]
    rad = json.loads((run_dir / "radiometry.json").read_text())
    assert rad["mode"] == "global" and len(rad["percentiles"]) == 3

    summary = tiling.run_summary(project, promote_to=None)
    assert summary["mosaics_share_grid"] is True
    assert summary["cross_scene_ids"]["infer"]["same_id_same_bounds"] is True
    assert all(s["checks"]["lattice_exact"] for s in summary["tile_sets"])


def test_mosaic_average_matches_block_mean(synth):
    project, _, _ = synth
    tiling.write_grid(project)
    cfg = tiling.load_tiling(project).values
    grid = tiling.frozen_grid(project, cfg)
    record = tiling.build_mosaic(project, "scene_b", cfg, grid)
    assert record["valid_px"] == sum(record["histograms"][0])
    again = tiling.build_mosaic(project, "scene_b", cfg, grid)
    assert again["reused"] is True and again["key"] == record["key"]
    with rasterio.open(project.paths.resolve(record["path"])) as src:
        data = src.read()
        valid = src.read_masks(1) > 0
    assert valid.any() and (data[:, ~valid] == 0).all()
    # a homogeneous source would map to the same value; random data average near the mean
    assert abs(float(data[:, valid].mean()) - 128.0) < 3.0
