"""Georeferencing and cross-tile de-duplication (plan 003, Steps 4 and 6)."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from nakkhu import georef, tiling

PIXEL = 0.5
WEST, NORTH = 331200.0, 3074880.0
TILE, STRIDE = 640, 512


def test_round_trip_on_synthetic_geotiff(tmp_path: Path):
    path = tmp_path / "t.tif"
    transform = Affine(PIXEL, 0, WEST + 1234.5, 0, -PIXEL, NORTH - 987.0)
    with rasterio.open(path, "w", driver="GTiff", width=640, height=640, count=1,
                       dtype="uint8", crs="EPSG:32645", transform=transform) as dst:
        dst.write(np.zeros((1, 640, 640), dtype="uint8"))
    with rasterio.open(path) as src:
        affine = src.transform
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(affine.c, affine.c + 320, 500),
                    rng.uniform(affine.f - 320, affine.f, 500)], axis=-1)
    back = georef.pixel_to_map(affine, georef.map_to_pixel(affine, pts))
    assert np.abs(back - pts).max() < 1e-6
    # corner convention: pixel (0, 0) is the outer top-left corner of the tile
    assert georef.pixel_to_map(affine, np.array([0.0, 0.0])).tolist() == [affine.c, affine.f]


def test_reprojection_only_when_not_work_crs():
    xy = np.array([[331500.0, 3060000.0]])
    assert georef.to_work_crs(xy, "EPSG:32645") is xy
    lonlat = georef.to_work_crs(xy, "EPSG:32645", "EPSG:4326")
    again = georef.to_work_crs(lonlat, "EPSG:4326")
    assert np.abs(again - xy).max() < 1e-3


def _index(cols_rows):
    rows = []
    for col, row in cols_rows:
        x0, y0 = col * STRIDE, row * STRIDE
        a = Affine(PIXEL, 0, WEST + x0 * PIXEL, 0, -PIXEL, NORTH - y0 * PIXEL)
        rows.append({"tile_id": tiling.tile_id(col, row), "col": col, "row": row,
                     "affine": [a.a, a.b, a.c, a.d, a.e, a.f], "crs": "EPSG:32645"})
    return pd.DataFrame(rows)


def _detection(tile, map_corners, conf):
    """A detection in ``tile`` (col, row) of a building with the given map corners."""
    col, row = tile
    a = Affine(PIXEL, 0, WEST + col * STRIDE * PIXEL, 0, -PIXEL, NORTH - row * STRIDE * PIXEL)
    px = georef.map_to_pixel(a, np.asarray(map_corners, dtype="float64"))
    rec = {"tile_id": tiling.tile_id(col, row), "conf": conf, "cls": 0}
    for k, (x, y) in enumerate(px, start=1):
        rec[f"x{k}"], rec[f"y{k}"] = float(x), float(y)
    return rec


def _square(cx, cy, half=5.0, angle=0.3):
    c, s = np.cos(angle), np.sin(angle)
    pts = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts]


GRID = georef.TileGrid(WEST, NORTH, PIXEL, TILE, STRIDE)
CORE_EDGE_X = WEST + (STRIDE + (TILE - STRIDE) / 2) * PIXEL  # core boundary between col 0 and 1
CORE_EDGE_Y = NORTH - (STRIDE + (TILE - STRIDE) / 2) * PIXEL


def _run(dets, tiles, **kw):
    frame, counts = georef.georeference(pd.DataFrame(dets), _index(tiles), GRID,
                                        iou_threshold=0.3, **kw)
    return frame, counts


def test_edge_straddling_obb_appears_once():
    tiles = [(0, 0), (1, 0)]
    corners = _square(CORE_EDGE_X - 1.0, NORTH - 100.0)  # in the overlap, centre in col 0 core
    frame, counts = _run([_detection((0, 0), corners, 0.8), _detection((1, 0), corners, 0.9)],
                         tiles)
    assert len(frame) == 1 and frame.iloc[0]["tile_id"] == "t_0000_0000"
    assert counts["not_owned"] == 1


def test_four_tile_corner_obb_appears_once():
    tiles = [(0, 0), (1, 0), (0, 1), (1, 1)]
    corners = _square(CORE_EDGE_X + 0.5, CORE_EDGE_Y - 0.5)  # centre in the (1, 1) core
    dets = [_detection(t, corners, 0.5 + 0.1 * i) for i, t in enumerate(tiles)]
    frame, _ = _run(dets, tiles)
    assert len(frame) == 1 and frame.iloc[0]["tile_id"] == "t_0001_0001"
    poly = frame.geometry.iloc[0]
    assert abs(poly.area - 100.0) < 1e-6 and frame.iloc[0]["area_m2"] == pytest.approx(100.0)


def test_truncated_duplicates_removed_by_nms():
    tiles = [(0, 0), (1, 0)]
    full = _square(CORE_EDGE_X, NORTH - 100.0, half=6.0, angle=0.0)
    left = [(x - 1.0, y) for x, y in full]  # centres fall on either side of the core edge
    right = [(x + 1.0, y) for x, y in full]
    frame, counts = _run([_detection((0, 0), left, 0.7), _detection((1, 0), right, 0.9)], tiles)
    assert counts["not_owned"] == 0 and counts["nms_suppressed"] == 1
    assert len(frame) == 1 and frame.iloc[0]["conf"] == 0.9


def test_two_adjacent_buildings_survive():
    tiles = [(0, 0), (1, 0)]
    a = _square(CORE_EDGE_X - 5.0, NORTH - 100.0, half=5.0, angle=0.0)
    b = _square(CORE_EDGE_X + 5.0, NORTH - 100.0, half=5.0, angle=0.0)  # shares an edge with a
    dets = [_detection((0, 0), a, 0.8), _detection((1, 0), b, 0.8),
            _detection((1, 0), a, 0.6), _detection((0, 0), b, 0.6)]  # each also seen in overlap
    frame, _ = _run(dets, tiles)
    assert len(frame) == 2
    assert sorted(frame["tile_id"]) == ["t_0000_0000", "t_0001_0000"]


def test_owner_outside_the_run_is_kept_and_area_filter():
    corners = _square(WEST + 10.0, NORTH - 10.0)  # in the outer margin of tile (0, 0)
    frame, _ = _run([_detection((0, 0), corners, 0.8)], [(0, 0)])
    assert len(frame) == 1
    frame, counts = _run([_detection((0, 0), corners, 0.8)], [(0, 0)], min_area_m2=150.0)
    assert len(frame) == 0 and counts["area_filtered"] == 1


def test_georef_run_writes_gpkg_layers(project):
    (project.paths.root / "configs" / "tiling.yaml").write_text(
        "georef: {dedup_iou: 0.3, min_area_m2: null, max_area_m2: null}\n", encoding="utf-8")
    index_path = project.paths.data_root / "interim" / "tiles" / "s" / "infer" / "index.parquet"
    index_path.parent.mkdir(parents=True)
    meta = {"grid": {"origin": {"west": WEST, "north": NORTH}, "pixel_size_m": PIXEL},
            "grid_sha256": "x", "tile_px": TILE, "stride_px": STRIDE}
    tiling.write_index(index_path, _index([(0, 0), (1, 0)]), meta)
    run_dir = project.paths.runs / "infer" / "r1"
    run_dir.mkdir(parents=True)
    corners = _square(CORE_EDGE_X - 1.0, NORTH - 100.0)
    pd.DataFrame([_detection((0, 0), corners, 0.8), _detection((1, 0), corners, 0.9)]).to_parquet(
        run_dir / georef.DETECTIONS_PX)
    (run_dir / georef.INPUTS_FILE).write_text(
        json.dumps({"tile_index": project.paths.relative(index_path)}), encoding="utf-8")
    result = georef.georef_run(project, "r1")
    assert result["kept"] == 1
    obb = gpd.read_file(run_dir / georef.DETECTIONS_GPKG, layer="obb")
    centroid = gpd.read_file(run_dir / georef.DETECTIONS_GPKG, layer="centroid")
    assert list(obb.columns) == ["run_id", "tile_id", "conf", "cls", "area_m2", "geometry"]
    assert obb.crs.to_epsg() == 32645 and len(centroid) == 1
    assert centroid.geometry.iloc[0].geom_type == "Point"
    assert obb.iloc[0]["run_id"] == "infer/r1"
