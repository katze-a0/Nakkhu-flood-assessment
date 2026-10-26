"""Pixel -> map georeferencing, OBB polygons and cross-tile de-duplication (plan 003).

Pixel convention (fixed by the prototype, ``prototype.ipynb`` cell 19, and checked against its
on-disk outputs in ``tests/test_georef_prototype.py``): detector coordinates are continuous
image coordinates with (0, 0) at the outer top-left corner of the tile, and map coordinates are
``affine * (x, y)`` with no half-pixel shift.

Cross-tile de-duplication: a detection belongs to the tile whose core (the central
``stride x stride`` px of the tile; the cores partition the grid) contains its centre. A
detection whose owning tile was not part of the run is kept. Remaining duplicates between
different tiles are removed by greedy rotated-IoU NMS (shapely polygons, STRtree).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from shapely.strtree import STRtree

from nakkhu.config import Project

WORK_CRS = "EPSG:32645"
CORNER_COLUMNS = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
DETECTIONS_PX = "detections_px.parquet"
INPUTS_FILE = "inputs.json"
DETECTIONS_GPKG = "detections.gpkg"


class GeorefError(ValueError):
    """Inputs cannot be georeferenced safely."""


def as_affine(value: Affine | Sequence[float]) -> Affine:
    """An :class:`Affine` from an Affine or its first six coefficients ``(a, b, c, d, e, f)``."""
    if isinstance(value, Affine):
        return value
    a, b, c, d, e, f = (float(v) for v in list(value)[:6])
    return Affine(a, b, c, d, e, f)


def pixel_to_map(affine: Affine | Sequence[float], xy: np.ndarray) -> np.ndarray:
    """Map coordinates of continuous pixel coordinates ``xy`` (``(..., 2)``)."""
    t = as_affine(affine)
    xy = np.asarray(xy, dtype="float64")
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f], axis=-1)


def map_to_pixel(affine: Affine | Sequence[float], xy: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pixel_to_map`."""
    return pixel_to_map(~as_affine(affine), xy)


def is_north_up(affine: Affine | Sequence[float]) -> bool:
    t = as_affine(affine)
    return t.b == 0 and t.d == 0 and t.a > 0 and t.e < 0


def to_work_crs(xy: np.ndarray, crs: str | CRS, work_crs: str = WORK_CRS) -> np.ndarray:
    """Reproject ``(..., 2)`` map coordinates to the working CRS (no-op if already there)."""
    src = CRS.from_user_input(crs)
    dst = CRS.from_user_input(work_crs)
    xy = np.asarray(xy, dtype="float64")
    if src == dst:
        return xy
    flat = xy.reshape(-1, 2)
    xs, ys = warp_transform(src, dst, flat[:, 0].tolist(), flat[:, 1].tolist())
    return np.stack([np.asarray(xs), np.asarray(ys)], axis=-1).reshape(xy.shape)


def obb_polygons(corners_map: np.ndarray) -> np.ndarray:
    """Shapely polygons from ``(N, 4, 2)`` map corners."""
    corners_map = np.asarray(corners_map, dtype="float64")
    if corners_map.size == 0:
        return np.array([], dtype=object)
    return shapely.polygons(corners_map)


# --- de-duplication -------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGrid:
    """The tile lattice of one tile index: grid origin, pixel size, tile size and stride."""

    west: float
    north: float
    pixel_size: float
    tile_px: int
    stride_px: int

    @property
    def margin_px(self) -> float:
        return (self.tile_px - self.stride_px) / 2.0

    def owner(self, xy_map: np.ndarray) -> np.ndarray:
        """``(N, 2)`` integer (col, row) of the tile whose core contains each map point."""
        xy_map = np.asarray(xy_map, dtype="float64").reshape(-1, 2)
        gx = (xy_map[:, 0] - self.west) / self.pixel_size
        gy = (self.north - xy_map[:, 1]) / self.pixel_size
        col = np.floor((gx - self.margin_px) / self.stride_px).astype("int64")
        row = np.floor((gy - self.margin_px) / self.stride_px).astype("int64")
        return np.stack([col, row], axis=-1)


def ownership_mask(
    tile_cols: np.ndarray, tile_rows: np.ndarray, centres_map: np.ndarray, grid: TileGrid,
    present: set[tuple[int, int]],
) -> np.ndarray:
    """True where a detection is kept by the centre-ownership rule."""
    owner = grid.owner(centres_map)
    keep = np.empty(len(owner), dtype=bool)
    for i, (col, row) in enumerate(owner):
        key = (int(col), int(row))
        keep[i] = key not in present or (key == (int(tile_cols[i]), int(tile_rows[i])))
    return keep


def cross_tile_nms(
    polygons: Sequence[Any] | np.ndarray, scores: np.ndarray, tiles: Sequence[Any] | np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy rotated-IoU NMS that only suppresses pairs from different tiles.

    Returns a boolean keep mask. Pairs within one tile are left to the detector's own NMS.
    """
    n = len(polygons)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    geoms = np.asarray(polygons, dtype=object)
    tree = STRtree(geoms)
    order = np.argsort(-np.asarray(scores, dtype="float64"), kind="stable")
    rank = np.empty(n, dtype="int64")
    rank[order] = np.arange(n)
    for i in order:
        if not keep[i]:
            continue
        for j in tree.query(geoms[i], predicate="intersects"):
            if j == i or not keep[j] or rank[j] < rank[i] or tiles[j] == tiles[i]:
                continue
            inter = shapely.area(shapely.intersection(geoms[i], geoms[j]))
            union = shapely.area(geoms[i]) + shapely.area(geoms[j]) - inter
            if union > 0 and inter / union > iou_threshold:
                keep[j] = False
    return keep


def area_mask(areas: np.ndarray, min_m2: float | None, max_m2: float | None) -> np.ndarray:
    keep = np.ones(len(areas), dtype=bool)
    if min_m2 is not None:
        keep &= areas >= float(min_m2)
    if max_m2 is not None:
        keep &= areas <= float(max_m2)
    return keep


# --- detections table -> map ----------------------------------------------------------------


def georeference(
    detections: pd.DataFrame,
    index: pd.DataFrame,
    grid: TileGrid,
    *,
    iou_threshold: float,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    work_crs: str = WORK_CRS,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Pixel OBB detections (``tile_id, conf, cls, x1..y4``) -> map polygons, de-duplicated.

    ``index`` is a tile index (``tile_id, col, row, affine, crs``). Returns the kept polygons
    (with ``tile_id, conf, cls, area_m2``) and the counts removed at each stage.
    """
    missing = [c for c in ("tile_id", "conf", "cls", *CORNER_COLUMNS) if c not in detections]
    if missing:
        raise GeorefError(f"detections lack columns {missing}")
    tiles = index.set_index("tile_id")
    unknown = sorted(set(detections["tile_id"]) - set(tiles.index))
    if unknown:
        raise GeorefError(f"{len(unknown)} detection tile ids are not in the tile index, "
                          f"e.g. {unknown[:3]}")
    corners_px = detections[list(CORNER_COLUMNS)].to_numpy("float64").reshape(-1, 4, 2)
    corners_map = np.empty_like(corners_px)
    for tile_id, rows in detections.groupby("tile_id", sort=False).indices.items():
        tile = tiles.loc[tile_id]
        affine = as_affine(tile["affine"])
        if not is_north_up(affine):
            raise GeorefError(f"tile {tile_id}: transform is not north-up")
        mapped = pixel_to_map(affine, corners_px[rows])
        corners_map[rows] = to_work_crs(mapped, tile["crs"], work_crs)
    polygons = obb_polygons(corners_map)
    centres = corners_map.mean(axis=1)
    areas = shapely.area(polygons) if len(polygons) else np.array([], dtype="float64")
    counts = {"input": len(detections)}

    keep = area_mask(np.asarray(areas), min_area_m2, max_area_m2)
    counts["area_filtered"] = int((~keep).sum())
    cols = tiles.loc[detections["tile_id"], "col"].to_numpy("int64")
    rows_ = tiles.loc[detections["tile_id"], "row"].to_numpy("int64")
    present = {(int(c), int(r)) for c, r in zip(tiles["col"], tiles["row"], strict=True)}
    owned = ownership_mask(cols, rows_, centres, grid, present)
    counts["not_owned"] = int((keep & ~owned).sum())
    keep &= owned
    idx = np.flatnonzero(keep)
    nms = cross_tile_nms(polygons[idx], detections["conf"].to_numpy()[idx],
                         detections["tile_id"].to_numpy()[idx], iou_threshold)
    counts["nms_suppressed"] = int((~nms).sum())
    final = idx[nms]
    counts["kept"] = len(final)
    frame = gpd.GeoDataFrame(
        {
            "tile_id": detections["tile_id"].to_numpy()[final],
            "conf": detections["conf"].to_numpy("float64")[final],
            "cls": detections["cls"].to_numpy("int64")[final],
            "area_m2": np.asarray(areas)[final],
        },
        geometry=list(polygons[final]),
        crs=work_crs,
    )
    return frame, counts


def tile_grid_from_metadata(meta: dict[str, Any]) -> TileGrid:
    grid = meta["grid"]
    return TileGrid(
        west=float(grid["origin"]["west"]),
        north=float(grid["origin"]["north"]),
        pixel_size=float(grid["pixel_size_m"]),
        tile_px=int(meta["tile_px"]),
        stride_px=int(meta["stride_px"]),
    )


def write_detections(path: str | Path, frame: gpd.GeoDataFrame, run_id: str) -> None:
    """Write layers ``obb`` (polygons) and ``centroid`` (points) to a GeoPackage atomically."""
    path = Path(path)
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.unlink(missing_ok=True)
    obb = frame.copy()
    obb.insert(0, "run_id", run_id)
    obb.to_file(tmp, layer="obb", driver="GPKG")
    centroid = obb.copy()
    centroid["geometry"] = obb.geometry.centroid
    centroid.to_file(tmp, layer="centroid", driver="GPKG")
    tmp.replace(path)


def load_inputs(run_dir: Path) -> dict[str, Any]:
    path = run_dir / INPUTS_FILE
    if not path.is_file():
        raise GeorefError(f"{path} is missing; an infer run records its tile index there")
    return json.loads(path.read_text(encoding="utf-8"))


def georef_run(project: Project, run_ref: str) -> dict[str, Any]:
    """Georeference ``runs/infer/<id>/detections_px.parquet`` into ``detections.gpkg``.

    Contract for an infer run (plan 007): ``detections_px.parquet`` with ``tile_id, conf, cls,
    x1, y1, ..., x4, y4`` (OBB corners in tile-image pixels) and ``inputs.json`` with
    ``tile_index`` (path of the ``index.parquet`` the tiles came from).
    """
    from nakkhu.runs import Run, atomic_write_json
    from nakkhu.tiling import load_tiling, read_index

    ref = run_ref.replace("\\", "/").strip("/")
    if ref.startswith("runs/"):
        ref = ref[len("runs/"):]
    stage, _, run_id = ref.rpartition("/")
    stage = stage or "infer"
    run_dir = project.paths.runs / stage / run_id
    if not run_dir.is_dir():
        raise GeorefError(f"run not found: {run_dir}")
    detections_path = run_dir / DETECTIONS_PX
    if not detections_path.is_file():
        raise GeorefError(f"{detections_path} is missing")
    inputs = load_inputs(run_dir)
    index, meta = read_index(project.paths.resolve(inputs["tile_index"]))
    cfg = load_tiling(project).values["georef"]
    run = Run(stage=stage, run_id=run_id, path=run_dir)
    with run.step("georef"):
        frame, counts = georeference(
            pd.read_parquet(detections_path), index, tile_grid_from_metadata(meta),
            iou_threshold=float(cfg["dedup_iou"]), min_area_m2=cfg.get("min_area_m2"),
            max_area_m2=cfg.get("max_area_m2"))
        write_detections(run_dir / DETECTIONS_GPKG, frame, run.ref)
        record = {"tile_index": inputs["tile_index"], "grid_sha256": meta["grid_sha256"],
                  "georef_config": cfg, "counts": counts}
        atomic_write_json(run_dir / "georef.json", record)
    return {"run": run.ref, **counts}
