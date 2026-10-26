"""Fixed tile grid, 0.5 m scene mosaics and tile cutting (plan 003).

- **Grid** (``configs/tiling.yaml`` ``grid``): EPSG:32645, 0.5 m pixels, 640 px tiles; the
  bounds of ``origin_from`` rounded outward to multiples of the tile size in metres. Every scene
  mosaic has exactly this transform and shape, so a window means the same ground in every scene.
- **Mosaic** (``data/interim/mosaic/<catalog>_0p5m.tif``): the plan 002 ``visual`` COGs
  (0.305 m) composited on their common native grid (first valid source wins, sources sorted by
  name), then resampled to the grid with ``Resampling.average`` over valid pixels only. A pixel
  is valid when at least ``mosaic.min_valid_share`` of its footprint is valid. Invalid pixels are
  0 with an internal mask. Per-band histograms of valid pixels are stored in a JSON sidecar and
  give the ``global`` 2-98 percentiles.
- **Tiles** (``data/interim/tiles/<scene>/<purpose>/``): ``images/<tile_id>.jpg`` (quality 95),
  ``masks/<tile_id>.png`` for partly valid tiles, and ``index.parquet`` with the tile affine.
  The tile set is written to a staging directory and swapped into place.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import math
import re
import shutil
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from nakkhu import radiometry
from nakkhu.acquire import geo
from nakkhu.config import Project, ResolvedConfig, config_sha256, load_config, load_yaml
from nakkhu.manifest import load_manifest
from nakkhu.runs import atomic_write_json, atomic_write_text, create_run

TILING_CONFIG = "configs/tiling.yaml"
PURPOSES = ("train", "infer")
INDEX_FILE = "index.parquet"
INDEX_META_KEY = b"nakkhu"
GDAL_ENV = {"GDAL_TIFF_INTERNAL_MASK": "YES"}


class TilingError(RuntimeError):
    """Inputs or the grid do not satisfy the plan 003 design."""


# --- config and grid ------------------------------------------------------------------------


def load_tiling(project: Project) -> ResolvedConfig:
    return load_config(project.paths.resolve(TILING_CONFIG))


def gsd_tag(pixel_size: float) -> str:
    return f"{pixel_size:g}".replace(".", "p") + "m"


@dataclass(frozen=True)
class Grid:
    crs: str
    pixel_size: float
    tile_px: int
    west: float
    north: float
    width_px: int
    height_px: int

    @property
    def tile_m(self) -> float:
        return self.tile_px * self.pixel_size

    @property
    def transform(self) -> Affine:
        return Affine(self.pixel_size, 0.0, self.west, 0.0, -self.pixel_size, self.north)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.west, self.north - self.height_px * self.pixel_size,
                self.west + self.width_px * self.pixel_size, self.north)

    def section(self, origin_from: str) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "pixel_size_m": self.pixel_size,
            "tile_px": self.tile_px,
            "origin_from": origin_from,
            "origin": {"west": self.west, "north": self.north},
            "width_px": self.width_px,
            "height_px": self.height_px,
        }

    def window_bounds(self, x0: int, y0: int, w: int, h: int) -> tuple[float, float, float, float]:
        p = self.pixel_size
        return (self.west + x0 * p, self.north - (y0 + h) * p,
                self.west + (x0 + w) * p, self.north - y0 * p)


def compute_grid(domain: BaseGeometry, crs: str, pixel_size: float, tile_px: int) -> Grid:
    """Grid on the bounds of ``domain`` rounded outward to multiples of the tile size."""
    tile_m = tile_px * pixel_size
    west, south, east, north = domain.bounds
    west = math.floor(west / tile_m) * tile_m
    south = math.floor(south / tile_m) * tile_m
    east = math.ceil(east / tile_m) * tile_m
    north = math.ceil(north / tile_m) * tile_m
    width = round((east - west) / pixel_size)
    height = round((north - south) / pixel_size)
    return Grid(crs, float(pixel_size), int(tile_px), float(west), float(north), width, height)


def _aoi(project: Project, key: str) -> BaseGeometry:
    return geo.read_geometry(geo.aoi_path(project, key))


def computed_grid(project: Project, cfg: dict[str, Any]) -> Grid:
    section = cfg["grid"]
    if section["crs"] != geo.WORK_CRS:
        raise TilingError(f"grid CRS {section['crs']} is not {geo.WORK_CRS}")
    return compute_grid(_aoi(project, section["origin_from"]), section["crs"],
                        float(section["pixel_size_m"]), int(section["tile_px"]))


def frozen_grid(project: Project, cfg: dict[str, Any]) -> Grid:
    """The grid in ``configs/tiling.yaml``, after checking it against a fresh computation."""
    section = cfg["grid"]
    if section.get("width_px") is None:
        raise TilingError("grid not written yet; run `nakkhu tile --write-grid`")
    grid = Grid(section["crs"], float(section["pixel_size_m"]), int(section["tile_px"]),
                float(section["origin"]["west"]), float(section["origin"]["north"]),
                int(section["width_px"]), int(section["height_px"]))
    fresh = computed_grid(project, cfg)
    if fresh != grid:
        raise TilingError(f"configs/tiling.yaml grid {grid} differs from the computed {fresh}; "
                          "rerun `nakkhu tile --write-grid` deliberately (it changes tile ids)")
    return grid


def grid_sha256(grid: Grid, origin_from: str) -> str:
    return config_sha256(grid.section(origin_from))


def write_grid(project: Project) -> Grid:
    """Compute the grid and write its ``origin``, ``width_px``, ``height_px`` into the config."""
    path = project.paths.resolve(TILING_CONFIG)
    cfg = load_yaml(path)
    grid = computed_grid(project, cfg)
    text = path.read_text(encoding="utf-8")
    replacements = {
        r"^  origin: .*$": f"  origin: {{west: {grid.west!r}, north: {grid.north!r}}}",
        r"^  width_px: .*$": f"  width_px: {grid.width_px}",
        r"^  height_px: .*$": f"  height_px: {grid.height_px}",
    }
    for pattern, line in replacements.items():
        text, n = re.subn(pattern, line, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise TilingError(f"{TILING_CONFIG}: no line matching {pattern!r} in the grid section")
    atomic_write_text(path, text)
    if frozen_grid(project, load_yaml(path)) != grid:
        raise TilingError("written grid does not read back")
    return grid


# --- scenes and mosaics ---------------------------------------------------------------------


def scene_entry(project: Project, scene: str) -> dict[str, Any]:
    manifest = project.paths.resolve(
        (project.config.values.get("paths") or {}).get("manifest", "configs/data/manifest.yaml"))
    for entry in load_manifest(manifest):
        if entry["id"] == scene:
            if entry.get("status") == "pending":
                raise TilingError(f"{scene} is pending in the data manifest")
            return entry
    raise TilingError(f"{scene} is not in the data manifest")


def scene_sources(project: Project, scene: str) -> list[Path]:
    base = project.paths.resolve(scene_entry(project, scene)["local_path"])
    files = sorted((base / "visual").glob("*.tif"))
    if not files:
        raise TilingError(f"no visual GeoTIFFs under {base / 'visual'}")
    return files


def mosaic_paths(project: Project, catalog: str, pixel_size: float) -> tuple[Path, Path]:
    stem = f"{catalog}_{gsd_tag(pixel_size)}"
    base = project.paths.data_root / "interim" / "mosaic"
    return base / f"{stem}.tif", base / f"{stem}.json"


@dataclass(frozen=True)
class Native:
    """The common native grid of a scene's sources and each source's pixel offset on it."""

    crs: str
    pixel: float
    west: float
    north: float
    offsets: tuple[tuple[int, int, int, int], ...]  # col, row, width, height per source


def native_grid(files: list[Path], work_crs: str) -> Native:
    infos = []
    for path in files:
        with rasterio.open(path) as src:
            t = src.transform
            if src.crs is None or src.crs.to_string() != work_crs:
                raise TilingError(f"{path.name}: CRS {src.crs} is not {work_crs} (STOP condition)")
            if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
                raise TilingError(f"{path.name}: transform is not north-up (STOP condition)")
            if abs(t.a + t.e) > 1e-12:
                raise TilingError(f"{path.name}: non-square pixels {t.a}, {t.e}")
            if src.count < 3:
                raise TilingError(f"{path.name}: {src.count} bands, expected RGB")
            infos.append((t.a, t.c, t.f, src.width, src.height))
    pixel = infos[0][0]
    if any(abs(i[0] - pixel) > 1e-12 for i in infos):
        raise TilingError("sources have different pixel sizes")
    west = min(i[1] for i in infos)
    north = max(i[2] for i in infos)
    offsets = []
    for _, c, f, w, h in infos:
        col, row = (c - west) / pixel, (north - f) / pixel
        if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
            raise TilingError("sources are not aligned on one native grid")
        offsets.append((round(col), round(row), w, h))
    return Native(work_crs, pixel, west, north, tuple(offsets))


def _win(col: int, row: int, width: int, height: int) -> Window:
    return Window.from_slices((row, row + height), (col, col + width))


_local = threading.local()


def _handles(files: list[Path]) -> list[Any]:
    key = tuple(str(f) for f in files)
    if getattr(_local, "key", None) != key:
        _local.key = key
        _local.handles = [rasterio.open(f) for f in files]
    return _local.handles


def mosaic_block(
    files: list[Path], native: Native, grid: Grid, window: Window, min_valid_share: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Composite and average-resample one grid window; None when no source data fall in it."""
    x0, y0, w, h = int(window.col_off), int(window.row_off), int(window.width), int(window.height)
    west, south, east, north = grid.window_bounds(x0, y0, w, h)
    p = native.pixel
    c0 = math.floor((west - native.west) / p) - 2
    c1 = math.ceil((east - native.west) / p) + 2
    r0 = math.floor((native.north - north) / p) - 2
    r1 = math.ceil((native.north - south) / p) + 2
    data = np.zeros((3, r1 - r0, c1 - c0), dtype="uint8")
    valid = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    hit = False
    for src, (sc, sr, sw, sh) in zip(_handles(files), native.offsets, strict=True):
        ic0, ic1 = max(c0, sc), min(c1, sc + sw)
        ir0, ir1 = max(r0, sr), min(r1, sr + sh)
        if ic0 >= ic1 or ir0 >= ir1:
            continue
        win = _win(ic0 - sc, ir0 - sr, ic1 - ic0, ir1 - ir0)
        mask = src.read_masks(1, window=win) > 0
        if not mask.any():
            continue
        block = src.read([1, 2, 3], window=win)
        sub = (slice(ir0 - r0, ir1 - r0), slice(ic0 - c0, ic1 - c0))
        take = mask & ~valid[sub]
        if take.any():
            hit = True
            target = data[:, sub[0], sub[1]]
            target[:, take] = block[:, take]
            valid[sub] |= take
    if not hit:
        return None
    weights = valid.astype("float32")
    source = np.concatenate([data.astype("float32") * weights, weights[None]], axis=0)
    out = np.zeros((4, h, w), dtype="float32")
    reproject(
        source=source,
        destination=out,
        src_transform=Affine(p, 0.0, native.west + c0 * p, 0.0, -p, native.north - r0 * p),
        src_crs=native.crs,
        dst_transform=Affine(grid.pixel_size, 0.0, west, 0.0, -grid.pixel_size, north),
        dst_crs=grid.crs,
        resampling=Resampling.average,
        src_nodata=-1.0,
        dst_nodata=-1.0,
        num_threads=1,
    )
    share = np.clip(out[3], 0.0, None)
    ok = share >= min_valid_share
    values = np.zeros((3, h, w), dtype="uint8")
    safe = np.where(share > 0, share, 1.0)
    for i in range(3):
        band = np.clip(np.rint(np.clip(out[i], 0.0, None) / safe), 0, 255).astype("uint8")
        band[~ok] = 0
        values[i] = band
    return values, ok


def _blocks(grid: Grid, size: int) -> Iterator[Window]:
    for y0 in range(0, grid.height_px, size):
        for x0 in range(0, grid.width_px, size):
            yield _win(x0, y0, min(size, grid.width_px - x0), min(size, grid.height_px - y0))


def _source_bounds(native: Native) -> BaseGeometry:
    p = native.pixel
    return shapely.union_all([
        box(native.west + c * p, native.north - (r + h) * p,
            native.west + (c + w) * p, native.north - r * p)
        for c, r, w, h in native.offsets])


def _bounded_map(func: Any, items: list[Any], workers: int) -> Iterator[tuple[Any, Any]]:
    """``(item, func(item))`` in order, with at most ``2 * workers`` results in flight."""
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        pending: list[tuple[Any, concurrent.futures.Future[Any]]] = []
        for item in items:
            pending.append((item, pool.submit(func, item)))
            if len(pending) >= 2 * workers:
                head, future = pending.pop(0)
                yield head, future.result()
        for head, future in pending:
            yield head, future.result()


def build_mosaic(
    project: Project, scene: str, cfg: dict[str, Any], grid: Grid, *, rebuild: bool = False
) -> dict[str, Any]:
    """Build (or reuse) the scene mosaic on ``grid``; returns its sidecar record."""
    scene_cfg = cfg["scenes"][scene]
    entry = scene_entry(project, scene)
    tif, sidecar = mosaic_paths(project, scene_cfg["catalog"], grid.pixel_size)
    mosaic_cfg = cfg["mosaic"]
    key = {
        "scene": scene,
        "catalog": scene_cfg["catalog"],
        "manifest_sha256": entry["sha256"],
        "grid_sha256": grid_sha256(grid, cfg["grid"]["origin_from"]),
        "resampling": mosaic_cfg["resampling"],
        "min_valid_share": float(mosaic_cfg["min_valid_share"]),
    }
    if not rebuild and tif.is_file() and sidecar.is_file():
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("key") == key:
            record["reused"] = True
            return record
    record = write_mosaic(project, scene_sources(project, scene), grid, tif, mosaic_cfg,
                          label=scene)
    record = {"key": key, **record}
    atomic_write_json(sidecar, record)
    record["reused"] = False
    return record


def write_mosaic(project: Project, files: list[Path], grid: Grid, tif: Path,
                 mosaic_cfg: dict[str, Any], *, label: str = "mosaic") -> dict[str, Any]:
    """Composite ``files`` (first valid pixel wins, in the given order) and average-resample
    them onto ``grid`` into ``tif``; returns the sidecar record without its key."""
    if mosaic_cfg["resampling"] != "average":
        raise TilingError("only average resampling is implemented (plan 003 design)")
    native = native_grid(files, grid.crs)
    footprint = _source_bounds(native)
    blocks = [w for w in _blocks(grid, int(mosaic_cfg["block_px"]))
              if footprint.intersects(box(*grid.window_bounds(
                  int(w.col_off), int(w.row_off), int(w.width), int(w.height))))]
    tif.parent.mkdir(parents=True, exist_ok=True)
    tmp = tif.with_name(f".{tif.name}.tmp")
    tmp.unlink(missing_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 3, "crs": grid.crs,
        "transform": grid.transform, "width": grid.width_px, "height": grid.height_px,
        "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate",
        "predictor": 2, "interleave": "pixel", "photometric": "RGB", "BIGTIFF": "IF_SAFER",
        "SPARSE_OK": "TRUE",
    }
    hist = np.zeros((3, 256), dtype="int64")
    written = 0
    started = dt.datetime.now(dt.UTC)
    min_share = float(mosaic_cfg["min_valid_share"])
    workers = max(1, int(mosaic_cfg.get("workers", 1)))
    with rasterio.Env(**GDAL_ENV), rasterio.open(tmp, "w", **profile) as dst:
        for window, result in _bounded_map(
                lambda w: mosaic_block(files, native, grid, w, min_share), blocks, workers):
            if result is None:
                continue
            values, ok = result
            dst.write(values, window=window)
            dst.write_mask((ok * 255).astype("uint8"), window=window)
            for i in range(3):
                hist[i] += np.bincount(values[i][ok], minlength=256)
            written += 1
    tmp.replace(tif)
    valid_px = int(hist[0].sum())
    if valid_px == 0:
        raise TilingError(f"{label}: mosaic has no valid pixels")
    return {
        "path": project.paths.relative(tif),
        "sources": [project.paths.relative(f) for f in files],
        "native_pixel_m": native.pixel,
        "blocks_written": written,
        "blocks_total": sum(1 for _ in _blocks(grid, int(mosaic_cfg["block_px"]))),
        "valid_px": valid_px,
        "valid_km2": valid_px * grid.pixel_size ** 2 / 1e6,
        "histograms": hist.tolist(),
        "seconds": round((dt.datetime.now(dt.UTC) - started).total_seconds(), 1),
        "created_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --- tiles ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    col: int
    row: int
    x0: int
    y0: int


def tile_id(col: int, row: int) -> str:
    return f"t_{col:04d}_{row:04d}"


def select_tiles(grid: Grid, stride: int, domain: BaseGeometry) -> list[TileSpec]:
    """Tiles on the ``stride`` lattice whose footprint overlaps ``domain`` with positive area."""
    ncols = math.ceil(grid.width_px / stride)
    nrows = math.ceil(grid.height_px / stride)
    specs = [TileSpec(tile_id(c, r), c, r, c * stride, r * stride)
             for r in range(nrows) for c in range(ncols)]
    boxes = np.array([box(*grid.window_bounds(s.x0, s.y0, grid.tile_px, grid.tile_px))
                      for s in specs], dtype=object)
    shapely.prepare(domain)
    hit = shapely.intersects(domain, boxes)
    areas = np.zeros(len(specs))
    areas[hit] = shapely.area(shapely.intersection(boxes[hit], domain))
    return [s for s, a in zip(specs, areas, strict=True) if a > 0]


def _read_window(src: Any, grid: Grid, x0: int, y0: int) -> tuple[np.ndarray, np.ndarray]:
    size = grid.tile_px
    window = _win(x0, y0, size, size)
    inside = x0 >= 0 and y0 >= 0 and x0 + size <= grid.width_px and y0 + size <= grid.height_px
    if inside:
        return src.read([1, 2, 3], window=window), src.read_masks(1, window=window) > 0
    data = np.zeros((3, size, size), dtype="uint8")
    valid = np.zeros((size, size), dtype=bool)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(grid.width_px, x0 + size), min(grid.height_px, y0 + size)
    if cx0 < cx1 and cy0 < cy1:
        win = _win(cx0, cy0, cx1 - cx0, cy1 - cy0)
        sub = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        data[:, sub[0], sub[1]] = src.read([1, 2, 3], window=win)
        valid[sub] = src.read_masks(1, window=win) > 0
    return data, valid


def valid_share(mosaic: Path, grid: Grid, geom: BaseGeometry, rows: int = 2048) -> float:
    """Share of ``geom`` covered by valid mosaic pixels (pixel centres inside ``geom``)."""
    west, south, east, north = geom.bounds
    p = grid.pixel_size
    c0 = max(0, math.floor((west - grid.west) / p))
    c1 = min(grid.width_px, math.ceil((east - grid.west) / p))
    r0 = max(0, math.floor((grid.north - north) / p))
    r1 = min(grid.height_px, math.ceil((grid.north - south) / p))
    inside_total = 0
    valid_total = 0
    shape = [mapping(geom)]
    with rasterio.open(mosaic) as src:
        for y in range(r0, r1, rows):
            h = min(rows, r1 - y)
            win = _win(c0, y, c1 - c0, h)
            inside = ~geometry_mask(shape, out_shape=(h, c1 - c0),
                                    transform=src.window_transform(win))
            mask = src.read_masks(1, window=win) > 0
            inside_total += int(inside.sum())
            valid_total += int((inside & mask).sum())
    return valid_total / inside_total if inside_total else 0.0


def index_metadata(path: str | Path) -> dict[str, Any]:
    meta = pq.read_schema(Path(path)).metadata or {}
    return json.loads(meta[INDEX_META_KEY])


def read_index(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    return pd.read_parquet(path), index_metadata(path)


def tiles_dir(project: Project, scene: str, purpose: str) -> Path:
    return project.paths.data_root / "interim" / "tiles" / scene / purpose


def write_index(path: Path, frame: pd.DataFrame, meta: dict[str, Any]) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[INDEX_META_KEY] = json.dumps(meta, sort_keys=True).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(schema_meta), path)


def cut_tiles(
    project: Project, scene: str, purpose: str, cfg: dict[str, Any], grid: Grid,
    mosaic: dict[str, Any], run_ref: str,
) -> dict[str, Any]:
    """Cut, stretch and write one scene/purpose tile set; returns computed statistics."""
    if purpose not in PURPOSES:
        raise TilingError(f"purpose must be one of {PURPOSES}")
    if purpose not in cfg["scenes"][scene]["purposes"]:
        raise TilingError(f"{scene} is not configured for purpose {purpose!r}")
    stride = int(cfg["strides"][purpose])
    domain_key = cfg["domains"][purpose]
    domain = _aoi(project, domain_key)
    extent = _aoi(project, "analysis_extent")
    buffer_m = float(cfg["test_buffer_m"])
    rad = cfg["radiometry"]
    mode = rad["mode"]
    percentiles = radiometry.scene_percentiles(mosaic["histograms"], rad["percentiles"])
    quality = int(cfg["tiles"]["jpeg_quality"])
    min_valid = float(cfg["tiles"]["min_valid_frac"])
    origin_from = cfg["grid"]["origin_from"]
    gsha = grid_sha256(grid, origin_from)
    specs = select_tiles(grid, stride, domain)

    target = tiles_dir(project, scene, purpose)
    staging = target.with_name(f".{purpose}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    (staging / "masks").mkdir()
    mosaic_path = project.paths.resolve(mosaic["path"])
    rows: list[dict[str, Any]] = []
    shapely.prepare(extent)
    with rasterio.Env(**GDAL_ENV), rasterio.open(mosaic_path) as src:
        if src.transform != grid.transform or (src.width, src.height) != (grid.width_px,
                                                                          grid.height_px):
            raise TilingError(f"{mosaic_path.name} is not on the frozen grid")
        for spec in specs:
            data, valid = _read_window(src, grid, spec.x0, spec.y0)
            frac = float(valid.mean())
            bounds = grid.window_bounds(spec.x0, spec.y0, grid.tile_px, grid.tile_px)
            footprint = box(*bounds)
            extent_frac = float(shapely.area(shapely.intersection(footprint, extent))
                                / footprint.area)
            dist = float(shapely.distance(footprint, extent))
            domain_label = ("test" if extent_frac > 0
                            else "buffer" if dist < buffer_m else "train")
            image = mask = None
            if frac > 0:
                rgb = radiometry.stretch(data, valid, mode, percentiles)
                image = f"images/{spec.tile_id}.jpg"
                radiometry.write_jpg(staging / image, rgb, quality)
                if frac < 1:
                    mask = f"masks/{spec.tile_id}.png"
                    radiometry.write_mask_png(staging / mask, valid)
            affine = grid.transform @ Affine.translation(spec.x0, spec.y0)
            rows.append({
                "tile_id": spec.tile_id, "col": spec.col, "row": spec.row,
                "x0_px": spec.x0, "y0_px": spec.y0,
                "affine": [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f],
                "crs": grid.crs, "bounds": list(bounds),
                "valid_frac": frac, "low_valid": frac < min_valid,
                "extent_frac": extent_frac, "dist_extent_m": dist, "domain": domain_label,
                "image": image, "mask": mask, "block_id": None, "grid_sha256": gsha,
            })
    frame = pd.DataFrame(rows)
    frame["block_id"] = frame["block_id"].astype("Int64")
    meta = {
        "scene": scene, "purpose": purpose, "catalog": cfg["scenes"][scene]["catalog"],
        "grid": grid.section(origin_from), "grid_sha256": gsha,
        "tile_px": grid.tile_px, "stride_px": stride, "domain": domain_key,
        "radiometry": {"mode": mode, "percentiles_q": rad["percentiles"],
                       "percentiles": percentiles},
        "jpeg_quality": quality, "min_valid_frac": min_valid, "test_buffer_m": buffer_m,
        "mosaic": mosaic["path"], "mosaic_key": mosaic["key"], "run": run_ref,
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_index(staging / INDEX_FILE, frame, meta)
    old = target.with_name(f".{purpose}.old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.replace(old)
    staging.replace(target)
    if old.exists():
        shutil.rmtree(old)
    return index_stats(frame, meta)


def index_stats(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    vf = frame["valid_frac"]
    with_image = frame["image"].notna()
    domains = {d: int((frame["domain"] == d).sum()) for d in ("train", "buffer", "test")}
    usable = {d: int(((frame["domain"] == d) & ~frame["low_valid"]).sum())
              for d in ("train", "buffer", "test")}
    return {
        "scene": meta["scene"], "purpose": meta["purpose"], "stride_px": meta["stride_px"],
        "tiles": len(frame), "with_image": int(with_image.sum()),
        "empty": int((~with_image).sum()), "low_valid": int(frame["low_valid"].sum()),
        "partly_valid": int(((vf > 0) & (vf < 1)).sum()),
        "valid_frac": {"min": float(vf.min()), "mean": float(vf.mean()),
                       "median": float(vf.median())},
        "tiles_by_domain": domains, "tiles_by_domain_valid_ge_min": usable,
        "radiometry": meta["radiometry"],
    }


# --- checks ---------------------------------------------------------------------------------


def coverage_of(frame: pd.DataFrame, geom: BaseGeometry) -> float:
    """Share of ``geom`` inside the union of the tile footprints."""
    footprints = shapely.union_all([box(*b) for b in frame["bounds"]])
    return float(shapely.area(shapely.intersection(geom, footprints)) / geom.area)


def lattice_ok(frame: pd.DataFrame, grid: Grid, stride: int) -> bool:
    """Every tile's bounds lie on the ``stride`` lattice from the grid origin, exactly."""
    step = stride * grid.pixel_size
    size = grid.tile_m
    for (west, south, east, north), col, row in zip(frame["bounds"], frame["col"], frame["row"],
                                                    strict=True):
        if west != grid.west + col * step or north != grid.north - row * step:
            return False
        if east - west != size or north - south != size:
            return False
    return True


# --- stage entry points ---------------------------------------------------------------------


def _resolved(cfg: dict[str, Any], **extra: Any) -> ResolvedConfig:
    return ResolvedConfig(values={"tiling": cfg, **extra}, source=TILING_CONFIG)


def run_tile(
    project: Project, scene: str, purpose: str, *, rebuild_mosaic: bool = False
) -> dict[str, Any]:
    cfg = load_tiling(project).values
    if scene not in cfg["scenes"]:
        raise TilingError(f"unknown scene {scene!r}; configured: {sorted(cfg['scenes'])}")
    run = create_run(project, "tiling", _resolved(cfg, scene=scene, purpose=purpose))
    with run.step("grid"):
        grid = frozen_grid(project, cfg)
    with run.step("mosaic"):
        mosaic = build_mosaic(project, scene, cfg, grid, rebuild=rebuild_mosaic)
        percentiles = radiometry.scene_percentiles(mosaic["histograms"],
                                                   cfg["radiometry"]["percentiles"])
        atomic_write_json(run.path / "radiometry.json", {
            "scene": scene, "mode": cfg["radiometry"]["mode"],
            "percentiles_q": cfg["radiometry"]["percentiles"], "percentiles": percentiles,
            "valid_px": mosaic["valid_px"], "mosaic": mosaic["path"],
            "mosaic_key": mosaic["key"], "mosaic_reused": mosaic["reused"],
        })
    with run.step("tiles"):
        stats = cut_tiles(project, scene, purpose, cfg, grid, mosaic, run.ref)
    stats["index"] = project.paths.relative(tiles_dir(project, scene, purpose) / INDEX_FILE)
    stats["mosaic"] = {k: mosaic[k] for k in ("path", "valid_px", "valid_km2", "reused")}
    run.write_summary(stats)
    return {"run": run.ref, **stats}


def _indices(project: Project) -> list[Path]:
    return sorted((project.paths.data_root / "interim" / "tiles").glob(f"*/*/{INDEX_FILE}"))


def run_summary(project: Project, promote_to: str | None = "tiling/tile_summary.json"
                ) -> dict[str, Any]:
    """Compute ``tile_summary.json`` from every tile index and mosaic, with the plan checks."""
    from nakkhu.artifacts import promote

    cfg = load_tiling(project).values
    run = create_run(project, "tiling", _resolved(cfg, summary=True))
    with run.step("summary"):
        grid = frozen_grid(project, cfg)
        gsha = grid_sha256(grid, cfg["grid"]["origin_from"])
        extent = _aoi(project, "analysis_extent")
        training = _aoi(project, "training_domain")
        sets = []
        ids_bounds: dict[str, dict[str, dict[str, list[float]]]] = {}
        for path in _indices(project):
            frame, meta = read_index(path)
            if meta["grid_sha256"] != gsha:
                raise TilingError(f"{path} was cut on another grid")
            stats = index_stats(frame, meta)
            stats["checks"] = {
                "lattice_exact": lattice_ok(frame, grid, int(meta["stride_px"])),
                "extent_tile_coverage": coverage_of(frame, extent),
                "unique_tile_ids": bool(frame["tile_id"].is_unique),
            }
            if meta["purpose"] == "train":
                stats["checks"]["training_domain_tile_coverage"] = coverage_of(frame, training)
            mosaic = project.paths.resolve(meta["mosaic"])
            stats["extent_valid_share"] = valid_share(mosaic, grid, extent)
            sets.append(stats)
            ids_bounds.setdefault(meta["purpose"], {})[meta["scene"]] = dict(
                zip(frame["tile_id"], (list(b) for b in frame["bounds"]), strict=True))
        cross = {}
        for purpose, scenes in ids_bounds.items():
            names = sorted(scenes)
            shared = 0
            same = True
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    common = set(scenes[a]) & set(scenes[b])
                    shared += len(common)
                    same &= all(scenes[a][t] == scenes[b][t] for t in common)
            cross[purpose] = {"scenes": names, "shared_id_pairs": shared,
                              "same_id_same_bounds": same}
        mosaics = {}
        for path in sorted((project.paths.data_root / "interim" / "mosaic").glob("*.tif")):
            with rasterio.open(path) as src:
                mosaics[path.name] = {"transform": list(src.transform)[:6],
                                      "width": src.width, "height": src.height,
                                      "crs": src.crs.to_string()}
        transforms = {json.dumps(m, sort_keys=True) for m in mosaics.values()}
        summary = {
            "grid": grid.section(cfg["grid"]["origin_from"]), "grid_sha256": gsha,
            "aoi_km2": {"analysis_extent": extent.area / 1e6,
                        "training_domain": training.area / 1e6},
            "tile_sets": sets,
            "cross_scene_ids": cross,
            "mosaics": mosaics,
            "mosaics_share_grid": len(transforms) == 1,
        }
        atomic_write_json(run.path / "tile_summary.json", summary)
    run.write_summary({"tile_sets": len(sets), "grid_sha256": gsha})
    if promote_to:
        promote(f"{run.ref}/tile_summary.json", promote_to, project.paths, replace=True)
    return summary
