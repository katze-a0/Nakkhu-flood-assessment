"""OSM stream burning and hydrological conditioning (plan 008, Step 2; factor H4).

The burn lowers the cells touched by OSM river lines by a fixed depth in the DEM used for flow
routing only. HAND is measured against the *reference* elevations (:func:`reference_elevation`):
the conditioned DEM, with each burned cell restored to at least its original elevation.

Conditioning methods:

- ``fill``: pysheds ``fill_pits`` + priority-flood ``fill_depressions`` + ``resolve_flats``.
- ``breach``: WhiteboxTools ``breach_depressions_least_cost`` (unbreached depressions filled),
  then pysheds ``fill_depressions`` + ``resolve_flats`` so no flat or pit is left for D8.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import LineString

from nakkhu.hydro.dem import GridSpec, read_raster, write_raster
from nakkhu.hydro.route import as_raster, pysheds_grid

WBT_NODATA = -9999.0


def read_overpass_lines(path: str | Path) -> gpd.GeoDataFrame:
    """LineStrings (EPSG:4326) of the ``way`` elements of an Overpass JSON with geometry."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for element in document.get("elements", []):
        coords = [(p["lon"], p["lat"]) for p in element.get("geometry") or []]
        if element.get("type") == "way" and len(coords) >= 2:
            tags = element.get("tags") or {}
            rows.append({"osm_way": element["id"], "name": tags.get("name"),
                         "geometry": LineString(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def read_lines(paths: Iterable[str | Path], crs: str) -> gpd.GeoDataFrame:
    """Union of line files (GeoJSON or Overpass JSON), reprojected to ``crs``."""
    frames = []
    for path in paths:
        path = Path(path)
        head = json.loads(path.read_text(encoding="utf-8"))
        frame = read_overpass_lines(path) if "elements" in head else gpd.read_file(path)
        frames.append(frame.to_crs(crs)[[c for c in ("name", "geometry") if c in frame]])
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    return merged[merged.geometry.notna() & ~merged.geometry.is_empty]


def burn_mask(lines: gpd.GeoDataFrame, grid: GridSpec, *, all_touched: bool = True) -> np.ndarray:
    geoms = [(geom, 1) for geom in lines.to_crs(grid.crs).geometry]
    if not geoms:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize(geoms, out_shape=grid.shape, transform=grid.transform,
                     all_touched=all_touched, fill=0, dtype="uint8").astype(bool)


def burn(dem: np.ndarray, mask: np.ndarray, depth_m: float) -> np.ndarray:
    return (dem.astype(np.float64) - float(depth_m) * mask)


def reference_elevation(conditioned: np.ndarray, mask: np.ndarray | None,
                        original: np.ndarray) -> np.ndarray:
    """HAND reference surface: conditioned elevations; burned cells take
    ``max(conditioned, original)``, i.e. the burn is undone without keeping a fill that only
    refilled the burned trench (adding the burn depth back would leave such cells up to the
    burn depth above the terrain)."""
    ref = conditioned.astype(np.float64)
    if mask is None:
        return ref
    return np.where(mask, np.maximum(ref, original.astype(np.float64)), ref)


def _fill_and_resolve(dem: np.ndarray, grid: GridSpec) -> np.ndarray:
    pgrid, _, _ = pysheds_grid(grid)
    raster = as_raster(dem.astype(np.float64), grid, np.nan)
    filled = pgrid.fill_depressions(pgrid.fill_pits(raster))
    return np.asarray(pgrid.resolve_flats(filled), dtype=np.float64)


def fill(dem: np.ndarray, grid: GridSpec) -> np.ndarray:
    return _fill_and_resolve(dem, grid)


def whitebox_tools() -> Any:
    """A quiet, single-threaded ``WhiteboxTools`` instance (downloads the binary on first use).

    One thread because the multithreaded least-cost breach is not bit-reproducible (mm-level
    differences between identical runs, checked 2026-10-05)."""
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_max_procs(1)
    return wbt


def whitebox_version(wbt: Any) -> str:
    text = wbt.version() or ""
    return text.strip().splitlines()[0] if text.strip() else "unknown"


def breach(dem: np.ndarray, grid: GridSpec, *, max_dist_cells: int,
           wbt: Any | None = None, workdir: str | Path | None = None) -> np.ndarray:
    wbt = wbt or whitebox_tools()
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        src, out = Path(tmp) / "dem.tif", Path(tmp) / "breached.tif"
        write_raster(src, np.nan_to_num(dem, nan=WBT_NODATA).astype(np.float32), grid,
                     nodata=WBT_NODATA, plain=True)
        rc = wbt.breach_depressions_least_cost(
            str(src), str(out), dist=int(max_dist_cells), max_cost=None, min_dist=True,
            flat_increment=None, fill=True,
        )
        if rc != 0 or not out.is_file():
            raise RuntimeError(f"WhiteboxTools breach_depressions_least_cost failed (rc={rc})")
        breached = read_raster(out).astype(np.float64)
    breached[np.isnan(dem)] = np.nan
    return _fill_and_resolve(breached, grid)


def condition(dem: np.ndarray, grid: GridSpec, method: str, *, max_dist_cells: int = 100,
              wbt: Any | None = None, workdir: str | Path | None = None) -> np.ndarray:
    if method == "fill":
        return fill(dem, grid)
    if method == "breach":
        return breach(dem, grid, max_dist_cells=max_dist_cells, wbt=wbt, workdir=workdir)
    raise ValueError(f"unknown conditioning method {method!r}; expected fill or breach")
