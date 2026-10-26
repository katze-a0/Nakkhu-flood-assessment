"""D8 flow directions, accumulation and catchments (plan 008, Steps 3-4), via pysheds 0.5.

Direction codes follow pysheds/ESRI: N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32; flats are
-1 and pits -2. pysheds 0.5 calls ``np.in1d``, which NumPy 2.4 removed; :func:`pysheds_grid`
restores it as ``np.isin`` (identical for the 1-D inputs pysheds passes) before importing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from nakkhu.hydro.dem import GridSpec

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # N, NE, E, SE, S, SW, W, NW
OFFSETS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
           16: (0, -1), 32: (-1, -1)}


def _numpy_compat() -> None:
    if not hasattr(np, "in1d"):
        setattr(np, "in1d", np.isin)  # noqa: B010 (removed in NumPy 2.4)


def pysheds_grid(grid: GridSpec, nodata: Any = np.nan) -> tuple[Any, Any, Any]:
    """``(Grid, Raster, ViewFinder factory)`` for ``grid``; imports pysheds lazily."""
    _numpy_compat()
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder

    crs = pyproj.Proj(grid.crs)

    def viewfinder(nodata_value: Any = nodata) -> Any:
        return ViewFinder(affine=grid.transform, shape=grid.shape, crs=crs, nodata=nodata_value)

    return Grid(viewfinder=viewfinder()), Raster, viewfinder


def as_raster(array: np.ndarray, grid: GridSpec, nodata: Any) -> Any:
    _, raster, viewfinder = pysheds_grid(grid, nodata)
    return raster(array, viewfinder=viewfinder(nodata))


def flowdir(dem: np.ndarray, grid: GridSpec) -> np.ndarray:
    """D8 directions (int64) of a conditioned DEM; NaN cells get 0."""
    pgrid, _, _ = pysheds_grid(grid)
    fdir = pgrid.flowdir(as_raster(dem.astype(np.float64), grid, np.nan), dirmap=DIRMAP,
                         nodata_out=np.int64(0))
    return np.asarray(fdir, dtype=np.int64)


def accumulation(fdir: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Number of cells draining through each cell, itself included (float64)."""
    pgrid, _, _ = pysheds_grid(grid)
    acc = pgrid.accumulation(as_raster(fdir, grid, np.int64(0)), dirmap=DIRMAP)
    return np.asarray(acc, dtype=np.float64)


def downstream_index(fdir: np.ndarray) -> np.ndarray:
    """Flat index of each cell's D8 receiver; -1 for pits, flats, nodata and grid exits."""
    height, width = fdir.shape
    rows, cols = np.indices(fdir.shape)
    ds = np.full(fdir.size, -1, dtype=np.int64)
    for code, (dr, dc) in OFFSETS.items():
        sel = fdir == code
        r, c = rows[sel] + dr, cols[sel] + dc
        inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        idx = np.flatnonzero(sel.ravel())
        ds[idx[inside]] = r[inside] * width + c[inside]
    return ds


def topological_order(acc: np.ndarray) -> np.ndarray:
    """Flat indices sorted so that every cell precedes its receiver (ascending accumulation)."""
    return np.argsort(acc.ravel(), kind="stable")


def catchment(ds: np.ndarray, acc: np.ndarray, outlet: int) -> np.ndarray:
    """Boolean flat mask of the cells draining to ``outlet`` (outlet included)."""
    order = topological_order(acc)[::-1]  # receivers before donors
    inside = np.zeros(ds.size, dtype=bool)
    inside[outlet] = True
    # Process downstream to upstream: a cell is inside iff its receiver is inside.
    for cell in order:
        receiver = ds[cell]
        if receiver >= 0 and inside[receiver]:
            inside[cell] = True
    return inside


def mask_to_polygon(mask: np.ndarray, grid: GridSpec) -> BaseGeometry:
    polygons = [
        shape(geom)
        for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform)
        if value == 1
    ]
    return unary_union(polygons)


def snap_outlet(
    acc: np.ndarray,
    grid: GridSpec,
    line: BaseGeometry,
    point: BaseGeometry,
    *,
    line_buffer_m: float,
    max_dist_m: float,
    area_km2: tuple[float, float] | None = None,
) -> int | None:
    """Largest-accumulation cell centred within ``line_buffer_m`` of ``line`` and
    ``max_dist_m`` of ``point`` (optionally with upstream area inside ``area_km2``)."""
    near = line.buffer(line_buffer_m).intersection(point.buffer(max_dist_m))
    if near.is_empty:
        return None
    idx = np.flatnonzero(grid.mask(near).ravel())
    if area_km2 is not None:
        area = acc.ravel()[idx] * grid.cell_area_m2 / 1e6
        idx = idx[(area >= area_km2[0]) & (area <= area_km2[1])]
    if idx.size == 0:
        return None
    return int(idx[np.argmax(acc.ravel()[idx])])
