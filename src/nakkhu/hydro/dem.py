"""The common analysis grid and DEM preparation (plan 008, Step 1).

Every DEM is warped onto one EPSG:32645 grid whose origin is snapped to the cell size, covering
the routing domain plus a buffer. Values are read raw: rasterio never applies a band's
scale/offset tags, and :func:`prepare_dem` only does so when asked. GEDTM30 carries a 0.1 scale
tag although its values are already metres, so the sweep config keeps ``apply_scale: false`` and
:func:`check_elevation_units` rejects any DEM that disagrees with the reference by ~1000 m.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry.base import BaseGeometry


class UnitsError(ValueError):
    """A raster's values are not plausible metres (e.g. a wrongly applied scale factor)."""


@dataclass(frozen=True)
class GridSpec:
    """A north-up raster grid: CRS, affine transform and shape."""

    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def res(self) -> float:
        return float(self.transform.a)

    @property
    def cell_area_m2(self) -> float:
        return abs(self.transform.a * self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return (left, top + self.height * self.transform.e, left + self.width * self.res, top)

    def profile(self, dtype: str, nodata: float | int | None) -> dict[str, Any]:
        return {
            "driver": "GTiff",
            "width": self.width,
            "height": self.height,
            "count": 1,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        t = self.transform
        return float(t.c + (col + 0.5) * t.a), float(t.f + (row + 0.5) * t.e)

    def rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of map coordinates (may fall outside the grid; north-up only)."""
        t = self.transform
        col = (np.asarray(x, dtype=float) - t.c) / t.a
        row = (np.asarray(y, dtype=float) - t.f) / t.e
        return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)

    def mask(self, geometry: BaseGeometry) -> np.ndarray:
        """True for cells whose centre lies inside ``geometry`` (grid CRS)."""
        return geometry_mask(
            [geometry], out_shape=self.shape, transform=self.transform, invert=True
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
        }


def grid_for_bounds(
    bounds: tuple[float, float, float, float], res: float, buffer_m: float, crs: str
) -> GridSpec:
    """Grid covering ``bounds`` + ``buffer_m``, edges snapped outward to multiples of ``res``."""
    left, bottom, right, top = bounds
    x0 = math.floor((left - buffer_m) / res) * res
    y0 = math.floor((bottom - buffer_m) / res) * res
    x1 = math.ceil((right + buffer_m) / res) * res
    y1 = math.ceil((top + buffer_m) / res) * res
    width, height = round((x1 - x0) / res), round((y1 - y0) / res)
    return GridSpec(crs=crs, transform=Affine(res, 0.0, x0, 0.0, -res, y1), width=width,
                    height=height)


def _horizontal_crs(crs: Any) -> Any:
    """The 2D CRS of a 3D geographic CRS (EPSG:4979 -> EPSG:4326), so that warping a geoid grid
    never triggers a vertical transformation; other CRSs are returned unchanged."""
    if crs is not None and crs.to_epsg() == 4979:
        return rasterio.crs.CRS.from_epsg(4326)
    return crs


def prepare_dem(
    path: str | Path,
    grid: GridSpec,
    *,
    resampling: str = "bilinear",
    apply_scale: bool = False,
) -> np.ndarray:
    """Warp band 1 of ``path`` onto ``grid`` as float32 with NaN for nodata.

    Raw stored values are used unless ``apply_scale`` is true, in which case the band's
    scale/offset tags are applied after warping.
    """
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_nodata = src.nodata
        scale, offset = float(src.scales[0]), float(src.offsets[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_crs=_horizontal_crs(src.crs),
            src_nodata=src_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling[resampling],
        )
    if src_nodata is not None and np.isfinite(src_nodata):
        out[out == np.float32(src_nodata)] = np.nan
    if apply_scale:
        out = out * np.float32(scale) + np.float32(offset)
    return out


VERTICAL_DATUMS = ("egm2008", "egm96", "ellipsoid")


def geoid_shift(grid: GridSpec, source: str, target: str,
                geoids: Mapping[str, str | Path]) -> np.ndarray | None:
    """Metres to add to heights in datum ``source`` to express them in datum ``target``.

    Heights relate as ``H = h - N`` (ellipsoidal height h, geoid undulation N), so
    ``H_target = H_source + N_source - N_target`` with ``N_ellipsoid = 0``. Each geoid grid
    (``geoids[name]``: a raster of N in metres, e.g. the PROJ-data NGA grids) is sampled
    bilinearly onto ``grid``. Returns ``None`` when the datums are equal.
    """
    for name in (source, target):
        if name not in VERTICAL_DATUMS:
            raise ValueError(f"unknown vertical datum {name!r}; expected one of {VERTICAL_DATUMS}")
    if source == target:
        return None

    def undulation(name: str) -> np.ndarray:
        if name == "ellipsoid":
            return np.zeros(grid.shape, dtype=np.float32)
        if name not in geoids:
            raise ValueError(f"no geoid grid configured for {name}")
        values = prepare_dem(geoids[name], grid, resampling="bilinear")
        if np.isnan(values).any():
            raise ValueError(f"geoid grid {name} does not cover the analysis grid")
        return values

    return (undulation(source) - undulation(target)).astype(np.float32)


def fill_voids(dem: np.ndarray, donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(filled, mask)``: NaN cells of ``dem`` take the ``donor`` value; ``mask`` marks them."""
    mask = np.isnan(dem) & np.isfinite(donor)
    return np.where(mask, donor, dem).astype(dem.dtype), mask


def check_elevation_units(
    dem: np.ndarray, reference: np.ndarray, domain: np.ndarray, max_median_abs_diff_m: float
) -> float:
    """Median |dem - reference| over ``domain``; raise :class:`UnitsError` above the limit."""
    diff = np.abs(dem[domain].astype(np.float64) - reference[domain].astype(np.float64))
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        raise UnitsError("no overlapping finite cells to compare elevations")
    median = float(np.median(diff))
    if median > max_median_abs_diff_m:
        raise UnitsError(
            f"median |dem - reference| = {median:.1f} m exceeds {max_median_abs_diff_m} m: "
            "check units and scale factors"
        )
    return median


def write_raster(
    path: str | Path,
    array: np.ndarray,
    grid: GridSpec,
    *,
    nodata: float | int | None = None,
    tags: dict[str, Any] | None = None,
    plain: bool = False,
) -> Path:
    """Tiled DEFLATE GeoTIFF; ``plain`` writes it untiled and uncompressed (WhiteboxTools 2.0
    cannot read floating-point predictors)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = grid.profile(str(array.dtype), nodata)
    if plain:
        for key in ("tiled", "blockxsize", "blockysize", "compress"):
            profile.pop(key)
    elif np.issubdtype(array.dtype, np.floating):
        profile["predictor"] = 3
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def read_raster(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read(1)
        nodata = src.nodata
    if nodata is not None and np.issubdtype(array.dtype, np.floating) and np.isfinite(nodata):
        array[array == nodata] = np.nan
    return array
