"""GLO-30 delineations on the geographic tile grid (pyflwdir D8).

Catchment c4 (research ``burn.py``): every OSM line burned ``burn_depth_m`` into the DEM, outlet
snapped to the burned Nakkhu cell next to the confluence. Routing domain (research
``basins.py``): unburned DEM, outlet on the Bagmati just below the confluence. Arithmetic and
candidate ordering follow the research scripts so the registered polygons are reproduced.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

NODATA = -9999.0  # the GLO-30 tile carries no nodata value; the research scripts passed -9999


class BasinError(RuntimeError):
    """No outlet candidate satisfies the snapping rule."""


def read_window(path: str | Path, bounds: Sequence[float]) -> tuple[np.ndarray, Any]:
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        return src.read(1, window=win).astype("float32"), src.window_transform(win)


def _flow(dem: np.ndarray, transform: Any) -> tuple[Any, np.ndarray]:
    import pyflwdir

    flw = pyflwdir.from_dem(data=dem, nodata=NODATA, transform=transform, latlon=True,
                            outlets="edge")
    return flw, flw.upstream_area(unit="km2")


def _basin_polygon(flw: Any, shape_: tuple[int, int], row: int, col: int,
                   transform: Any) -> BaseGeometry:
    basins = flw.basins(idxs=np.array([row * shape_[1] + col]))
    parts = [shape(g) for g, _ in features.shapes(basins.astype("int32"), mask=basins > 0,
                                                  transform=transform)]
    return unary_union(parts)


def _to_work(points: list[Point], work_crs: str) -> gpd.GeoSeries:
    return gpd.GeoSeries(points, crs="EPSG:4326").to_crs(work_crs)


def km2(geom: BaseGeometry, work_crs: str) -> float:
    return float(gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(work_crs).area.iloc[0] / 1e6)


def catchment_c4(dem_path: str | Path, lines: gpd.GeoDataFrame, cfg: dict[str, Any]
                 ) -> tuple[BaseGeometry, dict[str, Any]]:
    """Nakkhu catchment with the OSM burn; returns the polygon (EPSG:4326) and outlet record."""
    work = cfg["work_crs"]
    params = cfg["catchment_c4"]
    rule = params["outlet"]
    dem, tr = read_window(dem_path, cfg["dem_window"])
    burn = features.rasterize([(g, 1) for g in lines.to_crs("EPSG:4326").geometry],
                              out_shape=dem.shape, transform=tr, all_touched=True).astype(bool)
    burned = dem - float(params["burn_depth_m"]) * burn
    flw, upa = _flow(burned, tr)
    rows, cols = np.where(burn & (upa > rule["upa_min_km2"]) & (upa < rule["upa_max_km2"]))
    xs, ys = rasterio.transform.xy(tr, rows, cols)
    points = _to_work([Point(a, b) for a, b in zip(xs, ys, strict=True)], work)
    conf = _to_work([Point(cfg["confluence"])], work).iloc[0]
    cand = sorted([(upa[r, c], r, c, a, b)
                   for p, r, c, a, b in zip(points, rows, cols, xs, ys, strict=True)
                   if p.distance(conf) < rule["max_dist_m"]], reverse=True)
    if not cand:
        raise BasinError("catchment_c4: no burned outlet candidate near the confluence")
    u, r, c, a, b = cand[0]
    poly = _basin_polygon(flw, dem.shape, int(r), int(c), tr)
    stem = unary_union(lines[lines["name"].isin(params["stem_names"])].to_crs(work).geometry)
    poly_work = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(work).iloc[0]
    record = {"outlet_lonlat": [float(a), float(b)], "outlet_upa_km2": round(float(u), 4),
              "area_km2": round(km2(poly, work), 4), "burned_cells": int(burn.sum()),
              "stem_share_inside": round(stem.intersection(poly_work).length / stem.length, 4)}
    return poly, record


def routing_domain(dem_path: str | Path, lines: gpd.GeoDataFrame, cfg: dict[str, Any]
                   ) -> tuple[BaseGeometry, dict[str, Any]]:
    """Upper Bagmati basin to just below the confluence on the unburned DEM (EPSG:4326)."""
    work = cfg["work_crs"]
    rule = cfg["routing_domain"]["outlet"]
    dem, tr = read_window(dem_path, cfg["dem_window"])
    flw, upa = _flow(dem, tr)
    lon0, lat0 = cfg["confluence"]
    rows, cols = np.indices(dem.shape)
    xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel())
    xs = np.asarray(xs).reshape(dem.shape)
    ys = np.asarray(ys).reshape(dem.shape)
    sel = (np.abs(xs - lon0) < rule["window_deg"]) & (np.abs(ys - lat0) < rule["window_deg"])
    r_i, c_i = np.where(sel)
    points = _to_work([Point(xs[r, c], ys[r, c]) for r, c in zip(r_i, c_i, strict=True)], work)
    conf = _to_work([Point(lon0, lat0)], work).iloc[0]
    river = lines[lines["name"].str.contains(rule["line_name_contains"])].to_crs(work)
    zone = unary_union(river.geometry).buffer(rule["line_buffer_m"])
    cand = sorted([(upa[r, c], r, c) for (r, c), p in zip(zip(r_i, c_i, strict=True), points,
                                                         strict=True)
                   if zone.contains(p) and p.distance(conf) < rule["max_dist_m"]], reverse=True)
    if not cand:
        raise BasinError("routing_domain: no outlet candidate on the Bagmati near the confluence")
    u, r, c = cand[0]
    poly = _basin_polygon(flw, dem.shape, int(r), int(c), tr)
    record = {"outlet_lonlat": [float(xs[r, c]), float(ys[r, c])],
              "outlet_upa_km2": round(float(u), 4), "area_km2": round(km2(poly, work), 4)}
    return poly, record
