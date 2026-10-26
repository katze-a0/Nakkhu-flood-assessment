"""Analysis extent (research ``recommend.py``): (catchment c4 U confluence disc) n Maxar pre n post.

Footprints are the geometries of the registered Maxar STAC item JSONs (one per quadkey tile).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


class ExtentError(RuntimeError):
    """A footprint input is missing or empty."""


def footprint(stac_dirs: Iterable[Path], catalogs: Iterable[str], work_crs: str) -> BaseGeometry:
    """Union of the STAC item geometries of ``catalogs`` (EPSG:32645 by default)."""
    wanted = set(catalogs)
    geoms = []
    for directory in stac_dirs:
        files = sorted(Path(directory).glob("*.json"))
        if not files:
            raise ExtentError(f"no STAC item JSONs in {directory}")
        for path in files:
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["properties"].get("catalog_id") in wanted:
                geoms.append(shape(item["geometry"]))
    if not geoms:
        raise ExtentError(f"no STAC items of catalogs {sorted(wanted)}")
    return unary_union(gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(work_crs).geometry)


def analysis_extent(catchment: BaseGeometry, pre: BaseGeometry, post: BaseGeometry,
                    cfg: dict[str, Any]) -> tuple[BaseGeometry, dict[str, Any]]:
    """Extent polygon in EPSG:4326 from the catchment (EPSG:4326) and footprints (work CRS)."""
    work = cfg["work_crs"]
    params = cfg["analysis_extent"]
    catch = gpd.GeoSeries([catchment], crs="EPSG:4326").to_crs(work).iloc[0]
    conf = gpd.GeoSeries([Point(cfg["confluence"])], crs="EPSG:4326").to_crs(work).iloc[0]
    rec = catch.union(conf.buffer(params["disc_radius_m"])).intersection(pre).intersection(post)
    parts = [g for g in getattr(rec, "geoms", [rec]) if g.area > params["min_part_m2"]]
    rec = unary_union(parts)
    out = gpd.GeoSeries([rec], crs=work).to_crs("EPSG:4326").iloc[0]
    return out, {"area_km2": round(rec.area / 1e6, 4), "parts": len(parts)}
