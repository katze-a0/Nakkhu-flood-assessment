"""Named flood sites from ``configs/aoi/flood_reach_sites_2024.yaml`` (no live lookups)."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import Point

METHODS = ("osm_node", "nominatim", "map_read", "stated_in_source")


class SitesError(ValueError):
    """A site entry is incomplete."""


def site_frame(values: dict[str, Any]) -> gpd.GeoDataFrame:
    rows = []
    for i, site in enumerate(values.get("sites") or []):
        missing = [k for k in ("name", "lon", "lat", "method", "provenance") if not site.get(k)]
        if missing:
            raise SitesError(f"site {i}: missing {missing}")
        if site["method"] not in METHODS:
            raise SitesError(f"site {site['name']!r}: method must be one of {METHODS}")
        rows.append({"name": site["name"], "method": site["method"],
                     "coord_source": site["provenance"],
                     "geometry": Point(float(site["lon"]), float(site["lat"]))})
    if not rows:
        raise SitesError("no sites")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")
