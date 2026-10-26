"""OSM river lines from the dated Geofabrik PBF (no live Overpass query): by way id, or by tag + bbox."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, box


class LinesError(RuntimeError):
    """A configured way is missing from the PBF or cannot be built."""


def read_ways(pbf: str | Path, ways: Sequence[dict[str, Any]], check_names: Sequence[str] = (),
              check_bounds: Sequence[float] | None = None) -> tuple[gpd.GeoDataFrame, dict]:
    """The configured ways as LineStrings (EPSG:4326, config order) plus a provenance record.

    ``ways`` items are ``{id, name}``; the PBF name must equal ``name``. Ways tagged
    ``waterway=river`` with a name in ``check_names`` that touch ``check_bounds`` (lon/lat) but
    are not configured are listed under ``unlisted_same_name`` (reported, never added).
    """
    import osmium
    import osmium.filter
    import osmium.osm

    wanted = {int(w["id"]): w for w in ways}
    names = set(check_names)
    window = box(*check_bounds) if check_bounds is not None else None
    found: dict[int, dict[str, Any]] = {}
    unlisted = []
    processor = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
                 .with_locations()
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                 .with_filter(osmium.filter.KeyFilter("waterway")))
    for obj in processor:
        way: Any = obj  # EntityFilter(WAY) passes ways only
        if way.id not in wanted and (way.tags.get("name") not in names or window is None):
            continue
        coords = [(n.lon, n.lat) for n in way.nodes if n.location.valid()]
        if len(coords) != len(way.nodes) or len(coords) < 2:
            raise LinesError(f"way {way.id}: {len(way.nodes) - len(coords)} nodes without location")
        line = LineString(coords)
        record = {"id": way.id, "name": way.tags.get("name"),
                  "waterway": way.tags.get("waterway"), "version": way.version,
                  "timestamp": str(way.timestamp), "nodes": len(coords), "geometry": line}
        if way.id in wanted:
            found[way.id] = record
        elif record["waterway"] == "river" and window is not None and line.intersects(window):
            unlisted.append({k: v for k, v in record.items() if k != "geometry"})
    missing = [wid for wid in wanted if wid not in found]
    if missing:
        raise LinesError(f"ways not in {Path(pbf).name}: {missing}")
    renamed = [(wid, found[wid]["name"], w["name"]) for wid, w in wanted.items()
               if found[wid]["name"] != w["name"]]
    if renamed:
        raise LinesError(f"way names differ from the config (id, pbf, config): {renamed}")
    frame = gpd.GeoDataFrame(
        {"name": [found[w]["name"] for w in wanted], "osm_way": list(wanted)},
        geometry=[found[w]["geometry"] for w in wanted], crs="EPSG:4326")
    record = {"pbf": Path(pbf).name,
              "ways": [{k: v for k, v in found[w].items() if k != "geometry"} for w in wanted],
              "unlisted_same_name": unlisted}
    return frame, record


def ways_in_bbox(pbf: str | Path, tags: dict[str, str], bbox: Sequence[float],
                 timestamp: str = "") -> dict[str, Any]:
    """Overpass-style ``out geom`` document of the ways with ``tags`` that cross ``bbox``.

    ``bbox`` is (south, west, north, east) as in Overpass QL. A way is selected when any of its
    segments intersects the box, which is how Overpass ``way(bbox)`` behaves. Elements are
    sorted by id and carry ``bounds``, ``geometry`` (lat/lon per node) and all tags, so the result
    has the layout of the registered Overpass JSON.
    """
    import osmium
    import osmium.filter
    import osmium.osm

    south, west, north, east = (float(v) for v in bbox)
    window = box(west, south, east, north)
    processor = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
                 .with_locations()
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                 .with_filter(osmium.filter.TagFilter(*tags.items())))
    elements: list[dict[str, Any]] = []
    for obj in processor:
        way: Any = obj  # EntityFilter(WAY) passes ways only
        if not all(way.tags.get(k) == v for k, v in tags.items()):
            continue
        # 7 decimals = the OSM storage precision (as printed by Overpass)
        nodes = [(round(n.lon, 7), round(n.lat, 7)) for n in way.nodes if n.location.valid()]
        if len(nodes) != len(way.nodes):
            raise LinesError(f"way {way.id}: nodes without location")
        if len(nodes) < 2 or not LineString(nodes).intersects(window):
            continue
        lons = [p[0] for p in nodes]
        lats = [p[1] for p in nodes]
        elements.append({"type": "way", "id": way.id,
                         "bounds": {"minlat": min(lats), "minlon": min(lons),
                                    "maxlat": max(lats), "maxlon": max(lons)},
                         "geometry": [{"lat": lat, "lon": lon} for lon, lat in nodes],
                         "tags": dict(way.tags)})
    elements.sort(key=lambda e: int(e["id"]))
    return {"version": 0.6, "generator": f"nakkhu.aoi.lines from {Path(pbf).name}",
            "osm3s": {"timestamp_osm_base": timestamp,
                      "copyright": "The data included in this document is from "
                                   "www.openstreetmap.org. The data is made available under ODbL."},
            "elements": elements}
