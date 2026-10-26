"""Rebuilt vs registered geometry: area, IoU and Hausdorff distance in the work CRS."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry


def _work(geom: BaseGeometry, work_crs: str) -> BaseGeometry:
    return gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(work_crs).iloc[0]


def polygons(rebuilt: BaseGeometry, registered: BaseGeometry, work_crs: str,
             tol: dict[str, float]) -> dict[str, Any]:
    """Compare two EPSG:4326 polygons; ``match`` holds when every tolerance holds."""
    a, b = _work(rebuilt, work_crs), _work(registered, work_crs)
    union = a.union(b).area
    iou = a.intersection(b).area / union if union else 1.0
    rel = abs(a.area - b.area) / b.area if b.area else float(a.area > 0)
    haus = float(shapely.hausdorff_distance(a, b))
    exact = bool(shapely.equals_exact(shapely.normalize(rebuilt), shapely.normalize(registered),
                                      tolerance=0.0))
    return {"area_km2_rebuilt": round(a.area / 1e6, 6), "area_km2_registered": round(b.area / 1e6, 6),
            "area_rel_diff": rel, "iou": iou, "hausdorff_m": haus,
            "vertices_rebuilt": int(shapely.get_num_coordinates(rebuilt)),
            "vertices_registered": int(shapely.get_num_coordinates(registered)),
            "coordinates_identical": exact,
            "match": rel <= tol["area_rel"] and iou >= tol["iou_min"]
            and haus <= tol["hausdorff_m"]}


def lines(rebuilt: gpd.GeoDataFrame, registered: gpd.GeoDataFrame, work_crs: str,
          tol: dict[str, float]) -> dict[str, Any]:
    """Per OSM way (``osm_way``): length difference and Hausdorff distance."""
    a = rebuilt.set_index("osm_way").to_crs(work_crs)
    b = registered.set_index("osm_way").to_crs(work_crs)
    a_raw = rebuilt.set_index("osm_way")
    b_raw = registered.set_index("osm_way")
    ways = []
    for way in sorted(set(a.index) | set(b.index), key=int):
        if way not in a.index or way not in b.index:
            ways.append({"osm_way": int(way), "only_in": "rebuilt" if way in a.index else
                         "registered", "match": False})
            continue
        ga, gb = a.geometry[way], b.geometry[way]
        rel = abs(ga.length - gb.length) / gb.length
        haus = float(shapely.hausdorff_distance(ga, gb))
        ways.append({"osm_way": int(way), "name_equal": a_raw["name"][way] == b_raw["name"][way],
                     "length_m_rebuilt": round(ga.length, 3), "length_m_registered": round(gb.length, 3),
                     "length_rel_diff": rel, "hausdorff_m": haus,
                     "coordinates_identical": bool(shapely.equals_exact(
                         a_raw.geometry[way], b_raw.geometry[way], tolerance=0.0)),
                     "match": rel <= tol["length_rel"] and haus <= tol["hausdorff_m"]
                     and a_raw["name"][way] == b_raw["name"][way]})
    total_a, total_b = float(a.length.sum()), float(b.length.sum())
    return {"ways": ways, "total_length_km_rebuilt": round(total_a / 1e3, 4),
            "total_length_km_registered": round(total_b / 1e3, 4),
            "max_hausdorff_m": max((w.get("hausdorff_m", float("inf")) for w in ways), default=0.0),
            "match": all(w["match"] for w in ways)}


def points(rebuilt: gpd.GeoDataFrame, registered: gpd.GeoDataFrame, work_crs: str,
           tol: dict[str, float]) -> dict[str, Any]:
    """Points matched by ``name``: maximum displacement and name-set equality."""
    a = rebuilt.set_index("name").to_crs(work_crs)
    b = registered.set_index("name").to_crs(work_crs)
    common = [n for n in b.index if n in a.index]
    shift = max((float(a.geometry[n].distance(b.geometry[n])) for n in common), default=0.0)
    same = list(a.index) == list(b.index)
    return {"n_rebuilt": len(a), "n_registered": len(b), "names_and_order_equal": same,
            "max_shift_m": shift, "match": same and shift <= tol["hausdorff_m"]}


def overpass(rebuilt: dict[str, Any], registered: dict[str, Any]) -> dict[str, Any]:
    """Two Overpass-style ``out geom`` documents: element sets, tags and node coordinates."""
    a = {e["id"]: e for e in rebuilt["elements"]}
    b = {e["id"]: e for e in registered["elements"]}
    common = sorted(set(a) & set(b))
    tags = [i for i in common if a[i].get("tags") != b[i].get("tags")]
    geometry = [i for i in common if a[i].get("geometry") != b[i].get("geometry")]
    only_rebuilt = sorted(set(a) - set(b))
    only_registered = sorted(set(b) - set(a))
    return {"n_rebuilt": len(a), "n_registered": len(b), "only_rebuilt": only_rebuilt,
            "only_registered": only_registered, "tags_differ": tags, "geometry_differs": geometry,
            "elements_identical": rebuilt["elements"] == registered["elements"],
            "osm_base": [rebuilt.get("osm3s", {}).get("timestamp_osm_base"),
                         registered.get("osm3s", {}).get("timestamp_osm_base")],
            "match": not (only_rebuilt or only_registered or tags or geometry)}
