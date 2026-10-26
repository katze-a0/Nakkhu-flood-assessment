"""OSM greenhouse mask: greenhouses are not buildings (user decision, 2026-10-05).

Shared helper (plan 008 exposure counts; reusable by the plan 004 label code):

- :func:`extract_greenhouses` reads OSM areas (closed ways and multipolygon relations) whose tags
  match :data:`GREENHOUSE_TAGS` from a ``.osm.pbf`` inside a lon/lat bbox.
- :func:`load_greenhouses` caches that extract under ``{data_root}/interim/reference/`` keyed by
  the PBF's SHA-256 prefix and the tag list.
- :func:`greenhouse_overlap` flags footprints that overlap a greenhouse polygon: IoU >= ``iou_min``
  (default 0.3) with any greenhouse, or the footprint centroid inside one.

Tags (OSM wiki): ``building=greenhouse`` (greenhouse building), ``building=glasshouse``
(non-standard synonym), ``landuse=greenhouse_horticulture`` (area under greenhouses; the
non-standard ``landuse=greenhouse`` too), and the
lifecycle forms ``disused:building=greenhouse`` / ``abandoned:building=greenhouse``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

GREENHOUSE_TAGS: dict[str, tuple[str, ...]] = {
    "building": ("greenhouse", "glasshouse"),
    "landuse": ("greenhouse_horticulture", "greenhouse"),
    "disused:building": ("greenhouse",),
    "abandoned:building": ("greenhouse",),
}
DEFAULT_IOU = 0.3


def tag_pairs(tags: Mapping[str, Sequence[str]] = GREENHOUSE_TAGS) -> list[tuple[str, str]]:
    return [(k, v) for k, values in tags.items() for v in values]


def matching_tag(tags: Any, wanted: Mapping[str, Sequence[str]] = GREENHOUSE_TAGS) -> str | None:
    """``"key=value"`` of the first greenhouse tag in an OSM tag list, else None."""
    for key, values in wanted.items():
        value = tags.get(key)
        if value is not None and value in values:
            return f"{key}={value}"
    return None


def extract_greenhouses(pbf: str | Path, bbox_lonlat: tuple[float, float, float, float],
                        tags: Mapping[str, Sequence[str]] = GREENHOUSE_TAGS) -> gpd.GeoDataFrame:
    """Greenhouse areas (EPSG:4326) whose first outer-ring node lies in ``bbox_lonlat``."""
    import osmium
    import osmium.filter
    import osmium.geom
    import osmium.osm

    west, south, east, north = bbox_lonlat
    wkb = osmium.geom.WKBFactory()
    pairs = tag_pairs(tags)
    processor = (osmium.FileProcessor(str(pbf))
                 .with_areas(osmium.filter.TagFilter(*pairs))
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
                 .with_filter(osmium.filter.TagFilter(*pairs)))
    rows = []
    for obj in processor:
        area: Any = obj
        first = next(iter(next(iter(area.outer_rings()), [])), None)
        if first is None or not (south <= first.lat <= north and west <= first.lon <= east):
            continue
        try:
            geom = shapely.from_wkb(bytes.fromhex(wkb.create_multipolygon(area)))
        except RuntimeError:
            continue
        kind = "w" if area.from_way() else "r"
        rows.append({"source_id": f"{kind}{area.orig_id()}", "tag": matching_tag(area.tags, tags),
                     "geometry": geom})
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else \
        gpd.GeoDataFrame({"source_id": [], "tag": []}, geometry=[], crs="EPSG:4326")
    frame["geometry"] = shapely.make_valid(frame.geometry.values)
    return frame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_greenhouses(pbf: str | Path, bbox_lonlat: tuple[float, float, float, float],
                     cache_dir: str | Path,
                     tags: Mapping[str, Sequence[str]] = GREENHOUSE_TAGS) -> gpd.GeoDataFrame:
    """:func:`extract_greenhouses`, cached as GeoParquet per PBF digest, tags and bbox."""
    pbf = Path(pbf)
    key = hashlib.sha256(repr((_file_sha256(pbf), sorted(tag_pairs(tags)),
                               tuple(round(b, 6) for b in bbox_lonlat))).encode()).hexdigest()
    cache = Path(cache_dir) / f"osm_greenhouses_{key[:12]}.parquet"
    if cache.is_file():
        return gpd.read_parquet(cache)
    frame = extract_greenhouses(pbf, bbox_lonlat, tags)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(cache)
    return frame


def greenhouse_overlap(footprints: gpd.GeoSeries, greenhouses: gpd.GeoSeries,
                       iou_min: float = DEFAULT_IOU) -> pd.DataFrame:
    """Per footprint: ``max_iou`` with any greenhouse, ``centroid_inside`` and ``is_greenhouse``
    (IoU >= ``iou_min`` or centroid inside). Both inputs must share a projected CRS."""
    n = len(footprints)
    max_iou = np.zeros(n)
    centroid_inside = np.zeros(n, dtype=bool)
    if n and len(greenhouses):
        fp = np.asarray(footprints.values)
        gh = np.asarray(greenhouses.values)
        tree = shapely.STRtree(gh)
        i, j = tree.query(fp, predicate="intersects")
        if i.size:
            inter = shapely.area(shapely.intersection(fp[i], gh[j]))
            union = shapely.area(fp[i]) + shapely.area(gh[j]) - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            np.maximum.at(max_iou, i, iou)
        ci, _ = tree.query(shapely.centroid(fp), predicate="within")
        centroid_inside[ci] = True
    return pd.DataFrame({"max_iou": max_iou, "centroid_inside": centroid_inside,
                         "is_greenhouse": (max_iou >= iou_min) | centroid_inside})
