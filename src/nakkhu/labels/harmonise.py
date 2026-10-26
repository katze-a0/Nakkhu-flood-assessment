"""Step 1: load and harmonise the two label sources (Google Open Buildings v3, Microsoft).

Multipolygons are exploded into parts, invalid geometries are repaired with
``shapely.make_valid`` (kept only if the repair is polygonal) and empty parts are dropped. Each
Google polygon gets the per-S2-level-4-cell confidence thresholds for 80/85/90 % precision from
Google's ``score_thresholds_s2_level_4.csv`` (looked up by the polygon's published lat/lon). If a
polygon's cell has no thresholds, the fixed cut-offs of ``labels.fallback_thresholds`` are used
and the row is marked ``threshold_source = "fallback"``.

Only the ``fp_google_v3`` and ``fp_microsoft`` manifest entries are read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from nakkhu.config import Project
from nakkhu.manifest import load_manifest

# The only manifest entries this package may read.
LABEL_SOURCES = {"google": "fp_google_v3", "microsoft": "fp_microsoft"}
THRESHOLD_FILE = "score_thresholds_s2_level_4.csv"
THRESHOLD_COLUMNS = {
    "t80": "confidence_threshold_80%_precision",
    "t85": "confidence_threshold_85%_precision",
    "t90": "confidence_threshold_90%_precision",
}


class HarmoniseError(ValueError):
    """A label source is missing or malformed."""


@dataclass(frozen=True)
class Footprints:
    """Harmonised label sources in the working CRS, one row per polygon part."""

    google: gpd.GeoDataFrame
    microsoft: gpd.GeoDataFrame


def source_path(project: Project, key: str) -> Path:
    """Local GeoParquet of a label source (``google`` or ``microsoft``) from the data manifest."""
    if key not in LABEL_SOURCES:
        raise HarmoniseError(f"{key!r} is not a label source; allowed: {sorted(LABEL_SOURCES)}")
    entry_id = LABEL_SOURCES[key]
    manifest = (project.config.values.get("paths") or {}).get("manifest",
                                                               "configs/data/manifest.yaml")
    entries = {e["id"]: e for e in load_manifest(project.paths.resolve(manifest))}
    entry = entries.get(entry_id)
    if entry is None or not entry.get("local_path"):
        raise HarmoniseError(f"manifest entry {entry_id} is missing or not acquired")
    path = project.paths.resolve(entry["local_path"]) / f"{entry_id}.parquet"
    if not path.is_file():
        raise HarmoniseError(f"{path} not found")
    return path


def explode_clean(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Explode multipolygons, repair invalid geometries, drop empty or non-polygonal parts."""
    stats = {"input": len(frame)}
    frame = gpd.GeoDataFrame(frame.reset_index(drop=True).copy(), geometry="geometry",
                             crs=frame.crs)
    frame["src_index"] = np.arange(len(frame))
    geoms = frame.geometry.values
    invalid = ~shapely.is_valid(geoms)
    stats["invalid_repaired"] = int(invalid.sum())
    if invalid.any():
        repaired = shapely.make_valid(geoms[invalid])
        # Keep only the polygonal part of a repair (make_valid can return collections).
        repaired = np.array([_polygonal(g) for g in repaired], dtype=object)
        frame.loc[invalid, "geometry"] = gpd.GeoSeries(repaired, index=frame.index[invalid],
                                                       crs=frame.crs)
    exploded = frame.explode(index_parts=False, ignore_index=True)
    stats["parts_after_explode"] = len(exploded)
    geoms = exploded.geometry.values
    keep = (~shapely.is_empty(geoms)) & (shapely.get_type_id(geoms) == 3)
    keep &= shapely.area(geoms) > 0
    stats["dropped_empty_or_nonpolygon"] = int((~keep).sum())
    out = exploded.loc[keep].reset_index(drop=True)
    stats["output"] = len(out)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frame.crs), stats


def _polygonal(geom: Any) -> Any:
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def load_thresholds(path: str | Path) -> gpd.GeoDataFrame:
    """Google's per-S2-cell confidence thresholds (lon/lat polygons, EPSG:4326)."""
    table = pd.read_csv(path)
    missing = [c for c in THRESHOLD_COLUMNS.values() if c not in table.columns]
    if missing or "geometry" not in table.columns:
        raise HarmoniseError(f"{path}: missing columns {missing or ['geometry']}")
    frame = gpd.GeoDataFrame(
        table.rename(columns={v: k for k, v in THRESHOLD_COLUMNS.items()}),
        geometry=gpd.GeoSeries.from_wkt(table["geometry"]), crs="EPSG:4326",
    )
    return frame[["s2_token", "t80", "t85", "t90", "geometry"]]


def attach_thresholds(google: pd.DataFrame, thresholds: gpd.GeoDataFrame,
                      fallback: dict[str, float]) -> pd.DataFrame:
    """Add ``s2_token, t80, t85, t90, threshold_source`` to each Google polygon."""
    points = gpd.GeoDataFrame(
        {"_row": np.arange(len(google))},
        geometry=gpd.points_from_xy(google["longitude"], google["latitude"]), crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, thresholds, how="left", predicate="within")
    joined = joined.sort_values("_row").drop_duplicates("_row", keep="first")
    out = google.copy()
    for key in ("s2_token", "t80", "t85", "t90"):
        out[key] = joined[key].to_numpy()
    missing = out["t90"].isna().to_numpy()
    out["threshold_source"] = np.where(missing, "fallback", "s2_cell")
    for key in ("t80", "t85", "t90"):
        if key in fallback:
            out.loc[missing, key] = float(fallback[key])
    return out


def load_footprints(project: Project, labels_cfg: dict[str, Any],
                    clip: Any | None = None) -> tuple[Footprints, dict[str, Any]]:
    """Load both label sources, optionally keep only polygons intersecting ``clip``."""
    stats: dict[str, Any] = {}
    frames = {}
    for key in LABEL_SOURCES:
        path = source_path(project, key)
        frame = gpd.read_parquet(path)
        if clip is not None:
            frame = frame.loc[shapely.intersects(frame.geometry.values, clip)]
        frame, part_stats = explode_clean(frame)
        stats[key] = part_stats
        frames[key] = frame
    google = frames["google"]
    thr_path = source_path(project, "google").parent / THRESHOLD_FILE
    google = attach_thresholds(google, load_thresholds(thr_path),
                               labels_cfg.get("fallback_thresholds", {}))
    stats["google"]["threshold_source"] = google["threshold_source"].value_counts().to_dict()
    stats["google"]["s2_tokens"] = sorted(google["s2_token"].dropna().unique().tolist())
    keep = ["source_id", "confidence", "s2_token", "t80", "t85", "t90", "threshold_source",
            "geometry"]
    google = gpd.GeoDataFrame(google[keep].reset_index(drop=True), geometry="geometry",
                              crs=google.crs)
    microsoft = frames["microsoft"]
    microsoft = gpd.GeoDataFrame(
        microsoft[["source_id", "confidence", "geometry"]].reset_index(drop=True),
        geometry="geometry", crs=microsoft.crs)
    return Footprints(google=google, microsoft=microsoft), stats
