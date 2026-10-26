"""Freeze the acquired inputs (plan 002, Step 7): manifest snapshot, licence register, coverage.

``nakkhu data freeze`` verifies the manifest, then writes into a ``runs/data/<run_id>/`` run:

- ``manifest_snapshot.json``: every manifest entry plus the manifest SHA-256;
- ``licences.csv``: id, licence, attribution, redistributable, note;
- ``coverage.json``: computed share of the analysis extent (imagery) or routing domain (DEMs)
  with valid pixels, Sentinel-2 SCL cloud fractions and footprint counts.

The files are then promoted to ``artifacts/data/``. No pixel data are written.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_geom
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from nakkhu.acquire import _manifest_path, geo, load_sources, provenance
from nakkhu.artifacts import promote
from nakkhu.config import Project
from nakkhu.manifest import load_manifest, sha256_file, verify_manifest
from nakkhu.runs import atomic_write_json, atomic_write_text, create_run

ARTIFACT_DIR = "data"
SCL_CLOUD = (8, 9, 10)  # cloud medium / high probability, thin cirrus
SCL_SHADOW = (3,)


class FreezeError(RuntimeError):
    """The manifest does not verify, so nothing is frozen."""


def _grid(geom: BaseGeometry, res: float) -> tuple[Any, int, int]:
    west, south, east, north = geom.bounds
    width = int(np.ceil((east - west) / res))
    height = int(np.ceil((north - south) / res))
    return from_origin(west, north, res, res), width, height


def valid_share(files: list[Path], target: BaseGeometry, res: float,
                valid: Any = None, edge_m: float = 3.0) -> dict[str, Any]:
    """Share of ``target`` (EPSG:32645) covered by valid pixels of any of ``files``.

    Validity is the dataset mask, or ``valid(array, nodata)`` on band 1 when given. Each file is
    read decimated to about ``res`` and resampled (nearest) onto a common ``res`` grid. Grid
    cells on the target edge can miss data that ends exactly on the edge, so the share is also
    reported for the target eroded by ``edge_m`` (``share_interior``).
    """
    transform, width, height = _grid(target, res)
    inside = ~geometry_mask([mapping(target)], out_shape=(height, width), transform=transform)
    covered = np.zeros((height, width), dtype="uint8")
    for path in files:
        with rasterio.open(path) as src:
            factor = max(1, int(res / max(abs(src.res[0]), abs(src.res[1])) // 2))
            if src.crs.to_epsg() == 4326:
                factor = 1
            shape = (max(1, src.height // factor), max(1, src.width // factor))
            if valid is None:
                band = (src.dataset_mask(out_shape=shape, resampling=Resampling.nearest) > 0)
            else:
                data = src.read(1, out_shape=shape, resampling=Resampling.nearest)
                band = valid(data, src.nodata)
            src_transform = src.transform * src.transform.scale(src.width / shape[1],
                                                                src.height / shape[0])
            out = np.zeros((height, width), dtype="uint8")
            reproject(band.astype("uint8"), out, src_transform=src_transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=geo.WORK_CRS,
                      resampling=Resampling.nearest)
            covered |= out
    share = float((covered.astype(bool) & inside).sum() / inside.sum())
    core = ~geometry_mask([mapping(target.buffer(-edge_m))], out_shape=(height, width),
                          transform=transform)
    interior = float((covered.astype(bool) & core).sum() / core.sum())
    return {"share": round(share, 6), f"share_interior_{edge_m:g}m": round(interior, 6),
            "grid_m": res, "target_cells": int(inside.sum())}


def dem_holes(path: Path, target: BaseGeometry) -> dict[str, Any]:
    """Invalid (nodata / non-finite / < -1000) DEM cells inside ``target`` on the native grid."""
    with rasterio.open(path) as src:
        data = src.read(1)
        shape_geom = transform_geom(geo.WORK_CRS, src.crs, mapping(target))
        inside = ~geometry_mask([shape_geom], out_shape=data.shape, transform=src.transform)
        bad = ~np.isfinite(data) | (data < -1000)
        if src.nodata is not None:
            bad |= data == src.nodata
        return {"cells_in_target": int(inside.sum()), "invalid_cells": int((bad & inside).sum()),
                "share_valid": round(float(1 - (bad & inside).sum() / inside.sum()), 6)}


def scl_fractions(path: Path, target: BaseGeometry) -> dict[str, Any]:
    with rasterio.open(path) as src:
        scl = src.read(1)
        shape_geom = transform_geom(geo.WORK_CRS, src.crs, mapping(target))
        inside = ~geometry_mask([shape_geom], out_shape=scl.shape, transform=src.transform)
        values = scl[inside]
        observed = values[values != 0]
        return {
            "cells": int(inside.sum()),
            "no_data_share": round(float((values == 0).mean()), 6),
            "cloud_share": round(float(np.isin(observed, SCL_CLOUD).mean()), 6),
            "cloud_shadow_share": round(float(np.isin(observed, SCL_SHADOW).mean()), 6),
            "cloud_classes": list(SCL_CLOUD), "shadow_classes": list(SCL_SHADOW),
        }


def _products(project: Project, entry: dict[str, Any], kind: str) -> list[Path]:
    base = project.paths.resolve(entry["local_path"])
    return [base / p["path"] for p in provenance(project, entry).get("products", [])
            if p.get("kind") == kind]


def coverage(project: Project, entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {e["id"]: e for e in entries if e.get("status") != "pending"}
    extent = geo.extent(project)
    routing = geo.routing(project)
    training = geo.training(project)
    result: dict[str, Any] = {
        "aoi_km2": {"analysis_extent": round(extent.area / 1e6, 2),
                    "routing_domain": round(routing.area / 1e6, 2),
                    "training_domain": round(training.area / 1e6, 2)},
        "imagery_valid_share_of_extent": {},
        "dem_valid_share_of_routing_domain": {},
        "sentinel2_scl": {},
        "footprints": {},
    }
    imagery = result["imagery_valid_share_of_extent"]
    maxar = {sid: _products(project, e, "maxar_visual") for sid, e in by_id.items()
             if sid.startswith("maxar_") and not sid.endswith("_datamask")}
    for sid, files in sorted(maxar.items()):
        vrt_dir = files[0].parent
        tifs = sorted((vrt_dir / "visual").glob("*.tif"))
        imagery[sid] = valid_share(tifs, extent, 2.0)
    ge01 = [p for sid in ("maxar_ge01_20241006_0600", "maxar_ge01_20241006_0500")
            for p in sorted((project.paths.resolve(by_id[sid]["local_path"]) / "visual")
                            .glob("*.tif"))]
    imagery["maxar_ge01_20241006_combined"] = valid_share(ge01, extent, 2.0)
    for sid in sorted(s for s in by_id if s.startswith("oam_")):
        imagery[sid] = valid_share(_products(project, by_id[sid], "oam"), extent, 1.0)

    dems = result["dem_valid_share_of_routing_domain"]
    for sid in sorted(s for s in by_id if s.startswith(("dem_", "hand_"))):
        for path in _products(project, by_id[sid], "dem"):
            dems[sid] = dem_holes(path, routing)

    if "s2_l2a_45RUL" in by_id:
        for path in _products(project, by_id["s2_l2a_45RUL"], "s2"):
            if path.stem != "SCL":
                continue
            date = path.parent.name
            result["sentinel2_scl"][date] = {"extent": scl_fractions(path, extent),
                                             "routing_domain": scl_fractions(path, routing)}

    for sid in sorted(s for s in by_id if s.startswith("fp_")):
        for path in _products(project, by_id[sid], "footprints"):
            frame = gpd.read_parquet(path)
            centroids = frame.geometry.centroid
            result["footprints"][sid] = {
                "training_domain_intersecting": len(frame),
                "extent_centroid_within": int(centroids.within(extent).sum()),
                "with_confidence": int(frame["confidence"].notna().sum()),
            }
    ms = result["footprints"].get("fp_microsoft")
    if ms:
        ms["research_value_2026_02_03"] = 12965
        ms["note"] = ("Recount of the 2026-02-03 release file of quadkey 123131223 with the same "
                      "rule gives 12950 as well (identical in both releases), so the -15 "
                      "difference to the research value is not a release effect; the research "
                      "counting script was not kept and cannot be rerun.")
    return result


def licences(project: Project, entries: list[dict[str, Any]]) -> str:
    sources = load_sources(project)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["id", "licence", "attribution", "redistributable", "status"])
    for entry in sorted(entries, key=lambda e: e["id"]):
        redistributable = sources.get(entry["id"], {}).get("redistributable", "UNKNOWN")
        writer.writerow([entry["id"], entry["licence"], entry["attribution"], redistributable,
                         entry.get("status", "ok")])
    return out.getvalue()


def freeze(project: Project) -> list[dict[str, Any]]:
    manifest = _manifest_path(project)
    results = verify_manifest(manifest, project.paths)
    failed = [r.id for r in results if not r.ok]
    if failed:
        raise FreezeError(f"manifest does not verify: {failed}")
    entries = load_manifest(manifest)
    run = create_run(project, "data")
    with run.step("snapshot"):
        atomic_write_json(run.path / "manifest_snapshot.json", {
            "manifest": project.paths.relative(manifest),
            "manifest_sha256": sha256_file(manifest),
            "entries": entries,
        })
    with run.step("licences"):
        atomic_write_text(run.path / "licences.csv", licences(project, entries))
    with run.step("coverage"):
        atomic_write_json(run.path / "coverage.json", coverage(project, entries))
    run.write_summary({
        "entries": len(entries),
        "pending": sorted(e["id"] for e in entries if e.get("status") == "pending"),
        "bytes": sum(e.get("bytes") or 0 for e in entries),
    })
    promoted = []
    for name in ("manifest_snapshot.json", "licences.csv", "coverage.json"):
        promoted.append(promote(f"{run.ref}/{name}", f"{ARTIFACT_DIR}/{name}", project.paths,
                                replace=True))
    print(json.dumps(json.loads((run.path / "coverage.json").read_text()), indent=1)[:6000])
    return promoted
