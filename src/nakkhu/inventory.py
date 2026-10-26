"""D3: off-the-shelf footprint inventories as fixed (scored-as-is) runs (plan 007 Phase C).

``nakkhu detector inventory`` writes one ``runs/infer/<id>/`` per inventory with
``detections.gpkg`` (layer ``obb``: oriented envelope of each footprint, ``conf``) and
``inputs.json`` (``inventory: true``, ``model: inventory:<name>``), clipped to the val and test
regions (+ margin). Scoring is ``nakkhu eval --run infer/<id> --split val --fixed`` (test only
behind the prereg guard).

Conversion as in plan 004: Google v3 and Microsoft polygons are moved by the plan 004
registration shift of the train-grid tile holding their centroid (``registration_shifts.csv``;
same convention as ``labelqa.aligned_osm``), greenhouses are removed (shared rule), each polygon
becomes its minimum rotated rectangle, and rectangles under ``obb.min_area_m2`` (9 m2, the
label and detection filter) are dropped. ``*_unreg`` variants skip the shift (the product as a
user downloads it). Overture is descriptive only (prereg): scored unregistered, never a
reference. Inventories:

- ``google_t90`` (>= the per-cell 90 %-precision threshold; primary Google baseline),
  ``google_t80``, ``google_ge065`` (all of v3), ``google_ge075``; conf = Google confidence;
- ``microsoft`` (no score: conf 1); ``overture_unreg`` (no score: conf 1).
"""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from nakkhu.config import Project, ResolvedConfig

WORK_CRS = "EPSG:32645"


def _labels_cfg(project: Project) -> dict[str, Any]:
    from nakkhu.labels_cli import _arm_cfg

    return _arm_cfg(project, "L1")


def regions_clip(project: Project, margin_m: float = 100.0) -> Any:
    from nakkhu.evaluate.score import region_and_tiles

    parts = []
    for split in ("val", "test"):
        region, _, _ = region_and_tiles(project, split)
        parts.append(shapely.buffer(region, margin_m))
    return shapely.union_all(parts)


def shift_by_centroid_tile(geoms: np.ndarray, index: pd.DataFrame,
                           shifts: pd.DataFrame) -> np.ndarray:
    from nakkhu.labelqa import shift_map

    s = shifts.set_index("tile_id")
    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    ci, ti = shapely.STRtree(boxes).query(shapely.centroid(geoms), predicate="within")
    tile_of = np.full(len(geoms), -1)
    tile_of[ci[::-1]] = ti[::-1]
    out = np.asarray(geoms, dtype=object).copy()
    for t in np.unique(tile_of[tile_of >= 0]):
        row = index.iloc[int(t)]
        if row["tile_id"] not in s.index:
            continue
        sel = tile_of == t
        out[sel] = shift_map(out[sel], row["affine"], float(s.loc[row["tile_id"], "dx"]),
                             float(s.loc[row["tile_id"], "dy"]))
    return out


def to_obb(geoms: np.ndarray, min_area_m2: float) -> tuple[np.ndarray, np.ndarray]:
    env = shapely.oriented_envelope(np.asarray(geoms, dtype=object))
    keep = shapely.area(env) >= float(min_area_m2)
    return env, keep


def build_inventories(project: Project) -> list[dict[str, Any]]:
    from nakkhu.labels.build import greenhouse_areas
    from nakkhu.labels.harmonise import load_footprints
    from nakkhu.reference.greenhouse import greenhouse_overlap
    from nakkhu.runs import atomic_write_json, create_run
    from nakkhu.tiling import read_index

    cfg = _labels_cfg(project)
    clip = regions_clip(project)
    fps, fp_stats = load_footprints(project, cfg, clip=clip)
    index, _ = read_index(project.paths.data_root / "interim" / "tiles" /
                          "maxar_wv02_20240113" / "train" / "index.parquet")
    shifts = pd.read_csv(project.paths.artifacts / "labels" / "registration_shifts.csv")
    green = greenhouse_areas(project, cfg["greenhouse_filter"])
    min_area = float(cfg["obb"]["min_area_m2"])
    iou_min = float(cfg["greenhouse_filter"]["iou_min"])

    google = fps.google.to_crs(WORK_CRS)
    micro = fps.microsoft.to_crs(WORK_CRS)
    from nakkhu.manifest import load_manifest

    manifest = {e["id"]: e for e in load_manifest(project.paths.resolve(
        "configs/data/manifest.yaml"))}
    ov_path = project.paths.resolve(manifest["fp_overture"]["local_path"]) / \
        "fp_overture.parquet"
    over = gpd.read_parquet(ov_path).to_crs(WORK_CRS)
    over = over.loc[shapely.intersects(over.geometry.values, clip)]
    over = over.explode(index_parts=False)
    over = over.loc[over.geometry.geom_type == "Polygon"]

    conf_g = google["confidence"].to_numpy(dtype=float)
    t90 = google["t90"].to_numpy(dtype=float)
    t80 = google["t80"].to_numpy(dtype=float) if "t80" in google else None
    specs: list[tuple[str, gpd.GeoDataFrame, np.ndarray, np.ndarray, bool, str]] = []
    gmask = {"google_t90": conf_g >= t90, "google_ge065": conf_g >= 0.65,
             "google_ge075": conf_g >= 0.75}
    if t80 is not None:
        gmask["google_t80"] = conf_g >= t80
    for name, m in gmask.items():
        specs.append((name, google, m, conf_g, True, "Google Open Buildings v3"))
        if name == "google_t90":
            specs.append(("google_t90_unreg", google, m, conf_g, False,
                          "Google Open Buildings v3"))
    specs.append(("microsoft", micro, np.ones(len(micro), bool), np.ones(len(micro)), True,
                  "Microsoft GlobalMLBuildingFootprints"))
    specs.append(("microsoft_unreg", micro, np.ones(len(micro), bool), np.ones(len(micro)),
                  False, "Microsoft GlobalMLBuildingFootprints"))
    specs.append(("overture_unreg", over, np.ones(len(over), bool), np.ones(len(over)), False,
                  "Overture buildings (descriptive; merge of OSM/Google/Microsoft here)"))

    gh_cache: dict[str, np.ndarray] = {}
    out = []
    for name, frame, mask, conf, registered, source in specs:
        key = source
        if key not in gh_cache:
            gh_cache[key] = greenhouse_overlap(frame.geometry, green.geometry,
                                               iou_min)["is_greenhouse"].to_numpy()
        sel = mask & ~gh_cache[key]
        geoms = frame.geometry.values[sel]
        if registered:
            geoms = shift_by_centroid_tile(geoms, index, shifts)
        env, keep = to_obb(geoms, min_area)
        values = {"command": "detector inventory", "inventory": name, "source": source,
                  "registered": registered, "min_area_m2": min_area,
                  "greenhouse_iou_min": iou_min}
        run = create_run(project, "infer", ResolvedConfig(values=values,
                                                          source="nakkhu.inventory"))
        with run.step("inventory"):
            gdf = gpd.GeoDataFrame({"conf": np.asarray(conf)[sel][keep],
                                    "cls": np.zeros(int(keep.sum()), dtype="int64")},
                                   geometry=list(env[keep]), crs=WORK_CRS)
            from nakkhu.georef import write_detections

            write_detections(run.path / "detections.gpkg", gdf, run.ref)
            inputs = {"inventory": True, "model": f"inventory:{name}", "source": source,
                      "registered": registered, "n_input": int(mask.sum()),
                      "greenhouses_removed": int((mask & gh_cache[key]).sum()),
                      "below_min_area": int((~keep).sum()), "n_obb": int(keep.sum()),
                      "clip": "val and test regions + 100 m", "footprint_stats": fp_stats}
            atomic_write_json(run.path / "inputs.json", inputs)
        run.write_summary({k: v for k, v in inputs.items() if k != "footprint_stats"})
        out.append({"run": run.ref, "inventory": name, "n_obb": inputs["n_obb"]})
        print(json.dumps(out[-1]), flush=True)
    return out
