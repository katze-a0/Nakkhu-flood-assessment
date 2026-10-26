"""Sampling frames of the audit tasks (plan 006 Step 1). All frames lie in the test region.

``recall_points``: units = representative points (centroid, or a point on the surface when the
centroid falls outside) of every Google v3, Microsoft and OSM footprint, each shifted to the
imagery by its plan 004 per-tile registration (Google/Microsoft: joint footprint shift; OSM: OSM
shift; 24 px cap), kept when inside the region, and de-duplicated across sources within
``dedup_m`` (greedy in source order; points of the same source are never merged). Footprints
overlapping an OSM greenhouse area are dropped first (greenhouses are not buildings; the shared
rule of :mod:`nakkhu.reference.greenhouse` with the ``greenhouse_filter`` settings of
``configs/labels/common.yaml``, as for labels, the OSM reference and exposure counts). Overture
is never a source here (prereg ``audit.overture_rule``: it is a merge of the other three). A
unit's membership pattern lists the sources with a polygon within ``membership_tol_m`` of it;
the pattern is the stratum. A separate ``outside`` stratum holds uniform random points in the
region farther than ``outside_buffer_m`` from every footprint (greenhouse footprints included)
and outside every greenhouse area (its size is an area, not a count).

``precision_pool``: every detection of every listed run (confidence >= the run's frozen
threshold, centroid in the region), single-linkage clustered across runs at rotated IoU >=
``cluster_iou``; the cluster medoid is shown. Strata: share of runs containing the cluster x
distance-to-channel band.

``flood_trace``: observable buildings of the plan 009 change map in the region; strata:
positive, negative within ``near_channel_m`` of the channel, negative farther.

``precision_microsoft`` (plan 013, prereg v8, sitting 2): the Microsoft exposure units, i.e. the
polygons :func:`nakkhu.hydro.sweep.load_inventory` gives for ``microsoft`` (centroid in the
extent, greenhouses removed; the ids and order of its HAND table and D_t), each shown moved by
the joint footprint registration shift. Strata: ``agree`` (a detector_final box centroid lies
inside the moved polygon) or ``ms_only``, crossed with the channel-distance bands of the moved
polygon centroid. The unmoved centroid (``hand_x``, ``hand_y``) is the unit's HAND point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from nakkhu.acquire import geo
from nakkhu.audit.core import AuditError
from nakkhu.config import Project
from nakkhu.manifest import load_manifest

SOURCE_IDS = {"google": "fp_google_v3", "microsoft": "fp_microsoft", "osm": "fp_osm",
              "overture": "fp_overture"}
SOURCE_LETTER = {"google": "G", "microsoft": "M", "osm": "O", "overture": "V"}
OVERTURE_SHIFT = {"OpenStreetMap": "osm", "Google Open Buildings": "footprints",
                  "Microsoft ML Buildings": "footprints"}
OUTSIDE = "outside"
TRAIN_TILE_INDEX = "interim/tiles/maxar_wv02_20240113/train/index.parquet"   # under data_root
GOOGLE_T90_COLUMN = "confidence_threshold_90%_precision"


def region(project: Project, cfg: dict[str, Any]) -> Any:
    key = str(cfg.get("region", "analysis_extent"))
    return geo.read_geometry(geo.aoi_path(project, key))


def manifest_entries(project: Project) -> dict[str, dict[str, Any]]:
    manifest = (project.config.values.get("paths") or {}).get("manifest",
                                                               "configs/data/manifest.yaml")
    return {e["id"]: e for e in load_manifest(project.paths.resolve(manifest))}


def source_file(project: Project, source_id: str) -> Path:
    entry = manifest_entries(project).get(source_id)
    if entry is None or entry.get("status") == "pending":
        raise AuditError(f"{source_id} is missing or pending in the data manifest")
    return project.paths.resolve(entry["local_path"]) / f"{source_id}.parquet"


def google_t90(project: Project) -> float:
    """t90 of the S2 cell the Google file was cut from (plan 004 Step 1)."""
    entry = manifest_entries(project)[SOURCE_IDS["google"]]
    token = str((entry.get("params") or {}).get("token", "39f"))
    path = project.paths.resolve(entry["local_path"]) / "score_thresholds_s2_level_4.csv"
    table = pd.read_csv(path, usecols=["s2_token", GOOGLE_T90_COLUMN])
    row = table.loc[table["s2_token"].astype(str) == token]
    if row.empty:
        raise AuditError(f"no Google v3 t90 for S2 cell {token}")
    return float(row[GOOGLE_T90_COLUMN].iloc[0])


def channel(project: Project, cfg: dict[str, Any]) -> Any:
    from nakkhu.splits import read_lines

    lines = read_lines([project.paths.resolve(p) for p in cfg.get("channel_lines") or []])
    if len(lines) == 0:
        raise AuditError("no channel lines configured (channel_lines)")
    return shapely.union_all(lines)


# --- registration -----------------------------------------------------------------------------


def tile_index(project: Project) -> pd.DataFrame:
    path = project.paths.data_root / TRAIN_TILE_INDEX
    if not path.is_file():
        raise AuditError(f"{path} is missing (plan 003 train tiles of the WV02 scene)")
    return pd.read_parquet(path).sort_values("tile_id").reset_index(drop=True)


def load_shifts(project: Project, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Promoted plan 004 per-tile shifts (tile pixels, x right / y down)."""
    reg = cfg.get("registration") or {}
    out = {}
    for kind in ("footprints", "osm"):
        path = project.paths.artifacts / str(reg[kind])
        if not path.is_file():
            raise AuditError(f"artifacts/{reg[kind]} is missing (plan 004 registration)")
        out[kind] = pd.read_csv(path).set_index("tile_id")[["dx", "dy", "reg_ok"]]
    return out


def apply_shifts(geoms: np.ndarray, index: pd.DataFrame, shifts: pd.DataFrame) -> np.ndarray:
    """Move each polygon by the shift of the train-grid tile containing its centroid."""
    geoms = np.asarray(geoms, dtype=object)
    if len(geoms) == 0:
        return geoms
    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    ci, ti = shapely.STRtree(boxes).query(shapely.centroid(geoms), predicate="within")
    tile_of = np.full(len(geoms), -1)
    tile_of[ci[::-1]] = ti[::-1]
    out = geoms.copy()
    for t in np.unique(tile_of[tile_of >= 0]):
        tid = index.iloc[int(t)]["tile_id"]
        if tid not in shifts.index:
            continue
        a, e = (float(v) for v in np.asarray(index.iloc[int(t)]["affine"])[[0, 4]])
        dx, dy = float(shifts.loc[tid, "dx"]), float(shifts.loc[tid, "dy"])
        if dx == 0 and dy == 0:
            continue
        sel = tile_of == t
        out[sel] = shapely.transform(out[sel], lambda xy, s=(dx * a, dy * e): xy + np.array(s))
    return out


GREENHOUSE_SETTINGS = "configs/labels/common.yaml"


def greenhouse_areas(project: Project) -> tuple[gpd.GeoDataFrame, float]:
    """OSM greenhouse areas (working CRS) and the overlap IoU, exactly as the label code and the
    OSM reference use them (``greenhouse_filter`` in configs/labels/common.yaml)."""
    from nakkhu.labels.arms import load_arm_config
    from nakkhu.labels.build import greenhouse_areas as shared_areas

    opts = load_arm_config(project.paths.resolve(GREENHOUSE_SETTINGS))["greenhouse_filter"]
    return shared_areas(project, opts), float(opts["iou_min"])


def drop_greenhouses(polys: dict[str, gpd.GeoDataFrame], areas: gpd.GeoDataFrame,
                     iou_min: float) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, int]]:
    """Remove footprints overlapping a greenhouse (IoU >= ``iou_min`` or centroid inside)."""
    from nakkhu.reference.greenhouse import greenhouse_overlap

    out, dropped = {}, {}
    for name, frame in polys.items():
        flags = greenhouse_overlap(frame.geometry, areas.geometry, iou_min)
        keep = ~flags["is_greenhouse"].to_numpy()
        out[name] = frame.loc[keep].reset_index(drop=True)
        dropped[name] = int((~keep).sum())
    return out, dropped


def representative_points(geoms: np.ndarray) -> np.ndarray:
    cent = shapely.centroid(geoms)
    inside = shapely.contains(geoms, cent)
    out = np.asarray(cent, dtype=object).copy()
    out[~inside] = shapely.point_on_surface(geoms[~inside])
    return out


# --- footprints -------------------------------------------------------------------------------


def load_sources(project: Project, cfg: dict[str, Any], clip: Any,
                 sources: list[str]) -> dict[str, gpd.GeoDataFrame]:
    """Registered polygons of each source intersecting ``clip`` (exploded to parts)."""
    index = tile_index(project)
    shifts = load_shifts(project, cfg)
    out = {}
    for name in sources:
        frame = gpd.read_parquet(source_file(project, SOURCE_IDS[name]))
        frame = frame.loc[shapely.intersects(frame.geometry.values, clip.buffer(50.0))]
        frame = frame.explode(index_parts=False, ignore_index=True)
        geoms = frame.geometry.values
        keep = (shapely.get_type_id(np.asarray(geoms)) == 3) & (shapely.area(geoms) > 0)
        frame = frame.loc[keep].reset_index(drop=True)
        geoms = np.asarray(frame.geometry.values, dtype=object)
        if name in ("google", "microsoft"):
            geoms = apply_shifts(geoms, index, shifts["footprints"])
        elif name == "osm":
            geoms = apply_shifts(geoms, index, shifts["osm"])
        else:
            kinds = frame["source_dataset"].map(OVERTURE_SHIFT).fillna("footprints").to_numpy()
            for kind in ("footprints", "osm"):
                sel = kinds == kind
                geoms[sel] = apply_shifts(geoms[sel], index, shifts[kind])
        frame = frame.set_geometry(gpd.GeoSeries(geoms, crs=frame.crs))
        out[name] = frame
    return out


def dedup_cross_source(points: np.ndarray, source: np.ndarray, distance: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Greedy de-duplication in array order: a kept point absorbs every point of *another*
    source within ``distance``. Returns (kept mask, owner index of every point)."""
    n = len(points)
    tree = shapely.STRtree(points)
    a, b = tree.query(points, predicate="dwithin", distance=distance)
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    cuts = np.searchsorted(a, np.arange(n + 1))
    owner = np.full(n, -1)
    kept = np.zeros(n, dtype=bool)
    for i in range(n):
        if owner[i] >= 0:
            continue
        kept[i] = True
        owner[i] = i
        nb = b[cuts[i]:cuts[i + 1]]
        nb = nb[(owner[nb] < 0) & (source[nb] != source[i])]
        owner[nb] = i
    return kept, owner


def recall_frame(project: Project, cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict]:
    """Units of the recall_points frame with membership flags and stratum (pattern)."""
    tcfg = cfg["tasks"]["recall_points"]
    names = list(tcfg["sources"])
    if "overture" in names:
        raise AuditError("Overture is never an audit source or stratum (prereg "
                         "audit.overture_rule); remove it from tasks.recall_points.sources")
    area = region(project, cfg)
    raw = load_sources(project, cfg, area, names)
    gh, gh_iou = greenhouse_areas(project)
    gh = gh.loc[shapely.intersects(gh.geometry.values, area.buffer(50.0))].reset_index(drop=True)
    polys, gh_dropped = drop_greenhouses(raw, gh, gh_iou)
    pts, src, conf = [], [], []
    for k, name in enumerate(names):
        geoms = np.asarray(polys[name].geometry.values, dtype=object)
        rep = representative_points(geoms)
        inside = shapely.within(rep, area)
        pts.append(rep[inside])
        src.append(np.full(int(inside.sum()), k))
        c = polys[name]["confidence"].to_numpy(dtype=float) if "confidence" in polys[name] \
            else np.full(len(geoms), np.nan)
        conf.append(c[inside])
    points = np.concatenate(pts)
    source = np.concatenate(src)
    kept, owner = dedup_cross_source(points, source, float(tcfg["dedup_m"]))
    units = np.flatnonzero(kept)
    remap = np.full(len(points), -1)
    remap[units] = np.arange(len(units))
    origin = np.zeros((len(units), len(names)), dtype=bool)
    origin[remap[owner], source] = True
    frame = gpd.GeoDataFrame({"unit": np.arange(len(units))},
                             geometry=gpd.GeoSeries(points[units], crs=geo.WORK_CRS))
    tol = float(tcfg["membership_tol_m"])
    t90 = google_t90(project)
    for k, name in enumerate(names):
        geoms = np.asarray(polys[name].geometry.values, dtype=object)
        ui, pi = shapely.STRtree(geoms).query(points[units], predicate="dwithin", distance=tol)
        member = np.zeros(len(units), dtype=bool)
        member[ui] = True
        frame[f"in_{name}"] = member
        frame[f"origin_{name}"] = origin[:, k]
        if name == "google":
            c = polys[name]["confidence"].to_numpy(dtype=float)
            best = pd.Series(c[pi]).groupby(ui).max()
            gmax = np.full(len(units), np.nan)
            gmax[best.index.to_numpy()] = best.to_numpy()
            frame["google_conf_max"] = gmax
            frame["in_google_t90"] = np.nan_to_num(gmax, nan=-1.0) >= t90
    frame["stratum"] = ["".join(SOURCE_LETTER[n] if frame[f"in_{n}"].iat[i] else "-"
                                for n in names) for i in range(len(frame))]
    frame["x"] = shapely.get_x(frame.geometry.values)
    frame["y"] = shapely.get_y(frame.geometry.values)
    allpoly = np.concatenate([np.asarray(raw[n].geometry.values, dtype=object)
                              for n in names])
    stats = {"points_per_source": {n: int((source == k).sum()) for k, n in enumerate(names)},
             "units": int(len(units)), "google_t90": t90,
             "region_area_m2": float(area.area),
             "greenhouse_areas_near_region": int(len(gh)),
             "greenhouse_footprints_dropped": gh_dropped}
    return frame, {"stats": stats, "all_polygons": allpoly, "region": area,
                   "greenhouses": np.asarray(gh.geometry.values, dtype=object)}


def outside_area(area: Any, polygons: np.ndarray, buffer_m: float,
                 exclude: np.ndarray | None = None) -> Any:
    near = polygons[shapely.intersects(polygons, area.buffer(buffer_m))]
    covered = shapely.union_all(shapely.buffer(near, buffer_m, quad_segs=4))
    out = shapely.difference(area, covered)
    if exclude is not None and len(exclude):
        out = shapely.difference(out, shapely.union_all(shapely.make_valid(exclude)))
    return out


def outside_points(outside: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` uniform random points in ``outside`` by rejection from its bounding box."""
    xmin, ymin, xmax, ymax = outside.bounds
    shapely.prepare(outside)
    got: list[Any] = []
    while len(got) < n:
        xy = rng.uniform([xmin, ymin], [xmax, ymax], size=(4 * n + 16, 2))
        cand = shapely.points(xy)
        got.extend(cand[shapely.contains_xy(outside, xy[:, 0], xy[:, 1])][: n - len(got)])
    return np.asarray(got, dtype=object)


# --- precision pool (plan 007 detections) -----------------------------------------------------


def _run_threshold(project: Project, spec: dict[str, Any]) -> float:
    if spec.get("threshold") is not None:
        return float(spec["threshold"])
    ref = spec.get("threshold_from")
    if not ref:
        raise AuditError(f"run {spec.get('run')}: give `threshold` or `threshold_from`")
    import json

    path = project.paths.runs / str(ref) / "threshold.json"
    if not path.is_file():
        raise AuditError(f"{path} is missing (val eval of the same model)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return float(doc["threshold"])


def load_detections(project: Project, cfg: dict[str, Any], area: Any) -> gpd.GeoDataFrame:
    specs = cfg["tasks"]["precision_pool"].get("runs") or []
    if not specs:
        raise AuditError("precision_pool has no detection runs yet (plan 007 Phase A/B); list "
                         "them under tasks.precision_pool.runs in configs/audit/audit.yaml")
    frames = []
    for k, spec in enumerate(specs):
        path = project.paths.runs / str(spec["run"]) / "detections.gpkg"
        if not path.is_file():
            raise AuditError(f"{path} is missing (run `nakkhu georef --run {spec['run']}`)")
        det = gpd.read_file(path, layer="obb")
        det = det.loc[det["conf"].to_numpy(dtype=float) >= _run_threshold(project, spec)]
        det = det.loc[shapely.within(shapely.centroid(det.geometry.values), area)]
        frames.append(pd.DataFrame({
            "run_k": k, "run": str(spec["run"]), "model": str(spec.get("model", spec["run"])),
            "det": det.index.to_numpy(), "conf": det["conf"].to_numpy(dtype=float),
            "geometry": det.geometry.values}))
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=geo.WORK_CRS)


def iou_pairs(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = a < b
    a, b = a[keep], b[keep]
    inter = shapely.area(shapely.intersection(geoms[a], geoms[b]))
    union = shapely.area(geoms[a]) + shapely.area(geoms[b]) - inter
    return a, b, np.where(union > 0, inter / union, 0.0)


def cluster_labels(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, j in zip(a.tolist(), b.tolist(), strict=True):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def band(values: np.ndarray, edges: list[float], prefix: str) -> np.ndarray:
    idx = np.clip(np.searchsorted(np.asarray(edges[1:-1], dtype=float), values, side="right"),
                  0, len(edges) - 2)
    return np.array([f"{prefix}{i}" for i in idx])


def precision_frame(project: Project, cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict]:
    tcfg = cfg["tasks"]["precision_pool"]
    area = region(project, cfg)
    det = load_detections(project, cfg, area)
    geoms = np.asarray(det.geometry.values, dtype=object)
    a, b, iou = iou_pairs(geoms)
    run_k = det["run_k"].to_numpy()
    edge = (iou >= float(tcfg["cluster_iou"])) & (run_k[a] != run_k[b])
    labels = cluster_labels(len(det), a[edge], b[edge])
    det["cluster"] = labels
    n_runs = len(tcfg["runs"])
    score = np.zeros(len(det))
    np.add.at(score, a[edge], iou[edge])
    np.add.at(score, b[edge], iou[edge])
    det["medoid_score"] = score
    rep = det.sort_values(["cluster", "medoid_score", "conf"], ascending=[True, False, False])
    rep = rep.drop_duplicates("cluster")
    share = det.groupby("cluster")["run_k"].nunique().reindex(rep["cluster"]).to_numpy() / n_runs
    rep_geoms = np.asarray(rep.geometry.values, dtype=object)
    cent = shapely.centroid(rep_geoms)
    dist = shapely.distance(cent, channel(project, cfg))
    frame = gpd.GeoDataFrame({
        "unit": rep["cluster"].to_numpy(), "run_share": share, "dist_channel_m": dist,
        "x": shapely.get_x(cent), "y": shapely.get_y(cent)},
        geometry=gpd.GeoSeries(rep_geoms, crs=geo.WORK_CRS))
    frame["stratum"] = [f"{s}_{c}" for s, c in zip(
        band(share, list(tcfg["run_share_bands"]), "share"),
        band(dist, list(tcfg["channel_bands_m"]), "chan"), strict=True)]
    members = det[["cluster", "run", "model", "det", "conf"]].copy()
    stats = {"detections": int(len(det)), "clusters": int(len(frame)), "runs": n_runs}
    return frame, {"stats": stats, "members": members}


# --- Microsoft polygon precision (plan 013) ---------------------------------------------------


def polygon_strata(polygons: np.ndarray, boxes: np.ndarray, channel_geom: Any,
                   bands_m: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(agree, distance to the channel, stratum) per polygon: ``agree`` if the centroid of at
    least one box lies inside the polygon; the band is that of the polygon centroid."""
    polygons = np.asarray(polygons, dtype=object)
    agree = np.zeros(len(polygons), dtype=bool)
    boxes = np.asarray(boxes, dtype=object)
    if len(polygons) and len(boxes):
        # (box centroid index, polygon index) pairs with the centroid within the polygon
        _, hit = shapely.STRtree(polygons).query(shapely.centroid(boxes), predicate="within")
        agree[np.unique(hit)] = True
    dist = shapely.distance(shapely.centroid(polygons), channel_geom) if len(polygons) \
        else np.zeros(0)
    stratum = np.array([f"{'agree' if a else 'ms_only'}_{b}" for a, b in
                        zip(agree, band(np.asarray(dist, dtype=float), bands_m, "chan"),
                            strict=True)], dtype=object)
    return agree, np.asarray(dist, dtype=float), stratum


def microsoft_frame(project: Project, cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict]:
    from nakkhu.hydro.sweep import load_greenhouses, load_inventory, load_settings

    tcfg = cfg["tasks"]["precision_microsoft"]
    settings, _ = load_settings(project)
    extent = settings.read_polygon(settings.v["analysis_extent"])
    gh = load_greenhouses(settings)
    polys, info = load_inventory(settings, str(tcfg.get("inventory", "microsoft")), extent, gh)
    det, det_info = load_inventory(settings, str(tcfg.get("agree_with", "detector_final")),
                                   extent, gh)
    polys = polys.to_crs(geo.WORK_CRS)
    geoms = np.asarray(polys.geometry.values, dtype=object)
    hand_pt = shapely.centroid(geoms)
    moved = apply_shifts(geoms, tile_index(project), load_shifts(project, cfg)["footprints"])
    boxes = np.asarray(det.to_crs(geo.WORK_CRS).geometry.values, dtype=object)
    agree, dist, stratum = polygon_strata(moved, boxes, channel(project, cfg),
                                          list(tcfg["channel_bands_m"]))
    cent = shapely.centroid(moved)
    frame = gpd.GeoDataFrame({
        "unit": np.arange(len(polys)), "building_uid": polys["building_uid"].to_numpy(),
        "agree": agree, "dist_channel_m": dist,
        "x": shapely.get_x(cent), "y": shapely.get_y(cent),
        "hand_x": shapely.get_x(hand_pt), "hand_y": shapely.get_y(hand_pt),
        "stratum": stratum}, geometry=gpd.GeoSeries(moved, crs=geo.WORK_CRS))
    stats = {"polygons": int(len(frame)), "agree": int(agree.sum()),
             "inventory": {k: v for k, v in info.items() if not isinstance(v, list)},
             "agree_with": str(tcfg.get("agree_with", "detector_final")),
             "agree_boxes": int(len(boxes))}
    return frame, {"stats": stats}


# --- flood trace (plan 009 change map) --------------------------------------------------------


def flood_path(project: Project, tcfg: dict[str, Any], key: str) -> Path:
    """A flood_trace path with ``{inventory}`` filled from ``tasks.flood_trace.inventory``."""
    return project.paths.resolve(str(tcfg[key]).replace("{inventory}",
                                                        str(tcfg.get("inventory", ""))))


def flood_hashes(project: Project, tcfg: dict[str, Any]) -> dict[str, Any]:
    """Inventory id and the change-map / inventory hashes of the plan 009 sidecar (the
    change-map file is hashed again here, so an edited table is caught too)."""
    out: dict[str, Any] = {"inventory": tcfg.get("inventory")}
    cmap = flood_path(project, tcfg, "change_map")
    out["change_map_file_sha256"] = sha256_path(cmap) if cmap.is_file() else None
    meta_key = tcfg.get("change_map_meta")
    meta = flood_path(project, tcfg, "change_map_meta") if meta_key else None
    doc = json.loads(meta.read_text(encoding="utf-8")) if meta and meta.is_file() else {}
    for key in ("change_map_sha256", "inventory_sha256", "buildings_sha256"):
        out[key] = doc.get(key)
    return out


def sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def flood_frame(project: Project, cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict]:
    tcfg = cfg["tasks"]["flood_trace"]
    path = flood_path(project, tcfg, "change_map")
    if not path.is_file():
        raise AuditError(f"{path} is missing (plan 009 change map)")
    cmap = pd.read_parquet(path)
    if "geometry" not in cmap.columns:
        raise AuditError(f"{path} has no geometry column (plan 009 writes the registered "
                         "building geometry per inventory)")
    build = gpd.read_parquet(path)
    for col in ("building_id", "class"):
        if col not in build.columns:
            raise AuditError(f"change map lacks column {col!r}")
    build = build.to_crs(geo.WORK_CRS)
    if "observable" in build.columns:
        build = build.loc[build["observable"].astype(bool)]
    area = region(project, cfg)
    cent = shapely.centroid(np.asarray(build.geometry.values, dtype=object))
    inside = shapely.within(cent, area)
    build, cent = build.loc[inside].reset_index(drop=True), cent[inside]
    dist = shapely.distance(cent, channel(project, cfg))
    positive = build["class"].astype(str).str.lower().isin(["positive", "1", "true", "changed"])
    near = dist <= float(tcfg["near_channel_m"])
    stratum = np.where(positive, "positive", np.where(near, "negative_near", "negative_far"))
    frame = gpd.GeoDataFrame({
        "unit": np.arange(len(build)), "building_id": build["building_id"].to_numpy(),
        "change_class": build["class"].astype(str).to_numpy(), "dist_channel_m": dist,
        "x": shapely.get_x(cent), "y": shapely.get_y(cent), "stratum": stratum},
        geometry=gpd.GeoSeries(np.asarray(build.geometry.values, dtype=object),
                               crs=geo.WORK_CRS))
    return frame, {"stats": {"buildings": int(len(frame))}}


# --- flood trace on the Phase-2 sites (plan 014 P2-audit) -------------------------------------
# Label-blind order: the frame reads the site change maps, so it may be built only after
# prereg_v9_a4 is frozen and the site change maps exist (G6). The change maps are named, with
# their sha256, in ``tasks.flood_trace_p2.change_maps`` (written after G6).


def p2_require_freeze(project: Project, tcfg: dict[str, Any]) -> None:
    """Refuse unless the addendum named in ``requires_frozen`` (prereg_v9_a4) is frozen."""
    from nakkhu.evaluate.addendum import frozen_addenda

    want = str(tcfg.get("requires_frozen") or "")
    if not want:
        raise AuditError("flood_trace_p2 needs `requires_frozen` (the Phase-2 freeze)")
    if want not in frozen_addenda(project):
        raise AuditError(f"{want} is not frozen: no Phase-2 audit item may be drawn")


def p2_change_maps(project: Project, tcfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per site the change-map file and its sha256 (checked) from ``change_maps``."""
    from nakkhu.config import load_yaml

    path = project.paths.resolve(str(tcfg["change_maps"]))
    if not path.is_file():
        raise AuditError(f"{tcfg['change_maps']} is missing: the site change maps (G6) come "
                         "first")
    doc = load_yaml(path) or {}
    out = {}
    for site in tcfg["sites"]:
        entry = (doc.get("sites") or {}).get(site)
        if not entry:
            raise AuditError(f"{tcfg['change_maps']}: no change map for site {site}")
        f = project.paths.resolve(str(entry["file"]))
        if not f.is_file() or sha256_path(f) != entry["sha256"]:
            raise AuditError(f"site {site}: change map {entry['file']} missing or not the "
                             "pinned sha256")
        out[str(site)] = {"file": f, "sha256": str(entry["sha256"])}
    return out


def p2_site_path(project: Project, tcfg: dict[str, Any], key: str, site: str) -> Path:
    return project.paths.resolve(str(tcfg[key]).replace("{site}", str(site)))


def p2_hashes(project: Project, tcfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cm = project.paths.resolve(str(tcfg["change_maps"]))
    out["change_maps_config"] = sha256_path(cm) if cm.is_file() else None
    for site in tcfg["sites"]:
        for key in ("site_lines", "site_extent"):
            p = p2_site_path(project, tcfg, key, site)
            out[f"{key}_{site}"] = sha256_path(p) if p.is_file() else None
    if cm.is_file():
        from nakkhu.config import load_yaml

        for site, entry in ((load_yaml(cm) or {}).get("sites") or {}).items():
            f = project.paths.resolve(str(entry["file"]))
            out[f"change_map_{site}"] = sha256_path(f) if f.is_file() else None
    return out


def p2_frame(project: Project, cfg: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict]:
    """Observable buildings of each site's change map inside its extent; strata per site
    (``<site>:positive``, ``<site>:negative_near`` within ``near_channel_m`` of the site's main
    lines, ``<site>:negative_far``), the plain stratum in ``stratum_in_site``."""
    from nakkhu.audit.core import P2_TASK

    tcfg = cfg["tasks"][P2_TASK]
    p2_require_freeze(project, tcfg)
    maps = p2_change_maps(project, tcfg)
    parts, stats, unit = [], {}, 0
    for site in tcfg["sites"]:
        site = str(site)
        build = gpd.read_parquet(maps[site]["file"]).to_crs(geo.WORK_CRS)
        for col in ("building_id", "class", "observable"):
            if col not in build.columns:
                raise AuditError(f"site {site}: change map lacks column {col!r}")
        build = build.loc[build["observable"].astype(bool)]
        extent = geo.read_geometry(p2_site_path(project, tcfg, "site_extent", site))
        cent = shapely.centroid(np.asarray(build.geometry.values, dtype=object))
        inside = shapely.within(cent, extent)
        build, cent = build.loc[inside].reset_index(drop=True), cent[inside]
        lines = gpd.read_file(p2_site_path(project, tcfg, "site_lines", site)).to_crs(
            geo.WORK_CRS)
        dist = shapely.distance(cent, shapely.union_all(lines.geometry.values))
        positive = build["class"].astype(str).str.lower().isin(["positive", "1", "true",
                                                                "changed"])
        near = dist <= float(tcfg["near_channel_m"])
        plain = np.where(positive, "positive", np.where(near, "negative_near", "negative_far"))
        n = len(build)
        parts.append(gpd.GeoDataFrame({
            "unit": np.arange(unit, unit + n), "building_id": build["building_id"].to_numpy(),
            "change_class": build["class"].astype(str).to_numpy(), "dist_channel_m": dist,
            "x": shapely.get_x(cent), "y": shapely.get_y(cent), "site": site,
            "stratum_in_site": plain, "stratum": np.char.add(f"{site}:", plain.astype(str))},
            geometry=gpd.GeoSeries(np.asarray(build.geometry.values, dtype=object),
                                   crs=geo.WORK_CRS)))
        unit += n
        stats[site] = {"buildings": int(n), "change_map_sha256": maps[site]["sha256"]}
    frame = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=geo.WORK_CRS)
    return frame, {"stats": {"sites": stats, "buildings": int(len(frame))}}
