"""Label-arm builder: shared preparation (harmonise, match, register) and per-arm outputs.

``prepare`` runs Steps 1-3 once and caches the match graph and the raw registration under
``{data_root}/interim/labels/_common/`` (keyed by the source hashes, the tile grid and the
settings). ``build_arm`` then writes, per arm, ``{data_root}/interim/labels/<arm>/``:
``labels/<tile>.txt`` (YOLO-OBB), ``ignore/<tile>.png`` (only tiles with ignore pixels),
``tiles.txt`` (tiles admitted for training), ``instances.parquet`` (every tile-instance decision)
and ``tile_stats.parquet``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from nakkhu.config import Project
from nakkhu.labels import tiles as T
from nakkhu.labels.arms import Selection, select
from nakkhu.labels.harmonise import LABEL_SOURCES, Footprints, load_footprints
from nakkhu.labels.match import MatchResult, class_table, match
from nakkhu.labels.obb import (
    IGNORE,
    KEEP,
    ObbFilters,
    ignore_mask,
    tile_labels,
    write_ignore_png,
    write_yolo_obb,
)
from nakkhu.labels.register import null_peak_ratio, register_tile
from nakkhu.labels.sam2_refine import prompt_box
from nakkhu.manifest import load_manifest
from nakkhu.runs import atomic_write_text
from nakkhu.tiling import load_tiling

CACHE_VERSION = 1


class LabelBuildError(RuntimeError):
    """An arm cannot be built (missing inputs or settings)."""


@dataclass
class Prepared:
    index: pd.DataFrame
    tile_dir: Path
    footprints: Footprints
    matches: MatchResult
    tile_google: list[np.ndarray]
    tile_microsoft: list[np.ndarray]
    registration: pd.DataFrame
    stats: dict[str, Any]
    excluded: dict[str, np.ndarray] = field(default_factory=dict)

    def geoms(self, source: str) -> np.ndarray:
        frame = self.footprints.google if source == "google" else self.footprints.microsoft
        return frame.geometry.values


def labels_root(project: Project) -> Path:
    return project.paths.data_root / "interim" / "labels"


def _source_hashes(project: Project) -> dict[str, str]:
    manifest = (project.config.values.get("paths") or {}).get("manifest",
                                                               "configs/data/manifest.yaml")
    entries = {e["id"]: e for e in load_manifest(project.paths.resolve(manifest))}
    return {key: str(entries[eid]["sha256"]) for key, eid in LABEL_SOURCES.items()}


def _cache_key(project: Project, cfg: dict[str, Any], index: pd.DataFrame) -> str:
    material = {
        "version": CACHE_VERSION,
        "sources": _source_hashes(project),
        "grid": sorted(set(index["grid_sha256"])),
        "tiles": cfg["tiles"], "match": cfg["match"], "registration": cfg["registration"],
        "fallback": cfg.get("fallback_thresholds"),
    }
    text = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# --- registration workers (top level for process pools) -------------------------------------


def _register_worker(args: tuple[Any, ...]) -> tuple[float, float, float, str]:
    directory, row, wkb, reg = args
    rgb, valid = T.read_tile(directory, row)
    polys = shapely.from_wkb(np.asarray(wkb, dtype=object)) if len(wkb) else np.array([])
    out = register_tile(rgb, valid, polys, max_shift_px=float("inf"), min_peak_ratio=0.0,
                        sigma_px=float(reg["sigma_px"]), exclusion_px=int(reg["exclusion_px"]))
    return out.raw_dx, out.raw_dy, out.peak_ratio, out.reason


def _null_worker(args: tuple[Any, ...]) -> float:
    directory, row, wkb, reg = args
    rgb, valid = T.read_tile(directory, row)
    polys = shapely.from_wkb(np.asarray(wkb, dtype=object))
    return null_peak_ratio(rgb, valid, polys, float(reg["sigma_px"]), int(reg["exclusion_px"]))


def _pool_map(func: Any, items: list[Any], workers: int) -> list[Any]:
    if workers <= 1 or len(items) < 8:
        return [func(item) for item in items]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items, chunksize=max(1, len(items) // (workers * 8))))


def tile_polygons_px(prep: Prepared, i: int, sources: tuple[str, ...] = ("google", "microsoft")
                     ) -> np.ndarray:
    row = prep.index.iloc[i]
    parts = []
    for source in sources:
        idx = (prep.tile_google if source == "google" else prep.tile_microsoft)[i]
        parts.append(prep.geoms(source)[idx])
    geoms = np.concatenate(parts) if parts else np.array([], dtype=object)
    return T.to_px(geoms, row["affine"])


def decide_registration(raw: pd.DataFrame, reg: dict[str, Any], threshold: float) -> pd.DataFrame:
    out = raw.copy()
    magnitude = np.hypot(out["raw_dx"], out["raw_dy"])
    reason = np.where(out["reason"] == "no_footprints", "no_footprints",
                      np.where(magnitude > float(reg["max_shift_px"]), "shift_too_large",
                               np.where(out["peak_ratio"] >= threshold, "ok", "low_peak")))
    out["reason"] = reason
    out["reg_ok"] = reason == "ok"
    out["dx"] = np.where(out["reg_ok"], out["raw_dx"], 0.0)
    out["dy"] = np.where(out["reg_ok"], out["raw_dy"], 0.0)
    out["min_peak_ratio"] = threshold
    return out


def _null_threshold(prep_index: pd.DataFrame, directory: Path, wkb: list[list[bytes]],
                    cfg: dict[str, Any], eligible: np.ndarray, workers: int
                    ) -> tuple[float, np.ndarray]:
    rule = cfg["registration"]["min_peak_ratio"]
    if not isinstance(rule, dict):
        return float(rule), np.array([])
    rng = np.random.default_rng(int(rule["seed"]))
    pool = np.flatnonzero(eligible)
    if len(pool) < 2:
        raise LabelBuildError("too few eligible tiles for the registration null")
    n = int(rule["null_pairs"])
    image_i = rng.choice(pool, size=n, replace=True)
    offset = rng.integers(1, len(pool), size=n)
    pos = np.searchsorted(pool, image_i)
    fp_i = pool[(pos + offset) % len(pool)]
    items = [(directory, prep_index.iloc[a].to_dict(), wkb[b], cfg["registration"])
             for a, b in zip(image_i, fp_i, strict=True)]
    ratios = np.array(_pool_map(_null_worker, items, workers), dtype=float)
    return float(np.nanquantile(ratios, float(rule["null_quantile"]))), ratios


def prepare(project: Project, cfg: dict[str, Any], workers: int = 1,
            use_cache: bool = True) -> Prepared:
    directory = T.tile_dir(project, cfg["tiles"]["scene"], cfg["tiles"]["purpose"])
    index = T.load_index(directory)
    has_image = index["image"].notna().to_numpy()
    index = index.loc[has_image].reset_index(drop=True)
    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    clip = shapely.box(*shapely.total_bounds(boxes))
    footprints, fp_stats = load_footprints(project, cfg, clip=clip)
    g_geoms = footprints.google.geometry.values
    m_geoms = footprints.microsoft.geometry.values

    key = _cache_key(project, cfg, index)
    cache = labels_root(project) / "_common" / key
    stats: dict[str, Any] = {"cache_key": key, "footprints": fp_stats}
    if use_cache and (cache / "registration.parquet").is_file():
        matches = MatchResult(edges=pd.read_parquet(cache / "edges.parquet"),
                              google=pd.read_parquet(cache / "match_google.parquet"),
                              microsoft=pd.read_parquet(cache / "match_microsoft.parquet"))
        raw = pd.read_parquet(cache / "registration.parquet")
        cached_meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        null_ratios = pd.read_parquet(cache / "null.parquet")["peak_ratio"].to_numpy()
        threshold = float(cached_meta["min_peak_ratio"])
        stats["cache"] = "reused"
    else:
        stats["cache"] = "built"
        matches = match(g_geoms, footprints.google["confidence"].to_numpy(),
                        footprints.google["t90"].to_numpy(), m_geoms,
                        iou_min=float(cfg["match"]["iou_min"]),
                        contain_min=float(cfg["match"]["contain_min"]))
        raw, null_ratios, threshold = None, None, None
    ti_g, gi = shapely.STRtree(g_geoms).query(boxes, predicate="intersects")
    ti_m, mi = shapely.STRtree(m_geoms).query(boxes, predicate="intersects")
    tile_google = _group(ti_g, gi, len(index))
    tile_microsoft = _group(ti_m, mi, len(index))
    prep = Prepared(index=index, tile_dir=directory, footprints=footprints, matches=matches,
                    tile_google=tile_google, tile_microsoft=tile_microsoft,
                    registration=pd.DataFrame(), stats=stats)
    if raw is None:
        wkb = [list(shapely.to_wkb(tile_polygons_px(prep, i))) for i in range(len(index))]
        items = [(directory, index.iloc[i].to_dict(), wkb[i], cfg["registration"])
                 for i in range(len(index))]
        out = _pool_map(_register_worker, items, workers)
        raw = pd.DataFrame(out, columns=["raw_dx", "raw_dy", "peak_ratio", "reason"])
        raw.insert(0, "tile_id", index["tile_id"].to_numpy())
        eligible = (index["domain"].isin(cfg["tiles"]["train_domains"]).to_numpy()
                    & ~index["low_valid"].to_numpy(dtype=bool)
                    & np.array([len(w) > 0 for w in wkb]))
        threshold, null_ratios = _null_threshold(index, directory, wkb, cfg, eligible, workers)
        if use_cache:
            cache.mkdir(parents=True, exist_ok=True)
            matches.edges.to_parquet(cache / "edges.parquet")
            matches.google.to_parquet(cache / "match_google.parquet")
            matches.microsoft.to_parquet(cache / "match_microsoft.parquet")
            raw.to_parquet(cache / "registration.parquet")
            pd.DataFrame({"peak_ratio": null_ratios}).to_parquet(cache / "null.parquet")
            atomic_write_text(cache / "meta.json", json.dumps(
                {"min_peak_ratio": threshold, "key": key}, indent=1))
    assert raw is not None and null_ratios is not None and threshold is not None
    prep.registration = decide_registration(raw, cfg["registration"], threshold)
    load_exclusions(project, cfg, prep)
    stats["registration"] = registration_summary(prep.registration, index, null_ratios,
                                                 threshold, cfg)
    stats["match_classes"] = class_table(matches).to_dict("records")
    return prep


def greenhouse_areas(project: Project, opts: dict[str, Any]) -> gpd.GeoDataFrame:
    """Shared greenhouse areas (``nakkhu.reference.greenhouse``), with the exposure code's bbox:
    analysis extent + training domain in lon/lat, padded by ``bbox_pad_deg``; working CRS."""
    from nakkhu.reference import greenhouse

    aoi = project.config.values.get("aoi") or {}
    crs = project.config.values.get("crs", "EPSG:32645")
    shapes = [gpd.read_file(project.paths.resolve(aoi[key])).to_crs(crs).union_all()
              for key in ("analysis_extent", "training_domain") if aoi.get(key)]
    lonlat = gpd.GeoSeries(shapes, crs=crs).to_crs("EPSG:4326").total_bounds
    pad = float(opts["bbox_pad_deg"])
    bbox = (lonlat[0] - pad, lonlat[1] - pad, lonlat[2] + pad, lonlat[3] + pad)
    frame = greenhouse.load_greenhouses(project.paths.resolve(opts["pbf"]), bbox,
                                        project.paths.data_root / "interim" / "reference")
    return frame.to_crs(crs)


def load_exclusions(project: Project, cfg: dict[str, Any], prep: Prepared) -> None:
    """Flag footprints overlapping a greenhouse area (``greenhouse_filter`` in the config)."""
    from nakkhu.reference.greenhouse import greenhouse_overlap

    opts = cfg["greenhouse_filter"]
    areas = greenhouse_areas(project, opts)
    prep.excluded = {}
    for source in ("google", "microsoft"):
        frame = prep.footprints.google if source == "google" else prep.footprints.microsoft
        flags = greenhouse_overlap(frame.geometry, areas.geometry, float(opts["iou_min"]))
        prep.excluded[source] = flags["is_greenhouse"].to_numpy()
    prep.stats["excluded"] = {"greenhouse_areas": int(len(areas)),
                              "areas_by_tag": areas["tag"].value_counts().to_dict(),
                              **{s: int(v.sum()) for s, v in prep.excluded.items()}}


def _group(tile_idx: np.ndarray, inst_idx: np.ndarray, n: int) -> list[np.ndarray]:
    order = np.argsort(tile_idx, kind="stable")
    tile_idx, inst_idx = tile_idx[order], inst_idx[order]
    bounds = np.searchsorted(tile_idx, np.arange(n + 1))
    return [inst_idx[bounds[i]:bounds[i + 1]] for i in range(n)]


def registration_summary(reg: pd.DataFrame, index: pd.DataFrame, null_ratios: np.ndarray,
                         threshold: float, cfg: dict[str, Any]) -> dict[str, Any]:
    rule = cfg["registration"]["min_peak_ratio"]
    out: dict[str, Any] = {
        "min_peak_ratio": threshold,
        "rule": rule,
        "null_pairs_used": int(np.isfinite(null_ratios).sum()) if len(null_ratios) else 0,
        "by_domain": {},
    }
    merged = reg.merge(index[["tile_id", "domain", "low_valid"]], on="tile_id")
    for domain, part in merged.groupby("domain"):
        ok = part.loc[part["reg_ok"]]
        out["by_domain"][str(domain)] = {
            "tiles": len(part),
            "reasons": part["reason"].value_counts().to_dict(),
            "registered_share": float(part["reg_ok"].mean()),
            "median_dx_px": float(ok["dx"].median()) if len(ok) else None,
            "median_dy_px": float(ok["dy"].median()) if len(ok) else None,
        }
    return out


# --- arms -----------------------------------------------------------------------------------


def _arm_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    directory, row, wkb, actions, tile_px, filters = args
    _, valid = T.read_tile(directory, row) if T.has_mask(row) else (None, None)
    region = (T.valid_region(valid) if valid is not None
              else shapely.box(0, 0, tile_px, tile_px))
    polys = shapely.from_wkb(np.asarray(wkb, dtype=object)) if len(wkb) else np.array([])
    result = tile_labels(polys, np.asarray(actions, dtype=object), region, tile_px, filters)
    return {"boxes": result.boxes, "ignore": shapely.to_wkb(result.ignore),
            "decisions": result.decisions, "reasons": result.reasons,
            "building_area_px": result.building_area_px,
            "ignore_area_px": result.ignore_area_px}


def instance_polygons(prep: Prepared, i: int, selection: Selection,
                      membership: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
    """Registered pixel polygons of the selected instances intersecting tile ``i``."""
    row = prep.index.iloc[i]
    reg = prep.registration.iloc[i]
    sources, indices, actions, geoms = [], [], [], []
    for source, tile_lists in (("google", prep.tile_google), ("microsoft", prep.tile_microsoft)):
        idx = tile_lists[i]
        if len(idx) == 0:
            continue
        pos = membership[source][idx]
        take = pos >= 0
        if not take.any():
            continue
        sel_pos = pos[take]
        sources.append(np.full(take.sum(), source, dtype=object))
        indices.append(idx[take])
        actions.append(selection.action[sel_pos])
        geoms.append(prep.geoms(source)[idx[take]])
    if not geoms:
        empty = np.array([], dtype=object)
        return empty, empty, np.array([], dtype=np.int64), empty
    px = T.to_px(np.concatenate(geoms), row["affine"])
    px = shapely.transform(px, lambda xy: xy + np.array([reg["dx"], reg["dy"]]))
    return (np.concatenate(sources), np.concatenate(actions), np.concatenate(indices), px)


def membership_arrays(prep: Prepared, selection: Selection) -> dict[str, np.ndarray]:
    """Position of each source instance in ``selection`` (-1 if not selected)."""
    out = {}
    for source in ("google", "microsoft"):
        arr = np.full(len(prep.geoms(source)), -1, dtype=np.int64)
        mask = selection.source == source
        arr[selection.index[mask]] = np.flatnonzero(mask)
        out[source] = arr
    return out


@dataclass
class ArmInputs:
    """Per tile: sources, actions, indices and registered pixel polygons of one arm."""

    sources: list[np.ndarray]
    actions: list[np.ndarray]
    indices: list[np.ndarray]
    polygons: list[np.ndarray]
    reasons: list[np.ndarray]
    excluded: list[int] = field(default_factory=list)   # selected instances removed per tile
    geometry: list[np.ndarray] = field(default_factory=list)   # "footprint" or "sam2"


def arm_inputs(prep: Prepared, arm_cfg: dict[str, Any]) -> ArmInputs:
    selection = select(arm_cfg["rule"], prep.matches)
    member = membership_arrays(prep, selection)
    flags = prep.excluded or {s: np.zeros(len(prep.geoms(s)), dtype=bool)
                              for s in ("google", "microsoft")}
    out = ArmInputs([], [], [], [], [])
    for i in range(len(prep.index)):
        out.excluded.append(int(sum(
            ((member[s][lists[i]] >= 0) & flags[s][lists[i]]).sum()
            for s, lists in (("google", prep.tile_google),
                             ("microsoft", prep.tile_microsoft)))))
    member = {s: np.where(flags[s], -1, member[s]) for s in member}
    for i in range(len(prep.index)):
        s, a, idx, px = instance_polygons(prep, i, selection, member)
        out.sources.append(s)
        out.actions.append(a)
        out.indices.append(idx)
        out.polygons.append(px)
        out.reasons.append(np.full(len(idx), "", dtype=object))
        out.geometry.append(np.full(len(idx), "footprint", dtype=object))
    return out


def apply_sam2(prep: Prepared, inputs: ArmInputs, results: pd.DataFrame,
               min_iou: float, min_area_px: float, dilate_px: float,
               tile_px: int) -> np.ndarray:
    """Replace a kept footprint by its SAM2 mask polygon where mask/footprint IoU >=
    ``min_iou``; otherwise keep the registered footprint unchanged (Decisions 2026-10-06:
    refinement must not remove coverage). Records ``inputs.geometry`` per instance
    (``sam2`` / ``footprint``). Returns a per-tile bool array: SAM2 outputs complete."""
    lookup = {(t, s, int(k)): (iou, wkt) for t, s, k, iou, wkt in
              results[["tile_id", "source", "index", "iou", "mask_wkt"]].itertuples(index=False)}
    complete = np.ones(len(prep.index), dtype=bool)
    for i, tile_id in enumerate(prep.index["tile_id"]):
        actions = inputs.actions[i].copy()
        polys = inputs.polygons[i].copy()
        reasons = inputs.reasons[i].copy()
        geometry = inputs.geometry[i].copy()
        for j in np.flatnonzero(actions == KEEP):
            key = (tile_id, inputs.sources[i][j], int(inputs.indices[i][j]))
            if key not in lookup:
                # Missing only if the instance was prompted (same rule as the prompt table);
                # an unprompted one (tiny or outside the tile) keeps its action for Step 5.
                if prompt_box(polys[j], min_area_px, dilate_px, tile_px) is not None:
                    complete[i] = False
                continue
            iou, wkt = lookup[key]
            if iou is not None and np.isfinite(iou) and iou >= min_iou and wkt:
                polys[j] = shapely.from_wkt(wkt)
                geometry[j] = "sam2"
        inputs.actions[i], inputs.polygons[i], inputs.reasons[i] = actions, polys, reasons
        inputs.geometry[i] = geometry
    return complete


def build_arm(project: Project, prep: Prepared, arm_cfg: dict[str, Any], *, workers: int = 1,
              sam2_results: pd.DataFrame | None = None, admitted_by_gate: set[str] | None = None,
              write: bool = True) -> dict[str, Any]:
    """Build one arm; returns per-tile stats, instance decisions and the summary."""
    arm = arm_cfg["arm"]
    tile_px = int(load_tiling(project).values["grid"]["tile_px"])
    pixel = float(prep.index.iloc[0]["affine"][0])
    filters = ObbFilters.from_config(arm_cfg["obb"], pixel)
    inputs = arm_inputs(prep, arm_cfg)
    sam2_complete = np.ones(len(prep.index), dtype=bool)
    if arm_cfg.get("refine") == "sam2":
        if sam2_results is None:
            raise LabelBuildError(f"{arm} needs merged SAM2 outputs (nakkhu labels sam2-merge)")
        sam2_complete = apply_sam2(prep, inputs, sam2_results,
                                   float(arm_cfg["sam2"]["min_mask_iou"]),
                                   filters.min_area_m2 / pixel ** 2,
                                   float(arm_cfg["sam2"]["box_dilate_m"]) / pixel, tile_px)
    if arm_cfg.get("use_gate") and admitted_by_gate is None:
        raise LabelBuildError(f"{arm} needs gate.admit_fraction (chosen on val blocks)")
    items = [(prep.tile_dir, prep.index.iloc[i].to_dict(),
              list(shapely.to_wkb(inputs.polygons[i])), list(inputs.actions[i]), tile_px,
              filters) for i in range(len(prep.index))]
    results = _pool_map(_arm_worker, items, workers)

    out_dir = labels_root(project) / arm
    staging = out_dir.with_name(f".{arm}.staging")
    if write:
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "labels").mkdir(parents=True)
    rows, inst_rows = [], []
    split_of, usable = _split_lookup(project, arm_cfg)
    train_split = arm_cfg["split"]["train"]
    for i, res in enumerate(results):
        row = prep.index.iloc[i]
        tile_id = row["tile_id"]
        decisions = np.asarray(res["decisions"], dtype=object)
        reasons = np.asarray(res["reasons"], dtype=object)
        pre = inputs.reasons[i]
        reasons = np.where((pre != "") & (decisions == IGNORE), pre, reasons)
        n_keep = int((decisions == KEEP).sum())
        n_ignore = int((decisions == IGNORE).sum())
        n_drop = int((decisions == "drop").sum())
        ignore_frac = (res["ignore_area_px"] / res["building_area_px"]
                       if res["building_area_px"] > 0 else 0.0)
        not_admitted = []
        split = split_of.get(tile_id, "none")
        if split != train_split:
            not_admitted.append(f"split_{split}")
        if bool(row["low_valid"]) or not usable.get(tile_id, False):
            not_admitted.append("low_valid")
        if ignore_frac > filters.max_ignore_frac:
            not_admitted.append("ignore_frac")
        if not sam2_complete[i]:
            not_admitted.append("sam2_missing")
        if admitted_by_gate is not None and tile_id not in admitted_by_gate:
            not_admitted.append("gate")
        rows.append({"tile_id": tile_id, "domain": row["domain"], "split": split,
                     "excluded": inputs.excluded[i],
                     "low_valid": bool(row["low_valid"]), "keep": n_keep, "ignore": n_ignore,
                     "drop": n_drop, "ignore_frac": ignore_frac,
                     "building_area_m2": res["building_area_px"] * pixel ** 2,
                     "ignore_area_m2": res["ignore_area_px"] * pixel ** 2,
                     "sam2_complete": bool(sam2_complete[i]),
                     "sam2_replaced": int(((decisions == KEEP)
                                           & (inputs.geometry[i] == "sam2")).sum()),
                     "admitted": not not_admitted, "not_admitted": ";".join(not_admitted)})
        inst_rows.append(pd.DataFrame({
            "tile_id": tile_id, "source": inputs.sources[i], "index": inputs.indices[i],
            "action": inputs.actions[i], "decision": decisions, "reason": reasons,
            "geometry": inputs.geometry[i]}))
        if write:
            write_yolo_obb(staging / "labels" / f"{tile_id}.txt", res["boxes"], tile_px)
            region = shapely.from_wkb(res["ignore"])
            if not region.is_empty:
                mask = ignore_mask(region, tile_px)
                if mask.any():
                    write_ignore_png(staging / "ignore" / f"{tile_id}.png", mask)
    tile_stats = pd.DataFrame(rows)
    instances = pd.concat(inst_rows, ignore_index=True) if inst_rows else pd.DataFrame()
    if write:
        admitted = tile_stats.loc[tile_stats["admitted"], "tile_id"].tolist()
        atomic_write_text(staging / "tiles.txt", "".join(f"{t}\n" for t in admitted))
        tile_stats.to_parquet(staging / "tile_stats.parquet")
        instances.to_parquet(staging / "instances.parquet")
        atomic_write_text(staging / "arm.json", json.dumps(
            {"arm": arm, "config": arm_cfg, "cache_key": prep.stats["cache_key"]},
            indent=1, sort_keys=True, default=str))
        old = out_dir.with_name(f".{arm}.old")
        if old.exists():
            shutil.rmtree(old)
        if out_dir.exists():
            out_dir.replace(old)
        staging.replace(out_dir)
        if old.exists():
            shutil.rmtree(old)
    return {"tile_stats": tile_stats, "instances": instances}


def _split_lookup(project: Project, arm_cfg: dict[str, Any]) -> tuple[dict[str, str],
                                                                     dict[str, bool]]:
    from nakkhu.splits import load_split

    table, _ = load_split(project, arm_cfg["split"]["file"])
    return (dict(zip(table["tile_id"], table["split"], strict=True)),
            dict(zip(table["tile_id"], table["usable"], strict=True)))


def split_ids(project: Project, arm_cfg: dict[str, Any], split: str) -> list[str]:
    """Usable tiles of one plan 005 split (``train`` / ``val``)."""
    split_of, usable = _split_lookup(project, arm_cfg)
    return sorted(t for t, s in split_of.items() if s == split and usable.get(t, False))


def arm_stats_rows(arm: str, status: str, tile_stats: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Rows of ``arm_stats.csv``: per arm and plan 005 split (plus ``all``), tile-instance
    counts (``excluded`` = instances of the arm removed by the exclusion layer)."""
    if tile_stats is None or tile_stats.empty:
        return [{"arm": arm, "status": status, "split": "all"}]
    rows = []
    groups = [("all", tile_stats)] + [(str(d), p) for d, p in tile_stats.groupby("split")]
    for domain, part in groups:
        usable = part.loc[~part["low_valid"]]
        rows.append({
            "arm": arm, "status": status, "split": domain, "tiles": len(part),
            "tiles_valid": len(usable),
            "tiles_admitted": int(part["admitted"].sum()),
            "tiles_dropped_ignore": int(usable["not_admitted"].str.contains("ignore_frac").sum()),
            "excluded": int(part["excluded"].sum()),
            "kept": int(part["keep"].sum()), "ignored": int(part["ignore"].sum()),
            "dropped": int(part["drop"].sum()),
            "kept_admitted": int(part.loc[part["admitted"], "keep"].sum()),
            "kept_sam2_geometry": int(part["sam2_replaced"].sum()),
            "ignore_area_share": float(part["ignore_area_m2"].sum()
                                       / max(part["building_area_m2"].sum(), 1e-9)),
        })
    return rows


def footprints_frame(prep: Prepared, source: str) -> gpd.GeoDataFrame:
    return prep.footprints.google if source == "google" else prep.footprints.microsoft
