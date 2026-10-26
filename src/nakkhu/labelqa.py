"""Label quality against the held-out OSM reference (plan 004 Steps 6 and 8; R-OSM).

This is the only plan 004 module that reads OpenStreetMap. OSM is a *reference*: it never
enters a label arm (``nakkhu.labels`` cannot import it; see tests/test_labels.py).

- :func:`convention_table` (Step 8): 1:n table of Google / Microsoft (/ SAM2) units against OSM
  in conflict components, and F1 at IoU 0.3 and 0.5 of each unit convention against OSM.
- :func:`register_osm` (Decisions 2026-10-05): OSM is registered per tile to the imagery with the
  footprint method (Step 3: boundary vs gradient phase correlation, null-calibrated peak ratio,
  12 px cap) before any scoring; :func:`aligned_osm` returns OSM shifted by its tile's shift and
  is the entry point for plan 005 scoring.
- :func:`tile_label_quality` and :func:`g1` (Step 6): per-tile label quality (F1) against
  aligned OSM and Spearman rho(gate score, quality); G1 passes at rho >= 0.3 on test tiles.
  Test tiles are refused unless ``configs/prereg.yaml`` exists (plan 005 freezes it first).

Matching: greedy one-to-one by IoU (highest first), a pair counts at IoU >= the threshold.
OSM completeness here is unknown, so unmatched labels are not all false positives (caveat
reported with every number).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from PIL import Image

from nakkhu.config import Project
from nakkhu.labels import tiles as T
from nakkhu.labels.match import CONFLICT_CLASSES, candidate_pairs, overlap_stats
from nakkhu.labels.obb import read_yolo_obb
from nakkhu.manifest import load_manifest

OSM_ID = "fp_osm"
OSM_SHIFTS = "_osm/registration.parquet"   # under {data_root}/interim/labels/


class LabelQAError(RuntimeError):
    """A reference comparison would break the evaluation rules."""


def load_osm(project: Project, clip: Any | None = None,
             drop_greenhouses: bool = False) -> gpd.GeoDataFrame:
    """OSM building polygons (exploded, polygonal). ``drop_greenhouses`` removes buildings that
    overlap a shared greenhouse area (same rule as labels and exposure); scoring uses it."""
    manifest = (project.config.values.get("paths") or {}).get("manifest",
                                                               "configs/data/manifest.yaml")
    entries = {e["id"]: e for e in load_manifest(project.paths.resolve(manifest))}
    path = project.paths.resolve(entries[OSM_ID]["local_path"]) / f"{OSM_ID}.parquet"
    frame = gpd.read_parquet(path)
    if clip is not None:
        frame = frame.loc[shapely.intersects(frame.geometry.values, clip)]
    frame = frame.explode(index_parts=False, ignore_index=True)
    geoms = frame.geometry.values
    keep = (shapely.get_type_id(geoms) == 3) & (shapely.area(geoms) > 0)
    frame = frame.loc[keep].reset_index(drop=True)
    if drop_greenhouses:
        from nakkhu.labels.arms import load_arm_config
        from nakkhu.labels.build import greenhouse_areas
        from nakkhu.reference.greenhouse import greenhouse_overlap

        opts = load_arm_config(project.paths.resolve("configs/labels/common.yaml"))[
            "greenhouse_filter"]
        flags = greenhouse_overlap(frame.geometry, greenhouse_areas(project, opts).geometry,
                                   float(opts["iou_min"]))
        frame = frame.loc[~flags["is_greenhouse"].to_numpy()].reset_index(drop=True)
    return frame


def greedy_match(pred: np.ndarray, ref: np.ndarray, iou_min: float) -> tuple[int, int, int]:
    """(TP, FP, FN) of a greedy one-to-one IoU matching."""
    pred = np.asarray(pred, dtype=object)
    ref = np.asarray(ref, dtype=object)
    if len(pred) == 0 or len(ref) == 0:
        return 0, len(pred), len(ref)
    ip, ir = candidate_pairs(pred, ref)
    iou, _, _ = overlap_stats(pred, ref, ip, ir)
    order = np.argsort(-iou, kind="stable")
    used_p = np.zeros(len(pred), dtype=bool)
    used_r = np.zeros(len(ref), dtype=bool)
    tp = 0
    for k in order:
        if iou[k] < iou_min:
            break
        a, b = ip[k], ir[k]
        if not used_p[a] and not used_r[b]:
            used_p[a] = used_r[b] = True
            tp += 1
    return tp, len(pred) - tp, len(ref) - tp


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")


def spearman(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# --- Step 8: row-house convention -----------------------------------------------------------


def convention_table(prep: Any, osm: gpd.GeoDataFrame, tile_ids: set[str],
                     contain_min: float,
                     sam2_units: gpd.GeoDataFrame | None = None) -> tuple[pd.DataFrame,
                                                                          pd.DataFrame]:
    """1:n counts and F1 of Google vs Microsoft units in conflict components.

    Components are restricted to those whose members all lie in ``tile_ids`` tiles (by member
    centroid), so the region is chosen by the caller (never test tiles).
    Returns (per-component table, summary table).
    """
    index = prep.index.set_index("tile_id")
    region = shapely.union_all([shapely.box(*index.loc[t, "bounds"]) for t in sorted(tile_ids)])
    g, m = prep.footprints.google, prep.footprints.microsoft
    gm, mm = prep.matches.google, prep.matches.microsoft
    g_in = shapely.within(shapely.centroid(g.geometry.values), region)
    m_in = shapely.within(shapely.centroid(m.geometry.values), region)
    comps = pd.concat([
        pd.DataFrame({"component": gm["component"], "source": "google",
                      "i": np.arange(len(gm)), "cls": gm["cls"], "inside": g_in}),
        pd.DataFrame({"component": mm["component"], "source": "microsoft",
                      "i": np.arange(len(mm)), "cls": mm["cls"], "inside": m_in}),
    ])
    comps = comps.loc[comps["cls"].isin(CONFLICT_CLASSES)]
    full = comps.groupby("component")["inside"].transform("all")
    comps = comps.loc[full]
    osm_geoms = osm.geometry.values
    osm_tree = shapely.STRtree(osm_geoms)
    rows = []
    preds = {"google": [], "microsoft": [], "osm": set()}
    for comp_id, part in comps.groupby("component"):
        gi = part.loc[part["source"] == "google", "i"].to_numpy()
        mi = part.loc[part["source"] == "microsoft", "i"].to_numpy()
        members = np.concatenate([g.geometry.values[gi], m.geometry.values[mi]])
        hull = shapely.union_all(members)
        cand = osm_tree.query(hull, predicate="intersects")
        if len(cand):
            # OSM buildings with >= contain_min of their area inside the component's union.
            _, _, osm_inside = overlap_stats(np.array([hull], dtype=object), osm_geoms,
                                             np.zeros(len(cand), dtype=np.int64), cand)
            cand = cand[osm_inside >= contain_min]
        preds["google"].extend(g.geometry.values[gi])
        preds["microsoft"].extend(m.geometry.values[mi])
        preds["osm"].update(cand.tolist())
        rows.append({"component": int(comp_id), "cls": part["cls"].iloc[0],
                     "n_google": len(gi), "n_microsoft": len(mi), "n_osm": len(cand)})
    table = pd.DataFrame(rows)
    ref = osm_geoms[sorted(preds["osm"])] if preds["osm"] else np.array([], dtype=object)
    summary = []
    units = {"google_split": np.array(preds["google"], dtype=object),
             "microsoft_merged": np.array(preds["microsoft"], dtype=object)}
    if sam2_units is not None and len(sam2_units):
        units["sam2"] = sam2_units.geometry.values
    for name, pred in units.items():
        for thr in (0.3, 0.5):
            tp, fp, fn = greedy_match(pred, ref, thr)
            summary.append({"convention": name, "iou": thr, "units": len(pred),
                            "osm_refs": len(ref), "tp": tp, "fp": fp, "fn": fn,
                            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                            "f1": f1(tp, fp, fn)})
    if len(table):
        with_osm = table.loc[table["n_osm"] > 0]
        for label, cond in (("osm_eq_google", with_osm["n_osm"] == with_osm["n_google"]),
                            ("osm_eq_microsoft", with_osm["n_osm"] == with_osm["n_microsoft"]),
                            ("osm_between", (with_osm["n_osm"] > with_osm[["n_google",
                                             "n_microsoft"]].min(axis=1))
                             & (with_osm["n_osm"] < with_osm[["n_google",
                                                              "n_microsoft"]].max(axis=1)))):
            summary.append({"convention": label, "iou": None, "units": int(cond.sum()),
                            "osm_refs": len(with_osm), "tp": None, "fp": None, "fn": None,
                            "precision": None, "recall": None,
                            "f1": float(cond.mean()) if len(with_osm) else None})
    return table, pd.DataFrame(summary)


# --- Step 6: per-tile label error and G1 ----------------------------------------------------


def _guard_split(project: Project, domains: set[str],
                 access: tuple[str, str] | None = None,
                 note: str = "plan 004 G1") -> dict[str, Any] | None:
    """Refuse test tiles unless plan 005's freeze holds, then journal the access.

    Test scoring needs ``evaluate.guard.require_frozen`` to pass (live prereg hash = frozen hash,
    split hash = the one in the prereg, mirrored configs unchanged) and ``access`` = (run ref,
    config hash) so that the access is appended to the test journal before any scoring.
    """
    if "test" not in domains:
        return None
    from nakkhu.evaluate.guard import GuardError, record_test_access, require_frozen

    try:
        state = require_frozen(project)
    except GuardError as error:
        raise LabelQAError(str(error)) from error
    if access is None:
        raise LabelQAError("test scoring needs a run reference for the test journal")
    run_ref, cfghash = access
    return record_test_access(project, run_ref, cfghash, state["prereg_sha256"],
                              state["split_sha256"], note=note)


def osm_shift_path(project: Project) -> Path:
    from nakkhu.labels.build import labels_root

    return labels_root(project) / OSM_SHIFTS


def register_osm(project: Project, index: pd.DataFrame, directory: Path, cfg: dict[str, Any],
                 workers: int = 1) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-tile OSM registration with the footprint method, settings and null rule
    (``registration`` in ``configs/labels/common.yaml``).

    ``min_peak_ratio`` is calibrated on the OSM null (mismatched image/OSM pairs from train
    tiles, same quantile, pair count and seed as ``registration.min_peak_ratio``). The summary
    also gives the null acceptance rate (ratio >= threshold and shift within the cap) at the
    footprint cap and at the OSM cap. Every tile with an image gets a row; flagged tiles keep
    shift 0 and their reason.
    """
    from nakkhu.labels.build import (
        _pool_map,
        _register_worker,
        decide_registration,
    )

    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    osm = load_osm(project, shapely.box(*shapely.total_bounds(boxes)))
    geoms = osm.geometry.values
    ti, oi = shapely.STRtree(geoms).query(boxes, predicate="intersects")
    order = np.argsort(ti, kind="stable")
    ti, oi = ti[order], oi[order]
    cuts = np.searchsorted(ti, np.arange(len(index) + 1))
    wkb = [list(shapely.to_wkb(T.to_px(geoms[oi[cuts[i]:cuts[i + 1]]],
                                       index.iloc[i]["affine"])))
           for i in range(len(index))]
    reg = cfg["registration"]
    items = [(directory, index.iloc[i].to_dict(), wkb[i], reg) for i in range(len(index))]
    out = _pool_map(_register_worker, items, workers)
    raw = pd.DataFrame(out, columns=["raw_dx", "raw_dy", "peak_ratio", "reason"])
    raw.insert(0, "tile_id", index["tile_id"].to_numpy())
    eligible = (index["domain"].isin(cfg["tiles"]["train_domains"]).to_numpy()
                & ~index["low_valid"].to_numpy(dtype=bool)
                & np.array([len(w) > 0 for w in wkb]))
    null = _osm_null(index, directory, wkb, reg, eligible, workers)
    threshold = float(np.nanquantile(null["peak_ratio"], float(reg["min_peak_ratio"]
                                                               ["null_quantile"])))
    shifts = decide_registration(raw, reg, threshold)
    shifts.insert(1, "n_osm", [len(w) for w in wkb])
    shifts = shifts.merge(index[["tile_id", "domain", "low_valid"]], on="tile_id")
    pixel = float(index.iloc[0]["affine"][0])
    null_mag = np.hypot(null["dx"], null["dy"])
    passing = null["peak_ratio"] >= threshold
    caps = sorted({12.0, float(reg["max_shift_px"])})   # 12 px = the plan's original cap
    summary: dict[str, Any] = {
        "min_peak_ratio": threshold, "rule": reg["min_peak_ratio"],
        "max_shift_px": float(reg["max_shift_px"]),
        "null_pairs_used": int(np.isfinite(null["peak_ratio"]).sum()),
        "null_accept_rate_at_cap_px": {f"{c:g}": float((passing & (null_mag <= c)).mean())
                                       for c in caps},
        "by_domain": {}}
    groups = [("all", shifts), *((str(d), p) for d, p in shifts.groupby("domain"))]
    for domain, part in groups:
        ok = part.loc[part["reg_ok"]]
        mag = np.hypot(ok["dx"].to_numpy(), ok["dy"].to_numpy()) * pixel
        summary["by_domain"][domain] = {
            "tiles": len(part), "reasons": part["reason"].value_counts().to_dict(),
            "registered_share": float(part["reg_ok"].mean()),
            "dx_px_q05_q25_q50_q75_q95": _q(ok["dx"].to_numpy()),
            "dy_px_q05_q25_q50_q75_q95": _q(ok["dy"].to_numpy()),
            "shift_m_median": float(np.median(mag)) if len(mag) else None,
        }
    return shifts, summary


def _osm_null_worker(args: tuple[Any, ...]) -> tuple[float, float, float]:
    from nakkhu.labels.register import peak_and_ratio

    directory, row, wkb, reg = args
    from nakkhu.labels.register import boundary_raster, correlation_surface, gradient_magnitude

    rgb, valid = T.read_tile(directory, row)
    polys = shapely.from_wkb(np.asarray(wkb, dtype=object))
    boundary = boundary_raster(polys, rgb.shape[:2]) * valid
    if boundary.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    surface = correlation_surface(gradient_magnitude(rgb, valid), boundary,
                                  float(reg["sigma_px"]))
    dy, dx, ratio = peak_and_ratio(surface, int(reg["exclusion_px"]))
    return ratio, dx, dy


def _osm_null(index: pd.DataFrame, directory: Path, wkb: list[list[bytes]], reg: dict[str, Any],
              eligible: np.ndarray, workers: int) -> pd.DataFrame:
    """Mismatched (image, OSM) pairs drawn exactly like the footprint null (same seed and
    sampling as ``nakkhu.labels.build._null_threshold``), keeping the peak position too."""
    from nakkhu.labels.build import _pool_map

    rule = reg["min_peak_ratio"]
    rng = np.random.default_rng(int(rule["seed"]))
    pool = np.flatnonzero(eligible)
    n = int(rule["null_pairs"])
    image_i = rng.choice(pool, size=n, replace=True)
    offset = rng.integers(1, len(pool), size=n)
    fp_i = pool[(np.searchsorted(pool, image_i) + offset) % len(pool)]
    items = [(directory, index.iloc[a].to_dict(), wkb[b], reg)
             for a, b in zip(image_i, fp_i, strict=True)]
    out = np.array(_pool_map(_osm_null_worker, items, workers), dtype=float).reshape(-1, 3)
    return pd.DataFrame({"peak_ratio": out[:, 0], "dx": out[:, 1], "dy": out[:, 2]})


def _q(values: np.ndarray) -> list[float] | None:
    if not len(values):
        return None
    return [float(v) for v in np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])]


def load_osm_shifts(project: Project) -> pd.DataFrame:
    """Per-tile OSM shifts (``tile_id, dx, dy`` in tile pixels, ``reg_ok``, ``reason``)."""
    path = osm_shift_path(project)
    if not path.is_file():
        raise LabelQAError("OSM is not registered yet: run `nakkhu labels osm-register`")
    return pd.read_parquet(path)


def shift_map(geoms: np.ndarray, affine: Any, dx: float, dy: float) -> np.ndarray:
    """Shift map polygons by a tile-pixel shift (x right, y down) of a north-up tile."""
    a, e = float(list(affine)[0]), float(list(affine)[4])
    return shapely.transform(np.asarray(geoms, dtype=object),
                             lambda xy: xy + np.array([dx * a, dy * e]))


def aligned_osm(project: Project, index: pd.DataFrame, clip: Any | None = None,
                drop_greenhouses: bool = True) -> gpd.GeoDataFrame:
    """OSM buildings aligned to the imagery: each polygon moved by the shift of the plan 003
    train-grid tile containing its centroid (entry point for plan 005 scoring). Greenhouses are
    removed by default (shared rule, ``nakkhu.reference.greenhouse``)."""
    shifts = load_osm_shifts(project).set_index("tile_id")
    osm = load_osm(project, clip, drop_greenhouses)
    centroids = shapely.centroid(osm.geometry.values)
    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    ci, ti = shapely.STRtree(boxes).query(centroids, predicate="within")
    tile_of = np.full(len(osm), -1)
    tile_of[ci[::-1]] = ti[::-1]          # first containing tile wins
    geoms = osm.geometry.values.copy()
    for t in np.unique(tile_of[tile_of >= 0]):
        row = index.iloc[int(t)]
        if row["tile_id"] not in shifts.index:
            continue
        sel = tile_of == t
        s = shifts.loc[row["tile_id"]]
        geoms[sel] = shift_map(geoms[sel], row["affine"], float(s["dx"]), float(s["dy"]))
    out = osm.copy()
    out["geometry"] = geoms
    out["osm_tile"] = [index.iloc[int(t)]["tile_id"] if t >= 0 else None for t in tile_of]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=osm.crs)


def tile_label_quality(project: Project, arm_dir: Path, index: pd.DataFrame,
                       osm: gpd.GeoDataFrame, shifts: pd.DataFrame, domains: set[str],
                       tile_px: int, iou_thresholds: tuple[float, ...] = (0.3, 0.5),
                       access: tuple[str, str] | None = None) -> pd.DataFrame:
    """Per tile: TP/FP/FN and F1 (= label quality) of an arm's labels against OSM aligned with
    that tile's OSM shift (ignore pixels excluded).

    OSM buildings are counted for a tile when >= 50 % of their aligned area is inside it and
    their centroid is not in the arm's ignore region; each is compared as its minimum rotated
    rectangle (the prereg R-OSM geometry) with the arm's OBB labels, rotated-polygon IoU.
    """
    if "osm_tile" in osm.columns:   # marker column of aligned_osm output
        raise LabelQAError("pass unshifted OSM (load_osm): the tile's OSM shift is applied here")
    _guard_split(project, domains, access)
    osm_geoms = osm.geometry.values
    tree = shapely.STRtree(osm_geoms)
    shift = shifts.set_index("tile_id")
    rows = []
    for _, row in index.loc[index["domain"].isin(domains)].iterrows():
        path = arm_dir / "labels" / f"{row['tile_id']}.txt"
        if not path.is_file() or row["tile_id"] not in shift.index:
            continue
        _, corners = read_yolo_obb(path, tile_px)
        preds = T.to_map(np.array([shapely.Polygon(c) for c in corners], dtype=object),
                         row["affine"])
        tile = shapely.box(*row["bounds"])
        s = shift.loc[row["tile_id"]]
        refs = osm_geoms[tree.query(shapely.buffer(tile, 10.0), predicate="intersects")]
        refs = shift_map(refs, row["affine"], float(s["dx"]), float(s["dy"]))
        if len(refs):
            inside = shapely.area(shapely.intersection(refs, tile)) / shapely.area(refs)
            # R-OSM geometry (prereg): oriented envelope of each aligned OSM polygon, so boxes
            # are compared with boxes.
            refs = shapely.minimum_rotated_rectangle(refs[inside >= 0.5])
        ignore_path = arm_dir / "ignore" / f"{row['tile_id']}.png"
        if len(refs) and ignore_path.is_file():
            ignore = np.asarray(Image.open(ignore_path).convert("L")) > 0
            cpx = T.to_px(shapely.centroid(refs), row["affine"])
            xy = np.clip(np.floor(shapely.get_coordinates(cpx)).astype(int), 0, tile_px - 1)
            refs = refs[~ignore[xy[:, 1], xy[:, 0]]]
        rec: dict[str, Any] = {"tile_id": row["tile_id"], "domain": row["domain"],
                               "osm_reg_ok": bool(s["reg_ok"]),
                               "n_labels": len(preds), "n_osm": len(refs)}
        for thr in iou_thresholds:
            tp, fp, fn = greedy_match(preds, refs, thr)
            key = f"{int(round(thr * 100)):02d}"
            rec.update({f"tp_{key}": tp, f"fp_{key}": fp, f"fn_{key}": fn,
                        f"quality_{key}": f1(tp, fp, fn)})
        rows.append(rec)
    return pd.DataFrame(rows)


def g1(scores: pd.DataFrame, errors: pd.DataFrame, fractions: np.ndarray,
       iou_key: str = "50") -> tuple[dict[str, Any], pd.DataFrame]:
    """G1 statistic: Spearman rho(gate score, label quality) (pass: rho >= 0.3 on test tiles)
    and pooled F1 against the fraction of lowest-scoring tiles rejected."""
    merged = scores[["tile_id", "gate_score"]].merge(errors, on="tile_id")
    merged = merged.loc[(merged["n_labels"] + merged["n_osm"]) > 0]
    rho = spearman(merged["gate_score"], merged[f"quality_{iou_key}"])
    ordered = merged.sort_values(["gate_score", "tile_id"], ascending=[False, True])
    curve = []
    for frac in fractions:
        n_keep = int(np.ceil((1 - frac) * len(ordered)))
        part = ordered.iloc[:n_keep]
        tp, fp, fn = (int(part[f"{k}_{iou_key}"].sum()) for k in ("tp", "fp", "fn"))
        curve.append({"rejected_fraction": float(frac), "tiles": n_keep, "tp": tp, "fp": fp,
                      "fn": fn, "f1": f1(tp, fp, fn)})
    return {"spearman_rho": rho, "tiles": len(merged), "iou": iou_key}, pd.DataFrame(curve)


# --- G1 confirmatory: gate score vs audit-based label quality --------------------------------


def g1_audit_quality(items: pd.DataFrame, answers: dict[str, str],
                     clicked: dict[str, tuple[float, float]], tiles: pd.DataFrame,
                     obbs: dict[str, np.ndarray], min_answered: int = 5,
                     tolerance_m: float = 1.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-tile audit quality of the labels (prereg ``falsifiers.G1.confirmatory``).

    ``items`` are the recall_points items (``item_id, stratum, frame_n, x, y``), ``answers``
    their protocol-v2 answers and ``clicked`` the clicked roof point of each near-miss Yes.
    The ``outside`` stratum is excluded. A unit's point is its frame point, or the clicked roof
    point for a near-miss Yes; it belongs to the tile (``tile_id, bounds``) containing that
    point. Design weights w = N_h / n_h come from the answered (Yes/No) units of the whole
    sample (:func:`nakkhu.audit.estimators.answered`), Unsure excluded. Audit quality of a tile =
    sum(w y d) / sum(w y) over its answered units, d = 1 if the point lies within
    ``tolerance_m`` of one of the tile's label OBBs (``obbs[tile_id]``, map coordinates);
    NaN without a Yes unit. ``eligible`` = at least ``min_answered`` answered units.
    """
    from nakkhu.audit.estimators import answered

    units = items.loc[items["stratum"] != "outside"].copy()
    xy = units[["x", "y"]].to_numpy(dtype=float).copy()
    for i, item in enumerate(units["item_id"]):
        if item in clicked:
            xy[i] = clicked[item]
    units["px"], units["py"] = xy[:, 0], xy[:, 1]
    frame = answered(units.drop(columns=["x", "y"]), answers)   # its "y" is the Yes flag
    boxes = np.array([shapely.box(*b) for b in tiles["bounds"]], dtype=object)
    pts = shapely.points(frame[["px", "py"]].to_numpy(dtype=float))
    pi, ti = shapely.STRtree(boxes).query(pts, predicate="intersects")
    order = np.lexsort((ti, pi))
    tile_of = np.full(len(frame), -1)
    tile_of[pi[order][::-1]] = ti[order][::-1]       # lowest tile index wins on an edge
    frame["tile_id"] = [tiles["tile_id"].iloc[t] if t >= 0 else None for t in tile_of]
    hit = np.zeros(len(frame), dtype=bool)
    for tile, rows in frame.groupby("tile_id").indices.items():
        geoms = np.asarray(obbs.get(tile, np.array([], dtype=object)), dtype=object)
        if len(geoms):
            near, _ = shapely.STRtree(geoms).query(pts[rows], predicate="dwithin",
                                                   distance=tolerance_m)
            hit[rows[np.unique(near)]] = True
    frame["hit"] = hit
    rows = []
    for tile_id, g in frame.loc[frame["tile_id"].notna()].groupby("tile_id"):
        wy = (g["w"] * g["y"]).to_numpy(dtype=float)
        den = float(wy.sum())
        rows.append({"tile_id": tile_id, "answered": len(g), "yes": int(g["y"].sum()),
                     "yes_hit": int((g["y"] * g["hit"]).sum()),
                     "w_yes": den,
                     "audit_quality": float((wy * g["hit"].to_numpy()).sum()) / den
                     if den > 0 else float("nan"),
                     "eligible": len(g) >= min_answered})
    table = pd.DataFrame(rows, columns=["tile_id", "answered", "yes", "yes_hit", "w_yes",
                                        "audit_quality", "eligible"])
    counts = {"units": len(units), "answered_units": len(frame),
              "answered_units_in_tiles": int(frame["tile_id"].notna().sum()),
              "answered_units_outside_tiles": int(frame["tile_id"].isna().sum()),
              "clicked_points_used": int(frame["item_id"].isin(set(clicked)).sum())}
    return table, counts


def g1_confirmatory(scores: pd.DataFrame, quality: pd.DataFrame) -> dict[str, Any]:
    """Spearman rho(gate score, audit quality) over the eligible tiles with a Yes unit."""
    merged = quality.loc[quality["eligible"]].merge(scores[["tile_id", "gate_score"]],
                                                    on="tile_id", how="left")
    if merged["gate_score"].isna().any():
        raise LabelQAError("eligible tile(s) without a gate score: "
                           + ", ".join(merged.loc[merged["gate_score"].isna(), "tile_id"]))
    used = merged.loc[np.isfinite(merged["audit_quality"].to_numpy(dtype=float))]
    return {"spearman_rho": spearman(used["gate_score"], used["audit_quality"]),
            "tiles": len(used), "eligible_tiles": len(merged),
            "eligible_tiles_without_yes": int(len(merged) - len(used))}
