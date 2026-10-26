"""Scoring detections against aligned OSM on val or test blocks (plan 005 Steps 3-5).

Region membership: an object (prediction or reference) belongs to the region when its centroid
lies in it. Matching runs over the objects whose centroid lies within the region buffered by
``margin_m``, so a pair straddling the region edge is matched once; precision counts in-region
predictions and recall in-region references. Units: each in-region object is assigned by
centroid to its plan 003 train-grid tile and that tile's block (``split_v1.yaml``).

References are aligned OSM (:func:`nakkhu.labelqa.aligned_osm`) reduced to their oriented
envelope (minimum rotated rectangle), the geometry the detector predicts. Numbers are always
"against OSM": OSM completeness is unknown, so unmatched predictions are not all false.

Run contract (plan 007): ``runs/<stage>/<id>/detections.gpkg`` layer ``obb`` with ``conf`` and
OBB polygons in EPSG:32645 (``nakkhu georef``), optional ``inputs.json`` with ``model`` (key
shared by the val and test inference runs of one trained model) or ``inventory: true`` (a fixed
footprint product scored as-is with ``--fixed``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from nakkhu.config import Project, ResolvedConfig
from nakkhu.evaluate import bootstrap as B
from nakkhu.evaluate import ci, guard
from nakkhu.evaluate.match import iou_pairs, match_by_confidence
from nakkhu.evaluate.metrics import (
    AP_IOUS,
    ap_coco101,
    count_errors,
    f1_max_threshold,
    prf,
)

DEFAULTS: dict[str, Any] = {"iou_primary": 0.5, "iou_secondary": 0.3, "margin_m": 20.0,
                            "n_boot": 2000, "seed": 0, "level": 0.95,
                            "threshold_iou": 0.5}
DETECTIONS = "detections.gpkg"


class EvalError(RuntimeError):
    """A run cannot be scored under the evaluation rules."""


def _key(iou: float) -> str:
    return f"{int(round(iou * 100)):02d}"


@dataclass
class Scored:
    metrics: dict[str, Any]
    per_block: pd.DataFrame
    per_tile: pd.DataFrame
    preds: pd.DataFrame


def unit_of(geoms: np.ndarray, tiles: pd.DataFrame) -> np.ndarray:
    """Tile id of each geometry's centroid (``None`` outside every tile; first tile wins)."""
    out = np.full(len(geoms), None, dtype=object)
    if len(geoms) == 0 or tiles.empty:
        return out
    boxes = np.array([shapely.box(*b) for b in tiles["bounds"]], dtype=object)
    gi, ti = shapely.STRtree(boxes).query(shapely.centroid(geoms), predicate="intersects")
    first = pd.DataFrame({"g": gi, "t": ti}).sort_values(["g", "t"]).drop_duplicates("g")
    out[first["g"].to_numpy()] = tiles["tile_id"].to_numpy()[first["t"].to_numpy()]
    return out


def score(pred_geoms: np.ndarray, conf: np.ndarray, ref_geoms: np.ndarray, region: Any,
          tiles: pd.DataFrame, *, threshold: float | None, params: dict[str, Any] | None = None,
          ap_ci: bool = True) -> Scored:
    """Match, count and bootstrap one prediction set against one reference set.

    ``tiles``: ``tile_id, bounds, block_id`` of the scored tiles (units). ``threshold``: keep
    predictions with ``conf >= threshold`` for P/R/F1 and counts (None = all); AP never uses it.
    """
    p = {**DEFAULTS, **(params or {})}
    ious = sorted({float(p["iou_primary"]), float(p["iou_secondary"]), *AP_IOUS})
    pred_geoms = np.asarray(pred_geoms, dtype=object)
    ref_geoms = np.asarray(ref_geoms, dtype=object)
    conf = np.asarray(conf, dtype=float)
    shapely.prepare(region)
    wide = shapely.buffer(region, float(p["margin_m"]))
    shapely.prepare(wide)
    pc, rc = shapely.centroid(pred_geoms), shapely.centroid(ref_geoms)
    pk = shapely.intersects(wide, pc) if len(pc) else np.zeros(0, dtype=bool)
    rk = shapely.intersects(wide, rc) if len(rc) else np.zeros(0, dtype=bool)
    pred_geoms, conf, pc = pred_geoms[pk], conf[pk], pc[pk]
    ref_geoms, rc = ref_geoms[rk], rc[rk]
    p_in = shapely.intersects(region, pc) if len(pc) else np.zeros(0, dtype=bool)
    r_in = shapely.intersects(region, rc) if len(rc) else np.zeros(0, dtype=bool)
    pairs = iou_pairs(pred_geoms, ref_geoms)
    t = -np.inf if threshold is None else float(threshold)
    sel = p_in & (conf >= t)
    block_of_tile = tiles.set_index("tile_id")["block_id"]
    p_tile = unit_of(pred_geoms, tiles)
    r_tile = unit_of(ref_geoms, tiles)
    p_block = np.array([block_of_tile.get(x, "none") if x is not None else "none"
                        for x in p_tile], dtype=object)
    r_block = np.array([block_of_tile.get(x, "none") if x is not None else "none"
                        for x in r_tile], dtype=object)
    preds = pd.DataFrame({"tile_id": p_tile, "block_id": p_block, "conf": conf,
                          "in_region": p_in, "kept": sel})
    refs = pd.DataFrame({"tile_id": r_tile, "block_id": r_block, "in_region": r_in})
    for iou in ious:
        p2r, r2p = match_by_confidence(conf, len(ref_geoms), pairs, iou)
        preds[f"tp_{_key(iou)}"] = p2r >= 0
        matched_conf = np.where(r2p >= 0, conf[np.maximum(r2p, 0)], -np.inf)
        refs[f"tp_{_key(iou)}"] = (r2p >= 0) & (matched_conf >= t)

    blocks = sorted(set(tiles["block_id"]))
    pin, rin = preds.loc[preds["in_region"]], refs.loc[refs["in_region"]]
    kept = pin.loc[pin["kept"]]
    per_block = pd.DataFrame({"block_id": blocks})
    per_block["n_pred"] = per_block["block_id"].map(kept.groupby("block_id").size()).fillna(0)
    per_block["n_ref"] = per_block["block_id"].map(rin.groupby("block_id").size()).fillna(0)
    per_block["n_pred_all"] = per_block["block_id"].map(pin.groupby("block_id").size()
                                                        ).fillna(0)
    for iou in (p["iou_primary"], p["iou_secondary"]):
        k = _key(iou)
        per_block[f"tp_pred_{k}"] = per_block["block_id"].map(
            kept.groupby("block_id")[f"tp_{k}"].sum()).fillna(0)
        per_block[f"tp_ref_{k}"] = per_block["block_id"].map(
            rin.groupby("block_id")[f"tp_{k}"].sum()).fillna(0)
    per_tile = tiles[["tile_id", "block_id"]].copy()
    per_tile["n_pred"] = per_tile["tile_id"].map(kept.groupby("tile_id").size()).fillna(0)
    per_tile["n_ref"] = per_tile["tile_id"].map(rin.groupby("tile_id").size()).fillna(0)
    per_tile["abs_err"] = (per_tile["n_pred"] - per_tile["n_ref"]).abs()
    per_tile["err"] = per_tile["n_pred"] - per_tile["n_ref"]
    tb = per_tile.groupby("block_id").agg(abs_err=("abs_err", "sum"), err=("err", "sum"),
                                          n_tiles=("tile_id", "size"))
    for col in ("abs_err", "err", "n_tiles"):
        per_block[col] = per_block["block_id"].map(tb[col]).fillna(0)
    outside = int((pin["block_id"] == "none").sum() + (rin["block_id"] == "none").sum())

    n_ref = float(len(rin))
    metrics: dict[str, Any] = {
        "threshold": None if threshold is None else float(threshold),
        "n_pred_in_region": len(pin), "n_pred_kept": len(kept), "n_ref": len(rin),
        "blocks": len(blocks), "tiles": len(tiles), "objects_outside_units": outside,
        "reference": "aligned OSM (oriented envelope); metrics are against OSM",
    }
    W = B.multiplicities(len(blocks), int(p["n_boot"]), int(p["seed"]))
    cols = {c: per_block[c].to_numpy(dtype=float) for c in per_block.columns if c != "block_id"}
    level = float(p["level"])
    for iou in (p["iou_primary"], p["iou_secondary"]):
        k = _key(iou)
        point = prf(float(kept[f"tp_{k}"].sum()), float(len(kept)),
                    float(rin[f"tp_{k}"].sum()), n_ref)
        f1s = B.additive(cols, W, B.f1_stat(f"tp_pred_{k}", "n_pred", f"tp_ref_{k}", "n_ref"))
        ps = B.additive(cols, W, B.ratio_stat(f"tp_pred_{k}", "n_pred"))
        rs = B.additive(cols, W, B.ratio_stat(f"tp_ref_{k}", "n_ref"))
        metrics[f"iou_{k}"] = {"f1": B.summary(point["f1"], f1s, level),
                               "precision": B.summary(point["precision"], ps, level),
                               "recall": B.summary(point["recall"], rs, level)}
    block_index = {b: i for i, b in enumerate(blocks)}
    pb = np.array([block_index.get(b, -1) for b in pin["block_id"]], dtype=np.int64)
    n_ref_block = per_block["n_ref"].to_numpy(dtype=float)
    pin_conf = pin["conf"].to_numpy()
    ap: dict[str, Any] = {}
    for iou in AP_IOUS:
        k = _key(iou)
        ap[k] = ap_coco101(pin_conf, pin[f"tp_{k}"].to_numpy(), n_ref)
    in_unit = pb >= 0
    ap50_s = (B.ap_replicates(pin_conf[in_unit], pin["tp_50"].to_numpy()[in_unit],
                              pb[in_unit], n_ref_block, W) if ap_ci else np.array([]))
    metrics["ap50"] = B.summary(ap["50"], ap50_s, level)
    metrics["ap50_95"] = {"value": float(np.mean([ap[_key(i)] for i in AP_IOUS])),
                          "per_iou": ap}
    counts = count_errors(per_tile["n_pred"].to_numpy(), per_tile["n_ref"].to_numpy())
    mae_s = B.additive(cols, W, B.ratio_stat("abs_err", "n_tiles"))
    bias_s = B.additive(cols, W, B.ratio_stat("err", "n_tiles"))
    metrics["tile_count"] = {"mae": B.summary(counts["mae"], mae_s, level),
                             "bias": B.summary(counts["bias"], bias_s, level),
                             "units": counts["units"]}
    metrics["bootstrap"] = {"unit": "block", "n_boot": int(p["n_boot"]), "seed": int(p["seed"]),
                            "level": level, "interval": "percentile"}
    return Scored(metrics=metrics, per_block=per_block, per_tile=per_tile,
                  preds=preds.loc[preds["in_region"]].reset_index(drop=True))


def per_block_f1(per_block: pd.DataFrame, key: str = "50") -> np.ndarray:
    tp_p, n_p = per_block[f"tp_pred_{key}"].to_numpy(float), per_block["n_pred"].to_numpy(float)
    tp_r, n_r = per_block[f"tp_ref_{key}"].to_numpy(float), per_block["n_ref"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp_p / n_p, tp_r / n_r
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return np.where((n_p > 0) & (n_r > 0), f1, np.nan)


def compare_blocks(a: pd.DataFrame, b: pd.DataFrame, params: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
    """Paired block bootstrap of F1 differences (a - b) on the same replicates, plus a
    Wilcoxon signed-rank test on per-block F1 (secondary)."""
    p = {**DEFAULTS, **(params or {})}
    if list(a["block_id"]) != list(b["block_id"]):
        raise EvalError("the two evaluations have different blocks")
    W = B.multiplicities(len(a), int(p["n_boot"]), int(p["seed"]))
    out: dict[str, Any] = {"bootstrap": {"unit": "block", "n_boot": int(p["n_boot"]),
                                         "seed": int(p["seed"])}}
    for iou in (p["iou_primary"], p["iou_secondary"]):
        k = _key(iou)
        stat = B.f1_stat(f"tp_pred_{k}", "n_pred", f"tp_ref_{k}", "n_ref")
        fa = B.additive({c: a[c].to_numpy(float) for c in a.columns if c != "block_id"}, W, stat)
        fb = B.additive({c: b[c].to_numpy(float) for c in b.columns if c != "block_id"}, W, stat)
        pa = float(stat({c: np.array([a[c].sum()]) for c in a.columns if c != "block_id"})[0])
        pb_ = float(stat({c: np.array([b[c].sum()]) for c in b.columns if c != "block_id"})[0])
        out[f"delta_f1_{k}"] = B.summary(pa - pb_, fa - fb, float(p["level"]))
        out[f"wilcoxon_block_f1_{k}"] = ci.wilcoxon_paired(per_block_f1(a, k),
                                                           per_block_f1(b, k))
    return out


# --- run-level entry points -----------------------------------------------------------------


def _run_dir(project: Project, ref: str, default_stage: str) -> tuple[str, Path]:
    ref = ref.replace("\\", "/").strip("/")
    if ref.startswith("runs/"):
        ref = ref[len("runs/"):]
    stage, _, run_id = ref.rpartition("/")
    stage = stage or default_stage
    path = project.paths.runs / stage / run_id
    if not path.is_dir():
        raise EvalError(f"run not found: {path}")
    return f"{stage}/{run_id}", path


def _inputs(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "inputs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def eval_params(prereg: dict[str, Any] | None) -> dict[str, Any]:
    if not prereg:
        return dict(DEFAULTS)
    m, u = prereg["matching"], prereg["uncertainty"]["block_bootstrap"]
    return {"iou_primary": float(m["iou_primary"]), "iou_secondary": float(m["iou_secondary"]),
            "margin_m": float(m["region_margin_m"]), "n_boot": int(u["n_resamples"]),
            "seed": int(u["seed"]), "level": float(u["level"]),
            "threshold_iou": float(prereg["conf_threshold"]["iou"])}


def region_and_tiles(project: Project, split: str) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Scored region, the unit tiles (``tile_id, bounds, block_id``) and the full index."""
    from nakkhu.acquire import geo
    from nakkhu.splits import load_split
    from nakkhu.tiling import read_index

    frame, header = load_split(project)
    index, _ = read_index(project.paths.resolve(header["tile_index"]))
    index = index.sort_values("tile_id").reset_index(drop=True)
    members = frame.loc[frame["split"] == split, ["tile_id", "block_id"]]
    tiles = members.merge(index[["tile_id", "bounds"]], on="tile_id")
    boxes = np.array([shapely.box(*b) for b in tiles["bounds"]], dtype=object)
    union = shapely.union_all(boxes)
    region = shapely.intersection(geo.extent(project), union) if split == "test" else union
    return region, tiles.sort_values("tile_id").reset_index(drop=True), index


def reference_geoms(project: Project, index: pd.DataFrame, region: Any, margin_m: float
                    ) -> tuple[np.ndarray, pd.DataFrame]:
    from nakkhu.labelqa import aligned_osm, load_osm_shifts

    clip = shapely.buffer(region, margin_m + 50.0)
    osm = aligned_osm(project, index, clip)
    geoms = shapely.oriented_envelope(osm.geometry.values)
    return np.asarray(geoms, dtype=object), load_osm_shifts(project)


def load_detections(run_dir: Path) -> gpd.GeoDataFrame:
    path = run_dir / DETECTIONS
    if not path.is_file():
        raise EvalError(f"{path} is missing (plan 007 infer + `nakkhu georef`)")
    frame = gpd.read_file(path, layer="obb")
    if "conf" not in frame:
        raise EvalError(f"{path}: layer obb has no conf column")
    if frame.crs is not None and frame.crs.to_epsg() != 32645:
        frame = frame.to_crs("EPSG:32645")
    return frame


def _threshold_from(project: Project, ref: str, model: str, split_sha: str) -> dict[str, Any]:
    _, path = _run_dir(project, ref, "eval")
    file = path / "threshold.json"
    if not file.is_file():
        raise EvalError(f"{ref} has no threshold.json (it must be a val evaluation)")
    record = json.loads(file.read_text(encoding="utf-8"))
    if record.get("split") != "val":
        raise EvalError(f"{ref} is not a val evaluation")
    if record.get("model") != model:
        raise EvalError(f"threshold of model {record.get('model')!r}, not {model!r}")
    if record.get("split_sha256") != split_sha:
        raise EvalError("the val threshold was chosen on another split file")
    return record


def evaluate_run(project: Project, run_ref: str, split: str, *,
                 threshold_from: str | None = None, fixed: bool = False,
                 ap_ci: bool = True) -> dict[str, Any]:
    """``nakkhu eval --run <ref> --split val|test``: guard, journal (test), score, write."""
    from nakkhu.runs import atomic_write_json, create_run

    state = guard.check_access(project, split)        # raises for test unless frozen
    prereg = guard.load_prereg(project) if Path(project.paths.resolve(guard.PREREG)).is_file() \
        else None
    params = eval_params(prereg)
    ref, run_dir = _run_dir(project, run_ref, "infer")
    inputs = _inputs(run_dir)
    model = str(inputs.get("model") or ref)
    split_sha = guard.sha256_file(project.paths.resolve(guard.SPLIT_FILE))
    threshold_record: dict[str, Any] | None = None
    if split == "test":
        if fixed and not inputs.get("inventory"):
            raise EvalError("--fixed on test is only for fixed inventories (inputs.json "
                            "`inventory: true`); detectors use their val threshold")
        if not fixed:
            if not threshold_from:
                raise EvalError("test scoring of a detector needs --threshold-from <val eval>")
            threshold_record = _threshold_from(project, threshold_from, model, split_sha)
    values = {"command": "eval", "run": ref, "split": split, "model": model,
              "threshold_from": threshold_from, "fixed": fixed, "params": params,
              "prereg_sha256": state.get("prereg_sha256"), "split_sha256": split_sha}
    config = ResolvedConfig(values=values, source=guard.PREREG)
    access = None
    if split == "test":
        access = guard.record_test_access(project, ref, config.cfghash,
                                          state["prereg_sha256"], split_sha)
    run = create_run(project, "eval", config)
    with run.step("score"):
        region, tiles, index = region_and_tiles(project, split)
        refs, shifts = reference_geoms(project, index, region, params["margin_m"])
        dets = load_detections(run_dir)
        geoms, conf = np.asarray(dets.geometry.values, dtype=object), dets["conf"].to_numpy()
        if split == "val" and not fixed:
            first = score(geoms, conf, refs, region, tiles, threshold=None, params=params,
                          ap_ci=False)
            pin = first.preds
            n_ref = first.metrics["n_ref"]
            choice = f1_max_threshold(pin["conf"].to_numpy(),
                                      pin[f"tp_{_key(params['threshold_iou'])}"].to_numpy(),
                                      n_ref)
            threshold: float | None = choice["threshold"]
            threshold_record = {"split": "val", "model": model, "run": ref,
                                "split_sha256": split_sha, "iou": params["threshold_iou"],
                                "rule": "F1-max against aligned OSM on val", **choice}
            atomic_write_json(run.path / "threshold.json", threshold_record)
        elif threshold_record is not None:
            threshold = float(threshold_record["threshold"])
        else:
            threshold = None
        main = score(geoms, conf, refs, region, tiles, threshold=threshold, params=params,
                     ap_ci=ap_ci)
        registered = set(shifts.loc[shifts["reg_ok"], "tile_id"])
        sub_tiles = tiles.loc[tiles["tile_id"].isin(registered)].reset_index(drop=True)
        sub = None
        if len(sub_tiles):
            sub_boxes = np.array([shapely.box(*b) for b in sub_tiles["bounds"]], dtype=object)
            sub_region = shapely.intersection(region, shapely.union_all(sub_boxes))
            sub = score(geoms, conf, refs, sub_region, sub_tiles, threshold=threshold,
                        params=params, ap_ci=False)
        metrics = {"run": ref, "model": model, "split": split, "params": params,
                   "threshold_source": ("fixed inventory" if fixed else
                                        "val F1-max" if split == "val" else threshold_from),
                   "prereg_sha256": state.get("prereg_sha256"), "split_sha256": split_sha,
                   "all_tiles": main.metrics,
                   "osm_registered_tiles": None if sub is None else sub.metrics}
        if access is not None:
            metrics["test_access"] = access
        atomic_write_json(run.path / "metrics.json", metrics)
        main.per_block.to_csv(run.path / "per_block.csv", index=False)
        main.per_tile.to_csv(run.path / "per_tile.csv", index=False)
        main.preds.to_parquet(run.path / "preds.parquet", index=False)
    result = {"eval_run": run.ref, "split": split, "threshold": threshold,
              "f1_50": main.metrics["iou_50"]["f1"]["value"],
              "flagged": bool(access and access["flagged"])}
    run.write_summary(result)
    return result


def compare_runs(project: Project, a: str, b: str) -> dict[str, Any]:
    """Paired comparison of two eval runs on the same split and blocks."""
    from nakkhu.runs import create_run

    _, da = _run_dir(project, a, "eval")
    _, db = _run_dir(project, b, "eval")
    ma = json.loads((da / "metrics.json").read_text(encoding="utf-8"))
    mb = json.loads((db / "metrics.json").read_text(encoding="utf-8"))
    for key in ("split", "split_sha256"):
        if ma[key] != mb[key]:
            raise EvalError(f"evaluations differ in {key}")
    params = ma["params"]
    out = compare_blocks(pd.read_csv(da / "per_block.csv"), pd.read_csv(db / "per_block.csv"),
                         params)
    out.update({"a": a, "b": b, "split": ma["split"]})
    run = create_run(project, "eval", ResolvedConfig(values={"command": "compare", "a": a,
                                                             "b": b}, source=guard.PREREG))
    with run.step("compare"):
        run.write_summary(out)
    return {"compare_run": run.ref, **out}
