"""Final detector inventory ``detector_final`` (plan 007 "Final"): 5-seed clustered OBBs.

The prereg fixes the final seeds (``seeds.final``: 0-4), the per-model confidence rule
(``conf_threshold``: F1-max against aligned OSM on val, ties to the higher threshold) and the
detection post-processing, but not how the seeds are combined. The rule below was set on val
only (orchestrator decision log 2026-10-12) and reuses registered values wherever one exists:

1. Members: each seed's georeferenced detections (centre ownership + cross-tile NMS, prereg
   ``detection_postprocess``) at ``conf >=`` that seed's frozen val threshold (its own
   ``threshold.json``, the prereg rule per trained model).
2. Clustering: greedy, one box per seed per cluster. Boxes are visited in descending confidence
   (stable); a box joins the open cluster whose anchor (its first, highest-confidence box) has
   the highest rotated IoU with it, provided that IoU >= ``iou_min`` (0.5 = prereg
   ``matching.iou_primary`` and the audit ``cluster_iou``) and the cluster has no box of the
   same seed yet; otherwise it opens a new cluster. Unlike single linkage this cannot chain
   neighbouring row houses into one cluster.
3. Geometry: the cluster medoid (the member with the largest summed IoU to the other members;
   ties: higher confidence, then input order), an unmodified detector box.
4. Score: ``conf`` = sum of the members' confidences / number of seeds (an absent seed adds 0),
   so it rises with both seed agreement and confidence; ``conf_s<k>`` keep each seed's
   confidence (NaN when absent) and ``n_seeds`` the support.
5. Cut: the ensemble is one model for the prereg ``conf_threshold`` rule: the val ensemble run
   is scored with ``nakkhu eval --split val`` (F1-max on val), and ``detector_final`` = the
   test-region clusters with ``conf >=`` that threshold and centroid in the analysis extent.

``nakkhu detector final`` builds the val and test ensemble runs (``runs/infer/<id>/`` with
``inputs.json`` ``ensemble: true`` and one ``model`` key, so the guarded test scoring takes the
threshold from the val eval like any detector), scores only val, and writes the inventory. No
test metric is computed here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely

from nakkhu.config import Project, ResolvedConfig

WORK_CRS = "EPSG:32645"
IOU_MIN = 0.5


class EnsembleError(RuntimeError):
    """The final inventory cannot be built under the rule above."""


def _cross_pairs(geoms: np.ndarray, member: np.ndarray, iou_min: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs ``a < b`` of boxes from different members with rotated IoU >= ``iou_min``."""
    if len(geoms) == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=float)
    a, b = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = (a < b) & (member[a] != member[b])
    a, b = a[keep], b[keep]
    inter = shapely.area(shapely.intersection(geoms[a], geoms[b]))
    union = shapely.area(geoms[a]) + shapely.area(geoms[b]) - inter
    iou = np.divide(inter, union, out=np.zeros(len(a)), where=union > 0)
    ok = iou >= iou_min
    return a[ok].astype(np.int64), b[ok].astype(np.int64), iou[ok]


def greedy_clusters(geoms: npt.ArrayLike, conf: npt.ArrayLike, member: npt.ArrayLike,
                    iou_min: float = IOU_MIN) -> np.ndarray:
    """Cluster label per box (rule step 2): at most one box per member in a cluster."""
    geoms = np.asarray(geoms, dtype=object)
    conf = np.asarray(conf, dtype=float)
    member = np.asarray(member, dtype=np.int64)
    n = len(geoms)
    a, b, iou = _cross_pairs(geoms, member, iou_min)
    nbr: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, v in zip(a.tolist(), b.tolist(), iou.tolist(), strict=True):
        nbr[i].append((j, v))
        nbr[j].append((i, v))
    label = np.full(n, -1, dtype=np.int64)
    anchor_of: list[int] = []
    seeds_in: list[set[int]] = []
    for i in np.argsort(-conf, kind="stable").tolist():
        best, best_iou = -1, -1.0
        for j, v in nbr[i]:
            c = label[j]
            if c < 0 or anchor_of[c] != j or int(member[i]) in seeds_in[c]:
                continue
            if v > best_iou or (v == best_iou and c < best):
                best, best_iou = int(c), v
        if best < 0:
            best = len(anchor_of)
            anchor_of.append(i)
            seeds_in.append(set())
        label[i] = best
        seeds_in[best].add(int(member[i]))
    return label


def fuse(geoms: npt.ArrayLike, conf: npt.ArrayLike, member: npt.ArrayLike, n_members: int,
         iou_min: float = IOU_MIN) -> pd.DataFrame:
    """One row per cluster: medoid ``geometry``, ``conf``, ``n_seeds``, ``conf_s<k>``,
    ``anchor`` (box index) and ``medoid`` (box index)."""
    geoms = np.asarray(geoms, dtype=object)
    conf = np.asarray(conf, dtype=float)
    member = np.asarray(member, dtype=np.int64)
    if len(member) and (member.min() < 0 or member.max() >= n_members):
        raise EnsembleError("member index outside 0..n_members-1")
    label = greedy_clusters(geoms, conf, member, iou_min)
    n_cl = int(label.max()) + 1 if len(label) else 0
    per = np.full((n_cl, n_members), np.nan)
    per[label, member] = conf
    # medoid: summed IoU to the other members of the same cluster
    a, b, iou = _cross_pairs(geoms, member, 0.0)
    same = label[a] == label[b]
    tot = np.zeros(len(geoms))
    np.add.at(tot, a[same], iou[same])
    np.add.at(tot, b[same], iou[same])
    order = np.lexsort((np.arange(len(geoms)), -conf, -tot, label))
    first = order[np.r_[True, label[order][1:] != label[order][:-1]]] if len(order) else order
    anchors = np.lexsort((np.arange(len(geoms)), -conf, label))
    anchor = anchors[np.r_[True, label[anchors][1:] != label[anchors][:-1]]] if len(
        anchors) else anchors
    out = pd.DataFrame({"cluster": np.arange(n_cl),
                        "conf": np.nansum(per, axis=1) / n_members if n_cl else [],
                        "n_seeds": (~np.isnan(per)).sum(axis=1) if n_cl else [],
                        "medoid": first, "anchor": anchor})
    for k in range(n_members):
        out[f"conf_s{k}"] = per[:, k] if n_cl else []
    out["geometry"] = geoms[first] if n_cl else []
    return out


# --- runs -------------------------------------------------------------------------------------


def _infer_ref(ref: str) -> str:
    return ref.removeprefix("runs/").removeprefix("infer/")


def final_members(project: Project, arm: str, model: str, seeds: Sequence[int]
                  ) -> list[dict[str, Any]]:
    """Per seed: train run, val infer + eval (threshold), WV02 test-region infer run."""
    from nakkhu.detector import _test_infer_runs, load_registry

    reg = load_registry(project)
    tests = _test_infer_runs(project)
    out = []
    for seed in seeds:
        train_run = f"train/{arm}/{model}/seed{seed}"
        units = [u for u in reg.get("units", {}).values()
                 if u["train_run"] == train_run and u.get("eval_val")]
        if len(units) != 1:
            raise EnsembleError(f"{train_run}: {len(units)} evaluated units in the registry")
        u = units[0]
        thr = json.loads((project.paths.runs / u["eval_val"] / "threshold.json")
                         .read_text(encoding="utf-8"))
        if thr.get("split") != "val":
            raise EnsembleError(f"{u['eval_val']} is not a val evaluation")
        if train_run not in tests:
            raise EnsembleError(f"{train_run} has no WV02 test-region inference run")
        out.append({"seed": int(seed), "train_run": train_run, "infer_val": u["infer_val"],
                    "eval_val": u["eval_val"], "threshold": float(thr["threshold"]),
                    "infer_test": tests[train_run]})
    return out


def _load_member(project: Project, infer_ref: str, threshold: float) -> gpd.GeoDataFrame:
    from nakkhu.georef import georef_run

    d = project.paths.runs / "infer" / _infer_ref(infer_ref)
    path = d / "detections.gpkg"
    if not path.is_file():
        georef_run(project, f"infer/{_infer_ref(infer_ref)}")
    det = gpd.read_file(path, layer="obb", columns=["conf"])
    if det.crs is not None and det.crs.to_epsg() != 32645:
        det = det.to_crs(WORK_CRS)
    return det.loc[det["conf"].to_numpy(dtype=float) >= threshold].reset_index(drop=True)


def model_key(arm: str, model: str, seeds: Sequence[int]) -> str:
    return f"ensemble:{arm}/{model}/seeds{'-'.join(str(s) for s in seeds)}"


def build_run(project: Project, members: list[dict[str, Any]], region: str, key: str,
              iou_min: float = IOU_MIN) -> dict[str, Any]:
    """One ensemble infer run (all clusters, no ensemble cut) for ``region`` (val | test)."""
    from nakkhu.georef import write_detections
    from nakkhu.runs import atomic_write_json, create_run

    col = "infer_val" if region == "val" else "infer_test"
    frames = [_load_member(project, m[col], m["threshold"]) for m in members]
    geoms = np.concatenate([np.asarray(f.geometry.values, dtype=object) for f in frames])
    conf = np.concatenate([f["conf"].to_numpy(dtype=float) for f in frames])
    member = np.concatenate([np.full(len(f), k, dtype=np.int64) for k, f in enumerate(frames)])
    values = {"command": "detector final", "region": region, "model": key, "iou_min": iou_min,
              "members": [{k: m[k] for k in ("seed", "train_run", col, "threshold")}
                          for m in members],
              "rule": "nakkhu.ensemble (greedy one-box-per-seed clusters, medoid, mean conf)"}
    run = create_run(project, "infer", ResolvedConfig(values=values, source="nakkhu.ensemble"))
    with run.step("ensemble"):
        fused = fuse(geoms, conf, member, len(members), iou_min)
        gdf = gpd.GeoDataFrame(
            {"conf": fused["conf"].to_numpy(), "cls": np.zeros(len(fused), dtype="int64"),
             "n_seeds": fused["n_seeds"].to_numpy(dtype="int64"),
             **{f"conf_s{m['seed']}": fused[f"conf_s{k}"].to_numpy()
                for k, m in enumerate(members)}},
            geometry=list(fused["geometry"]), crs=WORK_CRS)
        write_detections(run.path / "detections.gpkg", gdf, run.ref)
        support = np.bincount(gdf["n_seeds"].to_numpy(), minlength=len(members) + 1)
        inputs = {"ensemble": True, "model": key, "region": region,
                  "scene": "maxar_wv02_20240113", "iou_min": iou_min,
                  "members": values["members"], "n_member_boxes": [len(f) for f in frames],
                  "clusters": len(gdf),
                  "clusters_by_support": {str(k): int(v) for k, v in enumerate(support)
                                          if k > 0}}
        atomic_write_json(run.path / "inputs.json", inputs)
    run.write_summary({k: v for k, v in inputs.items() if k != "members"})
    return {"run": run.ref, **inputs}


def inventory_frame(project: Project, test_run: str, threshold: float) -> gpd.GeoDataFrame:
    """``detector_final``: test-region clusters at ``conf >= threshold``, centroid in the
    analysis extent, with a stable ``building_id``."""
    from nakkhu.acquire import geo

    det = gpd.read_file(project.paths.runs / "infer" / _infer_ref(test_run) / "detections.gpkg",
                        layer="obb")
    extent = geo.extent(project)
    keep = (det["conf"].to_numpy(dtype=float) >= threshold) & shapely.within(
        shapely.centroid(det.geometry.values), extent)
    out = det.loc[keep].drop(columns=["run_id"], errors="ignore").reset_index(drop=True)
    cx = shapely.get_x(shapely.centroid(out.geometry.values))
    cy = shapely.get_y(shapely.centroid(out.geometry.values))
    order = np.lexsort((cx, -cy))                      # north to south, then west to east
    out = out.iloc[order].reset_index(drop=True)
    out.insert(0, "building_id", [f"df{i:06d}" for i in range(len(out))])
    out["confidence"] = out["conf"]
    out["area_m2"] = shapely.area(out.geometry.values)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=WORK_CRS)


def cli_final(project: Project, args: Any) -> int:
    """``nakkhu detector final``: val + test ensemble runs, val eval, inventory, promote."""
    from nakkhu.artifacts import promote
    from nakkhu.detector import load_registry, save_registry
    from nakkhu.evaluate.score import evaluate_run
    from nakkhu.runs import atomic_write_json, create_run

    seeds = [int(s) for s in args.seeds]
    members = final_members(project, args.arm, args.model, seeds)
    key = model_key(args.arm, args.model, seeds)
    val = build_run(project, members, "val", key)
    print(json.dumps({"val_run": val["run"], "clusters": val["clusters"]}), flush=True)
    res = evaluate_run(project, val["run"], "val")
    threshold = float(res["threshold"])
    print(json.dumps({"eval_val": res["eval_run"], "threshold": threshold,
                      "f1_50": res["f1_50"]}), flush=True)
    test = build_run(project, members, "test", key)
    inv = inventory_frame(project, test["run"], threshold)
    metrics = json.loads((project.paths.runs / res["eval_run"] / "metrics.json").read_text())
    a = metrics["all_tiles"]
    summary = {
        "inventory": "detector_final", "model": key, "seeds": seeds,
        "members": members, "val_run": val["run"], "test_run": test["run"],
        "eval_val": res["eval_run"], "threshold": threshold,
        "val": {"f1_50": a["iou_50"]["f1"], "precision_50": a["iou_50"]["precision"],
                "recall_50": a["iou_50"]["recall"], "f1_30": a["iou_30"]["f1"],
                "ap50": a["ap50"], "n_pred_kept": a["n_pred_kept"], "n_ref": a["n_ref"]},
        "val_clusters_by_support": val["clusters_by_support"],
        "test_clusters_by_support": test["clusters_by_support"],
        "buildings": len(inv),
        "buildings_by_support": {str(k): int(v) for k, v in
                                 inv["n_seeds"].value_counts().sort_index().items()},
        "note": "val metrics only; no test metric computed (prereg guard)"}
    run = create_run(project, "detector", ResolvedConfig(
        values={"command": "detector final", "model": key, "val_run": val["run"],
                "test_run": test["run"], "eval_val": res["eval_run"]},
        source="nakkhu.ensemble"))
    with run.step("inventory"):
        gpkg = run.path / "final_detections.gpkg"
        tmp = gpkg.with_name(".tmp-final_detections.gpkg")
        tmp.unlink(missing_ok=True)
        inv.to_file(tmp, layer="obb", driver="GPKG")
        tmp.replace(gpkg)
        inv.to_parquet(run.path / "inventory_final.parquet", index=False)
        pts = inv[["building_id", "conf", "n_seeds"]].copy()
        pts = gpd.GeoDataFrame(pts, geometry=inv.geometry.centroid, crs=WORK_CRS)
        (run.path / "final_points.geojson").write_text(pts.to_crs("EPSG:4326").to_json(),
                                                       encoding="utf-8")
        atomic_write_json(run.path / "final_model.json", summary)
    run.write_summary({k: v for k, v in summary.items() if k != "members"})
    reg = load_registry(project)
    reg["final"] = {"model": key, "seeds": seeds, "infer_val": val["run"],
                    "infer_test": test["run"], "eval_val": res["eval_run"],
                    "threshold": threshold, "inventory_run": run.ref,
                    "val_f1_50": res["f1_50"]}
    save_registry(project, reg)
    if args.promote:
        for f, dest in (("final_detections.gpkg", "inventory/final_detections.gpkg"),
                        ("final_points.geojson", "inventory/final_points.geojson"),
                        ("inventory_final.parquet", "detector/inventory_final.parquet"),
                        ("final_model.json", "detector/final_model.json")):
            promote(f"{run.ref}/{f}", dest, project.paths, replace=True)
    print(json.dumps({"run": run.ref, "threshold": threshold, "buildings": len(inv),
                      "val_f1_50": res["f1_50"]}, indent=1))
    return 0


def summary_path(project: Project) -> Path:
    return project.paths.artifacts / "detector" / "final_model.json"
