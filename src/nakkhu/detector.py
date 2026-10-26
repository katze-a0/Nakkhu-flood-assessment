"""Plan 007 bookkeeping: ingest offloaded units, the L8 round, artifact tables.

- ``nakkhu detector ingest <pulled dir>...``: for every finished unit record
  (``**/nk/done/<key>.json``, written by ``offload/detector/launcher.py``) check the SHA-256 of
  every output file, copy ``runs/train/<arm>/<model>/seed<k>/`` and its ``runs/infer/<id>/``
  runs into the local ``runs/``, write ``remote.json`` (platform, GPU, timings) next to
  ``train.json``, georeference the val predictions (``nakkhu georef``) and score them on val
  (``nakkhu eval --split val``). The unit registry is ``runs/detector/registry.json``.
- ``nakkhu detector selftrain``: L8 = L7 + one round (plan 004 Step 7) from the L7 seed-0 model.
- ``nakkhu detector summarise``: val tables (no test access).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project
from nakkhu.runs import atomic_write_json

REGISTRY = "detector/registry.json"


class IngestError(RuntimeError):
    """A pulled unit is incomplete or inconsistent."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def registry_path(project: Project) -> Path:
    return project.paths.runs / REGISTRY


def load_registry(project: Project) -> dict[str, Any]:
    p = registry_path(project)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_registry(project: Project, reg: dict[str, Any]) -> None:
    atomic_write_json(registry_path(project), reg)


def _platform_of(source: Path) -> dict[str, Any]:
    startup = sorted(source.rglob("startup.json"))
    finished = sorted(source.rglob("finished.json"))
    out: dict[str, Any] = {"source": source.name}
    if startup:
        s = json.loads(startup[0].read_text(encoding="utf-8"))
        out.update({k: s.get(k) for k in ("job", "host", "platform", "cpus", "setup_s",
                                          "start_utc", "nvidia_smi", "torch_fallback")})
        out["probe"] = s.get("probe")
        out["pin_check"] = {"matched": (s.get("pin_check") or {}).get("matched"),
                            "mismatched": (s.get("pin_check") or {}).get("mismatched")}
    if finished:
        f = json.loads(finished[0].read_text(encoding="utf-8"))
        out.update({"session_end_utc": f.get("end_utc"),
                    "session_wall_s": f.get("total_wall_s")})
    return out


def ingest_unit(project: Project, record: dict[str, Any], nk_root: Path,
                platform: dict[str, Any], *, evaluate: bool = True) -> dict[str, Any]:
    from nakkhu.evaluate.score import evaluate_run
    from nakkhu.georef import georef_run

    for rel, want in record["files"].items():
        p = nk_root / rel
        if not p.is_file():
            raise IngestError(f"{record['unit']}: {rel} missing in the pulled output")
        if _sha(p) != want:
            raise IngestError(f"{record['unit']}: {rel} SHA-256 mismatch")
    _, arm, model, seed_s = record["train_run"].split("/")
    seed = seed_s.removeprefix("seed")
    train_rel = f"runs/{record['train_run']}"
    dst = project.paths.root / train_rel
    src = nk_root / train_rel
    tjson = json.loads((src / "train.json").read_text(encoding="utf-8"))
    if dst.exists():
        old = dst / "train.json"
        if old.is_file() and json.loads(old.read_text())["best_pt_sha256"] == \
                tjson["best_pt_sha256"]:
            pass
        else:
            raise IngestError(f"{dst} exists with other weights; remove it to re-ingest")
    else:
        shutil.copytree(src, dst)
    atomic_write_json(dst / "remote.json", {"unit": record, "platform": platform})
    out: dict[str, Any] = {"unit": record["unit"], "train_run": f"train/{arm}/{model}/seed{seed}",
                           "arm": arm, "model": model, "seed": int(seed),
                           "job": record.get("job"), "platform": platform.get("source"),
                           "wall_s": record["wall_s"], "steps": record["steps"],
                           "start_utc": record["start_utc"], "end_utc": record["end_utc"],
                           "gpu": tjson.get("gpu"), "best_pt_sha256": tjson["best_pt_sha256"]}
    infer_rels = sorted({"/".join(Path(r).parts[:3]) for r in record["files"]
                         if r.startswith("runs/infer/")})
    for rel in infer_rels:
        d = project.paths.root / rel
        if not d.exists():
            shutil.copytree(nk_root / rel, d)
        inputs = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
        region = inputs["region"]
        tag = region if inputs.get("scene", "maxar_wv02_20240113") == "maxar_wv02_20240113"             else f"{region}_{inputs['scene']}"
        out[f"infer_{tag}"] = rel.removeprefix("runs/")
        out[f"ms_per_tile_{tag}"] = inputs.get("ms_per_tile")
        if region == "val":
            if not (d / "detections.gpkg").is_file():
                out["georef_val"] = georef_run(project, rel.removeprefix("runs/"))
            if evaluate:
                res = evaluate_run(project, rel.removeprefix("runs/"), "val")
                out["eval_val"] = res["eval_run"]
                out["val_f1_50"] = res["f1_50"]
                out["val_threshold"] = res["threshold"]
    return out


def cli_ingest(project: Project, args: argparse.Namespace) -> int:
    reg = load_registry(project)
    n = 0
    for source in args.source:
        source = Path(source)
        platform = _platform_of(source)
        for done in sorted(source.rglob("nk/done/*.json")):
            record = json.loads(done.read_text(encoding="utf-8"))
            key = record["unit"]
            if record.get("status") != "ok":
                print(f"skip {key}: {record.get('status')}", file=sys.stderr)
                reg.setdefault("failed", {})[f"{key}@{platform.get('source')}"] = record["status"]
                continue
            if key in reg.get("units", {}) and reg["units"][key].get("eval_val"):
                continue
            try:
                entry = ingest_unit(project, record, done.parent.parent, platform,
                                    evaluate=not args.no_eval)
            except IngestError as error:
                print(f"error: {error}", file=sys.stderr)
                continue
            reg.setdefault("units", {})[key] = entry
            reg.setdefault("sessions", {})[platform.get("source")] = platform
            save_registry(project, reg)
            n += 1
            print(json.dumps({k: entry.get(k) for k in ("unit", "val_f1_50", "eval_val")}))
    print(f"ingested {n} units")
    return 0


# --- L8 -------------------------------------------------------------------------------------

#: L7 = L6 (prereg v2: gate admit fraction 1.0). The L7 cells reuse the L6 runs instead of
#: retraining byte-identical inputs; ``_check_l7_alias`` proves the identity before use.
L7_ALIAS = "L6"


def _check_l7_alias(project: Project) -> None:
    root = project.paths.data_root / "interim" / "labels"
    a, b = root / "L6", root / "L7"
    if (a / "tiles.txt").read_bytes() != (b / "tiles.txt").read_bytes():
        raise SystemExit("L7 tiles.txt differs from L6: L7 must be trained itself")
    for sub in ("labels", "ignore"):
        names = sorted(p.name for p in (a / sub).iterdir())
        if names != sorted(p.name for p in (b / sub).iterdir()):
            raise SystemExit(f"L7 {sub}/ differs from L6")
        for n in names:
            if (a / sub / n).read_bytes() != (b / sub / n).read_bytes():
                raise SystemExit(f"L7 {sub}/{n} differs from L6")


def cli_selftrain(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.config import ResolvedConfig, load_yaml
    from nakkhu.labels.selftrain import apply_to_arm, conf_at_precision
    from nakkhu.runs import create_run

    cfg = load_yaml(project.paths.resolve("configs/labels/L8.yaml"))["selftrain"]
    infer_dir = project.paths.runs / args.train_preds.removeprefix("runs/")
    eval_dir = project.paths.runs / args.val_eval.removeprefix("runs/")
    inputs = json.loads((infer_dir / "inputs.json").read_text(encoding="utf-8"))
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    if inputs["region"] != "train" or not inputs["train_run"].startswith(("train/L7/",
                                                                          "train/L6/")):
        raise SystemExit("--train-preds must be an L7 (= L6) model's infer run on --region "
                         "train")
    if inputs["train_run"].startswith("train/L6/"):
        _check_l7_alias(project)
    if metrics["split"] != "val":
        raise SystemExit("--val-eval must be a val evaluation")
    eval_infer = json.loads((project.paths.runs / metrics["run"] / "inputs.json").read_text())
    if eval_infer["train_run"] != inputs["train_run"]:
        raise SystemExit("the val eval scores a different model")
    preds = pd.read_parquet(eval_dir / "preds.parquet")
    add_conf = conf_at_precision(preds["conf"], preds["tp_50"], 0.9)
    rule = "conf at 90 % precision vs aligned OSM on val (IoU 0.5)"
    if add_conf is None:
        add_conf_used = float("inf")
        rule += "; never reached on val, so no detection is added (drop step only)"
    else:
        add_conf_used = float(add_conf)
    labels_root = project.paths.data_root / "interim" / "labels"
    values = {"command": "detector selftrain", "train_preds": args.train_preds,
              "val_eval": args.val_eval, "add_conf": add_conf, "rule": rule,
              "support_conf": cfg["support_conf"], "overlap_iou": cfg["overlap_iou"]}
    run = create_run(project, "labels", ResolvedConfig(values=values, source="L8.yaml"))
    with run.step("selftrain"):
        det = pd.read_parquet(infer_dir / "detections_px.parquet")
        stats = apply_to_arm(labels_root / "L7", labels_root / "L8", det,
                             add_conf=add_conf_used, support_conf=float(cfg["support_conf"]),
                             overlap_iou=float(cfg["overlap_iou"]), tile_px=640)
        stats.to_csv(run.path / "selftrain_stats.csv", index=False)
        summary = {"add_conf": add_conf, "rule": rule, "tiles": len(stats),
                   "added": int(stats["added"].sum()), "dropped": int(stats["dropped"].sum()),
                   "labels": int(stats["labels"].sum()),
                   "val_max_precision_vs_osm": float(np.max(
                       np.cumsum(preds.sort_values("conf", ascending=False)["tp_50"].to_numpy())
                       / np.arange(1, len(preds) + 1))) if len(preds) else None}
        atomic_write_json(labels_root / "L8" / "selftrain.json", {**values, **summary,
                                                                  "run": run.ref})
    run.write_summary(summary)
    print(json.dumps({"run": run.ref, **summary}, indent=1))
    return 0


# --- tables ---------------------------------------------------------------------------------


def unit_rows(project: Project) -> pd.DataFrame:
    reg = load_registry(project)
    rows = []
    for _key, u in sorted(reg.get("units", {}).items()):
        if not u.get("eval_val") or u["model"].endswith(("-pilot", "-repro")):
            continue                      # pilot runs set the epochs; never reported
        m = json.loads((project.paths.runs / u["eval_val"] / "metrics.json").read_text())
        a = m["all_tiles"]
        tj = json.loads((project.paths.root / "runs" / u["train_run"] / "train.json").read_text())
        rows.append({
            "arm": u["arm"], "model": u["model"], "seed": u["seed"], "train_run": u["train_run"],
            "eval_val": u["eval_val"], "infer_val": u.get("infer_val"),
            "threshold": a["threshold"],
            "f1_50": a["iou_50"]["f1"]["value"], "precision_50": a["iou_50"]["precision"]["value"],
            "recall_50": a["iou_50"]["recall"]["value"], "f1_30": a["iou_30"]["f1"]["value"],
            "ap50": a["ap50"]["value"], "ap50_95": a["ap50_95"]["value"],
            "count_mae": a["tile_count"]["mae"]["value"],
            "count_bias": a["tile_count"]["bias"]["value"],
            "f1_50_reg_tiles": (m.get("osm_registered_tiles") or {}).get("iou_50", {})
            .get("f1", {}).get("value"),
            "ultra_val_map50_95_own_labels": tj["ultralytics_val"].get("best_map50_95"),
            "best_epoch": tj["ultralytics_val"].get("best_epoch"),
            "epochs_run": tj["ultralytics_val"].get("epochs_run"),
            "train_wall_s": tj["train_wall_s"], "unit_wall_s": u["wall_s"],
            "gpu": ";".join(tj.get("gpu") or []), "platform": u.get("platform"),
            "ms_per_tile_val": u.get("ms_per_tile_val"),
        })
    return pd.DataFrame(rows)


def seed_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Seed mean and sample SD (ddof 1) per (arm, model); L7 rows alias the L6 runs."""
    metrics = ["f1_50", "precision_50", "recall_50", "f1_30", "ap50", "ap50_95", "count_mae"]
    alias = rows.loc[rows["arm"] == L7_ALIAS].copy()
    if len(alias) and not (rows["arm"] == "L7").any():
        alias["arm"] = "L7"
        rows = pd.concat([rows, alias], ignore_index=True)
    out = []
    for (arm, model), g in rows.groupby(["arm", "model"]):  # ty: ignore[not-iterable]
        rec: dict[str, Any] = {"arm": arm, "model": model, "n_seeds": len(g),
                               "alias_of": L7_ALIAS if arm == "L7" else "",
                               "seeds": ";".join(str(s) for s in sorted(g["seed"])),
                               "runs": ";".join(g["train_run"])}
        for m in metrics:
            rec[f"{m}_mean"] = float(g[m].mean())
            rec[f"{m}_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else None
        out.append(rec)
    return pd.DataFrame(out)


TEST_EVIDENCE = (("falsifiers", "test_falsifiers.json"), ("cross_sensor", "cross_sensor.json"))


def scored_test_rows(reg: dict[str, Any]) -> pd.DataFrame:
    """One row per run scored on test (registry ``test``); read only, never a test access."""
    rows = [{"name": k, "f1_50": v.get("f1_50"), "threshold": v.get("threshold"),
             "eval_test": v["eval_test"]}
            for k, v in sorted(reg.get("test", {}).items()) if "eval_test" in v]
    return pd.DataFrame(rows, columns=["name", "f1_50", "threshold", "eval_test"])


def scored_test_evidence(reg: dict[str, Any]) -> list[tuple[str, str]]:
    """(source, artifact dest) of the journalled C1/C5 and C4 outputs, copied unchanged."""
    test = reg.get("test", {})
    return [(f"{test[key]['run']}/{name}", f"detector/{name}")
            for key, name in TEST_EVIDENCE if key in test]


def cli_summarise(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.config import ResolvedConfig
    from nakkhu.runs import create_run

    rows = unit_rows(project)
    if rows.empty:
        print("no evaluated units")
        return 1
    run = create_run(project, "detector", ResolvedConfig(values={"command": "summarise",
                                                                 "units": sorted(rows["train_run"])},
                                                         source="nakkhu.detector"))
    with run.step("tables"):
        rows.to_csv(run.path / "val_units.csv", index=False)
        table = seed_table(rows)
        table["split"] = "val"
        table.to_csv(run.path / "val_seed_summary.csv", index=False)
        reg = load_registry(project)
        cost = rows[["arm", "model", "seed", "train_run", "platform", "gpu", "train_wall_s",
                     "unit_wall_s"]].copy()
        cost["unit_gpu_h"] = cost["unit_wall_s"] / 3600
        cost.to_csv(run.path / "cost_units.csv", index=False)
        sessions = pd.DataFrame([{"session": k, **{c: v.get(c) for c in
                                                   ("job", "host", "start_utc",
                                                    "session_end_utc", "session_wall_s",
                                                    "setup_s")},
                                  "gpus": ";".join((v.get("probe") or {}).get("gpus") or [])}
                                 for k, v in reg.get("sessions", {}).items()])
        sessions.to_csv(run.path / "cost_sessions.csv", index=False)
        tests = scored_test_rows(reg)
        if not tests.empty:
            tests.to_csv(run.path / "test_scores.csv", index=False)
    run.write_summary({"units": len(rows), "cells": len(table), "test_scores": len(tests)})
    if args.promote:
        from nakkhu.artifacts import promote

        files = [("val_units.csv", "detector/val_units.csv"),
                 ("val_seed_summary.csv", "detector/val_seed_summary.csv"),
                 ("cost_units.csv", "detector/cost_units.csv"),
                 ("cost_sessions.csv", "detector/cost_sessions.csv")]
        if not tests.empty:
            files.append(("test_scores.csv", "detector/test_scores.csv"))
        for f, dest in files:
            promote(f"{run.ref}/{f}", dest, project.paths, replace=True)
        for src, dest in scored_test_evidence(reg):
            promote(src, dest, project.paths, replace=True)
    print(table.to_string(index=False))
    return 0


# --- D3 inventories -------------------------------------------------------------------------


def cli_inventory(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.evaluate.score import evaluate_run
    from nakkhu.inventory import build_inventories

    reg = load_registry(project)
    for inv in build_inventories(project):
        entry: dict[str, Any] = {"infer_val": inv["run"], "n_obb": inv["n_obb"]}
        if not args.no_eval:
            res = evaluate_run(project, inv["run"], "val", fixed=True)
            entry.update(eval_val=res["eval_run"], val_f1_50=res["f1_50"])
            print(json.dumps({"inventory": inv["inventory"], **entry}), flush=True)
        reg.setdefault("inventories", {})[inv["inventory"]] = entry
        save_registry(project, reg)
    return 0


# --- test scoring (once, behind the plan 005 guard) ------------------------------------------


def _test_infer_runs(project: Project) -> dict[str, str]:
    """``train run -> infer run`` of the WV02 test-region inference of each trained unit."""
    out: dict[str, str] = {}
    for f in sorted((project.paths.runs / "infer").glob("*/inputs.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if (d.get("region") == "test" and d.get("scene") == "maxar_wv02_20240113"
                and d.get("train_run")):          # ensemble / zero-shot runs have none
            out[d["train_run"]] = f"infer/{f.parent.name}"
    return out


def test_plan(project: Project) -> list[dict[str, Any]]:
    reg = load_registry(project)
    tests = _test_infer_runs(project)
    plan = []
    for _key, u in sorted(reg.get("units", {}).items()):
        if u["model"].endswith(("-pilot", "-repro")) or not u.get("eval_val"):
            continue
        plan.append({"kind": "model", "name": u["train_run"], "val_eval": u["eval_val"],
                     "test_infer": tests.get(u["train_run"])})
    final = reg.get("final")
    if final:                       # detector_final: threshold from its own val eval
        plan.append({"kind": "model", "name": final["model"], "val_eval": final["eval_val"],
                     "test_infer": final["infer_test"]})
    zs = reg.get("zeroshot", {})
    if zs.get("selected"):          # D3 SAM3: the variant chosen on val only
        z = zs[zs["selected"]["model"]]
        plan.append({"kind": "model", "name": zs["selected"]["model"],
                     "val_eval": z.get("eval_val"), "test_infer": z.get("infer_test")})
    for name, inv in sorted(reg.get("inventories", {}).items()):
        plan.append({"kind": "inventory", "name": f"inventory:{name}",
                     "test_infer": inv["infer_val"], "val_eval": inv.get("eval_val")})
    return plan


def _pooled_blocks(project: Project, evals: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(project.paths.runs / e / "per_block.csv") for e in evals]
    out = frames[0].copy()
    for f in frames[1:]:
        num = [c for c in out.columns if c != "block_id"]
        out[num] = out[num].to_numpy(float) + f[num].to_numpy(float)
    return out


C1_SEEDS = (0, 1, 2)
C1_CONSENSUS = ("L5", "L6", "L7", "L8")
C1_SINGLE = ("L1", "L2")
C5_FOOTPRINTS = ("google_t90", "microsoft")
FINAL_SEEDS = (0, 1, 2, 3, 4)
FINAL = ("L5", "yolo11n-obb")


def c1_arms(project: Project) -> dict[str, Any]:
    """Arms of C1 by mean val F1@0.5 over seeds 0-2 (YOLO11n-OBB; L7 = L6), never test."""
    rows = unit_rows(project)
    rows = rows.loc[(rows["model"] == "yolo11n-obb") & rows["seed"].isin(C1_SEEDS)]
    means = rows.groupby("arm")["f1_50"].mean()
    if "L6" in means and "L7" not in means:
        means["L7"] = means["L6"]
    cons = means.reindex([a for a in C1_CONSENSUS if a in means]).sort_values(ascending=False)
    single = means.reindex([a for a in C1_SINGLE if a in means]).sort_values(ascending=False)
    a, b = str(cons.index[0]), str(single.index[0])

    def runs(arm: str) -> list[str]:
        base = L7_ALIAS if arm == "L7" else arm
        return [f"train/{base}/yolo11n-obb/seed{s}" for s in C1_SEEDS]

    return {"consensus": a, "single": b, "val_means": {k: float(v) for k, v in means.items()},
            "consensus_runs": runs(a), "single_runs": runs(b)}


def falsifier_plan(project: Project, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Which scored runs C1, C5 and C4 read (C1/C5 read plan entries, scored first)."""
    from nakkhu.testanalyses import wv03_problems, wv03_units

    names = {p["name"] for p in plan}
    arms = c1_arms(project)
    final = [f"train/{FINAL[0]}/{FINAL[1]}/seed{s}" for s in FINAL_SEEDS]
    c5 = {fp: f"inventory:{fp}" for fp in C5_FOOTPRINTS}
    reg = load_registry(project)
    units = wv03_units(project, reg, list(FINAL_SEEDS), *FINAL)
    tests = _test_infer_runs(project)
    problems = [f"C1/C5 input {n} not in the score plan" for n in
                [*arms["consensus_runs"], *arms["single_runs"], *final, *c5.values()]
                if n not in names]
    problems += wv03_problems(project, reg)
    problems += [f"C4: {u['train_run']} lacks WV03 inference or val eval" for u in units
                 if not u["wv03_infer"] or not u["val_eval"]]
    problems += [f"C4: {r} lacks WV02 test inference" for r in final if r not in tests]
    return {"C1": arms, "C5": {"final_seeds": final, "footprints": c5, "min_seeds": 3},
            "C4": {"units": units, "registration": reg.get("wv03_registration")},
            "problems": problems}


def _per_block(project: Project, eval_ref: str) -> pd.DataFrame:
    return pd.read_csv(project.paths.runs / eval_ref / "per_block.csv")


def run_falsifiers(project: Project, fplan: dict[str, Any], results: dict[str, Any],
                   params: dict[str, Any]) -> dict[str, Any]:
    from nakkhu.testanalyses import c1, c5

    kw: dict[str, Any] = {"n_boot": int(params["n_boot"]), "seed": int(params["seed"]),
          "level": float(params["level"])}
    pb = {n: _per_block(project, results[n]["eval_test"]) for n in
          [*fplan["C1"]["consensus_runs"], *fplan["C1"]["single_runs"],
           *fplan["C5"]["final_seeds"], *fplan["C5"]["footprints"].values()]}
    out: dict[str, Any] = {
        "C1": {"arms": {k: fplan["C1"][k] for k in ("consensus", "single", "val_means")},
               **c1([pb[n] for n in fplan["C1"]["consensus_runs"]],
                    [pb[n] for n in fplan["C1"]["single_runs"]], **kw)},
        "C5": {fp: c5([pb[n] for n in fplan["C5"]["final_seeds"]], pb[name], **kw)
               for fp, name in fplan["C5"]["footprints"].items()}}
    out["C5_adopt_footprints"] = any(v["matches"] for v in out["C5"].values())
    return out


def cli_score_test(project: Project, args: argparse.Namespace) -> int:
    """Score every plan entry on test once, then C1, C5 and C4 (prereg v7); journalled.

    Refuses unless the prereg verifies, every input exists and the coordinator's hold file
    (``runs/eval/test_hold.yaml``, :func:`nakkhu.uncertainty.run.hold_state`) releases it.
    """
    from nakkhu.config import ResolvedConfig
    from nakkhu.evaluate.guard import (
        SPLIT_FILE,
        load_prereg,
        read_test_journal,
        record_test_access,
        require_frozen,
        sha256_file,
        verify,
    )
    from nakkhu.evaluate.score import eval_params, evaluate_run
    from nakkhu.georef import georef_run
    from nakkhu.runs import atomic_write_json, create_run
    from nakkhu.uncertainty.run import hold_state

    plan = test_plan(project)
    missing = [p["name"] for p in plan if not p["test_infer"]
               or (p["kind"] == "model" and not p.get("val_eval"))]
    state = verify(project)
    fplan = falsifier_plan(project, plan)
    held, why = hold_state(project)
    report = {"prereg": state.get("prereg_sha256"), "problems": state["problems"],
              "hold": why if held else f"off ({why})",
              "journal_entries": len(read_test_journal(project)),
              "to_score": len(plan), "missing_inputs": missing,
              "plan": [{k: p.get(k) for k in ("kind", "name", "test_infer", "val_eval")}
                       for p in plan],
              "falsifiers": fplan}
    print(json.dumps(report, indent=1, default=str))
    if args.dry_run:
        return 0 if not (state["problems"] or missing or fplan["problems"]) else 1
    if held or missing or state["problems"] or fplan["problems"]:
        print("refused: hold on or prerequisites missing", file=sys.stderr)
        return 1
    reg = load_registry(project)
    results = reg.setdefault("test", {})
    for p in plan:
        if p["name"] in results:
            continue
        d = project.paths.runs / p["test_infer"]
        if p["kind"] == "model":
            if not (d / "detections.gpkg").is_file():
                georef_run(project, p["test_infer"])
            res = evaluate_run(project, p["test_infer"], "test", threshold_from=p["val_eval"])
        else:
            res = evaluate_run(project, p["test_infer"], "test", fixed=True)
        results[p["name"]] = {"eval_test": res["eval_run"], "f1_50": res["f1_50"],
                              "threshold": res["threshold"], "flagged": res["flagged"]}
        save_registry(project, reg)
        print(json.dumps({p["name"]: results[p["name"]]}), flush=True)
    params = eval_params(load_prereg(project))
    frozen = require_frozen(project)
    split_sha = sha256_file(project.paths.resolve(SPLIT_FILE))
    if "falsifiers" not in results:
        access = record_test_access(project, "detector/falsifiers:C1,C5", "c1c5-v7",
                                    frozen["prereg_sha256"], split_sha,
                                    note="plan 007 C1 + C5 from the test evals")
        out = run_falsifiers(project, fplan, results, params)
        run = create_run(project, "detector", ResolvedConfig(
            values={"command": "score-test falsifiers", "C1": fplan["C1"]["consensus_runs"]},
            source="nakkhu.detector"))
        with run.step("falsifiers"):
            atomic_write_json(run.path / "test_falsifiers.json", {**out, "access": access})
        results["falsifiers"] = {"run": run.ref, "C1_pass": out["C1"]["pass"],
                                 "C5_adopt_footprints": out["C5_adopt_footprints"]}
        save_registry(project, reg)
        print(json.dumps(results["falsifiers"]), flush=True)
    if "cross_sensor" not in results:
        from nakkhu.testanalyses import run_cross_sensor

        access = record_test_access(project, "detector/cross_sensor:C4", "c4-v7",
                                    frozen["prereg_sha256"], split_sha,
                                    note="plan 007 C4 WV03 cross-sensor")
        out = run_cross_sensor(project, fplan["C4"]["units"], _test_infer_runs(project), params)
        run = create_run(project, "detector", ResolvedConfig(
            values={"command": "score-test cross_sensor"}, source="nakkhu.detector"))
        with run.step("cross_sensor"):
            atomic_write_json(run.path / "cross_sensor.json", {**out, "access": access})
        results["cross_sensor"] = {"run": run.ref, "pass": out["primary"]["pass"],
                                   "drop_pp_mean": out["primary"]["drop_pp_mean"]}
        save_registry(project, reg)
        print(json.dumps(results["cross_sensor"]), flush=True)
    return 0
