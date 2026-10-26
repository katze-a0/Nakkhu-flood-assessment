"""``nakkhu detector efficiency``: model size and logged predict timings (plan 007 Step 5).

One row per model of ``artifacts/detector/model_efficiency.csv``:

- ``params`` / ``gflops_640``: from ``runs/train/<arm>/<model>/seed<k>/weights/best.pt``
  (``--size-seed``, default 0), loaded unfused on CPU, via ultralytics ``get_num_params`` /
  ``get_flops(model, 640)``. One model is loaded at a time.
- ``{val,test}_ms_per_window_{mean,sd}``: mean and sample SD (ddof 1) over seeds of the logged
  ``ms_per_tile`` (predict wall per 640 px window, NMS included) in each
  ``runs/infer/<id>/inputs.json`` written by :func:`nakkhu.infer.infer_run`. Nothing is re-run
  or measured here; a timing that is not logged stays empty and is named in ``missing``.
- ``concurrent_gpu_load``: what the kernel's other T4 ran during each timed predict step
  (``training`` / ``inference`` / ``dataset build`` / ``other`` / ``idle`` / ``unknown``), from
  the pulled unit records ``{data_root}/interim/detector/offload/*/pulled/<kernel>/nk/done/*.json``
  (kernel, ``gpu_index``, unit window, output files) and the run journals' step intervals.

The run directory holds ``model_efficiency.csv`` and ``manifest.json`` (every input file with
its SHA-256 and the ultralytics / thop versions).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project
from nakkhu.runs import atomic_write_json, journal_status

MODELS = ("yolo11n-obb", "yolov8n-obb", "yolo11s-obb", "yolo11s-hbb")
SCENE = "maxar_wv02_20240113"
REGIONS = ("val", "test")
IMGSZ = 640
ARTIFACT = "detector/model_efficiency.csv"
OFFLOAD = "interim/detector/offload"
LOAD = {"dataset": "dataset build", "train": "training", "predict": "inference"}
NOTE = ("ms per 640 px window = logged predict wall / windows, NMS incl., on one T4 of a 2xT4 "
        "Kaggle kernel; concurrent_gpu_load = what the other T4 ran meanwhile; noisy. Params "
        "and GFLOPs: unfused best.pt on CPU")


class EfficiencyError(RuntimeError):
    """An input needed for the efficiency table is missing or inconsistent."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_size(weights: Path, imgsz: int = IMGSZ) -> tuple[int, float]:
    """(parameters, GFLOPs at ``imgsz``) of an ultralytics checkpoint, unfused, on CPU."""
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model, _ = load_checkpoint(str(weights), device="cpu", fuse=False)
    try:
        params = int(get_num_params(model))
        gflops = float(get_flops(model, imgsz))
    finally:
        del model
        gc.collect()
    if gflops <= 0:  # get_flops returns 0.0 when thop is missing or profiling fails
        raise EfficiencyError(f"{weights}: get_flops returned {gflops}")
    return params, gflops


def logged_timings(project: Project, arm: str,
                   scene: str = SCENE) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    """Every logged infer run of ``arm`` on ``scene``, keyed by (model, seed, region)."""
    out: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    infer = project.paths.runs / "infer"
    for p in sorted(infer.glob("*/inputs.json")) if infer.is_dir() else []:
        d = json.loads(p.read_text(encoding="utf-8"))
        parts = str(d.get("train_run", "")).split("/")
        if (len(parts) != 4 or parts[0] != "train" or parts[1] != arm
                or d.get("scene", SCENE) != scene or d.get("region") not in REGIONS
                or not parts[3].startswith("seed")):
            continue
        env_path = p.parent / "env.json"
        env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
        out.setdefault((parts[2], int(parts[3].removeprefix("seed")), d["region"]), []).append({
            "run": f"infer/{p.parent.name}", "inputs": p, "ms_per_tile": d.get("ms_per_tile"),
            "tiles": d.get("tiles"), "weights_sha256": d.get("weights_sha256"),
            "complete": journal_status(p.parent) == "complete",
            "gpus": sorted({g.get("name", "") for g in env.get("gpus") or []}),
        })
    return out


def _utc(text: str) -> float:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC).timestamp()


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return min(a[1], b[1]) - max(a[0], b[0])


def _steps(run_dir: Path, window: tuple[float, float] | None = None
           ) -> list[tuple[str, float, float]]:
    """(stage, start, end) of each ended step in ``run_dir``'s journal (inside ``window``)."""
    from nakkhu.runs import read_journal

    out = []
    for r in read_journal(run_dir):
        if r.get("event") != "end" or "start_utc" not in r:
            continue
        span = (_utc(r["start_utc"]), _utc(r["utc"]))
        if window is None or (span[0] >= window[0] - 1 and span[1] <= window[1] + 1):
            out.append((str(r["stage"]), *span))
    return out


def unit_records(project: Project) -> list[dict[str, Any]]:
    """Pulled unit records with the remote step intervals of the runs each unit produced."""
    out = []
    base = project.paths.data_root / OFFLOAD
    for p in sorted(base.glob("*/pulled/*/nk/done/*.json")) if base.is_dir() else []:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("gpu_index") is None or not rec.get("start_utc") or not rec.get("end_utc"):
            continue
        window = (_utc(rec["start_utc"]), _utc(rec["end_utc"]))
        infer_runs = sorted({"/".join(Path(f).parts[1:3]) for f in rec.get("files", {})
                             if f.startswith("runs/infer/")})
        dirs = [project.paths.runs / r for r in infer_runs]
        if "train" in (rec.get("steps") or {}) and rec.get("train_run"):
            dirs.append(project.paths.runs / rec["train_run"])
        intervals = [i for d in dirs for i in _steps(d, window) if i[0] in LOAD]
        out.append({"kernel": p.parents[2].name, "gpu": rec["gpu_index"], "window": window,
                    "infer_runs": set(infer_runs), "intervals": intervals, "path": p})
    return out


def concurrent_load(project: Project, infer_ref: str, records: list[dict[str, Any]]) -> str:
    """What the other GPU of the same kernel ran during ``infer_ref``'s predict step."""
    own = [r for r in records if infer_ref in r["infer_runs"]]
    pred = [(a, b) for stage, a, b in _steps(project.paths.runs / infer_ref)
            if stage == "predict"]
    if len(own) != 1 or len(pred) != 1:
        return "unknown"
    others = [r for r in records if r["kernel"] == own[0]["kernel"]
              and r["gpu"] != own[0]["gpu"] and _overlap(r["window"], pred[0]) > 0]
    if not others:
        return "idle"
    loads = [LOAD[stage] for r in others for stage, a, b in r["intervals"]
             if _overlap((a, b), pred[0]) > 0]
    return "+".join(dict.fromkeys(loads)) or "other"


def _load_summary(loads: dict[str, list[str]]) -> str:
    flat = {x for v in loads.values() for x in v}
    if len(flat) == 1:
        return flat.pop()
    return "; ".join(f"{region}={','.join(v)}" for region, v in loads.items() if v)


def _sd(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def build_table(project: Project, *, arm: str = "L5", models: tuple[str, ...] = MODELS,
                seeds: tuple[int, ...] = (0, 1, 2), size_seed: int = 0,
                size_fn: Any = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """The efficiency table and the list of input files it read (path, SHA-256, role)."""
    size_fn = size_fn or model_size
    timings = logged_timings(project, arm)
    records = unit_records(project)
    inputs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    rows = []
    for model in models:
        tdir = project.paths.runs / "train" / arm / model / f"seed{size_seed}"
        weights = tdir / "weights" / "best.pt"
        if not weights.is_file():
            raise EfficiencyError(f"{weights} missing")
        wsha = _sha(weights)
        tjson = json.loads((tdir / "train.json").read_text(encoding="utf-8"))
        if tjson.get("best_pt_sha256") != wsha:
            raise EfficiencyError(f"{weights}: SHA-256 differs from train.json")
        inputs.append({"role": "size", "path": project.paths.relative(weights), "sha256": wsha})
        params, gflops = size_fn(weights)
        row: dict[str, Any] = {"model": model, "arm": arm, "task": tjson.get("task"),
                               "size_run": f"train/{arm}/{model}/seed{size_seed}",
                               "params": params, "params_m": round(params / 1e6, 2),
                               "gflops_640": round(gflops, 2),
                               "seeds": ";".join(str(s) for s in seeds)}
        missing: list[str] = []
        gpus: set[str] = set()
        loads: dict[str, list[str]] = {}
        for region in REGIONS:
            vals, runs, windows = [], [], set()
            loads[region] = []
            for seed in seeds:
                cands = [c for c in timings.get((model, seed, region), []) if c["complete"]]
                if len(cands) != 1 or cands[0]["ms_per_tile"] is None:
                    why = "none logged" if not cands else (
                        f"{len(cands)} runs" if len(cands) > 1 else "no ms_per_tile")
                    missing.append(f"{region} seed{seed} ({why})")
                    continue
                c = cands[0]
                train_sha = json.loads((project.paths.runs / "train" / arm / model
                                        / f"seed{seed}" / "train.json")
                                       .read_text(encoding="utf-8"))["best_pt_sha256"]
                if c["weights_sha256"] != train_sha:
                    raise EfficiencyError(f"{c['run']}: weights differ from train/{arm}/"
                                          f"{model}/seed{seed}")
                vals.append(float(c["ms_per_tile"]))
                load = concurrent_load(project, c["run"], records)
                loads[region].append(load)
                own = [r for r in records if c["run"] in r["infer_runs"]]
                for r in records:
                    if own and r["kernel"] == own[0]["kernel"] and r["path"] not in seen:
                        seen.add(r["path"])
                        inputs.append({"role": "unit_record", "path":
                                       project.paths.relative(r["path"]),
                                       "sha256": _sha(r["path"])})
                runs.append(c["run"])
                windows.add(c["tiles"])
                gpus.update(c["gpus"])
                inputs.append({"role": f"timing_{region}", "path":
                               project.paths.relative(c["inputs"]), "sha256": _sha(c["inputs"])})
            row[f"{region}_n_seeds"] = len(vals)
            row[f"{region}_windows"] = ";".join(str(w) for w in sorted(windows))
            row[f"{region}_ms_per_window_mean"] = round(float(np.mean(vals)), 2) if vals else None
            sd = _sd(vals)
            row[f"{region}_ms_per_window_sd"] = None if sd is None else round(sd, 2)
            row[f"{region}_infer_runs"] = ";".join(runs)
            row[f"{region}_concurrent_gpu_load"] = ";".join(loads[region])
        row["timing_gpu"] = ";".join(sorted(gpus))
        row["concurrent_gpu_load"] = _load_summary(loads)
        row["missing"] = ";".join(missing)
        row["note"] = NOTE
        rows.append(row)
    return pd.DataFrame(rows), inputs


def cli_efficiency(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.config import ResolvedConfig
    from nakkhu.runs import create_run

    values = {"command": "efficiency", "arm": args.arm, "models": list(args.models),
              "seeds": list(args.seeds), "size_seed": args.size_seed, "imgsz": IMGSZ,
              "scene": SCENE, "device": "cpu"}
    run = create_run(project, "detector", ResolvedConfig(values=values,
                                                         source="nakkhu.efficiency"))
    try:
        with run.step("efficiency"):
            table, inputs = build_table(project, arm=args.arm, models=tuple(args.models),
                                        seeds=tuple(args.seeds), size_seed=args.size_seed)
            table.to_csv(run.path / "model_efficiency.csv", index=False)
            versions = {}
            for pkg in ("ultralytics", "ultralytics-thop", "torch"):
                try:
                    versions[pkg] = metadata.version(pkg)
                except metadata.PackageNotFoundError:
                    versions[pkg] = None
            atomic_write_json(run.path / "manifest.json", {
                "outputs": {"model_efficiency.csv": _sha(run.path / "model_efficiency.csv")},
                "inputs": inputs, "params": values, "packages": versions,
                "method": {"size": "ultralytics.nn.tasks.load_checkpoint(fuse=False) on CPU; "
                                   "get_num_params; get_flops(model, 640)",
                           "timing": "logged ms_per_tile in runs/infer/<id>/inputs.json; "
                                     "mean, SD ddof 1 over seeds; nothing re-run"}})
    except EfficiencyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    missing = [f"{m}: {x}" for m, x in zip(table["model"], table["missing"], strict=True) if x]
    run.write_summary({"models": len(table), "missing": missing})
    if not args.no_promote:
        from nakkhu.artifacts import promote

        promote(f"{run.ref}/model_efficiency.csv", ARTIFACT, project.paths, replace=True)
    cols = ["model", "params_m", "gflops_640", "val_ms_per_window_mean",
            "val_ms_per_window_sd", "test_ms_per_window_mean", "test_ms_per_window_sd"]
    print(f"run {run.ref}")
    print(table[cols].to_string(index=False))
    for m in missing:
        print(f"missing timing: {m}", file=sys.stderr)
    return 0
