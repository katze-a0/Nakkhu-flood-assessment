"""Detector training wrapper (plan 007 Step 2).

``nakkhu train --config configs/train/<model>.yaml --arm L5 --seed 0`` trains one
(arm, model, seed) unit into ``runs/train/<arm>/<model>/seed<k>/``:

- ``config.resolved.yaml``, ``env.json`` (packages, GPU name and driver, torch/CUDA build),
  ``journal.jsonl`` (steps ``dataset``, ``train``);
- ``results.csv`` and ``args.yaml`` (Ultralytics), ``weights/best.pt`` (gitignored; SHA-256 in
  ``train.json``), ``train.json`` (timings, epochs, best epoch, final Ultralytics val metrics
  on the arm's own val labels), ``summary.json``.

Every arm and model uses the same settings (``configs/train/base.yaml``): Ultralytics defaults
except ``epochs`` (frozen after the pilot) and ``seed``; the checkpoint is chosen by
Ultralytics val fitness (``best.pt``). Runtime-only settings (``workers``, ``cache``,
``plots``, ``max_det``) do not change the optimisation and are recorded with the run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from nakkhu.config import ConfigError, Project, ResolvedConfig, apply_overrides, load_yaml
from nakkhu.runs import Run, atomic_write_json, atomic_write_text, collect_env, journal_status

#: keys passed to ``YOLO.train`` (everything else in the config is wrapper metadata)
TRAIN_KEYS = ("epochs", "imgsz", "batch", "workers", "cache", "plots", "deterministic",
              "max_det", "patience", "amp", "val")


class TrainError(RuntimeError):
    """A training unit cannot run."""


def load_train_config(project: Project, path: str | Path,
                      overrides: tuple[str, ...] = ()) -> ResolvedConfig:
    """A model config merged over the file named by its ``extends`` key (one level)."""
    file = project.paths.resolve(str(path))
    values = load_yaml(file)
    parent = values.pop("extends", None)
    if parent:
        base = load_yaml(file.parent / parent)
        base.update(values)
        values = base
    values = apply_overrides(values, overrides)
    for key in ("model", "weights", "task", "imgsz"):
        if key not in values:
            raise ConfigError(f"{file}: missing {key!r}")
    return ResolvedConfig(values=values, source=project.paths.relative(file),
                          overrides=tuple(overrides))


def run_dir(project: Project, arm: str, model: str, seed: int) -> Path:
    return project.paths.runs / "train" / arm / model / f"seed{seed}"


def run_ref(arm: str, model: str, seed: int) -> str:
    return f"train/{arm}/{model}/seed{seed}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_path(project: Project, name: str) -> Path:
    return project.paths.data_root / "interim" / "detector" / "weights" / name


def torch_env() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "cuda_available": torch.cuda.is_available(), "cuda_devices": gpus}


def _final_metrics(results_csv: Path) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame.columns = [c.strip() for c in frame.columns]
    m50 = next((c for c in frame.columns if c.endswith("mAP50(B)")), None)
    m5095 = next((c for c in frame.columns if c.endswith("mAP50-95(B)")), None)
    out: dict[str, Any] = {"epochs_run": len(frame)}
    if m50 and m5095:
        fitness = frame[m5095]       # Ultralytics fitness = mAP50-95 (8.4)
        best = int(fitness.idxmax())
        out.update({"best_epoch": int(frame.loc[best, "epoch"]),
                    "best_fitness": float(fitness.max()),
                    "best_map50": float(frame.loc[best, m50]),
                    "best_map50_95": float(frame.loc[best, m5095]),
                    "fitness_column": m5095})
    return out


def train(project: Project, config: ResolvedConfig, arm: str, seed: int, *,
          device: str | None = None, force: bool = False) -> dict[str, Any]:
    """Train one unit; refuses to overwrite a complete run unless ``force``."""
    from nakkhu import dataset as ds

    cfg = config.values
    if cfg.get("epochs") in (None, ""):
        raise TrainError("epochs is not frozen (configs/train/base.yaml; set after the pilot "
                         "or pass --set epochs=N for the pilot)")
    model = str(cfg["model"])
    path = run_dir(project, arm, model, seed)
    if path.exists():
        if journal_status(path) == "complete" and not force:
            raise TrainError(f"{path} is complete (use --force to retrain)")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    values = {**cfg, "arm": arm, "seed": int(seed)}
    resolved = ResolvedConfig(values=values, source=config.source, overrides=config.overrides)
    atomic_write_text(path / "config.resolved.yaml", resolved.to_yaml())
    env = collect_env(project.paths.root)
    env["torch_env"] = torch_env()
    env["config"] = {"source": config.source, "overrides": list(config.overrides),
                     "sha256": resolved.sha256, "cfghash": resolved.cfghash}
    atomic_write_json(path / "env.json", env)
    run = Run(stage="train", run_id=f"{arm}/{model}/seed{seed}", path=path)
    with run.step("dataset"):
        t0 = time.time()
        built = ds.build(project, arm, str(cfg["task"]))
        dataset_s = time.time() - t0
    weights = weights_path(project, str(cfg["weights"]))
    if not weights.is_file():
        raise TrainError(f"pretrained weights missing: {weights}")
    if cfg.get("weights_sha256") and sha256(weights) != cfg["weights_sha256"]:
        raise TrainError(f"{weights.name}: SHA-256 differs from the config")
    with run.step("train"):
        from ultralytics import YOLO

        kwargs = {k: cfg[k] for k in TRAIN_KEYS if k in cfg and cfg[k] is not None}
        work = path / "ultra"
        t0 = time.time()
        yolo = YOLO(str(weights), task=str(cfg["task"]))
        yolo.train(data=str(ds.dataset_dir(project, arm, str(cfg["task"])) / "data.yaml"),
                   seed=int(seed), project=str(path), name="ultra", exist_ok=True,
                   device=device, verbose=False, **kwargs)
        train_s = time.time() - t0
        best = work / "weights" / "best.pt"
        if not best.is_file():
            raise TrainError("Ultralytics wrote no best.pt")
        (path / "weights").mkdir(exist_ok=True)
        shutil.move(str(best), path / "weights" / "best.pt")
        for name in ("results.csv", "args.yaml"):
            if (work / name).is_file():
                shutil.copy2(work / name, path / name)
        shutil.rmtree(work)                    # last.pt and per-epoch artefacts are not kept
    final = _final_metrics(path / "results.csv")
    record = {"run": run.ref, "arm": arm, "model": model, "seed": int(seed),
              "task": cfg["task"], "weights_init": cfg["weights"],
              "weights_init_sha256": sha256(weights),
              "best_pt_sha256": sha256(path / "weights" / "best.pt"),
              "best_pt_bytes": (path / "weights" / "best.pt").stat().st_size,
              "dataset": {"counts": built["counts"], "inputs_sha256": hashlib.sha256(
                  json.dumps(built["inputs"], sort_keys=True).encode()).hexdigest(),
                  "reused": built.get("reused", False), "build_s": round(dataset_s, 1)},
              "train_wall_s": round(train_s, 1), "device": device,
              "gpu": env["torch_env"].get("cuda_devices"),
              "ultralytics_val": final,
              "selection": "best.pt by Ultralytics fitness on the arm's own val labels"}
    if final.get("epochs_run"):
        record["s_per_epoch"] = round(train_s / final["epochs_run"], 2)
    atomic_write_json(path / "train.json", record)
    run.write_summary({k: record[k] for k in ("run", "best_pt_sha256", "train_wall_s",
                                              "ultralytics_val")})
    return record


def read_args(path: Path) -> dict[str, Any]:
    return yaml.safe_load((path / "args.yaml").read_text(encoding="utf-8"))
