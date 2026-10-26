"""Plan 010 spec: config, random streams and prerequisites (no test access here)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from nakkhu.config import Project, load_yaml

CONFIG = "configs/uncertainty/uncertainty.yaml"

# One fixed stream id per purpose: rng = default_rng([seed, STREAMS[purpose]]).
STREAMS = {
    "u3_bootstrap_recall": 1,
    "u3_bootstrap_precision": 2,
    "split_sample": 3,
    "split_sample_bootstrap": 4,
    "u2_audit_replicates_recall": 5,
    "u2_audit_replicates_precision": 6,
    "u2_share_ci": 7,
    "simulation": 8,
    "split_sample_block": 9,
    "u3_bootstrap_precision_main": 10,     # prereg v8: main-inventory polygon precision
}


def rng(seed: int, purpose: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), STREAMS[purpose]])


def stream_seed(seed: int, purpose: str) -> int:
    """An integer seed for code that takes ``seed: int`` (e.g. the plan 006 Rao-Wu weights)."""
    return int(rng(seed, purpose).integers(0, 2**31 - 1))


def load_spec(project: Project) -> dict[str, Any]:
    return load_yaml(project.paths.resolve(CONFIG))


def final_runs(spec: dict[str, Any]) -> list[str]:
    fm = spec["inputs"]["final_model"]
    return [str(fm["train_runs"]).format(seed=s) for s in fm["seeds"]]


def ensemble_key(spec: dict[str, Any]) -> str:
    """Model key of the plan 007 ensemble run (``nakkhu.ensemble.model_key``)."""
    fm = spec["inputs"]["final_model"]
    return f"ensemble:{fm['arm']}/{fm['model']}/seeds{'-'.join(str(s) for s in fm['seeds'])}"


def _inputs(project: Project, run: str) -> dict[str, Any]:
    path = project.paths.runs / str(run) / "inputs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def precision_pool_problems(project: Project, spec: dict[str, Any]) -> list[str]:
    """The sitting-2 pool must be exactly six WV02 test-region runs: the five final seeds and
    the ``detector_final`` ensemble run, each thresholded by ``threshold_from`` the val eval of
    the same model (prereg ``audit_precision_pool``)."""
    from nakkhu.audit import core as audit_core

    cfg = audit_core.load_audit_config(project)
    specs = cfg["tasks"]["precision_pool"].get("runs") or []
    if not specs:
        return ["audit.yaml tasks.precision_pool.runs is empty (plan 006 sitting 2 not set up)"]
    problems = []
    want = sorted([*final_runs(spec), ensemble_key(spec)])
    got = []
    for s in specs:
        inp = _inputs(project, str(s.get("run")))
        if inp.get("region") != "test" or inp.get("scene") != "maxar_wv02_20240113":
            problems.append(f"precision_pool run {s.get('run')} is not a WV02 test-region run")
        got.append(str(inp.get("model")) if inp.get("ensemble") else str(inp.get("train_run")))
        ref = s.get("threshold_from")
        if s.get("threshold") is not None or not ref:
            problems.append(f"precision_pool run {s.get('run')}: the threshold must come from "
                            "`threshold_from` (the val eval of the same model)")
            continue
        doc = project.paths.runs / str(ref) / "threshold.json"
        th = json.loads(doc.read_text(encoding="utf-8")) if doc.is_file() else {}
        if th.get("split") != "val" or th.get("model") != inp.get("model"):
            problems.append(f"{ref} is not the val eval of the model of {s.get('run')}")
    if sorted(got) != want:
        problems.append(f"precision_pool runs are {sorted(got)}, prereg wants {want}")
    return problems


def prerequisites(project: Project) -> list[str]:
    """Everything that must exist before ``nakkhu uncertainty run`` may open the test audit."""
    from nakkhu.evaluate.guard import load_prereg, verify

    problems = [f"prereg: {p}" for p in verify(project)["problems"]]
    try:
        if not (load_prereg(project).get("uncertainty_propagation") or {}).get("config"):
            problems.append("prereg has no uncertainty_propagation section (prereg v7)")
    except Exception as error:   # missing prereg is already reported by verify
        problems.append(str(error))
    spec = load_spec(project)
    inp = spec["inputs"]
    for key in ("detector_final", "blocks", "event_skill"):
        path = project.paths.resolve(str(inp[key]))
        if not path.is_file():
            problems.append(f"{inp[key]} missing ({key})")
    for task in inp["audit_tasks"].values():
        for name in ("items.parquet", "responses.jsonl"):
            if not (project.paths.root / "audit" / task / name).is_file():
                problems.append(f"audit/{task}/{name} missing")
    problems += precision_pool_problems(project, spec)
    return problems
