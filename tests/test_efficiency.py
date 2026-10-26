"""``nakkhu detector efficiency``: table from logged infer timings (synthetic runs)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from nakkhu import efficiency as ef
from nakkhu.config import Project

REAL_ROOT = Path(__file__).resolve().parents[1]
MS = {("m-obb", 0, "val"): 70.0, ("m-obb", 1, "val"): 74.0, ("m-obb", 2, "val"): 78.0,
      ("m-obb", 0, "test"): 50.0, ("m-obb", 1, "test"): 60.0,          # test seed 2 not logged
      ("m-hbb", 0, "val"): 51.0, ("m-hbb", 1, "val"): 52.0, ("m-hbb", 2, "val"): 53.0}


def _journal(run: Path, stage: str = "predict", t0: str = "00:00:00",
             t1: str = "00:00:20") -> None:
    a, b = f"2026-10-06T{t0}Z", f"2026-10-06T{t1}Z"
    (run / "journal.jsonl").write_text(
        json.dumps({"event": "start", "stage": stage, "utc": a}) + "\n"
        + json.dumps({"event": "end", "stage": stage, "utc": b, "start_utc": a,
                      "returncode": 0}) + "\n", encoding="utf-8")


def _infer_dir(project: Project, model: str, seed: int, region: str) -> Path:
    for p in (project.paths.runs / "infer").glob("*/inputs.json"):
        d = json.loads(p.read_text())
        if d["train_run"] == f"train/L5/{model}/seed{seed}" and d["region"] == region \
                and d["scene"] == ef.SCENE:
            return p.parent
    raise AssertionError((model, seed, region))


def _done(project: Project, kernel: str, key: str, gpu: int, t0: str, t1: str,
          infer: list[Path], train_run: str | None = None) -> None:
    d = project.paths.data_root / ef.OFFLOAD / "B" / "pulled" / kernel / "nk" / "done"
    d.mkdir(parents=True, exist_ok=True)
    files = {f"runs/infer/{i.name}/inputs.json": "x" for i in infer}
    steps = {"train": {"rc": 0}} if train_run else {"infer_val": {"rc": 0}}
    (d / f"{key}.json").write_text(json.dumps({
        "unit": key, "gpu_index": gpu, "start_utc": f"2026-10-06T{t0}Z",
        "end_utc": f"2026-10-06T{t1}Z", "files": files, "steps": steps,
        "train_run": train_run}))


def _fake_runs(project: Project) -> None:
    runs = project.paths.runs
    shas = {}
    for model in ("m-obb", "m-hbb"):
        for seed in (0, 1, 2):
            t = runs / "train" / "L5" / model / f"seed{seed}"
            (t / "weights").mkdir(parents=True)
            data = f"{model}{seed}".encode()
            (t / "weights" / "best.pt").write_bytes(data)
            shas[(model, seed)] = hashlib.sha256(data).hexdigest()
            (t / "train.json").write_text(json.dumps({
                "best_pt_sha256": shas[(model, seed)],
                "task": "obb" if model.endswith("obb") else "detect"}), encoding="utf-8")
    for i, ((model, seed, region), ms) in enumerate(sorted(MS.items())):
        d = runs / "infer" / f"20261006T00000{i}Z-abc"
        d.mkdir(parents=True)
        (d / "inputs.json").write_text(json.dumps({
            "train_run": f"train/L5/{model}/seed{seed}", "region": region,
            "scene": ef.SCENE, "tiles": 260 if region == "val" else 277, "ms_per_tile": ms,
            "weights_sha256": shas[(model, seed)]}), encoding="utf-8")
        (d / "env.json").write_text(json.dumps({"gpus": [{"name": "Tesla T4"}] * 2}))
        _journal(d)
    other = runs / "infer" / "20261006T009999Z-wv03"    # another scene: ignored
    other.mkdir()
    (other / "inputs.json").write_text(json.dumps({
        "train_run": "train/L5/m-obb/seed2", "region": "test", "scene": "maxar_wv03",
        "tiles": 9, "ms_per_tile": 999.0, "weights_sha256": shas[("m-obb", 2)]}))
    _journal(other)


def test_table_from_logged_timings(project: Project):
    _fake_runs(project)
    sizes = {"m-obb": (2_661_702, 6.7471), "m-hbb": (9_428_179, 21.67)}
    table, inputs = ef.build_table(project, models=("m-obb", "m-hbb"),
                                   size_fn=lambda w: sizes[w.parents[2].name])
    obb, hbb = (table.set_index("model").loc[m] for m in ("m-obb", "m-hbb"))
    assert obb["params"] == 2_661_702 and obb["params_m"] == 2.66
    assert obb["gflops_640"] == 6.75 and obb["size_run"] == "train/L5/m-obb/seed0"
    assert obb["val_ms_per_window_mean"] == 74.0 and obb["val_ms_per_window_sd"] == 4.0
    assert obb["val_n_seeds"] == 3 and obb["val_windows"] == "260"
    assert obb["test_ms_per_window_mean"] == 55.0          # seeds 0-1; other scene ignored
    assert obb["test_ms_per_window_sd"] == pytest.approx(7.07)
    assert obb["test_n_seeds"] == 2 and obb["missing"] == "test seed2 (none logged)"
    assert hbb["val_ms_per_window_mean"] == 52.0 and hbb["val_ms_per_window_sd"] == 1.0
    assert hbb["test_n_seeds"] == 0 and pd.isna(hbb["test_ms_per_window_mean"])
    assert hbb["timing_gpu"] == "Tesla T4" and "T4" in hbb["note"]
    assert hbb["concurrent_gpu_load"] == "unknown"          # no pulled unit records
    assert {i["role"] for i in inputs} == {"size", "timing_val", "timing_test"}


def test_concurrent_gpu_load_from_unit_records(project: Project):
    _fake_runs(project)
    val = {k: _infer_dir(project, "m-obb", k, "val") for k in (0, 1, 2)}
    _journal(val[0], t0="00:50:00", t1="00:50:20")      # other GPU training
    _done(project, "k1", "a", 0, "00:00:00", "00:50:21", [val[0]])
    trainer = project.paths.runs / "train" / "L5" / "x" / "seed0"
    trainer.mkdir(parents=True)
    _journal(trainer, "train", "00:01:00", "01:59:00")
    _done(project, "k1", "b", 1, "00:00:00", "02:00:00", [], "train/L5/x/seed0")
    _journal(val[1], t0="00:10:00", t1="00:10:20")      # other GPU predicting
    _done(project, "k2", "c", 0, "00:00:00", "00:10:21", [val[1]])
    other = project.paths.runs / "infer" / "20261006T001005Z-other"
    other.mkdir()
    _journal(other, t0="00:10:05", t1="00:10:30")
    _done(project, "k2", "d", 1, "00:00:00", "00:10:31", [other])
    _journal(val[2], t0="03:00:00", t1="03:00:20")      # other GPU finished: idle
    _done(project, "k3", "e", 0, "02:00:00", "03:00:21", [val[2]])
    _done(project, "k3", "f", 1, "00:00:00", "02:30:00", [])
    table, inputs = ef.build_table(project, models=("m-obb",), size_fn=lambda w: (1, 1.0))
    row = table.iloc[0]
    assert row["val_concurrent_gpu_load"] == "training;inference;idle"
    assert row["test_concurrent_gpu_load"] == "unknown;unknown"   # no unit records
    assert row["concurrent_gpu_load"] == \
        "val=training,inference,idle; test=unknown,unknown"
    assert sum(i["role"] == "unit_record" for i in inputs) == 6


def test_weights_mismatch_refused(project: Project):
    _fake_runs(project)
    p = next((project.paths.runs / "infer").glob("*/inputs.json"))
    d = json.loads(p.read_text())
    p.write_text(json.dumps(d | {"weights_sha256": "0" * 64}))
    with pytest.raises(ef.EfficiencyError, match="weights differ"):
        ef.build_table(project, models=("m-obb", "m-hbb"), size_fn=lambda w: (1, 1.0))


def test_cli_writes_manifest_and_promotes(project: Project, monkeypatch):
    _fake_runs(project)
    monkeypatch.setattr(ef, "model_size", lambda w: (1000, 2.0))
    args = argparse.Namespace(arm="L5", models=["m-obb", "m-hbb"], seeds=[0, 1, 2],
                              size_seed=0, no_promote=False)
    assert ef.cli_efficiency(project, args) == 0
    run = next((project.paths.runs / "detector").iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["params"]["seeds"] == [0, 1, 2] and len(manifest["inputs"]) == 2 + 8
    art = project.paths.artifacts / "detector" / "model_efficiency.csv"
    assert art.read_bytes() == (run / "model_efficiency.csv").read_bytes()
    assert hashlib.sha256(art.read_bytes()).hexdigest() == \
        manifest["outputs"]["model_efficiency.csv"]


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None,
                    reason="needs the `train` dependency group")
def test_model_size_reproduces_yolo11n_obb():
    weights = REAL_ROOT / "runs" / "train" / "L5" / "yolo11n-obb" / "seed0" / "weights" / "best.pt"
    if not weights.is_file():
        pytest.skip("trained weights not present")
    params, gflops = ef.model_size(weights)
    assert params == 2_661_702 and round(gflops, 2) == 6.75
