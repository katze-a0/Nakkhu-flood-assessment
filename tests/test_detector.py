"""Plan 007: dataset builder split guard, HBB conversion, 1-epoch CPU training smoke run."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from nakkhu import dataset as ds
from nakkhu.config import Project, load_project
from nakkhu.runs import journal_status

REAL_ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"t_0000_0000": "train", "t_0001_0000": "train", "t_0002_0000": "train",
          "t_0003_0000": "train", "t_0000_0001": "val", "t_0001_0001": "val",
          "t_0002_0001": "buffer", "t_0003_0001": "test"}


def _fake_project(root: Path, arm_tiles: list[str]) -> Project:
    rng = np.random.default_rng(0)
    split = {"schema_version": 1, "version": "split_v1",
             "tiles": {t: {"split": s, "block": "b_00_00", "usable": True}
                       for t, s in SPLITS.items()}}
    (root / "configs" / "splits").mkdir(parents=True)
    (root / "configs" / "splits" / "split_v1.yaml").write_text(yaml.safe_dump(split))
    tiles = root / "data" / "interim" / "tiles" / ds.SCENE / "train"
    arm = root / "data" / "interim" / "labels" / "LX"
    (tiles / "images").mkdir(parents=True)
    (arm / "labels").mkdir(parents=True)
    (arm / "ignore").mkdir()
    for t in SPLITS:
        img = np.full((640, 640, 3), 90, np.uint8)
        lines = []
        for _ in range(6):
            x, y = rng.integers(20, 560, 2)
            w, h = rng.integers(20, 60, 2)
            img[y:y + h, x:x + w] = 200
            c = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) / 640
            lines.append("0 " + " ".join(f"{v:.6f}" for v in c.reshape(-1)))
        Image.fromarray(img).save(tiles / "images" / f"{t}.jpg", quality=95)
        (arm / "labels" / f"{t}.txt").write_text("\n".join(lines) + "\n")
        mask = np.zeros((640, 640), bool)
        mask[:10, :10] = True
        Image.fromarray(mask).save(arm / "ignore" / f"{t}.png")
    (arm / "tiles.txt").write_text("\n".join(arm_tiles) + "\n")
    return load_project(root)


def test_check_ids_refuses_test_buffer_and_overlap():
    split = {t: {"split": s, "usable": True} for t, s in SPLITS.items()}
    ds.check_ids(split, ["t_0000_0000"], ["t_0000_0001"])
    for bad_train in (["t_0003_0001"], ["t_0002_0001"], ["t_0000_0001"]):
        with pytest.raises(ds.DatasetError):
            ds.check_ids(split, bad_train, ["t_0000_0001"])
    with pytest.raises(ds.DatasetError):
        ds.check_ids(split, ["t_0000_0000"], ["t_0003_0001"])


def test_built_dataset_has_no_test_or_buffer_tile(root: Path):
    project = _fake_project(root, ["t_0000_0000", "t_0001_0000", "t_0002_0000",
                                   "t_0003_0000"])
    out = ds.build(project, "LX")
    d = ds.dataset_dir(project, "LX")
    written = {p.stem for p in (d / "images").rglob("*.*")} | \
              {p.stem for p in (d / "labels").rglob("*.txt")}
    forbidden = {t for t, s in SPLITS.items() if s in ("test", "buffer")}
    assert written and not written & forbidden
    assert out["counts"]["train"]["tiles"] == 4 and out["counts"]["val"]["tiles"] == 2
    filled = np.array(Image.open(d / "images" / "train" / "t_0000_0000.png"))
    assert (filled[:10, :10] == filled[:10, :10].reshape(-1, 3).mean(0).round()).all()
    assert ds.build(project, "LX")["reused"]


def test_arm_listing_a_test_tile_is_refused(root: Path):
    project = _fake_project(root, ["t_0000_0000", "t_0003_0001"])
    with pytest.raises(ds.DatasetError):
        ds.build(project, "LX")


def test_obb_to_detect_envelope():
    line = "0 0.1 0.2 0.3 0.1 0.4 0.3 0.2 0.4\n"
    assert ds.obb_to_detect(line) == "0 0.250000 0.250000 0.300000 0.300000\n"


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None,
                    reason="needs the `train` dependency group")
def test_one_epoch_cpu_smoke_run_is_complete(root: Path):
    weights = REAL_ROOT / "data" / "interim" / "detector" / "weights" / "yolo11n-obb.pt"
    if not weights.is_file():
        pytest.skip("pretrained weights not downloaded")
    project = _fake_project(root, ["t_0000_0000", "t_0001_0000", "t_0002_0000",
                                   "t_0003_0000"])
    wdir = root / "data" / "interim" / "detector" / "weights"
    wdir.mkdir(parents=True)
    shutil.copy2(weights, wdir / weights.name)
    shutil.copytree(REAL_ROOT / "configs" / "train", root / "configs" / "train")
    from nakkhu.train import load_train_config, train

    cfg = load_train_config(project, "configs/train/yolo11n-obb.yaml",
                            ("epochs=1", "batch=2", "workers=0", "cache=false"))
    record = train(project, cfg, "LX", 0, device="cpu")
    run = root / "runs" / "train" / "LX" / "yolo11n-obb" / "seed0"
    assert journal_status(run) == "complete"
    for name in ("config.resolved.yaml", "env.json", "journal.jsonl", "train.json",
                 "summary.json", "results.csv", "args.yaml", "weights/best.pt"):
        assert (run / name).is_file(), name
    assert not (run / "ultra").exists()
    assert json.loads((run / "train.json").read_text())["best_pt_sha256"] == \
        record["best_pt_sha256"]


# --- score-test exit code and test-evidence promotion helpers ---------------------------


def _patch_score_test(monkeypatch, project, registry: dict, *, problems=(), held=False):
    import argparse

    from nakkhu import detector
    from nakkhu.evaluate import guard, score
    from nakkhu.uncertainty import run as urun

    plan = [{"kind": "model", "name": "train/L5/m/seed0", "test_infer": "infer/a",
             "val_eval": "eval/b"}]
    monkeypatch.setattr(detector, "test_plan", lambda project: plan)
    monkeypatch.setattr(detector, "falsifier_plan", lambda project, plan: {"problems": []})
    monkeypatch.setattr(detector, "load_registry", lambda project: registry)
    monkeypatch.setattr(detector, "save_registry", lambda project, reg: None)
    monkeypatch.setattr(guard, "verify", lambda project: {"prereg_sha256": "p",
                                                          "problems": list(problems)})
    monkeypatch.setattr(guard, "read_test_journal", lambda project: [])
    monkeypatch.setattr(guard, "load_prereg", lambda project: {})
    monkeypatch.setattr(guard, "require_frozen", lambda project: {"prereg_sha256": "p"})
    monkeypatch.setattr(guard, "sha256_file", lambda path: "s")
    monkeypatch.setattr(score, "eval_params", lambda prereg: {})
    monkeypatch.setattr(urun, "hold_state", lambda project: (held, "why"))

    def no_access(*a, **k):
        raise AssertionError("no test access expected")

    monkeypatch.setattr(guard, "record_test_access", no_access)
    monkeypatch.setattr(score, "evaluate_run", no_access)
    return lambda dry: detector.cli_score_test(project, argparse.Namespace(dry_run=dry))


def test_score_test_success_returns_zero_and_skips_scored(monkeypatch, project):
    reg = {"test": {"train/L5/m/seed0": {"eval_test": "eval/c", "f1_50": 0.1,
                                         "threshold": 0.2, "flagged": False},
                    "falsifiers": {"run": "detector/x"}, "cross_sensor": {"run": "detector/y"}}}
    run = _patch_score_test(monkeypatch, project, reg)
    assert run(False) == 0
    assert run(True) == 0


def test_score_test_refusals_return_one(monkeypatch, project):
    assert _patch_score_test(monkeypatch, project, {"test": {}}, problems=["drift"])(True) == 1
    assert _patch_score_test(monkeypatch, project, {"test": {}}, problems=["drift"])(False) == 1
    assert _patch_score_test(monkeypatch, project, {"test": {}}, held=True)(False) == 1


def test_scored_test_rows_and_evidence():
    from nakkhu.detector import scored_test_evidence, scored_test_rows

    reg = {"test": {"b": {"eval_test": "eval/2", "f1_50": 0.2, "threshold": None,
                          "flagged": False},
                    "a": {"eval_test": "eval/1", "f1_50": 0.1, "threshold": 0.3,
                          "flagged": False},
                    "falsifiers": {"run": "detector/f", "C1_pass": True},
                    "cross_sensor": {"run": "detector/c", "pass": True}}}
    rows = scored_test_rows(reg)
    assert list(rows.columns) == ["name", "f1_50", "threshold", "eval_test"]
    assert rows["name"].tolist() == ["a", "b"]
    assert rows["eval_test"].tolist() == ["eval/1", "eval/2"]
    assert scored_test_evidence(reg) == [
        ("detector/f/test_falsifiers.json", "detector/test_falsifiers.json"),
        ("detector/c/cross_sensor.json", "detector/cross_sensor.json")]
    assert scored_test_rows({}).empty and scored_test_evidence({}) == []
