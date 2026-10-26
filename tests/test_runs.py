import json
from pathlib import Path

import pytest
import yaml

from nakkhu.config import Project
from nakkhu.runs import (
    CONFIG_FILE,
    ENV_FILE,
    atomic_write_text,
    create_run,
    find_run_dir,
    journal_status,
    list_runs,
)


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_atomic_write_failure_keeps_old_content(tmp_path: Path, monkeypatch):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "old")

    def fail(self, other):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        atomic_write_text(target, "new")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_create_run_records_config_and_env(project: Project):
    run = create_run(project, "tiling")
    assert run.path.parent == project.paths.runs / "tiling"
    assert run.run_id.endswith(f"-{project.config.cfghash}")
    resolved = yaml.safe_load((run.path / CONFIG_FILE).read_text(encoding="utf-8"))
    assert resolved == project.config.values
    env = json.loads((run.path / ENV_FILE).read_text(encoding="utf-8"))
    for key in ("git", "uv_lock_sha256", "python", "packages", "os", "cpu", "ram_bytes", "gpus"):
        assert key in env
    assert set(env["git"]) == {"sha", "dirty"}
    assert env["config"]["sha256"] == project.config.sha256
    assert "pyyaml" in {name.lower() for name in env["packages"]}
    assert run.status == "empty"


def test_journal_status_transitions(project: Project):
    run = create_run(project, "tiling")
    started = run.start("tile")
    assert journal_status(run.path) == "running"
    run.end("tile", 0, started)
    assert journal_status(run.path) == "complete"
    with pytest.raises(RuntimeError), run.step("georef"):
        raise RuntimeError("boom")
    assert journal_status(run.path) == "failed"


def test_step_context_writes_start_and_end(project: Project):
    run = create_run(project, "tiling")
    with run.step("tile"):
        pass
    lines = (run.path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    start, end = (json.loads(line) for line in lines)
    assert start["event"] == "start" and end["event"] == "end"
    assert end["returncode"] == 0 and end["start_utc"] == start["utc"]


def test_find_and_list_runs(project: Project):
    run = create_run(project, "tiling")
    nested = run.path / "sub" / "file.csv"
    assert find_run_dir(nested, project.paths) == run.path.resolve()
    assert find_run_dir(project.paths.runs / "tiling", project.paths) is None
    assert [r.ref for r in list_runs(project.paths)] == [run.ref]
    assert list_runs(project.paths, "other") == []


def test_invalid_stage_name(project: Project):
    with pytest.raises(ValueError):
        create_run(project, "../escape")
