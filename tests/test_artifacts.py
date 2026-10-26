import hashlib
import re

import pytest
import yaml

from nakkhu.artifacts import PromotionError, load_index, promote
from nakkhu.config import Project
from nakkhu.runs import Run, create_run


def _complete_run(project: Project) -> Run:
    run = create_run(project, "evaluate")
    with run.step("score"):
        (run.path / "metrics.csv").write_text("metric,value\nf1,0.5\n", encoding="utf-8")
    return run


def test_promote_writes_index_entry(project: Project):
    run = _complete_run(project)
    source = f"{run.ref}/metrics.csv"
    entry = promote(source, "detector/metrics.csv", project.paths)

    target = project.paths.artifacts / "detector" / "metrics.csv"
    data = (run.path / "metrics.csv").read_bytes()
    assert target.read_bytes() == data
    index = yaml.safe_load(project.paths.artifact_index.read_text(encoding="utf-8"))
    assert index["artifacts"] == [entry]
    assert entry["path"] == "detector/metrics.csv"
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["bytes"] == len(data)
    assert entry["source_run"] == run.ref
    assert entry["source_file"] == "metrics.csv"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["promoted_utc"])


def test_refuses_incomplete_journal(project: Project):
    run = create_run(project, "evaluate")
    run.start("score")
    (run.path / "metrics.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="running"):
        promote(f"{run.ref}/metrics.csv", "metrics.csv", project.paths)
    run.end("score", 1)
    with pytest.raises(PromotionError, match="failed"):
        promote(f"{run.ref}/metrics.csv", "metrics.csv", project.paths)
    assert not (project.paths.artifacts / "metrics.csv").exists()


def test_refuses_empty_journal(project: Project):
    run = create_run(project, "evaluate")
    (run.path / "metrics.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="empty"):
        promote(f"{run.ref}/metrics.csv", "metrics.csv", project.paths)


def test_refuses_source_outside_runs(project: Project):
    stray = project.paths.root / "stray.csv"
    stray.write_text("x\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="run directory"):
        promote("stray.csv", "stray.csv", project.paths)


@pytest.mark.parametrize("dest", ["../escape.csv", "INDEX.yaml", "artifacts/"])
def test_refuses_bad_destination(project: Project, dest: str):
    run = _complete_run(project)
    with pytest.raises(PromotionError):
        promote(f"{run.ref}/metrics.csv", dest, project.paths)


def test_replace_and_size_limit(project: Project):
    run = _complete_run(project)
    source = f"{run.ref}/metrics.csv"
    promote(source, "artifacts/m.csv", project.paths)  # leading artifacts/ is stripped
    with pytest.raises(PromotionError, match="exists"):
        promote(source, "m.csv", project.paths)
    promote(source, "m.csv", project.paths, replace=True)
    assert [e["path"] for e in load_index(project.paths)["artifacts"]] == ["m.csv"]
    with pytest.raises(PromotionError, match="bytes"):
        promote(source, "big.csv", project.paths, max_bytes=4)
