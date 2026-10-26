"""Pre-registration addenda (plan 014): freeze after test access, verify, run-once journal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nakkhu.cli import main
from nakkhu.config import load_project
from nakkhu.evaluate import addendum, guard


def _frozen(root: Path):
    """prereg_v1 frozen, one primary test access journalled (the amend path is then closed)."""
    (root / "configs" / "splits").mkdir(parents=True)
    split = root / "configs" / "splits" / "split_v1.yaml"
    split.write_text("tiles: {}\n", encoding="utf-8")
    sha = hashlib.sha256(split.read_bytes()).hexdigest()
    (root / "configs" / "prereg.yaml").write_text(
        f"version: prereg_v1\nsplit: {{file: configs/splits/split_v1.yaml, sha256: {sha}}}\n",
        encoding="utf-8")
    project = load_project(root)
    base = guard.freeze(project)["prereg_sha256"]
    guard.record_test_access(project, "flood/validate:x", "c", base, sha, note="validate")
    (root / "configs" / "pinned.txt").write_bytes(b"blocks\n")
    return project, base


def _write(root: Path, base: str, name: str = "prereg_v1_a1", extra: str = "") -> Path:
    folder = root / addendum.ADDENDA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    pin = hashlib.sha256(b"blocks\n").hexdigest()
    path = folder / f"{name}.yaml"
    path.write_text(
        f"addendum: {name}\nbase_version: prereg_v1\nbase_sha256: {base}\n"
        "tests:\n  P1-1: {access: test, metric: auc}\n  P0: {access: none}\n"
        f"blocks:\n  sha256_of: configs/pinned.txt\n  sha256: {pin}\n" + extra,
        encoding="utf-8")
    return path


def test_addendum_freezes_after_test_access_and_verifies(root, tmp_path):
    project, base = _frozen(root)
    path = _write(root, base)
    plan = tmp_path / "014.md"
    plan.write_text("# Plan 014\n", encoding="utf-8")
    log = tmp_path / "log.md"
    log.write_text("## prereg_v1_a1\nfollow-up tests\n", encoding="utf-8")
    (root / "artifacts" / guard.CHANGELOG).parent.mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / guard.CHANGELOG).write_text("# old\n", encoding="utf-8")
    out = addendum.freeze_addendum(project, path, plan=plan, changelog=log)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert out["status"] == "frozen" and out["sha256"] == sha
    assert out["test_journal_entries"] == 1 and out["tests"] == {"P1-1": "test", "P0": "none"}
    art = root / "artifacts" / addendum.ARTIFACT_DIR
    assert (art / "prereg_v1_a1.yaml").read_bytes() == path.read_bytes()
    assert (art / "prereg_v1_a1.plan.md").read_bytes() == plan.read_bytes()
    assert json.loads((art / "index.json").read_text(encoding="utf-8"))[0]["sha256"] == sha
    text = (root / "artifacts" / guard.CHANGELOG).read_text(encoding="utf-8")
    assert text.startswith("# old\n") and "prereg_v1_a1" in text              # appended
    state = guard.verify(project)
    assert state["problems"] == [] and state["addenda"] == {"prereg_v1_a1": sha}
    assert guard.frozen_versions(project) == {"prereg_v1": base}          # prereg untouched
    assert addendum.freeze_addendum(project, path)["status"] == "already frozen"
    rc = main(["--root", str(root), "prereg", "verify"])
    assert rc == 0


def test_addendum_refusals(root, tmp_path):
    project, base = _frozen(root)
    with pytest.raises(addendum.AddendumError, match="base_sha256"):
        addendum.freeze_addendum(project, _write(root, "0" * 64))
    path = _write(root, base, extra="tests: {}\n")
    with pytest.raises(addendum.AddendumError, match="duplicate keys"):
        addendum.freeze_addendum(project, path)
    path = _write(root, base)
    path.rename(path.with_name("prereg_v1_a2.yaml"))
    with pytest.raises(addendum.AddendumError, match="not the file stem"):
        addendum.freeze_addendum(project, path.with_name("prereg_v1_a2.yaml"))
    path.with_name("prereg_v1_a2.yaml").unlink()
    path = _write(root, base, extra="requires: [prereg_v1_a0]\n")
    with pytest.raises(addendum.AddendumError, match="requires prereg_v1_a0"):
        addendum.freeze_addendum(project, path)
    path = _write(root, base)
    (root / "configs" / "pinned.txt").write_bytes(b"other\n")
    with pytest.raises(addendum.AddendumError, match="pinned.txt sha256"):
        addendum.freeze_addendum(project, path)
    (root / "configs" / "pinned.txt").write_bytes(b"blocks\n")
    log = tmp_path / "vague.md"
    log.write_text("changes\n", encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="must name prereg_v1_a1"):
        addendum.freeze_addendum(project, path, changelog=log)
    prereg = root / "configs" / "prereg.yaml"
    keep = prereg.read_bytes()
    prereg.write_text(prereg.read_text(encoding="utf-8") + "extra: 1\n", encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="changed after the freeze"):
        addendum.freeze_addendum(project, path)
    prereg.write_bytes(keep)
    other = root / "configs" / "elsewhere.yaml"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="lives in"):
        addendum.freeze_addendum(project, other)
    assert addendum.frozen_addenda(project) == {}


def test_frozen_addendum_is_immutable_and_tampering_fails_verify(root):
    project, base = _frozen(root)
    path = _write(root, base)
    addendum.freeze_addendum(project, path)
    keep = path.read_bytes()
    path.write_text(path.read_text(encoding="utf-8") + "late: change\n", encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="immutable"):
        addendum.freeze_addendum(project, path)
    assert any("edited after its freeze" in p for p in guard.verify(project)["problems"])
    with pytest.raises(guard.GuardError, match="edited after its freeze"):
        guard.require_frozen(project)               # primary scoring refuses as well
    path.write_bytes(keep)
    assert guard.verify(project)["problems"] == []
    copy = root / "artifacts" / addendum.ARTIFACT_DIR / "prereg_v1_a1.yaml"
    copy.write_text("altered\n", encoding="utf-8")
    assert any("frozen copy" in p for p in guard.verify(project)["problems"])
    copy.write_bytes(keep)
    (root / "configs" / "pinned.txt").write_bytes(b"other\n")
    assert any("prereg_v1_a1: blocks" in p for p in guard.verify(project)["problems"])


def test_addendum_test_runs_once_with_one_explicit_retry(root):
    project, base = _frozen(root)
    with pytest.raises(addendum.AddendumError, match="no frozen hash"):
        addendum.require_addendum(project, "prereg_v1_a1", "P1-1")
    addendum.freeze_addendum(project, _write(root, base))
    sha = addendum.frozen_addenda(project)["prereg_v1_a1"]
    with pytest.raises(addendum.AddendumError, match="declares no test"):
        addendum.require_addendum(project, "prereg_v1_a1", "P9")
    with pytest.raises(addendum.AddendumError, match="access: none"):
        addendum.record_addendum_access(project, "prereg_v1_a1", "P0", "followup/x", "c")
    with pytest.raises(addendum.AddendumError, match="retry needs exactly one"):
        addendum.require_addendum(project, "prereg_v1_a1", "P1-1", retry=True)
    out = addendum.record_addendum_access(project, "prereg_v1_a1", "P1-1", "followup/r1", "c")
    entry = out["entry"]
    assert entry["note"] == "prereg_v1_a1:P1-1" and entry["addendum_sha256"] == sha
    assert entry["prereg_sha256"] == base and entry["run"] == "followup/r1"
    with pytest.raises(addendum.AddendumError, match="already ran"):
        addendum.record_addendum_access(project, "prereg_v1_a1", "P1-1", "followup/r2", "c")
    again = addendum.record_addendum_access(project, "prereg_v1_a1", "P1-1", "followup/r2",
                                            "c", retry=True)
    assert again["entry"]["note"] == "prereg_v1_a1:P1-1 retry"
    with pytest.raises(addendum.AddendumError, match="retry needs exactly one"):
        addendum.record_addendum_access(project, "prereg_v1_a1", "P1-1", "followup/r3", "c",
                                        retry=True)
    assert len(addendum.prior_accesses(project, "prereg_v1_a1", "P1-1")) == 2
    assert len(guard.read_test_journal(project)) == 3


def test_cli_addendum_freeze(root, capsys):
    _, base = _frozen(root)
    path = _write(root, base)
    rc = main(["--root", str(root), "prereg", "addendum", "--file",
               "configs/prereg_addenda/prereg_v1_a1.yaml"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["status"] == "frozen"
    path.write_text(path.read_text(encoding="utf-8") + "x: 1\n", encoding="utf-8")
    rc = main(["--root", str(root), "prereg", "addendum", "--file", str(path)])
    assert rc == 1 and "immutable" in capsys.readouterr().err


def test_unused_addendum_can_be_superseded_once(root, tmp_path):
    project, base = _frozen(root)
    addendum.freeze_addendum(project, _write(root, base))
    (root / "configs" / "pinned.txt").write_bytes(b"blocks v2\n")    # a pinned file moves on
    assert any("prereg_v1_a1: blocks" in p for p in guard.verify(project)["problems"])
    pin = hashlib.sha256(b"blocks v2\n").hexdigest()
    folder = root / addendum.ADDENDA_DIR
    nxt = folder / "prereg_v1_a2.yaml"
    nxt.write_text(
        f"addendum: prereg_v1_a2\nbase_version: prereg_v1\nbase_sha256: {base}\n"
        "supersedes: prereg_v1_a1\n"
        "tests:\n  P1-1: {access: test}\n"
        f"blocks:\n  sha256_of: configs/pinned.txt\n  sha256: {pin}\n", encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="supersede_reason"):
        addendum.freeze_addendum(project, nxt)
    nxt.write_text(nxt.read_text(encoding="utf-8") + "supersede_reason: pin too broad\n",
                   encoding="utf-8")
    out = addendum.freeze_addendum(project, nxt)
    assert out["status"] == "frozen" and out["supersedes"] == "prereg_v1_a1"
    state = guard.verify(project)
    assert state["problems"] == [] and set(state["addenda"]) == {"prereg_v1_a1", "prereg_v1_a2"}
    assert addendum.superseded(project) == {"prereg_v1_a1": "prereg_v1_a2"}
    with pytest.raises(addendum.AddendumError, match="superseded by prereg_v1_a2"):
        addendum.require_addendum(project, "prereg_v1_a1", "P1-1")
    assert addendum.require_addendum(project, "prereg_v1_a2", "P1-1")["access"] == "test"
    old = folder / "prereg_v1_a1.yaml"                 # integrity of the old one still binds
    keep = old.read_bytes()
    old.write_text("addendum: x\n", encoding="utf-8")
    assert any("prereg_v1_a1 edited" in p for p in guard.verify(project)["problems"])
    old.write_bytes(keep)
    third = folder / "prereg_v1_a3.yaml"
    third.write_text(nxt.read_text(encoding="utf-8").replace("prereg_v1_a2", "prereg_v1_a3"),
                     encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="already superseded"):
        addendum.freeze_addendum(project, third)


def test_used_addendum_cannot_be_superseded(root):
    project, base = _frozen(root)
    addendum.freeze_addendum(project, _write(root, base))
    addendum.record_addendum_access(project, "prereg_v1_a1", "P1-1", "followup/r1", "c")
    nxt = root / addendum.ADDENDA_DIR / "prereg_v1_a2.yaml"
    pin = hashlib.sha256(b"blocks\n").hexdigest()
    nxt.write_text(
        f"addendum: prereg_v1_a2\nbase_version: prereg_v1\nbase_sha256: {base}\n"
        "supersedes: prereg_v1_a1\nsupersede_reason: change of mind\n"
        "tests:\n  P1-1: {access: test}\n"
        f"blocks:\n  sha256_of: configs/pinned.txt\n  sha256: {pin}\n", encoding="utf-8")
    with pytest.raises(addendum.AddendumError, match="has 1 journalled access"):
        addendum.freeze_addendum(project, nxt)
