"""Test-access journal under concurrent writers (exclusive cross-process lock)."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
from test_addendum import _frozen, _write

from nakkhu.config import load_project
from nakkhu.evaluate import addendum, guard


def _append_many(root: str, worker: int, n: int) -> None:
    project = load_project(Path(root))
    for i in range(n):
        guard.record_test_access(project, f"run/{worker}/{i}", "c", "p", "s", note="w")


def _hold_lock(root: str, ready: mp.Event, release: mp.Event) -> None:
    project = load_project(Path(root))
    with guard.journal_lock(project):
        ready.set()
        release.wait(30)


def _run_test_once(root: str, out: mp.Queue) -> None:
    project = load_project(Path(root))
    try:
        addendum.record_addendum_access(project, "prereg_v1_a1", "P1-1", "followup/x", "c")
        out.put("ok")
    except addendum.AddendumError as error:
        out.put(f"refused: {error}")


def test_concurrent_appends_keep_every_entry(root):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(root), w, 15)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    entries = guard.read_test_journal(load_project(root))
    assert len(entries) == 60
    assert {e["run"] for e in entries} == {f"run/{w}/{i}" for w in range(4) for i in range(15)}


def test_lock_times_out_while_another_process_holds_it(root):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(str(root), ready, release))
    holder.start()
    try:
        assert ready.wait(60)
        project = load_project(root)
        with pytest.raises(guard.GuardError, match="not acquired"):
            with guard.journal_lock(project, timeout_s=0.3):
                pass
    finally:
        release.set()
        holder.join(60)
    with guard.journal_lock(load_project(root), timeout_s=5):      # released by the holder
        pass


def test_lock_is_reentrant_in_one_process(project):
    with guard.journal_lock(project, timeout_s=1):
        with guard.journal_lock(project, timeout_s=1):
            guard.record_test_access(project, "r", "c", "p", "s")
    assert len(guard.read_test_journal(project)) == 1


def test_concurrent_runs_of_one_addendum_test_journal_once(root):
    project, base = _frozen(root)
    addendum.freeze_addendum(project, _write(root, base))
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_run_test_once, args=(str(root), out)) for _ in range(4)]
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(60)
    assert sorted(r == "ok" for r in results) == [False, False, False, True]
    assert all("already ran" in r for r in results if r != "ok")
    notes = [e["note"] for e in guard.read_test_journal(project)]
    assert notes.count("prereg_v1_a1:P1-1") == 1
