"""Audit configuration, task files, the append-only response log and the item queue.

Files per task (``<task>`` = ``recall_points``, ``precision_pool``, ``flood_trace`` or
``precision_microsoft``):

- ``audit/<task>/items.parquet``: the frozen design-size sample (tracked): ``item_id``,
  coordinates, ``stratum``, ``frame_n`` (stratum size N_h), ``design_n`` (n_h), ``rank`` (position
  in the seeded within-stratum permutation), ``pi_design`` = n_h / N_h, ``pilot``, ``order_key``.
- ``audit/allocation.yaml``: final sizes set from the timed pilot (written once).
- ``audit/<task>/responses.jsonl``: every event, appended and fsynced at once (raw evidence).
- ``audit/<task>/responses_pass1.jsonl``: a redone or restarted sitting's first pass, moved
  there unchanged by ``serve --redo`` or ``serve --restart``; the new ``responses.jsonl`` then
  opens with a ``redo`` event (``redo_of: pass1``, ``prior_answer_shown: true``: answers given
  with the pass-1 answer shown) or a ``restart`` event (``restart_of: pass1``,
  ``prior_answer_shown: false``, ``reason``: pass 1 void, served again blind) and holds only the
  new pass's answers (:func:`start_pass`, :func:`redo_state`).
- ``{data_root}/interim/audit/<task>/``: frame and chips (local only, never published).

``recall_nearmiss`` (sitting 1b) is derived from ``recall_points``: its items are the sitting-1
items answered No/Unsure in a footprint stratum (asked "is there a roof within the circle?") plus
a random overlap sample asked the original question again (inter-rater agreement). Its queue is
all items in ``order_key`` order: no pilot, no allocation, no repeats; each slot carries its own
question. Every task has an ``auditor`` (``tasks.<task>.auditor``); the server writes it into
every event, and sitting-1 events written before the field existed get it from a later
``auditor`` event of the same sitting (:func:`sitting_auditors`).

The queue of a task is a pure function of its items, the allocation and the seed: pilot items
first (recall_points), then the other active items in ``order_key`` order, with repeats for
self-agreement inserted at least ``repeat_min_gap`` items after their original and never inside
the pilot. A slot is ``<item_id>`` or ``<item_id>#r`` (the repeat); the page never sees either.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nakkhu.config import Project, load_yaml
from nakkhu.runs import utc_now

AUDIT_CONFIG = "configs/audit/audit.yaml"
ALLOCATION = "audit/allocation.yaml"
# Order fixes the sampling streams (TASKS.index): new tasks are appended, never inserted.
# precision_microsoft (plan 013, prereg v8, sitting 2) is absent from the pinned allocation
# file, so its size is its design size (active_items).
# flood_trace_p2 (plan 014 Phase 2, prereg_v9_a3 P2-audit): the v9 flood_trace protocol on the
# Phase-2 site change maps; items carry their ``site`` and ``repeats_per_site`` repeats per site.
P2_TASK = "flood_trace_p2"
TASKS = ("recall_points", "precision_pool", "flood_trace", "precision_microsoft", P2_TASK)
ALLOCATED_TASKS = ("recall_points", "precision_pool", "flood_trace")   # in audit/allocation.yaml
NEARMISS_TASK = "recall_nearmiss"
SERVE_TASKS = (*TASKS, NEARMISS_TASK)
FOLLOWUP_TASKS = ("damage",)
# Follow-up "tasks" for summaries and agreement: name -> (log task, question in that log).
FOLLOWUPS = {"damage": ("flood_trace", "damage"), "damage_p2": (P2_TASK, "damage")}
AUTHOR = "author"
ANSWERS = ("yes", "no", "unsure")
PILOT_TASK = "recall_points"


class AuditError(RuntimeError):
    """The audit cannot proceed as asked (missing inputs, frozen files, broken blindness)."""


def load_audit_config(project: Project) -> dict[str, Any]:
    path = project.paths.resolve(AUDIT_CONFIG)
    if not path.is_file():
        raise AuditError(f"{AUDIT_CONFIG} is missing")
    return load_yaml(path)


def task_config(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    tasks = cfg.get("tasks") or {}
    if task not in tasks:
        raise AuditError(f"unknown audit task {task!r}; one of {sorted(tasks)}")
    return tasks[task]


def task_dir(project: Project, task: str) -> Path:
    return project.paths.root / "audit" / task


def items_path(project: Project, task: str) -> Path:
    return task_dir(project, task) / "items.parquet"


def responses_path(project: Project, task: str) -> Path:
    return task_dir(project, task) / "responses.jsonl"


REDO_OF = "pass1"     # the one archived pass a redo may replace


def archive_path(project: Project, task: str) -> Path:
    """Answers of the first pass of a redone sitting (``responses_pass1.jsonl``)."""
    return task_dir(project, task) / f"responses_{REDO_OF}.jsonl"


def interim_dir(project: Project, task: str) -> Path:
    return project.paths.data_root / "interim" / "audit" / task


def chip_dir(project: Project, task: str) -> Path:
    return interim_dir(project, task) / "chips"


def examples_dir(project: Project, task: str) -> Path:
    return interim_dir(project, task) / "examples"


def allocation_path(project: Project) -> Path:
    return project.paths.root / ALLOCATION


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chip_name(task: str, item_id: str) -> str:
    """Opaque chip file name (no stratum, source or order in it)."""
    return hashlib.sha1(f"{task}:{item_id}".encode()).hexdigest()[:16] + ".jpg"


def load_items(project: Project, task: str) -> pd.DataFrame:
    path = items_path(project, task)
    if not path.is_file():
        raise AuditError(f"{path} is missing: run `nakkhu audit sample --task {task}`")
    return pd.read_parquet(path)


def load_allocation(project: Project) -> dict[str, Any] | None:
    path = allocation_path(project)
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- response log -----------------------------------------------------------------------------


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON line and fsync it before returning (a crash loses nothing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a torn last line after a crash: ignored, never repaired
    return out


@dataclass
class Replay:
    """Effective answers after replaying answer/undo events in order."""

    answers: dict[tuple[str, str], dict[str, Any]]   # (slot, question) -> answer event
    stack: list[tuple[str, str]]                     # effective answers, oldest first

    @property
    def effective(self) -> list[dict[str, Any]]:
        return [self.answers[key] for key in self.stack]


def replay(events: list[dict[str, Any]]) -> Replay:
    answers: dict[tuple[str, str], dict[str, Any]] = {}
    stack: list[tuple[str, str]] = []
    for event in events:
        kind = event.get("event")
        if kind == "answer":
            key = (str(event["slot"]), str(event["question"]))
            if key in answers:
                stack.remove(key)
            answers[key] = event
            stack.append(key)
        elif kind == "undo":
            key = (str(event["slot"]), str(event["question"]))
            if key in answers:
                del answers[key]
                stack.remove(key)
    return Replay(answers=answers, stack=stack)


def logged_ms(events: list[dict[str, Any]], auditor: str | None = None,
              default: str | None = None) -> int:
    """Time spent on items: every answer event counts (also answers later undone); with
    ``auditor`` only the answers of that auditor (resolved by :func:`event_auditor`)."""
    who = sitting_auditors(events)
    return int(sum(int(e.get("ms") or 0) for e in events if e.get("event") == "answer"
                   and (auditor is None or event_auditor(e, who, default) == auditor)))


def task_auditor(cfg: dict[str, Any], task: str) -> str | None:
    value = (cfg.get("tasks") or {}).get(task, {}).get("auditor")
    return str(value) if value else None


def sitting_auditors(events: list[dict[str, Any]]) -> dict[str, str]:
    """Sitting id -> auditor from ``auditor`` events (annotations appended after a sitting) and
    from the ``auditor`` field of start/answer events."""
    out: dict[str, str] = {}
    for e in events:
        if e.get("auditor") and e.get("sitting") and e.get("event") in ("auditor", "start",
                                                                         "answer"):
            out.setdefault(str(e["sitting"]), str(e["auditor"]))
    return out


def event_auditor(event: dict[str, Any], sittings: dict[str, str],
                  default: str | None = None) -> str | None:
    if event.get("auditor"):
        return str(event["auditor"])
    return sittings.get(str(event.get("sitting")), default)


def auditors_of(events: list[dict[str, Any]], default: str | None = None) -> list[str]:
    """Distinct auditors of the answer events (unknown -> ``"unrecorded"``)."""
    who = sitting_auditors(events)
    return sorted({event_auditor(e, who, default) or "unrecorded"
                   for e in events if e.get("event") == "answer"})


def cumulative_ms(project: Project, auditor: str = AUTHOR) -> int:
    """Logged time of one auditor over every task (the 120 min cap counts the author only),
    archived first passes of redone sittings included (that time was spent)."""
    try:
        cfg = load_audit_config(project)
    except AuditError:
        cfg = {}
    return sum(logged_ms(read_events(path), auditor, task_auditor(cfg, t) or AUTHOR)
               for t in SERVE_TASKS
               for path in (responses_path(project, t), archive_path(project, t)))


# --- redo / restart of a whole sitting -----------------------------------------------------------
# Both move pass 1 to ``responses_pass1.jsonl`` once and open a fresh log with a header event.
# ``redo`` (``--redo``): the page shows each slot's pass-1 answer (non-blind).
# ``restart`` (``--restart``): pass 1 is void; the same queue is served blind, nothing of pass 1
# reaches the page, and the header records ``restart_of``, ``prior_answer_shown: false`` and the
# reason.
PASS_MODES = {"redo": True, "restart": False}        # header event -> prior answer shown
PASS_FLAG = {"redo": "--redo", "restart": "--restart"}


def pass_of(header: dict[str, Any]) -> str:
    """The archived pass a redo/restart header replaces (``pass1``)."""
    return str(header.get("redo_of") or header.get("restart_of"))


def prior_shown(header: dict[str, Any] | None) -> bool:
    return bool(header) and bool(header.get("prior_answer_shown"))


def pass_fields(header: dict[str, Any] | None) -> dict[str, Any]:
    """Fields every event of a redo/restart carries (empty for a first pass)."""
    if not header:
        return {}
    return {f"{header['event']}_of": pass_of(header),
            "prior_answer_shown": prior_shown(header)}


def redo_state(project: Project, task: str) -> dict[str, Any] | None:
    """The ``redo`` or ``restart`` event opening ``responses.jsonl`` if the task's sitting is
    being redone or restarted, ``None`` for a first pass. Refuses an archive without such a log
    (or the reverse): passes never mix."""
    archive = archive_path(project, task)
    events = read_events(responses_path(project, task))
    header = events[0] if events and events[0].get("event") in PASS_MODES else None
    if any(e.get("event") in PASS_MODES for e in events[1 if header else 0:]):
        raise AuditError(f"{task}: a redo/restart event is not only the first line of "
                         "responses.jsonl")
    if header is not None and not archive.is_file():
        raise AuditError(f"{task}: responses.jsonl is a {header['event']} but {archive.name} is "
                         "missing")
    if header is None and archive.is_file():
        raise AuditError(f"{task}: {archive.name} exists but responses.jsonl is not a redo or "
                         f"restart of it; run `nakkhu audit serve --task {task} --restart` "
                         "(or --redo)")
    return header


def any_answers(project: Project, task: str) -> bool:
    """True if the task has an answer in its log or in an archived pass."""
    return any(e.get("event") == "answer"
               for path in (responses_path(project, task), archive_path(project, task))
               for e in read_events(path))


def start_pass(project: Project, task: str, auditor: str | None, mode: str = "redo",
               reason: str | None = None) -> dict[str, Any]:
    """Begin (once) or resume the redo (``mode="redo"``, pass-1 answers shown) or blind restart
    (``mode="restart"``, nothing of pass 1 shown; ``reason`` required) of a whole sitting: move
    ``responses.jsonl`` to ``responses_pass1.jsonl`` unchanged and open a new log with a
    ``redo``/``restart`` event. The archive is never overwritten; a second redo or restart, or
    resuming one mode with the other flag, is refused."""
    if mode not in PASS_MODES:
        raise AuditError(f"unknown pass mode {mode!r}; one of {sorted(PASS_MODES)}")
    log, archive = responses_path(project, task), archive_path(project, task)
    events = read_events(log)
    if events and events[0].get("event") in PASS_MODES:
        header = redo_state(project, task) or {}
        if header.get("event") != mode:                   # never switch blind <-> non-blind
            raise AuditError(f"{task}: a {header.get('event')} is in progress; resume it with "
                             f"`nakkhu audit serve --task {task} "
                             f"{PASS_FLAG[str(header.get('event'))]}`, not {PASS_FLAG[mode]}")
        return header                                     # in progress: resume
    if not auditor:
        raise AuditError(f"tasks.{task}.auditor is not set in {AUDIT_CONFIG}")
    reason = (reason or "").strip()
    if mode == "restart" and not reason:
        raise AuditError("a restart needs --reason (why pass 1 is void), recorded in the log")
    if archive.is_file():
        if log.is_file() and log.stat().st_size:
            raise AuditError(f"{archive} already exists and responses.jsonl is not a redo or "
                             "restart of it; refusing to overwrite an archived pass")
    else:
        if not any(e.get("event") == "answer" for e in events):
            raise AuditError(f"{task} has no answers: nothing to {mode} (serve without "
                             f"{PASS_FLAG[mode]})")
        log.rename(archive)                               # refuses an existing target
    if log.is_file():                                     # an empty log left by a crash
        log.unlink()
    header: dict[str, Any] = {
        "event": mode, f"{mode}_of": REDO_OF, "prior_answer_shown": PASS_MODES[mode],
        "archive": archive.name, "archive_sha256": sha256_file(archive),
        "items_sha256": sha256_file(items_path(project, task)),
        "auditor": auditor, "utc": utc_now()}
    if mode == "redo":
        header["note"] = ("whole sitting redone in the same queue order; the page showed the "
                          f"{REDO_OF} answer of every slot before it was answered (non-blind)")
    else:
        manifest = chip_dir(project, task) / "manifest.json"
        header.update(
            reason=reason,
            chips_manifest_sha256=sha256_file(manifest) if manifest.is_file() else None,
            note=(f"{REDO_OF} void; whole sitting served again from scratch in the same queue "
                  f"order, blind: no {REDO_OF} answer reaches the page and estimates use this "
                  "log only"))
    append_event(log, header)
    return header


def start_redo(project: Project, task: str, auditor: str | None) -> dict[str, Any]:
    """Non-blind redo (:func:`start_pass` with ``mode="redo"``)."""
    return start_pass(project, task, auditor, "redo")


def start_restart(project: Project, task: str, auditor: str | None,
                  reason: str | None) -> dict[str, Any]:
    """Blind restart (:func:`start_pass` with ``mode="restart"``)."""
    return start_pass(project, task, auditor, "restart", reason)


def prior_answers(project: Project, task: str) -> dict[tuple[str, str], str]:
    """(slot, question) -> effective answer of the archived pass (undo events honoured)."""
    state = replay(read_events(archive_path(project, task)))
    return {key: str(e["answer"]) for key, e in state.answers.items()}



# --- sizes and queue --------------------------------------------------------------------------


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Integer split of ``total`` proportional to ``weights`` (ties by sorted key)."""
    keys = sorted(weights)
    w = np.array([max(float(weights[k]), 0.0) for k in keys])
    if total <= 0 or w.sum() <= 0:
        return dict.fromkeys(keys, 0)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    rest = raw - base
    order = sorted(range(len(keys)), key=lambda i: (-rest[i], keys[i]))
    for i in order[: total - int(base.sum())]:
        base[i] += 1
    return {k: int(v) for k, v in zip(keys, base, strict=True)}


def stratum_sizes(items: pd.DataFrame, n_total: int) -> dict[str, int]:
    """Final per-stratum sizes m_h for a task total: proportional to the design sizes n_h,
    never below the pilot size p_h and never above n_h."""
    design = items.groupby("stratum")["design_n"].first().to_dict()
    pilot = items.loc[items["pilot"]].groupby("stratum").size().to_dict()
    floor = {k: int(pilot.get(k, 0)) for k in design}
    if n_total >= sum(design.values()):
        return {k: int(v) for k, v in design.items()}
    out = dict(floor)
    for _ in range(len(design) + 1):
        left = n_total - sum(out.values())
        free = {k: design[k] - out[k] for k in design if design[k] > out[k]}
        if left <= 0 or not free:
            break
        share = largest_remainder({k: float(design[k]) for k in free}, left)
        for k, v in share.items():
            out[k] += min(v, free[k])
    return out


@dataclass(frozen=True)
class Slot:
    slot: str
    item_id: str
    repeat: bool
    question: str | None = None     # recall_nearmiss: the item's own question (else the task)


def active_items(items: pd.DataFrame, allocation: dict[str, Any] | None,
                 task: str) -> pd.DataFrame:
    """Items in the final sample (rank < m_h); only the pilot before the allocation exists. A
    task the allocation does not list (drawn after it was written) keeps its design size."""
    if allocation is None:
        return items.loc[items["pilot"]]
    if task not in (allocation.get("tasks") or {}):
        return items
    total = int(allocation["tasks"][task]["n"])
    sizes = stratum_sizes(items, total)
    keep = items["rank"].to_numpy() < items["stratum"].map(sizes).to_numpy()
    return items.loc[keep | items["pilot"].to_numpy()]


def build_queue(items: pd.DataFrame, allocation: dict[str, Any] | None, task: str,
                cfg: dict[str, Any]) -> list[Slot]:
    if task == NEARMISS_TASK:
        ordered = items.sort_values("order_key")
        return [Slot(i, i, False, str(q)) for i, q in
                zip(ordered["item_id"], ordered["question"], strict=True)]
    if allocation is None and not items["pilot"].any():
        raise AuditError(f"{task} has no pilot items and no allocation: finish the pilot "
                         f"(sitting 1) first")
    active = active_items(items, allocation, task)
    pilot = active.loc[active["pilot"]].sort_values("order_key")
    rest = active.loc[~active["pilot"]].sort_values("order_key")
    primary = list(pilot["item_id"]) + list(rest["item_id"])
    slots = [Slot(i, i, False) for i in primary]
    if allocation is None:
        return slots
    rng = np.random.default_rng([int(cfg.get("seed", 0)), 6, TASKS.index(task)])
    groups = repeat_groups(items, task, cfg)
    if groups is not None:
        return insert_group_repeats(slots, rng, cfg, *groups, floor=len(pilot))
    return insert_repeats(slots, rng, cfg, floor=len(pilot))


def repeat_groups(items: pd.DataFrame, task: str, cfg: dict[str, Any]
                  ) -> tuple[dict[str, str], int] | None:
    """(item id -> group, repeats per group) for a task with ``repeats_per_site`` (the Phase-2
    audit: a fixed number of repeats per site), else None."""
    per = ((cfg.get("tasks") or {}).get(task) or {}).get("repeats_per_site")
    if per is None:
        return None
    if "site" not in items.columns:
        raise AuditError(f"{task}: repeats_per_site needs a `site` column in the items")
    group_of = dict(zip(items["item_id"].astype(str), items["site"].astype(str), strict=True))
    return group_of, int(per)


def insert_group_repeats(slots: list[Slot], rng: np.random.Generator, cfg: dict[str, Any],
                         group_of: dict[str, str], per_group: int, floor: int = 0
                         ) -> list[Slot]:
    """As :func:`insert_repeats`, but exactly ``per_group`` repeats drawn from the slots of each
    group (groups in sorted order; all of a smaller group)."""
    t = cfg.get("time") or {}
    n = len(slots)
    gap = int(t.get("repeat_min_gap", 20))
    inserts: dict[int, list[Slot]] = {}
    for group in sorted(set(group_of.values())):
        pos = [i for i, s in enumerate(slots) if group_of.get(s.item_id) == group]
        if not pos:
            continue
        chosen = sorted(int(p) for p in rng.choice(pos, size=min(per_group, len(pos)),
                                                   replace=False))
        for p in chosen:
            lo = min(max(p + gap, floor), n - 1)
            q = int(rng.integers(lo, n)) if lo < n - 1 else n - 1
            orig = slots[p]
            inserts.setdefault(q, []).append(Slot(f"{orig.item_id}#r", orig.item_id, True,
                                                  orig.question))
    out: list[Slot] = []
    for i, slot in enumerate(slots):
        out.append(slot)
        out.extend(inserts.get(i, []))
    return out


def insert_repeats(slots: list[Slot], rng: np.random.Generator, cfg: dict[str, Any],
                   floor: int = 0) -> list[Slot]:
    """Insert ``ceil(repeat_frac * n)`` repeat slots (``<item_id>#r``), each at least
    ``repeat_min_gap`` slots after its original and never before position ``floor`` (the
    pilot). The draw order of ``rng`` is fixed: the author's frozen queues depend on it."""
    t = cfg.get("time") or {}
    n = len(slots)
    n_rep = int(math.ceil(float(t.get("repeat_frac", 0.05)) * n))
    gap = int(t.get("repeat_min_gap", 20))
    chosen = sorted(rng.choice(n, size=min(n_rep, n), replace=False))
    inserts: dict[int, list[Slot]] = {}
    for p in chosen:
        lo = min(max(int(p) + gap, floor), n - 1)
        q = int(rng.integers(lo, n)) if lo < n - 1 else n - 1
        orig = slots[p]
        inserts.setdefault(q, []).append(Slot(f"{orig.item_id}#r", orig.item_id, True,
                                              orig.question))
    out: list[Slot] = []
    for i, slot in enumerate(slots):
        out.append(slot)
        out.extend(inserts.get(i, []))
    return out
