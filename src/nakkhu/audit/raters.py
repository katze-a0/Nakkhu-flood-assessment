"""Remote raters: secret invite tokens, per-rater answer logs and queues, pace checks.

A second (or third) independent rater answers a task through ``nakkhu audit share`` (the
server on 127.0.0.1 behind a cloudflared quick tunnel). Rules:

- **Tokens.** ``nakkhu audit invite --task <t> --rater <name>`` draws a random token
  (:data:`TOKEN_BYTES` = 24 bytes, 192 bits, URL-safe), prints it once and stores only its
  SHA-256 in an invite run ``runs/audit-invite/<run>/invite.json`` (local; ``runs/**`` is not
  tracked). The tracked ``summary.json`` of that run holds the task, the rater and the token id
  (first 8 hex of the hash), never the token or its full hash. The latest invite of a
  (task, rater) pair is the only valid one: ``--replace`` issues a new token, ``--revoke``
  withdraws it. Nothing ever logs a token; logs and the console show the token id.
- **Storage.** A rater's events go to ``audit/<task>/raters/<rater>/responses.jsonl`` (tracked,
  raw evidence like the author's), with ``auditor`` = ``rater`` = the rater name. They never
  touch ``audit/<task>/responses.jsonl`` (the pre-declared rater of the task, whose answers the
  primary estimators read), never count towards the author's 120 min cap, and the page shows
  nothing about any other rater.
- **Queue.** Every rater gets the same slots as the author queue (same active items, same
  follow-up rule) in their own random order, seeded by (``seed``, task, rater name), with their
  own repeat items (same ``repeat_frac`` and ``repeat_min_gap``). Re-inviting a rater keeps
  the order and the log (both depend on the name, not on the token).
- **Pace.** The instruction card asks for about :data:`PACE_TARGET_S` seconds per image; when
  :data:`FAST_COUNT` of the last :data:`FAST_WINDOW` answers took under :data:`FAST_MS`, the
  page shows a soft warning (logged as a ``pace_warning`` event; answers are never refused).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.audit import core
from nakkhu.audit.core import AUTHOR, AuditError, Slot
from nakkhu.config import Project, ResolvedConfig
from nakkhu.runs import atomic_write_text, create_run, list_runs, utc_now

TOKEN_BYTES = 24                       # 192-bit tokens (>= 128 bits required)
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
RATER_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
INVITE_STAGE = "audit-invite"
INVITE_FILE = "invite.json"
RATERS_DIR = "raters"

PACE_TARGET_S = (10, 15)              # instruction card: about 10-15 s per image
FAST_MS = 3000                        # an answer under 3 s counts as fast
FAST_WINDOW = 5                       # warn if FAST_COUNT of the last FAST_WINDOW answers
FAST_COUNT = 3                        # were fast (soft warning, at most once per window)
PACE_MESSAGE = ("You are answering very fast. Please take about {lo}-{hi} seconds per image "
                "and look at it carefully before you answer.")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_id(digest: str) -> str:
    """Short public id of a token (from its hash; safe to log and track)."""
    return digest[:8]


def check_rater_name(cfg: dict[str, Any], task: str, rater: str) -> str:
    """A rater name is lower-case ``[a-z][a-z0-9_]``; it may not be ``author`` or the task's
    pre-declared auditor (whose answers live in ``responses.jsonl``)."""
    if not RATER_RE.fullmatch(rater or ""):
        raise AuditError(f"rater name {rater!r} must match {RATER_RE.pattern}")
    declared = core.task_auditor(cfg, task)
    if rater in (AUTHOR, declared):
        raise AuditError(f"{rater!r} is the pre-declared auditor of {task} (answers in "
                         "responses.jsonl); a remote rater needs another name")
    return rater


def rater_dir(project: Project, task: str, rater: str) -> Path:
    return core.task_dir(project, task) / RATERS_DIR / rater


def rater_log(project: Project, task: str, rater: str) -> Path:
    return rater_dir(project, task, rater) / "responses.jsonl"


def rater_names(project: Project, task: str) -> list[str]:
    """Raters with an answer log for the task (directory names under ``raters/``)."""
    base = core.task_dir(project, task) / RATERS_DIR
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and RATER_RE.fullmatch(p.name) and (p / "responses.jsonl").is_file())


# --- invites --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Invite:
    task: str
    rater: str
    token_sha256: str | None       # None: revoked
    created_ns: int
    run: str

    @property
    def token_id(self) -> str:
        return token_id(self.token_sha256 or "") if self.token_sha256 else "revoked"


def load_invites(project: Project, task: str | None = None) -> dict[tuple[str, str], Invite]:
    """(task, rater) -> latest invite (a revoked one included, with ``token_sha256`` None)."""
    latest: dict[tuple[str, str], Invite] = {}
    for run in list_runs(project.paths, INVITE_STAGE):
        path = run.path / INVITE_FILE
        if not path.is_file():
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task is not None and rec.get("task") != task:
            continue
        inv = Invite(task=str(rec["task"]), rater=str(rec["rater"]),
                     token_sha256=rec.get("token_sha256"), created_ns=int(rec["created_ns"]),
                     run=run.ref)
        key = (inv.task, inv.rater)
        if key not in latest or inv.created_ns > latest[key].created_ns:
            latest[key] = inv
    return latest


def active_invites(project: Project, task: str) -> list[Invite]:
    return sorted((i for i in load_invites(project, task).values() if i.token_sha256),
                  key=lambda i: i.rater)


def invite(project: Project, cfg: dict[str, Any], task: str, rater: str, *,
           replace: bool = False, revoke: bool = False) -> tuple[str | None, dict[str, Any]]:
    """Issue (or revoke) a rater's token for a task. Returns (token, record): the token is
    returned exactly once and never written anywhere; the record carries its token id."""
    core.task_config(cfg, task)
    if task not in core.SERVE_TASKS:
        raise AuditError(f"raters answer one of {core.SERVE_TASKS}")
    check_rater_name(cfg, task, rater)
    current = load_invites(project, task).get((task, rater))
    live = current is not None and current.token_sha256 is not None
    if revoke:
        if not live:
            raise AuditError(f"{rater} has no valid invite for {task}")
    elif live and not replace:
        raise AuditError(f"{rater} already has a valid invite for {task} (token id "
                         f"{current.token_id}); pass --replace to issue a new token (the old "
                         "one stops working)")
    token = None if revoke else secrets.token_urlsafe(TOKEN_BYTES)
    digest = token_hash(token) if token else None
    created = time.time_ns()
    public = {"task": task, "rater": rater, "token_id": token_id(digest) if digest else None,
              "action": "revoke" if revoke else ("replace" if live else "invite"),
              "supersedes": current.token_id if live else None}
    run = create_run(project, INVITE_STAGE, ResolvedConfig(
        values={"command": "audit-invite", **public, "created_ns": created},
        source=core.AUDIT_CONFIG))
    with run.step("invite"):
        atomic_write_text(run.path / INVITE_FILE, json.dumps(
            {**public, "token_sha256": digest, "created_ns": created, "utc": utc_now()},
            indent=1, sort_keys=True))
        run.write_summary(public)
    return token, {**public, "run": run.ref}


def find_invite(invites: list[Invite], token: str) -> Invite | None:
    """The invite whose hash matches ``token`` (constant-time comparison), else None."""
    if not TOKEN_RE.fullmatch(token or ""):
        return None
    digest = token_hash(token)
    found = None
    for inv in invites:
        if inv.token_sha256 and hmac.compare_digest(inv.token_sha256, digest):
            found = inv
    return found


# --- per-rater queue ------------------------------------------------------------------------


def rater_seed(cfg: dict[str, Any], task: str, rater: str) -> list[int]:
    salt = int.from_bytes(hashlib.sha256(f"{task}\0{rater}".encode()).digest()[:8], "big")
    return [int(cfg.get("seed", 0)), 8, salt]


def rater_queue(items: pd.DataFrame, allocation: dict[str, Any] | None, task: str,
                cfg: dict[str, Any], rater: str) -> list[Slot]:
    """The author queue's primary slots in the rater's own seeded order, with the rater's own
    repeats (only if the author queue has repeats; never for ``recall_nearmiss``)."""
    if allocation is None and task != core.NEARMISS_TASK:
        raise AuditError(f"{task}: raters need {core.ALLOCATION} (the author's pilot sets the "
                         "sizes first)")
    base = core.build_queue(items, allocation, task, cfg)
    primary = [s for s in base if not s.repeat]
    rng = np.random.default_rng(rater_seed(cfg, task, rater))
    ordered = [primary[i] for i in rng.permutation(len(primary))]
    if not any(s.repeat for s in base):
        return ordered
    groups = core.repeat_groups(items, task, cfg)
    if groups is not None:
        return core.insert_group_repeats(ordered, rng, cfg, *groups, floor=0)
    return core.insert_repeats(ordered, rng, cfg, floor=0)


def chip_alias(key: str, slot: str) -> str:
    """Opaque per-rater, per-slot chip name: a repeat gets a name of its own, so not even the
    URL tells a rater that an image comes back."""
    return hmac.new(key.encode(), slot.encode(), hashlib.sha256).hexdigest()[:20] + ".jpg"


# --- pace -----------------------------------------------------------------------------------


def pace_text() -> str:
    lo, hi = PACE_TARGET_S
    return (f"Take your time: about {lo}-{hi} seconds per image is right. Look at the whole "
            "image before you answer.")


def fast_answers(ms: list[int]) -> int:
    """Answers under :data:`FAST_MS` among the last :data:`FAST_WINDOW`."""
    return sum(int(m) < FAST_MS for m in ms[-FAST_WINDOW:])


# --- rate limiting --------------------------------------------------------------------------


class RateLimiter:
    """Per-client sliding windows: at most ``per_min`` requests a minute, and a client with
    ``max_failures`` requests without a valid token inside ``block_s`` is blocked (404) for the
    rest of that window. Thread-safe; ``clock`` is injectable for tests."""

    def __init__(self, per_min: int = 600, max_failures: int = 20, block_s: float = 600.0,
                 clock: Any = time.monotonic) -> None:
        self.per_min, self.max_failures, self.block_s = per_min, max_failures, block_s
        self.clock = clock
        self.lock = threading.Lock()
        self.hits: dict[str, deque[float]] = {}
        self.fails: dict[str, deque[float]] = {}

    @staticmethod
    def _trim(q: deque[float], since: float) -> None:
        while q and q[0] < since:
            q.popleft()

    def allow(self, client: str) -> bool:
        with self.lock:
            now = self.clock()
            q = self.hits.setdefault(client, deque())
            self._trim(q, now - 60.0)
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True

    def fail(self, client: str) -> None:
        with self.lock:
            self.fails.setdefault(client, deque()).append(self.clock())

    def blocked(self, client: str) -> bool:
        with self.lock:
            q = self.fails.get(client)
            if not q:
                return False
            self._trim(q, self.clock() - self.block_s)
            return len(q) >= self.max_failures
