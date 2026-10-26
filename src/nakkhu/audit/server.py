"""Keyboard-only click page on 127.0.0.1 (plan 006 Step 3); stdlib ``http.server`` only.

The page gets one chip, one question and the progress; never an item id, stratum, source,
model, score or the fact that an item is a repeat (slots are addressed by queue position).
Every answer is appended to ``audit/<task>/responses.jsonl`` and fsynced before the reply, so a
crash or a closed tab loses nothing; a restart resumes at the first unanswered slot. Backspace
undoes the last answer (an ``undo`` event; nothing is deleted). Every event carries the task's
``auditor`` (``tasks.<task>.auditor``; serving refuses without one).

``recall_nearmiss`` slots carry their own question: the circle question, where a Yes needs a
click on the roof inside the circle (offset from the centre logged in metres; a click outside
the circle is refused), or the original sitting-1 crosshair question (overlap items).

A redone sitting (``serve --redo``, :func:`core.start_redo`) serves the same queue from scratch
and, by the author's explicit choice (non-blind), shows each slot's archived pass-1 answer, and
its follow-up answer, above the buttons; every redo event records ``redo_of`` and
``prior_answer_shown`` and the answer that was shown. A restarted sitting (``serve --restart``)
serves the same queue from scratch blind: the archived pass is never read, the page payload is
that of a first pass (no ``prior``, ``redo`` is null), and every event records ``restart_of``
and ``prior_answer_shown: false``.

Remote raters (:mod:`nakkhu.audit.raters`, ``nakkhu audit share``) use a separate server
(:class:`RaterHub`, :func:`make_rater_server`, still bound to 127.0.0.1): every request must
carry a valid invite token in its path (``/r/<token>/...``, the page uses relative URLs) or it
gets 404, with no page, chip or item metadata. Each rater has their own :class:`Session` (own
log, own seeded order and repeats, opaque per-slot chip names), and their page payload is
filtered to :data:`RATER_FIELDS`. The author's tokenless page refuses requests that came
through a proxy or tunnel. Every response of both servers carries :data:`SECURITY_HEADERS`
(``no-store``, ``X-Robots-Tag: noindex``, no referrer, no framing).
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

import numpy as np

from nakkhu.audit import core, raters
from nakkhu.audit.chips import ChipStyle, metres_per_px
from nakkhu.audit.core import (
    ANSWERS,
    NEARMISS_TASK,
    PILOT_TASK,
    AuditError,
    append_event,
    build_queue,
    chip_dir,
    chip_name,
    examples_dir,
    load_allocation,
    load_items,
    read_events,
    replay,
    responses_path,
)
from nakkhu.config import Project
from nakkhu.runs import utc_now, utc_stamp

HOST = "127.0.0.1"
CHIP_RE = re.compile(r"^[0-9a-f]{16}\.jpg$")
EXAMPLE_RE = re.compile(r"^(yes|no|unsure)_\d+\.jpg$")

HELP = {
    "recall_points": [
        "Look only at the open centre of the crosshair.",
        "Yes: the centre is on the roof of any building: house, apartment, shed, factory, "
        "temple, a building under construction with a roof or top slab. Attached row houses "
        "count.",
        "No: road, yard, courtyard, field, trees, water, bare ground, a wall alone.",
        "Plastic greenhouse or tunnel = No (greenhouses are not buildings here).",
        "Unsure: you really cannot tell (deep shadow, haze, the centre exactly on a roof "
        "edge).",
        "Go with your first impression. There is no feedback and no score.",
    ],
    "precision_pool": [
        "Look at the outlined box.",
        "Yes: the box marks a real building, or a block of attached buildings.",
        "No: the box is on a road, field, trees, vehicle or empty ground.",
        "Plastic greenhouse or tunnel = No (greenhouses are not buildings here).",
        "Unsure: you really cannot tell.",
        "Go with your first impression. There is no feedback and no score.",
    ],
    "precision_microsoft": [
        "Look at the outlined footprint.",
        "Yes: the outline marks a real building, or a block of attached buildings.",
        "An outline shifted a few metres that clearly belongs to one building or block = Yes.",
        "No: the outline is on a road, field, trees, vehicle or empty ground.",
        "Plastic greenhouse or tunnel = No (greenhouses are not buildings here).",
        "Unsure: you really cannot tell.",
        "Go with your first impression. There is no feedback and no score.",
    ],
    NEARMISS_TASK: [
        "Two kinds of image are mixed. Read the question above each image.",
        "Images with a CIRCLE: is any part of a building roof inside the circle? The small mark "
        "is the centre.",
        "Yes: press J (or the Yes button), then CLICK on that roof inside the circle. If "
        "several roofs are inside, click the one nearest the centre. Backspace cancels.",
        "No: nothing inside the circle is a roof (road, yard, courtyard, field, trees, water, "
        "bare ground, a wall alone).",
        "Images with a CROSSHAIR only: is the open centre of the crosshair on a building roof? "
        "Yes: house, apartment, shed, factory, temple, a building under construction with a "
        "roof or top slab; attached row houses count. No: road, yard, field, trees, water, bare "
        "ground, a wall alone.",
        "Plastic greenhouse or tunnel = No (greenhouses are not buildings here).",
        "Unsure: you really cannot tell (deep shadow, haze).",
        "Go with your first impression. There is no feedback and no score.",
    ],
    "flood_trace": [
        "Left: before (Jan 2024). Right: after the flood (Oct 2024). Same place, same outline.",
        "Yes: fresh water, mud, sand or debris touches the outlined building after, and was "
        "not there before.",
        "No: nothing new touches the building.",
        "Not observable: cloud, shadow, blur or lean hides the building.",
        "After a Yes you get one more question about visible structural damage.",
    ],
    core.P2_TASK: [
        "Left: before (Jan 2024). Right: after the flood (Oct 2024). Same place, same outline.",
        "Yes: fresh water, mud, sand or debris touches the outlined building after, and was "
        "not there before.",
        "No: nothing new touches the building.",
        "Not observable: cloud, shadow, blur or lean hides the building.",
        "After a Yes you get one more question about visible structural damage.",
        "Take about 10-15 seconds per image: look at the whole image before you answer.",
    ],
}


# The only keys a remote rater's page receives (blindness whitelist; tested).
RATER_FIELDS = frozenset({
    "done", "total", "sitting_ms", "used_min", "cap_min", "sitting_max_min", "help",
    "examples", "pilot", "message", "can_undo", "redo", "remaining_min", "phase", "pos", "qid",
    "question", "keys", "click_on_yes", "chip", "prefetch", "click_refused", "remote", "pace",
    "pace_warning"})


class Session:
    """State of one serving process for one task (thread-safe).

    ``rater`` (an invite) makes it a remote rater's session: answers go to the rater's own log
    (:func:`raters.rater_log`), the queue is the rater's own order with their own repeats, no
    redo/restart and no pilot, the cap counts the rater's own time, chip URLs are opaque
    per-slot aliases, and :meth:`state` is filtered to :data:`RATER_FIELDS`.
    """

    def __init__(self, project: Project, task: str, cfg: dict[str, Any],
                 rater: raters.Invite | None = None) -> None:
        if task not in core.SERVE_TASKS:
            raise AuditError(f"serve one of {core.SERVE_TASKS}; damage is asked after "
                             "flood_trace")
        self.project, self.task, self.cfg = project, task, cfg
        self.tcfg = core.task_config(cfg, task)
        self.rater = rater
        if rater is not None:
            if rater.task != task or not rater.token_sha256:
                raise AuditError(f"invite of {rater.rater} is not a valid {task} invite")
            self.auditor: str | None = raters.check_rater_name(cfg, task, rater.rater)
        else:
            self.auditor = core.task_auditor(cfg, task)
        if not self.auditor:
            raise AuditError(f"tasks.{task}.auditor is not set in {core.AUDIT_CONFIG}: say who "
                             "answers this sitting before serving it")
        self.prefix = "" if rater else "/"      # rater pages live under /r/<token>/ (relative)
        self.followup = self.tcfg.get("followup")
        self.items = load_items(project, task)
        self.examples_task = PILOT_TASK if task == NEARMISS_TASK else task
        self.style = ChipStyle.from_config(cfg)
        self.radius: dict[str, float] = {}
        if "radius_m" in self.items:
            self.radius = {str(i): float(r) for i, r in
                           zip(self.items["item_id"], self.items["radius_m"], strict=True)
                           if r == r}
        if rater is None:
            self.log = responses_path(project, task)
            self.redo = core.redo_state(project, task)      # refuses a half-made redo/restart
            self.sitting = f"{task}-{utc_stamp()}"
        else:                                               # never the author's files
            self.log = raters.rater_log(project, task, rater.rater)
            self.redo = None
            self.sitting = f"{task}-{rater.rater}-{utc_stamp()}"
        self.show_prior = core.prior_shown(self.redo)    # False on a blind restart
        self.prior = core.prior_answers(project, task) if self.show_prior else {}
        self.lock = threading.Lock()
        self.chips = chip_dir(project, task)
        self.aliases: dict[str, str] = {}               # rater chip alias -> item id
        self._shown: dict[tuple[int, str], float] = {}  # rater: (pos, qid) -> first served
        self._warned_at = -raters.FAST_WINDOW
        self._reload()
        missing = [s.item_id for s in self.queue
                   if not (self.chips / chip_name(task, s.item_id)).is_file()]
        if missing:
            raise AuditError(f"{len(missing)} chips missing under {self.chips}: run "
                             f"`nakkhu audit chips --task {task}`")
        self._started = False

    def _log(self, event: dict[str, Any]) -> None:
        """Append an event (preceded once by this sitting's ``start``); reads never write."""
        if not self._started:
            start = {"event": "start", "sitting": self.sitting, "utc": utc_now(),
                     "auditor": self.auditor, **self._redo_fields(), **self._rater_fields()}
            append_event(self.log, start)
            self.events.append(start)
            self._started = True
        event = {**event, "auditor": self.auditor, **self._rater_fields()}
        append_event(self.log, event)
        self.events.append(event)

    def _rater_fields(self) -> dict[str, Any]:
        if self.rater is None:
            return {}
        return {"rater": self.rater.rater, "token_id": self.rater.token_id}

    def chip_url(self, slot: core.Slot) -> str:
        if self.rater is None:
            return f"/chip/{chip_name(self.task, slot.item_id)}"
        return f"chip/{raters.chip_alias(str(self.rater.token_sha256), slot.slot)}"

    def chip_file(self, name: str) -> str | None:
        """File name under :attr:`chips` for a requested chip name (rater: an alias)."""
        if self.rater is None:
            return name if CHIP_RE.match(name) else None
        item = self.aliases.get(name)
        return chip_name(self.task, item) if item is not None else None

    def _redo_fields(self) -> dict[str, Any]:
        return core.pass_fields(self.redo)

    def _prior(self, slot: str, qid: str, main: bool) -> list[dict[str, Any]]:
        """Archived answers shown on a redo slot: the shown question's, then (main question
        only) the follow-up's if pass 1 had one."""
        asked = [qid] + ([str(self.followup)] if main and self.followup else [])
        out = []
        for i, q in enumerate(asked):
            raw = self.prior.get((slot, q))
            if i and raw is None:
                continue
            label = "Previous answer" if not i else "Previous follow-up answer"
            out.append({"label": label, "value": raw,
                        "text": self._question(q)[1].get(raw, raw) if raw else "none"})
        return out

    # --- state ----------------------------------------------------------------------------

    def _reload(self) -> None:
        self.allocation = load_allocation(self.project)
        if self.rater is None:
            self.queue = build_queue(self.items, self.allocation, self.task, self.cfg)
        else:
            self.queue = raters.rater_queue(self.items, self.allocation, self.task, self.cfg,
                                            self.rater.rater)
            key = str(self.rater.token_sha256)
            self.aliases = {raters.chip_alias(key, s.slot): s.item_id for s in self.queue}
        self.events = read_events(self.log)

    def _main_question(self, pos: int) -> str:
        return self.queue[pos].question or self.task

    def _needs_click(self, qid: str) -> bool:
        return qid == NEARMISS_TASK

    def _expected(self) -> tuple[int, str] | None:
        state = replay(self.events)
        for pos, slot in enumerate(self.queue):
            main = slot.question or self.task
            first = state.answers.get((slot.slot, main))
            if first is None:
                return pos, main
            if self.followup and first["answer"] == "yes" \
                    and (slot.slot, self.followup) not in state.answers:
                return pos, str(self.followup)
        return None

    def _question(self, qid: str) -> tuple[str, dict[str, str]]:
        q = core.task_config(self.cfg, qid)
        return str(q["question"]), dict(q["keys"])

    def _median_ms(self) -> float | None:
        ms = [int(e["ms"]) for e in replay(self.events).effective
              if e["question"] == self.task or self.task == NEARMISS_TASK]
        if ms:
            return float(np.median(ms))
        if self.rater is not None:          # never another rater's (the pilot's) timing
            return None
        pilot = (self.allocation or {}).get("pilot") or {}
        return float(pilot["median_s"]) * 1000.0 if pilot.get("median_s") else None

    def _examples(self) -> list[dict[str, str]]:
        """Chosen examples with captions (``examples.json`` written by ``audit chips``)."""
        directory = examples_dir(self.project, self.examples_task)
        manifest = directory / "examples.json"
        if not manifest.is_file():
            return []
        out = []
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            name = str(entry["file"])
            if EXAMPLE_RE.match(name) and (directory / name).is_file():
                out.append({"src": f"{self.prefix}example/{name}",
                            "answer": str(entry["caption"])})
        return out

    def _paced(self) -> bool:
        """Pace guidance and the soft warning: every remote rater, and every rater of a task
        with ``pace_all_raters`` (the Phase-2 audit, the author included)."""
        return self.rater is not None or bool(self.tcfg.get("pace_all_raters"))

    def state(self, message: str = "") -> dict[str, Any]:
        out = self._state(message)
        if self.rater is None:
            if self._paced():
                out.update(pace={"target_s": list(raters.PACE_TARGET_S),
                                 "text": raters.pace_text()})
                out.setdefault("pace_warning", False)
            return out
        if out.get("phase") == "run":       # server-side time on item (first time served)
            self._shown.setdefault((int(out["pos"]), str(out["qid"])), time.monotonic())
        out.update(remote=True, pace={"target_s": list(raters.PACE_TARGET_S),
                                      "text": raters.pace_text()})
        out.setdefault("pace_warning", False)
        return {k: v for k, v in out.items() if k in RATER_FIELDS}

    def _state(self, message: str = "") -> dict[str, Any]:
        t = self.cfg.get("time") or {}
        # The cap counts the author's time over every task; a rater's own time on this task.
        used_ms = core.cumulative_ms(self.project) if self.rater is None \
            else core.logged_ms(self.events)
        cap_ms = float(t.get("cap_min", 120)) * 60_000
        sitting_ms = sum(int(e.get("ms") or 0) for e in self.events
                         if e.get("event") == "answer" and e.get("sitting") == self.sitting)
        state = replay(self.events)
        done = sum((s.slot, self.task) in state.answers for s in self.queue)
        expected = self._expected()
        out: dict[str, Any] = {
            "task": self.task, "done": done, "total": len(self.queue),
            "sitting_ms": sitting_ms, "used_min": used_ms / 60_000,
            "cap_min": float(t.get("cap_min", 120)),
            "sitting_max_min": float(t.get("sitting_max_min", 35)),
            "help": HELP.get(self.task, []), "examples": self._examples(),
            "pilot": self.allocation is None, "message": message,
            "can_undo": bool(state.stack),
            # A blind restart looks like a first pass on the page (nothing of pass 1 is sent).
            "redo": {"of": core.pass_of(self.redo)} if self.show_prior else None,
        }
        med = self._median_ms()
        out["remaining_min"] = None if med is None else (len(self.queue) - done) * med / 60_000
        if used_ms >= cap_ms:
            out["phase"] = "stopped"
            out["message"] = (f"The {t.get('cap_min', 120)} min total limit is reached. "
                              "Stop here; nothing more is needed from you.")
            return out
        if expected is None:
            if self.allocation is None and self.task == PILOT_TASK and self.rater is None:
                out["phase"] = "pilot_done"
            else:
                out["phase"] = "done"
            return out
        pos, qid = expected
        question, keys = self._question(qid)
        slot = self.queue[pos]
        prefetch_n = int((self.cfg.get("chips") or {}).get("prefetch", 10))
        nxt = [self.chip_url(s) for s in self.queue[pos + 1: pos + 1 + prefetch_n]]
        main = qid == self._main_question(pos)
        out.update(phase="run", pos=pos, qid="main" if main else "followup",
                   question=question, keys=keys, click_on_yes=self._needs_click(qid),
                   chip=self.chip_url(slot), prefetch=nxt)
        if self.show_prior:
            out["prior"] = self._prior(slot.slot, qid, main)
        return out

    # --- actions --------------------------------------------------------------------------

    def click_offset(self, item_id: str, click: Any) -> dict[str, Any] | str:
        """Click in chip pixels -> offset from the centre in metres (east, north), or the reason
        it is refused."""
        try:
            x, y = float(click["x_px"]), float(click["y_px"])
        except (KeyError, TypeError, ValueError):
            return "Click on the roof inside the circle (Backspace cancels)."
        half = self.style.size_px / 2.0
        mpp = metres_per_px(self.style)
        dx, dy = (x - half) * mpp, (half - y) * mpp
        dist = float(np.hypot(dx, dy))
        radius = self.radius.get(item_id)
        if radius is None or not np.isfinite(dist) or dist > radius:
            return "That click is outside the circle: click the roof inside the circle."
        return {"click_dx_m": round(dx, 3), "click_dy_m": round(dy, 3),
                "click_m": round(dist, 3), "click_px": [round(x, 1), round(y, 1)]}

    def answer(self, pos: int, qid: str, answer: str, ms: int,
               click: Any = None) -> dict[str, Any]:
        with self.lock:
            if answer not in ANSWERS:
                return self.state("unknown answer")
            if self.state()["phase"] == "stopped":
                return self.state()
            expected = self._expected()
            if not 0 <= pos < len(self.queue):
                return self.state()
            qname = self._main_question(pos) if qid == "main" else str(self.followup)
            if expected is None or expected != (pos, qname):
                return self.state()          # stale or double submit: ignored
            slot = self.queue[pos]
            event = {"event": "answer", "slot": slot.slot, "item_id": slot.item_id,
                     "repeat": slot.repeat, "question": qname, "answer": answer,
                     "ms": max(0, int(ms)), "utc": utc_now(), "sitting": self.sitting,
                     "pos": pos, **self._redo_fields()}
            if self.show_prior:
                event["prior_answer"] = self.prior.get((slot.slot, qname))
            if answer == "yes" and self._needs_click(qname):
                offset = self.click_offset(slot.item_id, click)
                if isinstance(offset, str):
                    out = self.state(offset)
                    out["click_refused"] = True
                    return out
                event.update(offset)
            if self.rater is not None:
                shown = self._shown.pop((pos, qid), None)
                if shown is not None:
                    event["server_ms"] = int((time.monotonic() - shown) * 1000)
            self._log(event)
            if self.rater is None and self.allocation is None and self._expected() is None \
                    and self.task == PILOT_TASK:
                return self._finish_pilot()
            warning = self._pace_check()
            if warning:
                out = self.state(warning)
                out["pace_warning"] = True
                return out
            return self.state()

    def _pace_check(self) -> str | None:
        """Soft pace warning (logged; the answer is kept as given) for the paced raters."""
        if not self._paced():
            return None
        mine = [e for e in self.events
                if e.get("event") == "answer" and e.get("sitting") == self.sitting]
        fast = raters.fast_answers([int(e.get("ms") or 0) for e in mine])
        if fast < raters.FAST_COUNT or len(mine) - self._warned_at < raters.FAST_WINDOW:
            return None
        self._warned_at = len(mine)
        self._log({"event": "pace_warning", "utc": utc_now(), "sitting": self.sitting,
                   "fast": fast, "window": raters.FAST_WINDOW, "fast_ms": raters.FAST_MS})
        return raters.PACE_MESSAGE.format(lo=raters.PACE_TARGET_S[0], hi=raters.PACE_TARGET_S[1])

    def _finish_pilot(self) -> dict[str, Any]:
        from nakkhu.audit.workflow import write_allocation

        alloc = write_allocation(self.project, self.cfg)
        self._log({"event": "allocation", "utc": utc_now(), "sitting": self.sitting,
                   "median_s": alloc["pilot"]["median_s"]})
        self._reload()
        task = alloc["tasks"][self.task]
        state = self.state()
        rest = state["total"] - state["done"]
        minutes = rest * alloc["pilot"]["median_s"] / 60.0
        state["phase"] = "pilot_done"
        state["message"] = (f"Pilot done: {alloc['pilot']['median_s']:.1f} s per image. "
                            f"{rest} images left in this task (about {minutes:.0f} min). "
                            f"Final size {task['n']} plus repeats.")
        return state

    def undo(self) -> dict[str, Any]:
        with self.lock:
            stack = replay(self.events).stack
            if not stack:
                return self.state("nothing to undo")
            slot, question = stack[-1]
            event = {"event": "undo", "slot": slot, "question": question, "utc": utc_now(),
                     "sitting": self.sitting}
            self._log(event)
            return self.state()

    def pause(self, paused: bool) -> dict[str, Any]:
        with self.lock:
            event = {"event": "pause" if paused else "resume", "utc": utc_now(),
                     "sitting": self.sitting}
            self._log(event)
            return self.state()


def page_html() -> bytes:
    return resources.files("nakkhu.audit").joinpath("static/index.html").read_bytes()


# Sent with every response of both servers. Chips are Maxar CC BY-NC: never cached, indexed or
# framed; the page loads nothing from elsewhere and sends no referrer.
SECURITY_HEADERS = {
    "Cache-Control": "no-store, private",       # also keeps Cloudflare's edge from caching
    "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet, noimageindex",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'none'; img-src 'self' blob: data:; connect-src 'self'; "
                                "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
}
# A request that came through a proxy or tunnel (cloudflared adds Cf-*) never reaches the
# author's tokenless page: it gets 404 like any unknown path.
PROXY_HEADERS = ("Cf-Connecting-Ip", "Cf-Ray", "Cdn-Loop", "X-Forwarded-For", "Forwarded",
                 "X-Real-Ip")
MAX_BODY = 4096
NOT_FOUND = b"not found"
TOKEN_PATH = re.compile(r"^/r/([^/?#]+)(/[^?#]*)?$")
ROBOTS = b"User-agent: *\nDisallow: /\n"
# Asked for by browsers and crawlers without a token: answered without counting as a guess.
NO_TOKEN_PATHS = ("/robots.txt", "/favicon.ico")


class _Handler(BaseHTTPRequestHandler):
    """Routes of the click page for one :class:`Session`, with :data:`SECURITY_HEADERS`; no
    directory listing (only names matching the chip/example patterns are served)."""

    server_version = "audit"
    sys_version = ""
    page = b""

    def log_message(self, format: str, *args: Any) -> None:  # quiet console (paths hold tokens)
        return

    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _not_found(self) -> None:
        self._send(HTTPStatus.NOT_FOUND, NOT_FOUND, "text/plain; charset=utf-8")

    def _json(self, value: Any) -> None:
        self._send(HTTPStatus.OK, json.dumps(value).encode("utf-8"), "application/json")

    def _body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if not 0 <= length <= MAX_BODY:
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        return body if isinstance(body, dict) else {}

    def _route(self, session: Session, method: str, route: str) -> None:
        if method == "GET":
            if route in ("/", "/index.html"):
                self._send(HTTPStatus.OK, self.page, "text/html; charset=utf-8")
            elif route == "/api/state":
                with session.lock:
                    self._json(session.state())
            elif route.startswith("/chip/"):
                name = session.chip_file(route[len("/chip/"):])
                path = session.chips / name if name else None
                if path is None or not path.is_file():
                    self._not_found()
                else:
                    self._send(HTTPStatus.OK, path.read_bytes(), "image/jpeg")
            elif route.startswith("/example/"):
                name = route[len("/example/"):]
                path = examples_dir(session.project, session.examples_task) / name
                if not EXAMPLE_RE.match(name) or not path.is_file():
                    self._not_found()
                else:
                    self._send(HTTPStatus.OK, path.read_bytes(), "image/jpeg")
            else:
                self._not_found()
            return
        if method != "POST" or route not in ("/api/answer", "/api/undo", "/api/pause"):
            self._not_found()
            return
        body = self._body()
        if body is None:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"too large", "text/plain")
            return
        if route == "/api/answer":
            try:
                pos, ms = int(body.get("pos", -1)), int(body.get("ms") or 0)
            except (TypeError, ValueError):
                self._send(HTTPStatus.BAD_REQUEST, b"bad request", "text/plain")
                return
            self._json(session.answer(pos, str(body.get("qid")), str(body.get("answer")), ms,
                                      body.get("click")))
        elif route == "/api/undo":
            self._json(session.undo())
        else:
            self._json(session.pause(bool(body.get("paused"))))

    def _dispatch(self, method: str) -> None:
        raise NotImplementedError

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._dispatch("OPTIONS")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    """The author's local page (no token): refuses anything that came through a proxy."""

    class Handler(_Handler):
        page = page_html()

        def _dispatch(self, method: str) -> None:
            if any(self.headers.get(h) for h in PROXY_HEADERS):
                self._not_found()
                return
            self._route(session, method, self.path.split("?", 1)[0])

    return Handler


class RaterHub:
    """Remote raters of one task: token -> that rater's :class:`Session`. Invites are re-read
    at most every ``refresh_s`` seconds, so a ``--replace`` or ``--revoke`` takes effect on a
    running share without a restart."""

    def __init__(self, project: Project, task: str, cfg: dict[str, Any],
                 limiter: raters.RateLimiter | None = None, notify: Any = None,
                 refresh_s: float = 5.0) -> None:
        self.project, self.task, self.cfg = project, task, cfg
        self.limiter = limiter or raters.RateLimiter()
        self.notify = notify or (lambda message: None)
        self.refresh_s = refresh_s
        self.lock = threading.Lock()
        self.invites = raters.active_invites(project, task)
        if not self.invites:
            raise AuditError(f"no rater is invited for {task}: run `nakkhu audit invite --task "
                             f"{task} --rater <name>` first")
        self.sessions = {inv.rater: Session(project, task, cfg, inv) for inv in self.invites}
        self.seen: set[str] = set()
        self._refreshed = time.monotonic()

    def _refresh(self) -> None:
        if time.monotonic() - self._refreshed < self.refresh_s:
            return
        self._refreshed = time.monotonic()
        self.invites = raters.active_invites(self.project, self.task)

    def session_for(self, token: str) -> Session | None:
        with self.lock:
            self._refresh()
            inv = raters.find_invite(self.invites, token)
            if inv is None:
                return None
            session = self.sessions.get(inv.rater)
            if session is None or session.rater != inv:
                session = Session(self.project, self.task, self.cfg, inv)
                self.sessions[inv.rater] = session
            if inv.token_id not in self.seen:
                self.seen.add(inv.token_id)
                self.notify(f"rater {inv.rater} (token id {inv.token_id}) opened the page")
            return session


def make_rater_handler(hub: RaterHub) -> type[BaseHTTPRequestHandler]:
    """Every request needs a valid token in the path (``/r/<token>/...``); anything else gets
    404 (no page, no chip, no item metadata). Rate limited per client (the ``Cf-Connecting-Ip``
    of a tunnelled request, else the socket address)."""

    class Handler(_Handler):
        page = page_html()

        def _dispatch(self, method: str) -> None:
            client = self.headers.get("Cf-Connecting-Ip") or self.client_address[0]
            if hub.limiter.blocked(client):
                self._not_found()
                return
            if not hub.limiter.allow(client):
                self._send(HTTPStatus.TOO_MANY_REQUESTS, b"too many requests", "text/plain",
                           {"Retry-After": "60"})
                return
            route = self.path.split("?", 1)[0]
            if route in NO_TOKEN_PATHS:
                if route == "/robots.txt" and method == "GET":
                    self._send(HTTPStatus.OK, ROBOTS, "text/plain")
                else:
                    self._not_found()
                return
            match = TOKEN_PATH.match(route)
            session = hub.session_for(match.group(1)) if match else None
            if match is None or session is None:
                hub.limiter.fail(client)
                self._not_found()
                return
            rest = match.group(2)
            if rest is None:                 # /r/<token> -> /r/<token>/ (relative URLs)
                if method != "GET":
                    self._not_found()
                    return
                self._send(HTTPStatus.PERMANENT_REDIRECT, b"", "text/plain",
                           {"Location": f"/r/{match.group(1)}/"})
                return
            self._route(session, method, rest)

    return Handler


def make_rater_server(hub: RaterHub, port: int) -> ThreadingHTTPServer:
    server = bind_server(port)
    server.RequestHandlerClass = make_rater_handler(hub)
    return server


EXCLUSIVE = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)    # Windows only


class AuditServer(ThreadingHTTPServer):
    """Refuses a port another server already listens on. On Windows, SO_REUSEADDR (the
    http.server default) lets a second process bind the same port while the browser keeps
    reaching the first one (a stale sitting's page); SO_EXCLUSIVEADDRUSE forbids that."""

    allow_reuse_address = EXCLUSIVE is None      # keep fast restarts on POSIX

    def server_bind(self) -> None:
        if EXCLUSIVE is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, EXCLUSIVE, 1)
        super().server_bind()


def bind_server(port: int) -> ThreadingHTTPServer:
    """Bind the port (no handler yet): ``serve --redo`` claims it before moving any file, so a
    still-running server on the same port stops the redo before it starts."""
    try:
        return AuditServer((HOST, port), BaseHTTPRequestHandler)
    except OSError as error:
        raise OSError(f"{HOST}:{port} is already in use ({error}); another audit server is "
                      "probably still running: stop it (Ctrl+C in its terminal) or pass "
                      "--port") from error


def attach(server: ThreadingHTTPServer, session: Session) -> ThreadingHTTPServer:
    server.RequestHandlerClass = make_handler(session)
    return server


def make_server(session: Session, port: int) -> ThreadingHTTPServer:
    return attach(bind_server(port), session)
