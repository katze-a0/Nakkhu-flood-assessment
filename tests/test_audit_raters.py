"""Remote raters (plan 006 follow-up): tokens, blind payload, per-rater storage and order,
pace warning, kappa, and the share helper (cloudflared mocked; nothing is tunnelled)."""

from __future__ import annotations

import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import yaml
from PIL import Image

from nakkhu.audit import core, estimators, raters, sample
from nakkhu.audit.core import AuditError

TASK = "flood_trace"
CFG = {
    "schema_version": 1, "seed": 0,
    "chips": {"context_m": 64.0, "upscale": 2, "jpeg_quality": 85, "prefetch": 3},
    "time": {"cap_min": 120, "sitting_max_min": 35, "sitting_budget_min": 30, "pilot_items": 6,
             "floor_per_task": 10, "repeat_frac": 0.1, "repeat_min_gap": 3},
    "tasks": {
        "flood_trace": {"question": "Trace?", "keys": {"yes": "Yes", "no": "No",
                                                        "unsure": "Not observable"},
                        "design_n": 30, "min_per_stratum": 2, "followup": "damage",
                        "auditor": "author"},
        "damage": {"question": "Damage?", "keys": {"yes": "Yes", "no": "No",
                                                    "unsure": "Unsure"}},
    },
}
# Columns a rater must never see (class, site, stratum, HAND, distance).
SECRET_COLUMNS = ("stratum", "site", "hand_m", "dist_m", "change_class")


def _items():
    import pandas as pd

    rows, unit = [], 0
    for stratum, n in {"positive": 40, "negative_near": 120, "negative_far": 300}.items():
        for _ in range(n):
            rows.append({"unit": unit, "stratum": stratum, "x": 1000.25 + unit,
                         "y": 2000.75 + unit})
            unit += 1
    frame = pd.DataFrame(rows)
    design = sample.design_sizes(frame.groupby("stratum").size().to_dict(), 30, 2)
    items = sample.finish_items(sample.draw(frame, design, 0, TASK), TASK, 0, 0)
    items["site"] = "siteZebra"
    items["hand_m"] = 3.217
    items["dist_m"] = 123.456
    items["change_class"] = np.where(items["stratum"] == "positive", "flooded_cls", "dry_cls")
    return items


@pytest.fixture
def rproject(project):
    root = project.paths.root
    (root / "configs" / "audit").mkdir(parents=True)
    (root / "configs" / "audit" / "audit.yaml").write_text(yaml.safe_dump(CFG), encoding="utf-8")
    items = _items()
    path = core.items_path(project, TASK)
    path.parent.mkdir(parents=True)
    items.to_parquet(path, index=False)
    core.allocation_path(project).write_text(yaml.safe_dump(
        {"pilot": {"median_s": 2.0}, "tasks": {TASK: {"n": 30}}}), encoding="utf-8")
    chips = core.chip_dir(project, TASK)
    chips.mkdir(parents=True)
    for item in items["item_id"]:
        Image.new("RGB", (8, 8)).save(chips / core.chip_name(TASK, item))
    return project


def _invite(project, rater, **kw):
    return raters.invite(project, CFG, TASK, rater, **kw)


def _rater_session(project, rater="rater_2"):
    from nakkhu.audit.server import Session

    inv = core_invite(project, rater)
    return Session(project, TASK, CFG, inv)


def core_invite(project, rater):
    found = raters.load_invites(project, TASK).get((TASK, rater))
    if found is None or not found.token_sha256:
        _invite(project, rater)
        found = raters.load_invites(project, TASK)[(TASK, rater)]
    return found


def _all_bytes(directory):
    return b"".join(p.read_bytes() for p in directory.rglob("*") if p.is_file())


# --- tokens -----------------------------------------------------------------------------------


def test_invite_stores_only_a_hash_and_the_latest_token_wins(rproject):
    token, rec = _invite(rproject, "rater_2")
    assert token and len(token) >= 32 and raters.TOKEN_RE.fullmatch(token)
    assert len(token.encode()) * 6 // 8 >= 16                       # >= 128 bits
    runs_bytes = _all_bytes(rproject.paths.runs)
    assert token.encode() not in runs_bytes                          # never written
    inv = raters.load_invites(rproject, TASK)[(TASK, "rater_2")]
    assert inv.token_sha256 == raters.token_hash(token) and rec["token_id"] == inv.token_id
    run_dir = rproject.paths.runs / rec["run"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert inv.token_sha256 not in json.dumps(summary)               # tracked file: id only
    assert summary["token_id"] == inv.token_id
    with pytest.raises(AuditError, match="already has a valid invite"):
        _invite(rproject, "rater_2")
    new, rec2 = _invite(rproject, "rater_2", replace=True)
    active = raters.active_invites(rproject, TASK)
    assert raters.find_invite(active, token) is None                  # old token dead
    assert raters.find_invite(active, new).rater == "rater_2"
    assert rec2["supersedes"] == rec["token_id"]
    gone, _ = _invite(rproject, "rater_2", revoke=True)
    assert gone is None and raters.find_invite(raters.active_invites(rproject, TASK), new) is None
    for bad in ("author", "Rater", "r", "x" * 40, "../etc", "rater-2"):
        with pytest.raises(AuditError):
            _invite(rproject, bad)


def test_cli_invite_prints_the_token_once(rproject, capsys):
    from nakkhu.cli import main

    root = str(rproject.paths.root)
    assert main(["--root", root, "audit", "invite", "--task", TASK, "--rater", "rater_3"]) == 0
    out = capsys.readouterr().out
    inv = raters.load_invites(rproject, TASK)[(TASK, "rater_3")]
    tokens = [w for w in out.split() if raters.token_hash(w.strip("/").split("/")[-1])
              == inv.token_sha256 or raters.token_hash(w) == inv.token_sha256]
    assert tokens and inv.token_id in out
    assert main(["--root", root, "audit", "invite", "--task", TASK, "--rater", "rater_3"]) == 1


# --- queue and storage ------------------------------------------------------------------------


def test_each_rater_gets_the_same_items_in_their_own_order_with_own_repeats(rproject):
    items = core.load_items(rproject, TASK)
    alloc = core.load_allocation(rproject)
    author = core.build_queue(items, alloc, TASK, CFG)
    a = raters.rater_queue(items, alloc, TASK, CFG, "rater_2")
    b = raters.rater_queue(items, alloc, TASK, CFG, "rater_3")
    assert raters.rater_queue(items, alloc, TASK, CFG, "rater_2") == a      # seeded per rater
    prim = {s.item_id for s in author if not s.repeat}
    for q in (a, b):
        assert sorted(s.item_id for s in q if not s.repeat) == sorted(prim)
        reps = [(i, s) for i, s in enumerate(q) if s.repeat]
        assert len(reps) == math.ceil(0.1 * len(prim))
        for pos, slot in reps:
            orig = next(i for i, s in enumerate(q) if s.slot == slot.item_id)
            assert pos > orig
    order = [[s.slot for s in q if not s.repeat] for q in (author, a, b)]
    assert order[0] != order[1] != order[2] != order[0]
    assert {s.item_id for s in a if s.repeat} != {s.item_id for s in b if s.repeat}


def test_rater_answers_stay_in_the_rater_log(rproject):
    s = _rater_session(rproject, "rater_2")
    st = s.state()
    for _ in range(4):
        st = s.answer(st["pos"], st["qid"], "no", 12_000)
    log = raters.rater_log(rproject, TASK, "rater_2")
    events = core.read_events(log)
    answers = [e for e in events if e["event"] == "answer"]
    assert len(answers) == 4
    assert all(e["auditor"] == e["rater"] == "rater_2" and e["token_id"] for e in events)
    assert all("server_ms" in e for e in answers)
    assert not core.responses_path(rproject, TASK).exists()          # author log untouched
    assert core.cumulative_ms(rproject) == 0                         # not author time
    from nakkhu.audit.workflow import primary_answers

    assert primary_answers(rproject, TASK) == {}                     # estimators: author only
    assert raters.rater_names(rproject, TASK) == ["rater_2"]
    s2 = _rater_session(rproject, "rater_2")                         # resumes
    assert s2.state()["done"] == 4


# --- blindness --------------------------------------------------------------------------------


def _leaks(body: str, items) -> list[str]:
    words = [*items["item_id"], *set(items["stratum"]), *SECRET_COLUMNS, "siteZebra",
             "flooded_cls", "dry_cls", "3.217", "123.456", "item_id", "slot", "repeat",
             "frame_n", "pi_design", "order_key", '"task"', TASK, "#r", "rater_3", "author"]
    words += [f"{v:g}" for v in items["x"].head(5)]
    return [w for w in words if w in body]


def test_rater_payload_is_blind_through_a_whole_sitting(rproject):
    from nakkhu.audit.server import RATER_FIELDS

    _invite(rproject, "rater_3")                     # another rater: never visible
    items = core.load_items(rproject, TASK)
    s = _rater_session(rproject, "rater_2")
    st = s.state()
    seen_chips: dict[str, str] = {}
    answer = "yes"
    while st["phase"] == "run":
        body = json.dumps(st)
        assert set(st) <= RATER_FIELDS and not _leaks(body, items), _leaks(body, items)
        assert st["chip"].startswith("chip/") and st["remote"] and st["pace"]["target_s"]
        assert all(not e["src"].startswith("/") for e in st["examples"])
        slot = s.queue[st["pos"]]
        if st["qid"] == "main":
            assert st["chip"] != f"chip/{core.chip_name(TASK, slot.item_id)}"
            seen_chips[slot.slot] = st["chip"]
        st = s.answer(st["pos"], st["qid"], answer, 11_000)
        answer = {"yes": "no", "no": "unsure", "unsure": "yes"}[answer]
    assert st["phase"] == "done" and set(st) <= RATER_FIELDS and not _leaks(json.dumps(st), items)
    repeats = [sl for sl in s.queue if sl.repeat]
    assert repeats and all(seen_chips[r.slot] != seen_chips[r.item_id] for r in repeats)


# --- HTTP: tokens, headers, rate limits -----------------------------------------------------------


def _serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def _get(url, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def test_requests_without_a_valid_token_get_404(rproject):
    from nakkhu.audit.server import RaterHub, make_rater_server

    token, _ = _invite(rproject, "rater_2")
    dead, _ = _invite(rproject, "rater_3")
    _invite(rproject, "rater_3", revoke=True)
    hub = RaterHub(rproject, TASK, CFG, limiter=raters.RateLimiter(max_failures=1000),
                   refresh_s=0.0)
    server = make_rater_server(hub, 0)
    assert server.server_address[0] == "127.0.0.1"
    base = _serve(server)
    items = core.load_items(rproject, TASK)
    real_chip = core.chip_name(TASK, items["item_id"].iloc[0])
    try:
        code, body, headers = _get(f"{base}/r/{token}/api/state")
        assert code == 200 and headers["X-Robots-Tag"].startswith("noindex")
        state = json.loads(body)
        assert not _leaks(body.decode(), items)
        code, chip, headers = _get(f"{base}/r/{token}/{state['chip']}")
        assert code == 200 and chip[:2] == b"\xff\xd8"
        assert "no-store" in headers["Cache-Control"] and "noindex" in headers["X-Robots-Tag"]
        assert headers["Referrer-Policy"] == "no-referrer"
        code, page, _ = _get(f"{base}/r/{token}/")
        assert code == 200 and b"<title>" in page
        code, _, headers = _get(f"{base}/r/{token}")
        assert code == 308 and headers["Location"] == f"/r/{token}/"
        wrong = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
        refused = ["/", "/index.html", "/api/state", f"/chip/{real_chip}", "/r/", "/r//api/state",
                   f"/r/{wrong}/api/state", f"/r/{wrong}/{state['chip']}", f"/r/{dead}/",
                   f"/r/{dead}/api/state", f"/r/{token.upper()}/api/state",
                   f"/r/{token}/chip/", f"/r/{token}/chip/{real_chip}", f"/r/{token}/chip/..%2F",
                   f"/r/{token}/../items.parquet", f"/r/{token}/audit/", f"/r/{token}/static/"]
        for path in refused:
            code, body, headers = _get(base + path)
            assert (code, body) == (404, b"not found"), path
            assert "noindex" in headers["X-Robots-Tag"]
        for method in ("PUT", "DELETE", "OPTIONS", "PATCH"):
            assert _get(f"{base}/r/{token}/api/state", method=method)[0] == 404
        code, _, _ = _get(f"{base}/api/answer", "POST", json.dumps({"pos": 0}).encode())
        assert code == 404
        code, _, _ = _get(f"{base}/r/{token}/api/answer", "POST", b"x" * 5000,
                          {"Content-Type": "application/json"})
        assert code == 413
        code, body, _ = _get(f"{base}/r/{token}/api/answer", "POST", json.dumps(
            {"pos": state["pos"], "qid": state["qid"], "answer": "no", "ms": 9000}).encode())
        assert code == 200 and json.loads(body)["done"] == 1
        assert not core.responses_path(rproject, TASK).exists()
    finally:
        server.shutdown()
        server.server_close()


def test_rate_limits_block_guessing_and_floods(rproject):
    from nakkhu.audit.server import RaterHub, make_rater_server

    token, _ = _invite(rproject, "rater_2")
    now = [0.0]
    limiter = raters.RateLimiter(per_min=50, max_failures=3, block_s=600.0, clock=lambda: now[0])
    server = make_rater_server(RaterHub(rproject, TASK, CFG, limiter=limiter), 0)
    base = _serve(server)
    try:
        for _ in range(5):                                                   # not guesses
            assert _get(f"{base}/favicon.ico")[0] == 404
        code, body, _ = _get(f"{base}/robots.txt")
        assert code == 200 and b"Disallow: /" in body
        assert _get(f"{base}/r/{token}/api/state")[0] == 200
        for _ in range(3):
            assert _get(f"{base}/r/{'A' * 32}/api/state")[0] == 404
        assert _get(f"{base}/r/{token}/api/state")[0] == 404                 # blocked client
        assert _get(f"{base}/r/{token}/api/state",
                    headers={"Cf-Connecting-Ip": "203.0.113.9"})[0] == 200   # another client
        now[0] = 601.0                                                       # block expires
        assert _get(f"{base}/r/{token}/api/state")[0] == 200
        codes = [_get(f"{base}/r/{token}/api/state")[0] for _ in range(60)]
        assert 429 in codes and codes.count(200) <= 50
    finally:
        server.shutdown()
        server.server_close()


def test_author_page_refuses_tunnelled_requests(rproject):
    from nakkhu.audit.server import Session, make_server

    server = make_server(Session(rproject, TASK, CFG), 0)
    base = _serve(server)
    try:
        assert _get(f"{base}/api/state")[0] == 200
        for header in ("Cf-Connecting-Ip", "Cf-Ray", "X-Forwarded-For"):
            code, body, _ = _get(f"{base}/api/state", headers={header: "198.51.100.7"})
            assert (code, body) == (404, b"not found")
    finally:
        server.shutdown()
        server.server_close()


# --- pace -------------------------------------------------------------------------------------


def test_fast_answers_get_a_soft_warning_once_per_window(rproject):
    s = _rater_session(rproject)
    st = s.state()
    flags, messages = [], []
    for _ in range(8):
        st = s.answer(st["pos"], st["qid"], "no", 1200)
        flags.append(st["pace_warning"])
        messages.append(st["message"])
    assert flags == [False, False, True, False, False, False, False, True]
    assert "10-15 seconds" in messages[2] and not messages[3]
    assert "10-15 seconds" in s.state()["pace"]["text"]
    events = core.read_events(raters.rater_log(rproject, TASK, "rater_2"))
    assert sum(e["event"] == "pace_warning" for e in events) == 2
    assert sum(e["event"] == "answer" for e in events) == 8                # never refused
    from nakkhu.audit.server import Session

    author = Session(rproject, TASK, CFG)
    st = author.state()
    for _ in range(5):
        st = author.answer(st["pos"], st["qid"], "no", 900)
        assert "pace_warning" not in st and "pace" not in st


# --- kappa ------------------------------------------------------------------------------------


def test_cohen_and_fleiss_kappa_known_values():
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    k = estimators.kappa_ci(a, b, n_boot=400, seed=1)
    assert k["kappa"] == pytest.approx(0.4) and k["agreement"] == pytest.approx(0.7)
    assert k["ci_lo"] < 0.4 < k["ci_hi"] and k["n"] == 50
    # Fleiss (1971) worked example as on Wikipedia: 10 items, 14 raters, 5 categories.
    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                       [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                       [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert estimators.fleiss_from_counts(counts) == pytest.approx(0.20993, abs=1e-4)
    rows = [[c for c, n in zip("abcde", r, strict=True) for _ in range(n)] for r in counts]
    f = estimators.fleiss_kappa(rows, n_boot=300)
    assert f["kappa"] == pytest.approx(0.20993, abs=1e-4) and f["raters"] == 14
    assert f["ci_lo"] <= f["kappa"] <= f["ci_hi"]
    rng = np.random.default_rng(0)
    truth = rng.choice(["yes", "no"], 200)
    noisy = {r: dict(enumerate(np.where(rng.random(200) < 0.1,
                                        rng.choice(["yes", "no", "unsure"], 200), truth)))
             for r in ("author", "rater_2", "rater_3")}
    out = estimators.multi_rater_agreement(noisy, n_boot=200)
    assert set(out["pairwise"]) == {"author|rater_2", "author|rater_3", "rater_2|rater_3"}
    assert all(0.6 < v["all"]["kappa"] < 1 for v in out["pairwise"].values())
    assert out["fleiss"]["all"]["n"] == 200 and 0.6 < out["fleiss"]["all"]["kappa"] < 1
    assert estimators.multi_rater_agreement({k: noisy[k] for k in ("author", "rater_2")},
                                            n_boot=50)["fleiss"] is None
    same = estimators.kappa_ci(list(truth), list(truth), n_boot=50)
    assert same["kappa"] == pytest.approx(1.0)


def test_agreement_report_keeps_raters_apart(rproject):
    from nakkhu.audit.server import Session
    from nakkhu.audit.workflow import agreement

    def run(session, pick):
        st = session.state()
        while st["phase"] == "run":
            slot = session.queue[st["pos"]]
            st = session.answer(st["pos"], st["qid"], pick(slot.item_id, st["qid"]), 10_000)

    def truth(item, qid):
        return "no" if qid == "followup" else ("yes" if int(item[-2:]) % 3 else "no")

    run(Session(rproject, TASK, CFG), truth)
    run(_rater_session(rproject, "rater_2"), truth)
    run(_rater_session(rproject, "rater_3"),
        lambda i, q: "unsure" if int(i[-2:]) % 7 == 0 and q == "main" else truth(i, q))
    out = agreement(rproject, TASK, n_boot=100)
    assert out["pre_declared_rater"] == "author"
    assert set(out["raters"]) == {"author", "rater_2", "rater_3"}
    assert all(r["complete"] for r in out["raters"].values())
    assert out["raters"]["rater_2"]["self_agreement"]["repeats"] == 3
    pw = out["inter_rater"]["pairwise"]
    assert pw["author|rater_2"]["all"]["kappa"] == pytest.approx(1.0)
    assert pw["author|rater_3"]["all"]["kappa"] < 1.0
    assert out["inter_rater"]["fleiss"]["all"]["raters"] == 3
    assert (rproject.paths.artifacts / "audit" / f"{TASK}_agreement.json").is_file()
    from nakkhu.audit.workflow import status

    st = status(rproject)["tasks"][TASK]
    assert set(st["raters"]) == {"rater_2", "rater_3"}
    assert st["raters"]["rater_2"]["answered"] == st["raters"]["rater_2"]["queue"]


# --- share (cloudflared mocked) ---------------------------------------------------------------


class FakeTunnel:
    def __init__(self, text, exits=False):
        self.stdout = io.StringIO(text)
        self.pid = 4_000_000
        self.returncode = 0 if exits else None
        self.terminated = False
        self.args = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminate()


CF_OUT = ("2026-10-17T10:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
          "2026-10-17T10:00:01Z INF |  https://brave-lucky-otter-tide.trycloudflare.com  |\n"
          "2026-10-17T10:00:02Z INF GET /r/SECRETtokenSECRETtokenSECRETtok/api/state\n")


def _port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def test_share_starts_server_and_tunnel_and_stops_both(rproject):
    from nakkhu.audit import share as SH

    _invite(rproject, "rater_2")
    fake = FakeTunnel(CF_OUT)
    calls, lines, during = [], [], {}
    stop = threading.Event()

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return fake

    def out(line):
        lines.append(line)
        if line.startswith("Public base URL"):
            during.update(json.loads(SH.pidfile(rproject).read_text(encoding="utf-8")))
            during["port_open"] = _port_open(during["port"])
            stop.set()

    code = SH.share(rproject, TASK, 0, cloudflared="cloudflared-x", popen=popen, stop=stop,
                    out=out)
    assert code == 0
    cmd, kwargs = calls[0]
    assert cmd[:2] == ["cloudflared-x", "tunnel"] and cmd[-2] == "--url"
    assert cmd[-1] == f"http://127.0.0.1:{during['port']}"
    assert kwargs["stdin"] == subprocess.DEVNULL
    if sys.platform == "win32":
        assert kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    assert "Public base URL: https://brave-lucky-otter-tide.trycloudflare.com" in lines
    assert during["server_pid"] == os.getpid() and during["tunnel_pid"] == fake.pid
    assert during["url"].endswith(".trycloudflare.com") and during["port_open"]
    assert during["raters"] == {"rater_2": raters.active_invites(rproject, TASK)[0].token_id}
    assert fake.terminated and not SH.pidfile(rproject).exists()
    assert not _port_open(during["port"])
    log = SH.tunnel_log(rproject).read_text(encoding="utf-8")
    assert "SECRETtoken" not in log and "/r/<redacted>" in log


def test_share_refuses_while_a_previous_share_runs(rproject):
    from nakkhu.audit import share as SH

    _invite(rproject, "rater_2")
    path = SH.pidfile(rproject)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": TASK, "port": 8766, "server_pid": os.getpid(),
                                "tunnel_pid": None}), encoding="utf-8")
    with pytest.raises(AuditError, match="still running"):
        SH.share(rproject, TASK, 0, cloudflared="x", popen=lambda *a, **k: pytest.fail("ran"))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert SH.pid_alive(os.getpid()) and not SH.pid_alive(dead.pid) and not SH.pid_alive(0)
    path.write_text(json.dumps({"server_pid": dead.pid, "tunnel_pid": None}), encoding="utf-8")
    assert SH.previous_share(rproject) is None                       # stale: allowed
    fake = FakeTunnel("INF no url here\n", exits=True)
    assert SH.share(rproject, TASK, 0, cloudflared="x", popen=lambda *a, **k: fake,
                    out=lambda line: None, url_timeout=2.0) == 1
    assert not path.exists()


def test_share_needs_an_invite(rproject):
    from nakkhu.audit import share as SH

    with pytest.raises(AuditError, match="no rater is invited"):
        SH.share(rproject, TASK, 0, cloudflared="x", popen=lambda *a, **k: pytest.fail("ran"))
    assert not SH.pidfile(rproject).exists()


def test_serve_only_starts_the_token_server_without_a_tunnel(rproject):
    from nakkhu.audit import share as SH

    token, _ = _invite(rproject, "rater_2")
    stop, during, lines = threading.Event(), {}, []

    def out(line):
        lines.append(line)
        if line.startswith("Ctrl+C"):
            during.update(json.loads(SH.pidfile(rproject).read_text(encoding="utf-8")))
            base = f"http://127.0.0.1:{during['port']}"
            during["page"] = _get(f"{base}/r/{token}/")[0]
            during["no_token"] = _get(f"{base}/")[0]
            stop.set()

    assert SH.serve_only(rproject, TASK, 0, stop=stop, out=out) == 0
    assert during["tunnel_pid"] is None and during["tunnel"] == "external"
    assert during["page"] == 200 and during["no_token"] == 404
    assert not any(token in line for line in lines) and not SH.pidfile(rproject).exists()
