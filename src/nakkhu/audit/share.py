"""``nakkhu audit share``: the rater server on 127.0.0.1 behind a cloudflared quick tunnel.

One process serves every invited rater of one task (:class:`server.RaterHub`; a token is needed
for every request) and runs ``cloudflared tunnel --url http://127.0.0.1:<port>`` as a child.
The public base URL (``https://<random>.trycloudflare.com``) is printed; a rater's link is
``<base>/r/<token>/`` with the token printed once by ``nakkhu audit invite``. Ctrl+C (or the
tunnel exiting) stops both. The PIDs go to a pidfile (``{data_root}/interim/audit/share.json``,
local only); a second share refuses to start while the recorded server or tunnel is alive. The
cloudflared log goes to ``share-cloudflared.log`` beside it, with any ``/r/<token>`` redacted.

Quick tunnels are Cloudflare's free, account-less test tunnels: no uptime guarantee, a new
random hostname on every start (so raters need the new base URL after a restart; their tokens
stay valid) and a limit on concurrent requests. The author's own ``serve`` page is tokenless and
refuses tunnelled requests, so tunnelling its port exposes nothing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nakkhu.audit import core
from nakkhu.audit.core import AuditError
from nakkhu.audit.server import HOST, RaterHub, make_rater_server
from nakkhu.config import Project
from nakkhu.runs import atomic_write_text, utc_now

DEFAULT_PORT = 8766            # the author's serve uses 8765
URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")
REDACT_RE = re.compile(r"/r/[A-Za-z0-9_-]{8,}")
URL_TIMEOUT_S = 60.0


def pidfile(project: Project) -> Path:
    return project.paths.data_root / "interim" / "audit" / "share.json"


def tunnel_log(project: Project) -> Path:
    return pidfile(project).with_name("share-cloudflared.log")


def pid_alive(pid: int | None) -> bool:
    """True if a process with this PID is running (never signals it)."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259                              # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def previous_share(project: Project) -> dict[str, Any] | None:
    """The recorded share if its server or tunnel is still alive, else None (stale file)."""
    path = pidfile(project)
    if not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    alive = [k for k in ("server_pid", "tunnel_pid") if pid_alive(rec.get(k))]
    return {**rec, "alive": alive} if alive else None


def _write_pidfile(project: Project, rec: dict[str, Any]) -> None:
    atomic_write_text(pidfile(project), json.dumps(rec, indent=1, sort_keys=True))


def find_cloudflared(explicit: str | None = None) -> str:
    exe = explicit or shutil.which("cloudflared")
    if not exe:
        raise AuditError("cloudflared is not on PATH; pass --cloudflared <path>")
    return exe


def tunnel_command(exe: str, port: int) -> list[str]:
    return [exe, "tunnel", "--no-autoupdate", "--url", f"http://{HOST}:{port}"]


class TunnelReader:
    """Drains the tunnel's output into the redacted log and catches the public URL."""

    def __init__(self, stream: Any, log: Path) -> None:
        self.url: str | None = None
        self.found = threading.Event()
        self.closed = threading.Event()
        self.stream, self.log = stream, log
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as out:
            try:
                for line in iter(self.stream.readline, ""):
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", "replace")
                    if not line:
                        break
                    out.write(REDACT_RE.sub("/r/<redacted>", line))
                    out.flush()
                    match = URL_RE.search(line)
                    if match and self.url is None:
                        self.url = match.group(0)
                        self.found.set()
            except (OSError, ValueError):
                pass
        self.closed.set()


def stop_tunnel(proc: Any, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def serve_only(project: Project, task: str, port: int = DEFAULT_PORT, *,
               stop: threading.Event | None = None, out: Callable[[str], None] = print) -> int:
    """The rater server of ``task`` on 127.0.0.1 without a tunnel (the user runs their own
    tunnel to ``http://127.0.0.1:<port>``); every request still needs a valid token. Runs
    until Ctrl+C or ``stop``."""
    running = previous_share(project)
    if running:
        raise AuditError(f"a share is still running ({running.get('task')} on port "
                         f"{running.get('port')}, alive: {', '.join(running['alive'])}; pidfile "
                         f"{pidfile(project)}): stop it first (Ctrl+C in its terminal)")
    cfg = core.load_audit_config(project)
    hub = RaterHub(project, task, cfg, notify=out)
    server = make_rater_server(hub, port)
    stop = stop or threading.Event()
    rec: dict[str, Any] = {"task": task, "port": server.server_address[1],
                           "server_pid": os.getpid(), "tunnel_pid": None,
                           "started_utc": utc_now(), "url": None, "tunnel": "external",
                           "raters": {i.rater: i.token_id for i in hub.invites}}
    _write_pidfile(project, rec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    out(f"Rater server for '{task}' on http://{HOST}:{rec['port']}/ (no tunnel started).")
    out(f"Point your tunnel at http://{HOST}:{rec['port']}; each rater opens "
        "<tunnel URL>/r/<their token>/")
    for inv in hub.invites:
        out(f"  {inv.rater} (token id {inv.token_id})")
    out("Ctrl+C stops the server; the same command resumes.")
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        current = None
        try:
            current = json.loads(pidfile(project).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if current and current.get("server_pid") == os.getpid():
            pidfile(project).unlink(missing_ok=True)
        out("stopped (server)")
    return 0


def share(project: Project, task: str, port: int = DEFAULT_PORT, *,
          cloudflared: str | None = None, popen: Callable[..., Any] = subprocess.Popen,
          stop: threading.Event | None = None, out: Callable[[str], None] = print,
          url_timeout: float = URL_TIMEOUT_S) -> int:
    """Serve the invited raters of ``task`` through a quick tunnel until Ctrl+C, ``stop`` is
    set or the tunnel exits. Returns 0 on a clean stop, 1 if the tunnel failed."""
    running = previous_share(project)
    if running:
        raise AuditError(f"a share is still running ({running.get('task')} on port "
                         f"{running.get('port')}, alive: {', '.join(running['alive'])}; pidfile "
                         f"{pidfile(project)}): stop it first (Ctrl+C in its terminal)")
    exe = find_cloudflared(cloudflared)
    cfg = core.load_audit_config(project)
    hub = RaterHub(project, task, cfg, notify=out)
    server = make_rater_server(hub, port)
    stop = stop or threading.Event()
    rec: dict[str, Any] = {"task": task, "port": server.server_address[1],
                           "server_pid": os.getpid(), "tunnel_pid": None,
                           "started_utc": utc_now(), "url": None,
                           "raters": {i.rater: i.token_id for i in hub.invites}}
    _write_pidfile(project, rec)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proc = None
    code = 0
    try:
        kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE,
                                  "stderr": subprocess.STDOUT, "text": True,
                                  "encoding": "utf-8", "errors": "replace"}
        if sys.platform == "win32":     # Ctrl+C reaches only us; we stop the tunnel ourselves
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = popen(tunnel_command(exe, rec["port"]), **kwargs)
        rec["tunnel_pid"] = proc.pid
        _write_pidfile(project, rec)
        reader = TunnelReader(proc.stdout, tunnel_log(project))
        waited = 0.0
        while not reader.found.wait(0.5):
            waited += 0.5
            if proc.poll() is not None or reader.closed.is_set() or waited >= url_timeout \
                    or stop.is_set():
                out(f"error: cloudflared gave no public URL (see {tunnel_log(project)})")
                return 1
        rec["url"] = reader.url
        _write_pidfile(project, rec)
        out(f"Sharing '{task}' on http://{HOST}:{rec['port']}/ through {reader.url}")
        out(f"Public base URL: {reader.url}")
        for inv in hub.invites:
            out(f"  {inv.rater} (token id {inv.token_id}): {reader.url}/r/<their token>/")
        out("Send each rater their own link over a private channel. Ctrl+C stops the server "
            "and the tunnel; the same command resumes (with a new base URL).")
        while not stop.wait(1.0):
            if proc.poll() is not None:
                out(f"cloudflared exited (code {proc.returncode}); stopping")
                code = 1
                break
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            stop_tunnel(proc)
        server.shutdown()
        server.server_close()
        current = None
        try:
            current = json.loads(pidfile(project).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if current and current.get("server_pid") == os.getpid():
            pidfile(project).unlink(missing_ok=True)
        out("stopped (server and tunnel)")
    return code
