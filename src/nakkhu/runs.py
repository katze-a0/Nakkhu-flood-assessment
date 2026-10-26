"""Run directories: resolved config, environment record, stage journal, atomic writes.

Layout: ``runs/<stage>/<YYYYMMDDTHHMMSSZ>-<cfghash>/`` holding ``config.resolved.yaml``,
``env.json``, ``journal.jsonl`` and the stage outputs. Only ``summary.json`` is tracked in git.
A run is *complete* when its journal has at least one step and every started step ended with
return code 0.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from nakkhu.config import Project, ResolvedConfig
from nakkhu.paths import ProjectPaths

CONFIG_FILE = "config.resolved.yaml"
ENV_FILE = "env.json"
JOURNAL_FILE = "journal.jsonl"
SUMMARY_FILE = "summary.json"

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write via a temp file in the same directory, then ``Path.replace`` it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


# --- environment record ---------------------------------------------------------------------


def _run(*command: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_info(root: Path) -> dict[str, Any]:
    sha = _run("git", "rev-parse", "HEAD", cwd=root)
    status = _run("git", "status", "--porcelain", cwd=root)
    return {"sha": sha, "dirty": None if status is None else bool(status)}


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_name() -> str:
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _ram_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    with contextlib.suppress(ValueError, OSError, AttributeError):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return None


def _gpus() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = _run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    gpus = []
    for line in (out or "").splitlines():
        name, _, driver = line.partition(",")
        if name.strip():
            gpus.append({"name": name.strip(), "driver_version": driver.strip()})
    return gpus


def collect_env(root: Path) -> dict[str, Any]:
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }
    return {
        "recorded_utc": utc_now(),
        "git": _git_info(root),
        "uv_lock_sha256": _file_sha256(root / "uv.lock"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {"name": _cpu_name(), "logical_cores": os.cpu_count()},
        "ram_bytes": _ram_bytes(),
        "gpus": _gpus(),
    }


# --- journal --------------------------------------------------------------------------------


def read_journal(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / JOURNAL_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_journal(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / JOURNAL_FILE
    existing = path.read_bytes() if path.is_file() else b""
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_bytes(path, existing + line.encode("utf-8"))


def journal_status(run_dir: str | Path) -> str:
    """``complete``, ``running`` (a step never ended), ``failed`` (non-zero rc) or ``empty``."""
    records = read_journal(run_dir)
    if not records:
        return "empty"
    open_steps: dict[str, int] = {}
    failed = False
    for record in records:
        step = record.get("stage")
        if not isinstance(step, str):
            continue
        if record.get("event") == "start":
            open_steps[step] = open_steps.get(step, 0) + 1
        elif record.get("event") == "end":
            open_steps[step] = open_steps.get(step, 0) - 1
            failed = failed or record.get("returncode") != 0
    if any(count != 0 for count in open_steps.values()):
        return "running"
    return "failed" if failed else "complete"


def is_complete(run_dir: str | Path) -> bool:
    return journal_status(run_dir) == "complete"


# --- runs -----------------------------------------------------------------------------------


def _check_name(kind: str, name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError(f"{kind} name {name!r} must match {_NAME.pattern}")
    return name


@dataclass(frozen=True)
class Run:
    stage: str
    run_id: str
    path: Path

    @property
    def ref(self) -> str:
        return f"{self.stage}/{self.run_id}"

    def start(self, step: str) -> str:
        started = utc_now()
        _append_journal(self.path, {"event": "start", "stage": step, "utc": started})
        return started

    def end(self, step: str, returncode: int, started_utc: str | None = None) -> None:
        record = {"event": "end", "stage": step, "utc": utc_now(), "returncode": returncode}
        if started_utc is not None:
            record["start_utc"] = started_utc
        _append_journal(self.path, record)

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Journal a step: ``start`` on entry, ``end`` with rc 0 or 1 on exit."""
        started = self.start(_check_name("step", name))
        try:
            yield
        except BaseException:
            self.end(name, 1, started)
            raise
        self.end(name, 0, started)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        path = self.path / SUMMARY_FILE
        atomic_write_json(path, summary)
        return path

    @property
    def status(self) -> str:
        return journal_status(self.path)


def create_run(project: Project, stage: str, config: ResolvedConfig | None = None) -> Run:
    """Create ``runs/<stage>/<utc>-<cfghash>/`` with the resolved config and environment."""
    config = config or project.config
    _check_name("stage", stage)
    run_id = f"{utc_stamp()}-{config.cfghash}"
    path = project.paths.runs / stage / run_id
    path.mkdir(parents=True, exist_ok=False)
    atomic_write_text(path / CONFIG_FILE, config.to_yaml())
    env = collect_env(project.paths.root)
    env["config"] = {
        "source": config.source,
        "overrides": list(config.overrides),
        "sha256": config.sha256,
        "cfghash": config.cfghash,
    }
    atomic_write_json(path / ENV_FILE, env)
    return Run(stage=stage, run_id=run_id, path=path)


def find_run_dir(path: str | Path, paths: ProjectPaths) -> Path | None:
    """The run directory (``runs/<stage>/<run_id>``) containing ``path``, if any."""
    resolved = Path(path).resolve()
    runs = paths.runs.resolve()
    if runs not in resolved.parents:
        return None
    parts = resolved.relative_to(runs).parts
    if len(parts) < 3:
        return None
    return runs / parts[0] / parts[1]


def list_runs(paths: ProjectPaths, stage: str | None = None) -> list[Run]:
    if not paths.runs.is_dir():
        return []
    stages = [paths.runs / stage] if stage else sorted(p for p in paths.runs.iterdir() if p.is_dir())
    runs = []
    for stage_dir in stages:
        if not stage_dir.is_dir():
            continue
        for run_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            runs.append(Run(stage=stage_dir.name, run_id=run_dir.name, path=run_dir))
    return runs
