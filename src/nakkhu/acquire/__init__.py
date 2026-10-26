"""Data acquisition (plan 002): ``nakkhu data fetch <id> [--dry-run]``.

Static source metadata lives in ``configs/data/sources.yaml``; a real fetch fills
``{data_root}/raw/<id>/`` through a staging directory that is renamed into place only after the
fetcher succeeded, writes ``source.json`` (per-file URL, ETag, size, SHA-256) next to the data,
and records the entry in ``configs/data/manifest.yaml``. Re-running is a no-op while the entry
still verifies; interrupted downloads resume from ``{data_root}/raw/.cache/``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from nakkhu.acquire.fetchers import METHODS, Context, PlanRow
from nakkhu.acquire.http import session
from nakkhu.config import Project
from nakkhu.manifest import load_manifest, measure, save_manifest, verify_entry
from nakkhu.runs import atomic_write_json

SOURCES = "configs/data/sources.yaml"
ENTRY_TEXT_FIELDS = ("title", "url", "access", "licence", "attribution", "citation", "clip")


class SourceError(ValueError):
    """The source register is malformed or an id is unknown."""


def load_sources(project: Project) -> dict[str, dict[str, Any]]:
    path = project.paths.resolve(SOURCES)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources: dict[str, dict[str, Any]] = {"_stac": document.get("stac") or {}}
    for source in document.get("sources") or []:
        if source["id"] in sources:
            raise SourceError(f"duplicate source id {source['id']}")
        if source.get("method") not in METHODS:
            raise SourceError(f"{source['id']}: unknown method {source.get('method')!r}")
        sources[source["id"]] = source
    return sources


def source_ids(sources: dict[str, dict[str, Any]]) -> list[str]:
    return [key for key in sources if not key.startswith("_")]


def _manifest_path(project: Project) -> Path:
    section = project.config.values.get("paths") or {}
    return project.paths.resolve(section.get("manifest", "configs/data/manifest.yaml"))


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _context(project: Project, sources: dict[str, Any], source: dict[str, Any],
             staging: Path) -> Context:
    cache = project.paths.data_root / "raw" / ".cache"
    return Context(project=project, source=source, sources=sources, staging=staging,
                   cache=cache, s=session())


def dry_run(project: Project, ids: list[str]) -> tuple[list[tuple[str, PlanRow]], int]:
    sources = load_sources(project)
    rows: list[tuple[str, PlanRow]] = []
    for sid in ids:
        source = sources[sid]
        ctx = _context(project, sources, source, project.paths.data_root / "raw" / sid)
        for row in METHODS[source["method"]].plan(ctx):
            rows.append((sid, row))
    total = sum(row.size or 0 for _, row in rows if not row.partial)
    return rows, total


def _entry(source: dict[str, Any], retrieved: str, *, local_path: str | None,
           size: int | None, sha: str | None, notes: list[str], overrides: dict[str, Any],
           pending: bool = False) -> dict[str, Any]:
    date = retrieved[:10] if retrieved else dt.date.today().isoformat()
    entry: dict[str, Any] = {"id": source["id"]}
    for name in ENTRY_TEXT_FIELDS:
        entry[name] = str(overrides.get(name, source.get(name, ""))).replace("{date}", date)
    entry.update(bytes=size, sha256=sha, retrieved_utc=retrieved or None, local_path=local_path)
    entry["notes"] = " | ".join(n for n in [source.get("notes", ""), *notes] if n)
    if pending:
        entry["status"] = "pending"
        entry.update(bytes=None, sha256=None, retrieved_utc=None, local_path=None)
    return entry


def _upsert(project: Project, entry: dict[str, Any]) -> None:
    """Replace one manifest entry under an exclusive lock file (safe for parallel fetches)."""
    import time

    path = _manifest_path(project)
    lock = path.with_name(path.name + ".lock")
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        raise SourceError(f"manifest lock {lock} held for 60 s; remove it if stale")
    try:
        entries = [e for e in load_manifest(path) if e.get("id") != entry["id"]] + [entry]
        save_manifest(path, entries)
    finally:
        os.close(fd)
        lock.unlink()


def _current(project: Project, sid: str) -> dict[str, Any] | None:
    for entry in load_manifest(_manifest_path(project)):
        if entry.get("id") == sid:
            return entry
    return None


def fetch(project: Project, sid: str, *, force: bool = False) -> dict[str, Any]:
    """Fetch one source and record it in the manifest; returns the manifest entry."""
    sources = load_sources(project)
    if sid not in sources or sid.startswith("_"):
        raise SourceError(f"unknown source id {sid!r}")
    source = sources[sid]
    method = METHODS[source["method"]]
    if method.fetch is None:
        entry = _entry(source, "", local_path=None, size=None, sha=None, notes=[],
                       overrides={}, pending=True)
        _upsert(project, entry)
        return entry

    existing = _current(project, sid)
    if existing and not force and existing.get("status") != "pending":
        if verify_entry(existing, project.paths).ok:
            print(f"{sid}: up to date (verified); use --force to refetch")
            return existing

    raw = project.paths.data_root / "raw"
    staging = raw / ".staging" / sid
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    ctx = _context(project, sources, source, staging)
    retrieved = _utc_now()
    result = method.fetch(ctx)

    if isinstance(result, Path):  # AOI methods write a config file directly
        shutil.rmtree(staging)
        target = result
    else:
        provenance = {"id": sid, "retrieved_utc": retrieved, "method": source["method"],
                      "params": source.get("params") or {}, **ctx.prov}
        atomic_write_json(staging / "source.json", provenance)
        target = raw / sid
        old = raw / ".staging" / f"{sid}.old"
        if old.exists():
            shutil.rmtree(old)
        if target.exists():
            os.replace(target, old)
        os.replace(staging, target)
        if old.exists():
            shutil.rmtree(old)
    size, sha = measure(target)
    entry = _entry(source, retrieved, local_path=project.paths.relative(target), size=size,
                   sha=sha, notes=ctx.notes, overrides=ctx.overrides)
    _upsert(project, entry)
    return entry


def provenance(project: Project, entry: dict[str, Any]) -> dict[str, Any]:
    """``source.json`` of a fetched directory entry (empty for file entries)."""
    if not entry.get("local_path"):
        return {}
    path = project.paths.resolve(entry["local_path"]) / "source.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
