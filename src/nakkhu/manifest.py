"""Data manifest (``configs/data/manifest.yaml``): schema and SHA-256 verification.

Each entry describes one acquired dataset. ``local_path`` is a file or a directory; for a
directory, ``sha256`` is the tree digest (see :func:`sha256_tree`) and ``bytes`` the total size.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nakkhu.paths import PathError, ProjectPaths

REQUIRED_FIELDS = (
    "id",
    "title",
    "url",
    "access",
    "licence",
    "attribution",
    "citation",
    "bytes",
    "sha256",
    "retrieved_utc",
    "local_path",
    "clip",
    "notes",
)
ACCESS_VALUES = ("public", "registration", "request")
CLIP_VALUES = ("extent", "extent_buffer", "routing", "training", "none")
# Fields that must be present but may be empty.
OPTIONAL_VALUE_FIELDS = ("notes",)
# Fields that may be absent. ``status: pending`` marks a dataset that is registered but not yet
# acquired (e.g. it needs a user account); its measured fields stay empty and ``notes`` says why.
OPTIONAL_FIELDS = ("status",)
STATUS_VALUES = ("ok", "pending")
PENDING_EMPTY_FIELDS = ("bytes", "sha256", "retrieved_utc", "local_path")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


class ManifestError(ValueError):
    """The manifest file itself is unreadable or malformed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> tuple[int, str]:
    """Total bytes and digest of a directory: SHA-256 over sorted ``relpath\\tsha256\\n`` lines."""
    base = Path(path)
    digest = hashlib.sha256()
    total = 0
    for file in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = file.relative_to(base).as_posix()
        digest.update(f"{rel}\t{sha256_file(file)}\n".encode())
        total += file.stat().st_size
    return total, digest.hexdigest()


def measure(path: str | Path) -> tuple[int, str]:
    """``(bytes, sha256)`` of a file or directory as recorded in the manifest."""
    path = Path(path)
    if path.is_dir():
        return sha256_tree(path)
    return path.stat().st_size, sha256_file(path)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ManifestError(f"{path}: top level must be a mapping")
    entries = document.get("entries") or []
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ManifestError(f"{path}: 'entries' must be a list of mappings")
    return entries


def save_manifest(path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Rewrite the manifest atomically, keeping its leading comment block; entries sorted by id."""
    from nakkhu.runs import atomic_write_text

    path = Path(path)
    header: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("#"):
                break
            header.append(line)
    ordered = [
        {name: entry[name] for name in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS) if name in entry}
        for entry in sorted(entries, key=lambda e: str(e.get("id")))
    ]
    body = yaml.safe_dump(
        {"schema_version": 1, "entries": ordered}, sort_keys=False, allow_unicode=True, width=100
    )
    atomic_write_text(path, "\n".join(header) + ("\n" if header else "") + body)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def is_pending(entry: dict[str, Any]) -> bool:
    return entry.get("status") == "pending"


def validate_entry(entry: dict[str, Any]) -> list[str]:
    """Schema problems of one entry (empty list when valid)."""
    problems = []
    pending = is_pending(entry)
    may_be_empty = OPTIONAL_VALUE_FIELDS + (PENDING_EMPTY_FIELDS if pending else ())
    for name in REQUIRED_FIELDS:
        if name not in entry:
            problems.append(f"missing field '{name}'")
        elif name not in may_be_empty and _blank(entry[name]):
            problems.append(f"empty field '{name}'")
    status = entry.get("status")
    if status is not None and status not in STATUS_VALUES:
        problems.append(f"status must be one of {STATUS_VALUES}")
    if pending:
        if _blank(entry.get("notes")):
            problems.append("a pending entry must say why in 'notes'")
        filled = [name for name in PENDING_EMPTY_FIELDS if not _blank(entry.get(name))]
        if filled:
            problems.append(f"a pending entry must leave {filled} empty")
    unknown = sorted(set(entry) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        problems.append(f"unknown fields {unknown}")
    if not _blank(entry.get("access")) and entry.get("access") not in ACCESS_VALUES:
        problems.append(f"access must be one of {ACCESS_VALUES}")
    if not _blank(entry.get("clip")) and entry.get("clip") not in CLIP_VALUES:
        problems.append(f"clip must be one of {CLIP_VALUES}")
    size = entry.get("bytes")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        problems.append("bytes must be a non-negative integer")
    sha = entry.get("sha256")
    if not _blank(sha) and not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        problems.append("sha256 must be 64 lowercase hex digits")
    retrieved = entry.get("retrieved_utc")
    if isinstance(retrieved, str) and retrieved.strip():
        try:
            dt.datetime.fromisoformat(retrieved)
        except ValueError:
            problems.append("retrieved_utc must be ISO 8601")
    return problems


@dataclass
class EntryResult:
    id: str
    problems: list[str] = field(default_factory=list)
    pending: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_entry(entry: dict[str, Any], paths: ProjectPaths) -> EntryResult:
    result = EntryResult(
        id=str(entry.get("id", "<no id>")),
        problems=validate_entry(entry),
        pending=is_pending(entry),
    )
    local = entry.get("local_path")
    if result.pending or _blank(local):
        return result
    try:
        target = paths.resolve(str(local))
    except PathError as error:
        result.problems.append(str(error))
        return result
    if not target.exists():
        result.problems.append(f"local_path does not exist: {local}")
        return result
    size, sha = measure(target)
    if isinstance(entry.get("bytes"), int) and size != entry["bytes"]:
        result.problems.append(f"bytes mismatch: manifest {entry['bytes']}, on disk {size}")
    if isinstance(entry.get("sha256"), str) and sha != entry["sha256"]:
        result.problems.append(f"sha256 mismatch: manifest {entry['sha256']}, on disk {sha}")
    return result


def verify_manifest(path: str | Path, paths: ProjectPaths) -> list[EntryResult]:
    """Validate every entry and recompute the SHA-256 of every local file."""
    entries = load_manifest(path)
    results = [verify_entry(entry, paths) for entry in entries]
    seen: set[str] = set()
    for result in results:
        if result.id in seen:
            result.problems.append("duplicate id")
        seen.add(result.id)
    return results
