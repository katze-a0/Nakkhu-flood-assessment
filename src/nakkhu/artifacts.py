"""Artifact promotion: copy an approved run output into ``artifacts/`` and index it.

``artifacts/INDEX.yaml`` records, per promoted file, its path (relative to ``artifacts/``),
SHA-256, size, source run (``<stage>/<run_id>``), source file within the run and the UTC time of
promotion. Only outputs of complete runs (see :func:`nakkhu.runs.is_complete`) can be promoted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from nakkhu.paths import PathError, ProjectPaths
from nakkhu.runs import atomic_write_bytes, atomic_write_text, find_run_dir, journal_status, utc_now

MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
INDEX_SCHEMA_VERSION = 1


class PromotionError(RuntimeError):
    """A file cannot be promoted."""


def load_index(paths: ProjectPaths) -> dict[str, Any]:
    if not paths.artifact_index.is_file():
        return {"schema_version": INDEX_SCHEMA_VERSION, "artifacts": []}
    document = yaml.safe_load(paths.artifact_index.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise PromotionError(f"{paths.artifact_index}: top level must be a mapping")
    document.setdefault("schema_version", INDEX_SCHEMA_VERSION)
    document["artifacts"] = document.get("artifacts") or []
    return document


def _write_index(paths: ProjectPaths, document: dict[str, Any]) -> None:
    document["artifacts"] = sorted(document["artifacts"], key=lambda entry: entry["path"])
    header = "# Promoted artifacts. Written by `nakkhu artifacts promote`; do not edit by hand.\n"
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    atomic_write_text(paths.artifact_index, header + body)


def _resolve_source(source: str | Path, paths: ProjectPaths) -> Path:
    candidate = Path(source)
    if not candidate.is_absolute() and (paths.runs / candidate).is_file():
        return (paths.runs / candidate).resolve()
    try:
        return paths.resolve(source)
    except PathError as error:
        raise PromotionError(str(error)) from error


def _resolve_dest(dest: str, paths: ProjectPaths) -> tuple[Path, str]:
    pure = PurePosixPath(dest.replace("\\", "/"))
    if pure.is_absolute() or Path(dest).is_absolute():
        raise PromotionError("artifact path must be relative to artifacts/")
    if pure.parts and pure.parts[0] == "artifacts":
        pure = PurePosixPath(*pure.parts[1:])
    if not pure.parts:
        raise PromotionError("artifact path is empty")
    artifacts = paths.artifacts.resolve()
    target = (artifacts / pure).resolve()
    if artifacts not in target.parents:
        raise PromotionError(f"artifact path {dest!r} escapes artifacts/")
    if target == paths.artifact_index.resolve():
        raise PromotionError("cannot overwrite INDEX.yaml")
    return target, target.relative_to(artifacts).as_posix()


def promote(
    source: str | Path,
    dest: str,
    paths: ProjectPaths,
    *,
    replace: bool = False,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Copy ``source`` (a file inside a run directory) to ``artifacts/<dest>`` and index it."""
    src = _resolve_source(source, paths)
    if not src.is_file():
        raise PromotionError(f"source is not a file: {source}")
    run_dir = find_run_dir(src, paths)
    if run_dir is None:
        raise PromotionError(f"source is not inside a run directory under runs/: {source}")
    status = journal_status(run_dir)
    if status != "complete":
        raise PromotionError(f"run {run_dir.name} journal is {status}, not complete")
    data = src.read_bytes()
    if len(data) > max_bytes:
        raise PromotionError(f"{source} is {len(data)} bytes; artifacts must be <= {max_bytes}")

    target, rel = _resolve_dest(dest, paths)
    document = load_index(paths)
    if target.exists() and not replace:
        raise PromotionError(f"artifacts/{rel} exists; pass --replace to overwrite it")

    atomic_write_bytes(target, data)
    sha = hashlib.sha256(data).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
        raise PromotionError(f"copy of {source} does not match its source digest")

    runs = paths.runs.resolve()
    entry = {
        "path": rel,
        "sha256": sha,
        "bytes": len(data),
        "source_run": run_dir.relative_to(runs).as_posix(),
        "source_file": src.relative_to(run_dir).as_posix(),
        "promoted_utc": utc_now(),
    }
    document["artifacts"] = [e for e in document["artifacts"] if e.get("path") != rel] + [entry]
    _write_index(paths, document)
    return entry
