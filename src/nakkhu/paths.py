"""Project-root path resolution.

Every path the pipeline reads or writes resolves relative to the ``nakkhu/`` root. Absolute
paths are accepted only inside the root or inside the data root set in ``configs/project.yaml``
(which may live on another drive). A path may start with ``{data_root}`` to address the data
root explicitly, e.g. ``{data_root}/raw/dem_glo30/tile.tif``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT_TOKEN = "{data_root}"
ROOT_ENV_VAR = "NAKKHU_ROOT"


class PathError(ValueError):
    """A path is malformed or resolves outside the allowed roots."""


def _is_within(path: Path, base: Path) -> bool:
    return path == base or base in path.parents


def find_root(root: str | Path | None = None) -> Path:
    """Return the project root: explicit argument, then ``NAKKHU_ROOT``, then the checkout."""
    if root is not None:
        return Path(root).resolve()
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise PathError(f"cannot locate the nakkhu root; pass --root or set {ROOT_ENV_VAR}")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path

    @classmethod
    def create(cls, root: str | Path, data_root: str | Path = "data") -> ProjectPaths:
        root = Path(root).resolve()
        data = Path(data_root)
        data = (data if data.is_absolute() else root / data).resolve()
        return cls(root=root, data_root=data)

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def artifact_index(self) -> Path:
        return self.artifacts / "INDEX.yaml"

    def resolve(self, value: str | Path) -> Path:
        """Resolve ``value`` against the root; reject anything outside root and data root."""
        text = str(value)
        if not text.strip():
            raise PathError("empty path")
        if text.startswith(DATA_ROOT_TOKEN):
            rest = text[len(DATA_ROOT_TOKEN) :].lstrip("/\\")
            candidate = (self.data_root / rest).resolve()
            if not _is_within(candidate, self.data_root):
                raise PathError(f"{text!r} escapes the data root {self.data_root}")
            return candidate
        path = Path(text)
        candidate = (path if path.is_absolute() else self.root / path).resolve()
        if _is_within(candidate, self.root) or _is_within(candidate, self.data_root):
            return candidate
        raise PathError(f"{text!r} resolves outside the nakkhu root and the data root")

    def relative(self, path: str | Path) -> str:
        """Portable form of ``path``: POSIX relative to the root, or ``{data_root}/...``."""
        resolved = Path(path).resolve()
        if _is_within(resolved, self.root):
            return resolved.relative_to(self.root).as_posix()
        if _is_within(resolved, self.data_root):
            rest = resolved.relative_to(self.data_root).as_posix()
            return f"{DATA_ROOT_TOKEN}/{rest}"
        raise PathError(f"{path} is outside the nakkhu root and the data root")
