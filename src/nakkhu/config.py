"""YAML configs with dotted ``--set`` overrides and a stable config hash.

The resolved config (file values after overrides) is serialised to canonical JSON (sorted keys,
no whitespace) and hashed with SHA-256; the first 8 hex digits are the ``cfghash`` used in run
directory names, so reordering keys in a YAML file never changes the hash.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nakkhu.paths import ProjectPaths, find_root

PROJECT_CONFIG = "configs/project.yaml"


class ConfigError(ValueError):
    """A config file or override is invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return values


def apply_overrides(values: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``values`` with each ``a.b.c=value`` applied (value parsed as YAML).

    Only existing keys can be overridden, so a typo fails instead of adding a dead key.
    """
    result = copy.deepcopy(dict(values))
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"invalid override {override!r}; expected key=value")
        target: dict[str, Any] = result
        *parents, leaf = key.strip().split(".")
        for parent in parents:
            child = target.get(parent)
            if not isinstance(child, dict):
                raise ConfigError(f"unknown config path: {key}")
            target = child
        if leaf not in target:
            raise ConfigError(f"unknown config path: {key}")
        target[leaf] = yaml.safe_load(raw)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serialisable in a config: {type(value).__name__}")


def canonical_json(values: Mapping[str, Any]) -> str:
    return json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_jsonable
    )


def config_sha256(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def config_hash(values: Mapping[str, Any]) -> str:
    """Short config hash (first 8 hex digits of the canonical-JSON SHA-256)."""
    return config_sha256(values)[:8]


@dataclass(frozen=True)
class ResolvedConfig:
    values: dict[str, Any]
    source: str
    overrides: tuple[str, ...] = field(default=())

    @property
    def sha256(self) -> str:
        return config_sha256(self.values)

    @property
    def cfghash(self) -> str:
        return config_hash(self.values)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.values, sort_keys=True, allow_unicode=True)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ResolvedConfig:
    overrides = tuple(overrides)
    values = apply_overrides(load_yaml(path), overrides)
    return ResolvedConfig(values=values, source=Path(path).as_posix(), overrides=overrides)


@dataclass(frozen=True)
class Project:
    config: ResolvedConfig
    paths: ProjectPaths


def load_project(
    root: str | Path | None = None,
    config: str | Path = PROJECT_CONFIG,
    overrides: Iterable[str] = (),
) -> Project:
    """Load ``configs/project.yaml`` (plus overrides) and the paths it defines."""
    bootstrap = ProjectPaths.create(find_root(root))
    resolved = load_config(bootstrap.resolve(config), overrides)
    section = resolved.values.get("paths") or {}
    if not isinstance(section, dict):
        raise ConfigError("project config: 'paths' must be a mapping")
    data_root = section.get("data_root", "data")
    if not isinstance(data_root, str) or not data_root.strip():
        raise ConfigError("project config: paths.data_root must be a non-empty string")
    paths = ProjectPaths.create(bootstrap.root, data_root)
    source = bootstrap.relative(bootstrap.resolve(config))
    return Project(
        config=ResolvedConfig(values=resolved.values, source=source, overrides=resolved.overrides),
        paths=paths,
    )
