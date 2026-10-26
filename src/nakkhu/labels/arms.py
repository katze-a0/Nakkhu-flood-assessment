"""Label arms L1-L8: config loading and the instance rules (plan 004, Label Arms table)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nakkhu.config import ConfigError, ResolvedConfig, load_yaml
from nakkhu.labels.match import CONFLICT_CLASSES, MatchResult
from nakkhu.labels.obb import KEEP

ARMS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")
RULES = ("google_t90", "microsoft", "union", "intersection", "consensus")
LABELS_DIR = "configs/labels"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_arm_config(path: str | Path) -> dict[str, Any]:
    """An arm config with its ``extends`` chain resolved (keys of the arm file win)."""
    path = Path(path)
    values = load_yaml(path)
    parent = values.pop("extends", None)
    if parent:
        values = _merge(load_arm_config(path.parent / parent), values)
    return values


def arm_config(root: Path, arm: str) -> ResolvedConfig:
    if arm not in ARMS:
        raise ConfigError(f"unknown arm {arm!r}; expected one of {ARMS}")
    path = root / LABELS_DIR / f"{arm}.yaml"
    values = load_arm_config(path)
    if values.get("rule") not in RULES:
        raise ConfigError(f"{path}: rule must be one of {RULES}")
    return ResolvedConfig(values=values, source=f"{LABELS_DIR}/{arm}.yaml")


@dataclass(frozen=True)
class Selection:
    """Instances an arm uses: source (``google``/``microsoft``), index into that source, action."""

    source: np.ndarray
    index: np.ndarray
    action: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


def _sel(parts: list[tuple[str, np.ndarray, str]]) -> Selection:
    sources, indices, actions = [], [], []
    for source, mask, action in parts:
        idx = np.flatnonzero(mask)
        sources.append(np.full(len(idx), source, dtype=object))
        indices.append(idx)
        actions.append(np.full(len(idx), action, dtype=object))
    return Selection(np.concatenate(sources), np.concatenate(indices).astype(np.int64),
                     np.concatenate(actions))


def select(rule: str, matches: MatchResult) -> Selection:
    """Instances and keep actions for one arm rule (OSM never enters).

    ``consensus`` (L5, Decisions 2026-10-05): agree -> keep (Google geometry); split / merge /
    complex conflict -> keep Google's per-house units (Microsoft's merged polygons are not
    used); lone Google >= t90 -> keep; lone Google < t90 and lone Microsoft are dropped
    (background). Ignore regions then come only from the Step 5 filters and SAM2.
    """
    g, m = matches.google, matches.microsoft
    g_cls = g["cls"].to_numpy()
    hi = g["hi"].to_numpy(dtype=bool)
    if rule == "google_t90":
        return _sel([("google", hi, KEEP)])
    if rule == "microsoft":
        return _sel([("microsoft", np.ones(len(m), dtype=bool), KEEP)])
    if rule == "union":
        return _sel([("google", hi, KEEP),
                     ("microsoft", ~m["matches_hi"].to_numpy(dtype=bool), KEEP)])
    if rule == "intersection":
        return _sel([("google", g["n_matches"].to_numpy() > 0, KEEP)])
    if rule == "consensus":
        keep = (g_cls == "agree") | np.isin(g_cls, CONFLICT_CLASSES) | (g_cls == "lone_google_hi")
        return _sel([("google", keep, KEEP)])
    raise ValueError(f"unknown rule {rule!r}")
