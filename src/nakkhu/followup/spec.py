"""Settings of the plan 014 follow-up code: the frozen addendum (``prereg_v9_a3``) and the prep
pins (``configs/followup/prep.yaml``). Every test setting is read from the addendum file; the
prep file only says which runs and files hold the inputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nakkhu.config import Project, load_yaml

PREP_CONFIG = "configs/followup/prep.yaml"
PREP_STAGE = "followup-prep"
RUN_STAGE = "followup"
ARTIFACT_DIR = "followup"                    # under artifacts/
PREP_MANIFEST = f"{ARTIFACT_DIR}/prep_manifest.json"
V9_COLUMN = "glo30__breach_burn__tdrop__centroid"   # prereg v9 hand.primary in the HAND tables
P1_4_THRESHOLDS = (0.2, 0.5, 1.0, 2.0)


class FollowupError(RuntimeError):
    """A follow-up prep step or test may not run."""


def tlabel(threshold_km2: float) -> str:
    """Column label of a stream threshold: 0.075 -> ``t0p075`` (as the sweep's cfg ids)."""
    return "t" + f"{float(threshold_km2):g}".replace(".", "p")


def load_prep_config(project: Project, path: str = PREP_CONFIG) -> dict[str, Any]:
    cfg = load_yaml(project.paths.resolve(path))
    for key in ("addendum", "sweep_config", "inventories", "hand_tables", "dems", "team",
                "unosat"):
        if key not in cfg:
            raise FollowupError(f"{path}: `{key}` is required")
    return cfg


def load_spec(project: Project, addendum_id: str) -> dict[str, Any]:
    """The parsed addendum (duplicate keys refused, header checked)."""
    from nakkhu.evaluate.addendum import addendum_path, load_addendum

    return load_addendum(addendum_path(project, addendum_id))


def candidates_km2(spec: dict[str, Any]) -> list[float]:
    """The 16 drop-analysis candidates (``common.hand_grid.drop_analysis``), which are also
    P1-2's ``thresholds_km2``."""
    values = spec["common"]["hand_grid"]["drop_analysis"]["values"]["candidates_km2"]
    p12 = spec["tests"].get("P1-2", {}).get("thresholds_km2")
    if p12 is not None and [float(v) for v in p12] != [float(v) for v in values]:
        raise FollowupError("P1-2 thresholds_km2 differ from the drop-analysis candidates")
    return [float(v) for v in values]


def p1_4_thresholds(spec: dict[str, Any]) -> list[float]:
    """The fixed thresholds of P1-4 ``rules`` (``threshold 0.2 km2`` ...), checked against the
    YAML text."""
    rules = " ".join(str(r) for r in spec["tests"]["P1-4"]["rules"])
    found = [float(v) for v in re.findall(r"threshold (\d+(?:\.\d+)?)", rules)]
    if sorted(found) != sorted(P1_4_THRESHOLDS):
        raise FollowupError(f"P1-4 rules name thresholds {found}, code expects "
                            f"{list(P1_4_THRESHOLDS)}")
    return list(P1_4_THRESHOLDS)


def bootstrap_settings(spec: dict[str, Any]) -> tuple[int, int]:
    """``(n_boot, seed)`` of ``common.bootstrap_defaults`` (the C3 method)."""
    n_boot, seed, _ = spec["common"]["bootstrap_defaults"]["values"]
    return int(n_boot), int(seed)


def min_cell(spec: dict[str, Any]) -> int:
    """``common.min_cell``: >= N affected and >= N unaffected (N parsed from the text)."""
    match = re.search(r">=\s*(\d+)\s+affected", str(spec["common"]["min_cell"]))
    if not match:
        raise FollowupError("common.min_cell does not state '>= N affected'")
    return int(match.group(1))


def tier_edges(spec: dict[str, Any]) -> list[float]:
    return [float(e) for e in spec["common"]["hand_grid"]["tiers"]["values"]]


def vsizip(zip_path: Path, member: str) -> str:
    """GDAL path of a zip member (read in place; nothing is extracted)."""
    return f"/vsizip/{Path(zip_path).as_posix()}/{member}"
