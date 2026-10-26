"""Settings of the main-river recompute: the ensemble definition and the recompute config.

A *surface* is one DEM x conditioning routing (cfg id ``<dem>__<conditioning>__mr``); a *member*
is a surface x building statistic (``<cfg id>__<statistic>``), as the v9 HAND columns
(``glo30__breach_burn__tdrop__centroid``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nakkhu.config import Project, load_yaml

ENSEMBLE_CONFIG = "configs/hand/main_river_v1.yaml"
RECOMPUTE_CONFIG = "configs/mainriver/recompute_v1.yaml"
SUFFIX = "mr"
DIAGNOSTIC = ("breach_noburn", "fill_noburn")


class MainRiverError(RuntimeError):
    """A main-river step may not run or an input is inconsistent."""


def cfg_id(dem: str, conditioning: str) -> str:
    return f"{dem}__{conditioning}__{SUFFIX}"


@dataclass(frozen=True)
class Surface:
    cfg_id: str
    dem: str
    conditioning: str
    source: str                 # "canonical" | "dem_extension"
    sweep_run: str
    ensemble: str               # "primary" | "extended"
    routing_gate_pass: bool
    gate_reason: str
    is_primary: bool

    @property
    def computed(self) -> bool:
        return self.routing_gate_pass

    @property
    def in_sensitivity_range(self) -> bool:
        return self.routing_gate_pass and self.ensemble == "primary"


def load_ensemble(project: Project, path: str = ENSEMBLE_CONFIG) -> dict[str, Any]:
    cfg = load_yaml(project.paths.resolve(path))
    for key in ("version", "base_config", "stage", "drainage", "sources", "ensemble",
                "primary", "tiers_m", "inventories"):
        if key not in cfg:
            raise MainRiverError(f"{path}: `{key}` is required")
    return cfg


def load_recompute(project: Project, path: str = RECOMPUTE_CONFIG) -> dict[str, Any]:
    cfg = load_yaml(project.paths.resolve(path))
    for key in ("version", "ensemble_config", "uncertainty_spec", "uncertainty_spec_sha256",
                "run_stage", "artifact_dir", "analyses"):
        if key not in cfg:
            raise MainRiverError(f"{path}: `{key}` is required")
    return cfg


def sweep_dir(project: Project, ref: str) -> Path:
    stage, _, run_id = ref.partition("/")
    return project.paths.runs / stage / run_id


def sweep_interim(project: Project, ref: str) -> Path:
    summary = json.loads((sweep_dir(project, ref) / "summary.json").read_text(encoding="utf-8"))
    return project.paths.resolve(summary["interim"])


def sweep_grid(project: Project, ref: str) -> dict[str, Any]:
    summary = json.loads((sweep_dir(project, ref) / "summary.json").read_text(encoding="utf-8"))
    return dict(summary["grid"])


def surfaces(project: Project, cfg: dict[str, Any]) -> list[Surface]:
    """Every DEM x conditioning of the primary and the extended ensemble, with the routing gate
    of its source sweep (``burn_check.csv``)."""
    prim = cfg["primary"]
    out: list[Surface] = []
    for kind in ("ensemble", "extended"):
        block = cfg.get(kind)
        if not block:
            continue
        ref = str(cfg["sources"][block["source"]])
        gate = pd.read_csv(sweep_dir(project, ref) / "burn_check.csv")
        for dem in block["dems"]:
            for cond in block["conditioning"]:
                if cond in DIAGNOSTIC:
                    raise MainRiverError(f"{cond} is diagnostic and never an ensemble member")
                row = gate.loc[(gate["dem"] == dem) & (gate["conditioning"] == cond)]
                if row.empty:
                    raise MainRiverError(f"{ref}: no routing-gate row for {dem} x {cond}")
                ok = bool(row["gate_pass"].iloc[0])
                reason = "" if ok else str(row["gate_reason"].iloc[0])
                out.append(Surface(cfg_id(dem, cond), dem, cond, block["source"], ref,
                                   "primary" if kind == "ensemble" else "extended", ok, reason,
                                   kind == "ensemble" and dem == prim["dem"]
                                   and cond == prim["conditioning"]))
    ids = [s.cfg_id for s in out]
    if len(set(ids)) != len(ids):
        raise MainRiverError("a DEM x conditioning is listed twice")
    if sum(s.is_primary for s in out) != 1 or not next(s for s in out if s.is_primary).computed:
        raise MainRiverError("the primary surface must be one gate-passing ensemble member")
    return out


def primary_column(cfg: dict[str, Any]) -> str:
    p = cfg["primary"]
    return f"{cfg_id(p['dem'], p['conditioning'])}__{p['statistic']}"


def statistics(cfg: dict[str, Any]) -> list[str]:
    return [str(s) for s in cfg["ensemble"]["statistics"]]
