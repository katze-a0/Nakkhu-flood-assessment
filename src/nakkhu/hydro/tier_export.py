"""Per-building HAND-tier table of one inventory (``nakkhu hand tier-export``).

Descriptive only: one row per building with its id, centroid x/y (EPSG:32645 metres, rounded to
1 mm), the HAND value under the primary configuration and its tier. Nothing is recomputed:

- HAND comes from the ``building_hand_<inventory>.parquet`` table of a complete
  ``hand-attribute`` run, column ``<primary cfg_id>__<primary statistic>`` from the sweep config.
- Geometry and ids come from :func:`nakkhu.hydro.sweep.load_inventory` (extent filter, confidence
  cut, greenhouses removed, ids ``<inventory>:<i>``), the reader ``hand attribute`` used. The
  ids, the order and the polygon areas must match the HAND table row by row.
- Tier edges are the sweep config's ``tiers_m`` (``Settings.edges``), the edges of the
  ``hand attribute`` tables and of ``uncertainty run``.

Checks (any failure stops the run before a table is written, except the count check, which
writes ``count_check_<inventory>.csv`` first so the difference can be read):

- the inventory filter counts equal those in the attribute run summary;
- the per-tier counts equal the primary row of the attribute run's ``count_stability.csv``;
- the per-tier counts equal the promoted raw counts in ``artifacts/uncertainty/
  ppi_tier_counts.csv`` (``coding=primary, level=tier, estimator=naive``), whose SHA-256 must
  match ``artifacts/INDEX.yaml``. An inventory without raw rows there is reported as
  ``not_available``.

It reads no labels, no audit answers and no change map, and writes no test-journal entry.
The run records the SHA-256 of every file it reads.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu import hazard
from nakkhu.config import Project, ResolvedConfig

STAGE = "hand-tiers"
CRS = "EPSG:32645"
COLUMNS = ["building_id", "x", "y", "hand_m", "tier"]
PPI_ARTIFACT = "uncertainty/ppi_tier_counts.csv"
XY_DECIMALS = 3


class TierExportError(RuntimeError):
    """An input is missing or inconsistent, or a count check failed."""


def _sha256(path: Path) -> str:
    from nakkhu.evaluate.guard import sha256_file

    return sha256_file(path)


def primary_column(settings: Any) -> str:
    from nakkhu.hydro.sweep import primary_stat

    return f"{settings.primary().cfg_id}__{primary_stat(settings)}"


def hand_table_name(inventory: str) -> str:
    return f"building_hand_{inventory}.parquet"


def find_attribute_run(project: Project, inventory: str,
                       refs: Sequence[str] | None = None) -> Any:
    """The newest complete ``hand-attribute`` run (among ``refs`` if given) holding the
    inventory's HAND table."""
    from nakkhu.hydro.sweep import ATTRIBUTE_STAGE
    from nakkhu.runs import Run, list_runs

    if refs:
        runs = []
        for ref in refs:
            stage, _, run_id = ref.rpartition("/")
            if stage not in ("", ATTRIBUTE_STAGE):
                raise TierExportError(f"{ref} is not a {ATTRIBUTE_STAGE} run")
            runs.append(Run(ATTRIBUTE_STAGE, run_id,
                            project.paths.runs / ATTRIBUTE_STAGE / run_id))
    else:
        runs = list_runs(project.paths, ATTRIBUTE_STAGE)
    found = [r for r in runs
             if (r.path / hand_table_name(inventory)).is_file() and r.status == "complete"]
    if not found:
        raise TierExportError(f"no complete {ATTRIBUTE_STAGE} run has "
                              f"{hand_table_name(inventory)}")
    return sorted(found, key=lambda r: r.run_id)[-1]


def tier_table(frame: Any, hand: pd.DataFrame, column: str,
               edges: Sequence[float]) -> pd.DataFrame:
    """One row per building: id, centroid x/y, HAND (``column``) and tier label.

    ``frame`` is the inventory as ``load_inventory`` returns it (column ``building_uid``,
    projected CRS); ``hand`` is the attribute run's table for the same inventory."""
    from pyproj import CRS as ProjCRS

    if frame.crs is None or not ProjCRS.from_user_input(frame.crs).equals(
            ProjCRS.from_user_input(CRS)):
        raise TierExportError(f"inventory CRS is {frame.crs}, expected {CRS}")
    if column not in hand.columns:
        raise TierExportError(f"HAND table has no column {column}")
    if len(frame) != len(hand):
        raise TierExportError(f"inventory has {len(frame)} buildings, HAND table {len(hand)}")
    ids = frame["building_uid"].astype(str).to_numpy()
    if not np.array_equal(ids, hand["building_uid"].astype(str).to_numpy()):
        raise TierExportError("building ids or their order differ from the HAND table")
    if "area_m2" in hand.columns:
        area = frame.geometry.area.to_numpy(dtype=np.float64)
        if not np.allclose(area, hand["area_m2"].to_numpy(dtype=np.float64),
                           rtol=1e-6, atol=1e-3):
            raise TierExportError("polygon areas differ from the HAND table (other geometry)")
    values = hand[column].to_numpy(dtype=np.float32)
    labels = np.asarray(hazard.tier_labels(edges), dtype=object)
    index = hazard.tier_index(values, edges)
    tier = np.where(index >= 0, labels[np.clip(index, 0, None)], None)
    cent = frame.geometry.centroid
    return pd.DataFrame({"building_id": ids,
                         "x": cent.x.to_numpy().round(XY_DECIMALS),
                         "y": cent.y.to_numpy().round(XY_DECIMALS),
                         "hand_m": values, "tier": tier})[COLUMNS]


def tier_counts(table: pd.DataFrame, labels: Sequence[str]) -> dict[str, int]:
    counts = table["tier"].value_counts()
    return {lab: int(counts.get(lab, 0)) for lab in labels}


def stability_counts(stability: pd.DataFrame, inventory: str, column: str,
                     labels: Sequence[str]) -> dict[str, int]:
    """Per-tier counts of the primary row in an attribute run's ``count_stability.csv``."""
    cfg_id, stat = column.rsplit("__", 1)
    rows = stability[(stability["inventory"] == inventory) & (stability["cfg_id"] == cfg_id)
                     & (stability["statistic"] == stat)]
    if len(rows) != 1:
        raise TierExportError(f"count_stability.csv has {len(rows)} rows for {inventory} "
                              f"{column}, expected 1")
    missing = [f"n_{lab}" for lab in labels if f"n_{lab}" not in rows.columns]
    if missing:
        raise TierExportError(f"count_stability.csv lacks {missing}: its tier edges differ "
                              "from the sweep config")
    return {lab: int(rows.iloc[0][f"n_{lab}"]) for lab in labels}


def promoted_counts(ppi: pd.DataFrame, inventory: str,
                    labels: Sequence[str]) -> dict[str, int] | None:
    """Raw per-tier counts of ``ppi_tier_counts.csv`` (primary coding, naive), or None."""
    rows = ppi[(ppi["coding"] == "primary") & (ppi["inventory"] == inventory)
               & (ppi["level"] == "tier") & (ppi["estimator"] == "naive")]
    if rows.empty:
        return None
    by_tier = rows.set_index("tiers")["estimate"]
    if not by_tier.index.is_unique or set(by_tier.index) != set(labels):
        raise TierExportError(f"ppi_tier_counts.csv raw rows for {inventory} have tiers "
                              f"{sorted(by_tier.index)}, expected {list(labels)}")
    out = {}
    for lab in labels:
        value = float(by_tier[lab])
        if value != round(value):
            raise TierExportError(f"raw count for {inventory} {lab} is not whole: {value}")
        out[lab] = int(round(value))
    return out


def count_check(got: dict[str, int], references: dict[str, dict[str, int] | None]
                ) -> pd.DataFrame:
    """One row per tier: the export's count and each reference's count and difference."""
    rows = []
    for lab, n in got.items():
        row: dict[str, Any] = {"tier": lab, "n_export": n}
        for name, ref in references.items():
            row[f"n_{name}"] = None if ref is None else ref[lab]
            row[f"diff_{name}"] = None if ref is None else n - ref[lab]
        rows.append(row)
    return pd.DataFrame(rows)


def _index_sha(project: Project, rel: str) -> str | None:
    from nakkhu.artifacts import load_index

    for entry in load_index(project.paths)["artifacts"]:
        if entry.get("path") == rel:
            return str(entry["sha256"])
    return None


def _rel(project: Project, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.paths.root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def export_run(project: Project, inventories: Sequence[str],
               attribute_refs: Sequence[str] | None = None,
               config: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Write ``building_tiers_<inventory>.csv`` per inventory into one ``hand-tiers`` run."""
    from nakkhu.hydro.sweep import SWEEP_CONFIG, load_greenhouses, load_inventory, load_settings
    from nakkhu.runs import atomic_write_text, create_run

    settings, _ = load_settings(project, config or SWEEP_CONFIG)
    unknown = [n for n in inventories if n not in settings.v["inventories"]]
    if unknown:
        raise TierExportError(f"unknown inventory {unknown}")
    column = primary_column(settings)
    edges = list(settings.edges)
    labels = hazard.tier_labels(edges)
    sources = {n: find_attribute_run(project, n, attribute_refs) for n in inventories}
    ppi_path = project.paths.artifacts / PPI_ARTIFACT
    ppi_sha = _sha256(ppi_path) if ppi_path.is_file() else None
    ppi_index = _index_sha(project, PPI_ARTIFACT)
    if ppi_sha is not None and ppi_index is not None and ppi_sha != ppi_index:
        raise TierExportError(f"artifacts/{PPI_ARTIFACT} does not match INDEX.yaml")
    run = create_run(project, STAGE, ResolvedConfig(values={
        "command": "hand-tier-export", "inventories": list(inventories),
        "attribute_runs": {n: r.ref for n, r in sources.items()}, "hand_column": column,
        "tiers_m": edges, "tiers": labels, "columns": COLUMNS, "crs": CRS,
        "sweep_config": settings.config.source, "sweep_config_sha256": settings.config.sha256},
        source="cli"))
    summary: dict[str, Any] = {
        "hand_column": column, "tiers_m": edges, "tiers": labels, "columns": COLUMNS,
        "crs": CRS, "xy_decimals": XY_DECIMALS, "sweep_config": settings.config.source,
        "sweep_config_sha256": settings.config.sha256,
        "promoted_counts": {"path": f"artifacts/{PPI_ARTIFACT}", "sha256": ppi_sha,
                            "index_sha256": ppi_index},
        "reads_labels": False, "reads_audit_answers": False, "reads_change_map": False,
        "test_access": False, "inventories": {}}
    extent = settings.read_polygon(settings.v["analysis_extent"])
    greenhouses = load_greenhouses(settings)
    summary["greenhouses"] = None if greenhouses is None else {
        "n_areas": len(greenhouses),
        "by_tag": {str(k): int(v) for k, v in greenhouses["tag"].value_counts().items()}}
    ppi = pd.read_csv(ppi_path) if ppi_path.is_file() else None
    for name in inventories:
        source = sources[name]
        with run.step(f"export-{name.lower()}"):
            attr = json.loads((source.path / "summary.json").read_text(encoding="utf-8"))
            if attr.get("primary") not in (None, column):
                raise TierExportError(f"{source.ref} primary is {attr.get('primary')}, "
                                      f"the sweep config's is {column}")
            frame, info = load_inventory(settings, name, extent, greenhouses)
            recorded = (attr.get("inventories") or {}).get(name, {})
            diff = {k: (v, recorded.get(k)) for k, v in info.items()
                    if k in recorded and k != "cut_thresholds" and recorded[k] != v}
            if diff:
                raise TierExportError(f"{name}: inventory counts differ from {source.ref}: "
                                      f"{diff} (now, then)")
            if (summary["greenhouses"] is not None and attr.get("greenhouses") is not None
                    and attr["greenhouses"] != summary["greenhouses"]):
                raise TierExportError(f"greenhouse areas differ from {source.ref}")
            hand_path = source.path / hand_table_name(name)
            hand = pd.read_parquet(hand_path, columns=["building_uid", "area_m2", column])
            table = tier_table(frame, hand, column, edges)
            got = tier_counts(table, labels)
            stability_path = source.path / "count_stability.csv"
            refs: dict[str, dict[str, int] | None] = {
                "attribute_run": stability_counts(pd.read_csv(stability_path), name, column,
                                                  labels),
                "promoted": None if ppi is None else promoted_counts(ppi, name, labels)}
            check = count_check(got, refs)
            check_name = f"count_check_{name}.csv"
            atomic_write_text(run.path / check_name, check.to_csv(index=False,
                                                                 lineterminator="\n"))
            match = {k: (None if ref is None else ref == got) for k, ref in refs.items()}
            entry: dict[str, Any] = {
                "attribute_run": source.ref, "hand_table": _rel(project, hand_path),
                "hand_table_sha256": _sha256(hand_path),
                "count_stability_sha256": _sha256(stability_path),
                "attribute_summary_sha256": _sha256(source.path / "summary.json"),
                "inventory_path": settings.v["inventories"][name]["path"],
                "inventory_sha256": _sha256(settings.resolve(
                    settings.v["inventories"][name]["path"])),
                "filters": info, "rows": len(table),
                "n_missing_hand": int(table["tier"].isna().sum()),
                "counts": got, "count_check": check_name,
                "counts_match": {k: ("not_available" if v is None else v)
                                 for k, v in match.items()}}
            summary["inventories"][name] = entry
            bad = [k for k, v in match.items() if v is False]
            if bad:
                run.write_summary(summary)
                raise TierExportError(f"{name}: tier counts differ from {', '.join(bad)} "
                                      f"(see {run.ref}/{check_name})")
            out_name = f"building_tiers_{name}.csv"
            atomic_write_text(run.path / out_name, table.to_csv(index=False,
                                                               lineterminator="\n"))
            entry.update({"file": out_name, "output_sha256": _sha256(run.path / out_name),
                          "output_bytes": (run.path / out_name).stat().st_size})
    run.write_summary(summary)
    return run, summary
