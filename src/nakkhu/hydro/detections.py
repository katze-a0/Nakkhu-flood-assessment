"""Single-seed detector inventories as plan 008 inventory files (plan 009 D4 / plan 010 U2).

A sweep inventory with a ``detections`` entry, e.g.::

    seed0:
      path: "{data_root}/interim/inventories/seed0.parquet"
      detections: {arm: L5, model: yolo11n-obb, seed: 0}
      role: d4

is one trained unit's WV02 test-region inference run at that unit's own val F1-max threshold.
The run and threshold come from the plan 007 registry (:func:`nakkhu.ensemble.final_members`,
the same rule as the detector_final members and the plan 010 inventories). This module writes
the boxes with ``conf >= threshold`` to ``path`` (EPSG:32645 GeoParquet, plus a ``.json``
sidecar with the run, threshold and source SHA-256), so :func:`nakkhu.hydro.sweep.load_inventory`
(centroid in the extent, greenhouse filter) and ``nakkhu flood change`` read it like any other
inventory. Geometry only: nothing is scored, no reference is read. The file is regenerable from
the run and lives under ``data/interim`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nakkhu.config import Project

WORK_CRS = "EPSG:32645"


class DetectionInventoryError(RuntimeError):
    """A detection inventory cannot be written (unit not ingested, run not georeferenced)."""


def detection_entries(sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Sweep inventories defined by a ``detections`` entry."""
    return {name: entry for name, entry in (sweep.get("inventories") or {}).items()
            if isinstance(entry, dict) and entry.get("detections")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_one(project: Project, name: str, entry: dict[str, Any], resolve: Any
               ) -> dict[str, Any]:
    import geopandas as gpd

    from nakkhu.ensemble import final_members

    spec = entry["detections"]
    try:
        member = final_members(project, str(spec["arm"]), str(spec["model"]),
                               [int(spec["seed"])])[0]
    except Exception as error:      # unit not ingested / not evaluated yet
        raise DetectionInventoryError(f"{name}: {error}") from error
    gpkg = project.paths.runs / str(member["infer_test"]) / "detections.gpkg"
    if not gpkg.is_file():
        raise DetectionInventoryError(
            f"{name}: {gpkg} missing (run `nakkhu georef --run {member['infer_test']}`)")
    det = gpd.read_file(gpkg, layer="obb", columns=["conf"])
    det = det.to_crs(WORK_CRS) if det.crs is not None else det.set_crs(WORK_CRS)
    thr = float(member["threshold"])
    keep = det.loc[det["conf"].to_numpy(dtype=float) >= thr].reset_index(drop=True)
    out = gpd.GeoDataFrame({"conf": keep["conf"].to_numpy(dtype=float)},
                           geometry=keep.geometry.values, crs=WORK_CRS)
    path = Path(resolve(str(entry["path"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    meta = {"inventory": name, "train_run": member["train_run"],
            "infer_test": member["infer_test"], "eval_val": member["eval_val"],
            "threshold": thr, "rule": "conf >= the unit's own val F1-max threshold",
            "n_boxes_run": int(len(det)), "n_boxes_kept": int(len(out)),
            "detections_sha256": _sha256(gpkg), "inventory_sha256": _sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True),
                                         encoding="utf-8")
    return meta


def export(project: Project, names: Sequence[str] | None = None,
           sweep_config: str = "configs/hand/sweep.yaml"
           ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Write every requested detection inventory; ``(written, pending reasons)``."""
    from nakkhu.hydro.sweep import load_settings

    settings, _ = load_settings(project, sweep_config)
    entries = detection_entries(settings.v)
    unknown = [n for n in names or [] if n not in entries]
    if unknown:
        raise DetectionInventoryError(f"not detection inventories in {sweep_config}: {unknown}")
    written, pending = {}, {}
    for name in names or list(entries):
        try:
            written[name] = export_one(project, name, entries[name], settings.resolve)
        except DetectionInventoryError as error:
            pending[name] = str(error)
    return written, pending
