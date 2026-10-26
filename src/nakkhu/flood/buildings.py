"""Reference inventories for the change map (plan 009 Step 4).

Buildings are loaded with plan 008's own loader (:func:`nakkhu.hydro.sweep.load_inventory`:
centroid in the extent, Google v3 t90 cut, OSM greenhouses removed through
:mod:`nakkhu.reference.greenhouse`), so ``building_id`` equals plan 008's ``building_uid`` and
joins the building HAND tables row for row. Only the inventory definitions are read from the
sweep config; no HAND, DEM or channel raster is opened here. The polygons are then moved by
the plan 004 per-tile registration shifts onto the WV02 imagery (the change map samples
pixels, so footprints must sit on the roofs they describe).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from nakkhu.config import Project

#: the per-tile shift a source's polygons take (plan 004 registration)
SHIFT_KIND: dict[str, str | None] = {"google_v3": "footprints", "google_v3_uncut": "footprints",
                                     "microsoft": "footprints", "osm": "osm",
                                     "overture": None, "detector_final": None,
                                     # single-seed detector inventories (plan 009 D4)
                                     **{f"seed{k}": None for k in range(5)},
                                     **{f"L2_seed{k}": None for k in range(3)}}
# Overture is a conflation (not shifted); detector boxes already sit on the WV02 imagery.


def load_inventories(project: Project, names: Sequence[str], sweep_config: str,
                     registration: dict[str, str]) -> tuple[dict[str, gpd.GeoDataFrame],
                                                            dict[str, dict[str, Any]]]:
    from nakkhu.audit.frames import apply_shifts, tile_index
    from nakkhu.hydro.sweep import load_greenhouses, load_inventory, load_settings

    settings, _ = load_settings(project, sweep_config)
    extent = settings.read_polygon(settings.v["analysis_extent"])
    greenhouses = load_greenhouses(settings)
    index = tile_index(project)
    shifts = {kind: pd.read_csv(project.paths.artifacts / rel).set_index("tile_id")[
        ["dx", "dy", "reg_ok"]] for kind, rel in registration.items()}
    frames: dict[str, gpd.GeoDataFrame] = {}
    info: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in SHIFT_KIND:
            raise ValueError(f"no registration rule for inventory {name!r}")
        frame, meta = load_inventory(settings, name, extent, greenhouses)
        geoms = np.asarray(frame.geometry.values, dtype=object)
        kind = SHIFT_KIND[name]
        moved = geoms if kind is None else apply_shifts(geoms, index, shifts[kind])
        out = gpd.GeoDataFrame(
            {"building_id": frame["building_uid"].to_numpy(),
             **{c: frame[c].to_numpy() for c in ("source_id", "confidence")
                if c in frame.columns}},
            geometry=gpd.GeoSeries(moved, crs=frame.crs))
        frames[name] = out
        info[name] = {k: v for k, v in meta.items() if k != "cut_thresholds"}
        info[name]["registration"] = kind or "none"
    return frames, info
