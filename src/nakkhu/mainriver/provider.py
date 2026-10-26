"""Main-river HAND for the plan 010 analysis (``nakkhu.uncertainty.run.HandProvider``), read
from a complete ``hand-mr`` run: its ``surface_grid.csv`` and the hand_mr rasters it wrote.
Members are computed exactly as ``nakkhu mainriver surfaces`` (and ``hand attribute``) does."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from nakkhu.config import Project
from nakkhu.evaluate.guard import sha256_file
from nakkhu.mainriver import spec as S
from nakkhu.mainriver.surfaces import RASTER


class MainRiverHand:
    """``surfaces()`` lists the computed surfaces; ``in_sensitivity_range`` = the primary
    ensemble (gate-passing, canonical DEMs). Rasters are sha256-checked against the run."""

    def __init__(self, project: Project, run_dir: Path, cfg: dict) -> None:
        from nakkhu.hydro.sweep import load_settings

        self.settings, _ = load_settings(project, cfg["base_config"])
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.interim = project.paths.resolve(summary["interim"])
        table = pd.read_csv(run_dir / "surface_grid.csv")
        self.table = table[table["status"] == "done"].reset_index(drop=True)
        self.expected = {str(r.cfg_id): str(r.raster_sha256) for r in self.table.itertuples()}
        self.grid = self.settings.grid()
        if self.grid.to_dict() != summary["grid"]:
            raise S.MainRiverError("hand-mr run grid differs from the base sweep config")
        self.edges = tuple(float(e) for e in cfg["tiers_m"])
        self.buffer_m = float(self.settings.v["buffer_m"])
        if abs(self.buffer_m - 10.0) > 1e-9:
            raise S.MainRiverError("footprint_plus10m needs buffer_m = 10")
        self.index_tif = self.interim / "grid" / "cell_index.tif"
        if sha256_file(self.index_tif) != summary["cell_index_sha256"]:
            raise S.MainRiverError("cell_index.tif differs from the hand-mr run")
        self._cache: dict[str, np.ndarray] = {}

    def surfaces(self) -> pd.DataFrame:
        t = self.table
        return pd.DataFrame({"cfg_id": t["cfg_id"].astype(str), "dem": t["dem"].astype(str),
                             "in_sensitivity_range": t["in_sensitivity_range"].astype(bool),
                             "is_primary": t["is_primary"].astype(bool)})

    def _raster(self, cfg_id: str) -> np.ndarray:
        from nakkhu.hydro import dem as demmod

        if cfg_id not in self._cache:
            path = self.interim / cfg_id / RASTER
            if sha256_file(path) != self.expected[cfg_id]:
                raise S.MainRiverError(f"{path} differs from the hand-mr run")
            self._cache[cfg_id] = demmod.read_raster(path).ravel()
        return self._cache[cfg_id]

    def values(self, geoms: np.ndarray, stats: Sequence[str],
               cfg_ids: Sequence[str]) -> dict[str, np.ndarray]:
        import geopandas as gpd

        from nakkhu import hazard

        series = gpd.GeoSeries(np.asarray(geoms, dtype=object), crs=self.settings.crs)
        cent = hazard.centroid_cells(series, self.grid) if "centroid" in stats else None
        foot = hazard.cell_groups(series, self.index_tif) if "footprint_min" in stats else None
        buf = (hazard.cell_groups(series.buffer(self.buffer_m), self.index_tif)
               if "buffer_min" in stats else None)
        out: dict[str, np.ndarray] = {}
        for cfg in cfg_ids:
            vals = self._raster(cfg)
            for stat in stats:
                if stat == "centroid" and cent is not None:
                    out[f"{cfg}__{stat}"] = hazard.sample_centroids(vals, cent)
                elif stat == "footprint_min" and foot is not None:
                    out[f"{cfg}__{stat}"] = foot.minimum(vals)
                elif stat == "buffer_min" and buf is not None:
                    out[f"{cfg}__{stat}"] = buf.minimum(vals)
                else:
                    raise S.MainRiverError(f"unknown HAND statistic {stat!r}")
        return out
