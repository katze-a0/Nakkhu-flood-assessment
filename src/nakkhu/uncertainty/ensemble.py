"""Audited units and ensemble scores for U1 / U3 (prereg v5 ``detector_final`` and
``audit_precision_pool``).

``detector_final`` is built by plan 007 (``nakkhu.ensemble``): greedy one-box-per-seed clusters
of the five final seeds (each at its own val threshold), medoid geometry, ``conf`` = sum of the
member confidences / 5 (= seeds detecting / 5 x their mean confidence), cut at the ensemble's
own val F1-max threshold. That ``conf`` is the U1 raw existence score.

The sitting-2 pool holds six runs: the five seeds (own thresholds) and the ensemble test run at
its cut (the ``detector_final`` boxes). Audit units are the plan 006 single-linkage clusters of
those runs. A unit is in the ``detector_final`` domain if it contains an ensemble-run box; its
raw score is the highest ensemble ``conf`` among those boxes. A unit is in seed k's domain if it
contains a box of that seed (the plan 006 per-run precision domain).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def unit_domains(members: pd.DataFrame, final_run: str,
                 seed_runs: Sequence[str]) -> pd.DataFrame:
    """Per audit unit (``cluster`` of ``members.parquet``: cluster, run, det, conf): ``in_final``,
    ``raw_score`` (max ensemble conf, NaN outside the domain), ``n_final_boxes`` and one
    ``in_seed<k>`` column per seed run."""
    m = members.copy()
    m["run"] = m["run"].astype(str)
    units = pd.DataFrame({"unit": np.sort(m["cluster"].unique())})
    fin = m.loc[m["run"] == str(final_run)]
    g = fin.groupby("cluster")["conf"]
    units["n_final_boxes"] = units["unit"].map(g.size()).fillna(0).astype(int)
    units["in_final"] = units["n_final_boxes"] > 0
    units["raw_score"] = units["unit"].map(g.max())
    for k, run in enumerate(seed_runs):
        has = set(m.loc[m["run"] == str(run), "cluster"])
        units[f"in_seed{k}"] = units["unit"].isin(has)
    return units


def expected_count(p: np.ndarray, tiers: np.ndarray, n_tiers: int) -> np.ndarray:
    """Ensemble expected count per tier: sum of calibrated existence probabilities of the
    ``detector_final`` clusters."""
    p = np.asarray(p, dtype=float)
    tiers = np.asarray(tiers)
    ok = (tiers >= 0) & np.isfinite(p)
    return np.bincount(tiers[ok].astype(int), weights=p[ok], minlength=n_tiers)[:n_tiers]
