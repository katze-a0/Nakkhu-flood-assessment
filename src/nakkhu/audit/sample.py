"""Stratified sampling with a fixed seed (plan 006 Step 1) and the pilot allocation (Step 5).

Design sizes: ``min_per_stratum`` units per stratum (all of a smaller stratum), the rest
allocated proportional to sqrt(N_h) (water-filling at N_h), largest remainder. Within each
stratum the units are permuted with the seed; the first n_h are the sample and their position
is ``rank``. A final allocation m_h <= n_h keeps ranks < m_h, which is again a simple random
sample of the stratum, so the design-size draw never changes after the pilot.

Pilot (recall_points only): ``pilot_items`` units split over strata proportional to n_h; they
are the lowest ranks of their strata, so they stay in every final allocation.

Near-miss follow-up (``recall_nearmiss``, sitting 1b): every sitting-1 primary item answered
No/Unsure outside the ``exclude_strata`` (the area-sample ``outside`` points) gets the circle
question; a seeded simple random sample of ``overlap_n`` sitting-1 primary items (any answer,
any stratum) gets the original question again for inter-rater agreement. Both kinds are
interleaved in one seeded random order with opaque ids.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.audit.core import (
    ALLOCATED_TASKS,
    PILOT_TASK,
    TASKS,
    AuditError,
    largest_remainder,
    replay,
)

PREFIX = {"recall_points": "rp", "precision_pool": "pp", "flood_trace": "ft",
          "recall_nearmiss": "nm", "precision_microsoft": "pm", "flood_trace_p2": "fq"}
NEARMISS_COLUMNS = ["item_id", "task", "kind", "question", "parent_item", "stratum", "x", "y",
                    "radius_m", "order_key", "pilot"]


def design_sizes(frame_n: dict[str, int], total: int, min_per: int) -> dict[str, int]:
    """sqrt(N_h) allocation with a per-stratum minimum, capped at N_h."""
    keys = sorted(k for k, v in frame_n.items() if v > 0)
    total = min(total, sum(frame_n[k] for k in keys))
    out = {k: min(frame_n[k], min_per) for k in keys}
    if sum(out.values()) > total:          # minimums alone exceed the total: proportional
        return largest_remainder({k: float(out[k]) for k in keys}, total)
    for _ in range(len(keys) + 1):
        left = total - sum(out.values())
        free = [k for k in keys if out[k] < frame_n[k]]
        if left <= 0 or not free:
            break
        share = largest_remainder({k: math.sqrt(frame_n[k]) for k in free}, left)
        for k in free:
            out[k] += min(share[k], frame_n[k] - out[k])
    return out


def rebalance(sizes: dict[str, int], frame_n: dict[str, int], stratum: str, n: int,
              source: str) -> dict[str, int]:
    """Move items from ``source`` to ``stratum`` until it has ``n`` (capped at its frame and
    at what ``source`` can give above ``1``); the total is unchanged."""
    out = dict(sizes)
    if stratum not in out or source not in out:
        return out
    want = min(n, frame_n.get(stratum, 0)) - out[stratum]
    move = max(0, min(want, out[source] - 1))
    out[stratum] += move
    out[source] -= move
    return out


def draw(frame: pd.DataFrame, sizes: dict[str, int], seed: int, task: str) -> pd.DataFrame:
    """Seeded within-stratum permutation; returns the sampled rows with ``rank``."""
    rng = np.random.default_rng([seed, 1, TASKS.index(task)])
    parts = []
    for stratum in sorted(sizes):
        rows = frame.loc[frame["stratum"] == stratum].sort_values("unit")
        perm = rng.permutation(len(rows))[: sizes[stratum]]
        part = rows.iloc[perm].copy()
        part["rank"] = np.arange(len(part))
        part["frame_n"] = len(rows)
        part["design_n"] = sizes[stratum]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def finish_items(sample: pd.DataFrame, task: str, seed: int, pilot_items: int) -> pd.DataFrame:
    """Add pilot flags, presentation keys, opaque ids and design inclusion probabilities."""
    rng = np.random.default_rng([seed, 2, TASKS.index(task)])
    sample = sample.copy()
    sample["pi_design"] = np.where(sample["frame_n"] > 0,
                                   sample["design_n"] / sample["frame_n"].clip(lower=1), np.nan)
    sample["pilot"] = False
    if task == PILOT_TASK and pilot_items > 0:
        design = sample.groupby("stratum")["design_n"].first().to_dict()
        pilot = largest_remainder({k: float(v) for k, v in design.items()}, pilot_items)
        sample["pilot"] = sample["rank"].to_numpy() < sample["stratum"].map(pilot).to_numpy()
    sample["order_key"] = rng.random(len(sample))
    order = np.argsort(sample["order_key"].to_numpy(), kind="stable")
    ids = np.empty(len(sample), dtype=object)
    ids[order] = [f"{PREFIX[task]}{i + 1:04d}" for i in range(len(sample))]
    sample.insert(0, "item_id", ids)
    sample.insert(1, "task", task)
    return sample.sort_values("item_id").reset_index(drop=True)


# --- pilot allocation (Step 5) ----------------------------------------------------------------


def pilot_timing(events: list[dict[str, Any]], pilot_ids: set[str]) -> dict[str, Any]:
    """Seconds per pilot item from the effective first-question answers of pilot slots."""
    state = replay(events)
    ms = [int(e["ms"]) for e in state.effective
          if e["slot"] in pilot_ids and e["question"] == PILOT_TASK]
    if len(ms) < len(pilot_ids):
        raise AuditError(f"pilot incomplete: {len(ms)} of {len(pilot_ids)} items answered")
    s = np.asarray(ms, dtype=float) / 1000.0
    return {"n": len(ms), "median_s": float(np.median(s)), "mean_s": float(s.mean()),
            "q25_s": float(np.quantile(s, 0.25)), "q75_s": float(np.quantile(s, 0.75)),
            "total_s": float(s.sum())}


def allocation_from_pilot(cfg: dict[str, Any], timing: dict[str, Any],
                          design_totals: dict[str, int]) -> dict[str, Any]:
    """Final task sizes: as many items as fit one sitting's answering budget at the pilot median
    (times the task's ``time_factor``), within [floor_per_task, design size]."""
    t = cfg["time"]
    budget_s = float(t["sitting_budget_min"]) * 60.0
    rep = float(t["repeat_frac"])
    floor = int(t["floor_per_task"])
    tasks: dict[str, Any] = {}
    total_min = 0.0
    for task in ALLOCATED_TASKS:
        tcfg = cfg["tasks"][task]
        factor = float(tcfg.get("time_factor", 1.0))
        per_item = timing["median_s"] * factor
        fit = int(budget_s / (per_item * (1.0 + rep))) if per_item > 0 else design_totals[task]
        design = int(design_totals[task])
        n = min(design, max(floor, fit))
        minutes = n * (1.0 + rep) * per_item / 60.0
        total_min += minutes
        tasks[task] = {"n": n, "design_n": design, "fit_in_budget": fit,
                       "time_factor": factor, "expected_min": round(minutes, 1),
                       "provisional": task != PILOT_TASK}
    over = total_min > float(t["cap_min"])
    return {"rule": (f"n = min(design, max({floor}, floor({t['sitting_budget_min']} min / "
                     f"(pilot median s x time_factor x (1 + {rep})))))"),
            "pilot": timing, "tasks": tasks, "expected_total_min": round(total_min, 1),
            "cap_min": float(t["cap_min"]), "exceeds_cap": over,
            "note": ("precision_pool and flood_trace sizes are provisional: they assume the "
                     "recall_points pilot median times time_factor. Their page shows the "
                     "remaining time from their own logged median, a sitting can be split, and "
                     "the cap stops the page.")}


# --- near-miss follow-up (sitting 1b) ---------------------------------------------------------


def nearmiss_items(parent: pd.DataFrame, answers: dict[str, str], tcfg: dict[str, Any],
                   seed: int, parent_task: str = PILOT_TASK) -> pd.DataFrame:
    """Items of ``recall_nearmiss`` from the parent sample and its effective primary answers.

    ``answers`` maps parent item id -> sitting-1 answer (every parent item must have one).
    """
    from nakkhu.audit.core import NEARMISS_TASK

    missing = sorted(set(parent["item_id"]) - set(answers))
    if missing:
        raise AuditError(f"{len(missing)} {parent_task} items have no answer (e.g. "
                         f"{missing[0]}); finish sitting 1 first")
    parent = parent.sort_values("item_id").reset_index(drop=True)
    ans = parent["item_id"].map(answers)
    include = [str(a) for a in tcfg.get("include_answers", ["no", "unsure"])]
    if not set(include) <= {"yes", "no", "unsure"}:
        raise AuditError(f"include_answers {include} must be answers (quote yes/no in YAML)")
    exclude = [str(s) for s in tcfg.get("exclude_strata", ["outside"])]
    near = parent.loc[ans.isin(include).to_numpy() & ~parent["stratum"].isin(exclude).to_numpy()]
    rng = np.random.default_rng([seed, 7, 0])
    k = min(int(tcfg.get("overlap_n", 0)), len(parent))
    pick = np.sort(rng.choice(len(parent), size=k, replace=False)) if k else np.array([], int)
    over = parent.iloc[pick]
    radius = float(tcfg["radius_m"])
    parts = [pd.DataFrame({"kind": "nearmiss", "question": NEARMISS_TASK,
                           "parent_item": near["item_id"].to_numpy(),
                           "stratum": near["stratum"].to_numpy(),
                           "x": near["x"].to_numpy(float), "y": near["y"].to_numpy(float),
                           "radius_m": radius}),
             pd.DataFrame({"kind": "overlap", "question": parent_task,
                           "parent_item": over["item_id"].to_numpy(),
                           "stratum": over["stratum"].to_numpy(),
                           "x": over["x"].to_numpy(float), "y": over["y"].to_numpy(float),
                           "radius_m": np.nan})]
    out = pd.concat(parts, ignore_index=True)
    out["order_key"] = np.random.default_rng([seed, 8, 0]).random(len(out))
    out["pilot"] = False
    order = np.argsort(out["order_key"].to_numpy(), kind="stable")
    ids = np.empty(len(out), dtype=object)
    ids[order] = [f"{PREFIX[NEARMISS_TASK]}{i + 1:04d}" for i in range(len(out))]
    out["item_id"] = ids
    out["task"] = NEARMISS_TASK
    return out[NEARMISS_COLUMNS].sort_values("item_id").reset_index(drop=True)
