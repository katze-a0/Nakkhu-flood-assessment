"""C2 falsifier outcomes exactly as pre-registered (prereg v5 ``falsifiers.C2``)."""

from __future__ import annotations

from typing import Any

import numpy as np

ECE_MAX = 0.1
MISS_MAX = 0.10


def c2_outcome(u1: dict[str, Any], split: dict[str, Any], decomposition: dict[str, Any],
               *, ece_max: float = ECE_MAX, miss_max: float = MISS_MAX) -> dict[str, Any]:
    """``u1``: :func:`calibrate.calibrate_u1` result; ``split``: :func:`ppi.split_sample`;
    ``decomposition``: :func:`decompose.terrain_dominates` on HAND <= 3 m."""
    chosen = u1["chosen"]
    e = float(u1["methods"][chosen]["ece"])
    miss = float(split["miss_rate"])
    return {
        "ece": {"method": chosen, "value": e, "max": ece_max,
                "pass": bool(np.isfinite(e) and e <= ece_max),
                "other_methods": {m: float(d["ece"]) for m, d in u1["methods"].items()
                                  if m != chosen}},
        "ppi": {"miss_rate": miss, "max": miss_max,
                "pass": bool(np.isfinite(miss) and miss <= miss_max),
                "valid_repeats": int(split["valid"]), "rule": split["rule"]},
        "decomposition": {**decomposition, "gating": False,
                          "note": "a finding either way (prereg)"},
        "contribution_fails_if": "intervals not computable (non-finite) or not calibrated "
                                 "(ece or ppi fails)",
    }
