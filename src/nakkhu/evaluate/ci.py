"""Simple intervals and tests (plan 005 Step 4): Wilson, seed mean +- SD, Wilcoxon.

No significance language beyond these is used in the paper (plan 005).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import stats


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (no continuity correction)."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def seed_summary(values: npt.ArrayLike) -> dict[str, float]:
    """Mean and sample SD (ddof = 1) over seeds."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": len(v)}


def wilcoxon_paired(a: npt.ArrayLike, b: npt.ArrayLike) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test of paired per-block values (zero differences
    dropped, SciPy's default ``zero_method='wilcox'``)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if np.count_nonzero(d) == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(ok.sum())}
    result = stats.wilcoxon(x[ok], y[ok], alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "n": int(np.count_nonzero(d))}
