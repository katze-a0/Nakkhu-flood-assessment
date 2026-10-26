"""Change score and classes (plan 009 Step 5; rule pre-registered in prereg v2).

- Robust z per feature and strip over the buildings observable in that strip:
  z = (x - median) / (1.4826 MAD). A feature with MAD 0 is left out (NaN).
- Signed z: sign +1 or -1, or ``"abs"`` (z of |x|: change in either direction).
- Per building, the strip is the first in preference order in which it is observable.
- Score = mean of the available signed z of the building's strip (required features must
  all be present, optional ones such as Sentinel-2 may be missing).
- Class: ``positive`` if the score exceeds the boundary of a 2-component 1-D Gaussian mixture
  fitted by EM on the scores of observable buildings (the smallest score above the background
  component's mean where its posterior falls to 0.5; background = the larger-weight
  component), else ``negative``;
  unobservable buildings are ``not_observable``, never negative.

No input here is HAND, elevation, distance to a channel or an audit answer, and nothing is
tuned on them (plan 009 STOP condition).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826
CLASSES = ("positive", "negative", "not_observable")


def robust_z(x: np.ndarray, population: np.ndarray) -> np.ndarray:
    """z of every value against the median and MAD of ``x[population]`` (NaN-aware)."""
    x = np.asarray(x, dtype=np.float64)
    ref = x[population & np.isfinite(x)]
    if ref.size == 0:
        return np.full(x.shape, np.nan)
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * MAD_SCALE
    if mad == 0:
        return np.full(x.shape, np.nan)
    return (x - med) / mad


def signed_z(x: np.ndarray, population: np.ndarray, sign: int | str) -> np.ndarray:
    if sign == "abs":
        return robust_z(np.abs(np.asarray(x, dtype=np.float64)), population)
    if sign not in (1, -1):
        raise ValueError(f"feature sign must be 1, -1 or 'abs', not {sign!r}")
    return float(sign) * robust_z(x, population)


def score_strip(table: pd.DataFrame, signs: Mapping[str, int | str],
                required: Sequence[str], observable: np.ndarray) -> tuple[np.ndarray,
                                                                         pd.DataFrame]:
    """(score, signed z table) of one strip's feature table."""
    z = pd.DataFrame({f"z_{name}": signed_z(table[name].to_numpy(), observable, sign)
                      for name, sign in signs.items()}, index=table.index)
    values = z.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # rows without any feature -> NaN
        score = np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=1) \
            if values.shape[1] else np.full(len(table), np.nan)
    need = [f"z_{n}" for n in required if n in signs]
    if need:
        missing = ~np.isfinite(z[need].to_numpy()).all(axis=1)
        score = np.where(missing, np.nan, score)
    return np.where(observable, score, np.nan), z


@dataclass(frozen=True)
class Mixture:
    weights: tuple[float, float]
    means: tuple[float, float]
    sds: tuple[float, float]
    log_likelihood: float
    iterations: int
    converged: bool

    def posterior_background(self, x: np.ndarray) -> np.ndarray:
        bg = int(np.argmax(self.weights))
        dens = [self.weights[k] * _normal(x, self.means[k], self.sds[k]) for k in (0, 1)]
        total = dens[0] + dens[1]
        return np.divide(dens[bg], total, out=np.full(np.shape(x), 0.5), where=total > 0)


def _normal(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((np.asarray(x, dtype=np.float64) - mu) / sd) ** 2) / (
        sd * np.sqrt(2 * np.pi))


def fit_mixture(x: np.ndarray, init_quantiles: Sequence[float] = (0.25, 0.75),
                max_iter: int = 1000, tol: float = 1e-10) -> Mixture:
    """Two-component 1-D Gaussian mixture by EM; deterministic quantile initialisation."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("a mixture needs at least 10 finite scores")
    mu = np.quantile(x, list(init_quantiles)).astype(np.float64)
    sd = np.full(2, x.std())
    w = np.full(2, 0.5)
    floor = max(1e-6, 1e-3 * float(x.std()))
    previous = -np.inf
    converged = False
    it = 0
    ll = -np.inf
    while it < max_iter:
        it += 1
        dens = np.vstack([w[k] * _normal(x, mu[k], sd[k]) for k in range(2)])
        total = np.maximum(dens.sum(axis=0), 1e-300)
        ll = float(np.log(total).sum())
        resp = dens / total
        nk = resp.sum(axis=1)
        w = nk / x.size
        mu = (resp @ x) / nk
        sd = np.sqrt(np.maximum((resp * (x[None, :] - mu[:, None]) ** 2).sum(axis=1) / nk,
                                floor ** 2))
        if abs(ll - previous) < tol * max(1.0, abs(ll)):
            converged = True
            break
        previous = ll
    return Mixture(weights=(float(w[0]), float(w[1])), means=(float(mu[0]), float(mu[1])),
                   sds=(float(sd[0]), float(sd[1])), log_likelihood=ll, iterations=it,
                   converged=converged)


def boundary(m: Mixture) -> float:
    """Upper decision point of the background component (the one with the larger weight):
    the smallest x above its mean where its posterior falls to 0.5; +inf if it never does.

    Positive = scores above this point, where the other component is the more likely one.
    (The other component may be a wide one centred near the background; only its upper tail
    counts as change towards the flood signature.)"""
    bg = int(np.argmax(m.weights))
    ot = 1 - bg
    w1, m1, s1 = m.weights[bg], m.means[bg], m.sds[bg]
    w2, m2, s2 = m.weights[ot], m.means[ot], m.sds[ot]
    # log(w1 N1) = log(w2 N2)  ->  a x^2 + b x + c = 0
    a = 0.5 / s2 ** 2 - 0.5 / s1 ** 2
    b = m1 / s1 ** 2 - m2 / s2 ** 2
    c = (0.5 * m2 ** 2 / s2 ** 2 - 0.5 * m1 ** 2 / s1 ** 2
         + np.log((w1 * s2) / (w2 * s1)))
    if abs(a) < 1e-15:
        roots = np.array([-c / b]) if abs(b) > 1e-15 else np.array([])
    else:
        disc = b * b - 4 * a * c
        roots = np.array([]) if disc < 0 else np.array(
            [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)])
    roots = np.sort(roots[roots > m1])
    return float(roots[0]) if roots.size else float("inf")


def classify(score: np.ndarray, observable: np.ndarray, cut: float) -> np.ndarray:
    out = np.where(observable & np.isfinite(score),
                   np.where(score > cut, "positive", "negative"), "not_observable")
    return out.astype(object)


def choose_strip(observable: Mapping[str, np.ndarray], order: Sequence[str]) -> np.ndarray:
    """Per building, the first strip in ``order`` where it is observable, else ``""``."""
    n = len(next(iter(observable.values())))
    out = np.full(n, "", dtype=object)
    for strip in reversed(list(order)):
        out = np.where(observable[strip], strip, out)
    return out


def score_buildings(per_strip: Mapping[str, pd.DataFrame], order: Sequence[str],
                    flood_signs: Mapping[str, int | str], damage_signs: Mapping[str, int | str],
                    required: Sequence[str], gmm: Mapping[str, Any]) -> tuple[pd.DataFrame,
                                                                             dict[str, Any]]:
    """Flood and damage score/class for one inventory from its per-strip feature tables
    (same row order; each with an ``observable`` column and the feature columns)."""
    obs = {s: per_strip[s]["observable"].to_numpy(dtype=bool) for s in order}
    strip = choose_strip(obs, order)
    n = len(strip)
    out = pd.DataFrame(index=per_strip[order[0]].index)
    flood = np.full(n, np.nan)
    damage = np.full(n, np.nan)
    zcols: dict[str, np.ndarray] = {}
    for s in order:
        sel = strip == s
        fs, fz = score_strip(per_strip[s], flood_signs, required, obs[s])
        ds, dz = score_strip(per_strip[s], damage_signs, [], obs[s])
        flood[sel] = fs[sel]
        damage[sel] = ds[sel]
        for col in [*fz.columns, *dz.columns]:
            src = fz[col] if col in fz.columns else dz[col]
            zcols.setdefault(col, np.full(n, np.nan))[sel] = src.to_numpy()[sel]
    observable = strip != ""
    meta: dict[str, Any] = {}
    classes: dict[str, np.ndarray] = {}
    for name, score in (("flood", flood), ("damage", damage)):
        ok = observable & np.isfinite(score)
        if ok.sum() >= 10:
            mix = fit_mixture(score[ok], gmm.get("init_quantiles", (0.25, 0.75)),
                              int(gmm.get("max_iter", 1000)), float(gmm.get("tol", 1e-10)))
            cut = boundary(mix)
            meta[name] = {"mixture": mix.__dict__, "boundary": cut, "n_scored": int(ok.sum())}
        else:
            cut = float("inf")
            meta[name] = {"mixture": None, "boundary": cut, "n_scored": int(ok.sum())}
        classes[name] = classify(score, ok, cut)
    out["observable"] = observable & np.isfinite(flood)
    out["strip"] = strip
    out["score"] = flood
    out["class"] = classes["flood"]
    out["damage_score"] = damage
    out["damage_class"] = classes["damage"]
    for col, values in zcols.items():
        out[col] = values
    return out, meta
