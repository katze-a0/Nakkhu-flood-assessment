"""Per-band histogram matching of a GE01 strip to WV02 (plan 009 Step 3).

The matching population is the whole observable extent of the strip (valid in both images,
not cloud or shadow). It is never restricted by distance to the channel: that would bring a
predictor into the reference. Pure array code.
"""

from __future__ import annotations

import numpy as np


def histograms(image: np.ndarray, mask: np.ndarray, bins: int = 256) -> np.ndarray:
    """``(bands, bins)`` counts of an integer image (0..bins-1) over ``mask``."""
    out = np.zeros((image.shape[0], bins), dtype=np.int64)
    for k in range(image.shape[0]):
        values = np.clip(image[k][mask].astype(np.int64), 0, bins - 1)
        out[k] = np.bincount(values, minlength=bins)
    return out


def lut_from_histograms(src_hist: np.ndarray, ref_hist: np.ndarray) -> np.ndarray:
    """``(bands, bins)`` float lookup mapping each source level to the reference level of the
    same cumulative share (mid-rank CDFs, linear interpolation)."""
    bands, bins = src_hist.shape
    lut = np.zeros((bands, bins), dtype=np.float32)
    levels = np.arange(bins, dtype=np.float64)
    for k in range(bands):
        s, r = src_hist[k].astype(np.float64), ref_hist[k].astype(np.float64)
        if s.sum() == 0 or r.sum() == 0:
            lut[k] = levels
            continue
        s_cdf = (np.cumsum(s) - 0.5 * s) / s.sum()
        r_cdf = (np.cumsum(r) - 0.5 * r) / r.sum()
        keep = r > 0
        lut[k] = np.interp(s_cdf, r_cdf[keep], levels[keep])
    return lut


def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map an image (integer or float levels) through the per-band lookup (linear between levels)."""
    out = np.empty(image.shape, dtype=np.float32)
    levels = np.arange(lut.shape[1], dtype=np.float32)
    for k in range(image.shape[0]):
        out[k] = np.interp(image[k].astype(np.float32), levels, lut[k])
    return out


def match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram-match ``src`` to ``ref`` over ``mask`` (float arrays are rounded to levels
    for the histograms only)."""
    s = np.rint(src).astype(np.int64)
    r = np.rint(ref).astype(np.int64)
    lut = lut_from_histograms(histograms(s, mask, bins), histograms(r, mask, bins))
    return apply_lut(src, lut)
