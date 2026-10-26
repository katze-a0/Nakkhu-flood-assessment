"""Radiometry: 2-98 percentile stretch to 8-bit RGB, nodata masking and JPG export (plan 003).

Two modes:

- ``global`` (default): per-scene, per-band percentiles computed once over every valid pixel of
  the scene mosaic (from exact 256-bin histograms, see :func:`percentiles_from_histogram`), so
  the same ground value maps to the same output value in every tile.
- ``per_tile``: the prototype stretch (``archive/code/tif_to_jpg.ipynb`` cell 5,
  ``normalize_band``), percentiles per tile and over all pixels including nodata. Kept only to
  reproduce the prototype in the regression test.

Nodata stays 0 in the image; validity is carried separately as a boolean mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODES = ("global", "per_tile")


def normalize_band(band):
    """Percentile clip then stretch to 0-255."""
    # Verbatim port of the prototype (archive/code/tif_to_jpg.ipynb cell 5); do not edit.
    p2 = np.percentile(band, 2)
    p98 = np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    if p98 - p2 > 0:
        band = (band - p2) / (p98 - p2) * 255.0
    else:
        band = np.zeros_like(band)
    return band.astype("uint8")


def per_tile_stretch(rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Prototype per-tile stretch of a ``(3, H, W)`` array -> ``(H, W, 3)`` uint8.

    As in the prototype, percentiles include nodata pixels; nodata pixels (``~valid``) are then
    set to black.
    """
    bands = [normalize_band(rgb[i].astype("float32")) for i in range(3)]
    if valid is not None:
        for band in bands:
            band[~valid] = 0
    return np.stack(bands, axis=-1)


def percentiles_from_histogram(hist: Sequence[int] | np.ndarray, q: float) -> float:
    """``np.percentile(values, q)`` (linear method) of integer values given as a histogram.

    ``hist[v]`` counts the occurrences of value ``v``. The result is exactly what numpy returns
    for the expanded array, without materialising it.
    """
    counts = np.asarray(hist, dtype=np.int64)
    n = int(counts.sum())
    if n == 0:
        raise ValueError("empty histogram")
    position = (n - 1) * (q / 100.0)
    lower = int(np.floor(position))
    upper = min(lower + 1, n - 1)
    cumulative = np.cumsum(counts)
    v_lower = int(np.searchsorted(cumulative, lower, side="right"))
    v_upper = int(np.searchsorted(cumulative, upper, side="right"))
    return float(v_lower + (position - lower) * (v_upper - v_lower))


def scene_percentiles(
    histograms: Sequence[Any] | np.ndarray, q: Sequence[float] = (2.0, 98.0)
) -> list[list[float]]:
    """Per-band ``[low, high]`` percentiles from per-band histograms of valid pixels."""
    return [[percentiles_from_histogram(h, q[0]), percentiles_from_histogram(h, q[1])]
            for h in histograms]


def global_stretch(
    rgb: np.ndarray, valid: np.ndarray, percentiles: Sequence[Sequence[float]]
) -> np.ndarray:
    """Stretch ``(3, H, W)`` with fixed scene percentiles -> ``(H, W, 3)`` uint8.

    Same arithmetic as :func:`normalize_band` (float32 clip, scale, truncate), but with the
    scene's ``[low, high]`` per band; nodata pixels are set to 0.
    """
    out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype="uint8")
    for i in range(3):
        low, high = float(percentiles[i][0]), float(percentiles[i][1])
        band = np.clip(rgb[i].astype("float32"), low, high)
        if high - low > 0:
            band = (band - low) / (high - low) * 255.0
        else:
            band = np.zeros_like(band)
        out[:, :, i] = band.astype("uint8")
    out[~valid] = 0
    return out


def stretch(
    rgb: np.ndarray,
    valid: np.ndarray,
    mode: str = "global",
    percentiles: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    if mode == "global":
        if percentiles is None:
            raise ValueError("global stretch needs the scene percentiles")
        return global_stretch(rgb, valid, percentiles)
    if mode == "per_tile":
        return per_tile_stretch(rgb, valid)
    raise ValueError(f"unknown radiometry mode {mode!r}; expected one of {MODES}")


def write_jpg(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    """Write an ``(H, W, 3)`` uint8 image as JPEG (Pillow, as in the prototype)."""
    Image.fromarray(image).save(path, "JPEG", quality=quality)


def write_mask_png(path: str | Path, valid: np.ndarray) -> None:
    """Write a validity mask as a 1-bit PNG (white = valid)."""
    Image.fromarray(valid.astype(bool)).save(path, "PNG")
