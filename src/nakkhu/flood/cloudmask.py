"""Observability mask of a GE01 strip against WV02 (plan 009 Step 2).

Works on a coarse aggregate (mean of ``agg_px`` x ``agg_px`` 0.5 m pixels, 4 m by default):

- cloud core: bright and low-saturation in GE01 (raw DN) but not bright in WV02, connected
  area >= ``min_area_m2``; grown into connected cells that are brighter than WV02 after
  histogram matching and still low-saturation (cloud halo and thin edges);
- shadow: matched GE01 much darker than WV02, dark and low-saturation, connected area >=
  ``min_area_m2`` (smaller dark patches are building shadows, handled per pixel instead);
- both dilated by ``dilate_m``. Masked cells are *not observable*, never "not affected".

Pure array code: no HAND, DEM or river input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage as ndi


def brightness(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32).mean(axis=0)


def saturation(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    mx, mn = rgb.max(axis=0), rgb.min(axis=0)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0).astype(np.float32)


def area_filter(mask: np.ndarray, cell_m2: float, min_area_m2: float) -> np.ndarray:
    """Connected components (8-neighbour) of ``mask`` with area >= ``min_area_m2``."""
    labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    area = np.bincount(labels.ravel(), minlength=n + 1) * cell_m2
    keep = area >= min_area_m2
    keep[0] = False
    return keep[labels]


def grow(core: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Cells of ``core | candidate`` connected (8-neighbour) to a core cell."""
    labels, n = ndi.label(core | candidate, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(core, dtype=bool)
    hit = np.zeros(n + 1, dtype=bool)
    hit[np.unique(labels[core])] = True
    hit[0] = False
    return hit[labels]


def dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return mask.copy()
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    disc = (x * x + y * y) <= radius_cells * radius_cells
    return ndi.binary_dilation(mask, structure=disc)


def cloud_core(post_raw: np.ndarray, pre: np.ndarray, valid: np.ndarray, cell_m2: float,
               opts: Mapping[str, Any]) -> np.ndarray:
    core = (valid & (brightness(post_raw) >= float(opts["bright_min"]))
            & (saturation(post_raw) <= float(opts["sat_max"]))
            & (brightness(pre) < float(opts["pre_bright_max"])))
    return area_filter(core, cell_m2, float(opts["min_area_m2"]))


def cloud_mask(core: np.ndarray, post_matched: np.ndarray, pre: np.ndarray, valid: np.ndarray,
               opts: Mapping[str, Any]) -> np.ndarray:
    halo = (valid & (brightness(post_matched) - brightness(pre) >= float(opts["grow_delta_min"]))
            & (saturation(post_matched) <= float(opts["grow_sat_max"])))
    return grow(core, halo) & valid


def shadow_mask(post_matched: np.ndarray, pre: np.ndarray, valid: np.ndarray, cell_m2: float,
                opts: Mapping[str, Any]) -> np.ndarray:
    bp, bq = brightness(post_matched), brightness(pre)
    dark = (valid & (bp <= float(opts["ratio_max"]) * bq) & (bp <= float(opts["bright_max"]))
            & (saturation(post_matched) <= float(opts["sat_max"])))
    return area_filter(dark, cell_m2, float(opts["min_area_m2"]))


def observability(post_raw: np.ndarray, pre: np.ndarray, valid: np.ndarray, cell_m: float,
                  opts: Mapping[str, Any], match) -> dict[str, np.ndarray]:
    """Cloud, shadow and the dilated unobservable mask on the aggregate grid.

    ``match(post, pre, mask)`` histogram-matches ``post`` to ``pre`` over ``mask`` (the
    provisional matching that puts the halo and shadow tests on the WV02 scale).
    """
    cell_m2 = cell_m * cell_m
    core = cloud_core(post_raw, pre, valid, cell_m2, opts["cloud"])
    radius = int(round(float(opts["dilate_m"]) / cell_m))
    provisional = match(post_raw, pre, valid & ~dilate(core, radius))
    cloud = cloud_mask(core, provisional, pre, valid, opts["cloud"])
    shadow = shadow_mask(provisional, pre, valid, cell_m2, opts["shadow"]) & ~cloud
    unobservable = (dilate(cloud, radius) | dilate(shadow, radius)) & valid
    return {"cloud": cloud, "shadow": shadow, "unobservable": unobservable}
