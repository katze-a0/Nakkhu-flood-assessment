"""Building features of the change map (plan 009 Step 4). All features are post - pre.

Pixel features (0.5 m, WV02 and histogram-matched, co-registered GE01): brightness =
mean(R, G, B); saturation = (max - min) / max; sediment index = (R - B) / (R + B); texture = std
of brightness in 3 x 3 px. A building's ring and roof feature is the median of the per-pixel
difference over its usable pixels (valid in both images and not darker than
``pixel_dark_max`` in either: building shadows move between January and October).

Geometry per building: roof = footprint; ground ring = footprint buffered ``ring_m[1]`` minus
footprint buffered ``ring_m[0]``, minus every footprint of the ground mask (neighbouring roofs
are not ground). Observable (per strip) = no unobservable mask cell and only valid post pixels
within ``observable_buffer_m`` of the footprint, and at least ``min_ring_px`` usable ring pixels.

Sentinel-2 (10 m): MNDWI, BSI and NDVI on 23 Sep and 1 Oct 2024, mean over the clear pixels of
the 3 x 3 window around the building centroid.

Pure array and geometry code: no HAND, DEM or river input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from scipy import ndimage as ndi

PIXEL_FEATURES = ("brightness", "saturation", "sediment", "texture")
S2_FEATURES = ("mndwi", "bsi", "ndvi")


def pixel_features(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel features of a ``(3, h, w)`` RGB array (any numeric scale; DN here)."""
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[0], rgb[1], rgb[2]
    bright = (r + g + b) / 3.0
    mx, mn = rgb.max(axis=0), rgb.min(axis=0)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    sed = np.where(r + b > 0, (r - b) / np.maximum(r + b, 1e-6), 0.0)
    mean = ndi.uniform_filter(bright, 3, mode="nearest")
    mean_sq = ndi.uniform_filter(bright * bright, 3, mode="nearest")
    tex = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return {"brightness": bright, "saturation": sat.astype(np.float32),
            "sediment": sed.astype(np.float32), "texture": tex.astype(np.float32)}


@dataclass(frozen=True)
class Window:
    """A pixel window of the 0.5 m grid: top-left map coordinate and size."""

    x0: float
    y0: float
    height: int
    width: int
    pixel: float = 0.5

    def pixel_box(self, bounds: tuple[float, float, float, float]
                  ) -> tuple[tuple[int, int, int, int], bool]:
        """((r0, r1, c0, c1), clipped): pixels whose centres may fall in ``bounds``, clipped to
        the window; ``clipped`` is True when the window cut the box."""
        minx, miny, maxx, maxy = bounds
        c0 = int(np.floor((minx - self.x0) / self.pixel))
        c1 = int(np.ceil((maxx - self.x0) / self.pixel))
        r0 = int(np.floor((self.y0 - maxy) / self.pixel))
        r1 = int(np.ceil((self.y0 - miny) / self.pixel))
        clipped = c0 < 0 or r0 < 0 or c1 > self.width or r1 > self.height
        return (max(0, r0), min(self.height, r1), max(0, c0), min(self.width, c1)), clipped

    def centres(self, r0: int, r1: int, c0: int, c1: int) -> tuple[np.ndarray, np.ndarray]:
        cols = self.x0 + (np.arange(c0, c1) + 0.5) * self.pixel
        rows = self.y0 - (np.arange(r0, r1) + 0.5) * self.pixel
        xx, yy = np.meshgrid(cols, rows)
        return xx, yy


@dataclass(frozen=True)
class BuildingMasks:
    box: tuple[int, int, int, int]
    clipped: bool
    roof: np.ndarray
    ring: np.ndarray
    zone: np.ndarray        # footprint buffered by observable_buffer_m


def building_masks(polygon: Any, window: Window, built: np.ndarray, ring_m: Sequence[float],
                   observable_buffer_m: float) -> BuildingMasks | None:
    inner_m, outer_m = float(ring_m[0]), float(ring_m[1])
    outer = shapely.buffer(polygon, outer_m)
    zone_geom = shapely.buffer(polygon, observable_buffer_m) if observable_buffer_m != outer_m \
        else outer
    reach = max(outer_m, observable_buffer_m)
    minx, miny, maxx, maxy = polygon.bounds
    (r0, r1, c0, c1), clipped = window.pixel_box((minx - reach, miny - reach, maxx + reach,
                                                  maxy + reach))
    if r1 <= r0 or c1 <= c0:
        return None
    xx, yy = window.centres(r0, r1, c0, c1)
    inner = shapely.buffer(polygon, inner_m)
    for geom in (polygon, inner, outer, zone_geom):
        shapely.prepare(geom)
    roof = shapely.contains_xy(polygon, xx, yy)
    ring = (shapely.contains_xy(outer, xx, yy) & ~shapely.contains_xy(inner, xx, yy)
            & ~built[r0:r1, c0:c1])
    zone = shapely.contains_xy(zone_geom, xx, yy)
    return BuildingMasks(box=(r0, r1, c0, c1), clipped=clipped, roof=roof, ring=ring, zone=zone)


def strip_features(masks: BuildingMasks, diffs: Mapping[str, np.ndarray], usable: np.ndarray,
                   post_valid: np.ndarray, unobservable: np.ndarray,
                   min_ring_px: int) -> dict[str, Any]:
    """Ring/roof medians of the per-pixel differences for one strip, plus observability."""
    r0, r1, c0, c1 = masks.box
    use = usable[r0:r1, c0:c1]
    ring = masks.ring & use
    roof = masks.roof & use
    zone = masks.zone
    n_ring, n_roof = int(ring.sum()), int(roof.sum())
    masked = bool((unobservable[r0:r1, c0:c1] & zone).any())
    invalid = bool((~post_valid[r0:r1, c0:c1] & zone).any())
    if masks.clipped:
        reason = "outside the image window"
    elif invalid:
        reason = "no valid post pixels"
    elif masked:
        reason = "cloud or shadow"
    elif n_ring < min_ring_px:
        reason = "too little usable ground ring"
    else:
        reason = ""
    out: dict[str, Any] = {"observable": reason == "", "reason": reason,
                           "n_ring_px": n_ring, "n_roof_px": n_roof}
    for name in PIXEL_FEATURES:
        d = diffs[name][r0:r1, c0:c1]
        out[f"ring_d_{name}"] = float(np.median(d[ring])) if n_ring else float("nan")
        out[f"roof_d_{name}"] = float(np.median(d[roof])) if n_roof else float("nan")
    return out


# --- Sentinel-2 -------------------------------------------------------------------------------


def s2_indices(bands: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """MNDWI, BSI and NDVI from reflectance bands B02, B03, B04, B08, B11 (same grid).

    Reflectance below 0 (L2A offset on very dark pixels) is clipped to 0, so every index lies
    in [-1, 1]; a zero denominator gives NaN."""
    def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        s = a + b
        ok = s > 0
        return np.where(ok, (a - b) / np.where(ok, s, 1.0), np.nan)

    b02, b03, b04, b08, b11 = (np.clip(bands[k].astype(np.float64), 0.0, None)
                               for k in ("B02", "B03", "B04", "B08", "B11"))
    return {"mndwi": nd(b03, b11), "ndvi": nd(b08, b04),
            "bsi": nd(b11 + b04, b08 + b02)}


def s2_window_mean(values: np.ndarray, clear: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                   window_px: int, min_clear_px: int) -> np.ndarray:
    """Mean of ``values`` over the clear pixels of the ``window_px`` square around each
    (row, col); NaN where fewer than ``min_clear_px`` are clear or the window leaves the grid."""
    half = window_px // 2
    h, w = values.shape
    out = np.full(len(rows), np.nan)
    good = clear & np.isfinite(values)
    vals = np.where(good, values, 0.0)
    for k, (r, c) in enumerate(zip(rows, cols, strict=True)):
        if r - half < 0 or c - half < 0 or r + half >= h or c + half >= w:
            continue
        g = good[r - half:r + half + 1, c - half:c + half + 1]
        n = int(g.sum())
        if n >= min_clear_px:
            out[k] = float(vals[r - half:r + half + 1, c - half:c + half + 1].sum() / n)
    return out
