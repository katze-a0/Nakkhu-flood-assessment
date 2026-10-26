"""Step 3: per-tile registration of footprints to the image by phase correlation.

The rasterised footprint boundaries (union of Google and Microsoft, 1 px lines) are
phase-correlated with the image gradient magnitude. Both inputs get a Hann window; the
normalised cross-power spectrum is low-passed with a Gaussian of ``sigma_px`` (this widens the
correlation peak to a few pixels and suppresses the high-frequency noise that pure whitening
amplifies). The shift ``(dx, dy)`` in pixels is the one to *add* to footprint pixel coordinates
(x right, y down) so that they line up with the image.

``peak_ratio`` is the height of the highest correlation peak over the highest value outside a
``exclusion_px`` radius around it. A tile is registered (``reg_ok``) when the shift is within
``max_shift_px`` and the peak ratio is at least ``min_peak_ratio``; otherwise the shift is
set to 0 and the tile is flagged with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from rasterio.features import rasterize


@dataclass(frozen=True)
class Registration:
    dx: float
    dy: float
    peak_ratio: float
    raw_dx: float
    raw_dy: float
    ok: bool
    reason: str


def gradient_magnitude(rgb: np.ndarray, valid: np.ndarray | None = None,
                       erode_px: int = 2) -> np.ndarray:
    """Sobel gradient magnitude of the grey image; zero on and near invalid pixels."""
    grey = rgb.astype(np.float64).mean(axis=-1) if rgb.ndim == 3 else rgb.astype(np.float64)
    pad = np.pad(grey, 1, mode="edge")
    gx = ((pad[:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:])
          - (pad[:-2, :-2] + 2 * pad[1:-1, :-2] + pad[2:, :-2]))
    gy = ((pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:])
          - (pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:]))
    mag = np.hypot(gx, gy)
    if valid is not None:
        mag[~erode(valid, erode_px)] = 0.0
    return mag


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion with a square of side ``2 * radius + 1`` (outside counts as invalid)."""
    mask = mask.astype(bool)
    if radius <= 0:
        return mask
    pad = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones_like(mask)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= pad[dy:dy + h, dx:dx + w]
    return out


def boundary_raster(polygons_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """1 px boundary lines of polygons given in tile pixel coordinates."""
    lines = [shapely.boundary(p) for p in polygons_px if p is not None and not p.is_empty]
    if not lines:
        return np.zeros(shape, dtype=np.float64)
    out = rasterize(((line, 1) for line in lines), out_shape=shape, fill=0, all_touched=True,
                    dtype="uint8")
    return out.astype(np.float64)


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def correlation_surface(reference: np.ndarray, moving: np.ndarray,
                        sigma_px: float = 1.0) -> np.ndarray:
    """Low-passed phase-correlation surface; its peak is at the shift of ``moving`` onto
    ``reference`` (``reference(x) ~ moving(x - s)``), in FFT order (wrap-around)."""
    window = _hann2d(reference.shape)
    a = np.fft.fft2((reference - reference.mean()) * window)
    b = np.fft.fft2((moving - moving.mean()) * window)
    cross = a * np.conj(b)
    cross /= np.maximum(np.abs(cross), 1e-12)
    if sigma_px > 0:
        fy = np.fft.fftfreq(reference.shape[0])[:, None]
        fx = np.fft.fftfreq(reference.shape[1])[None, :]
        cross *= np.exp(-2 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(cross))


def peak_and_ratio(surface: np.ndarray, exclusion_px: int = 4) -> tuple[float, float, float]:
    """(dy, dx) of the highest peak (signed, wrap-around) and the peak ratio."""
    h, w = surface.shape
    iy, ix = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak = float(surface[iy, ix])
    yy = np.minimum(np.abs(np.arange(h) - iy), h - np.abs(np.arange(h) - iy))[:, None]
    xx = np.minimum(np.abs(np.arange(w) - ix), w - np.abs(np.arange(w) - ix))[None, :]
    outside = (yy > exclusion_px) | (xx > exclusion_px)
    second = float(surface[outside].max()) if outside.any() else 0.0
    ratio = peak / second if second > 0 else float("inf")
    dy = iy - h if iy > h // 2 else iy
    dx = ix - w if ix > w // 2 else ix
    return float(dy), float(dx), ratio


def register_tile(rgb: np.ndarray, valid: np.ndarray | None, polygons_px: np.ndarray, *,
                  max_shift_px: float, min_peak_ratio: float, sigma_px: float = 1.0,
                  exclusion_px: int = 4) -> Registration:
    """Register one tile; see the module docstring."""
    shape = rgb.shape[:2]
    boundary = boundary_raster(polygons_px, shape)
    if boundary.sum() == 0:
        return Registration(0.0, 0.0, float("nan"), float("nan"), float("nan"), False,
                            "no_footprints")
    grad = gradient_magnitude(rgb, valid)
    if valid is not None:
        boundary = boundary * valid
    surface = correlation_surface(grad, boundary, sigma_px)
    dy, dx, ratio = peak_and_ratio(surface, exclusion_px)
    if float(np.hypot(dx, dy)) > max_shift_px:
        return Registration(0.0, 0.0, ratio, dx, dy, False, "shift_too_large")
    if not ratio >= min_peak_ratio:
        return Registration(0.0, 0.0, ratio, dx, dy, False, "low_peak")
    return Registration(dx, dy, ratio, dx, dy, True, "ok")


def null_peak_ratio(rgb: np.ndarray, valid: np.ndarray | None, polygons_px: np.ndarray,
                    sigma_px: float = 1.0, exclusion_px: int = 4) -> float:
    """Peak ratio of a deliberately mismatched (image, footprints) pair: the null distribution
    used to set ``min_peak_ratio``."""
    boundary = boundary_raster(polygons_px, rgb.shape[:2])
    if boundary.sum() == 0:
        return float("nan")
    if valid is not None:
        boundary = boundary * valid
    surface = correlation_surface(gradient_magnitude(rgb, valid), boundary, sigma_px)
    return peak_and_ratio(surface, exclusion_px)[2]
