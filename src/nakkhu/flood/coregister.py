"""Per-block co-registration of the GE01 strips to WV02 (plan 009 Step 1).

Phase correlation on ground features: brightness, high-passed, with footprints, clouds and
invalid pixels set to zero after mean removal, Hann-windowed. The sub-pixel peak comes from a
parabola through the peak and its neighbours. Convention: ``(dy, dx)`` is the shift that moves
the post image onto the pre image, ``post[r + dy, c + dx] ~ pre[r, c]`` (rows down, columns
right, in pixels), so a co-registered post pixel at (r, c) is read from (r + dy, c + dx).

Pure array code: no HAND, DEM or river input (plan 009 STOP condition).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class Shift:
    dy: float
    dx: float
    peak_ratio: float
    ground_frac: float

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dy, self.dx))


def highpass(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Brightness minus its Gaussian blur (removes illumination and season gradients)."""
    image = image.astype(np.float32)
    return image - ndi.gaussian_filter(image, sigma)


def _prepare(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = highpass(image)
    if mask.any():
        out = out - out[mask].mean()
    out[~mask] = 0.0
    window = np.outer(np.hanning(out.shape[0]), np.hanning(out.shape[1])).astype(np.float32)
    return out * window


def _parabola(m1: float, c: float, p1: float) -> float:
    denom = m1 - 2.0 * c + p1
    return 0.0 if denom == 0 else float(0.5 * (m1 - p1) / denom)


def phase_correlation(pre: np.ndarray, post: np.ndarray, pre_mask: np.ndarray,
                      post_mask: np.ndarray | None = None, exclusion_px: int = 5) -> Shift:
    """Shift of ``post`` onto ``pre`` from the masked phase correlation of two equal windows.

    Each image gets its own mask (pre: footprints and invalid pixels; post: cloud, shadow and
    invalid pixels). A mask shared by both images would correlate with itself and pull the
    peak to zero shift. ``ground_frac`` is the share of pixels usable in both.
    """
    post_mask = pre_mask if post_mask is None else post_mask
    if not (pre.shape == post.shape == pre_mask.shape == post_mask.shape):
        raise ValueError("pre, post and masks must have the same shape")
    a = _prepare(pre, pre_mask)
    b = _prepare(post, post_mask)
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    cross = np.conj(fa) * fb
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft2(cross, s=a.shape)
    corr = np.fft.fftshift(corr)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    r, c = int(peak[0]), int(peak[1])
    h, w = corr.shape
    sub_r = _parabola(corr[(r - 1) % h, c], corr[r, c], corr[(r + 1) % h, c])
    sub_c = _parabola(corr[r, (c - 1) % w], corr[r, c], corr[r, (c + 1) % w])
    dy = r - h // 2 + sub_r
    dx = c - w // 2 + sub_c
    masked = corr.copy()
    masked[max(0, r - exclusion_px):r + exclusion_px + 1,
           max(0, c - exclusion_px):c + exclusion_px + 1] = -np.inf
    second = float(masked.max())
    ratio = float(corr[r, c] / second) if second > 0 else float("inf")
    return Shift(dy=float(dy), dx=float(dx), peak_ratio=ratio,
                 ground_frac=float((pre_mask & post_mask).mean()))


def shift_image(image: np.ndarray, dy: float, dx: float, order: int = 1,
                cval: float = 0.0) -> np.ndarray:
    """``out[r, c] = image[r + dy, c + dx]`` (bilinear for order 1)."""
    return ndi.shift(image.astype(np.float32), (-dy, -dx), order=order, mode="constant",
                     cval=cval)


def accept(shift: Shift, max_shift_px: float, min_peak_ratio: float,
           min_ground_frac: float) -> tuple[bool, str]:
    if shift.ground_frac < min_ground_frac:
        return False, "too little clear ground"
    if not np.isfinite(shift.peak_ratio) or shift.peak_ratio < min_peak_ratio:
        return False, "weak correlation peak"
    if shift.magnitude > max_shift_px:
        return False, "shift above cap"
    return True, "ok"


def fill_rejected(keys: list[tuple[int, int]], shifts: dict[tuple[int, int], tuple[float, float]],
                  accepted: dict[tuple[int, int], bool]) -> dict[tuple[int, int],
                                                               tuple[float, float, str]]:
    """Final (dy, dx, source) per block: its own shift if accepted, else the median of the
    accepted blocks among its 8 neighbours, else the median of every accepted block."""
    good = [k for k in keys if accepted.get(k)]
    if not good:
        raise ValueError("no block passed the co-registration checks")
    overall = (float(np.median([shifts[k][0] for k in good])),
               float(np.median([shifts[k][1] for k in good])))
    out: dict[tuple[int, int], tuple[float, float, str]] = {}
    for key in keys:
        if accepted.get(key):
            out[key] = (*shifts[key], "own")
            continue
        r, c = key
        near = [(r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                if (i or j) and accepted.get((r + i, c + j))]
        if near:
            out[key] = (float(np.median([shifts[k][0] for k in near])),
                        float(np.median([shifts[k][1] for k in near])), "neighbours")
        else:
            out[key] = (*overall, "all blocks")
    return out


def piecewise_shift(read: Callable[[int, int, int, int], np.ndarray], row0: int, col0: int,
                    height: int, width: int, block_px: int, field: Callable[[int, int],
                                                                             tuple[float, float]],
                    margin: int, order: int = 1) -> np.ndarray:
    """Co-registered window ``[row0:row0+height, col0:col0+width]`` of a post image.

    ``read(r0, c0, h, w)`` returns a ``(bands, h, w)`` array of the post image on the global
    pixel grid (zeros outside); ``field(block_row, block_col)`` gives the block's (dy, dx).
    Each block-aligned part of the window is shifted by its own block's shift.
    """
    probe = read(row0, col0, 1, 1)
    out = np.zeros((probe.shape[0], height, width), dtype=np.float32)
    br0, br1 = row0 // block_px, (row0 + height - 1) // block_px
    bc0, bc1 = col0 // block_px, (col0 + width - 1) // block_px
    for br in range(br0, br1 + 1):
        for bc in range(bc0, bc1 + 1):
            r_lo = max(row0, br * block_px)
            r_hi = min(row0 + height, (br + 1) * block_px)
            c_lo = max(col0, bc * block_px)
            c_hi = min(col0 + width, (bc + 1) * block_px)
            dy, dx = field(br, bc)
            iy, ix = int(np.floor(dy)), int(np.floor(dx))
            src = read(r_lo + iy - margin, c_lo + ix - margin, r_hi - r_lo + 2 * margin,
                       c_hi - c_lo + 2 * margin)
            fy, fx = dy - iy, dx - ix
            for k in range(src.shape[0]):
                moved = shift_image(src[k], fy, fx, order=order)
                out[k, r_lo - row0:r_hi - row0, c_lo - col0:c_hi - col0] = \
                    moved[margin:margin + r_hi - r_lo, margin:margin + c_hi - c_lo]
    return out
