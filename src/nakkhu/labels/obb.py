"""Step 5: polygon -> oriented box, keep/ignore/drop filters, ignore masks, YOLO-OBB files.

Per instance, in tile pixel coordinates (after registration):

1. area < ``min_area_m2`` -> drop (applies to keep and ignore instances alike);
2. less than ``min_inside_frac`` of the polygon inside the valid tile region -> ignore;
3. rectangularity (polygon area / minimum-rotated-rectangle area) < ``min_rectangularity``
   -> ignore;
4. otherwise the label is the minimum rotated rectangle of the polygon's part inside the tile,
   with corners clipped to the tile (Ultralytics rejects coordinates outside [0, 1]).

The ignore region of a tile is the union of its ignore polygons (clipped to the tile) minus the
kept boxes; it is filled with the tile mean colour when the training set is materialised
(Ultralytics has no ignore loss). A tile whose ignore area exceeds ``max_ignore_frac`` of its
building area (kept boxes plus ignore region) is not admitted for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from PIL import Image
from rasterio.features import rasterize

KEEP, IGNORE, DROP = "keep", "ignore", "drop"


@dataclass(frozen=True)
class ObbFilters:
    pixel_size_m: float = 0.5
    min_area_m2: float = 9.0
    min_inside_frac: float = 0.5
    min_rectangularity: float = 0.6
    max_ignore_frac: float = 0.15

    @classmethod
    def from_config(cls, cfg: dict[str, Any], pixel_size_m: float) -> ObbFilters:
        return cls(pixel_size_m=pixel_size_m, min_area_m2=float(cfg["min_area_m2"]),
                   min_inside_frac=float(cfg["min_inside_frac"]),
                   min_rectangularity=float(cfg["min_rectangularity"]),
                   max_ignore_frac=float(cfg["max_ignore_frac"]))


@dataclass
class TileLabels:
    """Result of :func:`tile_labels` for one tile and one arm."""

    boxes: np.ndarray                      # (n, 4, 2) pixel corners of kept boxes
    ignore: Any                            # ignore region polygon (pixel coordinates)
    decisions: list[str] = field(default_factory=list)   # per input: keep / ignore / drop
    reasons: list[str] = field(default_factory=list)
    building_area_px: float = 0.0
    ignore_area_px: float = 0.0

    @property
    def ignore_frac(self) -> float:
        return self.ignore_area_px / self.building_area_px if self.building_area_px > 0 else 0.0


def mrr_corners(geom: Any) -> np.ndarray:
    """Four corners (4, 2) of the minimum rotated rectangle, counter-clockwise in a y-down
    image frame is not guaranteed; the order is the rectangle's exterior ring order."""
    rect = shapely.minimum_rotated_rectangle(geom)
    if rect.geom_type != "Polygon":   # degenerate (line/point): pad to a zero-area box
        coords = np.asarray(rect.coords if hasattr(rect, "coords") else [], dtype=float)
        if len(coords) == 0:
            return np.zeros((4, 2))
        return np.array([coords[0], coords[-1], coords[-1], coords[0]], dtype=float)
    return np.asarray(rect.exterior.coords, dtype=float)[:4]


def rectangularity(geom: Any) -> float:
    rect = shapely.minimum_rotated_rectangle(geom)
    area = shapely.area(rect)
    return float(shapely.area(geom) / area) if area > 0 else 0.0


def tile_labels(polygons_px: np.ndarray, actions: np.ndarray, region: Any,
                tile_px: int, filters: ObbFilters) -> TileLabels:
    """Apply the filters to one tile's instances (``actions``: keep or ignore per polygon)."""
    tile_box = shapely.box(0, 0, tile_px, tile_px)
    px_area = filters.pixel_size_m ** 2
    boxes: list[np.ndarray] = []
    keep_polys: list[Any] = []
    ignore_polys: list[Any] = []
    decisions: list[str] = []
    reasons: list[str] = []
    for poly, action in zip(polygons_px, actions, strict=True):
        area = float(shapely.area(poly))
        if area * px_area < filters.min_area_m2:
            decisions.append(DROP)
            reasons.append("small")
            continue
        inside = shapely.intersection(poly, region)
        inside_frac = float(shapely.area(inside)) / area if area > 0 else 0.0
        clipped = shapely.intersection(poly, tile_box)
        if action == IGNORE:
            decisions.append(IGNORE)
            reasons.append("rule")
            ignore_polys.append(clipped)
            continue
        if inside_frac < filters.min_inside_frac:
            decisions.append(IGNORE)
            reasons.append("outside")
            ignore_polys.append(clipped)
            continue
        if rectangularity(poly) < filters.min_rectangularity:
            decisions.append(IGNORE)
            reasons.append("rectangularity")
            ignore_polys.append(clipped)
            continue
        corners = np.clip(mrr_corners(clipped if area > shapely.area(clipped) else poly),
                          0.0, float(tile_px))
        boxes.append(corners)
        keep_polys.append(shapely.Polygon(corners))
        decisions.append(KEEP)
        reasons.append("ok")
    kept = shapely.union_all(keep_polys) if keep_polys else shapely.Polygon()
    ignore = shapely.union_all(ignore_polys) if ignore_polys else shapely.Polygon()
    ignore = shapely.difference(ignore, kept) if not ignore.is_empty else ignore
    ignore = shapely.intersection(ignore, tile_box)
    ignore_area = float(shapely.area(ignore))
    building = float(shapely.area(kept)) + ignore_area
    return TileLabels(boxes=np.array(boxes, dtype=float).reshape(-1, 4, 2), ignore=ignore,
                      decisions=decisions, reasons=reasons, building_area_px=building,
                      ignore_area_px=ignore_area)


def write_yolo_obb(path: str | Path, boxes: np.ndarray, tile_px: int, cls: int = 0) -> None:
    """Ultralytics OBB format: ``cls x1 y1 x2 y2 x3 y3 x4 y4`` normalised to [0, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for corners in np.asarray(boxes, dtype=float).reshape(-1, 4, 2):
        values = (corners / float(tile_px)).reshape(-1)
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_obb(path: str | Path, tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Classes (n,) and pixel corners (n, 4, 2) from a YOLO-OBB label file."""
    text = Path(path).read_text(encoding="utf-8").split()
    values = np.array(text, dtype=float).reshape(-1, 9) if text else np.zeros((0, 9))
    return values[:, 0].astype(int), values[:, 1:].reshape(-1, 4, 2) * float(tile_px)


def ignore_mask(region: Any, tile_px: int) -> np.ndarray:
    if region is None or region.is_empty:
        return np.zeros((tile_px, tile_px), dtype=bool)
    return rasterize([(region, 1)], out_shape=(tile_px, tile_px), fill=0,
                     dtype="uint8").astype(bool)


def write_ignore_png(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(bool)).save(path, "PNG")


def fill_ignore(rgb: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Fill ignore pixels with the mean colour of the tile's valid pixels (plan 007 uses this
    when it materialises an arm's training images)."""
    out = rgb.copy()
    if not mask.any():
        return out
    base = valid if valid is not None else np.ones(mask.shape, dtype=bool)
    mean = rgb[base].reshape(-1, rgb.shape[-1]).mean(axis=0) if base.any() else 0
    out[mask] = np.round(mean).astype(rgb.dtype)
    return out
