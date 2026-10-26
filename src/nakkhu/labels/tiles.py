"""Tile access for label building: index, image, validity mask, map <-> pixel polygons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from PIL import Image

from nakkhu.config import Project

TRAIN_SCENE = "maxar_wv02_20240113"
TRAIN_PURPOSE = "train"


def tile_dir(project: Project, scene: str = TRAIN_SCENE, purpose: str = TRAIN_PURPOSE) -> Path:
    return project.paths.data_root / "interim" / "tiles" / scene / purpose


def load_index(directory: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(Path(directory) / "index.parquet")
    return frame.sort_values("tile_id").reset_index(drop=True)


def has_mask(row: Any) -> bool:
    mask = row["mask"] if "mask" in row else None
    return isinstance(mask, str) and bool(mask)


def read_tile(directory: str | Path, row: Any) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and boolean validity mask (all valid when no mask file exists)."""
    directory = Path(directory)
    rgb = np.asarray(Image.open(directory / row["image"]).convert("RGB"))
    if has_mask(row):
        valid = np.asarray(Image.open(directory / row["mask"]).convert("L")) > 0
    else:
        valid = np.ones(rgb.shape[:2], dtype=bool)
    return rgb, valid


def to_px(geoms: np.ndarray, affine: Any) -> np.ndarray:
    """Map polygons -> tile pixel coordinates (x right, y down; north-up affine only)."""
    a, _, c, _, e, f = (float(v) for v in list(affine)[:6])
    return shapely.transform(np.asarray(geoms, dtype=object),
                             lambda xy: np.column_stack(((xy[:, 0] - c) / a,
                                                         (xy[:, 1] - f) / e)))


def to_map(geoms: np.ndarray, affine: Any) -> np.ndarray:
    a, _, c, _, e, f = (float(v) for v in list(affine)[:6])
    return shapely.transform(np.asarray(geoms, dtype=object),
                             lambda xy: np.column_stack((xy[:, 0] * a + c, xy[:, 1] * e + f)))


def valid_region(valid: np.ndarray) -> Any:
    """Polygon of the valid pixels of a tile (pixel coordinates)."""
    if valid.all():
        return shapely.box(0, 0, valid.shape[1], valid.shape[0])
    from rasterio.features import shapes

    parts = [shapely.geometry.shape(geom) for geom, value in
             shapes(valid.astype("uint8"), mask=valid) if value == 1]
    return shapely.union_all(parts) if parts else shapely.Polygon()
