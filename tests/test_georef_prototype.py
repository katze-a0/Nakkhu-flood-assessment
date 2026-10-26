"""Regression against the prototype's on-disk georeferencing (plan 003, Step 5).

Reads ``datasets/Nakkhu_Building_Analytics_Paper01/`` read-only. The prototype mosaic affine is
fitted on these same rows (``.tmp/prototype-inventory.md`` 1.1), so this test is circular for
accuracy: it checks the pixel convention (no half-pixel shift), the tile-offset parsing and the
axis signs only. Do not report it as georeferencing accuracy.

The prototype pixel was not exactly 0.5 m: a least-squares fit over all rows gives about
0.5000011 m (E) and 0.5000048 m (N), so a 0.5 m affine misses by up to 0.14 m at the far edge of
the 28,800 px mosaic. The pixel sizes are therefore fitted here; the origin must still come out
as the plan's (330323.102, 3061574.731) to 1 mm, which a half-pixel shift (0.25 m) would break.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import Affine

from nakkhu.georef import pixel_to_map

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "Nakkhu_Building_Analytics_Paper01"
ORIGIN = (330323.102, 3061574.731)
TILE = re.compile(r"tile_x(\d{5})_y(\d{5})")
SAMPLE, SEED, TOLERANCE_M = 200, 0, 0.0015

pytestmark = pytest.mark.skipif(not DATASET.is_dir(), reason="prototype datasets/ absent")


def _pairs() -> pd.DataFrame:
    frames = []
    for folder in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        pixel = folder / f"{folder.name}_analytics.csv"
        geo = folder / f"{folder.name}_geo_locations.csv"
        if not pixel.is_file() or not geo.is_file():
            continue
        px = pd.read_csv(pixel)
        gx = pd.read_csv(geo)
        assert len(px) == len(gx), folder.name
        match = TILE.fullmatch(folder.name)
        assert match, folder.name
        frame = px[["confidence", "center_x", "center_y", "area_px"]].copy()
        frame[["geo_confidence", "easting", "northing", "geo_area_px"]] = gx[
            ["confidence", "easting_utm_x", "northing_utm_y", "area_px"]].to_numpy()
        frame["tile"] = folder.name
        frame["tile_x"], frame["tile_y"] = int(match.group(1)), int(match.group(2))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.c_[np.ones(len(x)), x]
    (offset, scale), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(offset), float(scale)


def test_port_reproduces_prototype_utm_coordinates():
    rows = _pairs()
    c, a = _fit((rows["tile_x"] + rows["center_x"]).to_numpy(), rows["easting"].to_numpy())
    f, e = _fit((rows["tile_y"] + rows["center_y"]).to_numpy(), rows["northing"].to_numpy())
    assert abs(c - ORIGIN[0]) < 1e-3 and abs(f - ORIGIN[1]) < 1e-3
    assert abs(a - 0.5) < 1e-5 and abs(e + 0.5) < 1e-5
    mosaic = Affine(a, 0.0, c, 0.0, e, f)
    rng = np.random.default_rng(SEED)
    sample = rows.iloc[np.sort(rng.choice(len(rows), size=SAMPLE, replace=False))]
    assert sample["tile"].nunique() > 50  # spread across tiles
    # pairing: the two CSVs of a tile agree row by row
    assert (sample["confidence"] == sample["geo_confidence"]).all()
    assert (sample["area_px"] == sample["geo_area_px"]).all()
    errors = []
    for _, row in sample.iterrows():
        tile_affine = mosaic @ Affine.translation(row["tile_x"], row["tile_y"])
        east, north = pixel_to_map(tile_affine, np.array([row["center_x"], row["center_y"]]))
        errors.append(max(abs(east - row["easting"]), abs(north - row["northing"])))
    assert max(errors) <= TOLERANCE_M
