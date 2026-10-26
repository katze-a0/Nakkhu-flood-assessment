import numpy as np

from nakkhu import radiometry


def prototype_normalize_band(band):
    """Percentile clip then stretch to 0-255."""
    # Copy of archive/code/tif_to_jpg.ipynb cell 5 (the reference for the per_tile mode).
    p2 = np.percentile(band, 2)
    p98 = np.percentile(band, 98)
    band = np.clip(band, p2, p98)
    if p98 - p2 > 0:
        band = (band - p2) / (p98 - p2) * 255.0
    else:
        band = np.zeros_like(band)
    return band.astype("uint8")


def prototype_tile(rgb: np.ndarray, nodata: int | None) -> np.ndarray:
    """The prototype's per-tile loop body: float32 bands, stretch, nodata to black."""
    r, g, b = (rgb[i].astype("float32") for i in range(3))
    mask = (r == nodata) | (g == nodata) | (b == nodata) if nodata is not None else None
    r, g, b = prototype_normalize_band(r), prototype_normalize_band(g), prototype_normalize_band(b)
    if mask is not None:
        r[mask] = 0
        g[mask] = 0
        b[mask] = 0
    return np.stack([r, g, b], axis=-1)


def synthetic_scene(seed: int = 0, shape: tuple[int, int, int] = (3, 200, 300)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = rng.gamma(2.0, 30.0, size=shape).clip(1, 255).astype("uint8")
    scene[:, :20, :35] = 0  # a nodata corner
    return scene


def test_per_tile_matches_prototype_byte_for_byte():
    scene = synthetic_scene()
    valid = ~(scene == 0).any(axis=0)
    ours = radiometry.per_tile_stretch(scene, valid)
    ref = prototype_tile(scene, nodata=0)
    assert ours.dtype == ref.dtype == np.uint8
    assert ours.tobytes() == ref.tobytes()
    band = scene[1].astype("float32")
    assert radiometry.normalize_band(band).tobytes() == prototype_normalize_band(band).tobytes()


def test_constant_band_goes_to_zero_like_the_prototype():
    band = np.full((8, 8), 7.0, dtype="float32")
    assert radiometry.normalize_band(band).tobytes() == prototype_normalize_band(band).tobytes()


def test_histogram_percentiles_equal_numpy():
    rng = np.random.default_rng(1)
    for size in (1, 2, 3, 17, 1000, 12345):
        values = rng.integers(0, 256, size=size)
        hist = np.bincount(values, minlength=256)
        for q in (0.0, 2.0, 50.0, 98.0, 100.0):
            assert radiometry.percentiles_from_histogram(hist, q) == np.percentile(values, q)


def test_global_stretch_is_identical_across_tiles_of_one_scene():
    scene = synthetic_scene(seed=2)
    valid = ~(scene == 0).any(axis=0)
    hist = [np.bincount(scene[i][valid], minlength=256) for i in range(3)]
    pct = radiometry.scene_percentiles(hist)
    # two overlapping tiles cut from the scene; overlap = columns 100..199
    a = (slice(None), slice(0, 200), slice(0, 200))
    b = (slice(None), slice(0, 200), slice(100, 300))
    out_a = radiometry.global_stretch(scene[a], valid[a[1:]], pct)
    out_b = radiometry.global_stretch(scene[b], valid[b[1:]], pct)
    assert out_a[:, 100:200].tobytes() == out_b[:, 0:100].tobytes()
    whole = radiometry.global_stretch(scene, valid, pct)
    assert whole[:, 100:300].tobytes() == out_b.tobytes()
    # the per-tile stretch differs on the same overlap (the inconsistency global removes)
    pa = radiometry.per_tile_stretch(scene[a], valid[a[1:]])
    pb = radiometry.per_tile_stretch(scene[b], valid[b[1:]])
    assert pa[:, 100:200].tobytes() != pb[:, 0:100].tobytes()
    assert (out_a[~valid[a[1:]]] == 0).all()
