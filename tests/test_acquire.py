from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box

from nakkhu.acquire import geo, load_sources
from nakkhu.acquire.fetchers import _tile_bounds
from nakkhu.cli import main
from nakkhu.config import load_project
from nakkhu.manifest import (
    ACCESS_VALUES,
    CLIP_VALUES,
    load_manifest,
    save_manifest,
    validate_entry,
)

REAL_ROOT = Path(__file__).resolve().parents[1]


def _pending(**overrides) -> dict:
    entry = {
        "id": "dem_x", "title": "X", "url": "https://example.org", "access": "registration",
        "licence": "public domain", "attribution": "X", "citation": "X", "bytes": None,
        "sha256": None, "retrieved_utc": None, "local_path": None, "clip": "routing",
        "notes": "pending: needs user account", "status": "pending",
    }
    entry.update(overrides)
    return entry


def test_pending_entry_is_valid_without_measurements():
    assert validate_entry(_pending()) == []


def test_pending_entry_needs_reason_and_empty_measurements():
    assert any("why" in p for p in validate_entry(_pending(notes="")))
    assert any("leave" in p for p in validate_entry(_pending(bytes=10)))
    assert any("status" in p for p in validate_entry(_pending(status="maybe")))


def test_clip_none_is_allowed():
    assert "none" in CLIP_VALUES


def test_save_manifest_keeps_header_and_sorts(tmp_path: Path):
    path = tmp_path / "manifest.yaml"
    path.write_text("# header line\nschema_version: 1\nentries: []\n", encoding="utf-8")
    save_manifest(path, [_pending(id="b"), _pending(id="a")])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# header line\n")
    assert [e["id"] for e in load_manifest(path)] == ["a", "b"]


def test_real_source_register_is_complete():
    sources = load_sources(load_project(REAL_ROOT))
    ids = [key for key in sources if not key.startswith("_")]
    assert len(ids) == len(set(ids))
    for sid in ids:
        source = sources[sid]
        for name in ("title", "url", "access", "licence", "attribution", "citation", "clip",
                     "redistributable"):
            assert str(source.get(name, "")).strip(), f"{sid}: empty {name}"
        assert source["access"] in ACCESS_VALUES, sid
        assert source["clip"] in CLIP_VALUES, sid


def test_quadkey_tile_bounds():
    assert _tile_bounds("0") == pytest.approx((-180.0, 0.0, 0.0, 85.0511287798), abs=1e-6)
    west, south, east, north = _tile_bounds("123131223")
    assert west < 85.3 < east and south < 27.6 < north


def _tile(path: Path, x0: float, y0: float, value: int) -> Path:
    data = np.full((3, 20, 30), value, dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=30, height=20, count=3, dtype="uint8",
                       crs="EPSG:32645", transform=from_origin(x0, y0, 1.0, 1.0)) as dst:
        dst.write(data)
        dst.write_mask(np.full((20, 30), 255, dtype="uint8"))
    return path


def test_vrt_mosaics_adjacent_tiles(tmp_path: Path):
    a = _tile(tmp_path / "a.tif", 1000.0, 2000.0, 10)
    b = _tile(tmp_path / "b.tif", 1030.0, 1980.0, 20)  # east and south of a
    vrt = tmp_path / "m.vrt"
    geo.write_vrt(vrt, [a, b])
    with rasterio.open(vrt) as src:
        assert (src.width, src.height) == (60, 40)
        assert src.bounds == (1000.0, 1960.0, 1060.0, 2000.0)
        data = src.read(1)
        assert data[0, 0] == 10 and data[-1, -1] == 20 and data[-1, 0] == 0


def test_clip_raster_masks_outside_polygon(tmp_path: Path):
    src = _tile(tmp_path / "a.tif", 1000.0, 2000.0, 50)
    dest = tmp_path / "clip.tif"
    info = geo.clip_raster(str(src), dest, box(1005, 1985, 1015, 1995), mask_polygon=True)
    assert info is not None and info["valid_pixels"] == 100
    with rasterio.open(dest) as clip:
        assert (clip.width, clip.height) == (10, 10)
        assert clip.read(1).min() == 50
    assert geo.clip_raster(str(src), tmp_path / "none.tif", box(0, 0, 10, 10)) is None


def test_fetch_pending_records_entry(root: Path, capsys):
    register = {"schema_version": 1, "sources": [{
        "id": "dem_x", "method": "pending", "title": "X", "url": "https://example.org",
        "access": "registration", "licence": "public domain", "attribution": "X",
        "citation": "X", "clip": "routing", "redistributable": "yes",
        "notes": "pending: needs user account"}]}
    (root / "configs" / "data" / "sources.yaml").write_text(yaml.safe_dump(register),
                                                           encoding="utf-8")
    assert main(["--root", str(root), "data", "fetch", "dem_x", "--dry-run"]) == 0
    assert main(["--root", str(root), "data", "fetch", "dem_x"]) == 0
    assert main(["--root", str(root), "data", "verify"]) == 0
    out = capsys.readouterr().out
    assert "PEND  dem_x" in out and "1 entries, 0 failed, 1 pending" in out
    assert main(["--root", str(root), "data", "fetch", "nope"]) == 2


def test_footprint_clip_defaults_to_training_and_follows_clip():
    from types import SimpleNamespace

    from nakkhu.acquire.fetchers import _footprint_clip

    project = load_project(REAL_ROOT)
    name, geom = _footprint_clip(SimpleNamespace(project=project, source={"clip": "training"}))
    assert name == "the training domain" and geom.equals(geo.training(project))
    assert _footprint_clip(SimpleNamespace(project=project, source={}))[1].equals(geom)
    name, routing = _footprint_clip(SimpleNamespace(project=project, source={"clip": "routing"}))
    assert name == "the `routing` clip" and routing.equals(geo.clip_geometry(project, "routing"))
    assert routing.area > geom.area
