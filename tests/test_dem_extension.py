"""DEM extension (2026-10-17): credentials, OpenTopography / NSIDC fetchers, vertical datum
harmonisation, void fill, derived sweep configs and the label-free DEM diagnostics."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import requests
import yaml
from affine import Affine
from rasterio.transform import from_origin

from nakkhu.acquire import credentials, fetchers, geo, load_sources
from nakkhu.config import load_project
from nakkhu.hydro import dem as demmod
from nakkhu.hydro import dem_diagnostics as diag
from nakkhu.hydro import route
from nakkhu.hydro.dem import GridSpec
from nakkhu.hydro.sweep import SweepError, extend_config, load_settings

REAL_ROOT = Path(__file__).resolve().parents[1]
SECRET = "s3cr3t-key-value-0123456789"


def small_grid(height: int = 5, width: int = 6, res: float = 30.0) -> GridSpec:
    return GridSpec(crs="EPSG:32645", transform=Affine(res, 0, 330000, 0, -res, 3060000),
                    width=width, height=height)


# --- credentials ----------------------------------------------------------------------------


def test_secret_from_env_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "secrets.env"
    path.write_text("# comment\nOPENTOPOGRAPHY_API_KEY=\"from-file\"\nOTHER=x\n", encoding="utf-8")
    monkeypatch.setenv(credentials.SECRETS_ENV, str(path))
    monkeypatch.delenv("OPENTOPOGRAPHY_API_KEY", raising=False)
    assert credentials.secret("OPENTOPOGRAPHY_API_KEY") == "from-file"
    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "from-env")
    assert credentials.secret("OPENTOPOGRAPHY_API_KEY") == "from-env"
    with pytest.raises(credentials.MissingCredential) as error:
        credentials.secret("EARTHDATA_TOKEN_MISSING")
    assert "from-file" not in str(error.value)


def test_redact_masks_values_and_credential_query_params():
    text = (f"GET https://h/API/globaldem?demtype=X&API_Key={SECRET}&south=1 failed; "
            f"https://d.cloudfront.net/x.tif?Expires=1&Signature=abc&Key-Pair-Id=K token {SECRET}")
    out = credentials.redact(text, SECRET)
    assert SECRET not in out and "Signature=abc" not in out and "Key-Pair-Id=K" not in out
    assert "API_Key=<redacted>" in out and "demtype=X" in out and "south=1" in out


# --- OpenTopography fetcher -----------------------------------------------------------------


def _geotiff_bytes() -> bytes:
    data = np.arange(12, dtype="int16").reshape(3, 4) + 1000
    buffer = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(driver="GTiff", width=4, height=3, count=1, dtype="int16",
                          crs="EPSG:4326", transform=from_origin(85.0, 28.0, 0.25, 0.25),
                          nodata=-32768) as dst:
            dst.write(data, 1)
        buffer.write(memfile.read())
    return buffer.getvalue()


class _Response:
    def __init__(self, status: int, content: bytes, text: str = ""):
        self.status_code, self.content, self.text = status, content, text
        self.headers = {"Content-Type": "application/octet-stream"}

    def iter_content(self, size: int):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Session:
    def __init__(self, response: _Response):
        self.response, self.calls = response, []

    def get(self, url, params=None, **kwargs):
        self.calls.append((url, dict(params or {})))
        return self.response


def _ot_context(tmp_path: Path, session: _Session, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", SECRET)
    source = {"id": "dem_test", "params": {"demtype": "SRTMGL1", "pad_deg": 0.01}}
    ctx = fetchers.Context(project=None, source=source, sources={}, staging=tmp_path,  # type: ignore[arg-type]
                           cache=tmp_path / "cache", s=session)  # type: ignore[arg-type]
    monkeypatch.setattr(fetchers, "opentopo_bounds", lambda ctx: {
        "south": 27.5, "north": 27.8, "west": 85.1, "east": 85.5})
    monkeypatch.setattr(fetchers, "_dem_clip", lambda ctx, src, name: ctx.notes.append("clip"))
    return ctx


def test_opentopo_fetch_records_redacted_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = _Session(_Response(200, _geotiff_bytes()))
    ctx = _ot_context(tmp_path, session, monkeypatch)
    fetchers.fetch_opentopo(ctx)
    assert session.calls[0][1]["API_Key"] == SECRET  # sent to OpenTopography only
    record = ctx.prov["files"][0]
    assert record["url"].endswith("API_Key=<redacted>") and "demtype=SRTMGL1" in record["url"]
    assert record["sha256"] == hashlib.sha256(_geotiff_bytes()).hexdigest()
    assert record["raster"]["crs"] == "EPSG:4326"
    assert SECRET not in json.dumps(ctx.prov) + json.dumps(ctx.notes)


def test_opentopo_error_message_is_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    session = _Session(_Response(401, b"", text=f"Invalid API_Key={SECRET}"))
    ctx = _ot_context(tmp_path, session, monkeypatch)
    with pytest.raises(fetchers.FetchError) as error:
        fetchers.fetch_opentopo(ctx)
    assert SECRET not in str(error.value) and "HTTP 401" in str(error.value)


def test_earthdata_token_only_sent_to_earthdata_hosts():
    seen = []

    def response(status: int) -> requests.Response:
        r = requests.Response()
        r.status_code, r._content, r._content_consumed = status, b"", True
        return r

    class Redirecting:
        def get(self, url, headers=None, **kwargs):
            seen.append((url, dict(headers or {})))
            if "earthdatacloud" in url:
                r = response(303)
                r.headers["Location"] = "https://x.cloudfront.net/f.tif?Sig=1"
                return r
            return response(206)

    final = fetchers.earthdata_signed_url(
        "https://data.nsidc.earthdatacloud.nasa.gov/p/f.tif", SECRET, Redirecting())  # type: ignore[arg-type]
    assert final.startswith("https://x.cloudfront.net/")
    assert seen[0][1]["Authorization"] == f"Bearer {SECRET}"
    assert "Authorization" not in seen[1][1]


def test_real_sources_register_new_dems_and_geoids():
    sources = load_sources(load_project(REAL_ROOT))
    for sid, method in (("dem_nasadem", "opentopo_globaldem"), ("dem_srtmgl1", "opentopo_globaldem"),
                        ("dem_aw3d30", "opentopo_globaldem"), ("dem_hma8m_mos", "nsidc_cmr_clip"),
                        ("geoid_egm96", "cog_clip"), ("geoid_egm2008", "cog_clip")):
        assert sources[sid]["method"] == method
    text = (REAL_ROOT / "configs" / "data" / "sources.yaml").read_text(encoding="utf-8")
    assert "API_Key=" not in text.replace("API_Key=<redacted>", "")


# --- merge of granule clips -----------------------------------------------------------------


def _tile(path: Path, x0: float, values: np.ndarray) -> Path:
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype="float32", crs="EPSG:32645",
                       transform=from_origin(x0, 3000080.0, 8.0, 8.0), nodata=-9999.0) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def test_merge_first_valid_adjacent_tiles(tmp_path: Path):
    a = np.full((4, 3), 100.0)
    a[0, 0] = -9999.0
    b = np.full((4, 3), 200.0)
    left = _tile(tmp_path / "a.tif", 300000.0, a)
    right = _tile(tmp_path / "b.tif", 300024.0, b)
    info = geo.merge_first_valid([left, right], tmp_path / "m.tif")
    with rasterio.open(tmp_path / "m.tif") as src:
        merged = src.read(1)
        assert src.nodata == -9999.0
    assert merged.shape == (4, 6)
    assert merged[0, 0] == -9999.0 and merged[1, 1] == 100.0 and merged[1, 4] == 200.0
    assert info["cells_valid"] == 23 and info["cells_valid_in_several"] == 0


# --- vertical datum and voids ---------------------------------------------------------------


def _geoid(path: Path, value: float) -> Path:
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=1, dtype="float32",
                       crs="EPSG:4979", transform=from_origin(84.0, 29.0, 0.25, 0.25)) as dst:
        dst.write(np.full((8, 8), value, dtype="float32"), 1)
    return path


def test_geoid_shift_signs(tmp_path: Path):
    grid = small_grid()
    geoids = {"egm96": _geoid(tmp_path / "n96.tif", -51.0),
              "egm2008": _geoid(tmp_path / "n08.tif", -49.0)}
    # H08 = H96 + N96 - N08 = H96 - 2; H08 = h - N08 = h + 49
    assert np.allclose(demmod.geoid_shift(grid, "egm96", "egm2008", geoids), -2.0)
    assert np.allclose(demmod.geoid_shift(grid, "ellipsoid", "egm2008", geoids), 49.0)
    assert demmod.geoid_shift(grid, "egm2008", "egm2008", geoids) is None
    with pytest.raises(ValueError):
        demmod.geoid_shift(grid, "navd88", "egm2008", geoids)


def test_real_geoid_grids_match_proj_if_fetched():
    """Recorded check: EGM96 -> EGM2008 at the Nakkhu is -1.927 m and ellipsoid -> EGM2008
    +49.268 m (pyproj with the PROJ-data network grids gave the same values on 2026-10-17)."""
    paths = {"egm96": REAL_ROOT / "data/raw/geoid_egm96/geoid_egm96_routing2km.tif",
             "egm2008": REAL_ROOT / "data/raw/geoid_egm2008/geoid_egm2008_routing2km.tif"}
    if not all(p.is_file() for p in paths.values()):
        pytest.skip("geoid grids not fetched")
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True).transform(
        85.3108, 27.66)
    grid = GridSpec("EPSG:32645", Affine(30, 0, x - 15, 0, -30, y + 15), 1, 1)
    assert demmod.geoid_shift(grid, "egm96", "egm2008", paths)[0, 0] == pytest.approx(-1.927,
                                                                                      abs=2e-3)
    assert demmod.geoid_shift(grid, "ellipsoid", "egm2008", paths)[0, 0] == pytest.approx(
        49.268, abs=2e-3)


def test_fill_voids_uses_donor_only_on_nan():
    dem = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    donor = np.array([[9.0, 8.0], [np.nan, 6.0]], dtype=np.float32)
    filled, mask = demmod.fill_voids(dem, donor)
    assert filled[0, 1] == 8.0 and filled[0, 0] == 1.0 and filled[1, 1] == 4.0
    assert np.isnan(filled[1, 0]) and mask.tolist() == [[False, True], [False, False]]


# --- derived sweep configs ------------------------------------------------------------------


def test_extend_config_merges_sections_and_selects():
    base = {"dems": {"a": {"p": 1}, "b": {"p": 2}}, "conditioning": {"x": {}, "y": {}},
            "inventories": {"i": {}}, "grid": {"res_m": 30}, "statistics": ["s1", "s2"]}
    ext = {"extends": "base.yaml", "dems": {"b": {"p": 3}, "c": {"p": 4}},
           "statistics": ["s1"], "only": {"dems": ["c", "a"], "conditioning": ["y"]}}
    merged = extend_config(base, ext)
    assert list(merged["dems"]) == ["c", "a"] and merged["dems"]["c"] == {"p": 4}
    assert list(merged["conditioning"]) == ["y"] and merged["statistics"] == ["s1"]
    assert merged["grid"] == {"res_m": 30} and merged["extends"] == "base.yaml"
    assert base["dems"]["b"] == {"p": 2}  # base untouched
    with pytest.raises(SweepError):
        extend_config(base, {"only": {"dems": ["zz"]}})
    with pytest.raises(SweepError):
        extend_config(base, {"only": {"grid": ["res_m"]}})


def test_real_sweep_dems_config_keeps_pinned_base():
    """sweep_dems.yaml extends the canonical sweep without editing it: the base file still has
    the SHA-256 pinned by prereg_v9_a1 (common.hand_grid)."""
    project = load_project(REAL_ROOT)
    settings, surface = load_settings(project, "configs/hand/sweep_dems.yaml")
    assert surface is None
    assert list(settings.v["dems"]) == ["glo30", "fabdem", "nasadem", "srtmgl1", "aw3d30",
                                        "hma8m"]
    assert list(settings.conditionings()) == ["fill_burn", "breach_burn"]
    assert settings.stage == "hand-dems" and settings.attribute_stage == "hand-dems-attribute"
    assert settings.v["dems"]["srtmgl1"]["vertical"] == "egm96"
    assert settings.v["dems"]["hma8m"]["vertical"] == "ellipsoid"
    assert settings.v["thresholds_km2"] == [0.05, 0.1, 0.25, 0.5, 1, 2]
    assert settings.v["burn"]["depth_m"] == 60
    addendum = yaml.safe_load((REAL_ROOT / "configs/prereg_addenda/prereg_v9_a1.yaml")
                              .read_text(encoding="utf-8"))
    pinned = json.dumps(addendum)
    digest = hashlib.sha256((REAL_ROOT / "configs/hand/sweep.yaml").read_bytes()).hexdigest()
    assert digest in pinned


# --- diagnostics ----------------------------------------------------------------------------


def test_diff_stats_quartiles():
    out = diag.diff_stats(np.array([1.0, 2.0, 3.0, 4.0, np.nan]), np.zeros(5), (2.5,))
    assert out["n"] == 4 and out["median_m"] == 2.5 and out["iqr_m"] == pytest.approx(1.5)
    assert out["share_abs_gt_2.5m"] == 0.5


def test_first_stream_cell_and_drain_main():
    # 1 x 6 strip draining east (code 1); stream cells at columns 2 and 5; main mask at col 5.
    grid = GridSpec("EPSG:32645", Affine(30, 0, 0, 0, -30, 30), 6, 1)
    fdir = np.ones((1, 6), dtype=np.int64)
    fdir[0, 5] = 0  # exits
    ds = route.downstream_index(fdir)
    streams = np.zeros(6, dtype=bool)
    streams[[2, 5]] = True
    first = diag.first_stream_cell(ds, streams, np.array([0, 3, 2, -1]))
    assert first.tolist() == [2, 5, 2, -1]
    no_stream = diag.first_stream_cell(ds, np.zeros(6, dtype=bool), np.array([0]))
    assert no_stream.tolist() == [-1]
    acc = np.arange(1, 7, dtype=np.float64).reshape(1, 6)  # cells drained
    main = np.zeros((1, 6), dtype=bool)
    main[0, 5] = True
    # threshold = 3 cells of 900 m2: streams are columns 2..5 -> col 0 drains to col 2 (trib.)
    t = 3 * 900 / 1e6
    out = diag.drain_main_summary(ds, acc, main, np.array([0, 4, 5]), grid, t)
    assert out["n_buildings"] == 3 and out["n_drains_main"] == 1
    assert out["n_drains_tributary"] == 2 and out["share_drains_main"] == pytest.approx(1 / 3)


def test_main_mask_needs_line_and_area():
    import geopandas as gpd
    from shapely.geometry import LineString

    grid = GridSpec("EPSG:32645", Affine(30, 0, 0, 0, -30, 90), 3, 3)
    lines = gpd.GeoDataFrame({"name": ["m"]}, geometry=[LineString([(5, 75), (85, 75)])],
                             crs="EPSG:32645")
    acc = np.array([[1, 2000, 2000], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
    mask = diag.main_mask(lines, grid, acc, 1.0)  # 1 km2 = 1111 cells of 900 m2
    assert mask.tolist() == [[False, True, True], [False] * 3, [False] * 3]


def test_diagnostics_module_reads_no_labels():
    text = (REAL_ROOT / "src" / "nakkhu" / "hydro" / "dem_diagnostics.py").read_text(
        encoding="utf-8")
    for forbidden in ("flood.validate", "change_map", "responses", "audit", "test_journal",
                      "roc_auc", "record_test_access"):
        assert forbidden not in text.split('"""', 2)[2], forbidden


def test_real_native_hma_config_chains_extends():
    settings, _ = load_settings(load_project(REAL_ROOT),
                                "configs/hand/sweep_dems_hma8m_native.yaml")
    assert settings.v["grid"]["res_m"] == 8 and list(settings.v["dems"]) == ["glo30", "hma8m"]
    assert settings.v["dems"]["hma8m"]["void_fill"] == "glo30"  # from sweep_dems.yaml
    assert settings.v["geoids"]["egm96"]["manifest_id"] == "geoid_egm96"
    assert settings.v["primary"]["threshold_km2"] is None
    assert settings.stage == "hand-dems-8m" and settings.grid().res == 8.0


def test_canonical_sweep_skips_dems_not_in_its_vertical_datum():
    """The canonical sweep has no `vertical` conversions: an EGM96 DEM fetched later must be
    skipped there (not mixed into the EGM2008 grid), while sweep_dems.yaml converts it."""
    project = load_project(REAL_ROOT)
    canonical, _ = load_settings(project, "configs/hand/sweep.yaml")
    derived, _ = load_settings(project, "configs/hand/sweep_dems.yaml")
    for name in ("glo30", "fabdem", "gedtm30"):
        assert canonical.datum_problem(canonical.v["dems"][name]) == ""
    for name in ("nasadem", "srtmgl1", "aw3d30"):
        assert "not the sweep target egm2008" in canonical.datum_problem(canonical.v["dems"][name])
        if canonical.resolve(canonical.v["dems"][name]["path"]).is_file():
            status, reason = canonical.dem_status(name)
            assert status == "skipped" and "vertical" in reason
        assert derived.datum_problem(derived.v["dems"][name]) == ""
    assert derived.datum_problem(derived.v["dems"]["hma8m"]) == ""
    assert canonical.datum_problem({"path": "x.tif"}) == ""        # undocumented: target assumed
