"""Per-method fetchers. Each method has ``plan`` (dry run: remote files and sizes, no writes) and
``fetch`` (fills the staging directory and records provenance in ``ctx.prov``)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from nakkhu.acquire import credentials, geo
from nakkhu.acquire.http import (
    FetchError,
    RangeFile,
    RemoteInfo,
    download,
    get_json,
    head,
    post_json,
)
from nakkhu.config import Project
from nakkhu.manifest import sha256_file


@dataclass
class Context:
    project: Project
    source: dict[str, Any]
    sources: dict[str, dict[str, Any]]
    staging: Path
    cache: Path
    s: requests.Session
    prov: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, Any]:
        return self.source.get("params") or {}

    def rel(self, path: Path) -> str:
        return path.relative_to(self.staging).as_posix()

    def add_file(self, record: dict[str, Any]) -> None:
        self.prov.setdefault("files", []).append(record)

    def add_product(self, path: Path, **info: Any) -> None:
        self.prov.setdefault("products", []).append({"path": self.rel(path), **info})


@dataclass
class PlanRow:
    url: str
    size: int | None
    note: str = ""
    partial: bool = False  # windowed / range read: only part of ``size`` is transferred


def _info_row(info: RemoteInfo, note: str = "") -> PlanRow:
    return PlanRow(info.final_url, info.content_length, note)


# --- AOI ------------------------------------------------------------------------------------

AOI_SCRIPT_DIR = "configs/aoi/scripts"


def _workspace_path(ctx: Context, rel: str) -> Path:
    return (ctx.project.paths.root / rel).resolve()


def plan_aoi_copy(ctx: Context) -> list[PlanRow]:
    src = _workspace_path(ctx, ctx.params["src"])
    return [PlanRow(str(src), src.stat().st_size if src.exists() else None, "local copy")]


def fetch_aoi_copy(ctx: Context) -> Path:
    src = _workspace_path(ctx, ctx.params["src"])
    dest = ctx.project.paths.resolve(ctx.params["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    feature = ctx.params.get("feature")
    if feature:
        document = json.loads(src.read_text(encoding="utf-8"))
        document["features"] = [f for f in document["features"]
                                if f["properties"].get("name") == feature]
        if len(document["features"]) != 1:
            raise FetchError(f"{src}: expected one feature named {feature}")
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_text(json.dumps(document), encoding="utf-8")
        tmp.replace(dest)
    else:
        shutil.copyfile(src, dest.with_name(dest.name + ".tmp"))
        dest.with_name(dest.name + ".tmp").replace(dest)
    scripts = []
    script_dir = ctx.project.paths.resolve(AOI_SCRIPT_DIR)
    for name in ctx.params.get("scripts") or []:
        script = src.parent / "scripts" / name
        script_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(script, script_dir / name)
        scripts.append(f"{AOI_SCRIPT_DIR}/{name} sha256 {sha256_file(script)}")
    ctx.notes.append(f"copied from .tmp/data/{src.name} (sha256 {sha256_file(src)})"
                     + (f", feature {feature}" if feature else ""))
    if scripts:
        ctx.notes.append("generating script: " + "; ".join(scripts))
    if dest.suffix == ".geojson":
        frame = gpd.read_file(dest)
        if frame.geom_type.iloc[0].endswith("Polygon"):
            area = frame.to_crs(geo.WORK_CRS).area.sum() / 1e6
            ctx.notes.append(f"area recomputed in EPSG:32645: {area:.2f} km2")
        else:
            ctx.notes.append(f"{len(frame)} features")
    return dest


def plan_aoi_build(ctx: Context) -> list[PlanRow]:
    return [PlanRow(f"nakkhu aoi build --only {ctx.params['product']}", 0, "local build")]


def fetch_aoi_build(ctx: Context) -> Path:
    """Rebuild one derived AOI product (``nakkhu.aoi``) and register the rebuilt file."""
    from nakkhu.aoi import BUILD_CONFIG
    from nakkhu.aoi.build import build

    product = ctx.params["product"]
    run, summary = build(ctx.project, [product])
    result = summary["products"][product]
    dest = ctx.project.paths.resolve(ctx.params["dest"])
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(run.path / result["output"], tmp)
    tmp.replace(dest)
    config = ctx.project.paths.resolve(BUILD_CONFIG)
    ctx.notes.append(f"built by `nakkhu aoi build --only {product}` (src/nakkhu/aoi/; "
                     f"{BUILD_CONFIG} sha256 {sha256_file(config)}), run {run.ref}")
    ctx.notes.append(f"previous registered file sha256 {result['registered_sha256']}: "
                     f"match={result['comparison']['match']} (details in the run summary)")
    frame = gpd.read_file(dest)
    if frame.geom_type.iloc[0].endswith("Polygon"):
        ctx.notes.append(f"area recomputed in EPSG:32645: "
                         f"{frame.to_crs(geo.WORK_CRS).area.sum() / 1e6:.2f} km2")
    else:
        ctx.notes.append(f"{len(frame)} features")
    return dest


def plan_aoi_training_domain(ctx: Context) -> list[PlanRow]:
    return [PlanRow(f"derived from {ctx.params['datamask']}", 0, "local")]


def fetch_aoi_training_domain(ctx: Context) -> Path:
    mask_dir = ctx.project.paths.data_root / "raw" / ctx.params["datamask"] / "data-mask"
    files = sorted(mask_dir.glob("*.gpkg"))
    if not files:
        raise FetchError(f"no data masks in {mask_dir}; fetch {ctx.params['datamask']} first")
    parts = [gpd.read_file(f).to_crs(geo.WORK_CRS).geometry.union_all() for f in files]
    domain = shapely.union_all(parts)
    domain = shapely.make_valid(domain).buffer(0)
    area = domain.area / 1e6
    frame = gpd.GeoDataFrame(
        {"name": ["training_domain_wv02_10300100F5AB8000"], "area_km2": [round(area, 2)],
         "definition": [f"union of {len(files)} ARD data-mask polygons of catalog "
                        "10300100F5AB8000 (WorldView-2, 2024-01-13)"]},
        geometry=[domain], crs=geo.WORK_CRS,
    ).to_crs("EPSG:4326")
    dest = ctx.project.paths.resolve(ctx.params["dest"])
    tmp = dest.with_name(dest.name + ".tmp")
    frame.to_file(tmp, driver="GeoJSON")
    tmp.replace(dest)
    ctx.notes.append(f"area {area:.2f} km2 (EPSG:32645) from {len(files)} data masks; "
                     "computed by nakkhu.acquire.fetchers.fetch_aoi_training_domain")
    return dest


# --- Maxar ----------------------------------------------------------------------------------

_items_cache: dict[str, list[dict[str, Any]]] = {}


def maxar_items(ctx: Context, catalog_id: str) -> list[dict[str, Any]]:
    """Live STAC items of one Maxar acquisition (sorted by quadkey), with absolute hrefs."""
    if catalog_id not in _items_cache:
        base = ctx.sources["_stac"]["maxar_acquisitions"]
        col_url = urljoin(base, f"{catalog_id}_collection.json")
        collection = get_json(col_url, ctx.s)
        items = []
        for link in collection["links"]:
            if link.get("rel") != "item":
                continue
            item_url = urljoin(col_url, link["href"])
            item = get_json(item_url, ctx.s)
            item["_url"] = item_url
            for asset in item["assets"].values():
                asset["href"] = urljoin(item_url, asset["href"])
            items.append(item)
        _items_cache[catalog_id] = sorted(items, key=lambda i: i["properties"]["quadkey"])
    return _items_cache[catalog_id]


def _item_bbox(item: dict[str, Any]) -> BaseGeometry:
    return box(*item["properties"]["proj:bbox"])


def _save_items(ctx: Context, items: list[dict[str, Any]]) -> None:
    out = ctx.staging / "stac"
    out.mkdir(parents=True, exist_ok=True)
    for item in items:
        q = item["properties"]["quadkey"]
        clean = {k: v for k, v in item.items() if k != "_url"}
        (out / f"{q}.json").write_text(json.dumps(clean, indent=1, sort_keys=True),
                                       encoding="utf-8")
    ctx.prov["stac_items"] = [i["_url"] for i in items]
    ctx.prov["assets_per_item"] = {
        i["properties"]["quadkey"]: sorted(i["assets"]) for i in items
    }
    keys = sorted({k for i in items for k in i["assets"]})
    ctx.notes.append(
        f"{len(items)} STAC items; assets available: {', '.join(keys)}; "
        "pan_analytic/ms_analytic not fetched; no cloud-mask asset (data-mask = valid data only)"
    )


def _clip_items(ctx: Context) -> list[dict[str, Any]]:
    items = maxar_items(ctx, ctx.params["catalog_id"])
    clip = geo.clip_geometry(ctx.project, ctx.source["clip"])
    return [i for i in items if _item_bbox(i).intersects(clip)]


def plan_maxar_scene(ctx: Context) -> list[PlanRow]:
    return [_info_row(head(i["assets"]["visual"]["href"], ctx.s), "whole COG")
            for i in maxar_items(ctx, ctx.params["catalog_id"])]


def fetch_maxar_scene(ctx: Context) -> None:
    items = maxar_items(ctx, ctx.params["catalog_id"])
    _save_items(ctx, items)
    tifs = []
    for item in items:
        q = item["properties"]["quadkey"]
        dest = ctx.staging / "visual" / f"{ctx.params['catalog_id']}_{q}-visual.tif"
        got = download(item["assets"]["visual"]["href"], dest, ctx.cache, s=ctx.s)
        ctx.add_file(got.record(ctx.rel(dest)))
        tifs.append(dest)
    vrt = ctx.staging / f"{ctx.params['catalog_id']}-visual.vrt"
    geo.write_vrt(vrt, tifs)
    ctx.add_product(vrt, kind="maxar_visual")


def plan_maxar_clip(ctx: Context) -> list[PlanRow]:
    return [_info_row(head(i["assets"]["visual"]["href"], ctx.s),
                      "downloaded to cache, clipped, cache deleted")
            for i in _clip_items(ctx)]


def fetch_maxar_clip(ctx: Context) -> None:
    items = _clip_items(ctx)
    _save_items(ctx, items)
    clip = geo.clip_geometry(ctx.project, ctx.source["clip"])
    tifs = []
    for item in items:
        q = item["properties"]["quadkey"]
        cached = ctx.cache / f"{ctx.params['catalog_id']}_{q}-visual.tif"
        got = download(item["assets"]["visual"]["href"], cached, ctx.cache, s=ctx.s)
        dest = ctx.staging / "visual" / f"{ctx.params['catalog_id']}_{q}-visual-clip.tif"
        info = geo.clip_raster(str(cached), dest, clip, mask_polygon=True)
        record = got.record(f"(not kept) {cached.name}")
        record["clipped_to"] = ctx.rel(dest) if info else None
        ctx.add_file(record)
        if info:
            tifs.append(dest)
            ctx.prov.setdefault("clips", []).append({"path": ctx.rel(dest), **info,
                                                     "source": item["assets"]["visual"]["href"]})
        cached.unlink()
    vrt = ctx.staging / f"{ctx.params['catalog_id']}-visual.vrt"
    geo.write_vrt(vrt, tifs)
    ctx.add_product(vrt, kind="maxar_visual")
    ctx.notes.append(
        f"clip = analysis extent + {geo.EXTENT_BUFFER_M:.0f} m on the source 0.305 m grid; "
        "DEFLATE (lossless w.r.t. decoded source JPEG Q75); outside the polygon masked"
    )


def plan_maxar_datamask(ctx: Context) -> list[PlanRow]:
    return [_info_row(head(i["assets"]["data-mask"]["href"], ctx.s))
            for i in maxar_items(ctx, ctx.params["catalog_id"])]


def fetch_maxar_datamask(ctx: Context) -> None:
    items = maxar_items(ctx, ctx.params["catalog_id"])
    _save_items(ctx, items)
    for item in items:
        q = item["properties"]["quadkey"]
        dest = ctx.staging / "data-mask" / f"{ctx.params['catalog_id']}_{q}-data-mask.gpkg"
        got = download(item["assets"]["data-mask"]["href"], dest, ctx.cache, s=ctx.s,
                       progress=False)
        ctx.add_file(got.record(ctx.rel(dest)))


# --- DEMs -----------------------------------------------------------------------------------


def plan_cog_clip(ctx: Context) -> list[PlanRow]:
    row = _info_row(head(ctx.params["url"], ctx.s))
    if ctx.params.get("keep_source"):
        row.note = "whole tile + clip"
    else:
        row.note, row.partial = "windowed read only", True
    return [row]


def _dem_clip(ctx: Context, source: str, name: str, *, label: str | None = None) -> None:
    """Clip ``source`` to the bbox of the routing domain + 2 km (plus ``params.pad_m``, used
    for coarse grids such as the geoid models so that bilinear sampling has neighbours)."""
    clip = geo.clip_geometry(ctx.project, "routing")
    pad = float(ctx.params.get("pad_m", 0))
    dest = ctx.staging / f"{name}_routing2km.tif"
    info = geo.clip_raster(source, dest, box(*clip.bounds).buffer(pad, join_style="mitre")
                           if pad else box(*clip.bounds))
    if info is None:
        raise FetchError(f"{label or source} does not cover the routing domain")
    if label:
        info["source"] = label
    ctx.add_product(dest, kind=ctx.params.get("product_kind", "dem"), **info)
    ctx.notes.append(f"vertical datum: {ctx.params.get('vertical_datum', 'UNVERIFIED')}; "
                     "clip = bbox of routing domain + 2 km on the native grid"
                     + (f" + {pad:g} m pad" if pad else ""))


def fetch_cog_clip(ctx: Context) -> None:
    url = ctx.params["url"]
    if ctx.params.get("keep_source"):
        dest = ctx.staging / "source" / Path(url).name
        got = download(url, dest, ctx.cache, s=ctx.s)
        ctx.add_file(got.record(ctx.rel(dest)))
        _dem_clip(ctx, str(dest), ctx.source["id"])
    else:
        info = head(url, ctx.s)
        ctx.add_file({"path": None, "url": url, "etag": info.etag,
                      "last_modified": info.last_modified,
                      "content_length": info.content_length, "read": "windowed (/vsicurl/)"})
        _dem_clip(ctx, f"/vsicurl/{url}", ctx.source["id"])


def plan_zip_member_clip(ctx: Context) -> list[PlanRow]:
    info = head(ctx.params["url"], ctx.s)
    return [PlanRow(info.final_url, info.content_length,
                    f"zip; only member {ctx.params['member']} is read via HTTP Range",
                    partial=True)]


def fetch_zip_member_clip(ctx: Context) -> None:
    url = ctx.params["url"]
    member = ctx.params["member"]
    info = head(url, ctx.s)
    if not info.accept_ranges or info.content_length is None:
        raise FetchError(f"{url}: server does not support range reads")
    remote = RangeFile(info.final_url, info.content_length, ctx.s)
    dest = ctx.staging / "source" / member
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote) as archive:
        zinfo = archive.getinfo(member)
        digest = hashlib.sha256()
        with archive.open(zinfo) as handle, dest.open("wb") as out:  # CRC-32 checked on read
            while chunk := handle.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
    ctx.add_file({
        "path": ctx.rel(dest), "url": url, "zip_etag": info.etag,
        "zip_last_modified": info.last_modified, "zip_content_length": info.content_length,
        "zip_member": member, "zip_member_crc32": f"{zinfo.CRC:08x}",
        "zip_member_compressed_bytes": zinfo.compress_size, "bytes": dest.stat().st_size,
        "sha256": digest.hexdigest(), "range_bytes_transferred": remote.transferred,
    })
    ctx.notes.append(
        f"member {member} read from the {info.content_length}-byte zip with HTTP Range "
        f"({remote.transferred} bytes transferred, CRC-32 verified); the zip itself was not "
        "downloaded, so only its ETag/size are recorded"
    )
    licence_url = ctx.params.get("licence_url")
    if licence_url:
        lic = ctx.staging / "source" / "license.txt"
        got = download(licence_url, lic, ctx.cache, s=ctx.s, progress=False)
        ctx.add_file(got.record(ctx.rel(lic)))
    _dem_clip(ctx, str(dest), ctx.source["id"])


def plan_pending(ctx: Context) -> list[PlanRow]:
    return [PlanRow(ctx.source["url"], None, ctx.source.get("notes", "pending"))]


# OpenTopography Global DEM API (plan 008 account-gated DEMs). The API key is read at run time
# (``acquire.credentials``) and masked in every recorded URL and message.
OPENTOPO_API = "https://portal.opentopography.org/API/globaldem"
OPENTOPO_KEY = "OPENTOPOGRAPHY_API_KEY"


def opentopo_bounds(ctx: Context) -> dict[str, float]:
    """South/north/west/east (degrees) of the routing domain + 2 km bbox plus ``pad_deg``,
    rounded outward to 1e-4 degrees."""
    west, south, east, north = geo.lonlat_bounds(geo.clip_geometry(ctx.project, "routing"))
    pad = float(ctx.params.get("pad_deg", 0.01))
    return {"south": math.floor((south - pad) * 1e4) / 1e4,
            "north": math.ceil((north + pad) * 1e4) / 1e4,
            "west": math.floor((west - pad) * 1e4) / 1e4,
            "east": math.ceil((east + pad) * 1e4) / 1e4}


def opentopo_query(ctx: Context) -> dict[str, Any]:
    return {"demtype": ctx.params["demtype"], **opentopo_bounds(ctx), "outputFormat": "GTiff"}


def _recorded_url(base: str, query: dict[str, Any]) -> str:
    """The request URL as recorded: the query plus ``API_Key=<redacted>``."""
    from urllib.parse import urlencode

    return f"{base}?{urlencode(query)}&API_Key={credentials.REDACTED}"


def plan_opentopo(ctx: Context) -> list[PlanRow]:
    return [PlanRow(_recorded_url(OPENTOPO_API, opentopo_query(ctx)), None,
                    "OpenTopography Global DEM API (API key, size known after the request)")]


def fetch_opentopo(ctx: Context) -> None:
    key = credentials.secret(OPENTOPO_KEY)
    query = opentopo_query(ctx)
    recorded = _recorded_url(OPENTOPO_API, query)
    dest = ctx.staging / "source" / f"{ctx.source['id']}_opentopography.tif"
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest, size, headers = hashlib.sha256(), 0, {}
    for attempt in range(4):
        try:
            with ctx.s.get(OPENTOPO_API, params={**query, "API_Key": key}, stream=True,
                           timeout=(30, 600)) as r:
                if r.status_code != 200:
                    body = credentials.redact(r.text[:300], key)
                    raise FetchError(f"OpenTopography {query['demtype']}: HTTP {r.status_code}: "
                                     f"{body}")
                headers = {k: r.headers.get(k) for k in ("Content-Type", "Content-Length",
                                                         "Content-Disposition", "Date")}
                digest, size = hashlib.sha256(), 0
                with dest.open("wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            break
        except requests.RequestException as error:
            if attempt == 3:
                raise FetchError(credentials.redact(f"OpenTopography request failed: {error}",
                                                    key)) from None
    import rasterio

    with rasterio.open(dest) as src:
        native = {"crs": str(src.crs), "res": list(src.res), "width": src.width,
                  "height": src.height, "nodata": src.nodata, "dtype": src.dtypes[0],
                  "bounds": list(src.bounds)}
    ctx.add_file({"path": ctx.rel(dest), "url": recorded, "request": query,
                  "response_headers": headers, "bytes": size, "sha256": digest.hexdigest(),
                  "raster": native})
    ctx.notes.append(f"OpenTopography Global DEM API demtype={query['demtype']}, bbox S "
                     f"{query['south']} N {query['north']} W {query['west']} E {query['east']} "
                     f"(routing domain + 2 km + {ctx.params.get('pad_deg', 0.01)} deg); "
                     f"response {size} bytes, {native['crs']}, {native['res'][0]:.10f} deg")
    _dem_clip(ctx, str(dest), ctx.source["id"])


# NASA Earthdata (NSIDC) granules found through CMR (public search) and read with an Earthdata
# bearer token. The token is attached only to Earthdata hosts; the data host answers with a
# short-lived signed URL, which is read windowed through /vsicurl/ and never recorded.
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
EARTHDATA_TOKEN = "EARTHDATA_TOKEN"
EARTHDATA_HOSTS = ("earthdata.nasa.gov", "earthdatacloud.nasa.gov")


def cmr_granules(ctx: Context) -> list[dict[str, Any]]:
    """CMR granules of ``short_name``/``version`` whose footprint intersects the routing domain +
    2 km (exact polygon test after the CMR bbox query), sorted by granule id."""
    west, south, east, north = geo.lonlat_bounds(geo.clip_geometry(ctx.project, "routing"))
    document = get_json(CMR_GRANULES, ctx.s, params={
        "short_name": ctx.params["short_name"], "version": ctx.params["version"],
        "bounding_box": f"{west},{south},{east},{north}", "page_size": 2000})
    target = shapely.box(west, south, east, north)
    out = []
    for item in document.get("items", []):
        umm = item["umm"]
        polys = []
        geometry = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        for gpoly in geometry.get("GPolygons", []):
            ring = [(p["Longitude"], p["Latitude"])
                    for p in gpoly["Boundary"]["Points"]]
            polys.append(shapely.Polygon(ring))
        footprint = shapely.union_all(polys) if polys else shapely.Polygon()
        if not footprint.intersects(target):
            continue
        urls = [u["URL"] for u in umm.get("RelatedUrls", [])
                if u.get("Type") == "GET DATA" and u["URL"].startswith("https://")]
        archive = (umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation")
                   or [{}])[0]
        out.append({"granule_ur": umm["GranuleUR"], "url": urls[0] if urls else None,
                    "size_bytes": archive.get("SizeInBytes"),
                    "checksum": archive.get("Checksum"),
                    "production": umm.get("DataGranule", {}).get("ProductionDateTime"),
                    "footprint_wkt": footprint.wkt,
                    "bbox_overlap_share": round(footprint.intersection(target).area
                                                / target.area, 6)})
    return sorted(out, key=lambda g: g["granule_ur"])


def earthdata_signed_url(url: str, token: str, s: requests.Session) -> str:
    """Follow the Earthdata redirect chain with the bearer token (Earthdata hosts only) and
    return the final, short-lived data URL (never recorded)."""
    current = url
    for _ in range(8):
        host = requests.utils.urlparse(current).netloc
        headers = {"Range": "bytes=0-0"}
        if host.endswith(EARTHDATA_HOSTS):
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = s.get(current, headers=headers, allow_redirects=False, timeout=TIMEOUT_AUTH)
        except requests.RequestException as error:
            raise FetchError(credentials.redact(f"Earthdata request failed: {error}",
                                                token)) from None
        r.close()
        if r.status_code in (301, 302, 303, 307, 308):
            current = urljoin(current, r.headers["Location"])
            continue
        if r.status_code in (200, 206):
            return current
        raise FetchError(f"Earthdata {url}: HTTP {r.status_code} (token rejected or no access)")
    raise FetchError(f"Earthdata {url}: too many redirects")


TIMEOUT_AUTH = (30, 120)


def plan_nsidc_cmr(ctx: Context) -> list[PlanRow]:
    return [PlanRow(g["url"] or g["granule_ur"], g["size_bytes"],
                    f"{g['granule_ur']}: bbox overlap {g['bbox_overlap_share']:.3f}; windowed read "
                    "only (Earthdata token)", partial=True)
            for g in cmr_granules(ctx)]


def fetch_nsidc_cmr(ctx: Context) -> None:
    """Windowed clip of every intersecting granule to the routing domain + 2 km bbox on the
    native grid, then one mosaic VRT (``<id>_routing2km.vrt``) and a merged GeoTIFF."""
    token = credentials.secret(EARTHDATA_TOKEN)
    granules = cmr_granules(ctx)
    if not granules:
        raise FetchError(f"CMR: no {ctx.params['short_name']} granule intersects the routing "
                         "domain + 2 km")
    clip = geo.clip_geometry(ctx.project, "routing")
    clips: list[Path] = []
    for g in granules:
        signed = earthdata_signed_url(g["url"], token, ctx.s)
        dest = ctx.staging / "granules" / f"{Path(g['granule_ur']).stem}_routing2km.tif"
        try:
            info = geo.clip_raster(f"/vsicurl/{signed}", dest, box(*clip.bounds))
        except Exception as error:  # noqa: BLE001 - re-raised without the signed URL
            raise FetchError(credentials.redact(str(error).replace(signed, "<signed url>"),
                                                token)) from None
        record = {"path": ctx.rel(dest) if info else None, "url": g["url"],
                  "granule_ur": g["granule_ur"], "cmr_size_bytes": g["size_bytes"],
                  "cmr_checksum": g["checksum"], "production": g["production"],
                  "footprint_wkt": g["footprint_wkt"],
                  "read": "windowed (/vsicurl/ on the signed URL, not recorded)"}
        if info:
            info["source"] = g["url"]
            record.update({"bytes": dest.stat().st_size, "sha256": sha256_file(dest),
                           "window": info["window"], "crs": info["crs"], "res": info["res"]})
            clips.append(dest)
        ctx.add_file(record)
    merged = ctx.staging / f"{ctx.source['id']}_routing2km.tif"
    info = geo.merge_first_valid(clips, merged)
    ctx.add_product(merged, kind="dem", **info, sources=[ctx.rel(c) for c in clips],
                    rule="first valid pixel in granule-id order; cells valid in several "
                         "granules and their largest difference are counted")
    ctx.notes.append(f"{len(granules)} {ctx.params['short_name']} granule(s) intersect the "
                     f"routing domain + 2 km bbox ({', '.join(g['granule_ur'] for g in granules)}"
                     "); windowed reads of the bbox on the native grid; vertical datum: "
                     f"{ctx.params.get('vertical_datum', 'UNVERIFIED')}")


# --- Footprints -----------------------------------------------------------------------------


def _footprint_clip(ctx: Context) -> tuple[str, BaseGeometry]:
    """The source's ``clip`` polygon (default ``training``, the domain of every plan 002
    footprint source) and how the notes name it."""
    clip = str(ctx.source.get("clip") or "training")
    name = "the training domain" if clip == "training" else f"the `{clip}` clip"
    return name, geo.clip_geometry(ctx.project, clip)


def _finish_footprints(ctx: Context, frame: gpd.GeoDataFrame, name: str) -> None:
    """Reproject, fix geometries, keep features intersecting the source's clip polygon
    (the training domain for the plan 002 sources), write."""
    domain_name, domain = _footprint_clip(ctx)
    frame = frame.to_crs(geo.WORK_CRS)
    invalid = int((~frame.geometry.is_valid).sum())
    frame["geometry"] = shapely.make_valid(frame.geometry.values)
    frame["geometry"] = frame.geometry.apply(_polygonal)
    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()]
    frame = frame[frame.geometry.intersects(domain)].reset_index(drop=True)
    for column in ("source", "source_id", "confidence", "release"):
        if column not in frame:
            frame[column] = None
    frame = frame[["source", "source_id", "confidence", "release",
                   *[c for c in frame.columns
                     if c not in ("source", "source_id", "confidence", "release", "geometry")],
                   "geometry"]]
    dest = ctx.staging / f"{name}.parquet"
    frame.to_parquet(dest, index=False, compression="zstd")
    extent = geo.extent(ctx.project)
    centroids = frame.geometry.centroid
    n_extent = int(centroids.within(extent).sum())
    ctx.add_product(dest, kind="footprints", features=len(frame), invalid_fixed=invalid,
                    in_extent_centroid=n_extent)
    ctx.notes.append(f"{len(frame)} features intersect {domain_name}; {n_extent} have "
                     f"their centroid in the extent; {invalid} invalid geometries fixed")


def _polygonal(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        return shapely.union_all(polys) if polys else shapely.Polygon()
    return shapely.Polygon()


def _tile_bounds(quadkey: str) -> tuple[float, float, float, float]:
    x = y = 0
    z = len(quadkey)
    for i, digit in enumerate(quadkey):
        mask = 1 << (z - i - 1)
        d = int(digit)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    n = 2**z

    def lat(yy: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def _ms_rows(ctx: Context) -> list[dict[str, str]]:
    index = ctx.cache / "ms_dataset_links.csv"
    got = download(ctx.params["index"], index, ctx.cache, s=ctx.s, progress=False)
    ctx.prov["index"] = got.record("(not kept) dataset-links.csv")
    rows = list(csv.DictReader(index.read_text(encoding="utf-8").splitlines()))
    domain = shapely.box(*geo.lonlat_bounds(_footprint_clip(ctx)[1]))
    return [r for r in rows if r["Location"] == ctx.params["location"]
            and shapely.box(*_tile_bounds(r["QuadKey"])).intersects(domain)]


def plan_ms_footprints(ctx: Context) -> list[PlanRow]:
    return [_info_row(head(r["Url"], ctx.s), f"quadkey {r['QuadKey']}") for r in _ms_rows(ctx)]


def fetch_ms_footprints(ctx: Context) -> None:
    frames = []
    for row in _ms_rows(ctx):
        dest = ctx.staging / "source" / f"{row['Location']}_{row['QuadKey']}.csv.gz"
        got = download(row["Url"], dest, ctx.cache, s=ctx.s)
        ctx.add_file(got.record(ctx.rel(dest)) | {"quadkey": row["QuadKey"],
                                                   "upload_date": row["UploadDate"]})
        features = []
        with gzip.open(dest, "rt", encoding="utf-8") as handle:
            for n, line in enumerate(handle):
                feature = json.loads(line)
                props = feature.get("properties") or {}
                features.append({"source_id": f"{row['QuadKey']}:{n}",
                                 "confidence": props.get("confidence"),
                                 "height": props.get("height"),
                                 "geometry": shape(feature["geometry"])})
        frames.append(gpd.GeoDataFrame(features, crs="EPSG:4326"))
    frame = pd.concat(frames, ignore_index=True)
    frame["source"] = "microsoft"
    frame["release"] = ctx.params["release"]
    _finish_footprints(ctx, gpd.GeoDataFrame(frame, crs="EPSG:4326"), "fp_microsoft")


def plan_google_v3(ctx: Context) -> list[PlanRow]:
    base, token = ctx.params["base"], ctx.params["token"]
    return [
        _info_row(head(f"{base}polygons_s2_level_4_gzip/{token}_buildings.csv.gz", ctx.s),
                  "downloaded to cache, filtered to the training domain, cache deleted"),
        _info_row(head(f"{base}score_thresholds_s2_level_4.csv", ctx.s)),
    ]


def fetch_google_v3(ctx: Context) -> None:
    base, token = ctx.params["base"], ctx.params["token"]
    thresholds = ctx.staging / "score_thresholds_s2_level_4.csv"
    got = download(f"{base}score_thresholds_s2_level_4.csv", thresholds, ctx.cache, s=ctx.s,
                   progress=False)
    ctx.add_file(got.record(ctx.rel(thresholds)))
    row = [r for r in csv.DictReader(thresholds.read_text(encoding="utf-8").splitlines())
           if r.get("s2_token") == token]
    if row:
        ctx.prov["thresholds_for_cell"] = row[0]
        ctx.notes.append(f"S2 cell {token} thresholds: " + ", ".join(
            f"{k}={v}" for k, v in row[0].items() if k not in ("s2_token", "geometry")))
    cached = ctx.cache / f"{token}_buildings.csv.gz"
    got = download(f"{base}polygons_s2_level_4_gzip/{token}_buildings.csv.gz", cached,
                   ctx.cache, s=ctx.s)
    ctx.add_file(got.record(f"(not kept) {cached.name}"))
    west, south, east, north = geo.lonlat_bounds(_footprint_clip(ctx)[1])
    pad = 0.005
    keep: list[dict[str, Any]] = []
    total = 0
    with gzip.open(cached, "rt", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            total += 1
            lat, lon = float(record["latitude"]), float(record["longitude"])
            if south - pad <= lat <= north + pad and west - pad <= lon <= east + pad:
                keep.append(record)
    frame = pd.DataFrame(keep)
    frame = gpd.GeoDataFrame(
        {"source": "google_open_buildings_v3", "source_id": frame["full_plus_code"],
         "confidence": frame["confidence"].astype(float), "release": ctx.params["release"],
         "area_in_meters": frame["area_in_meters"].astype(float),
         "latitude": frame["latitude"].astype(float),
         "longitude": frame["longitude"].astype(float)},
        geometry=shapely.from_wkt(frame["geometry"].values), crs="EPSG:4326",
    )
    ctx.notes.append(f"{total} rows scanned in {token}_buildings.csv.gz; the 8.4 GB cell file "
                     "was deleted after filtering (sha256/md5 kept in source.json)")
    _finish_footprints(ctx, frame, "fp_google_v3")
    cached.unlink()


def plan_osm_pbf(ctx: Context) -> list[PlanRow]:
    return [_info_row(head(ctx.params["url"], ctx.s), "dated snapshot (redirect target)")]


def fetch_osm_pbf(ctx: Context) -> None:
    info = head(ctx.params["url"], ctx.s)
    dated = info.final_url  # nepal-latest redirects to nepal-YYMMDD.osm.pbf
    dest = ctx.staging / "source" / Path(dated).name
    got = download(dated, dest, ctx.cache, s=ctx.s)
    ctx.add_file(got.record(ctx.rel(dest)))
    frame, stamp = osm_buildings(dest, geo.lonlat_bounds(_footprint_clip(ctx)[1]))
    ctx.prov["osm_replication_timestamp"] = stamp
    frame["release"] = stamp or Path(dated).name
    ctx.notes.append(f"snapshot {Path(dated).name}, osmosis_replication_timestamp {stamp}; "
                     "building=* ways and multipolygon relations (pyosmium areas); "
                     "building nodes ignored")
    _finish_footprints(ctx, frame, "fp_osm")


def osm_buildings(pbf: Path, lonlat_bounds: tuple[float, float, float, float],
                  pad: float = 0.005) -> tuple[gpd.GeoDataFrame, str]:
    """building=* ways and multipolygon relations (pyosmium areas) of a PBF whose first outer
    vertex lies in the padded lon/lat bounds (EPSG:4326), and the replication timestamp."""
    import osmium
    import osmium.filter
    import osmium.geom
    import osmium.io
    import osmium.osm

    header = osmium.io.Reader(str(pbf), osmium.osm.osm_entity_bits.NOTHING).header()
    stamp = header.get("osmosis_replication_timestamp", "")
    west, south, east, north = lonlat_bounds
    wkb = osmium.geom.WKBFactory()
    rows = []
    processor = (osmium.FileProcessor(str(pbf))
                 .with_areas(osmium.filter.KeyFilter("building"))
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
                 .with_filter(osmium.filter.KeyFilter("building")))
    for obj in processor:
        area: Any = obj
        first = next(iter(next(iter(area.outer_rings()), [])), None)
        if first is None:
            continue
        lon, lat = first.lon, first.lat
        if not (south - pad <= lat <= north + pad and west - pad <= lon <= east + pad):
            continue
        try:
            geom = shapely.from_wkb(bytes.fromhex(wkb.create_multipolygon(area)))
        except RuntimeError:
            continue
        kind = "w" if area.from_way() else "r"
        rows.append({"source_id": f"{kind}{area.orig_id()}",
                     "building": area.tags.get("building"),
                     "osm_version": area.version, "osm_timestamp": str(area.timestamp),
                     "geometry": geom})
    frame = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    frame["source"] = "osm"
    frame["confidence"] = np.nan
    return frame, stamp


def plan_overture(ctx: Context) -> list[PlanRow]:
    return [PlanRow(ctx.source["url"], None, "bbox query via uvx overturemaps (size unknown)")]


def fetch_overture(ctx: Context) -> None:
    west, south, east, north = geo.lonlat_bounds(geo.training(ctx.project))
    bbox = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"
    dest = ctx.staging / "source" / "overture_buildings_bbox.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = ["uvx", ctx.params["cli"], "download", f"--bbox={bbox}", "-f", "geoparquet",
               "-t", "building", "-r", ctx.params["release"], "-o", str(dest)]
    subprocess.run(command, check=True)
    ctx.add_file({"path": ctx.rel(dest), "url": ctx.source["url"], "bbox": bbox,
                  "command": " ".join(command[:-1] + ["<dest>"]),
                  "bytes": dest.stat().st_size, "sha256": sha256_file(dest)})
    raw = gpd.read_parquet(dest)
    if raw.crs is None:
        raw = raw.set_crs("EPSG:4326")

    def first_source(value: Any, key: str) -> Any:
        if value is None or len(value) == 0:
            return None
        return value[0].get(key)

    frame = gpd.GeoDataFrame(
        {"source": "overture", "source_id": raw["id"],
         "confidence": raw["sources"].apply(lambda v: first_source(v, "confidence")),
         "release": ctx.params["release"],
         "source_dataset": raw["sources"].apply(lambda v: first_source(v, "dataset")),
         "source_record_id": raw["sources"].apply(lambda v: first_source(v, "record_id"))},
        geometry=raw.geometry, crs=raw.crs,
    )
    counts = frame["source_dataset"].value_counts().to_dict()
    ctx.prov["overture_source_datasets"] = counts
    ctx.notes.append("primary source datasets: " + ", ".join(f"{k} {v}" for k, v in
                                                            counts.items()))
    _finish_footprints(ctx, frame, "fp_overture")


# --- Sentinel-2 and OAM ---------------------------------------------------------------------


def _s2_items(ctx: Context) -> list[dict[str, Any]]:
    items = []
    for date in ctx.params["dates"]:
        body = {"collections": ["sentinel-2-l2a"],
                "query": {"grid:code": {"eq": ctx.params["tile"]}},
                "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z", "limit": 10}
        found = post_json(f"{ctx.sources['_stac']['earth_search']}/search", body, ctx.s)
        if len(found["features"]) != 1:
            raise FetchError(f"Sentinel-2 {ctx.params['tile']} {date}: "
                             f"{len(found['features'])} items")
        items.append(found["features"][0])
    return items


def plan_s2_l2a(ctx: Context) -> list[PlanRow]:
    rows = []
    for item in _s2_items(ctx):
        for key in ctx.params["assets"]:
            info = head(item["assets"][key]["href"], ctx.s)
            row = _info_row(info, f"{item['id']} {key}: windowed read of the bbox")
            row.partial = True
            rows.append(row)
    return rows


def fetch_s2_l2a(ctx: Context) -> None:
    bbox = box(*geo.routing(ctx.project).bounds)
    for item in _s2_items(ctx):
        date = item["properties"]["datetime"][:10]
        out = ctx.staging / date
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{item['id']}.json").write_text(json.dumps(item, indent=1, sort_keys=True),
                                                encoding="utf-8")
        for key, band in ctx.params["assets"].items():
            href = item["assets"][key]["href"]
            dest = out / f"{band}.tif"
            info = geo.clip_raster(f"/vsicurl/{href}", dest, bbox)
            if info is None:
                raise FetchError(f"{href} does not overlap the routing domain")
            meta = head(href, ctx.s)
            ctx.add_file({"path": ctx.rel(dest), "url": href, "etag": meta.etag,
                          "last_modified": meta.last_modified,
                          "content_length": meta.content_length, "read": "windowed"})
            ctx.add_product(dest, kind="s2", date=date, band=band, item=item["id"],
                            scale=(item["assets"][key].get("raster:bands") or [{}])[0]
                            .get("scale"),
                            offset=(item["assets"][key].get("raster:bands") or [{}])[0]
                            .get("offset"))
        ctx.notes.append(f"{date}: {item['id']} ({item['properties']['datetime']}, scene cloud "
                         f"{item['properties'].get('eo:cloud_cover')} %)")
    ctx.notes.append("clip = bbox of the routing domain on the native 10/20 m grid; "
                     "reflectance = DN * 0.0001 - 0.1 (BOA offset, from the STAC raster:bands)")


def _oam_record(ctx: Context) -> dict[str, Any]:
    document = get_json(f"https://api.openaerialmap.org/meta/{ctx.params['meta_id']}", ctx.s)
    return document["results"]


def plan_oam(ctx: Context) -> list[PlanRow]:
    record = _oam_record(ctx)
    return [_info_row(head(record["uuid"], ctx.s), "whole native GeoTIFF + 0.25 m extent copy")]


def fetch_oam(ctx: Context) -> None:
    record = _oam_record(ctx)
    (ctx.staging / "oam_meta.json").write_text(
        json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    dest = ctx.staging / "source" / Path(record["uuid"]).name
    got = download(record["uuid"], dest, ctx.cache, s=ctx.s)
    ctx.add_file(got.record(ctx.rel(dest)))
    extent = geo.extent(ctx.project)
    copy = ctx.staging / f"{ctx.source['id']}_extent_0p25m.tif"
    info = geo.resampled_copy(dest, copy, extent, ctx.params["copy_resolution"])
    if info:
        ctx.add_product(copy, kind="oam", **info)
    ctx.add_product(dest, kind="oam_native")
    props = record.get("properties") or {}
    licence = props.get("license") or record.get("license")
    provider = record.get("provider")
    ctx.overrides["attribution"] = (f"{record.get('title')} by {provider} (OAM provider field, "
                                    f"verbatim), via OpenAerialMap, {licence}")
    ctx.notes.append(f"OAM provider (verbatim): {provider!r}; licence field: {licence!r}; "
                     f"acquired {record.get('acquisition_start')}; platform "
                     f"{record.get('platform')}, sensor {props.get('sensor')}, gsd "
                     f"{record.get('gsd')}; native file kept whole")


@dataclass(frozen=True)
class Method:
    plan: Callable[[Context], list[PlanRow]]
    fetch: Callable[[Context], Any] | None


METHODS: dict[str, Method] = {
    "aoi_copy": Method(plan_aoi_copy, fetch_aoi_copy),
    "aoi_build": Method(plan_aoi_build, fetch_aoi_build),
    "aoi_training_domain": Method(plan_aoi_training_domain, fetch_aoi_training_domain),
    "maxar_scene": Method(plan_maxar_scene, fetch_maxar_scene),
    "maxar_clip": Method(plan_maxar_clip, fetch_maxar_clip),
    "maxar_datamask": Method(plan_maxar_datamask, fetch_maxar_datamask),
    "cog_clip": Method(plan_cog_clip, fetch_cog_clip),
    "zip_member_clip": Method(plan_zip_member_clip, fetch_zip_member_clip),
    "pending": Method(plan_pending, None),
    "opentopo_globaldem": Method(plan_opentopo, fetch_opentopo),
    "nsidc_cmr_clip": Method(plan_nsidc_cmr, fetch_nsidc_cmr),
    "ms_footprints": Method(plan_ms_footprints, fetch_ms_footprints),
    "google_v3": Method(plan_google_v3, fetch_google_v3),
    "osm_pbf": Method(plan_osm_pbf, fetch_osm_pbf),
    "overture": Method(plan_overture, fetch_overture),
    "s2_l2a": Method(plan_s2_l2a, fetch_s2_l2a),
    "oam": Method(plan_oam, fetch_oam),
}
