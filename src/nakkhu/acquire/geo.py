"""AOI geometry and windowed raster clipping on the source grid (no resampling)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import rasterio.transform
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from nakkhu.config import Project

WORK_CRS = "EPSG:32645"
# Fixed buffers from plan 002: Maxar clips use extent + 500 m, DEMs use routing domain + 2 km.
EXTENT_BUFFER_M = 500.0
ROUTING_BUFFER_M = 2000.0
TRAINING_DOMAIN = "configs/aoi/training_domain_wv02_20240113.geojson"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "8",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_TIFF_INTERNAL_MASK": "YES",
}


def read_geometry(path: str | Path, crs: str = WORK_CRS) -> BaseGeometry:
    frame = gpd.read_file(path)
    return frame.to_crs(crs).geometry.union_all()


def aoi_path(project: Project, key: str) -> Path:
    aoi = project.config.values.get("aoi") or {}
    if key == "training_domain":
        return project.paths.resolve(aoi.get(key, TRAINING_DOMAIN))
    return project.paths.resolve(aoi[key])


def extent(project: Project) -> BaseGeometry:
    return read_geometry(aoi_path(project, "analysis_extent"))


def routing(project: Project) -> BaseGeometry:
    return read_geometry(aoi_path(project, "routing_domain"))


def training(project: Project) -> BaseGeometry:
    return read_geometry(aoi_path(project, "training_domain"))


def clip_geometry(project: Project, clip: str) -> BaseGeometry:
    """The polygon (EPSG:32645) a manifest ``clip`` value stands for."""
    if clip == "extent":
        return extent(project)
    if clip == "extent_buffer":
        return extent(project).buffer(EXTENT_BUFFER_M)
    if clip == "routing":
        return routing(project).buffer(ROUTING_BUFFER_M)
    if clip == "training":
        return training(project)
    raise ValueError(f"no clip geometry for {clip!r}")


def lonlat_bounds(geom: BaseGeometry, crs: str = WORK_CRS) -> tuple[float, float, float, float]:
    return transform_bounds(crs, "EPSG:4326", *geom.bounds, densify_pts=21)


def _win(col: int, row: int, width: int, height: int) -> Window:
    return Window.from_slices((row, row + height), (col, col + width))


def outward_window(src: Any, bounds: tuple[float, float, float, float]) -> Window | None:
    """Pixel-aligned window covering ``bounds`` (in the source CRS), clipped to the dataset."""
    win = from_bounds(*bounds, transform=src.transform)
    col0 = max(0, math.floor(win.col_off))
    row0 = max(0, math.floor(win.row_off))
    col1 = min(src.width, math.ceil(win.col_off + win.width))
    row1 = min(src.height, math.ceil(win.row_off + win.height))
    if col1 <= col0 or row1 <= row0:
        return None
    return _win(col0, row0, col1 - col0, row1 - row0)


def _profile(src: Any, window: Window, count: int, **overrides: Any) -> dict[str, Any]:
    profile = src.profile.copy()
    for key in ("photometric", "jpeg_quality", "jpegtablesmode", "compress", "predictor",
                "zlevel", "interleave", "blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    float_data = np.dtype(src.dtypes[0]).kind == "f"
    profile.update(
        driver="GTiff",
        width=int(window.width),
        height=int(window.height),
        count=count,
        transform=src.window_transform(window),
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=3 if float_data else 2,
        zlevel=6,
        bigtiff="IF_SAFER",
        interleave="pixel" if count > 1 else "band",
    )
    profile.update(overrides)
    return profile


def clip_raster(
    source: str,
    dest: str | Path,
    geom: BaseGeometry,
    geom_crs: str = WORK_CRS,
    *,
    mask_polygon: bool = False,
    chunk_rows: int = 2048,
) -> dict[str, Any] | None:
    """Copy the part of ``source`` covering ``geom`` to ``dest`` on the source grid.

    The output is a tiled, DEFLATE-compressed GeoTIFF (lossless with respect to the decoded
    source pixels). With ``mask_polygon`` the pixels outside ``geom`` are set to 0 and excluded
    by an internal mask, as are pixels the source itself masks. Returns ``None`` when ``geom``
    does not overlap the source.
    """
    dest = Path(dest)
    with rasterio.Env(**GDAL_ENV), rasterio.open(source) as src:
        src_geom = transform_geom(geom_crs, src.crs, mapping(geom))
        src_shape = shape(src_geom)
        window = outward_window(src, src_shape.bounds)
        if window is None:
            return None
        if mask_polygon and not src_shape.intersects(box(*src.window_bounds(window))):
            return None
        has_mask = mask_polygon or any(
            rasterio.enums.MaskFlags.per_dataset in flags for flags in src.mask_flag_enums
        )
        profile = _profile(src, window, src.count)
        if np.dtype(src.dtypes[0]).kind != "f" and src.count == 3:
            profile["photometric"] = "RGB"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        valid_pixels = 0
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, int(window.height), chunk_rows):
                rows = min(chunk_rows, int(window.height) - row)
                sub = _win(int(window.col_off), int(window.row_off) + row, int(window.width), rows)
                out_win = _win(0, row, int(window.width), rows)
                data = src.read(window=sub)
                if has_mask:
                    mask = src.dataset_mask(window=sub) > 0
                    if mask_polygon:
                        inside = ~geometry_mask(
                            [src_geom], out_shape=(rows, int(window.width)),
                            transform=src.window_transform(sub), all_touched=False,
                        )
                        mask &= inside
                    data[:, ~mask] = 0
                    dst.write_mask(np.where(mask, 255, 0).astype("uint8"), window=out_win)
                    valid_pixels += int(mask.sum())
                dst.write(data, window=out_win)
            for key, value in src.tags().items():
                dst.update_tags(**{key: value})
            dst.scales = src.scales
            dst.offsets = src.offsets
            if any(src.units):
                dst.units = src.units
            if any(src.descriptions):
                dst.descriptions = src.descriptions
        tmp.replace(dest)
        return {
            "source": source,
            "window": {"col_off": int(window.col_off), "row_off": int(window.row_off),
                       "width": int(window.width), "height": int(window.height)},
            "crs": str(src.crs),
            "res": list(src.res),
            "valid_pixels": valid_pixels if has_mask else None,
        }


def resampled_copy(
    source: str | Path,
    dest: str | Path,
    geom: BaseGeometry,
    resolution: float,
    geom_crs: str = WORK_CRS,
) -> dict[str, Any] | None:
    """Area-average copy of ``source`` at ``resolution`` (source CRS units), masked to ``geom``."""
    dest = Path(dest)
    with rasterio.Env(**GDAL_ENV), rasterio.open(source) as src:
        src_geom = transform_geom(geom_crs, src.crs, mapping(geom))
        window = outward_window(src, shape(src_geom).bounds)
        if window is None:
            return None
        factor = resolution / src.res[0]
        width = max(1, round(window.width / factor))
        height = max(1, round(window.height / factor))
        data = src.read(window=window, out_shape=(src.count, height, width),
                        resampling=Resampling.average)
        mask = src.dataset_mask(window=window, out_shape=(height, width),
                                resampling=Resampling.nearest) > 0
        bounds = src.window_bounds(window)
        west, south, east, north = bounds
        transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
        mask &= ~geometry_mask([src_geom], out_shape=(height, width), transform=transform)
        data[:, ~mask] = 0
        profile = _profile(src, window, src.count, width=width, height=height,
                           transform=transform)
        if src.count == 3:
            profile["photometric"] = "RGB"
        tmp = Path(str(dest) + ".tmp")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
            dst.write_mask(np.where(mask, 255, 0).astype("uint8"))
        tmp.replace(dest)
        return {"source": str(source), "resolution": resolution, "width": width,
                "height": height, "resampling": "average"}


def merge_first_valid(sources: Sequence[str | Path], dest: str | Path) -> dict[str, Any]:
    """Merge single-band rasters on one CRS and pixel grid into ``dest`` (first valid value
    wins, in the given order). Cells valid in more than one source are counted and their
    largest absolute difference is reported, so a value conflict is visible."""
    from rasterio.merge import merge

    dest = Path(dest)
    with rasterio.open(sources[0]) as first:
        nodata = first.nodata
        crs = first.crs
        profile = _profile(first, _win(0, 0, first.width, first.height), 1)
    mosaic, transform = merge([str(s) for s in sources], method="first", nodata=nodata)
    valid_count = np.zeros(mosaic.shape[1:], dtype=np.int16)
    max_diff = 0.0
    for path in sources:
        with rasterio.open(path) as src:
            if src.crs != crs:
                raise ValueError(f"{path}: CRS {src.crs} differs from {crs}")
            win = from_bounds(*src.bounds, transform=transform)
            r0, c0 = round(win.row_off), round(win.col_off)
            band = src.read(1)
            ok = np.isfinite(band) & ((band != nodata) if nodata is not None else True)
            sub = mosaic[0, r0:r0 + src.height, c0:c0 + src.width]
            both = ok & (valid_count[r0:r0 + src.height, c0:c0 + src.width] > 0)
            if both.any():
                max_diff = max(max_diff, float(np.nanmax(np.abs(sub[both] - band[both]))))
            valid_count[r0:r0 + src.height, c0:c0 + src.width] += ok
    profile.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=transform,
                   nodata=nodata, crs=crs)
    tmp = dest.with_name(dest.name + ".tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
    tmp.replace(dest)
    return {"width": int(mosaic.shape[2]), "height": int(mosaic.shape[1]),
            "crs": str(crs), "res": [float(transform.a), float(-transform.e)],
            "cells_valid": int((valid_count > 0).sum()),
            "cells_valid_in_several": int((valid_count > 1).sum()),
            "max_abs_diff_where_several_m": max_diff}


_GDAL_TYPES = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "float32": "Float32"}


def write_vrt(dest: str | Path, sources: Sequence[str | Path]) -> None:
    """Mosaic VRT (paths relative to the VRT) over GeoTIFFs on one CRS and pixel grid."""
    from xml.sax.saxutils import escape

    dest = Path(dest)
    metas = []
    for path in sources:
        with rasterio.open(path) as src:
            metas.append((Path(path), src.transform, src.width, src.height, src.count,
                          src.dtypes[0], src.crs, src.colorinterp,
                          any(rasterio.enums.MaskFlags.per_dataset in f
                              for f in src.mask_flag_enums)))
    res_x, res_y = metas[0][1].a, metas[0][1].e
    x0 = min(m[1].c for m in metas)
    y0 = max(m[1].f for m in metas)
    x1 = max(m[1].c + m[2] * res_x for m in metas)
    y1 = min(m[1].f + m[3] * res_y for m in metas)
    width = round((x1 - x0) / res_x)
    height = round((y0 - y1) / -res_y)
    count, dtype, crs = metas[0][4], _GDAL_TYPES[metas[0][5]], metas[0][6]

    def source_xml(meta: tuple, band: str, kind: str = "SimpleSource") -> str:
        path, transform, w, h = meta[0], meta[1], meta[2], meta[3]
        xoff = round((transform.c - x0) / res_x)
        yoff = round((transform.f - y0) / res_y)
        rel = escape(path.relative_to(dest.parent).as_posix())
        return (f"<{kind}><SourceFilename relativeToVRT=\"1\">{rel}</SourceFilename>"
                f"<SourceBand>{band}</SourceBand>"
                f"<SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{w}\" ySize=\"{h}\"/>"
                f"<DstRect xOff=\"{xoff}\" yOff=\"{yoff}\" xSize=\"{w}\" ySize=\"{h}\"/></{kind}>")

    parts = [f"<VRTDataset rasterXSize=\"{width}\" rasterYSize=\"{height}\">",
             f"<SRS>{escape(crs.to_wkt())}</SRS>",
             f"<GeoTransform>{x0!r}, {res_x!r}, 0.0, {y0!r}, 0.0, {res_y!r}</GeoTransform>"]
    for band in range(1, count + 1):
        interp = metas[0][7][band - 1].name.capitalize()
        parts.append(f"<VRTRasterBand dataType=\"{dtype}\" band=\"{band}\">"
                     f"<ColorInterp>{interp}</ColorInterp>")
        parts += [source_xml(m, str(band)) for m in metas]
        parts.append("</VRTRasterBand>")
    if all(m[8] for m in metas):
        parts.append("<MaskBand><VRTRasterBand dataType=\"Byte\">")
        parts += [source_xml(m, "mask,1") for m in metas]
        parts.append("</VRTRasterBand></MaskBand>")
    parts.append("</VRTDataset>")
    dest.write_text("\n".join(parts) + "\n", encoding="utf-8")
