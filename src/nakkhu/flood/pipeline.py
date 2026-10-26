"""``nakkhu flood change``: build the automatic flood-trace and damage change map (plan 009
Steps 1-5) and promote its artifacts.

Order: buildings (registered) -> 4 m observability masks per strip (Step 2) -> per-block
co-registration with held-back verification (Step 1) -> histogram matching over the observable
extent (Step 3) -> building features and Sentinel-2 support (Step 4) -> scores and classes per
inventory (Step 5). The masks are computed before the shifts are known; their 12 m dilation
covers the few-metre offsets (the shifts are reported in ``coregistration.csv``).

Inputs are the WV02 / GE01 mosaics, Sentinel-2, the reference footprints and their plan 004
registration shifts only: the resolved config holds no HAND, DEM or channel path, and the
modules that compute pixels and scores import none (tests/test_flood.py).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features as rfeatures
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window as RWindow

from nakkhu.acquire import geo
from nakkhu.artifacts import promote
from nakkhu.config import Project, load_config
from nakkhu.flood import change as C
from nakkhu.flood import cloudmask as M
from nakkhu.flood import coregister as R
from nakkhu.flood import features as F
from nakkhu.flood import normalise as N
from nakkhu.flood.buildings import load_inventories
from nakkhu.runs import Run, atomic_write_json, atomic_write_text, create_run

CHANGE_CONFIG = "configs/flood/change.yaml"
STAGE = "flood"
S2_ASSETS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir16"}
RESEARCH_CLEAR_SHARE = 0.85      # .tmp/research-area-imagery.md (rough, +-10 points)


class FloodError(RuntimeError):
    """The change map cannot be built."""


class Grid:
    """The shared 0.5 m grid of the mosaics (plan 003) and a processing window on it."""

    def __init__(self, transform: Any, height: int, width: int):
        self.transform = transform
        self.height, self.width = height, width
        self.pixel = float(transform.a)
        self.x0, self.y0 = float(transform.c), float(transform.f)

    def rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.floor((self.y0 - np.asarray(y)) / self.pixel).astype(np.int64),
                np.floor((np.asarray(x) - self.x0) / self.pixel).astype(np.int64))

    def window(self, r0: int, c0: int, h: int, w: int) -> F.Window:
        return F.Window(x0=self.x0 + c0 * self.pixel, y0=self.y0 - r0 * self.pixel, height=h,
                        width=w, pixel=self.pixel)

    def affine(self, r0: int, c0: int, pixel: float | None = None) -> Any:
        from affine import Affine

        p = pixel or self.pixel
        return Affine(p, 0, self.x0 + c0 * self.pixel, 0, -p, self.y0 - r0 * self.pixel)


def _reader(path: Path) -> Callable[[int, int, int, int], np.ndarray]:
    """``read(r0, c0, h, w)`` -> float32 (R, G, B, valid) with zeros outside the image."""
    ds = rasterio.open(path)

    def read(r0: int, c0: int, h: int, w: int) -> np.ndarray:
        rgb = ds.read(window=RWindow.from_slices((r0, r0 + h), (c0, c0 + w), boundless=True), boundless=True, fill_value=0)
        valid = (rgb > 0).any(axis=0)
        return np.concatenate([rgb.astype(np.float32), valid[None].astype(np.float32)])

    return read


def aggregate(read: Callable, r0: int, c0: int, h: int, w: int, agg: int,
              stripe: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Mean RGB of ``agg`` x ``agg`` cells and their validity (all pixels valid)."""
    hh, ww = h // agg, w // agg
    rgb = np.zeros((3, hh, ww), dtype=np.float32)
    valid = np.zeros((hh, ww), dtype=bool)
    step = (stripe // agg) * agg
    for top in range(0, hh * agg, step):
        n = min(step, hh * agg - top)
        block = read(r0 + top, c0, n, ww * agg)
        cells = block.reshape(4, n // agg, agg, ww, agg)
        rgb[:, top // agg:(top + n) // agg] = cells[:3].mean(axis=(2, 4))
        valid[top // agg:(top + n) // agg] = cells[3].min(axis=(1, 3)) > 0
    return rgb, valid


def _rasterize(geoms: list[Any], shape: tuple[int, int], transform: Any) -> np.ndarray:
    if not geoms:
        return np.zeros(shape, dtype=bool)
    return rfeatures.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool)


class ChangeMap:
    def __init__(self, project: Project, config: str = CHANGE_CONFIG,
                 inventories: list[str] | None = None):
        self.project = project
        overrides = [f"buildings.inventories={json.dumps(inventories)}"] if inventories else []
        self.cfg = load_config(project.paths.resolve(config), overrides)
        self.v = self.cfg.values
        self.timings: dict[str, float] = {}

    def resolve(self, value: str) -> Path:
        return self.project.paths.resolve(
            value.replace("{data_root}", str(self.project.paths.data_root)))

    # -- set-up ------------------------------------------------------------------------------
    def open_grid(self) -> None:
        paths = [self.resolve(self.v["scenes"]["pre"]["mosaic"])]
        paths += [self.resolve(s["mosaic"]) for s in self.v["scenes"]["post"]]
        specs = []
        for p in paths:
            if not p.is_file():
                raise FloodError(f"mosaic {p} is missing (plan 003 `nakkhu tile`)")
            with rasterio.open(p) as ds:
                specs.append((tuple(ds.transform), ds.height, ds.width, ds.crs.to_string()))
        if len(set(specs)) != 1:
            raise FloodError("mosaics do not share one grid (plan 003 drift)")
        t, h, w, crs = specs[0]
        if crs != geo.WORK_CRS:
            raise FloodError(f"mosaic CRS {crs} != {geo.WORK_CRS}")
        with rasterio.open(paths[0]) as ds:
            self.grid = Grid(ds.transform, h, w)
        self.pre_read = _reader(paths[0])
        self.strips = [str(s["strip"]) for s in self.v["scenes"]["post"]]
        self.post_read = {str(s["strip"]): _reader(self.resolve(s["mosaic"]))
                          for s in self.v["scenes"]["post"]}
        self.extent = geo.read_geometry(geo.aoi_path(self.project, str(self.v["region"])))
        cb = self.v["coregistration"]
        self.block_px = int(round(float(cb["block_m"]) / self.grid.pixel))
        self.agg = int(self.v["mask"]["agg_px"])
        if self.block_px % self.agg:
            raise FloodError("block size must be a multiple of the mask aggregation")
        minx, miny, maxx, maxy = self.extent.bounds
        rr, cc = self.grid.rowcol(np.array([minx, maxx]), np.array([maxy, miny]))
        rows, cols = (int(rr[0]), int(rr[1])), (int(cc[0]), int(cc[1]))
        b = self.block_px
        self.br = range(int(rows[0]) // b, int(rows[1]) // b + 1)
        self.bc = range(int(cols[0]) // b, int(cols[1]) // b + 1)
        self.win_r0, self.win_c0 = self.br[0] * b, self.bc[0] * b
        self.win_h = min(len(self.br) * b, self.grid.height - self.win_r0)
        self.win_w = min(len(self.bc) * b, self.grid.width - self.win_c0)
        self.win_h -= self.win_h % self.agg
        self.win_w -= self.win_w % self.agg
        self.blocks = [(r, c) for r in self.br for c in self.bc
                       if shapely.intersects(self.block_box(r, c), self.extent)]

    def block_box(self, br: int, bc: int, offset_px: int = 0) -> Any:
        b, p = self.block_px, self.grid.pixel
        x0 = self.grid.x0 + (bc * b + offset_px) * p
        y0 = self.grid.y0 - (br * b + offset_px) * p
        return shapely.box(x0, y0 - b * p, x0 + b * p, y0)

    # -- buildings ---------------------------------------------------------------------------
    def load_buildings(self) -> None:
        bcfg = self.v["buildings"]
        sweep = load_config(self.project.paths.resolve(str(bcfg["hand_sweep_config"]))).values
        self.inv_source: dict[str, Path] = {}
        self.pending: list[str] = []
        for name in dict.fromkeys([*bcfg["inventories"], *bcfg["ground_mask_from"]]):
            path = self.resolve(str(sweep["inventories"][name]["path"]))
            if path.is_file():
                self.inv_source[name] = path
            elif name in bcfg["ground_mask_from"]:
                raise FloodError(f"ground-mask inventory {name} is missing ({path})")
            else:
                self.pending.append(name)
        self.inventories = [n for n in bcfg["inventories"] if n not in self.pending]
        names = list(self.inv_source)
        self.inv, self.inv_info = load_inventories(
            self.project, names, str(bcfg["hand_sweep_config"]), dict(bcfg["registration"]))
        built = [g for n in bcfg["ground_mask_from"] for g in self.inv[n].geometry.values]
        self.built = np.asarray(built, dtype=object)
        self.built_tree = shapely.STRtree(self.built)

    def built_raster(self, r0: int, c0: int, h: int, w: int, buffer_m: float = 0.0) -> np.ndarray:
        win = self.grid.window(r0, c0, h, w)
        box = shapely.box(win.x0, win.y0 - h * win.pixel, win.x0 + w * win.pixel, win.y0)
        idx = self.built_tree.query(box.buffer(buffer_m))
        geoms = list(self.built[idx])
        if buffer_m:
            geoms = list(shapely.buffer(np.asarray(geoms, dtype=object), buffer_m))
        return _rasterize(geoms, (h, w), self.grid.affine(r0, c0))

    # -- Step 2: masks -----------------------------------------------------------------------
    def build_masks(self, run: Run) -> dict[str, Any]:
        mcfg = self.v["mask"]
        cell = self.agg * self.grid.pixel
        pre4, pre_valid4 = aggregate(self.pre_read, self.win_r0, self.win_c0, self.win_h,
                                     self.win_w, self.agg)
        t4 = self.grid.affine(self.win_r0, self.win_c0, cell)
        self.extent4 = _rasterize([self.extent], pre_valid4.shape, t4)
        self.unobs4: dict[str, np.ndarray] = {}
        self.valid4: dict[str, np.ndarray] = {}
        stats: dict[str, Any] = {"cell_m": cell, "extent_cells": int(self.extent4.sum()),
                                 "strips": {}}
        clear_any = np.zeros_like(self.extent4)
        for strip in self.strips:
            post4, post_valid4 = aggregate(self.post_read[strip], self.win_r0, self.win_c0,
                                           self.win_h, self.win_w, self.agg)
            valid = pre_valid4 & post_valid4 & self.extent4
            masks = M.observability(post4, pre4, valid, cell, mcfg, N.match)
            unobs = masks["unobservable"] | ~valid
            self.unobs4[strip] = unobs
            self.valid4[strip] = valid
            clear = valid & ~masks["unobservable"]
            clear_any |= clear
            n = int(self.extent4.sum())
            stats["strips"][strip] = {
                "valid_share": float(valid.sum() / n),
                "cloud_share": float((masks["cloud"] & valid).sum() / n),
                "shadow_share": float((masks["shadow"] & valid).sum() / n),
                "observable_share": float(clear.sum() / n)}
            with rasterio.open(run.path / f"mask_{strip}.tif", "w", driver="GTiff",
                               height=unobs.shape[0], width=unobs.shape[1], count=1,
                               dtype="uint8", crs=geo.WORK_CRS, transform=t4,
                               compress="deflate") as dst:
                code = np.where(~valid, 0, np.where(masks["cloud"], 2,
                                                    np.where(masks["shadow"], 3,
                                                             np.where(masks["unobservable"], 4,
                                                                      1))))
                dst.write(code.astype(np.uint8), 1)
        n = int(self.extent4.sum())
        stats["observable_share_either_strip"] = float(clear_any.sum() / n)
        stats["research_estimate"] = {"clear_share_both_strips": RESEARCH_CLEAR_SHARE,
                                      "tolerance_points": 10,
                                      "note": "rough brightness mask, shadows not removed"}
        stats["difference_points"] = round(100 * (stats["observable_share_either_strip"]
                                                  - RESEARCH_CLEAR_SHARE), 2)
        stats["within_tolerance"] = abs(stats["difference_points"]) <= 10
        stats["mask_codes"] = {"0": "no data / outside", "1": "observable", "2": "cloud",
                               "3": "shadow", "4": "dilation halo"}
        return stats

    def unobs_full(self, strip: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        """0.5 m view of the 4 m unobservable mask (True outside the mask window)."""
        rr = (np.arange(r0, r0 + h) - self.win_r0) // self.agg
        cc = (np.arange(c0, c0 + w) - self.win_c0) // self.agg
        m4 = self.unobs4[strip]
        okr = (rr >= 0) & (rr < m4.shape[0])
        okc = (cc >= 0) & (cc < m4.shape[1])
        out = np.ones((h, w), dtype=bool)
        sub = m4[np.ix_(np.clip(rr, 0, m4.shape[0] - 1), np.clip(cc, 0, m4.shape[1] - 1))]
        out[np.ix_(okr, okc)] = sub[np.ix_(okr, okc)]
        return out

    # -- Step 1: co-registration -------------------------------------------------------------
    def coregister(self) -> pd.DataFrame:
        cb = self.v["coregistration"]
        b = self.block_px
        rows: list[dict[str, Any]] = []
        self.field: dict[str, dict[tuple[int, int], tuple[float, float]]] = {}
        buffer_m = float(cb["footprint_buffer_m"])
        lims = (float(cb["max_shift_px"]), float(cb["min_peak_ratio"]),
                float(cb["min_ground_frac"]))
        built_cache = {key: self.built_raster(key[0] * b, key[1] * b, b, b, buffer_m)
                       for key in self.blocks}
        for strip in self.strips:
            est: dict[tuple[int, int], tuple[float, float]] = {}
            ok: dict[tuple[int, int], bool] = {}
            for key in self.blocks:
                r0, c0 = key[0] * b, key[1] * b
                pre = self.pre_read(r0, c0, b, b)
                post = self.post_read[strip](r0, c0, b, b)
                pre_mask = (pre[3] > 0) & ~built_cache[key]
                post_mask = (post[3] > 0) & ~self.unobs_full(strip, r0, c0, b, b)
                s = R.phase_correlation(pre[:3].mean(axis=0), post[:3].mean(axis=0),
                                        pre_mask, post_mask)
                good, why = R.accept(s, *lims)
                est[key], ok[key] = (s.dy, s.dx), good
                rows.append({"strip": strip, "set": "estimate", "block_row": key[0],
                             "block_col": key[1], "dy_px": s.dy, "dx_px": s.dx,
                             "peak_ratio": s.peak_ratio, "ground_frac": s.ground_frac,
                             "accepted": good, "reason": why})
            final = R.fill_rejected(self.blocks, est, ok)
            self.field[strip] = {k: (v[0], v[1]) for k, v in final.items()}
            for row in rows:
                if row["strip"] == strip and row["set"] == "estimate":
                    k = (row["block_row"], row["block_col"])
                    row["applied_dy_px"], row["applied_dx_px"], row["applied_from"] = final[k]
            # held-back verification blocks (offset by half a block), after correction
            half = int(round(float(cb["holdout_offset_m"]) / self.grid.pixel))
            for br in self.br:
                for bc in self.bc:
                    box = self.block_box(br, bc, half)
                    if not shapely.intersects(box, self.extent):
                        continue
                    r0, c0 = br * b + half, bc * b + half
                    pre = self.pre_read(r0, c0, b, b)
                    post = self.coregistered(strip, r0, c0, b, b)
                    built = self.built_raster(r0, c0, b, b, buffer_m)
                    pre_mask = (pre[3] > 0) & ~built
                    post_mask = (post[3] > 0.999) & ~self.unobs_full(strip, r0, c0, b, b)
                    s = R.phase_correlation(pre[:3].mean(axis=0), post[:3].mean(axis=0),
                                            pre_mask, post_mask)
                    good, why = R.accept(s, *lims)
                    rows.append({"strip": strip, "set": "held_back", "block_row": br,
                                 "block_col": bc, "offset_px": half, "dy_px": s.dy,
                                 "dx_px": s.dx, "peak_ratio": s.peak_ratio,
                                 "ground_frac": s.ground_frac, "accepted": good, "reason": why,
                                 "residual_px": s.magnitude})
        return pd.DataFrame(rows)

    def shift_of(self, strip: str) -> Callable[[int, int], tuple[float, float]]:
        field = self.field[strip]
        keys = list(field)

        def lookup(br: int, bc: int) -> tuple[float, float]:
            if (br, bc) in field:
                return field[(br, bc)]
            near = min(keys, key=lambda k: (k[0] - br) ** 2 + (k[1] - bc) ** 2)
            return field[near]

        return lookup

    def coregistered(self, strip: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        return R.piecewise_shift(self.post_read[strip], r0, c0, h, w, self.block_px,
                                 self.shift_of(strip), margin=4)

    # -- Step 3: histogram matching ----------------------------------------------------------
    def histogram_luts(self) -> dict[str, Any]:
        bins = int(self.v["normalise"]["bins"])
        b = self.block_px
        self.lut: dict[str, np.ndarray] = {}
        out: dict[str, Any] = {}
        for strip in self.strips:
            hp = np.zeros((3, bins), dtype=np.int64)
            hq = np.zeros((3, bins), dtype=np.int64)
            for br, bc in self.blocks:
                r0, c0 = br * b, bc * b
                pre = self.pre_read(r0, c0, b, b)
                post = self.coregistered(strip, r0, c0, b, b)
                inside = self.extent_full(r0, c0, b, b)
                mask = ((pre[3] > 0) & (post[3] > 0.999) & inside
                        & ~self.unobs_full(strip, r0, c0, b, b))
                hp += N.histograms(np.rint(post[:3]).astype(np.int64), mask, bins)
                hq += N.histograms(pre[:3].astype(np.int64), mask, bins)
            self.lut[strip] = N.lut_from_histograms(hp, hq)
            out[strip] = {"pixels": int(hp[0].sum()),
                          "lut_at_64_128_192": [[round(float(self.lut[strip][k][v]), 2)
                                                 for v in (64, 128, 192)] for k in range(3)]}
        return out

    def extent_full(self, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        rr = (np.arange(r0, r0 + h) - self.win_r0) // self.agg
        cc = (np.arange(c0, c0 + w) - self.win_c0) // self.agg
        e4 = self.extent4
        okr = (rr >= 0) & (rr < e4.shape[0])
        okc = (cc >= 0) & (cc < e4.shape[1])
        out = np.zeros((h, w), dtype=bool)
        sub = e4[np.ix_(np.clip(rr, 0, e4.shape[0] - 1), np.clip(cc, 0, e4.shape[1] - 1))]
        out[np.ix_(okr, okc)] = sub[np.ix_(okr, okc)]
        return out

    # -- Step 4: building features -----------------------------------------------------------
    def building_features(self) -> dict[str, dict[str, pd.DataFrame]]:
        bcfg, mcfg = self.v["buildings"], self.v["mask"]
        ring_m = [float(v) for v in bcfg["ring_m"]]
        zone_m = float(bcfg["observable_buffer_m"])
        reach = max(ring_m[1], zone_m)
        min_ring = int(bcfg["min_ring_px"])
        dark = float(mcfg["pixel_dark_max"])
        b = self.block_px
        names = list(self.inventories)
        results: dict[str, dict[str, list[dict[str, Any] | None]]] = {
            n: {s: [None] * len(self.inv[n]) for s in self.strips} for n in names}
        cent = {n: shapely.centroid(np.asarray(self.inv[n].geometry.values, dtype=object))
                for n in names}
        rc = {n: self.grid.rowcol(shapely.get_x(cent[n]), shapely.get_y(cent[n])) for n in names}
        empty = {"observable": False, "reason": "outside the processed blocks", "n_ring_px": 0,
                 "n_roof_px": 0, **{f"{p}_d_{f}": float("nan") for p in ("ring", "roof")
                                    for f in F.PIXEL_FEATURES}}
        for br, bc in self.blocks:
            members = {n: np.nonzero((rc[n][0] // b == br) & (rc[n][1] // b == bc))[0]
                       for n in names}
            if not any(len(m) for m in members.values()):
                continue
            r0, c0 = br * b, bc * b
            pad = 4
            for n, m in members.items():
                if len(m):
                    bounds = shapely.bounds(np.asarray(self.inv[n].geometry.values,
                                                       dtype=object)[m])
                    x_lo = self.grid.x0 + c0 * self.grid.pixel
                    y_hi = self.grid.y0 - r0 * self.grid.pixel
                    x_hi, y_lo = x_lo + b * self.grid.pixel, y_hi - b * self.grid.pixel
                    over = max(0.0, x_lo - bounds[:, 0].min(), bounds[:, 2].max() - x_hi,
                               y_lo - bounds[:, 1].min(), bounds[:, 3].max() - y_hi)
                    pad = max(pad, int(np.ceil((over + reach) / self.grid.pixel)) + 4)
            wr0, wc0, h, w = r0 - pad, c0 - pad, b + 2 * pad, b + 2 * pad
            win = self.grid.window(wr0, wc0, h, w)
            pre = self.pre_read(wr0, wc0, h, w)
            pre_f = F.pixel_features(pre[:3])
            pre_ok = (pre[3] > 0) & (pre_f["brightness"] >= dark)
            built = self.built_raster(wr0, wc0, h, w)
            strip_data = {}
            for strip in self.strips:
                post = self.coregistered(strip, wr0, wc0, h, w)
                valid = post[3] > 0.999
                matched = N.apply_lut(post[:3], self.lut[strip])
                post_f = F.pixel_features(matched)
                diffs = {k: post_f[k] - pre_f[k] for k in F.PIXEL_FEATURES}
                usable = pre_ok & valid & (post_f["brightness"] >= dark)
                strip_data[strip] = (diffs, usable, valid, self.unobs_full(strip, wr0, wc0, h, w))
                del post, matched, post_f
            cache: dict[bytes, dict[str, dict[str, Any]]] = {}
            for n, m in members.items():
                geoms = np.asarray(self.inv[n].geometry.values, dtype=object)
                for i in m:
                    key = shapely.to_wkb(geoms[i])
                    if key not in cache:
                        masks = F.building_masks(geoms[i], win, built, ring_m, zone_m)
                        cache[key] = {s: (dict(empty, reason="empty window") if masks is None
                                          else F.strip_features(masks, *strip_data[s], min_ring))
                                      for s in self.strips}
                    for s in self.strips:
                        results[n][s][i] = cache[key][s]
        out: dict[str, dict[str, pd.DataFrame]] = {}
        for n in names:
            out[n] = {s: pd.DataFrame([r if r is not None else empty for r in results[n][s]])
                      for s in self.strips}
        return out

    # -- Sentinel-2 --------------------------------------------------------------------------
    def s2_date(self, folder: Path) -> tuple[dict[str, np.ndarray], np.ndarray, Any]:
        items = sorted(folder.glob("*.json"))
        if not items:
            raise FloodError(f"no STAC item in {folder}")
        item = json.loads(items[0].read_text(encoding="utf-8"))
        with rasterio.open(folder / "B02.tif") as ref:
            shape, transform, crs = (ref.height, ref.width), ref.transform, ref.crs
        bands: dict[str, np.ndarray] = {}
        for band, asset in S2_ASSETS.items():
            rb = item["assets"][asset]["raster:bands"][0]
            with rasterio.open(folder / f"{band}.tif") as ds:
                raw = np.zeros(shape, dtype=np.float64)
                reproject(ds.read(1), raw, src_transform=ds.transform, src_crs=ds.crs,
                          dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
                          src_nodata=0, dst_nodata=0)
            data = raw * float(rb.get("scale", 1.0)) + float(rb.get("offset", 0.0))
            data[raw == 0] = np.nan
            bands[band] = data
        with rasterio.open(folder / "SCL.tif") as ds:
            scl = np.zeros(shape, dtype=np.uint8)
            reproject(ds.read(1), scl, src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest)
        clear = ~np.isin(scl, [int(v) for v in self.v["sentinel2"]["scl_unusable"]])
        return bands, clear, transform

    def s2_features(self) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        scfg = self.v["sentinel2"]
        pre_b, pre_clear, t = self.s2_date(self.resolve(scfg["pre"]))
        post_b, post_clear, t2 = self.s2_date(self.resolve(scfg["post"]))
        if tuple(t) != tuple(t2):
            raise FloodError("Sentinel-2 dates are not on one grid")
        pre_i, post_i = F.s2_indices(pre_b), F.s2_indices(post_b)
        clear = pre_clear & post_clear
        out: dict[str, pd.DataFrame] = {}
        inv_t = ~t
        for n in self.inventories:
            cent = shapely.centroid(np.asarray(self.inv[n].geometry.values, dtype=object))
            cols, rows = inv_t * (shapely.get_x(cent), shapely.get_y(cent))
            rows = np.floor(rows).astype(np.int64)
            cols = np.floor(cols).astype(np.int64)
            out[n] = pd.DataFrame({f"s2_d_{k}": F.s2_window_mean(
                post_i[k] - pre_i[k], clear, rows, cols, int(scfg["window_px"]),
                int(scfg["min_clear_px"])) for k in F.S2_FEATURES})
        ext = _rasterize([self.extent], clear.shape, t)
        stats = {"clear_share_extent_both_dates": float(clear[ext].mean()),
                 "pre_clear_share_extent": float(pre_clear[ext].mean()),
                 "post_clear_share_extent": float(post_clear[ext].mean())}
        return out, stats

    # -- run ---------------------------------------------------------------------------------
    def timed(self, name: str, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        value = fn()
        self.timings[name] = round(time.perf_counter() - start, 1)
        return value

    def execute(self) -> tuple[Run, dict[str, Any]]:
        run = create_run(self.project, STAGE, self.cfg)
        summary: dict[str, Any] = {"config": self.cfg.source, "config_sha256": self.cfg.sha256,
                                   "version": self.v.get("version")}
        with run.step("setup"):
            self.open_grid()
            self.timed("buildings", self.load_buildings)
            summary["blocks"] = len(self.blocks)
            summary["pending_inventories"] = self.pending
            summary["inventories"] = self.inv_info
        with run.step("masks"):
            mask_stats = self.timed("masks", lambda: self.build_masks(run))
            atomic_write_json(run.path / "observable_mask_stats.json", mask_stats)
        with run.step("coregister"):
            coreg = self.timed("coregister", self.coregister)
            coreg.to_csv(run.path / "coregistration.csv", index=False)
            held = coreg[(coreg["set"] == "held_back") & coreg["accepted"]]
            summary["coregistration"] = {
                s: {"blocks_accepted": int(((coreg["strip"] == s) & (coreg["set"] == "estimate")
                                            & coreg["accepted"]).sum()),
                    "blocks": len(self.blocks),
                    "median_shift_px": float(np.median(np.hypot(
                        coreg.loc[(coreg["strip"] == s) & (coreg["set"] == "estimate"),
                                  "applied_dy_px"],
                        coreg.loc[(coreg["strip"] == s) & (coreg["set"] == "estimate"),
                                  "applied_dx_px"]))),
                    "held_back_accepted": int((held["strip"] == s).sum()),
                    "held_back_median_residual_px": float(held.loc[held["strip"] == s,
                                                                   "residual_px"].median())}
                for s in self.strips}
            limit = float(self.v["coregistration"]["verify_max_median_px"])
            for v in summary["coregistration"].values():
                v["verify_pass"] = bool(v["held_back_median_residual_px"] <= limit)
            summary["coregistration_verify"] = {s: v["verify_pass"]
                                                for s, v in summary["coregistration"].items()}
        with run.step("normalise"):
            summary["normalise"] = self.timed("normalise", self.histogram_luts)
        with run.step("features"):
            feats = self.timed("features", self.building_features)
            s2, s2_stats = self.timed("sentinel2", self.s2_features)
            summary["sentinel2"] = s2_stats
        with run.step("score"):
            scfg = self.v["score"]
            per_inv: dict[str, Any] = {}
            gate = bool(self.v["coregistration"].get("gate", True))
            primary_strips = [s for s in self.strips
                              if not gate or summary["coregistration_verify"][s]]
            summary["primary_strips"] = primary_strips
            summary["sensitivity_strips"] = self.strips
            if not primary_strips:
                raise FloodError("no strip passed the co-registration gate")
            self.outputs: list[str] = []
            for n in self.inventories:
                per_strip = {s: pd.concat([feats[n][s], s2[n]], axis=1) for s in self.strips}
                scored, meta = C.score_buildings(per_strip, primary_strips, scfg["flood"],
                                                 scfg["damage"], scfg["required"], scfg["gmm"])
                sens, sens_meta = C.score_buildings(per_strip, self.strips, scfg["flood"],
                                                    scfg["damage"], scfg["required"],
                                                    scfg["gmm"])
                table = self.output_table(n, per_strip, scored, sens, primary_strips)
                path = run.path / f"change_map_buildings_{n}.parquet"
                table.to_parquet(path, index=False, compression="zstd")
                per_inv[n] = self.inventory_summary(table, meta)
                per_inv[n]["sensitivity_all_strips"] = {
                    "n_observable": int(table["observable_all_strips"].sum()),
                    "n_positive": int((table["class_all_strips"] == "positive").sum()),
                    "flood_boundary": sens_meta["flood"]["boundary"]}
                source = self.inv_source[n]
                meta_doc = {"inventory": n, "change_map": path.name,
                            "change_map_sha256": _sha256(path),
                            "inventory_source": self.project.paths.relative(source),
                            "inventory_sha256": _sha256(source),
                            "buildings_sha256": hashlib.sha256(
                                b"".join(str(i).encode() + b"\0" + w for i, w in zip(
                                    table["building_id"], shapely.to_wkb(np.asarray(
                                        table.geometry.values, dtype=object)), strict=True))
                            ).hexdigest(),
                            "n_buildings": len(table), "primary_strips": primary_strips,
                            "run": run.ref, "config_sha256": self.cfg.sha256}
                atomic_write_json(run.path / f"change_map_buildings_{n}.json", meta_doc)
                per_inv[n]["hashes"] = {k: meta_doc[k] for k in
                                        ("change_map_sha256", "inventory_sha256",
                                         "buildings_sha256")}
                pts = gpd.GeoDataFrame(
                    {"building_id": table["building_id"], "class": table["class"],
                     "class_all_strips": table["class_all_strips"]},
                    geometry=table.geometry.centroid, crs=table.crs).to_crs("EPSG:4326")
                atomic_write_text(run.path / f"change_map_points_{n}.geojson",
                                  pts.to_json(drop_id=True))
                if n in self.v["buildings"].get("audit_inventories", []):
                    self.outputs += [f"change_map_buildings_{n}.parquet",
                                     f"change_map_buildings_{n}.json",
                                     f"change_map_points_{n}.geojson"]
            summary["by_inventory"] = per_inv
            pd.DataFrame([{"inventory": n, **{k: v for k, v in s.items()
                                              if not isinstance(v, dict)},
                           **{f"all_strips_{k}": v for k, v in
                              s["sensitivity_all_strips"].items()}}
                          for n, s in per_inv.items()]).to_csv(
                run.path / "change_map_by_inventory.csv", index=False)
        summary["timings_s"] = self.timings
        summary["primary_inventory"] = self.v["buildings"]["primary_inventory"]
        summary["audit_inventories"] = self.v["buildings"].get("audit_inventories", [])
        run.write_summary(summary)
        return run, summary

    def output_table(self, name: str, per_strip: dict[str, pd.DataFrame],
                     scored: pd.DataFrame, sens: pd.DataFrame,
                     primary_strips: list[str]) -> gpd.GeoDataFrame:
        base = self.inv[name]
        cols: dict[str, Any] = {"building_id": base["building_id"].to_numpy(),
                                "inventory": name}
        for c in ("source_id", "confidence"):
            if c in base.columns:
                cols[c] = base[c].to_numpy()
        for c in ("score", "class", "observable", "strip", "damage_score", "damage_class"):
            cols[c] = scored[c].to_numpy()
        for c in ("score", "class", "observable", "strip"):
            cols[f"{c}_all_strips"] = sens[c].to_numpy()
        reasons = np.full(len(base), "", dtype=object)
        for s in self.strips:
            r = per_strip[s]["reason"].to_numpy(dtype=object)
            reasons = np.where(reasons == "", r, reasons)
        gated = sens["observable"].to_numpy() & ~scored["observable"].to_numpy() & np.isin(
            sens["strip"].to_numpy(dtype=object), [s for s in self.strips
                                                  if s not in primary_strips])
        reasons = np.where(gated, "strip failed the co-registration gate", reasons)
        cols["unobservable_reason"] = np.where(scored["observable"].to_numpy(), "",
                                               np.where(reasons == "", "missing feature",
                                                        reasons))
        strip = scored["strip"].to_numpy(dtype=object)
        feature_cols = ["n_ring_px", "n_roof_px",
                        *[f"{p}_d_{f}" for p in ("ring", "roof") for f in F.PIXEL_FEATURES]]
        for c in feature_cols:
            values = np.full(len(base), np.nan)
            for s in self.strips:
                sel = strip == s
                values[sel] = per_strip[s][c].to_numpy(dtype=float)[sel]
            cols[c] = values
        for k in F.S2_FEATURES:
            cols[f"s2_d_{k}"] = per_strip[self.strips[0]][f"s2_d_{k}"].to_numpy()
        for c in scored.columns:
            if c.startswith("z_"):
                cols[c] = scored[c].to_numpy()
        for s in self.strips:
            cols[f"observable_{s}"] = per_strip[s]["observable"].to_numpy(dtype=bool)
        return gpd.GeoDataFrame(cols, geometry=base.geometry.values, crs=base.crs)

    @staticmethod
    def inventory_summary(table: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
        cls = table["class"].value_counts().to_dict()
        obs = table["observable"].to_numpy(dtype=bool)
        out: dict[str, Any] = {
            "n_buildings": len(table),
            "n_observable": int(obs.sum()),
            "observable_share": float(obs.mean()) if len(table) else float("nan"),
            "n_positive": int(cls.get("positive", 0)),
            "n_negative": int(cls.get("negative", 0)),
            "n_not_observable": int(cls.get("not_observable", 0)),
            "positive_share_of_observable": float(cls.get("positive", 0) / obs.sum())
            if obs.sum() else float("nan"),
            "n_damage_positive": int((table["damage_class"] == "positive").sum()),
            "n_strip_0600": int((table["strip"] == "0600").sum()),
            "n_strip_0500": int((table["strip"] == "0500").sum()),
            "n_s2_missing_observable": int((obs & table["s2_d_mndwi"].isna()).sum()),
            "flood_boundary": meta["flood"]["boundary"],
            "damage_boundary": meta["damage"]["boundary"],
        }
        out["unobservable_reasons"] = table.loc[~obs, "unobservable_reason"].value_counts() \
            .to_dict()
        out["flood_mixture"] = meta["flood"]["mixture"]
        out["damage_mixture"] = meta["damage"]["mixture"]
        return out


ARTIFACTS = {"coregistration.csv": "flood/coregistration.csv",
             "observable_mask_stats.json": "flood/observable_mask_stats.json",
             "change_map_by_inventory.csv": "flood/change_map_by_inventory.csv"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


BY_INVENTORY = "change_map_by_inventory.csv"
BY_INVENTORY_MERGED = "change_map_by_inventory_merged.csv"


def merge_by_inventory(new: pd.DataFrame, old: pd.DataFrame | None) -> pd.DataFrame:
    """The promoted per-inventory table after a run: rows of ``new`` replace the rows of the
    same ``inventory`` in ``old``, other rows of ``old`` are kept (in their order), new
    inventories are appended. A run over one inventory no longer drops the others."""
    if old is None or old.empty or "inventory" not in old.columns:
        return new.reset_index(drop=True)
    ids = set(new["inventory"].astype(str))
    kept = old.loc[~old["inventory"].astype(str).isin(ids)]
    order = list(dict.fromkeys([*old["inventory"].astype(str), *new["inventory"].astype(str)]))
    both = pd.concat([kept, new], ignore_index=True)
    both["_order"] = both["inventory"].astype(str).map({k: i for i, k in enumerate(order)})
    return both.sort_values("_order", kind="stable").drop(columns="_order").reset_index(
        drop=True)


def run_change(project: Project, config: str = CHANGE_CONFIG, *,
               inventories: list[str] | None = None,
               promote_outputs: bool = True) -> tuple[Run, dict[str, Any]]:
    """Build the change map; promote the shared outputs and, per audit inventory that exists
    (``buildings.audit_inventories``), its table, hash sidecar and points. The promoted
    ``change_map_by_inventory.csv`` is merged by inventory id (:func:`merge_by_inventory`);
    the run keeps its own rows in ``change_map_by_inventory.csv`` and the promoted table in
    ``change_map_by_inventory_merged.csv``."""
    cm = ChangeMap(project, config, inventories)
    run, summary = cm.execute()
    if promote_outputs:
        for src, dest in ARTIFACTS.items():
            if src == BY_INVENTORY:
                current = project.paths.artifacts / dest
                old = pd.read_csv(current) if current.is_file() else None
                merged = merge_by_inventory(pd.read_csv(run.path / src), old)
                merged.to_csv(run.path / BY_INVENTORY_MERGED, index=False)
                src = BY_INVENTORY_MERGED
            promote(run.path / src, dest, project.paths, replace=True)
        for src in cm.outputs:
            promote(run.path / src, f"flood/{src}", project.paths, replace=True)
    return run, summary
