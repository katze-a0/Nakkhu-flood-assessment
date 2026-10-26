"""Local audit chips (plan 006 Step 2), written only under ``{data_root}/interim/audit/``.

A chip is ``context_m`` x ``context_m`` of the 0.305 m ``visual`` source resampled to
``upscale`` x its native pixels (bilinear), with a crosshair with an open centre for points, or a
thin white-under-black double outline for boxes and footprints. ``flood_trace`` chips put the
pre (WV02) and post (GE01, first strip with valid pixels) panels side by side at identical
geometry; a ``scenes.oam`` scene would add a third panel where it covers the building (empty
since prereg v9: the OAM Dukuchhap mosaics are pre-flood imagery, plans/oam_panel_check.md).
A panel is valid at the building only where the raster mask is valid and the pixels are not
near-black (no-data stored as 0 under an all-valid mask). ``recall_nearmiss`` chips are the
sitting-1 view: overlap items get the identical crosshair chip, near-miss items a circle of radius
``radius_m`` (white under black) around a small centre mark. Maxar pixels never leave this
directory: nothing here writes to ``artifacts/``, ``figures/`` or the manuscript.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from nakkhu.audit.core import P2_TASK, AuditError, chip_name
from nakkhu.config import Project

NATIVE_M = 0.30517578125


@dataclass(frozen=True)
class ChipStyle:
    context_m: float = 64.0
    upscale: int = 2
    quality: int = 90
    gap_px: int = 7
    arm_px: int = 26

    @property
    def size_px(self) -> int:
        return int(round(self.context_m / NATIVE_M)) * self.upscale

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ChipStyle:
        c = cfg.get("chips") or {}
        return cls(context_m=float(c.get("context_m", 64.0)), upscale=int(c.get("upscale", 2)),
                   quality=int(c.get("jpeg_quality", 90)),
                   gap_px=int(c.get("crosshair_gap_px", 7)),
                   arm_px=int(c.get("crosshair_arm_px", 26)))


def scene_raster(project: Project, scene_id: str) -> Path:
    """The scene's visual VRT (or its single GeoTIFF) from the data manifest."""
    from nakkhu.audit.frames import manifest_entries

    entry = manifest_entries(project).get(scene_id)
    if entry is None or entry.get("status") == "pending":
        raise AuditError(f"{scene_id} is missing or pending in the data manifest")
    base = project.paths.resolve(entry["local_path"])
    found = sorted(base.glob("*.vrt")) or sorted(base.glob("*.tif"))
    if not found:
        raise AuditError(f"no VRT or GeoTIFF under {base}")
    return found[0]


class Reader:
    """Open rasters once; read north-up RGB windows at a fixed output size."""

    def __init__(self) -> None:
        self._open: dict[Path, Any] = {}

    def dataset(self, path: Path) -> Any:
        if path not in self._open:
            self._open[path] = rasterio.open(path)
        return self._open[path]

    def read(self, path: Path, bounds: tuple[float, float, float, float], size: int
             ) -> np.ndarray:
        return self.read_valid(path, bounds, size)[0]

    def read_valid(self, path: Path, bounds: tuple[float, float, float, float], size: int
                   ) -> tuple[np.ndarray, np.ndarray]:
        """RGB window and its validity: the raster mask (every band valid; outside the raster
        is invalid) and not near-black (:func:`valid_pixels`)."""
        ds = self.dataset(path)
        window = from_bounds(*bounds, transform=ds.transform)
        # A window inside the raster is read without `boundless`: rasterio's boundless
        # read_masks returned an all-invalid mask for some interior windows (seen in tests on a
        # plain GeoTIFF), which made a covered building look outside its strip.
        inside = (window.col_off >= 0 and window.row_off >= 0
                  and window.col_off + window.width <= ds.width
                  and window.row_off + window.height <= ds.height)
        data = ds.read([1, 2, 3], window=window, out_shape=(3, size, size),
                       boundless=not inside, fill_value=0, resampling=Resampling.bilinear)
        masks = ds.read_masks([1, 2, 3], window=window, out_shape=(3, size, size),
                              boundless=not inside, resampling=Resampling.nearest)
        rgb = np.moveaxis(data, 0, -1).astype(np.uint8)
        return rgb, valid_pixels(rgb, (masks > 0).all(axis=0))

    def close(self) -> None:
        for ds in self._open.values():
            ds.close()
        self._open.clear()


def chip_bounds(x: float, y: float, style: ChipStyle) -> tuple[float, float, float, float]:
    half = style.size_px / style.upscale * NATIVE_M / 2.0
    return x - half, y - half, x + half, y + half


def to_px(xy: np.ndarray, bounds: tuple[float, float, float, float], size: int) -> np.ndarray:
    west, _, east, north = bounds
    scale = size / (east - west)
    return np.column_stack([(xy[:, 0] - west) * scale, (north - xy[:, 1]) * scale])


def draw_crosshair(img: Image.Image, cx: float, cy: float, style: ChipStyle) -> None:
    d = ImageDraw.Draw(img)
    g, a = style.gap_px, style.gap_px + style.arm_px
    arms = [((cx - a, cy), (cx - g, cy)), ((cx + g, cy), (cx + a, cy)),
            ((cx, cy - a), (cx, cy - g)), ((cx, cy + g), (cx, cy + a))]
    for width, colour in ((5, (255, 255, 255)), (2, (0, 0, 0))):
        for p, q in arms:
            d.line([p, q], fill=colour, width=width)


def draw_circle(img: Image.Image, cx: float, cy: float, r_px: float) -> None:
    """Circle of radius ``r_px`` centred on (cx, cy): a white ring under a thin black one."""
    d = ImageDraw.Draw(img)
    for width, colour in ((4, (255, 255, 255)), (2, (0, 0, 0))):
        d.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=colour, width=width)


def nearmiss_style(style: ChipStyle, tcfg: dict[str, Any]) -> ChipStyle:
    """Centre mark of the near-miss chip (smaller than the crosshair, so roofs inside the circle
    stay visible)."""
    from dataclasses import replace

    return replace(style, gap_px=int(tcfg.get("mark_gap_px", 3)),
                   arm_px=int(tcfg.get("mark_arm_px", 8)))


def metres_per_px(style: ChipStyle) -> float:
    return NATIVE_M / style.upscale


def draw_outline(img: Image.Image, geom: Any, bounds: tuple[float, float, float, float],
                 size: int) -> None:
    d = ImageDraw.Draw(img)
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for width, colour in ((4, (255, 255, 255)), (1, (0, 0, 0))):
        for poly in polys:
            ring = to_px(np.asarray(poly.exterior.coords), bounds, size)
            d.line([tuple(p) for p in ring], fill=colour, width=width, joint="curve")


# A pixel whose brightest band is at most this is no-data, not image: no-data stored as 0 under
# an all-valid mask (the OAM mosaics) reads as 0-1 after the bilinear resampling.
DARK_MAX = 4


def valid_pixels(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean (h, w): inside the raster mask (if given) and brighter than ``DARK_MAX``."""
    valid = rgb.max(axis=-1) > DARK_MAX
    return valid if mask is None else valid & mask.astype(bool)


def valid_share(rgb: np.ndarray, centre: tuple[float, float], radius: int = 12,
                valid: np.ndarray | None = None) -> float:
    """Share of valid pixels in a (2 radius)^2 patch at ``centre``; ``valid`` is the per-pixel
    validity (:meth:`Reader.read_valid`), else it is taken from ``rgb`` alone."""
    if valid is None:
        valid = valid_pixels(rgb)
    cx, cy = int(centre[0]), int(centre[1])
    patch = valid[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
    return float(patch.mean()) if patch.size else 0.0


def render_single(reader: Reader, raster: Path, x: float, y: float, style: ChipStyle,
                  geom: Any | None = None, circle_m: float | None = None
                  ) -> tuple[Image.Image, float]:
    bounds = chip_bounds(x, y, style)
    rgb, valid = reader.read_valid(raster, bounds, style.size_px)
    img = Image.fromarray(rgb, "RGB")
    centre = to_px(np.array([[x, y]]), bounds, style.size_px)[0]
    if circle_m is not None:
        draw_circle(img, float(centre[0]), float(centre[1]), circle_m / metres_per_px(style))
    if geom is None:
        draw_crosshair(img, float(centre[0]), float(centre[1]), style)
    else:
        draw_outline(img, geom, bounds, style.size_px)
    return img, valid_share(rgb, (float(centre[0]), float(centre[1])), valid=valid)


def render_pair(reader: Reader, panels: Sequence[tuple[str, Path]], x: float, y: float,
                style: ChipStyle, geom: Any) -> tuple[Image.Image, list[float]]:
    """Panels side by side with a caption strip (only the date label of each panel)."""
    tiles, shares = [], []
    for _, raster in panels:
        img, share = render_single(reader, raster, x, y, style, geom)
        tiles.append(img)
        shares.append(share)
    gutter, caption = 10, 26
    width = len(tiles) * style.size_px + (len(tiles) - 1) * gutter
    out = Image.new("RGB", (width, style.size_px + caption), (24, 24, 24))
    d = ImageDraw.Draw(out)
    for i, (img, (label, _)) in enumerate(zip(tiles, panels, strict=True)):
        x0 = i * (style.size_px + gutter)
        out.paste(img, (x0, caption))
        d.text((x0 + 6, 6), label, fill=(235, 235, 235))
    return out, shares


def write_jpeg(img: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{path.name}")
    img.save(tmp, "JPEG", quality=quality)
    tmp.replace(path)


def p2_rasters(project: Project, cfg: dict[str, Any]) -> dict[str, Any]:
    """Phase-2 audit scenes (``tasks.flood_trace_p2.scenes``): per site the pre strips in their
    order of preference, and the post strips."""
    tcfg = ((cfg.get("tasks") or {}).get(P2_TASK) or {})
    sc = tcfg.get("scenes") or {}
    pre = {str(site): [(str(s), scene_raster(project, str(s))) for s in scenes]
           for site, scenes in (sc.get("pre") or {}).items()}
    post = [(str(s), scene_raster(project, str(s))) for s in sc.get("post") or []]
    if not pre or not post:
        raise AuditError(f"tasks.{P2_TASK}.scenes needs `pre` (per site) and `post`")
    return {"pre": pre, "post": post}


def _first_valid(reader: Reader, panels: list[tuple[str, Path]], x: float, y: float,
                 style: ChipStyle, geom: Any) -> tuple[str, Path] | None:
    for sid, raster in panels:
        _, share = render_single(reader, raster, x, y, style, geom)
        if share >= 0.9:
            return sid, raster
    return None


def render_p2(reader: Reader, scenes: dict[str, Any], row: Any, style: ChipStyle, geom: Any,
              entry: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Phase-2 chip: the pre strip that holds valid pixels at the building (first in the site's
    order; the first strip if none does), then the first valid post strip, side by side with
    the v9 flood_trace captions."""
    site = str(row.site)
    if site not in scenes["pre"]:
        raise AuditError(f"no pre scene configured for site {site}")
    pre = _first_valid(reader, scenes["pre"][site], row.x, row.y, style, geom) \
        or scenes["pre"][site][0]
    post = _first_valid(reader, scenes["post"], row.x, row.y, style, geom)
    panels = [("Before: Jan 2024", pre[1])]
    if post is not None:
        panels.append(("After: 6 Oct 2024", post[1]))
    img, shares = render_pair(reader, panels, row.x, row.y, style, geom)
    return img, {**entry, "pre": pre[0], "post": post[0] if post else None, "valid": shares}


def render_task(project: Project, task: str, items: Any, cfg: dict[str, Any], out_dir: Path,
                *, force: bool = False) -> dict[str, Any]:
    """Render every item's chip into ``out_dir``; returns a manifest (kept next to the chips)."""
    root = (project.paths.data_root / "interim").resolve()
    if root not in out_dir.resolve().parents:
        raise AuditError(f"chips must be written under {root}, not {out_dir}")
    style = ChipStyle.from_config(cfg)
    mark = nearmiss_style(style, (cfg.get("tasks") or {}).get(task) or {})
    scenes = cfg.get("scenes") or {}
    reader = Reader()
    manifest: dict[str, Any] = {}
    try:
        pre = scene_raster(project, str(scenes["pre"])) if task != P2_TASK else None
        posts = [(s, scene_raster(project, str(s))) for s in scenes.get("post") or []] \
            if task == "flood_trace" else []
        p2_scenes = p2_rasters(project, cfg) if task == P2_TASK else {}
        oams = []
        if task == "flood_trace":
            for s in scenes.get("oam") or []:
                try:
                    oams.append((s, scene_raster(project, str(s))))
                except AuditError:
                    continue
        for row in items.itertuples(index=False):
            name = chip_name(task, str(row.item_id))
            path = out_dir / name
            point_task = task in ("recall_points", "recall_nearmiss")
            geom = None if point_task else shapely.from_wkb(row.wkb)
            entry: dict[str, Any] = {"file": name}
            if path.is_file() and not force:
                manifest[str(row.item_id)] = {**entry, "reused": True}
                continue
            if task == P2_TASK:
                img, entry = render_p2(reader, p2_scenes, row, style, geom, entry)
            elif task == "flood_trace":
                post = None
                for sid, raster in posts:
                    _, share = render_single(reader, raster, row.x, row.y, style, geom)
                    if share >= 0.9:
                        post = (sid, raster)
                        break
                panels = [("Before: Jan 2024", pre)]
                if post is not None:
                    panels.append(("After: 6 Oct 2024", post[1]))
                for sid, raster in oams:
                    _, share = render_single(reader, raster, row.x, row.y, style, geom)
                    if share >= 0.9:
                        panels.append(("Drone (date unverified)", raster))
                        entry["oam"] = sid
                        break
                img, shares = render_pair(reader, panels, row.x, row.y, style, geom)
                entry.update(post=post[0] if post else None, valid=shares)
            elif task == "recall_nearmiss" and getattr(row, "kind", "") == "nearmiss":
                img, share = render_single(reader, pre, row.x, row.y, mark,
                                           circle_m=float(row.radius_m))
                entry.update(valid=share, radius_m=float(row.radius_m))
            else:
                img, share = render_single(reader, pre, row.x, row.y, style, geom)
                entry["valid"] = share
            write_jpeg(img, path, style.quality)
            manifest[str(row.item_id)] = entry
    finally:
        reader.close()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True),
                                           encoding="utf-8")
    return manifest
