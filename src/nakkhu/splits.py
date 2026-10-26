"""Spatial-block split of the plan 003 train grid (plan 005 Step 1, Option A).

Rule (``configs/splits/split.yaml``):

- **test**: every tile overlapping the 13.34 km2 analysis extent (the whole extent is held out);
- **val**: whole 3 x 3-tile blocks (960 m) in the training region, drawn with a fixed seed and
  stratified by Microsoft building density x distance to the nearest OSM river line, from the
  blocks that hold no test or test-buffer tile and whose nine tiles are fully valid;
- **buffer**: every other tile closer than 320 m to a test or val tile (never trained on, never
  scored);
- **train**: the rest. Train tiles below ``min_valid_frac`` stay ``train`` but ``usable: false``.

The split file ``configs/splits/split_v1.yaml`` holds no timestamps, so a rebuild from the same
inputs is byte-identical (``nakkhu split build --check``). Its SHA-256 goes into
``configs/prereg.yaml`` and every eval run. The plan 003 tile index is not modified.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

from nakkhu.config import Project, ResolvedConfig, config_sha256, load_yaml

SPLIT_RULE = "configs/splits/split.yaml"
SPLIT_FILE = "configs/splits/split_v1.yaml"
SPLITS = ("train", "val", "test", "buffer")
ARTIFACT_DIR = "splits"
WORK_CRS = "EPSG:32645"


class SplitError(RuntimeError):
    """The split cannot be built or would change after being frozen."""


# --- geometry helpers -----------------------------------------------------------------------


def tile_boxes(index: pd.DataFrame) -> np.ndarray:
    return np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)


def block_ids(cols: np.ndarray, rows: np.ndarray, block_tiles: int) -> np.ndarray:
    bc = np.asarray(cols, dtype=np.int64) // block_tiles
    br = np.asarray(rows, dtype=np.int64) // block_tiles
    return np.array([f"b_{c:02d}_{r:02d}" for c, r in zip(bc, br, strict=True)], dtype=object)


def near(boxes: np.ndarray, member: np.ndarray, distance_m: float) -> np.ndarray:
    """Non-member tiles whose footprint is closer than ``distance_m`` to a member tile."""
    member = np.asarray(member, dtype=bool)
    if not member.any():
        return np.zeros(len(boxes), dtype=bool)
    union = shapely.union_all(boxes[member])
    return (~member) & (shapely.distance(boxes, union) < distance_m)


def read_lines(paths: Sequence[str | Path], crs: str = WORK_CRS) -> np.ndarray:
    """Line geometries of GeoJSON or raw Overpass JSON files, reprojected to ``crs``."""
    geoms: list[Any] = []
    for path in paths:
        path = Path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        if "elements" in document:
            lines = [LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
                     for e in document["elements"]
                     if e.get("type") == "way" and len(e.get("geometry") or []) >= 2]
            frame = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        else:
            frame = gpd.read_file(path)
        geoms.extend(frame.to_crs(crs).geometry.values)
    return np.array([g for g in geoms if g is not None and not g.is_empty], dtype=object)


def centroid_counts(index: pd.DataFrame, geoms: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Polygons per tile, by centroid (first containing tile wins on shared edges)."""
    out = np.zeros(len(index), dtype=np.int64)
    if len(geoms) == 0:
        return out
    if mask is not None:
        geoms = np.asarray(geoms, dtype=object)[np.asarray(mask, dtype=bool)]
    ci, ti = shapely.STRtree(tile_boxes(index)).query(shapely.centroid(geoms),
                                                      predicate="intersects")
    first = pd.DataFrame({"c": ci, "t": ti}).sort_values(["c", "t"]).drop_duplicates("c")
    np.add.at(out, first["t"].to_numpy(), 1)
    return out


# --- the split rule -------------------------------------------------------------------------


def _quantile_bins(values: pd.Series, q: int) -> np.ndarray:
    """Equal-count bins 0..q-1 by rank (ties broken by the series order, which is sorted)."""
    ranks = values.rank(method="first").to_numpy() - 1
    return np.floor(ranks * q / len(values)).astype(int)


def allocate(sizes: dict[Any, int], total: int) -> dict[Any, int]:
    """Proportional allocation with largest remainder (exact integer arithmetic; ties by
    sorted stratum key)."""
    n = sum(sizes.values())
    out = {k: (total * v) // n for k, v in sizes.items()}
    rest = {k: (total * v) % n for k, v in sizes.items()}
    left = total - sum(out.values())
    order = sorted(sizes, key=lambda k: (-rest[k], k))
    for k in order[:left]:
        out[k] += 1
    return out


def select_val_blocks(blocks: pd.DataFrame, fraction: float, seed: int, density_q: int,
                      distance_q: int) -> pd.DataFrame:
    """Stratified seeded draw of val blocks among ``blocks.eligible``.

    ``blocks``: ``block_id, eligible, density, distance_m``. Returns the eligible blocks with
    ``stratum`` and ``val`` columns, sorted by block id.
    """
    elig = blocks.loc[blocks["eligible"]].sort_values("block_id").reset_index(drop=True)
    if elig.empty:
        raise SplitError("no eligible val blocks")
    elig["density_bin"] = _quantile_bins(elig["density"], density_q)
    elig["distance_bin"] = _quantile_bins(elig["distance_m"], distance_q)
    elig["stratum"] = [f"d{a}_r{b}" for a, b in zip(elig["density_bin"], elig["distance_bin"],
                                                    strict=True)]
    total = int(round(fraction * len(elig)))
    sizes = elig.groupby("stratum").size().to_dict()
    quota = allocate(sizes, total)
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for stratum in sorted(quota):
        ids = sorted(elig.loc[elig["stratum"] == stratum, "block_id"])
        if quota[stratum]:
            chosen.update(rng.choice(ids, size=quota[stratum], replace=False).tolist())
    elig["val"] = elig["block_id"].isin(chosen)
    return elig


def assign(index: pd.DataFrame, extent: Any, rule: dict[str, Any], ms_count: np.ndarray,
           rivers: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tile split table and block table from the plan 003 index and the rule."""
    index = index.sort_values("tile_id").reset_index(drop=True)
    k = int(rule["block_tiles"])
    buffer_m = float(rule["buffer_m"])
    boxes = tile_boxes(index)
    test = shapely.area(shapely.intersection(boxes, extent)) > 0
    test_buffer = near(boxes, test, buffer_m)
    frame = pd.DataFrame({
        "tile_id": index["tile_id"].to_numpy(), "col": index["col"].to_numpy(),
        "row": index["row"].to_numpy(),
        "block_id": block_ids(index["col"].to_numpy(), index["row"].to_numpy(), k),
        "valid_frac": index["valid_frac"].to_numpy(dtype=float),
        "ms_count": np.asarray(ms_count, dtype=np.int64),
    })
    frame["held"] = test | test_buffer
    grouped = frame.groupby("block_id")
    blocks = pd.DataFrame({
        "n_tiles": grouped.size(),
        "held": grouped["held"].any(),
        "full": grouped["valid_frac"].apply(lambda v: int((v >= 1.0).sum())),
        "ms_count": grouped["ms_count"].sum(),
        "valid_km2": grouped["valid_frac"].sum() * (boxes[0].area / 1e6),
    }).reset_index()
    geoms = [shapely.union_all(boxes[(frame["block_id"] == b).to_numpy()])
             for b in blocks["block_id"]]
    blocks["geometry"] = geoms
    blocks["density"] = blocks["ms_count"] / blocks["valid_km2"].where(blocks["valid_km2"] > 0)
    centres = shapely.centroid(np.array(geoms, dtype=object))
    river_union = shapely.union_all(rivers) if len(rivers) else None
    blocks["distance_m"] = (shapely.distance(centres, river_union) if river_union is not None
                            else np.nan)
    blocks["eligible"] = (~blocks["held"]) & (blocks["n_tiles"] == k * k) & (blocks["full"]
                                                                             == k * k)
    val_cfg = rule["val"]
    chosen = select_val_blocks(blocks, float(val_cfg["fraction_of_eligible_blocks"]),
                               int(val_cfg["seed"]), int(val_cfg["density_quantiles"]),
                               int(val_cfg["distance_quantiles"]))
    blocks = blocks.merge(chosen[["block_id", "stratum", "val"]], on="block_id", how="left")
    blocks["val"] = blocks["val"].astype("boolean").fillna(False).astype(bool)
    val_blocks = set(blocks.loc[blocks["val"], "block_id"])
    val = frame["block_id"].isin(val_blocks).to_numpy()
    val_buffer = near(boxes, val | test, buffer_m) & ~test & ~val
    val_near = near(boxes, val, buffer_m)
    split = np.where(test, "test", np.where(val, "val",
                     np.where(test_buffer | val_buffer, "buffer", "train")))
    reason = np.where(split != "buffer", "",
                      np.where(test_buffer & val_near, "test+val",
                               np.where(test_buffer, "test", "val")))
    frame["split"] = split
    frame["buffer_of"] = reason
    frame["usable"] = frame["valid_frac"] >= float(rule["min_valid_frac"])
    frame = frame.drop(columns=["held"])
    counts = frame.groupby(["block_id", "split"]).size().unstack(fill_value=0)
    for name in SPLITS:
        blocks[f"n_{name}"] = blocks["block_id"].map(counts[name] if name in counts else {}
                                                     ).fillna(0).astype(int)
    blocks["role"] = np.select(
        [blocks["val"], blocks["n_test"] > 0, blocks["n_train"] == blocks["n_tiles"]],
        ["val", "test", "train"], default="mixed")
    return frame, blocks


def check_separation(frame: pd.DataFrame, boxes: np.ndarray, buffer_m: float) -> dict[str, Any]:
    """Verify: no train tile within ``buffer_m`` of a test or val tile; disjoint and complete."""
    split = frame["split"].to_numpy()
    train = split == "train"
    out: dict[str, Any] = {}
    for other in ("test", "val"):
        member = split == other
        if member.any():
            d = shapely.distance(boxes[train], shapely.union_all(boxes[member]))
            out[f"min_train_to_{other}_m"] = float(d.min()) if len(d) else None
        else:
            out[f"min_train_to_{other}_m"] = None
    out["unique_tile_ids"] = bool(frame["tile_id"].is_unique)
    out["every_tile_assigned"] = bool(frame["split"].isin(SPLITS).all())
    out["separation_ok"] = all(v is None or v >= buffer_m for k, v in out.items()
                               if k.startswith("min_train"))
    return out


def split_yaml(frame: pd.DataFrame, blocks: pd.DataFrame, header: dict[str, Any]) -> str:
    """Deterministic text of ``split_v1.yaml`` (no timestamps)."""
    lines = [
        "# Generated by `nakkhu split build` from configs/splits/split.yaml (plan 005).",
        "# Do not edit: its SHA-256 is frozen in configs/prereg.yaml.",
        f"schema_version: {header['schema_version']}",
        f"version: {header['version']}",
        f"rule_sha256: {header['rule_sha256']}",
        f"grid_sha256: {header['grid_sha256']}",
        f"tile_index: {header['tile_index']}",
        f"block_tiles: {header['block_tiles']}",
        f"buffer_m: {header['buffer_m']}",
        "val_blocks: [" + ", ".join(sorted(blocks.loc[blocks["val"], "block_id"])) + "]",
        "tiles:",
    ]
    for row in frame.sort_values("tile_id").to_dict("records"):
        extra = f", buffer_of: {row['buffer_of']}" if row["split"] == "buffer" else ""
        lines.append(f"  {row['tile_id']}: {{split: {row['split']}, block: {row['block_id']}, "
                     f"usable: {'true' if row['usable'] else 'false'}{extra}}}")
    return "\n".join(lines) + "\n"


# --- loading the frozen split ---------------------------------------------------------------


def load_split(project: Project, path: str = SPLIT_FILE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``(tile table, header)`` of a split file: columns ``tile_id, split, block_id, usable,
    buffer_of``."""
    values = load_yaml(project.paths.resolve(path))
    tiles = values.pop("tiles")
    frame = pd.DataFrame([{"tile_id": t, "split": v["split"], "block_id": v["block"],
                           "usable": bool(v["usable"]), "buffer_of": v.get("buffer_of", "")}
                          for t, v in sorted(tiles.items())])
    return frame, values


def split_tile_ids(project: Project, split: str, usable_only: bool = False) -> list[str]:
    frame, _ = load_split(project)
    sel = frame["split"] == split
    if usable_only:
        sel &= frame["usable"]
    return sorted(frame.loc[sel, "tile_id"])


# --- summary --------------------------------------------------------------------------------


def _q(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    qs = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {k: float(v) for k, v in zip(("q05", "q25", "q50", "q75", "q95"), qs, strict=True)}


def summarise(frame: pd.DataFrame, blocks: pd.DataFrame, extra: pd.DataFrame,
              checks: dict[str, Any], provisional_domain: pd.Series) -> dict[str, Any]:
    """Computed per-split counts; ``extra`` holds per-tile ``google, google_t90, microsoft,
    distance_m, osm_reg_ok`` aligned with ``frame``."""
    tile_km2 = 0.1024
    out: dict[str, Any] = {"checks": checks, "by_split": {}, "val_blocks": [],
                           "blocks": {"total": int(len(blocks)),
                                      "eligible_for_val": int(blocks["eligible"].sum()),
                                      "by_role": blocks["role"].value_counts().sort_index()
                                      .to_dict()}}
    for name in SPLITS:
        sel = (frame["split"] == name).to_numpy()
        part = extra.loc[sel]
        valid_km2 = frame.loc[sel, "valid_frac"].to_numpy() * tile_km2
        density = np.divide(part["microsoft"].to_numpy(dtype=float), valid_km2,
                            out=np.full(len(part), np.nan), where=valid_km2 > 0)
        record: dict[str, Any] = {
            "tiles": int(sel.sum()),
            "usable_tiles": int((sel & frame["usable"].to_numpy()).sum()),
            "blocks": int(frame.loc[sel, "block_id"].nunique()),
            "valid_km2": float(valid_km2.sum()),
            "instances_by_centroid": {k: int(part[k].sum())
                                      for k in ("google", "google_t90", "microsoft")},
            "microsoft_per_km2_tile": _q(density),
            "distance_to_river_m_tile": _q(part["distance_m"].to_numpy()),
        }
        if "osm_reg_ok" in part and part["osm_reg_ok"].notna().any():
            record["osm_registered_tiles"] = int(part["osm_reg_ok"].fillna(False).sum())
        if name == "buffer":
            record["buffer_of"] = frame.loc[sel, "buffer_of"].value_counts().sort_index().to_dict()
        out["by_split"][name] = record
    for row in blocks.loc[blocks["val"]].sort_values("block_id").itertuples(index=False):
        out["val_blocks"].append({"block_id": row.block_id, "stratum": row.stratum,
                                  "microsoft_per_km2": float(row.density),
                                  "distance_m": float(row.distance_m)})
    strata = blocks.loc[blocks["eligible"]].groupby("stratum")
    out["strata"] = {s: {"eligible": int(len(p)), "val": int(p["val"].sum())}
                     for s, p in strata}
    cross = pd.crosstab(provisional_domain.to_numpy(), frame["split"].to_numpy())
    out["vs_plan003_domain"] = {str(d): {str(s): int(v) for s, v in row.items() if v}
                                for d, row in cross.iterrows()}
    return out


def blocks_frame(blocks: pd.DataFrame) -> gpd.GeoDataFrame:
    cols = ["block_id", "role", "n_tiles", *(f"n_{s}" for s in SPLITS), "eligible", "stratum",
            "ms_count", "density", "distance_m"]
    frame = gpd.GeoDataFrame(blocks[cols].copy(), geometry=list(blocks["geometry"]),
                             crs=WORK_CRS)
    frame["stratum"] = frame["stratum"].fillna("")
    return gpd.GeoDataFrame(frame.sort_values("block_id").reset_index(drop=True),
                            geometry="geometry", crs=WORK_CRS)


# --- stage entry point ----------------------------------------------------------------------


def _inputs(project: Project, rule: dict[str, Any]) -> dict[str, Any]:
    from nakkhu.acquire import geo
    from nakkhu.labels.harmonise import load_footprints
    from nakkhu.tiling import read_index, tiles_dir

    index_path = tiles_dir(project, rule["tiles"]["scene"], rule["tiles"]["purpose"]) / \
        "index.parquet"
    if not index_path.is_file():
        raise SplitError(f"{index_path} not found (plan 003 `nakkhu tile`)")
    index, meta = read_index(index_path)
    index = index.sort_values("tile_id").reset_index(drop=True)
    extent = geo.extent(project)
    rivers = read_lines([project.paths.resolve(p) for p in rule["rivers"]])
    labels_cfg = load_yaml(project.paths.resolve("configs/labels/common.yaml"))
    clip = shapely.box(*shapely.total_bounds(tile_boxes(index)))
    fps, _ = load_footprints(project, labels_cfg, clip)
    g = fps.google
    extra = pd.DataFrame({
        "google": centroid_counts(index, g.geometry.values),
        "google_t90": centroid_counts(index, g.geometry.values,
                                      (g["confidence"] >= g["t90"]).to_numpy()),
        "microsoft": centroid_counts(index, fps.microsoft.geometry.values),
    })
    centres = shapely.centroid(tile_boxes(index))
    extra["distance_m"] = shapely.distance(centres, shapely.union_all(rivers))
    return {"index": index, "meta": meta, "index_path": index_path, "extent": extent,
            "rivers": rivers, "extra": extra}


def osm_registration(project: Project) -> pd.DataFrame | None:
    """Per-tile OSM registration flags if plan 004 has registered OSM (informational only)."""
    from nakkhu.labelqa import osm_shift_path

    path = osm_shift_path(project)
    return pd.read_parquet(path, columns=["tile_id", "reg_ok"]) if path.is_file() else None


def build(project: Project, *, check: bool = False, replace: bool = False) -> dict[str, Any]:
    """Build the split in a run; write ``split_v1.yaml`` (or compare with it under ``check``)
    and promote ``split_summary.json`` and ``blocks.geojson``."""
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json, atomic_write_text, create_run

    rule = load_yaml(project.paths.resolve(SPLIT_RULE))
    run = create_run(project, "splits", ResolvedConfig(
        values={"command": "split-build", "check": check, "rule": rule}, source=SPLIT_RULE))
    with run.step("split"):
        inputs = _inputs(project, rule)
        index = inputs["index"]
        frame, blocks = assign(index, inputs["extent"], rule,
                               inputs["extra"]["microsoft"].to_numpy(), inputs["rivers"])
        checks = check_separation(frame, tile_boxes(index), float(rule["buffer_m"]))
        if not checks["separation_ok"] or not checks["unique_tile_ids"]:
            raise SplitError(f"split checks failed: {checks}")
        header = {"schema_version": 1, "version": rule["version"],
                  "rule_sha256": config_sha256(rule), "grid_sha256": inputs["meta"]["grid_sha256"],
                  "tile_index": project.paths.relative(inputs["index_path"]),
                  "block_tiles": int(rule["block_tiles"]), "buffer_m": float(rule["buffer_m"])}
        text = split_yaml(frame, blocks, header)
        atomic_write_text(run.path / "split_v1.yaml", text)
        extra = inputs["extra"].copy()
        reg = osm_registration(project)
        if reg is not None:
            extra["osm_reg_ok"] = frame[["tile_id"]].merge(reg, on="tile_id", how="left")[
                "reg_ok"].to_numpy()
        summary = summarise(frame, blocks, extra, checks, index["domain"])
        summary["split_file"] = SPLIT_FILE
        summary["split_sha256"] = _sha256_text(text)
        summary["rule_sha256"] = header["rule_sha256"]
        summary["grid_sha256"] = header["grid_sha256"]
        summary["osm_registration_read"] = reg is not None
        atomic_write_json(run.path / "split_summary.json", summary)
        blocks_frame(blocks).to_crs("EPSG:4326").to_file(run.path / "blocks.geojson",
                                                          driver="GeoJSON")
    target = project.paths.resolve(SPLIT_FILE)
    result: dict[str, Any] = {"run": run.ref, "split_sha256": summary["split_sha256"],
                              "by_split": {k: v["tiles"] for k, v in summary["by_split"].items()},
                              "checks": checks}
    if check:
        result["identical_to_frozen"] = target.is_file() and target.read_bytes() == text.encode()
        run.write_summary(result)
        return result
    if target.is_file() and target.read_bytes() != text.encode() and not replace:
        raise SplitError(f"{SPLIT_FILE} exists and differs; the split is frozen (pass --replace "
                         "only if no test tile has been scored and prereg is re-frozen)")
    atomic_write_text(target, text)
    for name in ("split_summary.json", "blocks.geojson"):
        promote(run.path / name, f"{ARTIFACT_DIR}/{name}", project.paths, replace=True)
    result["written"] = SPLIT_FILE
    run.write_summary(result)
    return result


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(project: Project, path: str = SPLIT_FILE) -> str:
    import hashlib

    return hashlib.sha256(project.paths.resolve(path).read_bytes()).hexdigest()
