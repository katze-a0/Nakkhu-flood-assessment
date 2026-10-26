"""``nakkhu uncertainty run``: U1-U3 and D4 as ONE journalled test access (prereg v5).

Order (as for the plan 005/007 one-shot test commands): every precondition is checked first and
nothing is read from the test audit until all hold; then the access is journalled in
``runs/eval/test_journal.jsonl`` (:func:`nakkhu.evaluate.guard.record_test_access`), and only
then are the answers loaded and the analysis computed. Preconditions:

1. ``prereg verify`` passes (prereg v5 and every pin) and the prereg has the ``uncertainty``
   section;
2. the coordinator's hold is off: ``runs/eval/test_hold.yaml`` exists with ``hold: false`` and
   a non-empty ``by`` (fail-safe: a missing file means the hold is ON);
3. the journal has no earlier ``uncertainty/run`` entry (one shot);
4. every input exists (:meth:`FileInputs.problems`): sittings 1, 1b and 2 complete, the
   sitting-2 pool exactly as pre-registered, ``detector_final``, the L2 test runs, the HAND
   sweep, the plan 005 blocks and plan 009 ``by_inventory.csv``.

The computation (:func:`analyse`) is a pure function of an :class:`Inputs` bundle, so it is
tested end to end on synthetic fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import shapely

from nakkhu.config import Project, ResolvedConfig, load_yaml
from nakkhu.uncertainty import calibrate as C
from nakkhu.uncertainty import decompose as D
from nakkhu.uncertainty import ppi as P
from nakkhu.uncertainty import spec as S
from nakkhu.uncertainty.falsifiers import c2_outcome

FINAL = "detector_final"
# Count units of the estimators (plan 013, prereg v8 uncertainty_propagation.count_units):
# exposure is counted in real objects (polygons / boxes); totals over recall-frame points are
# upper bounds; a ratio of an object count to a point-level recall mixes units.
COUNT_UNIT = {"precision_only": "objects", "naive": "objects", "ensemble_expected": "objects",
              "ppi_pp": "frame_points", "ht": "frame_points", "ratio": "mixed"}
RUN_NAME = "uncertainty/run"
HOLD_FILE = "eval/test_hold.yaml"           # under runs/
STAGE = "uncertainty"
EDGES = (0.0, 3.0, 6.0, 9.0, 12.0)          # prereg hand.tiers
STAT_ALIAS = {"footprint_plus10m": "buffer_min"}   # sweep statistic name (buffer_m 10)
OUTPUTS = ("calibration_curve.csv", "ece.csv", "tier_counts_factorial.parquet",
           "variance_decomposition.csv", "ppi_tier_counts.csv", "coverage_check.json",
           "inventory_comparison.csv", "falsifiers.json")


class UncertaintyError(RuntimeError):
    """The run is refused or an input is malformed."""


# --- HAND provider ----------------------------------------------------------------------------


class HandProvider(Protocol):
    @property
    def edges(self) -> Sequence[float]: ...

    def surfaces(self) -> pd.DataFrame:
        """``cfg_id``, ``dem``, ``in_sensitivity_range``, ``is_primary``."""
        ...

    def values(self, geoms: np.ndarray, stats: Sequence[str],
               cfg_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """``{"<cfg_id>__<stat>": HAND per geometry}`` (sweep statistic names)."""
        ...


class SweepHand:
    """HAND from the latest complete plan 008 sweep, computed exactly as ``hand attribute``
    (centroid cell; minimum over footprint cells; minimum over the footprint + buffer_m)."""

    def __init__(self, project: Project) -> None:
        from nakkhu.hydro.sweep import latest_sweep, load_settings

        self.settings, _ = load_settings(project)
        sweep = latest_sweep(project)
        summary = json.loads((sweep.path / "summary.json").read_text(encoding="utf-8"))
        self.interim = project.paths.resolve(summary["interim"])
        grid = pd.read_csv(sweep.path / "config_grid.csv")
        self.grid_table = grid[grid["status"] == "done"].reset_index(drop=True)
        self.primary = self.settings.primary().cfg_id
        self.grid = self.settings.grid()
        self.edges = tuple(float(e) for e in self.settings.edges)
        self.buffer_m = float(self.settings.v["buffer_m"])
        if abs(self.buffer_m - 10.0) > 1e-9:
            raise UncertaintyError(f"sweep buffer_m is {self.buffer_m}, the spec's "
                                   "footprint_plus10m needs 10 m")
        self._cache: dict[str, np.ndarray] = {}

    def surfaces(self) -> pd.DataFrame:
        g = self.grid_table
        return pd.DataFrame({"cfg_id": g["cfg_id"].astype(str), "dem": g["dem"].astype(str),
                             "in_sensitivity_range": g["in_sensitivity_range"].astype(bool),
                             "is_primary": g["cfg_id"].astype(str) == self.primary})

    def _raster(self, cfg_id: str) -> np.ndarray:
        from nakkhu.hydro import dem as demmod

        if cfg_id not in self._cache:
            self._cache[cfg_id] = demmod.read_raster(self.interim / cfg_id / "hand.tif").ravel()
        return self._cache[cfg_id]

    def values(self, geoms: np.ndarray, stats: Sequence[str],
               cfg_ids: Sequence[str]) -> dict[str, np.ndarray]:
        import geopandas as gpd

        from nakkhu import hazard

        series = gpd.GeoSeries(np.asarray(geoms, dtype=object), crs=self.settings.crs)
        index_tif = self.interim / "grid" / "cell_index.tif"
        cent = hazard.centroid_cells(series, self.grid) if "centroid" in stats else None
        foot = hazard.cell_groups(series, index_tif) if "footprint_min" in stats else None
        buf = (hazard.cell_groups(series.buffer(self.buffer_m), index_tif)
               if "buffer_min" in stats else None)
        out: dict[str, np.ndarray] = {}
        for cfg in cfg_ids:
            vals = self._raster(cfg)
            for stat in stats:
                if stat == "centroid" and cent is not None:
                    out[f"{cfg}__{stat}"] = hazard.sample_centroids(vals, cent)
                elif stat == "footprint_min" and foot is not None:
                    out[f"{cfg}__{stat}"] = foot.minimum(vals)
                elif stat == "buffer_min" and buf is not None:
                    out[f"{cfg}__{stat}"] = buf.minimum(vals)
                else:
                    raise UncertaintyError(f"unknown HAND statistic {stat!r}")
        return out


# --- inputs -----------------------------------------------------------------------------------


@dataclass
class Inputs:
    """Everything the analysis reads, already in memory (geometries in EPSG:32645).

    - ``prec``: answered-or-not sitting-2 units: unit, stratum, frame_n, answer, x, y (tier
      point), in_final, raw_score, in_seed0..4.
    - ``recall``: sitting-1 primary items without ``outside``: item_id, stratum, frame_n, x, y,
      in_microsoft (1 m membership of a registered Microsoft polygon, plan 006 frame).
    - ``codings``: per answer coding name, item_id, answer, px, py (detection point).
    - ``frame``: population units of the recall frame without ``outside``: stratum, x, y,
      in_microsoft.
    - ``final``: detector_final: conf, geometry. ``seeds``: five arrays of seed boxes.
    - ``inventories``: other descriptive inventories (L2 seeds, footprints): geometries.
    - ``inv_prec``: per footprint inventory with its own polygon-precision task (plan 013:
      ``microsoft`` -> sitting-2 ``precision_microsoft``): item_id, stratum, frame_n, answer,
      x, y (the polygon's HAND point, the centroid its HAND table uses).
    """

    prec: pd.DataFrame
    recall: pd.DataFrame
    codings: dict[str, pd.DataFrame]
    outside: dict[str, int]
    frame: pd.DataFrame
    final: pd.DataFrame
    seeds: list[np.ndarray]
    inventories: dict[str, np.ndarray]
    blocks: pd.DataFrame                    # block_id, geometry
    event_skill: pd.DataFrame
    hand: HandProvider
    meta: dict[str, Any] = field(default_factory=dict)
    # OSM greenhouse areas (prereg `greenhouses`: not buildings; applies to every exposure
    # count): any box with IoU >= greenhouse_iou or its centroid inside one is dropped.
    greenhouses: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    greenhouse_iou: float = 0.3
    inv_prec: dict[str, pd.DataFrame] = field(default_factory=dict)


def not_greenhouse(geoms: np.ndarray, greenhouses: np.ndarray, iou_min: float) -> np.ndarray:
    """True where a geometry is NOT a greenhouse under the shared rule
    (``nakkhu.reference.greenhouse.greenhouse_overlap``)."""
    import geopandas as gpd

    from nakkhu.reference.greenhouse import greenhouse_overlap

    geoms = np.asarray(geoms, dtype=object)
    if not len(geoms) or not len(greenhouses):
        return np.ones(len(geoms), dtype=bool)
    flags = greenhouse_overlap(gpd.GeoSeries(geoms), gpd.GeoSeries(np.asarray(
        greenhouses, dtype=object)), iou_min)
    return ~flags["is_greenhouse"].to_numpy(dtype=bool)


def drop_greenhouses(inp: Inputs) -> tuple[Inputs, dict[str, int]]:
    """detector_final, the seeds, the other inventories and the sitting-2 units (by their tier
    box, column ``geometry`` if present) without greenhouses; counts removed per input."""
    import dataclasses

    gh, iou = inp.greenhouses, float(inp.greenhouse_iou)
    removed: dict[str, int] = {}
    keep = not_greenhouse(inp.final["geometry"].to_numpy(), gh, iou)
    removed["detector_final"] = int((~keep).sum())
    final = inp.final.loc[keep].reset_index(drop=True)
    seeds = []
    for k, s in enumerate(inp.seeds):
        m = not_greenhouse(s, gh, iou)
        removed[f"seed{k}"] = int((~m).sum())
        seeds.append(np.asarray(s, dtype=object)[m])
    invs = {}
    for name, g in inp.inventories.items():
        m = not_greenhouse(g, gh, iou)
        removed[name] = int((~m).sum())
        invs[name] = np.asarray(g, dtype=object)[m]
    prec = inp.prec
    if "geometry" in prec.columns:
        m = not_greenhouse(prec["geometry"].to_numpy(), gh, iou)
        removed["precision_units"] = int((~m).sum())
        prec = prec.loc[m].reset_index(drop=True)
    inv_prec = {}
    for name, units in inp.inv_prec.items():
        if "geometry" in units.columns:
            m = not_greenhouse(units["geometry"].to_numpy(), gh, iou)
            removed[f"precision_units_{name}"] = int((~m).sum())
            units = units.loc[m].reset_index(drop=True)
        inv_prec[name] = units
    return dataclasses.replace(inp, final=final, seeds=seeds, inventories=invs,
                               prec=prec, inv_prec=inv_prec), removed


def coding_names(spec: dict[str, Any]) -> list[str]:
    return ["primary", *[str(s["name"]) for s in spec["answers"]["sensitivities"]]]


def recall_codings(items: pd.DataFrame, s1: dict[str, str],
                   near: dict[str, dict[str, Any]], spec: dict[str, Any]
                   ) -> dict[str, pd.DataFrame]:
    """Answer and detection point per sitting-1 item for every pre-registered coding.

    ``near[parent]`` = {"answer", "click_m", "click_dx_m", "click_dy_m"} of the near-miss item.
    Protocol v2 detection points are the clicked roof for a near-miss Yes that turned the item
    to Yes (prereg ``audit_recall.estimators.model_recall_point``); else the item's own point."""
    from nakkhu.audit.estimators import combine_recall, unsure_as

    xy = items.set_index("item_id")[["x", "y"]]
    out = {}
    entries = [("primary", spec["answers"]["primary"]),
               *[(str(s["name"]), s) for s in spec["answers"]["sensitivities"]]]
    for name, c in entries:
        if c["protocol"] == "v1":
            ans = dict(s1)
        elif c["protocol"] == "v2":
            ans = combine_recall(s1, near, float(c["radius_m"]))
        else:
            raise UncertaintyError(f"unknown protocol {c['protocol']!r}")
        if c.get("unsure") == "as_yes":
            ans = unsure_as(ans, "yes")
        elif c.get("unsure") == "as_no":
            ans = unsure_as(ans, "no")
        rows = []
        for item in items["item_id"]:
            px, py = float(xy.at[item, "x"]), float(xy.at[item, "y"])
            n = near.get(item)
            if (c["protocol"] == "v2" and ans.get(item) == "yes" and s1.get(item) != "yes"
                    and n is not None and n.get("click_dx_m") is not None):
                px, py = px + float(n["click_dx_m"]), py + float(n["click_dy_m"])
            rows.append({"item_id": item, "answer": ans.get(item), "px": px, "py": py})
        out[name] = pd.DataFrame(rows)
    return out


def precision_units(items: pd.DataFrame, answers: dict[str, str], members: pd.DataFrame,
                    boxes: pd.DataFrame, final_run: str, seed_runs: Sequence[str]
                    ) -> pd.DataFrame:
    """Sitting-2 units with their answer, domains, raw score and tier point (the centroid of
    the highest-conf ensemble box for detector_final-domain units, else the unit point)."""
    from nakkhu.uncertainty.ensemble import unit_domains

    dom = unit_domains(members, final_run, seed_runs).set_index("unit")
    m = members.assign(run=members["run"].astype(str))
    fin = m.loc[m["run"] == str(final_run)].merge(
        boxes.assign(run=boxes["run"].astype(str))[["run", "det", "geometry"]],
        on=["run", "det"], how="left")
    fin = fin.sort_values(["cluster", "conf"], ascending=[True, False]).drop_duplicates("cluster")
    cent = shapely.centroid(np.asarray(fin["geometry"].to_numpy(), dtype=object))
    pt = pd.DataFrame({"tx": shapely.get_x(cent), "ty": shapely.get_y(cent)},
                      index=fin["cluster"].to_numpy())
    u = items[["item_id", "unit", "stratum", "frame_n", "x", "y"]].copy()
    u["answer"] = u["item_id"].map(answers)
    u = u.join(dom, on="unit")
    u = u.join(pt, on="unit")
    has = u["tx"].notna() & u["in_final"].fillna(False).astype(bool)
    u.loc[has, "x"] = u.loc[has, "tx"]
    u.loc[has, "y"] = u.loc[has, "ty"]
    # tier box: the highest-conf ensemble box, else the unit's medoid box (items wkb)
    box = pd.Series(fin["geometry"].to_numpy(), index=fin["cluster"].to_numpy())
    geom = u["unit"].map(box)
    if "wkb" in items.columns:
        med = pd.Series(shapely.from_wkb(items["wkb"].to_numpy()), index=items.index)
        geom = geom.where(has, med)
    u["geometry"] = geom.to_numpy()
    return u.drop(columns=["tx", "ty"])


# --- helpers ----------------------------------------------------------------------------------


def _points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return shapely.points(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def near_any(x: np.ndarray, y: np.ndarray, geoms: np.ndarray, tol: float) -> np.ndarray:
    """True where a point lies within ``tol`` metres of any geometry."""
    pts = _points(x, y)
    out = np.zeros(len(pts), dtype=bool)
    geoms = np.asarray(geoms, dtype=object)
    if len(geoms) and len(pts):
        hit, _ = shapely.STRtree(geoms).query(pts, predicate="dwithin", distance=tol)
        out[np.unique(hit)] = True
    return out


def block_of(x: np.ndarray, y: np.ndarray, blocks: pd.DataFrame) -> np.ndarray:
    pts = _points(x, y)
    out = np.full(len(pts), "none", dtype=object)
    geoms = np.asarray(blocks["geometry"].to_numpy(), dtype=object)
    if len(pts) and len(geoms):
        pi, bi = shapely.STRtree(geoms).query(pts, predicate="within")
        out[pi[::-1]] = blocks["block_id"].astype(str).to_numpy()[bi[::-1]]
    return out


def tiers(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    from nakkhu import hazard

    return hazard.tier_index(np.asarray(values, dtype=float), edges).astype(int)


def _counts(t: np.ndarray, n: int, w: np.ndarray | None = None) -> np.ndarray:
    ok = t >= 0
    return np.bincount(t[ok], weights=None if w is None else np.asarray(w)[ok],
                       minlength=n)[:n].astype(float)


def _hand_points(hand: HandProvider, x: np.ndarray, y: np.ndarray,
                 cfgs: Sequence[str]) -> dict[str, np.ndarray]:
    return hand.values(_points(x, y), ["centroid"], cfgs)


# --- the analysis -----------------------------------------------------------------------------


def analyse(inp: Inputs, spec: dict[str, Any]) -> dict[str, Any]:
    """U1, U3 (all codings), the split-sample check and its simulation, U2 primary and
    descriptive, D4 and the C2 outcome; returns the output tables."""
    from nakkhu import hazard

    inp, gh_removed = drop_greenhouses(inp)
    edges = tuple(inp.hand.edges) or EDGES
    labels = hazard.tier_labels(edges)
    n_t = len(labels)
    surf = inp.hand.surfaces()
    in_range = surf.loc[surf["in_sensitivity_range"]].sort_values("cfg_id")
    cfgs = in_range["cfg_id"].tolist()
    primary = surf.loc[surf["is_primary"], "cfg_id"].iloc[0]
    if primary not in cfgs:
        raise UncertaintyError(f"primary surface {primary} is not in the routing-gate range")
    dem_of = dict(zip(in_range["cfg_id"], in_range["dem"], strict=True))
    pcol = f"{primary}__centroid"
    tol = float(spec["answers"]["detection_tolerance_m"])
    u1s, u3s, u2s = spec["u1"], spec["u3"], spec["u2"]
    boot = u3s["bootstrap"]
    final_geoms = np.asarray(inp.final["geometry"].to_numpy(), dtype=object)
    final_cent = shapely.centroid(final_geoms)
    fx, fy = shapely.get_x(final_cent), shapely.get_y(final_cent)

    # --- tiers ---------------------------------------------------------------------------
    prec = inp.prec.reset_index(drop=True)
    prec_h = _hand_points(inp.hand, prec["x"].to_numpy(), prec["y"].to_numpy(), cfgs)
    rec = inp.recall.reset_index(drop=True)
    rec_h = _hand_points(inp.hand, rec["x"].to_numpy(), rec["y"].to_numpy(), cfgs)
    frame = inp.frame
    frame_h = _hand_points(inp.hand, frame["x"].to_numpy(), frame["y"].to_numpy(), [primary])
    final_h = _hand_points(inp.hand, fx, fy, [primary])
    prec["tier"] = tiers(prec_h[pcol], edges)
    rec["tier"] = tiers(rec_h[pcol], edges)
    frame_t = tiers(frame_h[pcol], edges)
    final_t = tiers(final_h[pcol], edges)
    prec["block"] = block_of(prec["x"].to_numpy(), prec["y"].to_numpy(), inp.blocks)
    rec["block"] = block_of(rec["x"].to_numpy(), rec["y"].to_numpy(), inp.blocks)

    prim = inp.codings["primary"].set_index("item_id")
    p_ans = prec.loc[prec["answer"].isin(["yes", "no"])].reset_index(drop=True)
    p_ans["y"] = (p_ans["answer"] == "yes").astype(float)
    p_ans["in_final"] = p_ans["in_final"].fillna(False).astype(bool)
    r_prim = rec.join(prim, on="item_id")
    r_prim = r_prim.loc[r_prim["answer"].isin(["yes", "no"])]

    # U3 inventories (plan 013, prereg v8): the first is primary (the main inventory). A
    # footprint inventory needs its own polygon-precision units (inp.inv_prec).
    main_inv = str(spec["inputs"].get("main_inventory", "detector_final"))
    u3_invs = [str(v) for v in u3s.get("inventories") or ["detector_final"]]
    if FINAL not in u3_invs:
        raise UncertaintyError("u3.inventories must list detector_final (C2 gates on it)")
    other_prec: dict[str, pd.DataFrame] = {}
    for inv in u3_invs:
        if inv == FINAL:
            continue
        if inv not in inp.inv_prec or f"in_{inv}" not in rec.columns \
                or f"in_{inv}" not in frame.columns:
            raise UncertaintyError(f"u3 inventory {inv} needs its precision units and the "
                                   f"recall-frame flag in_{inv}")
        units = inp.inv_prec[inv]
        units = units.loc[units["answer"].isin(["yes", "no"])].reset_index(drop=True)
        uh = _hand_points(inp.hand, units["x"].to_numpy(), units["y"].to_numpy(), [primary])
        other_prec[inv] = units.assign(tier=tiers(uh[pcol], edges),
                                       real=(units["answer"] == "yes").astype(float))

    # --- pooling (answered counts only; one grouping for every inventory) -----------------
    answered = np.minimum(_counts(r_prim["tier"].to_numpy(), n_t),
                          _counts(p_ans.loc[p_ans["in_final"], "tier"].to_numpy(), n_t))
    for units in other_prec.values():
        answered = np.minimum(answered, _counts(units["tier"].to_numpy(), n_t))
    group = P.pool_tiers(answered.astype(int).tolist(), int(u3s["pooling"]["min_answered"]))
    gmap = np.asarray(group)
    n_g = max(group) + 1

    def grp(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return np.where(t >= 0, gmap[np.clip(t, 0, n_t - 1)], -1)

    # --- U1 ----------------------------------------------------------------------------------
    pw_all = P.design_weights(p_ans["stratum"].to_numpy(), p_ans["frame_n"].to_numpy())
    dom = p_ans["in_final"].to_numpy()
    u1 = C.calibrate_u1(p_ans.loc[dom, "raw_score"].to_numpy(dtype=float),
                        p_ans.loc[dom, "y"].to_numpy(), pw_all[dom],
                        p_ans.loc[dom, "block"].to_numpy(), bins=int(u1s["ece"]["bins"]),
                        methods=tuple(u1s["methods"]))
    p_final = u1["predict"](inp.final["conf"].to_numpy(dtype=float))
    ece_rows, curve_rows = [], []
    for m, d in u1["methods"].items():
        ece_rows.append({"method": m, "ece": d["ece"], "ece_unweighted": d["ece_unweighted"],
                         "brier": d["brier"], "n_scored": d["n_scored"], "n": d["n"],
                         "chosen": m == u1["chosen"]})
        curve_rows.append(d["reliability"].assign(method=m))
    ece = pd.DataFrame(ece_rows)
    curve = pd.concat(curve_rows, ignore_index=True)

    # --- U3, per inventory (plan 013, prereg v8) ------------------------------------------
    # Exposure is counted in real objects: precision_only = D_t p_g (objects x the audited
    # precision of those objects) is the exposure count, primary for the main inventory;
    # naive = raw D_t. PPI++ / HT estimate T_g over recall-frame POINTS (several points can sit
    # on one roof), so they are upper bounds on buildings. The ratio D_t p_g / r_g divides an
    # object count by a point-level recall: mixed units, a sensitivity row only.
    p_rep = P.replicate_weights(p_ans["stratum"].to_numpy(), p_ans["frame_n"].to_numpy(),
                                int(boot["n_boot"]),
                                S.stream_seed(int(boot["seed"]), "u3_bootstrap_precision"))
    frame_g = grp(frame_t)
    pred_frame = {FINAL: near_any(frame["x"].to_numpy(), frame["y"].to_numpy(), final_geoms,
                                  tol)}
    d_ts = {FINAL: _counts(final_t, n_t)}
    prec_in: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {FINAL: (
        {"stratum": p_ans["stratum"].to_numpy(), "frame_n": p_ans["frame_n"].to_numpy(),
         "y": p_ans["y"].to_numpy(), "group": grp(p_ans["tier"].to_numpy()), "final": dom},
        p_rep)}
    for inv, units in other_prec.items():
        pred_frame[inv] = frame[f"in_{inv}"].to_numpy(dtype=bool)
        h = inp.hand.values(np.asarray(inp.inventories[inv], dtype=object), ["centroid"],
                            [primary])
        d_ts[inv] = _counts(tiers(h[pcol], edges), n_t)
        prec_in[inv] = (
            {"stratum": units["stratum"].to_numpy(), "frame_n": units["frame_n"].to_numpy(),
             "y": units["real"].to_numpy(), "group": grp(units["tier"].to_numpy()),
             "final": np.ones(len(units), dtype=bool)},
            P.replicate_weights(units["stratum"].to_numpy(), units["frame_n"].to_numpy(),
                                int(boot["n_boot"]),
                                S.stream_seed(int(boot["seed"]), "u3_bootstrap_precision_main")))
    f_tot = {inv: {g: float(pf[frame_g == g].sum()) for g in range(n_g)}
             for inv, pf in pred_frame.items()}

    def role(est: str, inv: str, coding: str) -> str:
        if est in ("ppi_pp", "ht"):
            return "upper_bound_frame_points"
        if est == "ratio":
            return "sensitivity_mixed_units"
        if est == "naive":
            return "raw_count"
        return "primary" if inv == main_inv and coding == "primary" else "secondary"

    rows: list[dict[str, Any]] = []
    split_inputs: dict[str, pd.DataFrame] = {}
    for name, cod in inp.codings.items():
        r = rec.join(cod.set_index("item_id"), on="item_id")
        r = r.loc[r["answer"].isin(["yes", "no"])].reset_index(drop=True)
        y = (r["answer"] == "yes").to_numpy(dtype=float)
        g = grp(r["tier"].to_numpy())
        rep = P.replicate_weights(r["stratum"].to_numpy(), r["frame_n"].to_numpy(),
                                  int(boot["n_boot"]),
                                  S.stream_seed(int(boot["seed"]), "u3_bootstrap_recall"))
        for inv in u3_invs:
            if inv == FINAL:
                f = near_any(r["x"].to_numpy(), r["y"].to_numpy(), final_geoms, tol)
                d = near_any(r["px"].to_numpy(), r["py"].to_numpy(), final_geoms, tol)
            else:          # recall-frame membership flag at the unit point (plan 006 frame)
                f = d = r[f"in_{inv}"].fillna(False).to_numpy(dtype=bool)
            f, d = f.astype(float), d.astype(float)
            tag = {"coding": name, "inventory": inv}
            for k in range(n_g):
                dm = (g == k).astype(float)
                members = [labels[t] for t in range(n_t) if group[t] == k]
                for est, lam in (("ppi_pp", None), ("ht", 0.0)):
                    res = P.ppi_total(r["stratum"].to_numpy(), r["frame_n"].to_numpy(), y * dm,
                                      f * dm, f_tot[inv][k], lam=lam, rep_w=rep)
                    rows.append({**tag, "level": "group", "group": k,
                                 "tiers": "+".join(members), "estimator": est,
                                 "count_unit": COUNT_UNIT[est], "role": role(est, inv, name),
                                 "estimate": res["estimate"], "ci_lo": res["ci_lo"],
                                 "ci_hi": res["ci_hi"], "lambda": res["lambda"],
                                 "se_lin": res["se_lin"], "se_boot": res["se_boot"],
                                 "n_answered_in_group": int(dm.sum())})
            rc = P.ratio_counts(d_ts[inv], group, prec_in[inv][0],
                                {"stratum": r["stratum"].to_numpy(),
                                 "frame_n": r["frame_n"].to_numpy(), "y": y, "group": g, "d": d},
                                prec_in[inv][1], rep)
            for row in rc.to_dict("records"):
                base = {**tag, "level": "tier", "group": int(row["group"]),
                        "tiers": labels[int(row["tier"])], "lambda": np.nan, "se_lin": np.nan,
                        "se_boot": np.nan, "n_answered_in_group": np.nan}
                rows.append({**base, "estimator": "naive", "count_unit": COUNT_UNIT["naive"],
                             "role": role("naive", inv, name), "estimate": row["naive"],
                             "ci_lo": np.nan, "ci_hi": np.nan})
                rows.append({**base, "estimator": "precision_only",
                             "count_unit": COUNT_UNIT["precision_only"],
                             "role": role("precision_only", inv, name),
                             "estimate": row["precision_only"],
                             "ci_lo": row["precision_only_ci"][0],
                             "ci_hi": row["precision_only_ci"][1],
                             "precision": row["precision"]})
                rows.append({**base, "estimator": "ratio", "count_unit": COUNT_UNIT["ratio"],
                             "role": role("ratio", inv, name), "estimate": row["ratio"],
                             "ci_lo": row["ratio_ci"][0], "ci_hi": row["ratio_ci"][1],
                             "precision": row["precision"], "recall": row["recall"]})
            if name == "primary":
                split_inputs[inv] = pd.DataFrame({
                    "stratum": r["stratum"].to_numpy(), "frame_n": r["frame_n"].to_numpy(),
                    "y": y, "f": f, "group": g, "block": r["block"].to_numpy()})
    for t in range(n_t):
        rows.append({"coding": "all", "inventory": FINAL, "level": "tier", "group": group[t],
                     "tiers": labels[t], "estimator": "ensemble_expected",
                     "count_unit": COUNT_UNIT["ensemble_expected"], "role": "secondary",
                     "estimate": float(np.sum(p_final[final_t == t])),
                     "ci_lo": np.nan, "ci_hi": np.nan})
    ppi_rows = pd.DataFrame(rows)
    split_sample_input = split_inputs[FINAL]          # C2.ppi gates on detector_final (frozen)
    frame_f = pred_frame[FINAL]

    # --- split-sample check and simulation ----------------------------------------------
    ss = u3s["split_sample"]
    assert split_sample_input is not None
    sp_in = split_sample_input.loc[split_sample_input["group"] >= 0]
    primary_mode = str(ss["unit"])
    secondary = [str(m) for m in ss.get("secondary_descriptive") or []]
    splits = {m: P.split_sample(sp_in, f_tot[FINAL], repeats=int(ss["repeats"]),
                                seed=int(ss["seed"]), n_boot=int(ss["n_boot"]),
                                min_per_stratum=int(ss["min_per_stratum"]), mode=m)
              for m in [primary_mode, *secondary]}
    split = splits[primary_mode]
    sim = u3s["simulation"]
    pop = plugin_population(frame["stratum"].to_numpy(), frame_g, frame_f.astype(float),
                            block_of(frame["x"].to_numpy(), frame["y"].to_numpy(), inp.blocks),
                            split_sample_input, S.stream_seed(int(sim["seed"]), "simulation"))
    n_h = sp_in.groupby("stratum").size().astype(int).to_dict()
    cover = P.simulate_coverage(pop, n_h, replicates=int(sim["replicates"]),
                                seed=int(sim["seed"]), n_boot=int(sim["n_boot"]))
    expect = {m: P.simulate_split_miss(pop, n_h, samples=int(sim["replicates"]),
                                       seed=int(sim["seed"]), n_boot=int(sim["n_boot"]),
                                       mode=m) for m in splits}
    band = [float(v) for v in sim["coverage_band"]]
    rate_rows = [{"mode": m, "role": "gating" if m == primary_mode else "descriptive",
                  "observed_miss_rate": splits[m]["miss_rate"],
                  "expected_miss_rate": expect[m]["expected_miss_rate"],
                  "observed_a_ci_misses_b_rate": splits[m]["a_ci_misses_b_rate"],
                  "expected_a_ci_misses_b_rate": expect[m]["expected_a_ci_misses_b_rate"],
                  "valid_repeats": splits[m]["valid"], "inventory": FINAL} for m in splits]
    # the other U3 inventories (Microsoft): same check, gating mode only; reported, never
    # gating (C2 stays on detector_final as frozen; plan 013 Open Question 3)
    other_split: dict[str, Any] = {}
    for inv, sp in split_inputs.items():
        if inv == FINAL:
            continue
        sp_o = sp.loc[sp["group"] >= 0]
        obs = P.split_sample(sp_o, f_tot[inv], repeats=int(ss["repeats"]), seed=int(ss["seed"]),
                             n_boot=int(ss["n_boot"]),
                             min_per_stratum=int(ss["min_per_stratum"]), mode=primary_mode)
        pop_o = plugin_population(frame["stratum"].to_numpy(), frame_g,
                                  pred_frame[inv].astype(float),
                                  block_of(frame["x"].to_numpy(), frame["y"].to_numpy(),
                                           inp.blocks),
                                  sp, S.stream_seed(int(sim["seed"]), "simulation"))
        exp_o = P.simulate_split_miss(pop_o, n_h, samples=int(sim["replicates"]),
                                      seed=int(sim["seed"]), n_boot=int(sim["n_boot"]),
                                      mode=primary_mode)
        other_split[inv] = {"split_sample": obs, "expected_split": exp_o}
        rate_rows.append({"mode": primary_mode, "role": "reported", "inventory": inv,
                          "observed_miss_rate": obs["miss_rate"],
                          "expected_miss_rate": exp_o["expected_miss_rate"],
                          "observed_a_ci_misses_b_rate": obs["a_ci_misses_b_rate"],
                          "expected_a_ci_misses_b_rate": exp_o["expected_a_ci_misses_b_rate"],
                          "valid_repeats": obs["valid"]})
    coverage = {"split_sample": split, "split_sample_by_mode": splits,
                "split_sample_other_inventories": other_split,
                "split_rates": rate_rows, "simulation": cover,
                "expected_split": expect[primary_mode], "expected_split_by_mode": expect,
                "coverage_band": band,
                "coverage_in_band": bool(band[0] <= cover["coverage_pooled"] <= band[1]),
                "outside_not_in_any_source": inp.outside,
                "greenhouses_removed": gh_removed,
                "tier_groups": {labels[t]: int(group[t]) for t in range(n_t)},
                "answered_per_tier": {labels[t]: int(answered[t]) for t in range(n_t)}}

    # --- U2 primary: seed x surface x audit replicate ----------------------------------
    pr = u2s["primary"]
    n_b = int(pr["audit_replicates"]["n"])
    a_seed = int(pr["audit_replicates"]["seed"])
    seeds_cfg = [int(s) for s in pr["factors"]["seed"]]
    p_rep2 = P.replicate_weights(p_ans["stratum"].to_numpy(), p_ans["frame_n"].to_numpy(),
                                 n_b, S.stream_seed(a_seed, "u2_audit_replicates_precision"))
    r = r_prim.reset_index(drop=True)
    ry = (r["answer"] == "yes").to_numpy(dtype=float)
    r_rep2 = P.replicate_weights(r["stratum"].to_numpy(), r["frame_n"].to_numpy(), n_b,
                                 S.stream_seed(a_seed, "u2_audit_replicates_recall"))
    p_idx = np.flatnonzero(prec["answer"].isin(["yes", "no"]).to_numpy())
    r_pos = r_prim.index.to_numpy()              # positions in rec (RangeIndex)
    dmat = np.zeros((len(seeds_cfg), len(cfgs), n_t))
    pmat = np.zeros((len(seeds_cfg), len(cfgs), n_g, n_b))
    rmat = np.zeros_like(pmat)
    for si, s in enumerate(seeds_cfg):
        boxes = np.asarray(inp.seeds[si], dtype=object)
        bc = shapely.centroid(boxes)
        bh = _hand_points(inp.hand, shapely.get_x(bc), shapely.get_y(bc), cfgs)
        in_s = p_ans[f"in_seed{s}"].fillna(False).to_numpy(dtype=float)
        det_s = near_any(r["px"].to_numpy(), r["py"].to_numpy(), boxes, tol).astype(float)
        for hi, cfg in enumerate(cfgs):
            col = f"{cfg}__centroid"
            dmat[si, hi] = _counts(tiers(bh[col], edges), n_t)
            pg = grp(tiers(prec_h[col][p_idx], edges))
            rg = grp(tiers(rec_h[col][r_pos], edges))
            for k in range(n_g):
                pk = (pg == k).astype(float) * in_s
                rk = (rg == k).astype(float)
                pmat[si, hi, k] = P.weighted_ratio(p_rep2, p_ans["y"].to_numpy() * pk, pk)
                rmat[si, hi, k] = P.weighted_ratio(r_rep2, ry * det_s * rk, ry * rk)
    target = [labels.index(str(t)) for t in pr["target_tiers"]]
    ci = u2s["share_ci"]
    strata = [dem_of[c] for c in cfgs]
    names = ["seed", "surface", "audit"]
    dec_rows: list[dict[str, Any]] = []
    finding: dict[str, Any] = {"computable": False}
    responses = [("hand_le_3m", target), *[(labels[t], [t]) for t in range(n_t)]]
    # prereg v8: the primary response is unit-consistent (seed boxes x their audited precision,
    # D p); the v7 ratio D p / r (boxes over a point-level recall) is a mixed-unit sensitivity.
    ones = np.ones_like(rmat)
    analyses = [("primary", rmat if pr.get("recall_correction", True) else ones)]
    for extra in u2s.get("sensitivity") or []:
        analyses.append((str(extra["name"]), rmat if extra.get("recall_correction") else ones))
    for analysis, rdiv in analyses:
        for resp, tt in responses:
            yarr = D.corrected_counts(dmat, pmat, rdiv, group, tt)
            if not np.all(np.isfinite(yarr)):
                dec_rows.append({"analysis": analysis, "response": resp, "term": None,
                                 "status": "not computable (zero recall or empty domain)"})
                continue
            terms = D.share_ci(yarr, names, axis=1, strata=strata, n_boot=int(ci["n_boot"]),
                               seed=S.stream_seed(int(ci["seed"]), "u2_share_ci"),
                               level=float(ci["level"]))
            for row in terms.to_dict("records"):
                dec_rows.append({"analysis": analysis, "response": resp, "status": "ok", **row})
            for row in D.factor_shares(terms, names).to_dict("records"):
                dec_rows.append({"analysis": analysis, "response": resp, "status": "ok",
                                 "term": row["factor"], "kind": "factor",
                                 "first_order": row["first_order"], "total": row["total"]})
            if resp == "hand_le_3m" and analysis == "primary":
                finding = {"computable": True, **D.terrain_dominates(terms, "surface",
                                                                     ["seed", "audit"])}

    # --- U2 descriptive and D4 ---------------------------------------------------------
    desc = u2s["descriptive"]
    stats = [STAT_ALIAS.get(s, s) for s in spec["inputs"]["hand_statistics_sensitivity"]]
    long_rows = []
    for inv in desc["inventories"]:
        inv = str(inv)
        w = None
        if inv in ("detector_final", "ensemble_expected"):
            geoms = final_geoms
            w = p_final if inv == "ensemble_expected" else None
        elif inv.startswith("seed"):
            geoms = np.asarray(inp.seeds[seeds_cfg.index(int(inv[4:]))], dtype=object)
        else:
            geoms = np.asarray(inp.inventories[inv], dtype=object)
        vals = inp.hand.values(geoms, stats, cfgs)
        for key, v in vals.items():
            c = _counts(tiers(v, edges), n_t, w)
            long_rows += [{"inventory": inv, "member": key, "tier": labels[t], "tier_index": t,
                           "count": float(c[t])} for t in range(n_t)]
    long = pd.DataFrame(long_rows)
    fact_problems = D.check_factorial(long.drop(columns="tier").rename(
        columns={"tier_index": "tier"}))
    for t in range(n_t):
        sub = long.loc[long["tier_index"] == t]
        arr = D.to_array(sub, ["inventory", "member"], "count")
        for row in D.anova_terms(arr, ["inventory", "member"]).to_dict("records"):
            dec_rows.append({"analysis": "descriptive", "response": labels[t], "status": "ok",
                             **row})
    variance = pd.DataFrame(dec_rows)

    skill = inp.event_skill
    prim_member = f"{primary}__centroid"
    baseline = str((spec.get("d4") or {}).get("baseline") or main_inv)   # v8: microsoft
    base = long.loc[(long["inventory"] == baseline) & (long["member"] == prim_member)]
    if base.empty:
        raise UncertaintyError(f"D4 baseline {baseline} is not a u2 descriptive inventory")
    base_c = base.set_index("tier")["count"]
    d4 = []
    for inv in desc["inventories"]:
        if inv == "ensemble_expected":
            continue
        sub = long.loc[(long["inventory"] == inv) & (long["member"] == prim_member)]
        cnt = sub.set_index("tier")["count"]
        row: dict[str, Any] = {"inventory": inv, "baseline": baseline,
                               "total": float(cnt.sum()),
                               "delta_total_pct": 100 * (cnt.sum() / base_c.sum() - 1)
                               if base_c.sum() else np.nan}
        for lab in labels:
            row[f"n_{lab}"] = float(cnt.get(lab, 0.0))
            b = float(base_c.get(lab, 0.0))
            row[f"delta_pct_{lab}"] = 100 * (row[f"n_{lab}"] / b - 1) if b else np.nan
        ev = skill.loc[skill["inventory"].astype(str) == str(inv)] if len(skill) else skill
        ref = None
        for want in ("R1", "R1-aud"):
            if len(ev) and (ev["reference"] == want).any():
                ref = want
                break
        if ref is not None:
            e = ev.loc[ev["reference"] == ref].iloc[0]
            row.update({"event_reference": ref, "event_auc": float(e["auc"]),
                        "event_pr_auc": float(e["pr_auc"])})
        else:
            row.update({"event_reference": None, "event_auc": np.nan, "event_pr_auc": np.nan})
        d4.append(row)

    fals = c2_outcome(u1, split, finding)
    fals["ppi"].update({"split_mode": split["mode"], "inventory": FINAL,
                        "expected_miss_rate": expect[primary_mode]["expected_miss_rate"],
                        "descriptive_splits": [r for r in rate_rows
                                               if r["role"] == "descriptive"],
                        "reported_other_inventories": [r for r in rate_rows
                                                       if r["role"] == "reported"]})
    fals["intervals_computable"] = bool(
        np.isfinite(ppi_rows.loc[(ppi_rows["coding"] == "primary")
                                 & (ppi_rows["inventory"] == FINAL)
                                 & (ppi_rows["estimator"] == "ppi_pp"),
                                 ["ci_lo", "ci_hi"]].to_numpy(dtype=float)).all())
    fals["main_inventory"] = main_inv
    fals["decomposition_finding"] = finding
    fals["factorial_problems"] = fact_problems
    fals["u1_chosen"] = u1["chosen"]
    return {"calibration_curve.csv": curve, "ece.csv": ece,
            "tier_counts_factorial.parquet": long,
            "variance_decomposition.csv": variance, "ppi_tier_counts.csv": ppi_rows,
            "coverage_check.json": coverage, "inventory_comparison.csv": pd.DataFrame(d4),
            "falsifiers.json": fals}


def plugin_population(strata: np.ndarray, group: np.ndarray, f: np.ndarray, blocks: np.ndarray,
                      sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Synthetic population on the real frame (strata, groups, blocks, detector predictions);
    Y drawn with the observed answered Yes share per (stratum, f) cell (stratum, then overall
    share where a cell is empty): known truth for the coverage simulation."""
    gen = np.random.default_rng(seed)
    overall = float(sample["y"].mean()) if len(sample) else 0.5
    by_s = sample.groupby("stratum")["y"].mean().to_dict()
    by_sf = sample.groupby(["stratum", "f"])["y"].mean().to_dict()
    q = np.array([by_sf.get((s, float(v)), by_s.get(s, overall))
                  for s, v in zip(strata, f, strict=True)], dtype=float)
    pop = pd.DataFrame({"stratum": strata, "y": (gen.random(len(q)) < q).astype(float),
                        "f": np.asarray(f, dtype=float), "group": group, "block": blocks})
    pop = pop.loc[pop["group"] >= 0].reset_index(drop=True)
    pop["frame_n"] = pop.groupby("stratum")["stratum"].transform("size")
    return pop


# --- file inputs (the real project) -------------------------------------------------------


class FileInputs:
    """Reads the pinned and promoted inputs of the real project."""

    def problems(self, project: Project, spec: dict[str, Any]) -> list[str]:
        from nakkhu.audit import core as audit_core
        from nakkhu.audit.workflow import require_complete

        out = [p for p in S.prerequisites(project) if not p.startswith("prereg")]
        cfg = audit_core.load_audit_config(project)
        for task in spec["inputs"]["audit_tasks"].values():
            try:
                require_complete(project, cfg, str(task))
            except Exception as error:   # missing items/responses or open slots
                out.append(f"audit {task} not complete ({error})")
        try:
            self._l2_runs(project)
        except Exception as error:
            out.append(f"L2 test-region runs: {error}")
        try:
            from nakkhu.hydro.sweep import latest_sweep

            latest_sweep(project)
        except Exception as error:
            out.append(f"HAND sweep: {error}")
        return out

    @staticmethod
    def _l2_runs(project: Project) -> list[dict[str, Any]]:
        from nakkhu.ensemble import final_members

        members = final_members(project, "L2", "yolo11n-obb", [0, 1, 2])
        for m in members:
            path = project.paths.runs / str(m["infer_test"]) / "detections.gpkg"
            if not path.is_file():
                raise UncertaintyError(f"{path} missing (georef the run first)")
        return members

    def load(self, project: Project, spec: dict[str, Any]) -> Inputs:
        import geopandas as gpd

        from nakkhu.audit import core as audit_core
        from nakkhu.audit import frames
        from nakkhu.hydro.sweep import load_greenhouses, load_inventory, load_settings

        cfg = audit_core.load_audit_config(project)
        region = frames.region(project, cfg)

        # sitting 2
        items = audit_core.load_items(project, "precision_pool")
        ev = audit_core.replay(audit_core.read_events(
            audit_core.responses_path(project, "precision_pool")))
        p_ans = {e["item_id"]: e["answer"] for e in ev.effective
                 if e["question"] == "precision_pool" and not e.get("repeat")}
        members = pd.read_parquet(audit_core.task_dir(project, "precision_pool")
                                  / "members.parquet")
        boxes = frames.load_detections(project, cfg, region)
        final_run, seed_runs = self._pool_roles(project, cfg, spec)
        prec = precision_units(items, p_ans, members, pd.DataFrame(boxes), final_run,
                               seed_runs)
        seeds = [np.asarray(boxes.loc[boxes["run"].astype(str) == r].geometry.values,
                            dtype=object) for r in seed_runs]

        # sittings 1 + 1b
        rp = audit_core.load_items(project, "recall_points")
        rp_ev = audit_core.replay(audit_core.read_events(
            audit_core.responses_path(project, "recall_points")))
        s1 = {e["item_id"]: e["answer"] for e in rp_ev.effective
              if e["question"] == "recall_points" and not e.get("repeat")}
        nm = audit_core.load_items(project, audit_core.NEARMISS_TASK)
        nm_ev = audit_core.replay(audit_core.read_events(
            audit_core.responses_path(project, audit_core.NEARMISS_TASK)))
        parent = dict(zip(nm["item_id"], nm["parent_item"], strict=True))
        kind = dict(zip(nm["item_id"], nm["kind"], strict=True))
        keys = ("answer", "click_m", "click_dx_m", "click_dy_m")
        near = {parent[e["item_id"]]: {k: e.get(k) for k in keys} for e in nm_ev.effective
                if kind.get(e["item_id"]) == "nearmiss"}
        units = rp.loc[rp["stratum"] != frames.OUTSIDE].drop_duplicates("item_id")
        outside = rp.loc[rp["stratum"] == frames.OUTSIDE, "item_id"].map(s1)
        codings = recall_codings(units[["item_id", "x", "y"]], s1, near, spec)
        frame_gdf, _ = frames.recall_frame(project, cfg)
        fpts = np.asarray(frame_gdf.geometry.values, dtype=object)
        frame = pd.DataFrame({"stratum": frame_gdf["stratum"].to_numpy(),
                              "x": shapely.get_x(fpts), "y": shapely.get_y(fpts),
                              "in_microsoft": frame_gdf["in_microsoft"].to_numpy(dtype=bool)})

        # main-inventory polygon precision (plan 013; sitting 2 precision_microsoft)
        inv_prec: dict[str, pd.DataFrame] = {}
        main_inv = str(spec["inputs"].get("main_inventory", "detector_final"))
        main_task = spec["inputs"]["audit_tasks"].get("precision_main")
        if main_inv != "detector_final" and main_task:
            pm = audit_core.load_items(project, str(main_task))
            pm_ev = audit_core.replay(audit_core.read_events(
                audit_core.responses_path(project, str(main_task))))
            pm_ans = {e["item_id"]: e["answer"] for e in pm_ev.effective
                      if e["question"] == str(main_task) and not e.get("repeat")}
            inv_prec[main_inv] = pd.DataFrame({
                "item_id": pm["item_id"].to_numpy(), "stratum": pm["stratum"].to_numpy(),
                "frame_n": pm["frame_n"].to_numpy(), "answer": pm["item_id"].map(pm_ans),
                "building_uid": pm["building_uid"].to_numpy(),
                "x": pm["hand_x"].to_numpy(dtype=float), "y": pm["hand_y"].to_numpy(dtype=float)})

        # inventories
        final = gpd.read_parquet(project.paths.resolve(spec["inputs"]["detector_final"]))
        final = final.to_crs("EPSG:32645")
        invs: dict[str, np.ndarray] = {}
        for k, m in enumerate(self._l2_runs(project)):
            det = gpd.read_file(project.paths.runs / str(m["infer_test"]) / "detections.gpkg",
                                layer="obb").to_crs("EPSG:32645")
            det = det.loc[(det["conf"].to_numpy(dtype=float) >= float(m["threshold"]))
                          & shapely.within(shapely.centroid(det.geometry.values), region)]
            invs[f"L2_seed{k}"] = np.asarray(det.geometry.values, dtype=object)
        settings, _ = load_settings(project)
        extent = settings.read_polygon(settings.v["analysis_extent"])
        gh = load_greenhouses(settings)
        for name in ("google_v3", "microsoft", "overture", "osm"):
            frm, _ = load_inventory(settings, name, extent, gh)
            invs[name] = np.asarray(frm.geometry.values, dtype=object)
        blocks = gpd.read_file(project.paths.resolve(spec["inputs"]["blocks"])).to_crs(
            "EPSG:32645")
        return Inputs(
            prec=prec, recall=units[["item_id", "stratum", "frame_n", "x", "y",
                                     "in_microsoft"]].reset_index(drop=True), codings=codings,
            inv_prec=inv_prec,
            outside={"yes": int((outside == "yes").sum()),
                     "n": int(outside.isin(["yes", "no"]).sum())},
            frame=frame, final=pd.DataFrame({"conf": final["conf"].to_numpy(dtype=float),
                                             "geometry": final.geometry.values}),
            seeds=seeds, inventories=invs,
            blocks=pd.DataFrame({"block_id": blocks["block_id"].astype(str),
                                 "geometry": blocks.geometry.values}),
            event_skill=pd.read_csv(project.paths.resolve(spec["inputs"]["event_skill"])),
            hand=SweepHand(project), meta={"final_run": final_run, "seed_runs": seed_runs},
            greenhouses=(np.asarray(gh.geometry.values, dtype=object) if gh is not None
                         else np.zeros(0, dtype=object)),
            greenhouse_iou=float(settings.v["greenhouse_filter"]["iou_min"]))

    @staticmethod
    def _pool_roles(project: Project, cfg: dict[str, Any], spec: dict[str, Any]
                    ) -> tuple[str, list[str]]:
        runs = [str(s["run"]) for s in cfg["tasks"]["precision_pool"]["runs"]]
        want = S.final_runs(spec)
        final_run, by_train = None, {}
        for run in runs:
            inp = S._inputs(project, run)
            if inp.get("ensemble"):
                final_run = run
            else:
                by_train[str(inp.get("train_run"))] = run
        if final_run is None or sorted(by_train) != sorted(want):
            raise UncertaintyError("precision_pool runs do not match the prereg pool")
        return final_run, [by_train[t] for t in want]


# --- the guarded run ----------------------------------------------------------------------------


def hold_state(project: Project) -> tuple[bool, str]:
    """(held, reason). Fail-safe: no file, a malformed file or anything but ``hold: false``
    with a ``by`` keeps the hold on."""
    path = project.paths.runs / HOLD_FILE
    if not path.is_file():
        return True, f"runs/{HOLD_FILE} missing (the hold is on until the coordinator writes " \
                     "`hold: false` and `by: <name>` there)"
    try:
        doc = load_yaml(path) or {}
    except Exception as error:
        return True, f"runs/{HOLD_FILE} unreadable ({error})"
    if doc.get("hold") is not False or not str(doc.get("by") or "").strip():
        return True, f"runs/{HOLD_FILE} does not release the hold (hold: {doc.get('hold')!r}, " \
                     f"by: {doc.get('by')!r})"
    return False, f"released by {doc['by']}" + (f" at {doc['utc']}" if doc.get("utc") else "")


def guard_problems(project: Project) -> list[str]:
    from nakkhu.evaluate.guard import load_prereg, read_test_journal, verify

    out = [f"prereg: {p}" for p in verify(project)["problems"]]
    try:
        if not (load_prereg(project).get("uncertainty_propagation") or {}).get("config"):
            out.append("prereg has no uncertainty_propagation section (prereg v7)")
    except Exception as error:
        out.append(str(error))
    held, why = hold_state(project)
    if held:
        out.append(f"hold: {why}")
    done = [e for e in read_test_journal(project) if e.get("run") == RUN_NAME]
    if done:
        out.append(f"already run once ({done[0].get('utc')}); the analysis is one test access")
    return out


def run_uncertainty(project: Project, *, dry_run: bool = False, loader: Any = None,
                    spec: dict[str, Any] | None = None,
                    analyse_fn: Callable[[Inputs, dict[str, Any]], dict[str, Any]] = analyse
                    ) -> dict[str, Any]:
    """The one-shot command. Returns ``{"ready", "problems"}`` when refused or with
    ``dry_run``; otherwise the run ref, the journal entry and the promoted artifacts."""
    from nakkhu.artifacts import promote
    from nakkhu.evaluate.guard import record_test_access, require_frozen, sha256_file
    from nakkhu.runs import atomic_write_json, create_run

    spec = S.load_spec(project) if spec is None else spec
    loader = FileInputs() if loader is None else loader
    problems = guard_problems(project)
    problems += loader.problems(project, spec)
    if problems or dry_run:
        return {"ready": not problems, "problems": problems, "ran": False}
    guard = require_frozen(project)
    values = {"command": "uncertainty-run",
              "config_sha256": sha256_file(project.paths.resolve(S.CONFIG)),
              "prereg_sha256": guard["prereg_sha256"]}
    cfg = ResolvedConfig(values=values, source=S.CONFIG)
    access = record_test_access(project, RUN_NAME, cfg.cfghash, guard["prereg_sha256"],
                                guard["split_sha256"], note="nakkhu uncertainty run")
    run = create_run(project, STAGE, cfg)
    with run.step("load"):
        inputs = loader.load(project, spec)
    with run.step("analyse"):
        out = analyse_fn(inputs, spec)
        for name in OUTPUTS:
            obj = out[name]
            path = run.path / name
            if name.endswith(".json"):
                atomic_write_json(path, _jsonable(obj))
            elif name.endswith(".parquet"):
                obj.to_parquet(path, index=False)
            else:
                obj.to_csv(path, index=False)
        fals = out["falsifiers.json"]
        run.write_summary({"test_access": access["entry"], "flagged": access["flagged"],
                           "c2": {"ece_pass": fals["ece"]["pass"],
                                  "ppi_pass": fals["ppi"]["pass"],
                                  "terrain_dominates": fals["decomposition_finding"].get(
                                      "terrain_dominates")}})
    dest = str(spec["outputs"]["dir"]).removeprefix("artifacts/")
    promoted = [str(promote(run.path / n, f"{dest}/{n}", project.paths, replace=True))
                for n in OUTPUTS]
    return {"ready": True, "problems": [], "ran": True, "run": run.ref,
            "test_access": access, "artifacts": promoted}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict("records"))
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if callable(obj):
        return None
    return obj


def write_hold(path: Path, hold: bool, by: str) -> None:
    """Helper for tests and for the coordinator: write the hold file."""
    from nakkhu.runs import atomic_write_text, utc_now

    atomic_write_text(path, f"hold: {str(hold).lower()}\nby: {by}\nutc: '{utc_now()}'\n")
