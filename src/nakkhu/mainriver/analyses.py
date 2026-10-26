"""The label-reading analyses of the main-river recompute. Called only by
:func:`nakkhu.mainriver.remote.execute` after the receipt is verified (remote host).

- :func:`exposure`: plan 010 U1-U3 + D4 (``nakkhu.uncertainty.run.analyse``, the frozen spec
  ``configs/uncertainty/uncertainty.yaml`` unchanged) with :class:`MainRiverHand` as the HAND
  provider. D4 carries no event AUC (``event_skill: none``).
- :func:`event`: plan 009 V2-V4 / H2-H6 per inventory (``flood.validate.score_inventory``) with
  the main-river members as HAND columns, the H6 ECE and Brier skill score, the UNOSAT
  reference (plan 014 P1-8 labels) with every competitor, and the team recipe scored on its
  own coverage.

Inputs that hold no label (prep tables, rasters, footprints) are loaded by :func:`event_inputs`
and :func:`exposure_inputs`, which :func:`check_inputs` also runs label-free (answers withheld).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project
from nakkhu.mainriver import spec as S

EVENT_TABLES = ("tier_hit_rates", "cumulative_skill", "odds_ratios", "competitors_auc",
                "config_skill_grid", "config_skill_range", "calibration_h6", "h6_reliability",
                "by_inventory", "roc_curves", "pr_curves")
TEAM_KINDS = ("acc", "dig")
TEAM_DEMS = ("glo30", "fabdem", "dem12.5")
TEAM_CONDS = ("fill", "breach_burn")
BUFFER_M = 100.0


# --- exposure (plan 010) ---------------------------------------------------------------------


def uncertainty_spec(project: Project, rc: dict[str, Any]) -> dict[str, Any]:
    from nakkhu.config import load_yaml
    from nakkhu.evaluate.guard import sha256_file

    path = project.paths.resolve(rc["uncertainty_spec"])
    if sha256_file(path) != rc["uncertainty_spec_sha256"]:
        raise S.MainRiverError(f"{rc['uncertainty_spec']} is not the pinned frozen spec")
    return load_yaml(path)


def exposure(project: Project, rc: dict[str, Any], ens: dict[str, Any], hand_run: Path,
             loader: Any = None) -> dict[str, Any]:
    """U1-U3 + D4 outputs (the eight plan 010 files) with main-river HAND."""
    from nakkhu.mainriver.provider import MainRiverHand
    from nakkhu.uncertainty.run import FileInputs, analyse

    spec = uncertainty_spec(project, rc)
    inp = (loader or FileInputs()).load(project, spec)
    empty = pd.DataFrame(columns=["inventory", "reference", "auc", "pr_auc"])
    inp = dataclasses.replace(inp, hand=MainRiverHand(project, hand_run, ens),
                              event_skill=empty)
    return analyse(inp, spec)


# --- event (plan 009) ------------------------------------------------------------------------


def hand_table(project: Project, hand_run: Path, inventory: str) -> pd.DataFrame:
    return pd.read_parquet(hand_run / f"building_hand_mr_{inventory}.parquet")


def surface_meta(hand_run: Path, inventory: str) -> pd.DataFrame:
    cs = pd.read_csv(hand_run / "count_stability.csv")
    cs = cs.loc[cs["inventory"].astype(str) == inventory]
    cs = cs.assign(surface=cs["cfg_id"].astype(str) + "__" + cs["statistic"].astype(str))
    cols = ["surface", "cfg_id", "dem", "conditioning", "threshold", "threshold_km2",
            "statistic", "diagnostic", "routing_gate_pass", "in_sensitivity_range", "ensemble"]
    return cs[[c for c in cols if c in cs.columns]].drop_duplicates("surface")


def predictors(project: Project, rc: dict[str, Any], hand_run: Path, inventory: str
               ) -> pd.DataFrame:
    """Per building (v9 order): the prep core columns (x, y, block, distance, hand_v9), the
    GLO-30 elevation and ASF HAND competitors (``validate._sample_raster``, the corrected
    sampler of journal 51) and every main-river member column."""
    from nakkhu.flood import validate as V

    core = pd.read_parquet(project.paths.artifacts / "followup" / f"prep_core_{inventory}.parquet")
    mr = hand_table(project, hand_run, inventory)
    if not np.array_equal(core["building_uid"].astype(str).to_numpy(),
                          mr["building_uid"].astype(str).to_numpy()):
        raise S.MainRiverError(f"{inventory}: prep core and hand-mr building order differ")
    x, y = core["x"].to_numpy(dtype=float), core["y"].to_numpy(dtype=float)
    pred = core.rename(columns={"distance": "dist_channel_m"}).copy()
    root = project.paths.data_root
    pred["elev_glo30_m"] = V._sample_raster(root / V.ELEVATION, x, y, "EPSG:32645")
    pred["asf_hand_m"] = V._sample_raster(root / V.ASF_HAND, x, y, "EPSG:32645")
    members = [c for c in mr.columns if c.endswith(("__centroid", "__footprint_min",
                                                     "__buffer_min"))]
    for c in members:
        pred[c] = mr[c].to_numpy(dtype=np.float32)
    return pred


def team_columns(project: Project, inventory: str) -> pd.DataFrame:
    path = project.paths.artifacts / "followup" / f"prep_team_{inventory}.parquet"
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def event_inputs(project: Project, rc: dict[str, Any], ens: dict[str, Any], hand_run: Path
                 ) -> dict[str, Any]:
    """Everything the event analysis reads that holds no label."""
    out: dict[str, Any] = {"pred": {}, "meta": {}, "team": {}}
    for inv in rc["analyses"]["event"]["inventories"]:
        out["pred"][inv] = predictors(project, rc, hand_run, inv)
        out["meta"][inv] = surface_meta(hand_run, inv)
        out["team"][inv] = team_columns(project, inv)
    return out


def ece(p: np.ndarray, y: np.ndarray, w: np.ndarray | None, bins: int) -> float:
    """Expected calibration error over equal-count bins of p (design-weighted with w)."""
    from nakkhu.flood.validate import reliability

    ok = np.isfinite(p)
    sw = np.ones(int(ok.sum())) if w is None else np.asarray(w, dtype=float)[ok]
    rel = reliability(p, y, bins, w=w)
    order = np.argsort(np.asarray(p, dtype=float)[ok], kind="stable")
    parts = np.array_split(order, bins)
    wsum = np.array([sw[ix].sum() for ix in parts if len(ix)])
    gap = np.abs(rel["p_mean"].to_numpy() - rel["observed"].to_numpy())
    return float(np.sum(wsum * gap) / wsum.sum())


def h6_skill(sub: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, pcol: str,
             bins: int) -> list[dict[str, Any]]:
    """ECE and Brier skill score of the H6 out-of-fold probabilities (same fits as
    ``validate.h6_calibration``), vs a constant forecast at the (weighted) prevalence."""
    from nakkhu import hazard
    from nakkhu.flood import validate as V

    sub, y, w, hand = V._primary_rows(sub, y, w, pcol)
    blocks = sub["block_id"].to_numpy()
    prev = float(np.average(y, weights=w))
    rows = []
    for method in V.H6_METHODS:
        p = hazard.calibrate_out_of_fold(hand, y.astype(float), blocks, method, w)
        ok = np.isfinite(p)
        wk = None if w is None else np.asarray(w)[ok]
        b = hazard.brier(p[ok], y[ok], wk)
        p_ok = float(np.average(y[ok], weights=wk))
        rows.append({"method": method, "n_scored": int(ok.sum()), "brier": b,
                     "prevalence": prev, "prevalence_scored": p_ok,
                     "brier_reference": p_ok * (1 - p_ok),
                     "bss": 1.0 - b / (p_ok * (1 - p_ok)),
                     "ece": ece(p, y.astype(float), w, bins), "ece_bins": bins,
                     "weighted": w is not None})
    return rows


def labels_of(project: Project, inventories: list[str]) -> dict[str, Any]:
    """The label-bearing inputs (change maps, sitting-3 answers, flood_trace estimates)."""
    from nakkhu.flood import validate as V

    items, ft, dmg = V.audit_answers(project)
    change = {}
    for inv in inventories:
        path = V.change_map_path(project, inv)
        if path is None:
            raise S.MainRiverError(f"change map of {inv} missing")
        change[inv] = pd.read_parquet(path, columns=V.CHANGE_COLUMNS)
    return {"items": items, "ft": ft, "dmg": dmg, "change": change,
            "estimates": V.load_audit_estimates(project)}


def _scores(frame: pd.DataFrame, pcol: str) -> dict[str, np.ndarray]:
    d = frame["dist_channel_m"].to_numpy(dtype=float)
    return {"hand_mr": -frame[pcol].to_numpy(dtype=float),
            "distance_to_channel": -d,
            "river_buffer_100m": (d <= BUFFER_M).astype(float),
            "glo30_elevation": -frame["elev_glo30_m"].to_numpy(dtype=float),
            "asf_glo30_hand": -frame["asf_hand_m"].to_numpy(dtype=float),
            "hand_v9": -frame["hand_v9"].to_numpy(dtype=float)}


def compare_all(frame: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, pcol: str,
                ev: dict[str, Any], **tags: Any) -> list[dict[str, Any]]:
    """hand_mr vs every competitor: AUCs and paired block-bootstrap delta AUC."""
    from nakkhu.followup import stats as ST

    sc = _scores(frame, pcol)
    blocks = frame["block_id"].astype(str).to_numpy()
    rows = []
    for comp in ("distance_to_channel", "river_buffer_100m", "glo30_elevation",
                 "asf_glo30_hand", "hand_v9"):
        r = ST.pair(y, sc["hand_mr"], sc[comp], blocks, w, int(ev["n_boot"]), int(ev["seed"]),
                    int(ev["min_cell"]))
        rows.append({**tags, "candidate": "hand_mr", "candidate_column": pcol,
                     "comparator": comp, **r})
    return rows


def unosat(project: Project, rc: dict[str, Any], pred: pd.DataFrame, r1: tuple | None,
           pcol: str, ev: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The UNOSAT FL20240928NPL reference R-UN (P1-8 labels) on Microsoft: rows = buildings
    with centroid in the UNOSAT analysis extent (un_all) and the R1 rows among them."""
    import geopandas as gpd
    import shapely

    from nakkhu.followup import run as FR
    from nakkhu.followup import spec as FS

    cfg = FS.load_prep_config(project, rc["prep_config"])
    path = FR.unosat_zip(project, cfg)
    u = cfg["unosat"]
    layer = {k: gpd.read_file(FS.vsizip(path, f"{u['members_dir']}/{v}.shp"))
             for k, v in u["layers"].items()}
    for frame in layer.values():
        frame["geometry"] = shapely.make_valid(frame.geometry.values)
    fp = FR._footprints(project, cfg, "microsoft", pred)
    ae = layer["analysis_extent"].to_crs(fp.crs).geometry.union_all()
    inside = gpd.GeoSeries(gpd.points_from_xy(pred["x"], pred["y"]), crs=fp.crs).within(ae)
    inside = inside.to_numpy()
    lab = FR.polygon_hits(fp, layer["flood_extent"])
    info = {"n_microsoft": len(pred), "n_un_all": int(inside.sum()),
            "affected_un_all": int(lab[inside].sum()),
            "n_outside_analysis_extent": int((~inside).sum()),
            "rows_rule": "building centroid within the UNOSAT analysis extent (P1-8 un_all)"}
    rows = compare_all(pred.loc[inside].reset_index(drop=True), lab[inside], None, pcol, ev,
                       inventory="microsoft", reference="R-UN", rows="un_all")
    if r1 is not None:
        sub, _, _ = r1
        pos = pd.Series(np.arange(len(pred)), index=pred["building_uid"])
        idx = pos.loc[sub["building_uid"]].to_numpy()
        sel = inside[idx]
        info.update({"n_un_r1rows": int(sel.sum()), "affected_un_r1rows": int(lab[idx][sel].sum())})
        rows += compare_all(sub.loc[sel].reset_index(drop=True), lab[idx][sel], None, pcol, ev,
                            inventory="microsoft", reference="R-UN", rows="un_r1rows")
    return rows, info


def team_rows(sub: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, team: pd.DataFrame,
              pcol: str, ev: dict[str, Any], **tags: Any) -> list[dict[str, Any]]:
    """Each team-recipe member on its OWN finite rows within the reference: vs distance and
    vs main-river HAND on exactly those rows."""
    from nakkhu.followup import stats as ST

    if team.empty:
        return []
    t = sub[["building_uid"]].merge(team, on="building_uid", how="left")
    blocks = sub["block_id"].astype(str).to_numpy()
    d = -sub["dist_channel_m"].to_numpy(dtype=float)
    h = -sub[pcol].to_numpy(dtype=float)
    rows = []
    for kind in TEAM_KINDS:
        for dem in TEAM_DEMS:
            for cnd in TEAM_CONDS:
                col = f"hand_team_{kind}__{dem}__{cnd}"
                if col not in t.columns:
                    continue
                v = t[col].to_numpy(dtype=float)
                own = np.isfinite(v) & np.isfinite(d) & np.isfinite(h)
                for comp, s2 in (("distance_to_channel", d), ("hand_mr", h)):
                    r = ST.pair(np.asarray(y)[own], -v[own], s2[own], blocks[own],
                                None if w is None else np.asarray(w)[own], int(ev["n_boot"]),
                                int(ev["seed"]), int(ev["min_cell"]))
                    rows.append({**tags, "member": col, "recipe": kind, "dem": dem,
                                 "conditioning": cnd, "rows": "own coverage",
                                 "n_reference_rows": len(v), "comparator": comp, **r})
    return rows


def event(project: Project, rc: dict[str, Any], ens: dict[str, Any], hand_run: Path,
          inputs: dict[str, Any] | None = None, labels: dict[str, Any] | None = None
          ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    from nakkhu import hazard
    from nakkhu.flood import validate as V

    ev = rc["analyses"]["event"]
    invs = [str(v) for v in ev["inventories"]]
    inputs = inputs or event_inputs(project, rc, ens, hand_run)
    labels = labels or labels_of(project, invs)
    pcol = S.primary_column(ens)
    edges = [float(e) for e in ens["tiers_m"]]
    tl = hazard.tier_labels(edges)
    tables: dict[str, list[pd.DataFrame]] = {k: [] for k in V.VALIDATION_ARTIFACTS}
    skill, un_rows, team, damage = [], [], [], []
    summary: dict[str, Any] = {"primary_column": pcol, "inventories": {},
                               "reporting": "post hoc (main-river HAND chosen after the event "
                                            "labels were seen, plan 014 P1-1)"}
    for inv in invs:
        pred, change = inputs["pred"][inv], labels["change"][inv]
        differ = set(change["building_id"]) ^ set(pred["building_uid"])
        if differ:
            raise S.MainRiverError(f"{inv}: change map and HAND table hold different buildings")
        inv_sum, parts = V.score_inventory(inv, pred, change, labels["estimates"],
                                           labels["items"], labels["ft"], labels["dmg"], pcol,
                                           tl, edges, int(ev["n_boot"]), int(ev["seed"]),
                                           inputs["meta"][inv])
        summary["inventories"][inv] = inv_sum
        for key, part in parts.items():
            tables[key] += part
        refs, _, _, _ = V.inventory_references(pred, change, labels["estimates"],
                                               labels["items"], labels["ft"])
        for ref, (sub, y, w) in refs.items():
            tag = {"inventory": inv, "reference": ref}
            skill += [{**tag, **r} for r in h6_skill(sub, y, w, pcol, int(ev["h6_ece_bins"]))]
            team += team_rows(sub, y, w, inputs["team"][inv], pcol, ev, **tag,
                              weighted=w is not None)
        if "V2_damage" in inv_sum:
            damage += [{"inventory": inv, **r} for r in inv_sum["V2_damage"]["by_tier"]]
        if inv == "microsoft":
            rows, info = unosat(project, rc, pred, refs.get("R1"), pcol, ev)
            un_rows += rows
            summary["unosat"] = info
    summary["primary"] = V.primary_outcome(summary, "microsoft")
    summary["primary_detector_final"] = V.primary_outcome(summary, "detector_final")
    out = {f"{k}.csv": (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
           for k, v in tables.items()}
    out["h6_skill.csv"] = pd.DataFrame(skill)
    out["unosat_competitors.csv"] = pd.DataFrame(un_rows)
    out["team_own_coverage.csv"] = pd.DataFrame(team)
    out["damage_by_tier.csv"] = pd.DataFrame(damage)
    return out, V.jsonable(summary)


# --- roc: ROC curves of main-river HAND and distance to the channel (figure) -----------------

ROC_PREDICTORS = ("hand_mr", "distance_to_channel")


def roc_inputs(project: Project, rc: dict[str, Any], ens: dict[str, Any], hand_run: Path
               ) -> dict[str, pd.DataFrame]:
    """Label-free, per inventory: building_uid, block_id, distance to the channel (prep core,
    as :func:`predictors`) and the primary main-river HAND column."""
    pcol = S.primary_column(ens)
    out = {}
    for inv in rc["analyses"]["roc"]["inventories"]:
        core = pd.read_parquet(project.paths.artifacts / "followup" /
                               f"prep_core_{inv}.parquet",
                               columns=["building_uid", "block_id", "distance"])
        mr = pd.read_parquet(hand_run / f"building_hand_mr_{inv}.parquet",
                             columns=["building_uid", pcol])
        if not np.array_equal(core["building_uid"].astype(str).to_numpy(),
                              mr["building_uid"].astype(str).to_numpy()):
            raise S.MainRiverError(f"{inv}: prep core and hand-mr building order differ")
        out[inv] = core.rename(columns={"distance": "dist_channel_m"}).assign(
            **{pcol: mr[pcol].to_numpy(dtype=np.float32)})
    return out


def roc_predictor_curves(project: Project, rc: dict[str, Any], ens: dict[str, Any],
                         hand_run: Path, inputs: dict[str, pd.DataFrame] | None = None,
                         change: dict[str, pd.DataFrame] | None = None,
                         estimates: dict[str, Any] | None = None
                         ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full ROC curves (every distinct score) of main-river HAND and distance to the channel on
    the R1 change-map reference, unweighted, per inventory. Rows and scores are those of the
    event analysis (``validate._primary_rows``, then the finite rows of each predictor, as
    ``validate.competitor_rows``), so each curve's area equals the promoted AUC."""
    from nakkhu.flood import validate as V

    cfg = rc["analyses"]["roc"]
    invs = [str(v) for v in cfg["inventories"]]
    inputs = inputs or roc_inputs(project, rc, ens, hand_run)
    if change is None:
        change = {}
        for inv in invs:
            path = V.change_map_path(project, inv)
            if path is None:
                raise S.MainRiverError(f"change map of {inv} missing")
            change[inv] = pd.read_parquet(path, columns=["building_id", "class", "observable"])
    estimates = estimates or V.load_audit_estimates(project)
    pcol = S.primary_column(ens)
    parts, curves = [], []
    for inv in invs:
        pred, cm = inputs[inv], change[inv]
        if set(cm["building_id"]) ^ set(pred["building_uid"]):
            raise S.MainRiverError(f"{inv}: change map and HAND table hold different buildings")
        r1 = V.r1_reference(cm, estimates)
        if not r1["usable"]:
            raise S.MainRiverError(f"{inv}: the drop rule removes the change map (R1)")
        ymap = dict(zip(r1["building_id"], r1["y"], strict=True))
        sub = pred.loc[pred["building_uid"].isin(set(r1["building_id"]))].reset_index(drop=True)
        y = sub["building_uid"].map(ymap).to_numpy(dtype=int)
        sub, y, _, hand = V._primary_rows(sub, y, None, pcol)
        scores = {"hand_mr": -hand, "distance_to_channel": -sub["dist_channel_m"].to_numpy(
            dtype=float)}
        for name in ROC_PREDICTORS:
            s = scores[name]
            good = np.isfinite(s)
            tag = {"inventory": inv, "reference": "R1", "predictor": name}
            parts.append(V.roc_curve(s[good], y[good]).assign(**tag))
            curves.append({**tag, "n": int(good.sum()), "affected": int(y[good].sum()),
                           "auc": V.roc_auc(s[good], y[good]),
                           "score": "-HAND (m)" if name == "hand_mr" else "-distance (m)"})
    summary = {"primary_column": pcol, "reference": "R1 (change-map class, observable "
               "buildings, unweighted)", "curves": curves,
               "reporting": "post hoc (figure curves of the promoted main_river_v1:event AUCs)"}
    return pd.concat(parts, ignore_index=True), summary


# --- label-free input check (remote host, before the reservation) --------------------------


def check_inputs(project: Project, rc: dict[str, Any], ens: dict[str, Any], hand_run: Path
                 ) -> dict[str, Any]:
    """Load every input of both analyses with the answers WITHHELD (audit responses read as
    empty; only the building ids of the change maps), and record every file opened."""
    import sys

    from nakkhu.audit import core as audit_core
    from nakkhu.flood import validate as V
    from nakkhu.mainriver.provider import MainRiverHand
    from nakkhu.uncertainty.run import FileInputs

    opened: set[str] = set()
    root = str(project.paths.root.resolve())

    def hook(event: str, args: tuple) -> None:
        if event == "open" and args and isinstance(args[0], (str, Path)):
            p = str(Path(args[0]).resolve()) if not str(args[0]).startswith("/vsi") \
                else str(args[0])
            if p.startswith(root):
                opened.add(Path(p).relative_to(root).as_posix())

    sys.addaudithook(hook)
    real = audit_core.read_events

    def withheld(path: Any) -> list:
        if Path(path).name.startswith("responses"):
            return []
        return real(path)

    setattr(audit_core, "read_events", withheld)  # noqa: B010
    out: dict[str, Any] = {"ok": True, "problems": []}
    try:
        spec = uncertainty_spec(project, rc)
        inp = FileInputs().load(project, spec)
        hand = MainRiverHand(project, hand_run, ens)
        surf = hand.surfaces()
        out["exposure"] = {"precision_units": len(inp.prec), "recall_items": len(inp.recall),
                           "frame_units": len(inp.frame), "detector_final": len(inp.final),
                           "seeds": [len(s) for s in inp.seeds],
                           "inventories": {k: len(v) for k, v in inp.inventories.items()},
                           "inv_prec": {k: len(v) for k, v in inp.inv_prec.items()},
                           "greenhouses": len(inp.greenhouses), "blocks": len(inp.blocks),
                           "surfaces": surf.to_dict("records"),
                           "answers_withheld": bool(inp.prec["answer"].isna().all())}
        ev_in = event_inputs(project, rc, ens, hand_run)
        out["event"] = {}
        for inv, pred in ev_in["pred"].items():
            cm = V.change_map_path(project, inv)
            ids = pd.read_parquet(cm, columns=["building_id"]) if cm is not None else None
            out["event"][inv] = {
                "n": len(pred), "members": int(sum("__" in c for c in pred.columns)),
                "elev_finite": int(np.isfinite(pred["elev_glo30_m"]).sum()),
                "asf_finite": int(np.isfinite(pred["asf_hand_m"]).sum()),
                "team_columns": len(ev_in["team"][inv].columns),
                "change_map_same_buildings": None if ids is None else
                bool(set(ids["building_id"]) == set(pred["building_uid"]))}
        from nakkhu.followup import run as FR
        from nakkhu.followup import spec as FS

        FR.unosat_zip(project, FS.load_prep_config(project, rc["prep_config"]))
        for task in ("flood_trace",):
            if not audit_core.responses_path(project, task).is_file():
                out["problems"].append(f"audit/{task}/responses.jsonl missing")
        if not (project.paths.artifacts / V.ESTIMATES).is_file():
            out["problems"].append("flood_trace estimates missing")
    except Exception as error:   # report, never raise: the check is diagnostic
        import traceback

        out["ok"] = False
        out["problems"].append(f"{type(error).__name__}: {error}")
        out["traceback"] = traceback.format_exc()
    finally:
        setattr(audit_core, "read_events", real)  # noqa: B010
    out["ok"] = out["ok"] and not out["problems"]
    out["files_opened"] = sorted(opened)
    return out
