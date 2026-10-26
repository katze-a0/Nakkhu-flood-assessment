"""``nakkhu followup run --test <id>`` (plan 014 Step 4): one declared Phase-1 test, once.

Order (nothing label-bearing is opened before step 4):

1. label-free prerequisites (:func:`prerequisites`): the test is declared ``access: test`` in
   the addendum and implemented here; ``prereg verify`` has no problem; the coordinator's hold
   (``runs/eval/test_hold.yaml``) is off **and** names this addendum; the journal holds no
   undeclared ``<addendum>:`` entry and no earlier access of this test (unless ``--retry``);
   flood validate's prerequisites hold (sitting 3 complete, estimates current, change maps);
   the promoted prep is intact and was built from the pinned v9 HAND tables; the change maps
   hold the prep's buildings (ids only); test-specific data (UNOSAT zip and its register
   entry for P1-8, the team zips and the promoted P1-10 reproduction check for P1-9 / P1-10, a
   prepared DEM for P1-4-<dem>, a registered reference for P1-5-ext);
2. a run directory ``runs/followup/<utc>-<cfghash>/``;
3. :func:`nakkhu.evaluate.addendum.record_addendum_access` (note ``<addendum>:<test>``) and a
   receipt (steps 1-3: :func:`nakkhu.followup.remote.reserve`, on the laptop);
4. only then, on a remote host and after :func:`nakkhu.followup.remote.verify_receipt`, the
   change-map classes, the audit estimates and answers are read and the test is scored into
   the run directory (:func:`nakkhu.followup.remote.execute`); the laptop verifies the pulled
   run and promotes its CSVs to ``artifacts/followup/``.

This module holds the checks and the scoring; :mod:`nakkhu.followup.remote` the three steps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project, load_yaml
from nakkhu.followup import analyses as A
from nakkhu.followup import spec as S

PHASE1 = ("P1-1", "P1-2", "P1-3", "P1-4", "P1-5", "P1-6", "P1-7", "P1-8", "P1-9", "P1-10")
P1_4_LATE = ("P1-4-srtmgl1", "P1-4-nasadem", "P1-4-aw3d30", "P1-4-hma8m", "P1-4-tandemx12",
             "P1-4-uav")
EXTERNAL = "P1-5-ext"
EXTERNAL_CONFIG = "configs/followup/p1_5_ext.yaml"
OUTPUTS = {"P1-1": ("p1_1_main_river.csv", "p1_1_tiers.csv"),
           "P1-2": ("p1_2_threshold_curve.csv",), "P1-3": ("p1_3_joint.csv", "p1_3_bands.csv"),
           "P1-4": ("p1_4_dems.csv",), "P1-5": ("p1_5_labels.csv",),
           "P1-6": ("p1_6_models.csv", "p1_6_importance.csv"), "P1-7": ("p1_7_metrics.csv",),
           "P1-8": ("p1_8_unosat.csv",), "P1-9": ("p1_9_team_surface.csv",),
           "P1-10": ("p1_10_team_recipe.csv",), EXTERNAL: ("p1_5_ext.csv",)}
SURFACE_TYPES = {"glo30": "DSM", "srtmgl1": "DSM", "nasadem": "DSM", "aw3d30": "DSM",
                 "hma8m": "DSM", "fabdem": "DTM", "gedtm30": "DTM"}


def outputs_of(test: str) -> tuple[str, ...]:
    from nakkhu.followup.explore import EXPLORE_P2
    from nakkhu.followup.explore import OUTPUTS as EXPLORE_OUTPUTS

    if test == EXPLORE_P2:
        return EXPLORE_OUTPUTS
    if test == "P2":
        from nakkhu.followup.confirm import OUTPUTS as P2_OUTPUTS

        return P2_OUTPUTS
    if test == "P2-audit":
        from nakkhu.followup.p2audit import OUTPUTS as AUDIT_OUTPUTS

        return AUDIT_OUTPUTS
    if test in P1_4_LATE:
        return (f"p1_4_{test.removeprefix('P1-4-')}.csv",)
    return OUTPUTS[test]


def implemented(test: str) -> bool:
    return test in PHASE1 or test in P1_4_LATE or test in (EXTERNAL, "P2", "P2-audit")


# --- label-free prerequisites -----------------------------------------------------------------


def journal_problems(project: Project, addendum_id: str, spec: dict[str, Any]) -> list[str]:
    """Every journal entry of the addendum must be a declared test (or ``explore``)."""
    from nakkhu.evaluate.guard import read_test_journal

    declared = set(spec["tests"])
    out = []
    for i, e in enumerate(read_test_journal(project), start=1):
        note = str(e.get("note") or "")
        if not note.startswith(f"{addendum_id}:"):
            continue
        test = note[len(addendum_id) + 1:].removesuffix(" retry")
        if test.startswith("explore ") or test in declared:
            continue
        out.append(f"journal entry {i} has an undeclared note {note!r}")
    return out


def hold_problems(project: Project, addendum_id: str) -> list[str]:
    from nakkhu.uncertainty.run import HOLD_FILE, hold_state

    held, why = hold_state(project)
    if held:
        return [f"hold: {why}"]
    doc = load_yaml(project.paths.runs / HOLD_FILE) or {}
    if str(doc.get("addendum") or "") != addendum_id:
        return [f"hold: runs/{HOLD_FILE} releases the hold but its `addendum` is "
                f"{doc.get('addendum')!r}, not {addendum_id}"]
    return []


def external_config(project: Project) -> dict[str, Any] | None:
    path = project.paths.resolve(EXTERNAL_CONFIG)
    return load_yaml(path) if path.is_file() else None


def prerequisites(project: Project, test: str, cfg: dict[str, Any], spec: dict[str, Any],
                  *, retry: bool = False) -> tuple[list[str], dict[str, Any]]:
    from nakkhu.artifacts import load_index
    from nakkhu.evaluate.addendum import prior_accesses
    from nakkhu.evaluate.guard import verify
    from nakkhu.flood import validate as V
    from nakkhu.followup import prep as P

    addendum_id = cfg["addendum"]
    problems: list[str] = []
    ctx: dict[str, Any] = {}
    from nakkhu.followup.explore import EXPLORE_P2

    if test == EXPLORE_P2:
        # the one Phase-2 preparation explore run (P1-explore rule: after P1-1 to P1-10)
        if "P1-explore" not in spec["tests"]:
            return [f"{addendum_id} declares no P1-explore"], ctx
        for p1 in PHASE1:
            if not prior_accesses(project, addendum_id, p1):
                problems.append(f"explore runs are allowed only after P1-1 to P1-10; "
                                f"{p1} has no journal entry")
        t = {"access": "test"}
    else:
        t = spec["tests"].get(test)
    if t is None:
        return [f"{addendum_id} declares no test {test!r}"], ctx
    if t.get("access") != "test":
        problems.append(f"{test} is access: none (not a journalled test)")
    if not implemented(test) and test != EXPLORE_P2:
        problems.append(f"{test} is not implemented by `nakkhu followup run` (Phase 2 and "
                        "explore runs have their own commands)")
    problems += [f"prereg: {p}" for p in verify(project)["problems"]]
    problems += hold_problems(project, addendum_id)
    problems += journal_problems(project, addendum_id, spec)
    prior = prior_accesses(project, addendum_id, test)
    if prior and not retry:
        problems.append(f"{addendum_id}:{test} already ran ({len(prior)} journal entry)")
    if retry and (len(prior) != 1 or str(prior[0].get("note", "")).endswith(" retry")):
        problems.append(f"{addendum_id}:{test}: a retry needs exactly one earlier, failed "
                        "access and is allowed once")
    problems += [f"validate: {p}" for p in V.prerequisites(project, None, cfg["inventories"])]
    try:
        tables, manifest = P.load_prep(project)
        ctx["prep"], ctx["manifest"] = tables, manifest
    except S.FollowupError as error:
        problems.append(f"prep: {error}")
        return problems, ctx
    if manifest.get("addendum") != addendum_id:
        problems.append(f"prep was built for {manifest.get('addendum')}, not {addendum_id}")
    for inv in cfg["inventories"]:
        want = cfg["hand_tables"][inv]["sha256"]
        if manifest["inputs"]["hand_tables"].get(inv) != want:
            problems.append(f"prep {inv}: HAND table is not the pinned v9 table")
        cm = V.change_map_path(project, inv)
        if cm is None:
            problems.append(f"change map of {inv} missing")
            continue
        ids = pd.read_parquet(cm, columns=["building_id"])["building_id"]
        if set(ids) != set(tables[inv]["building_uid"]):
            problems.append(f"{inv}: change map and prep hold different buildings")
    index = {e["path"] for e in load_index(project.paths)["artifacts"]}
    if test == "P2":
        from nakkhu.followup import confirm

        problems += [f"P2: {p}" for p in confirm.prerequisites(project, addendum_id)]
    if test == "P2-audit":
        from nakkhu.followup import p2audit

        problems += [f"P2-audit: {p}" for p in p2audit.prerequisites(project)]
    if test in ("P1-8",):
        try:
            unosat_zip(project, cfg)
        except S.FollowupError as error:
            problems.append(str(error))
    if test in ("P1-9", "P1-10"):
        try:
            P.team_zip(project, cfg, spec)
        except S.FollowupError as error:
            problems.append(str(error))
        if f"{S.ARTIFACT_DIR}/prep_team_reproduction.csv" not in index:
            problems.append("the P1-10 label-free reproduction check (artifacts/followup/"
                            "prep_team_reproduction.csv) is not promoted")
    if test in P1_4_LATE:
        dem = test.removeprefix("P1-4-")
        if dem not in manifest["dems"]:
            problems.append(f"{dem} is not prepared (add it to {S.PREP_CONFIG} and re-run "
                            "`nakkhu followup prep`)")
    if test == EXTERNAL:
        ext = external_config(project)
        if ext is None:
            problems.append(f"{EXTERNAL_CONFIG} missing: P1-5-ext needs a registered reference "
                            "(provenance, date, licence, sha256) first")
        else:
            for key in ("name", "file", "sha256", "licence", "date", "polygons", "coverage"):
                if not ext.get(key):
                    problems.append(f"{EXTERNAL_CONFIG}: `{key}` is required")
    return problems, ctx


def unosat_zip(project: Project, cfg: dict[str, Any]) -> Path:
    from nakkhu.followup.prep import sha256
    from nakkhu.manifest import load_manifest

    u = cfg["unosat"]
    entry = {e["id"]: e for e in load_manifest(project.paths.resolve(u["manifest"]))}.get(u["id"])
    if entry is None:
        raise S.FollowupError(f"{u['manifest']} has no entry {u['id']}")
    path = project.paths.resolve(entry["local_path"])
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise S.FollowupError(f"UNOSAT zip {path} missing or not the registered sha256")
    return path


# --- labels (only after the journal entry) ------------------------------------------------------


def load_labels(project: Project, inventories: list[str], *, score: bool = False
                ) -> dict[str, Any]:
    from nakkhu.flood import validate as V

    cols = [*V.CHANGE_COLUMNS, *(["score"] if score else [])]
    items, ft, _ = V.audit_answers(project)
    change = {}
    for inv in inventories:
        path = V.change_map_path(project, inv)
        if path is None:
            raise S.FollowupError(f"change map of {inv} missing")
        change[inv] = pd.read_parquet(path, columns=cols)
    return {"estimates": V.load_audit_estimates(project), "items": items, "ft": ft,
            "change": change}


def references(prep: dict[str, pd.DataFrame], labels: dict[str, Any], *,
               keep: tuple[str, ...] = ("R1", "R1-aud", "R1-all-strips")) -> list[A.Ref]:
    """Every reference of each inventory (``validate.inventory_references``), restricted to
    the v9 primary rows (finite hand_v9)."""
    from nakkhu.flood import validate as V

    out = []
    for inv, pred in prep.items():
        change = labels["change"][inv]
        refs, _, _, _ = V.inventory_references(pred, change, labels["estimates"],
                                               labels["items"], labels["ft"])
        for name, (sub, y, w) in refs.items():
            if name not in keep:
                continue
            ok = np.isfinite(sub[A.HAND_V9].to_numpy(dtype=float))
            extra: dict[str, Any] = {}
            if "score" in change.columns:
                smap = dict(zip(change["building_id"], change["score"], strict=True))
                extra["score"] = sub["building_uid"].map(smap).to_numpy(dtype=float)[ok]
            out.append(A.Ref(inv, name, sub.loc[ok].reset_index(drop=True), np.asarray(y)[ok],
                             None if w is None else np.asarray(w)[ok], extra))
    return out


def pick(refs: list[A.Ref], *pairs: tuple[str, str]) -> list[A.Ref]:
    return [r for r in refs if (r.inventory, r.reference) in pairs]


STANDARD = (("microsoft", "R1"), ("detector_final", "R1"), ("microsoft", "R1-aud"))


# --- per-test dispatch --------------------------------------------------------------------------


def score_test(project: Project, test: str, cfg: dict[str, Any], spec: dict[str, Any],
               prep: dict[str, pd.DataFrame], manifest: dict[str, Any], ctx: A.Ctx,
               jobs: int = 1) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    labels = load_labels(project, list(prep), score=test == "P1-5")
    refs = references(prep, labels)
    std = pick(refs, *STANDARD)
    info: dict[str, Any] = {"references": [{**r.tag(), "n": len(r.y), "affected": int(r.y.sum()),
                                            "n_blocks": int(len(set(r.blocks)))} for r in refs]}
    if test == "P1-1":
        main, tiers = A.p1_1(refs, spec, ctx)
        return {"p1_1_main_river.csv": main, "p1_1_tiers.csv": tiers}, info
    if test == "P1-2":
        return {"p1_2_threshold_curve.csv": A.p1_2(std, spec, ctx)}, info
    if test == "P1-3":
        joint = [row for r in std for row in A.p1_3_joint(r, ctx)]
        bands = [row for r in std for row in A.p1_3_bands(r, spec, ctx)]
        return {"p1_3_joint.csv": pd.DataFrame(joint), "p1_3_bands.csv": pd.DataFrame(bands)}, info
    if test == "P1-4" or test in P1_4_LATE:
        dems = list(spec["tests"]["P1-4"]["dems"]) if test == "P1-4" else \
            [test.removeprefix("P1-4-")]
        table = A.p1_4(std, dems, manifest["dems"], S.p1_4_thresholds(spec), ctx, SURFACE_TYPES)
        return {outputs_of(test)[0]: table}, info
    if test == "P1-5":
        return {"p1_5_labels.csv": A.p1_5(audit_ref(prep, labels),
                                          pick(refs, ("microsoft", "R1"),
                                               ("detector_final", "R1")), spec, ctx)}, info
    if test == "P1-6":
        from nakkhu.followup import ml

        ms = pick(refs, ("microsoft", "R1"))[0]
        det = (pick(refs, ("detector_final", "R1")) or [None])[0]
        models, importance, details = ml.p1_6(ms, det, spec, ctx, jobs=jobs)
        info["folds"] = details
        return {"p1_6_models.csv": models, "p1_6_importance.csv": importance}, info
    if test == "P1-7":
        return {"p1_7_metrics.csv": A.p1_7(std, ctx)}, info
    if test == "P1-8":
        return {"p1_8_unosat.csv": p1_8(project, cfg, prep, refs, ctx)}, info
    if test == "P1-9":
        return {"p1_9_team_surface.csv": A.p1_9(std, ctx, _mask_cuts())}, info
    if test == "P1-10":
        t = cfg["team"]
        table = A.p1_10(std, ctx, t["dems"], t["conditionings"])
        repro = project.paths.artifacts / S.ARTIFACT_DIR / "prep_team_reproduction.csv"
        info["label_free_reproduction"] = pd.read_csv(repro).to_dict("records")
        return {"p1_10_team_recipe.csv": table}, info
    if test == EXTERNAL:
        return {"p1_5_ext.csv": p1_5_ext(project, prep, refs, ctx)}, info
    raise S.FollowupError(f"{test} has no scoring code")


def _mask_cuts() -> dict[str, float]:
    from nakkhu.followup.prep import MASK_CUTS

    return dict(MASK_CUTS)


def audit_ref(prep: dict[str, pd.DataFrame], labels: dict[str, Any]) -> A.Ref | None:
    """The answered sitting-3 items (yes / no) of the audited inventory, with their R1 class."""
    from nakkhu.flood import validate as V

    aud = V.r1_aud(labels["items"], labels["ft"])
    if "building_id" not in aud.columns:
        return None
    for inv, pred in prep.items():
        if aud["building_id"].isin(pred["building_uid"]).any():
            sub = pred.merge(aud.rename(columns={"y": "affected", "w": "design_w"}),
                             left_on="building_uid", right_on="building_id")
            change = labels["change"][inv]
            cls = dict(zip(change["building_id"], change["class"].astype(str), strict=True))
            r1 = (sub["building_uid"].map(cls) == "positive").to_numpy(dtype=int)
            return A.Ref(inv, "sitting-3 items", sub, sub["affected"].to_numpy(dtype=int),
                         sub["design_w"].to_numpy(dtype=float),
                         {"r1": r1, "stratum": sub["stratum"].to_numpy()})
    return None


def _footprints(project: Project, cfg: dict[str, Any], inv: str,
                prep: pd.DataFrame) -> Any:
    from nakkhu.hydro.sweep import load_greenhouses, load_inventory, load_settings

    settings, _ = load_settings(project, cfg["sweep_config"])
    extent = settings.read_polygon(settings.v["analysis_extent"])
    frame, _ = load_inventory(settings, inv, extent, load_greenhouses(settings))
    cent = frame.geometry.centroid
    if not ((frame["building_uid"].to_numpy() == prep["building_uid"].to_numpy()).all()
            and np.array_equal(cent.x.to_numpy(), prep["x"].to_numpy())):
        raise S.FollowupError(f"{inv}: inventory differs from the prep buildings")
    return frame


def polygon_hits(footprints: Any, polygons: Any, buffer_m: float = 0.0) -> np.ndarray:
    import geopandas as gpd

    geom = footprints.geometry.buffer(buffer_m) if buffer_m else footprints.geometry
    left = gpd.GeoDataFrame({"_i": np.arange(len(footprints))}, geometry=geom,
                            crs=footprints.crs)
    hit = gpd.sjoin(left, polygons[["geometry"]].to_crs(footprints.crs), how="inner",
                    predicate="intersects")
    out = np.zeros(len(footprints), dtype=int)
    out[np.unique(hit["_i"].to_numpy())] = 1
    return out


def p1_8(project: Project, cfg: dict[str, Any], prep: dict[str, pd.DataFrame],
         refs: list[A.Ref], ctx: A.Ctx) -> pd.DataFrame:
    import geopandas as gpd

    path = unosat_zip(project, cfg)
    u = cfg["unosat"]
    layer = {k: gpd.read_file(S.vsizip(path, f"{u['members_dir']}/{v}.shp"))
             for k, v in u["layers"].items()}
    import shapely

    invalid = {k: int((~f.geometry.is_valid).sum()) for k, f in layer.items()}
    for frame in layer.values():                # the Spot-6 flood MultiPolygon is invalid
        frame["geometry"] = shapely.make_valid(frame.geometry.values)
    t = (S.load_spec(project, cfg["addendum"])["tests"]["P1-8"]["labels"])
    match = re.search(r"within\s+([\d.]+)\s*m", str(t["R-UN-pts"]))
    if match is None:
        raise S.FollowupError("P1-8 R-UN-pts states no radius")
    radius = float(match.group(1))
    inv = "microsoft"
    table = prep[inv]
    fp = _footprints(project, cfg, inv, table)
    crs = fp.crs
    ae = layer["analysis_extent"].to_crs(crs).geometry.union_all()
    inside = gpd.GeoSeries(gpd.points_from_xy(table["x"], table["y"]), crs=crs).within(ae)
    inside = inside.to_numpy()
    r_un = polygon_hits(fp, layer["flood_extent"])
    r_pts = polygon_hits(fp, layer["affected_points"].to_crs(crs).buffer(radius).to_frame(
        "geometry").set_geometry("geometry"))
    un_all = A.Ref(inv, "un_all", table.loc[inside].reset_index(drop=True), r_un[inside], None,
                   {"subset": "un_all"})
    rows: list[dict[str, Any]] = [{"part": "rows", "rows": "un_all", "n": int(inside.sum()),
                                   "invalid_geometries_made_valid": json.dumps(invalid),
                                   "R-UN affected": int(r_un[inside].sum()),
                                   "R-UN-pts affected": int(r_pts[inside].sum())}]
    for name, lab in (("R-UN", r_un), ("R-UN-pts", r_pts)):
        rows += A.external_reference(un_all, lab[inside], name, "un_all", ctx)
    r1 = pick(refs, (inv, "R1"))[0]
    pos = pd.Series(np.arange(len(table)), index=table["building_uid"])
    idx = pos.loc[r1.frame["building_uid"]].to_numpy()
    sel = inside[idx]
    r1_rows = r1.subset(sel, "un_r1rows")
    for name, lab in (("R-UN", r_un), ("R-UN-pts", r_pts)):
        rows += A.external_reference(r1_rows, lab[idx][sel], name, "un_r1rows", ctx)
        rows.append({**A.agreement(lab[idx][sel], r1_rows.y, name, "R1"), "rows": "un_r1rows"})
        rows += [{**r, "rows": "un_r1rows"} for r in A.within_label(r1_rows, lab[idx][sel],
                                                                     name, ctx)]
    out = pd.DataFrame(rows)
    out["caveat"] = ("UNOSAT: not field validated; whether a DEM / HAND mask was used is "
                     "unverified")
    return out


def p1_5_ext(project: Project, prep: dict[str, pd.DataFrame], refs: list[A.Ref],
             ctx: A.Ctx) -> pd.DataFrame:
    import geopandas as gpd

    from nakkhu.followup.prep import sha256

    ext = external_config(project) or {}
    path = project.paths.resolve(ext["file"])
    if sha256(path) != ext["sha256"]:
        raise S.FollowupError(f"{ext['file']} is not the registered sha256")
    polys = gpd.read_file(S.vsizip(path, ext["polygons"]) if path.suffix == ".zip"
                          else path, layer=ext.get("polygons_layer"))
    cover = gpd.read_file(S.vsizip(path, ext["coverage"]) if path.suffix == ".zip"
                          else project.paths.resolve(ext["coverage"]))
    cfg = S.load_prep_config(project)
    rows: list[dict[str, Any]] = []
    for inv, table in prep.items():
        fp = _footprints(project, cfg, inv, table)
        area = cover.to_crs(fp.crs).geometry.union_all()
        inside = gpd.GeoSeries(gpd.points_from_xy(table["x"], table["y"]),
                               crs=fp.crs).within(area).to_numpy()
        lab = polygon_hits(fp, polys)
        rows_ref = A.Ref(inv, "coverage", table.loc[inside].reset_index(drop=True),
                         lab[inside], None, {"subset": "inside coverage"})
        rows += A.external_reference(rows_ref, lab[inside], ext["name"], "inside coverage", ctx)
        r1 = pick(refs, (inv, "R1"))
        if r1:
            pos = pd.Series(np.arange(len(table)), index=table["building_uid"])
            idx = pos.loc[r1[0].frame["building_uid"]].to_numpy()
            sel = inside[idx]
            rows.append({**A.agreement(lab[idx][sel], r1[0].y[sel], ext["name"], "R1"),
                         "inventory": inv})
    return pd.DataFrame(rows).assign(reference=ext["name"], licence=ext["licence"])


# --- the runner -------------------------------------------------------------------------------
#
# There is no in-process "reserve and score" path any more (plan 014, 2026-10-17: the laptop is
# orchestrator only). A test is reserved on the laptop (`nakkhu followup reserve`), executed on
# a remote host from a bundle with its receipt (`nakkhu followup run --receipt`), and promoted on
# the laptop after verification (`nakkhu followup promote` / `merge`): see
# :mod:`nakkhu.followup.remote`.


def run_test(project: Project, test: str, *, retry: bool = False, dry_run: bool = False,
             config: str = S.PREP_CONFIG, **_: Any) -> dict[str, Any]:
    """Label-free dry run of the refusal checks only; anything else is refused."""
    if not dry_run:
        return {"ready": False, "test": test, "problems": [
            "local execution is refused: reserve on the laptop (`nakkhu followup reserve`), "
            "execute remotely with the receipt (`nakkhu followup run --receipt`), then "
            "`nakkhu followup promote` / `merge`"]}
    from nakkhu.followup.remote import reserve

    return reserve(project, test, retry=retry, dry_run=True, config=config)
