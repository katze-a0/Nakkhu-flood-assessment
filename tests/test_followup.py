"""Plan 014 follow-up code (prereg_v9_a3): label-free predictors, statistics, every Phase-1 test
path and the runner's guard order, on synthetic data."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from affine import Affine
from shapely.geometry import LineString, Point, box

from nakkhu.audit import core as audit_core
from nakkhu.config import ResolvedConfig, load_project
from nakkhu.evaluate import addendum as addendum_mod
from nakkhu.evaluate import guard
from nakkhu.flood import validate as V
from nakkhu.followup import ml
from nakkhu.followup import prep as P
from nakkhu.followup import remote as RM
from nakkhu.followup import run as R
from nakkhu.followup import spec as S
from nakkhu.followup import stats as ST
from nakkhu.followup import surfaces as F
from nakkhu.hydro import route
from nakkhu.hydro.dem import GridSpec

REAL_ROOT = Path(__file__).resolve().parents[1]
REAL_A3 = REAL_ROOT / "configs" / "prereg_addenda" / "prereg_v9_a3.yaml"
ADD = "prereg_v1_a1"


def grid(h: int = 6, w: int = 7, res: float = 30.0) -> GridSpec:
    return GridSpec(crs="EPSG:32645", transform=Affine(res, 0, 330000, 0, -res, 3060000),
                    width=w, height=h)


# --- surfaces ---------------------------------------------------------------------------------


def test_walk_downstream_first_cell_and_path_length():
    g = grid(3, 4)
    fdir = np.array([[2, 1, 1, 4], [1, 1, 1, 4], [1, 1, 1, 0]])   # SE, E ... into col 3, S
    ds = route.downstream_index(fdir)
    target = np.zeros((3, 4), bool)
    target[2, 3] = True
    first, length = F.walk_downstream(ds, target, np.array([0, 4, 11, -1]), g.width, g.res)
    assert first.tolist() == [11, 11, 11, -1]
    # cell 0 -> 5 (diagonal) -> 6 -> 7 -> 11 ; cell 4 -> 5 -> 6 -> 7 -> 11
    assert length[0] == pytest.approx(30 * np.sqrt(2) + 3 * 30)
    assert length[1] == pytest.approx(4 * 30) and length[2] == 0 and np.isnan(length[3])
    none = np.zeros((3, 4), bool)
    f2, l2 = F.walk_downstream(ds, none, np.array([0]), g.width, g.res)
    assert f2[0] == -1 and np.isnan(l2[0])


def test_main_mask_hand_above_and_rem():
    g = grid(5, 5)
    line = gpd.GeoDataFrame(geometry=[LineString([(330000 + 75, 3060000 - 1),
                                                  (330000 + 75, 3060000 - 149)])],
                            crs=g.crs)
    acc = np.full(g.shape, 2000.0)
    acc[:, 2][:2] = 10.0                                  # small area: dropped from M
    main = F.main_mask(line, g, acc, 1.0)
    assert main[:, 2][2:].all() and not main[:2, 2].any() and main.sum() >= 3
    z = np.arange(25, dtype=float).reshape(5, 5)
    h = F.hand_above(z, np.array([24, 3, 5]), np.array([12, 12, -1]))
    assert h[0] == 12 and h[1] == 0 and np.isnan(h[2])      # negative clamped, NaN path
    z_raw = np.full(g.shape, 100.0)
    z_raw[4, 2] = 90.0
    rem = F.hand_mr_rem(g, main, z_raw, np.array([2 * 5 + 0, 4 * 5 + 4, -1]), 60.0)
    assert rem[1] == pytest.approx(10.0)                  # nearest M (4,2): min within 60 m
    assert rem[0] == pytest.approx(10.0) and np.isnan(rem[2])      # (2,2) reaches (4,2)
    near = F.hand_mr_rem(g, main, z_raw, np.array([2 * 5 + 0]), 30.0)
    assert near[0] == pytest.approx(0.0)                  # 30 m: (4,2) is out of reach


def test_terrain_features_on_a_plane():
    g = grid(9, 9)
    rows, cols = np.indices(g.shape)
    z = 100.0 + 0.1 * cols * g.res                       # 10 % slope rising east
    slope = F.horn_slope_deg(z, g.res)
    assert slope[4, 4] == pytest.approx(np.degrees(np.arctan(0.1)))
    plan, profile = F.zt_curvatures(z, g.res)
    assert abs(plan[4, 4]) < 1e-12 and abs(profile[4, 4]) < 1e-12
    assert abs(F.tpi(z, g.res, 60.0)[4, 4]) < 1e-9
    bowl = 100.0 + ((rows - 4) ** 2 + (cols - 4) ** 2) * 1.0
    assert F.tpi(bowl, g.res, 60.0)[4, 4] < 0                 # a pit sits below its ring
    twi = F.twi(np.full(g.shape, 4.0), g, np.full(g.shape, 45.0))
    assert twi[0, 0] == pytest.approx(np.log(4 * 900 / 30 / 1.0))
    assert F.count_within(np.array([0, 50, 300.0]), np.zeros(3), 100).tolist() == [1, 1, 0]


def test_team_recipe_fill_drains_a_clipped_basin_to_its_outlet():
    g = grid(12, 12)
    rows, cols = np.indices(g.shape)
    z = 100.0 + np.abs(cols - 6) * 2.0 + (11 - rows) * 1.0          # valley flowing south
    z[5, 6] = 99.0 + 6                                              # a pit to fill
    basin = (cols >= 1) & (cols <= 10) & (rows >= 1)
    outlet = 11 * 12 + 6
    dig = np.zeros(g.shape, bool)
    dig[:, 6] = True
    res = F.team_recipe(z, g, basin, "fill", burn=None, burn_depth_m=60, max_dist_cells=10,
                        stream_area_km2=5 * g.cell_area_m2 / 1e6, digitized=dig, outlet=outlet)
    assert res["acc"].ravel()[outlet] == basin.sum()                # everything drains out
    assert np.isnan(res["hand_acc"][~basin]).all()
    assert np.isfinite(res["hand_acc"][basin]).all()
    assert (res["hand_dig"][basin][dig[basin]] == 0).all()
    assert res["z_ref"].ravel()[outlet] == z.ravel()[outlet]        # outlet reference restored
    agree = F.tier_agreement(res["hand_acc"][basin], res["hand_acc"][basin], [0, 3, 6, 9, 12])
    assert agree["kappa_linear"] == 1.0 and agree["median_abs_diff_m"] == 0
    with pytest.raises(ValueError):
        F.team_recipe(z, g, basin, "carve", burn=None, burn_depth_m=60, max_dist_cells=10,
                      stream_area_km2=1, digitized=dig)


# --- statistics -------------------------------------------------------------------------------


def _toy(n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    h = rng.gamma(2, 4, n)
    d = h * 20 + rng.uniform(0, 200, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1 - 0.4 * h)))).astype(int)
    blocks = np.array([f"b{i % 9}" for i in range(n)])
    return h, d, y, blocks


def test_block_bootstrap_shares_validate_resamples():
    h, d, y, blocks = _toy()
    bb = V.block_bootstrap_delta_auc(y, -h, -d, blocks, 200, 0)
    mine = ST.block_bootstrap(lambda i: V.roc_auc(-h[i], y[i]) - V.roc_auc(-d[i], y[i]),
                              blocks, 200, 0)
    lo, hi = ST.percentile_ci(mine)
    assert (lo, hi) == pytest.approx((bb["boot_lo"], bb["boot_hi"]))
    assert mine.size == bb["n_boot_valid"]


def test_pair_min_cell_and_delong():
    h, d, y, blocks = _toy()
    row = ST.pair(y, -h, -d, blocks, None, 50, 0, 10)
    assert row["scored"] and row["delong_p"] is not None and row["n"] == len(y)
    assert row["auc_candidate"] == pytest.approx(V.roc_auc(-h, y))
    few = np.zeros_like(y)
    few[:5] = 1
    assert not ST.pair(few, -h, -d, blocks, None, 50, 0, 10)["scored"]
    hn = h.copy()
    hn[:7] = np.nan
    r2 = ST.pair(y, -hn, -d, blocks, np.ones(len(y)), 20, 0, 10)
    assert r2["n_nan_candidate"] == 7 and r2["n"] == len(y) - 7 and r2["delong_p"] is None


def test_splines_logistic_and_metrics():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 500)
    k = ST.ns_knots(x)
    B = ST.ns_basis(x, k)
    assert B.shape == (500, 4) and np.allclose(B[:, 0], (np.clip(x, k[0], k[-1]) - k[0])
                                                / (k[-1] - k[0]))
    # natural spline: linear beyond the boundary knots (clamped -> constant)
    assert np.allclose(ST.ns_basis(np.array([k[-1] + 5]), k), ST.ns_basis(np.array([k[-1]]), k))
    y = (rng.random(500) < 1 / (1 + np.exp(-(x - 5)))).astype(float)
    X = np.column_stack([np.ones(500), B])
    b0 = ST.logistic_fit(X, y)
    b1 = ST.logistic_fit(X, y, lam=100.0)
    assert np.abs(b1[1:]).sum() < np.abs(b0[1:]).sum()
    cov = V.cluster_robust_cov(X, y, b0, np.arange(500) % 10)
    assert ST.wald_chi2(b0, cov, [1, 2, 3, 4])["p"] < 0.01
    yb = y.astype(bool)
    assert ST.standardised_pauc(np.where(yb, 1.0, 0.0), yb) == pytest.approx(1.0)
    assert 0.3 < ST.standardised_pauc(rng.random(500), yb) < 0.7
    assert ST.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert ST.ece(np.full(500, y.mean()), y) == pytest.approx(0.0, abs=0.06)
    band = np.where(x < 5, "a", "b")
    s = ST.stratified_auc(x, yb, band, ["a", "b"])
    assert 0.5 < s < 1.0
    sk = ST.binary_skill(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (sk["H"], sk["FAR"], sk["CSI"]) == (0.5, 0.5, pytest.approx(1 / 3))


# --- the real addendum's settings ------------------------------------------------------------


def test_spec_reads_the_frozen_addendum_settings():
    spec = yaml.safe_load(REAL_A3.read_text(encoding="utf-8"))
    assert len(S.candidates_km2(spec)) == 16 and S.candidates_km2(spec)[4] == 0.2
    assert S.p1_4_thresholds(spec) == [0.2, 0.5, 1.0, 2.0]
    assert S.bootstrap_settings(spec) == (2000, 0) and S.min_cell(spec) == 10
    assert S.tier_edges(spec) == [0, 3, 6, 9, 12]
    ml.check_spec(spec)
    configs = ml.hgb_configs(spec)
    assert len(configs) == 60 and configs == ml.hgb_configs(spec)
    assert {c["max_features"] for c in configs} <= {0.5, 1.0}
    assert ml.gam_lambdas(spec) == [0.001, 0.01, 0.1, 1, 10, 100]
    assert S.tlabel(0.075) == "t0p075" and S.tlabel(1.0) == "t1"
    assert sum(c == -1 for c in ml.monotone_cst()) == 14


def test_prep_and_runner_modules_read_no_labels():
    """Prep code never names a change-map class, observability or audit answer; the runner
    reads labels only in load_labels / audit_ref (after the journal entry)."""
    import inspect

    text = inspect.getsource(P) + inspect.getsource(F)
    for word in ("audit_answers", "load_audit_estimates", "\"class\"", "observable",
                 "responses"):
        assert word not in text, word
    res = inspect.getsource(RM.reserve)
    assert "record_addendum_access(" in res and "score_test" not in res
    assert "load_labels" not in res and "load_labels" not in inspect.getsource(R.prerequisites)
    ex = inspect.getsource(RM.execute)
    assert ex.index("verify_receipt(") < ex.index("score_test(")
    sh = inspect.getsource(RM.execute_shards)
    assert "_p1_6_data(" in sh and "verify_receipt" not in sh   # only reached via execute
    for fn in (RM.verify_receipt, RM.execute, RM.execute_shards, RM.bundle_problems):
        assert "record_test_access" not in inspect.getsource(fn)
        assert "record_addendum_access" not in inspect.getsource(fn)


# --- a synthetic project with frozen prereg + addendum, prep, labels ----------------------------


def _neutralised_addendum(base_sha: str) -> str:
    doc = yaml.safe_load(REAL_A3.read_text(encoding="utf-8"))

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {({"sha256_of": "file", "mirror_of": "mirror", "constant_of": "constant"}
                     .get(k, k)): strip(v) for k, v in node.items()}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    doc = strip(doc)
    doc.update({"addendum": ADD, "base_version": "prereg_v1", "base_sha256": base_sha})
    doc.pop("supersedes", None)
    doc.pop("supersede_reason", None)
    return yaml.safe_dump(doc, sort_keys=False)


THR = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5]


def _world(inv: str, n: int, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 2000, n), rng.uniform(0, 1500, n)
    hand = np.clip(rng.gamma(2.0, 4.0, n) - 1.0, 0, None)
    dist = hand * 25 + rng.uniform(0, 200, n)
    truth = rng.random(n) < 1 / (1 + np.exp(-(0.8 - 0.35 * hand)))
    uid = [f"{inv}:{i}" for i in range(n)]
    block = [f"b{int(a // 500)}_{int(b // 500)}" for a, b in zip(x, y, strict=True)]
    core = pd.DataFrame({"building_uid": uid, "x": 330000 + x, "y": 3060000 + y,
                         "block_id": block, "cell_v9": np.arange(n), "hand_v9": hand,
                         "distance": dist, "dist_cell": dist + rng.normal(0, 5, n),
                         "area_m2": rng.uniform(40, 300, n)})
    groups = {"core": core}
    for dem, thr in (("glo30", THR), ("fabdem", THR), ("gedtm30", THR),
                     ("srtmgl1", [0.2, 0.5, 1, 2])):
        cols: dict[str, Any] = {"building_uid": uid}
        for t in thr:
            lab = S.tlabel(t)
            cols[f"hand_any__{dem}__{lab}"] = hand * (0.5 + t / 5) + rng.normal(0, 1, n).clip(0)
            cols[f"dist_stream__{dem}__{lab}"] = dist * rng.uniform(0.2, 1.0, n)
            cols[f"drains_main__{dem}__{lab}"] = (rng.random(n) < min(0.3 + t / 2, 1)).astype(
                float)
        mr = hand + rng.normal(0, 1.0, n).clip(0)
        mr[:3] = np.nan
        cols.update({f"hand_mr_d8__{dem}": mr, f"hand_mr_rem__{dem}": hand * 1.1,
                     f"flowdist_main__{dem}": dist * 1.3, f"elev__{dem}": 1300 + hand})
        groups[dem] = pd.DataFrame(cols)
    groups["features"] = pd.DataFrame({
        "building_uid": uid, "slope_deg": rng.uniform(0, 20, n), "curv_plan": rng.normal(0, 1, n),
        "curv_profile": rng.normal(0, 1, n), "tpi_150": rng.normal(0, 3, n),
        "tpi_500": rng.normal(0, 5, n), "twi": rng.normal(8, 2, n),
        "log10_area_cell_km2": rng.uniform(-3, 0, n),
        "log10_area_first_drop_km2": rng.uniform(-0.7, 1, n),
        "log10_area_main_km2": rng.uniform(0, 2, n), "elev_glo30": 1300 + hand,
        "elev_fabdem": 1297 + hand, "dsm_excess": np.full(n, 3.0),
        "area_m2": core["area_m2"], "density_100m": rng.poisson(5, n).astype(float)})
    team = {"building_uid": uid, "hand_team": np.where(x < 1400, hand * 0.9, np.nan),
            "team_basin": (x < 1400).astype(float)}
    for key, cut in P.MASK_CUTS.items():
        team[f"team_mask_{key}"] = np.where(x < 1400, (hand < cut).astype(float), np.nan)
    for kind in ("acc", "dig"):
        for dem in ("dem12.5", "glo30", "fabdem"):
            for c in ("fill", "breach_burn"):
                team[f"hand_team_{kind}__{dem}__{c}"] = np.where(x < 1400, hand + rng.normal(
                    0, 0.5, n).clip(0), np.nan)
    groups["team"] = pd.DataFrame(team)
    observable = rng.random(n) > 0.1
    cls = np.where(~observable, "not_observable",
                   np.where(truth ^ (rng.random(n) < 0.05), "positive", "negative"))
    change = pd.DataFrame({"building_id": uid, "class": cls, "observable": observable,
                           "class_all_strips": cls, "observable_all_strips": observable,
                           "score": np.where(cls == "positive", 2.0, -1.0) + rng.normal(0, 1, n)})
    return groups, change


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def world(root, monkeypatch):
    from nakkhu.artifacts import promote
    from nakkhu.runs import create_run

    # prereg_v1 frozen + one primary access; the addendum frozen after it
    (root / "configs" / "splits").mkdir(parents=True)
    split = root / "configs" / "splits" / "split_v1.yaml"
    split.write_text("tiles: {}\n", encoding="utf-8")
    sha = hashlib.sha256(split.read_bytes()).hexdigest()
    (root / "configs" / "prereg.yaml").write_text(
        f"version: prereg_v1\nsplit: {{file: configs/splits/split_v1.yaml, sha256: {sha}}}\n",
        encoding="utf-8")
    project = load_project(root)
    base = guard.freeze(project)["prereg_sha256"]
    guard.record_test_access(project, "flood/validate:x", "c", base, sha, note="validate")
    folder = root / addendum_mod.ADDENDA_DIR
    folder.mkdir(parents=True)
    (folder / f"{ADD}.yaml").write_text(_neutralised_addendum(base), encoding="utf-8")
    addendum_mod.freeze_addendum(project, folder / f"{ADD}.yaml")
    hold = root / "runs" / "eval" / "test_hold.yaml"
    hold.write_text(f"hold: false\nby: test\naddendum: {ADD}\n", encoding="utf-8")
    # prep config and a promoted synthetic prep
    (root / "configs" / "followup").mkdir(parents=True)
    pins = {"microsoft": "m" * 64, "detector_final": "d" * 64}
    prep_cfg = {"addendum": ADD, "sweep_config": "configs/hand/sweep.yaml",
                "inventories": ["microsoft", "detector_final"],
                "hand_tables": {k: {"run": "x", "sha256": v} for k, v in pins.items()},
                "dems": {}, "team": {"dems": ["dem12.5", "glo30", "fabdem"],
                                     "conditionings": ["fill", "breach_burn"]},
                "unosat": {"manifest": "configs/data/manifest_followup.yaml",
                           "id": "ref_unosat", "members_dir": "U",
                           "layers": {"flood_extent": "flood", "affected_points": "pts",
                                      "analysis_extent": "ae"}}}
    (root / S.PREP_CONFIG).write_text(yaml.safe_dump(prep_cfg), encoding="utf-8")
    run = create_run(project, S.PREP_STAGE, ResolvedConfig(values={"x": 1}, source="t"))
    files, changes, groups_all = [], {}, {}
    with run.step("prep"):
        for inv, n, seed in (("microsoft", 700, 3), ("detector_final", 600, 4)):
            groups, change = _world(inv, n, seed)
            changes[inv], groups_all[inv] = change, groups
            for g, frame in groups.items():
                name = P.GROUP_FILES.format(group=g, inventory=inv)
                frame.to_parquet(run.path / name, index=False)
                files.append(name)
        pd.DataFrame([{"scope": "covered building cells", "kappa_linear": 0.9}]).to_csv(
            run.path / "prep_team_reproduction.csv", index=False)
        files.append("prep_team_reproduction.csv")
        manifest = {"run": run.ref, "addendum": ADD, "inventories": ["microsoft",
                                                                     "detector_final"],
                    "groups": ["core", "glo30", "fabdem", "gedtm30", "srtmgl1", "features",
                               "team"],
                    "dems": {"glo30": {"drop_threshold_km2": 0.2, "routing_gate_pass": True},
                             "fabdem": {"drop_threshold_km2": 0.2, "routing_gate_pass": True},
                             "gedtm30": {"drop_threshold_km2": 0.15,
                                         "routing_gate_pass": False},
                             "srtmgl1": {"drop_threshold_km2": 0.2,
                                         "routing_gate_pass": False}},
                    "files": {f: P.sha256(run.path / f) for f in files},
                    "inputs": {"hand_tables": pins}}
        _write_json(run.path / "prep_manifest.json", manifest)
    for f in [*files, "prep_manifest.json"]:
        promote(run.path / f, f"{S.ARTIFACT_DIR}/{f}", project.paths)
    # change maps and the sitting-3 audit (Microsoft items)
    flood = root / "artifacts" / "flood"
    flood.mkdir(parents=True)
    for inv, change in changes.items():
        change.to_parquet(flood / f"change_map_buildings_{inv}.parquet", index=False)
    ms, cm = groups_all["microsoft"]["core"], changes["microsoft"]
    obs = np.flatnonzero(cm["observable"].to_numpy())
    strata = np.where(cm["class"].to_numpy()[obs] == "positive", "positive",
                      np.where(ms["distance"].to_numpy()[obs] <= 300, "negative_near",
                               "negative_far"))
    rng = np.random.default_rng(9)
    rows = []
    for s in ("positive", "negative_near", "negative_far"):
        members = obs[strata == s]
        pick = rng.choice(members, min(60, len(members)), replace=False)
        rows += [{"item_id": f"ft{len(rows) + k}", "building_id": ms["building_uid"][i],
                  "stratum": s, "frame_n": len(members), "design_n": len(pick)}
                 for k, i in enumerate(pick)]
    items = pd.DataFrame(rows)
    audit_core.task_dir(project, "flood_trace").mkdir(parents=True)
    items.to_parquet(audit_core.items_path(project, "flood_trace"), index=False)
    pos = dict(zip(ms["building_uid"], cm["class"] == "positive", strict=True))
    events = [{"event": "answer", "slot": i, "item_id": i, "question": "flood_trace",
               "answer": ("yes" if pos[b] ^ (rng.random() < 0.15) else "no"), "ms": 900}
              for i, b in zip(items["item_id"], items["building_id"], strict=True)]
    audit_core.responses_path(project, "flood_trace").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    _write_json(root / "artifacts" / V.ESTIMATES, {"estimates": {
        "ppv": {"estimate": 0.9, "wilson_lo": 0.8, "wilson_hi": 0.95}, "npv": {"estimate": 0.9},
        "drop_change_map": False}})
    monkeypatch.setattr(V, "prerequisites", lambda *a, **k: [])
    monkeypatch.setattr(P, "team_zip", lambda *a, **k: Path("team.zip"))
    monkeypatch.setattr(RM, "code_state", lambda project: {"commit": COMMIT, "dirty": False})
    return {"project": project, "groups": groups_all, "changes": changes}


def _journal(project) -> list[dict]:
    return guard.read_test_journal(project)


COMMIT = "c0ffee" * 6 + "c0ff"


def _remote_clone(project, dest: Path, commit: str = COMMIT):
    """What the bundle ships to a remote host: the tree without the live journal, plus the
    journal copy and a bundle manifest of the code paths."""
    import shutil

    src = project.paths.root
    jp = guard.journal_path(project)
    shutil.copytree(src, dest, ignore=lambda d, names: [
        n for n in names if Path(d, n) in (jp, jp.with_name(jp.name + ".lock"))])
    copy = dest / RM.JOURNAL_COPY
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(jp.read_bytes())
    files = {}
    for top in RM.CODE_PATHS:
        base = dest / top
        for f in ([base] if base.is_file() else base.rglob("*") if base.is_dir() else []):
            if f.is_file():
                files[f.relative_to(dest).as_posix()] = P.sha256(f)
    (dest / RM.BUNDLE_MANIFEST).write_text(json.dumps(
        {"git_head": commit, "dirty": False, "files": files}), encoding="utf-8")
    return load_project(dest)


def _pull(remote_project, project, run_ref: str) -> None:
    """Copy the remote run dir back (new files only; reserved files must be identical)."""
    stage, run_id = run_ref.split("/", 1)
    a = remote_project.paths.runs / stage / run_id
    b = project.paths.runs / stage / run_id
    for f in a.rglob("*"):
        if f.is_file():
            t = b / f.relative_to(a)
            if t.exists():
                assert t.read_bytes() == f.read_bytes(), t
            else:
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(f.read_bytes())


_CLONES = [0]


def _run(project, test, *, retry=False, dry_run=False, shards=("0/1",)):
    """reserve (laptop) -> execute on a remote clone with the receipt -> pull -> promote /
    merge (laptop): the only way a declared test runs."""
    res = RM.reserve(project, test, retry=retry, dry_run=dry_run, n_boot=25)
    if not res["ready"] or dry_run:
        return res
    receipt = project.paths.root / res["receipt"]
    for spec_ in (shards if test in RM.SHARDED else (None,)):
        _CLONES[0] += 1
        remote = _remote_clone(project, project.paths.root.parent / f"remote{_CLONES[0]}")
        RM.execute(remote, remote.paths.root / res["receipt"], test, shard=spec_)
        _pull(remote, project, res["run"])
    out = (RM.merge_p1_6(project) if test in RM.SHARDED else RM.promote_test(project, test))
    return {**out, "receipt": receipt, "journal_note": res["journal_note"]}


@pytest.mark.parametrize("test", ["P1-1", "P1-2", "P1-3", "P1-4", "P1-5", "P1-7", "P1-9",
                                  "P1-10"])
def test_every_phase1_test_runs_once_and_journals_first(world, test, monkeypatch):
    project = world["project"]
    seen: dict[str, int] = {}
    real = R.load_labels

    def spy(*a, **k):
        seen["journal_at_label_read"] = len(_journal(project))
        return real(*a, **k)

    monkeypatch.setattr(R, "load_labels", spy)
    dry = _run(project, test, dry_run=True)
    assert dry["ready"] and dry["dry_run"] and len(_journal(project)) == 1
    out = _run(project, test)
    assert out["ready"], out
    journal = _journal(project)
    assert len(journal) == 2 and journal[1]["note"] == f"{ADD}:{test}"
    assert journal[1]["addendum_sha256"] and seen["journal_at_label_read"] == 2
    for name in R.outputs_of(test):
        frame = pd.read_csv(project.paths.artifacts / S.ARTIFACT_DIR / name)
        assert len(frame) and (frame["reporting"] == f"post hoc (prereg_v9_a3 {test})").all()
    again = _run(project, test)
    assert not again["ready"] and any("already ran" in p for p in again["problems"])
    assert len(_journal(project)) == 2


def test_p1_outputs_are_sensible(world):
    project = world["project"]
    assert _run(project, "P1-1")["ready"]
    main = pd.read_csv(project.paths.artifacts / "followup" / "p1_1_main_river.csv")
    assert set(main["reference"]) == {"R1", "R1-aud", "R1-all-strips"}
    ms = main[(main["inventory"] == "microsoft") & (main["reference"] == "R1")]
    assert (ms["n_nan_candidate"] <= 3).all() and ms["scored"].all()
    aud = main[main["reference"] == "R1-aud"]
    assert aud["weighted"].all() and aud["delong_p"].isna().all()
    tiers = pd.read_csv(project.paths.artifacts / "followup" / "p1_1_tiers.csv")
    assert {"hit_rate", "odds_ratio"} <= set(tiers["kind"]) and (tiers["nan_path_count"] <= 3).all()
    assert _run(project, "P1-4")["ready"]
    p4 = pd.read_csv(project.paths.artifacts / "followup" / "p1_4_dems.csv")
    assert set(p4.loc[p4["status"].str.startswith("not prepared"), "dem"]) == {
        "nasadem", "aw3d30", "hma8m"}
    assert (p4.loc[p4["dem"] == "gedtm30", "routing_gate_pass"] == False).all()  # noqa: E712
    assert "vs glo30 same rule" in set(p4["comparison"].dropna())


def test_p1_6_models_with_a_small_budget(world, monkeypatch):
    project = world["project"]
    monkeypatch.setattr(ml, "hgb_configs", lambda spec: [
        {"learning_rate": 0.1, "max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 20,
         "l2_regularization": 0.0, "max_features": 1.0},
        {"learning_rate": 0.05, "max_iter": 30, "max_leaf_nodes": 7, "min_samples_leaf": 50,
         "l2_regularization": 1.0, "max_features": 0.5}])
    monkeypatch.setattr(ml, "gam_lambdas", lambda spec: [1.0, 100.0])
    monkeypatch.setattr(ml, "REPEATS", 2)
    out = _run(project, "P1-6", shards=("0/4", "1-2/4", "3/4"))
    assert out["ready"], out
    assert out["merge"]["duplicates"] == 0 and out["merge"]["n_units"] == 3 * 12
    models = pd.read_csv(project.paths.artifacts / "followup" / "p1_6_models.csv")
    oof = models[models["row"] == "oof"]
    assert set(oof["family"]) == {"M-C", "M-D", "M-E"} and (oof["auc"] > 0.55).all()
    assert (models["row"] == "oof_detector_final").sum() == 3
    imp = pd.read_csv(project.paths.artifacts / "followup" / "p1_6_importance.csv")
    assert set(imp["group"]) == {"hand", "distance", "elevation", "terrain", "flow",
                                 "building"}


def _unosat_zip(tmp: Path) -> Path:
    src = tmp / "u"
    src.mkdir()
    crs = "EPSG:32645"
    gpd.GeoDataFrame(geometry=[box(330000, 3060000, 330600, 3061500)], crs=crs).to_file(
        src / "flood.shp")
    gpd.GeoDataFrame(geometry=[Point(330100 + 50 * i, 3060100 + 50 * i) for i in range(20)],
                     crs=crs).to_file(src / "pts.shp")
    gpd.GeoDataFrame(geometry=[box(329000, 3059000, 332500, 3062000)], crs=crs).to_file(
        src / "ae.shp")
    path = tmp / "unosat.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f"U/{f.name}")
    return path


def test_p1_8_unosat_reference(world, monkeypatch, tmp_path):
    project = world["project"]
    made = _unosat_zip(tmp_path)
    inside = project.paths.root / "data" / "unosat.zip"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(made.read_bytes())
    monkeypatch.setattr(R, "unosat_zip", lambda p, c: p.paths.root / "data" / "unosat.zip")
    monkeypatch.setattr(RM, "extra_inputs", lambda p, c, t: [R.unosat_zip(p, c)]
                        if t == "P1-8" else [])
    core = world["groups"]["microsoft"]["core"]
    fp = gpd.GeoDataFrame({"building_uid": core["building_uid"]},
                          geometry=gpd.points_from_xy(core["x"], core["y"]).buffer(5),
                          crs="EPSG:32645")
    monkeypatch.setattr(R, "_footprints", lambda *a: fp)
    out = _run(project, "P1-8")
    assert out["ready"], out
    t = pd.read_csv(project.paths.artifacts / "followup" / "p1_8_unosat.csv")
    assert {"R-UN", "R-UN-pts"} <= set(t["label"].dropna().str.split(" ").str[0])
    agree = t[t["part"] == "agreement"]
    assert len(agree) == 2 and agree["kappa"].notna().all()
    assert t["caveat"].str.contains("not field validated").all()


def test_runner_refusals_happen_before_any_journal_entry(world):
    project, root = world["project"], world["project"].paths.root
    hold = root / "runs" / "eval" / "test_hold.yaml"
    hold.write_text("hold: true\n", encoding="utf-8")
    assert any(p.startswith("hold:") for p in _run(project, "P1-1")["problems"])
    hold.write_text("hold: false\nby: user\naddendum: prereg_v9_a2\n", encoding="utf-8")
    assert any("not prereg_v1_a1" in p for p in _run(project, "P1-1")["problems"])
    hold.write_text(f"hold: false\nby: user\naddendum: {ADD}\n", encoding="utf-8")
    for test, why in (("P2", "p2_inputs.yaml missing"), ("P1-explore", "access: none"),
                      ("P1-4-nasadem", "not prepared"), ("P1-5-ext", "p1_5_ext.yaml"),
                      ("P9", "declares no test")):
        problems = " | ".join(_run(project, test)["problems"])
        assert why in problems, (test, problems)
    entry = _journal(project)[0]
    guard.record_test_access(project, "x", "c", entry["prereg_sha256"], entry["split_sha256"],
                             note=f"{ADD}:P1-99")
    assert any("undeclared" in p for p in _run(project, "P1-1")["problems"])
    assert len(_journal(project)) == 2


def test_tampered_prep_or_prereg_is_refused(world):
    project = world["project"]
    art = project.paths.artifacts / S.ARTIFACT_DIR / "prep_core_microsoft.parquet"
    keep = art.read_bytes()
    art.write_bytes(keep + b"x")
    assert any("altered" in p for p in _run(project, "P1-2")["problems"])
    art.write_bytes(keep)
    live = project.paths.resolve(f"{addendum_mod.ADDENDA_DIR}/{ADD}.yaml")
    text = live.read_text(encoding="utf-8")
    live.write_text(text + "\nextra: 1\n", encoding="utf-8")
    assert any("prereg:" in p for p in _run(project, "P1-2")["problems"])
    live.write_text(text, encoding="utf-8")
    assert len(_journal(project)) == 1


def test_retry_is_allowed_once_after_a_failed_run(world, monkeypatch):
    project = world["project"]
    monkeypatch.setattr(R, "score_test", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        _run(project, "P1-7")
    assert _journal(project)[-1]["note"] == f"{ADD}:P1-7"
    monkeypatch.undo()
    monkeypatch.setattr(V, "prerequisites", lambda *a, **k: [])
    monkeypatch.setattr(RM, "code_state", lambda project: {"commit": COMMIT, "dirty": False})
    out = _run(project, "P1-7", retry=True)
    assert out["ready"] and _journal(project)[-1]["note"] == f"{ADD}:P1-7 retry"
    assert not _run(project, "P1-7", retry=True)["ready"]


def test_external_reference_path(world, monkeypatch, tmp_path):
    project = world["project"]
    src = tmp_path / "ext"
    src.mkdir()
    gpd.GeoDataFrame(geometry=[box(330000, 3060000, 330500, 3061500)],
                     crs="EPSG:32645").to_file(src / "poly.shp")
    gpd.GeoDataFrame(geometry=[box(329000, 3059000, 332500, 3062000)],
                     crs="EPSG:32645").to_file(src / "cover.shp")
    path = tmp_path / "ext.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f"E/{f.name}")
    data = project.paths.root / "data" / "ext.zip"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(path.read_bytes())
    (project.paths.root / R.EXTERNAL_CONFIG).write_text(yaml.safe_dump({
        "name": "TEST-EXT", "file": "data/ext.zip", "sha256": P.sha256(data),
        "licence": "CC0", "date": "2024-10-01", "polygons": "E/poly.shp",
        "coverage": "E/cover.shp"}), encoding="utf-8")

    def fp(project, cfg, inv, table):
        return gpd.GeoDataFrame({"building_uid": table["building_uid"]},
                                geometry=gpd.points_from_xy(table["x"], table["y"]).buffer(5),
                                crs="EPSG:32645")

    monkeypatch.setattr(R, "_footprints", fp)
    out = _run(project, "P1-5-ext")
    assert out["ready"], out
    t = pd.read_csv(project.paths.artifacts / "followup" / "p1_5_ext.csv")
    assert (t["reference"] == "TEST-EXT").all() and (t["part"] == "agreement").sum() == 2


# --- reserve locally, execute remotely ----------------------------------------------------------


def _reserved(world, test="P1-7"):
    project = world["project"]
    res = RM.reserve(project, test, n_boot=25)
    assert res["ready"], res
    return project, res


def test_local_execution_is_refused(world):
    project, res = _reserved(world)
    assert not R.run_test(project, "P1-3")["ready"]
    assert "local execution is refused" in R.run_test(project, "P1-3")["problems"][0]
    # the laptop holds the live journal: executing there is refused even with a receipt
    with pytest.raises(RM.ReceiptError, match="live test journal"):
        RM.execute(project, project.paths.root / res["receipt"], "P1-7")
    assert not (project.paths.runs / res["run"] / RM.EXECUTION_FILE).exists()


def test_cli_run_without_receipt_refuses(world, capsys):
    from nakkhu.cli import main

    project = world["project"]
    rc = main(["--root", str(project.paths.root), "followup", "run", "--test", "P1-3"])
    assert rc == 1 and "refused without --receipt" in capsys.readouterr().err
    assert len(_journal(project)) == 1


def test_reserve_writes_journal_then_receipt(world):
    project, res = _reserved(world)
    receipt = json.loads((project.paths.root / res["receipt"]).read_text(encoding="utf-8"))
    lines = guard.journal_path(project).read_text(encoding="utf-8").splitlines()
    line = lines[receipt["journal"]["index"] - 1]
    assert hashlib.sha256(line.encode()).hexdigest() == receipt["journal"]["line_sha256"]
    assert json.loads(line)["note"] == f"{ADD}:P1-7" and receipt["code_commit"] == COMMIT
    assert {"artifacts/flood/change_map_buildings_microsoft.parquet",
            "audit/flood_trace/responses.jsonl"} <= set(receipt["inputs"])
    assert not RM.reserve(project, "P1-7")["ready"]          # run-once holds at reserve time


def test_reserve_refuses_dirty_code(world, monkeypatch):
    project = world["project"]
    monkeypatch.setattr(RM, "code_state", lambda p: {"commit": COMMIT, "dirty": True})
    out = RM.reserve(project, "P1-7")
    assert not out["ready"] and any("differ from HEAD" in p for p in out["problems"])
    assert len(_journal(project)) == 1


@pytest.mark.parametrize("tamper", ["journal_copy", "commit", "input", "prep", "code",
                                    "no_bundle", "test"])
def test_receipt_verification_refusals(world, tamper, tmp_path):
    project, res = _reserved(world)
    remote = _remote_clone(project, tmp_path / "r",
                           commit="f" * 40 if tamper == "commit" else COMMIT)
    root = remote.paths.root
    test = "P1-7"
    if tamper == "journal_copy":
        p = root / RM.JOURNAL_COPY
        lines = p.read_text(encoding="utf-8").splitlines()
        lines[-1] = lines[-1].replace("P1-7", "P1-3")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif tamper == "input":
        (root / "audit" / "flood_trace" / "responses.jsonl").write_text("{}\n")
    elif tamper == "prep":
        f = root / "artifacts" / S.ARTIFACT_DIR / "prep_core_microsoft.parquet"
        f.write_bytes(f.read_bytes() + b"x")
    elif tamper == "code":
        (root / "configs" / "followup" / "extra.yaml").write_text("x: 1\n")
    elif tamper == "no_bundle":
        (root / RM.BUNDLE_MANIFEST).unlink()
    elif tamper == "test":
        test = "P1-3"
    with pytest.raises(RM.ReceiptError):
        RM.execute(remote, root / res["receipt"], test)
    assert not (remote.paths.runs / res["run"] / RM.EXECUTION_FILE).exists()
    assert not guard.journal_path(remote).exists()           # the remote never writes one


def test_remote_executes_once_and_promotion_verifies(world, tmp_path):
    project, res = _reserved(world)
    remote = _remote_clone(project, tmp_path / "r")
    rec = remote.paths.root / res["receipt"]
    RM.verify_receipt(remote, rec, "P1-7")                   # dry-run verification only
    out = RM.execute(remote, rec, "P1-7")
    assert out["ready"] and not guard.journal_path(remote).exists()
    with pytest.raises(RM.ReceiptError, match="already executed"):
        RM.execute(remote, rec, "P1-7")
    with pytest.raises(RM.ReceiptError, match="not pulled"):
        RM.promote_test(project, "P1-7")
    _pull(remote, project, res["run"])
    run_dir = project.paths.runs / res["run"]
    csv = run_dir / "p1_7_metrics.csv"
    keep = csv.read_bytes()
    csv.write_bytes(keep + b"\n")
    with pytest.raises(RM.ReceiptError, match="altered since execution"):
        RM.promote_test(project, "P1-7")
    csv.write_bytes(keep)
    ex = json.loads((run_dir / RM.EXECUTION_FILE).read_text(encoding="utf-8"))
    (run_dir / RM.EXECUTION_FILE).write_text(json.dumps({**ex, "receipt_sha256": "0" * 64}))
    with pytest.raises(RM.ReceiptError, match="not made from this receipt"):
        RM.promote_test(project, "P1-7")
    (run_dir / RM.EXECUTION_FILE).write_text(json.dumps(ex))
    done = RM.promote_test(project, "P1-7")
    assert done["promoted"] == ["followup/p1_7_metrics.csv"]
    assert len(_journal(project)) == 2


def test_parse_shard():
    assert RM.parse_shard("3/8") == ([3], 8)
    assert RM.parse_shard("0-2/8") == ([0, 1, 2], 8)
    assert RM.parse_shard("1,5/8") == ([1, 5], 8)
    for bad in ("8/8", "1", "2-1/4", "1,1/4", "a/4"):
        with pytest.raises(RM.ReceiptError):
            RM.parse_shard(bad)


def _small_p1_6(monkeypatch):
    monkeypatch.setattr(ml, "hgb_configs", lambda spec: [
        {"learning_rate": 0.1, "max_iter": 10, "max_leaf_nodes": 5, "min_samples_leaf": 20,
         "l2_regularization": 0.0, "max_features": 1.0},
        {"learning_rate": 0.05, "max_iter": 15, "max_leaf_nodes": 5, "min_samples_leaf": 40,
         "l2_regularization": 1.0, "max_features": 0.5}])
    monkeypatch.setattr(ml, "gam_lambdas", lambda spec: [1.0, 100.0])
    monkeypatch.setattr(ml, "REPEATS", 2)


def test_p1_6_shards_resume_merge_and_equal_the_in_process_run(world, monkeypatch, tmp_path):
    project = world["project"]
    _small_p1_6(monkeypatch)
    res = RM.reserve(project, "P1-6", n_boot=25)
    assert res["ready"], res
    remote = _remote_clone(project, tmp_path / "r")
    rec = remote.paths.root / res["receipt"]
    with pytest.raises(RM.ReceiptError, match="--shard"):
        RM.execute(remote, rec, "P1-6")
    RM.execute(remote, rec, "P1-6", shard="0/3")
    path = RM.shard_path(remote.paths.runs / res["run"], 0, 3)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:50], encoding="utf-8")  # killed
    again = RM.execute(remote, rec, "P1-6", shard="0/3")
    assert again["units_run"] == 1                         # only the truncated unit is redone
    _pull(remote, project, res["run"])
    with pytest.raises(RM.ReceiptError, match="missing"):
        RM.merge_p1_6(project)
    remote2 = _remote_clone(project, tmp_path / "r2")
    RM.execute(remote2, remote2.paths.root / res["receipt"], "P1-6", shard="1-2/3", jobs=2)
    _pull(remote2, project, res["run"])
    out = RM.merge_p1_6(project)
    assert out["ready"] and out["merge"]["duplicates"] == 0
    with pytest.raises(RM.ReceiptError, match="already merged"):
        RM.merge_p1_6(project)
    merged = pd.read_csv(project.paths.artifacts / "followup" / "p1_6_models.csv")
    # the same units in one process give the same table
    from nakkhu.followup import analyses as A

    prep, _ = P.load_prep(project)
    refs = R.references(prep, R.load_labels(project, list(prep)))
    spec = S.load_spec(project, ADD)
    ctx = A.Ctx(n_boot=25, seed=0, min_cell=S.min_cell(spec), edges=S.tier_edges(spec))
    models, imp, _ = ml.p1_6(R.pick(refs, ("microsoft", "R1"))[0],
                             R.pick(refs, ("detector_final", "R1"))[0], spec, ctx, repeats=2)
    assert len(merged) == len(models)
    for col in ("auc", "pr_auc", "log_loss", "brier", "delta_auc"):
        if col in models:
            np.testing.assert_allclose(merged[col].to_numpy(float), models[col].to_numpy(float),
                                       rtol=0, atol=1e-12, equal_nan=True)
    mimp = pd.read_csv(project.paths.artifacts / "followup" / "p1_6_importance.csv")
    np.testing.assert_allclose(mimp["auc_drop_mean"], imp["auc_drop_mean"], atol=1e-12)


def test_p1_6_merge_refuses_duplicates(world, monkeypatch, tmp_path):
    project = world["project"]
    _small_p1_6(monkeypatch)
    res = RM.reserve(project, "P1-6", n_boot=25)
    remote = _remote_clone(project, tmp_path / "r")
    RM.execute(remote, remote.paths.root / res["receipt"], "P1-6", shard="0/1")
    _pull(remote, project, res["run"])
    run_dir = project.paths.runs / res["run"]
    first = RM.shard_path(run_dir, 0, 1).read_text(encoding="utf-8").splitlines()[0]
    (run_dir / RM.SHARD_DIR / "p1_6_000of002.jsonl").write_text(first + "\n", encoding="utf-8")
    with pytest.raises(RM.ReceiptError, match="duplicate"):
        RM.merge_p1_6(project)


def test_unit_permutations_reproduce_permutation_importance(world, monkeypatch):
    project = world["project"]
    _small_p1_6(monkeypatch)
    prep, _ = P.load_prep(project)
    refs = R.references(prep, R.load_labels(project, list(prep)))
    data = ml.prepare(R.pick(refs, ("microsoft", "R1"))[0], None)
    spec = S.load_spec(project, ADD)
    cv = ml.cross_validate("M-D", spec, data["X"], data["y"], data["blocks"], data["Xd"],
                           data["blocks_d"])
    old = ml.permutation_importance(cv, data["X"], data["y"], data["blocks"], 2, 0)
    units = {u["key"]: u for u in ml.run_units(ml.unit_keys(data, ("M-D",)), spec, data,
                                               repeats=2, seed=0)}
    for b in data["ids"]:
        np.testing.assert_array_equal(units[ml.unit_key("M-D", b)]["p"],
                                      cv["p"][data["blocks"] == b])
    new = ml.importance_from_units("M-D", units, data, cv["p"], 2)
    assert [r["auc_drop_mean"] for r in new] == [r["auc_drop_mean"] for r in old]


# --- Phase-2 preparation explore run (selection + M-C refit) ------------------------------------


def test_explore_p2_needs_phase1_then_runs_once(world):
    from nakkhu.followup import explore as E

    project = world["project"]
    early = _run(project, E.EXPLORE_P2)
    assert not early["ready"] and any("only after P1-1 to P1-10" in p
                                      for p in early["problems"])
    entry = _journal(project)[0]
    for t in R.PHASE1:
        guard.record_test_access(project, f"followup/{t}", "c", entry["prereg_sha256"],
                                 entry["split_sha256"], note=f"{ADD}:{t}")
    n0 = len(_journal(project))
    out = _run(project, E.EXPLORE_P2)
    assert out["ready"], out
    journal = _journal(project)
    assert len(journal) == n0 + 1 and journal[-1]["note"] == f"{ADD}:{E.EXPLORE_P2}"
    art = project.paths.artifacts / S.ARTIFACT_DIR
    members = pd.read_csv(art / "p2_select_hand_members.csv")
    assert len(members) == 6 and members["n"].nunique() == 1
    team = world["groups"]["microsoft"]["team"]
    assert members["n_covered"].iloc[0] < len(team)
    rule = json.loads((art / "p2_select_rule.json").read_text(encoding="utf-8"))
    assert rule["pick"]["dem"] in E.DEM_ORDER and rule["tie_margin"] == 0.005
    model = json.loads((art / "p2_mc_model.json").read_text(encoding="utf-8"))
    assert model["family"] == "M-C" and model["lambda"] in model["lambda_grid"]
    assert len(model["knots"]) == 28 and len(model["coefficients"]) == 1 + 28 * 4
    prep, _ = P.load_prep(project)
    p = E.predict(model, prep["microsoft"])
    assert np.isfinite(p).sum() > 0.9 * len(p) and ((p > 0) & (p < 1))[np.isfinite(p)].all()
    again = _run(project, E.EXPLORE_P2)
    assert not again["ready"] and any("already ran" in p for p in again["problems"])
    assert len(_journal(project)) == n0 + 1


def test_explore_rule_ties_and_order():
    from nakkhu.followup import explore as E

    t = pd.DataFrame([{"predictor": p, "dem": d, "column": c, "auc": a}
                      for (p, d, c), a in zip(E.MEMBERS, (0.906, 0.909, 0.910, 0.902, 0.899,
                                                          0.880), strict=True)])
    rule = E.apply_rule(t)
    assert rule["best_member"] == "hand_mr_rem x glo30"
    assert rule["pick"]["predictor"] == "hand_mr_d8" and rule["pick"]["dem"] == "glo30"
    t.loc[2, "auc"] = 0.920          # hand_mr_rem x glo30 now beats every other by > 0.005
    assert E.apply_rule(t)["pick"]["predictor"] == "hand_mr_rem"


# --- P2: the one-shot Phase-2 confirmation ----------------------------------------------------


def _p2_site_frames(site: str, n: int, seed: int, *, signal: float = 1.0, nan_share: float = 0.1,
                    n_blocks: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic site: predictor table (all M-C features) and change map."""
    rng = np.random.default_rng(seed)
    groups, _ = _world("microsoft", n, seed)
    feats = groups["features"].drop(columns=["building_uid", "area_m2"])
    glo = groups["glo30"].drop(columns=["building_uid"])
    fab = groups["fabdem"][["hand_mr_d8__fabdem"]]
    core = groups["core"]
    hand = core["hand_v9"].to_numpy()
    mr = hand + rng.normal(0, 0.5, n).clip(0)
    mr[rng.random(n) < nan_share] = np.nan
    table = pd.concat([core.drop(columns=["block_id"]), feats, glo, fab], axis=1)
    table["hand_mr_d8__glo30"] = mr
    table["building_uid"] = [f"{site}/microsoft:{i}" for i in range(n)]
    table["block_id"] = [f"b_{k % n_blocks:02d}_00" for k in range(n)]
    truth = rng.random(n) < 1 / (1 + np.exp(-(1.0 - signal * 0.4 * hand)))
    observable = rng.random(n) > 0.15
    cls = np.where(~observable, "not_observable", np.where(truth, "positive", "negative"))
    change = pd.DataFrame({"building_id": table["building_uid"], "class": cls,
                           "observable": observable})
    return table, change


def _p2_model(world):
    from nakkhu.followup import explore as E

    project = world["project"]
    spec = S.load_spec(project, ADD)
    prep, _ = P.load_prep(project)
    labels = R.load_labels(project, list(prep))
    ms = R.pick(R.references(prep, labels), ("microsoft", "R1"))[0]
    return E.refit_mc(ms, spec)[0]


def test_p2_statistics_units():
    from nakkhu.followup import confirm as C

    assert C.holm({"H2a": 0.02, "H2b": 0.04}) == {"H2a": True, "H2b": True}
    assert C.holm({"H2a": 0.03, "H2b": 0.04}) == {"H2a": False, "H2b": False}
    assert C.holm({"H2a": 0.01, "H2b": 0.06}) == {"H2a": True, "H2b": False}
    assert C.one_sided_p(np.array([0.1, -0.1, np.nan, 0.2])) == (1 + 2) / 5
    assert abs(C.mde(26) - 0.048) < 1e-12 and C.mde(104) < C.mde(26)
    assert C.size_check(np.array(["a"] * 30), np.ones(30), True)["mode"] == "descriptive"
    blocks = np.array([f"A:{k}" for k in range(30)] * 6)
    assert C.size_check(blocks, np.ones(180), True)["mode"] == "confirmatory"
    assert C.size_check(blocks, np.ones(180), False)["mode"] == "descriptive"
    # within-site resampling: a score that is perfect in every block of site A and inverted in
    # every block of site B gives the same delta in every resample (site mixes are fixed)
    y = np.tile([0, 1], 40)
    strata = np.array(["A"] * 40 + ["B"] * 40)
    blk = np.array([f"{s}:{(i // 2) % 4}" for i, s in enumerate(strata)])
    s1 = np.where(strata == "A", y + 0.1, -y).astype(float)
    d = C.stratified_deltas(y, s1, np.zeros(80), blk, strata, 50, 0)
    assert np.isfinite(d).all() and np.allclose(d, d[0])


def test_p2_score_sites_pools_gates_and_restricts_distance(world):
    from nakkhu.followup import confirm as C

    model = _p2_model(world)
    ta, ca = _p2_site_frames("A", 900, 11, n_blocks=20)
    tb, cb = _p2_site_frames("B", 600, 12, nan_share=0.3, n_blocks=12)
    tn, cn = _p2_site_frames("N", 300, 13, n_blocks=8)
    good = {"ppv": {"estimate": 0.9, "wilson_lo": 0.8, "wilson_hi": 0.95}, "npv": {},
            "drop_change_map": False}
    bad = {"ppv": {"estimate": 0.5, "wilson_lo": 0.4, "wilson_hi": 0.6}, "npv": {},
           "drop_change_map": True}
    sites = [C.Site("A", "test", ta, ca, good), C.Site("B", "test", tb, cb, good),
             C.Site("N", "negative_control", tn, cn, None)]
    summary, tables = C.score_sites(sites, model, n_boot=40, seed=0)
    rows = tables["p2_rows.csv"]
    h2a = rows[(rows["part"] == "hypothesis") & (rows["hypothesis"] == "H2a")].iloc[0]
    assert h2a["size_n"] < summary["size_check"]["pooled_all_rows"]["n"]
    assert summary["nan"]["H2a"]["n_nan"] > 0 and summary["sites"]["B"]["share_nan_H2a"] > 0.2
    assert summary["size_check"]["H2a"]["n_blocks"] == 32
    assert set(summary["hypotheses"]) == {"H2a", "H2b"} and summary["pool"] == ["A", "B"]
    for d in summary["hypotheses"].values():
        assert d["mode"] in ("confirmatory", "descriptive")
        if d["mode"] == "descriptive":
            assert d["decision"] is None
    assert len(tables["p2_negative_control.csv"]) == 1
    assert {"hit_rate", "odds_ratio"} <= set(tables["p2_tiers.csv"]["kind"])
    # distance is scored on exactly the hypothesis rows: same n for candidate and comparator
    ra = C.site_rows(C.Site("A", "test", ta, ca, good))
    rb = C.site_rows(C.Site("B", "test", tb, cb, good))
    pooled = pd.concat([ra, rb], ignore_index=True)
    ok = np.isfinite(pooled["hand_mr_d8__glo30"].to_numpy(dtype=float))
    y = pooled[C.R1_COL].to_numpy()
    assert abs(h2a["auc_distance_or_comparator"]
               - V.roc_auc(-pooled["distance"].to_numpy()[ok], y[ok])) < 1e-12
    # an audit drop removes the site from the pool; a pool below 26 blocks is descriptive
    sites[1] = C.Site("B", "test", tb, cb, bad)
    summary2, _ = C.score_sites(sites, model, n_boot=20, seed=0)
    assert summary2["pool"] == ["A"] and summary2["sites"]["B"]["kept"] is False
    assert summary2["size_check"]["H2a"]["n_blocks"] == 20
    assert all(d["mode"] == "descriptive" and d["decision"] is None
               for d in summary2["hypotheses"].values())


def test_p2_refuses_without_a4_and_audit_then_runs_once(world):
    from nakkhu.followup import confirm as C

    project, root = world["project"], world["project"].paths.root
    assert any("p2_inputs.yaml missing" in p for p in _run(project, "P2")["problems"])
    model = _p2_model(world)
    base = root / "artifacts" / "p2test"
    base.mkdir(parents=True)
    (base / "model.json").write_text(json.dumps(model), encoding="utf-8")
    est: dict[str, Any] = {"primary_rater": "author", "sites": {}}
    sites_doc = {}
    for s, n, seed, nb in (("A", 900, 21, 20), ("B", 700, 22, 12)):
        t, c = _p2_site_frames(s, n, seed, n_blocks=nb)
        t.to_parquet(base / f"{s}_pred.parquet", index=False)
        c.to_parquet(base / f"{s}_change.parquet", index=False)
        est["sites"][s] = {"ppv": {"estimate": 0.9, "wilson_lo": 0.8, "wilson_hi": 0.95},
                           "npv": {}, "drop_change_map": False}
        sites_doc[s] = {"role": "test",
                        "predictors": {"file": f"artifacts/p2test/{s}_pred.parquet",
                                       "sha256": P.sha256(base / f"{s}_pred.parquet")},
                        "change_map": {"file": f"artifacts/p2test/{s}_change.parquet",
                                       "sha256": P.sha256(base / f"{s}_change.parquet")}}
    (base / "est.json").write_text(json.dumps(est), encoding="utf-8")
    inputs = {"addendum_a4": "prereg_v1_a2",
              "model": {"file": "artifacts/p2test/model.json",
                        "sha256": P.sha256(base / "model.json")},
              "audit_estimates": {"file": "artifacts/p2test/est.json",
                                  "sha256": P.sha256(base / "est.json")},
              "sites": sites_doc}
    (root / "configs" / "followup" / "phase2").mkdir(parents=True, exist_ok=True)
    (root / C.INPUTS).write_text(yaml.safe_dump(inputs), encoding="utf-8")
    problems = " | ".join(_run(project, "P2")["problems"])
    assert "prereg_v1_a2 is not frozen" in problems and "P2-audit has 0" in problems
    a4 = root / addendum_mod.ADDENDA_DIR / "prereg_v1_a2.yaml"
    a4.write_text(yaml.safe_dump({"addendum": "prereg_v1_a2", "base_version": "prereg_v1",
                                  "base_sha256": _journal(project)[0]["prereg_sha256"],
                                  "requires": [ADD], "tests": {"x": {"access": "none"}}}),
                  encoding="utf-8")
    addendum_mod.freeze_addendum(project, a4)
    addendum_mod.record_addendum_access(project, ADD, "P2-audit", "audit/x", "c")
    n0 = len(_journal(project))
    out = _run(project, "P2")
    assert out["ready"], out
    assert _journal(project)[-1]["note"] == f"{ADD}:P2" and len(_journal(project)) == n0 + 1
    doc = json.loads((project.paths.artifacts / "followup" / "p2_confirm.json").read_text(
        encoding="utf-8"))
    assert doc["pool"] == ["A", "B"] and set(doc["hypotheses"]) == {"H2a", "H2b"}
    assert doc["size_check"]["H2a"]["n_blocks"] == 32
    again = _run(project, "P2")
    assert not again["ready"] and any("already ran" in p for p in again["problems"])
    # an altered input is refused before any journal entry
    (base / "est.json").write_text("{}", encoding="utf-8")
    assert "sha256 differs" in " | ".join(C.prerequisites(project, ADD))


def test_p2_audit_reservation_refuses_without_its_task(world):
    out = _run(world["project"], "P2-audit")
    assert not out["ready"] and any(p.startswith("P2-audit:") for p in out["problems"])
    assert len(_journal(world["project"])) == 1
