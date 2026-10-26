"""``nakkhu uncertainty run`` end to end on synthetic fixtures only (no real audit, detection,
HAND or test tile is read)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
import yaml

from nakkhu.config import load_project
from nakkhu.evaluate import guard
from nakkhu.uncertainty import run as R
from nakkhu.uncertainty import spec as S

REPO = Path(__file__).resolve().parents[1]
CFGS = ["glo30__breach_burn__tdrop", "glo30__breach_burn__t1", "fabdem__breach_burn__tdrop",
        "fabdem__breach_burn__t1"]


class FakeHand:
    """HAND rises west to east (0-15 m) with a per-surface and per-statistic offset."""

    edges = R.EDGES

    def surfaces(self) -> pd.DataFrame:
        return pd.DataFrame({"cfg_id": CFGS, "dem": [c.split("__")[0] for c in CFGS],
                             "in_sensitivity_range": True,
                             "is_primary": [c == CFGS[0] for c in CFGS]})

    def values(self, geoms, stats, cfg_ids):
        x = shapely.get_x(shapely.centroid(np.asarray(geoms, dtype=object)))
        out = {}
        for cfg in cfg_ids:
            off = 0.4 * CFGS.index(cfg) - 0.5
            for stat in stats:
                s_off = {"centroid": 0.0, "footprint_min": -0.6, "buffer_min": -1.2}[stat]
                out[f"{cfg}__{stat}"] = x / 2000.0 * 15.0 + off + s_off
        return out


def _box(x: float, y: float, s: float = 8.0):
    return shapely.box(x - s / 2, y - s / 2, x + s / 2, y + s / 2)


def synthetic_inputs(seed: int = 0) -> R.Inputs:
    gen = np.random.default_rng(seed)
    n_b = 700
    bx, by = gen.uniform(20, 1980, n_b), gen.uniform(20, 1980, n_b)
    buildings = np.array([_box(float(x), float(y)) for x, y in zip(bx, by, strict=True)], dtype=object)
    seeds = []
    hits = gen.random((5, n_b)) < 0.7
    for k in range(5):
        fp = [_box(*gen.uniform(20, 1980, 2)) for _ in range(40)]
        seeds.append(np.array([*buildings[hits[k]], *fp], dtype=object))
    n_hits = hits.sum(axis=0)
    fin = n_hits >= 3
    final_fp = np.array([_box(*gen.uniform(20, 1980, 2)) for _ in range(60)], dtype=object)
    final_geoms = np.array([*buildings[fin], *final_fp], dtype=object)
    conf = np.r_[n_hits[fin] / 5 * gen.uniform(0.4, 0.9, fin.sum()),
                 gen.uniform(0.1, 0.4, len(final_fp))]
    final = pd.DataFrame({"conf": conf, "geometry": final_geoms})

    # recall frame: building points (some offset) + empty points; strata A / B
    real = gen.random(n_b) < 0.85
    fx = np.r_[bx[real] + gen.normal(0, 1.0, real.sum()), gen.uniform(20, 1980, 250)]
    fy = np.r_[by[real] + gen.normal(0, 1.0, real.sum()), gen.uniform(20, 1980, 250)]
    is_b = np.r_[np.ones(real.sum(), bool), np.zeros(250, bool)]
    strat = np.where(gen.random(len(fx)) < 0.6, "A", "B")
    frame = pd.DataFrame({"stratum": strat, "x": fx, "y": fy})
    pick = np.concatenate([gen.choice(np.flatnonzero(strat == s), 70, replace=False)
                           for s in ("A", "B")])
    big_n = {s: int((strat == s).sum()) for s in ("A", "B")}
    recall = pd.DataFrame({"item_id": [f"rp{i:04d}" for i in range(len(pick))],
                           "stratum": strat[pick], "frame_n": [big_n[s] for s in strat[pick]],
                           "x": fx[pick], "y": fy[pick]})
    s1 = {}
    for item, b in zip(recall["item_id"], is_b[pick], strict=True):
        u = gen.random()
        s1[item] = "unsure" if u < 0.05 else ("yes" if b else "no")
    near = {item: {"answer": "yes", "click_m": 3.0, "click_dx_m": 2.0, "click_dy_m": 2.0}
            for item, a in list(s1.items())[:10] if a != "yes"}
    spec = yaml.safe_load((REPO / S.CONFIG).read_text(encoding="utf-8"))
    codings = R.recall_codings(recall[["item_id", "x", "y"]], s1, near, spec)

    # sitting-2 units: detector_final boxes (domain) and minority boxes
    n_fin = len(final_geoms)
    unit_f = gen.choice(n_fin, 90, replace=False)
    minority = np.array([_box(*gen.uniform(20, 1980, 2)) for _ in range(40)], dtype=object)
    rows = []
    for k, i in enumerate(unit_f):
        c = shapely.centroid(final_geoms[i])
        rows.append({"item_id": f"pp{k:04d}", "unit": k, "stratum": "share2_chan0",
                     "frame_n": n_fin, "x": c.x, "y": c.y, "in_final": True,
                     "raw_score": conf[i], "answer": "yes" if i < fin.sum() else "no",
                     "n_final_boxes": 1,
                     **{f"in_seed{s}": bool(gen.random() < 0.8) for s in range(5)}})
    for k, g in enumerate(minority):
        c = shapely.centroid(g)
        rows.append({"item_id": f"pq{k:04d}", "unit": 1000 + k, "stratum": "share0_chan0",
                     "frame_n": 200, "x": c.x, "y": c.y, "in_final": False,
                     "raw_score": np.nan, "answer": "no" if gen.random() < 0.8 else "yes",
                     "n_final_boxes": 0,
                     **{f"in_seed{s}": bool(gen.random() < 0.3) for s in range(5)}})
    prec = pd.DataFrame(rows)
    invs = {name: buildings[gen.random(n_b) < share]
            for name, share in (("L2_seed0", 0.3), ("L2_seed1", 0.35), ("L2_seed2", 0.32),
                                ("google_v3", 0.5), ("microsoft", 0.4), ("overture", 0.7),
                                ("osm", 0.35))}
    blocks = pd.DataFrame({"block_id": [f"b{i}{j}" for i in range(4) for j in range(4)],
                           "geometry": [shapely.box(500 * i, 500 * j, 500 * (i + 1),
                                                    500 * (j + 1))
                                        for i in range(4) for j in range(4)]})
    skill = pd.DataFrame({"inventory": ["detector_final", "google_v3", "google_v3"],
                          "reference": ["R1", "R1", "R1-aud"], "auc": [0.7, 0.65, 0.6],
                          "pr_auc": [0.2, 0.18, 0.15]})
    # Microsoft (main inventory, prereg v8): its polygons plus false outlines, the recall-frame
    # flag in_microsoft and a polygon-precision sample (sitting 2 precision_microsoft)
    ms_false = np.array([_box(*gen.uniform(20, 1980, 2)) for _ in range(50)], dtype=object)
    ms = np.array([*invs["microsoft"], *ms_false], dtype=object)
    invs["microsoft"] = ms
    in_ms = R.near_any(fx, fy, ms, 1.0)
    frame["in_microsoft"] = in_ms
    recall["in_microsoft"] = in_ms[pick]
    n_real = len(ms) - len(ms_false)
    strata = np.where(np.arange(len(ms)) < n_real, "agree_chan0", "ms_only_chan0")
    pm = []
    for s in ("agree_chan0", "ms_only_chan0"):
        idx = np.flatnonzero(strata == s)
        for i in gen.choice(idx, 40, replace=False):
            c = shapely.centroid(ms[i])
            pm.append({"item_id": f"pm{len(pm):04d}", "stratum": s, "frame_n": len(idx),
                       "answer": "yes" if i < n_real else "no", "x": c.x, "y": c.y})
    return R.Inputs(prec=prec, recall=recall, codings=codings, outside={"yes": 0, "n": 50},
                    frame=frame, final=final, seeds=seeds, inventories=invs, blocks=blocks,
                    event_skill=skill, hand=FakeHand(),
                    inv_prec={"microsoft": pd.DataFrame(pm)})


def small_spec() -> dict:
    spec = yaml.safe_load((REPO / S.CONFIG).read_text(encoding="utf-8"))
    s = copy.deepcopy(spec)
    s["u3"]["bootstrap"]["n_boot"] = 60
    s["u3"]["split_sample"].update(repeats=8, n_boot=40)
    s["u3"]["simulation"].update(replicates=8, n_boot=40)
    s["u2"]["primary"]["audit_replicates"]["n"] = 6
    s["u2"]["share_ci"]["n_boot"] = 10
    return s


class SyntheticLoader:
    def __init__(self, problems=None):
        self._problems = problems or []
        self.loaded = 0

    def problems(self, project, spec):
        return list(self._problems)

    def load(self, project, spec):
        self.loaded += 1
        return synthetic_inputs()


def _project(root: Path, spec: dict):
    (root / "configs" / "splits").mkdir(parents=True)
    split = root / "configs" / "splits" / "split_v1.yaml"
    split.write_text("tiles: {}\n", encoding="utf-8")
    (root / "configs" / "uncertainty").mkdir()
    cfg = root / S.CONFIG
    cfg.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    sha = hashlib.sha256(split.read_bytes()).hexdigest()
    (root / "configs" / "prereg.yaml").write_text(
        f"version: prereg_v5\nsplit: {{file: configs/splits/split_v1.yaml, sha256: {sha}}}\n"
        f"uncertainty_propagation:\n  config:\n    sha256_of: {S.CONFIG}\n"
        f"    sha256: {hashlib.sha256(cfg.read_bytes()).hexdigest()}\n", encoding="utf-8")
    project = load_project(root)
    guard.freeze(project)
    return project


def test_analyse_synthetic_outputs_are_complete():
    out = R.analyse(synthetic_inputs(), small_spec())
    assert set(out) == set(R.OUTPUTS)
    ppi = out["ppi_tier_counts.csv"]
    prim = ppi.loc[(ppi["coding"] == "primary") & (ppi["estimator"] == "ppi_pp")]
    assert len(prim) >= 1 and np.isfinite(prim[["estimate", "ci_lo", "ci_hi"]]).all().all()
    assert set(ppi["coding"]) >= {"primary", "v1", "v2_r4", "v2_unsure_yes", "v2_unsure_no"}
    assert {"naive", "ratio", "ht", "precision_only", "ensemble_expected"} <= set(
        ppi["estimator"])
    assert set(ppi["inventory"]) == {"microsoft", "detector_final"}
    # prereg v8 count units: one primary row set = Microsoft objects x polygon precision
    primary = ppi.loc[ppi["role"] == "primary"]
    assert set(primary["inventory"]) == {"microsoft"} and set(primary["coding"]) == {"primary"}
    assert set(primary["estimator"]) == {"precision_only"}
    assert set(primary["count_unit"]) == {"objects"}
    assert set(ppi.loc[ppi["estimator"].isin(["ppi_pp", "ht"]), "role"]) == {
        "upper_bound_frame_points"}
    assert set(ppi.loc[ppi["estimator"] == "ratio", "role"]) == {"sensitivity_mixed_units"}
    assert set(ppi.loc[ppi["estimator"] == "ratio", "count_unit"]) == {"mixed"}
    fals = out["falsifiers.json"]
    assert set(fals) >= {"ece", "ppi", "decomposition", "intervals_computable"}
    assert isinstance(fals["ece"]["pass"], bool) and isinstance(fals["ppi"]["pass"], bool)
    var = out["variance_decomposition.csv"]
    assert {"primary", "ratio_mixed_units", "descriptive"} == set(var["analysis"])
    cov = out["coverage_check.json"]
    assert cov["split_sample"]["repeats"] == 8 and cov["split_sample"]["mode"] == "stratified_random"
    roles = {r["mode"]: r["role"] for r in cov["split_rates"]
             if r["inventory"] == "detector_final"}
    assert roles == {"stratified_random": "gating", "block": "descriptive"}
    reported = [r for r in cov["split_rates"] if r["inventory"] == "microsoft"]
    assert [r["role"] for r in reported] == ["reported"]          # never gating (C2 frozen)
    assert fals["ppi"]["inventory"] == "detector_final"
    assert fals["ppi"]["reported_other_inventories"] == reported
    assert all(np.isfinite(r["expected_miss_rate"]) for r in cov["split_rates"])
    assert out["falsifiers.json"]["ppi"]["split_mode"] == "stratified_random"
    long = out["tier_counts_factorial.parquet"]
    assert long["member"].nunique() == len(CFGS) * 3 and long["inventory"].nunique() == 14
    assert fals["factorial_problems"] == []
    d4 = out["inventory_comparison.csv"].set_index("inventory")
    assert d4.loc["microsoft", "delta_total_pct"] == pytest.approx(0.0)     # v8 baseline
    assert set(d4["baseline"]) == {"microsoft"}
    assert d4.loc["detector_final", "delta_total_pct"] != pytest.approx(0.0)
    assert d4.loc["google_v3", "event_reference"] == "R1"
    assert set(out["ece.csv"]["method"]) == {"temperature", "isotonic"}


def test_recall_codings_use_clicked_point_and_bounds():
    items = pd.DataFrame({"item_id": ["a", "b", "c"], "x": [0.0, 10.0, 20.0],
                          "y": [0.0, 0.0, 0.0]})
    s1 = {"a": "yes", "b": "no", "c": "unsure"}
    near = {"b": {"answer": "yes", "click_m": 5.0, "click_dx_m": 3.0, "click_dy_m": 4.0},
            "c": {"answer": "unsure", "click_m": None, "click_dx_m": None, "click_dy_m": None}}
    spec = yaml.safe_load((REPO / S.CONFIG).read_text(encoding="utf-8"))
    out = R.recall_codings(items, s1, near, spec)
    p = out["primary"].set_index("item_id")
    assert p.loc["b", "answer"] == "yes" and (p.loc["b", "px"], p.loc["b", "py"]) == (13.0, 4.0)
    assert out["v2_r4"].set_index("item_id").loc["b", "answer"] == "no"   # 5 m > 4 m
    assert out["v1"].set_index("item_id").loc["b", "answer"] == "no"
    assert out["v2_unsure_yes"].set_index("item_id").loc["c", "answer"] == "yes"
    assert out["v2_unsure_no"].set_index("item_id").loc["c", "answer"] == "no"


def test_precision_units_tier_point_is_the_final_box():
    items = pd.DataFrame({"item_id": ["i0", "i1"], "unit": [0, 1], "stratum": "s",
                          "frame_n": 10, "x": [0.0, 50.0], "y": [0.0, 0.0]})
    members = pd.DataFrame({"cluster": [0, 0, 1], "run": ["s0", "ens", "s0"],
                            "det": [0, 0, 1], "conf": [0.5, 0.4, 0.3]})
    boxes = pd.DataFrame({"run": ["s0", "ens", "s0"], "det": [0, 0, 1],
                          "geometry": [_box(1, 1), _box(2, 2), _box(50, 0)]})
    u = R.precision_units(items, {"i0": "yes"}, members, boxes, "ens", ["s0"]).set_index("unit")
    assert bool(u.loc[0, "in_final"]) and (u.loc[0, "x"], u.loc[0, "y"]) == (2.0, 2.0)
    assert not bool(u.loc[1, "in_final"]) and u.loc[1, "x"] == 50.0
    assert u.loc[0, "answer"] == "yes" and pd.isna(u.loc[1, "answer"])


def test_run_refuses_while_held_and_journals_nothing(root):
    project = _project(root, small_spec())
    loader = SyntheticLoader()
    res = R.run_uncertainty(project, loader=loader, spec=small_spec())
    assert not res["ran"] and any(p.startswith("hold:") for p in res["problems"])
    R.write_hold(project.paths.runs / R.HOLD_FILE, True, "coordinator")
    res = R.run_uncertainty(project, loader=loader, spec=small_spec())
    assert any("does not release" in p for p in res["problems"])
    R.write_hold(project.paths.runs / R.HOLD_FILE, False, "coordinator")
    res = R.run_uncertainty(project, loader=SyntheticLoader(["detector_final missing"]),
                            spec=small_spec())
    assert res["problems"] == ["detector_final missing"]
    dry = R.run_uncertainty(project, loader=loader, spec=small_spec(), dry_run=True)
    assert dry["ready"] and not dry["ran"]
    assert loader.loaded == 0 and guard.read_test_journal(project) == []


def test_run_refuses_when_prereg_drifts(root):
    project = _project(root, small_spec())
    R.write_hold(project.paths.runs / R.HOLD_FILE, False, "coordinator")
    cfg = root / S.CONFIG
    cfg.write_text(cfg.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    res = R.run_uncertainty(project, loader=SyntheticLoader(), spec=small_spec())
    assert any("uncertainty.yaml sha256" in p for p in res["problems"])
    assert guard.read_test_journal(project) == []


def test_run_end_to_end_once(root):
    project = _project(root, small_spec())
    R.write_hold(project.paths.runs / R.HOLD_FILE, False, "coordinator")
    loader = SyntheticLoader()
    res = R.run_uncertainty(project, loader=loader, spec=small_spec())
    assert res["ran"] and loader.loaded == 1
    journal = guard.read_test_journal(project)
    assert [e["run"] for e in journal] == [R.RUN_NAME]
    for name in R.OUTPUTS:
        assert (project.paths.artifacts / "uncertainty" / name).is_file()
    again = R.run_uncertainty(project, loader=loader, spec=small_spec())
    assert not again["ran"] and any("already run once" in p for p in again["problems"])
    assert loader.loaded == 1 and len(guard.read_test_journal(project)) == 1


def test_greenhouses_dropped_from_every_inventory_and_unit():
    """Greenhouses are not buildings (prereg `greenhouses`): boxes overlapping an OSM
    greenhouse area leave detector_final, the seeds, the other inventories and the audit units
    before any tier count, D4 or U2 row."""
    base = synthetic_inputs()
    base.prec["geometry"] = [_box(float(x), float(y)) for x, y in zip(base.prec["x"],
                                                                      base.prec["y"],
                                                                      strict=True)]
    gh = np.array([shapely.box(0, 0, 600, 2000)], dtype=object)   # west strip
    inp = dataclasses.replace(base, greenhouses=gh, greenhouse_iou=0.3)
    clean, removed = R.drop_greenhouses(inp)
    for name, geoms in [("detector_final", clean.final["geometry"].to_numpy()),
                        *[(f"seed{k}", s) for k, s in enumerate(clean.seeds)],
                        *clean.inventories.items()]:
        cx = shapely.get_x(shapely.centroid(np.asarray(geoms, dtype=object)))
        assert (cx > 596).all(), name                       # nothing left in the strip
        assert removed[name] > 0
    assert removed["precision_units"] > 0 and (clean.prec["x"] > 596).all()
    out_gh = R.analyse(inp, small_spec())
    out = R.analyse(base, small_spec())
    assert out_gh["coverage_check.json"]["greenhouses_removed"]["detector_final"] == removed[
        "detector_final"]
    d4_gh = out_gh["inventory_comparison.csv"].set_index("inventory")
    d4 = out["inventory_comparison.csv"].set_index("inventory")
    assert d4_gh.loc["detector_final", "total"] == d4.loc["detector_final", "total"] - removed[
        "detector_final"]
    assert d4_gh.loc["seed0", "total"] == d4.loc["seed0", "total"] - removed["seed0"]


def test_u3_main_inventory_counts_objects_and_needs_its_precision_units():
    """Microsoft polygons all audited Yes -> its exposure count D_t p_g equals the raw D_t; the
    analysis refuses a u3 inventory without its own polygon-precision units."""
    base = synthetic_inputs()
    pm = base.inv_prec["microsoft"].assign(answer="yes")
    out = R.analyse(dataclasses.replace(base, inv_prec={"microsoft": pm}), small_spec())
    ppi = out["ppi_tier_counts.csv"]
    ms = ppi.loc[(ppi["inventory"] == "microsoft") & (ppi["coding"] == "primary")]
    po = ms.loc[ms["estimator"] == "precision_only"].set_index("tiers")["estimate"]
    raw = ms.loc[ms["estimator"] == "naive"].set_index("tiers")["estimate"]
    assert raw.sum() > 0
    pd.testing.assert_series_equal(po, raw, check_names=False)
    with pytest.raises(R.UncertaintyError, match="precision units"):
        R.analyse(dataclasses.replace(base, inv_prec={}), small_spec())
    spec = small_spec()
    spec["u3"]["inventories"] = ["microsoft"]
    with pytest.raises(R.UncertaintyError, match="detector_final"):
        R.analyse(base, spec)
