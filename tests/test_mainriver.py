"""Main-river recompute (nakkhu.mainriver): synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import LineString

from nakkhu.hydro.dem import GridSpec
from nakkhu.mainriver import analyses as A
from nakkhu.mainriver import remote as R
from nakkhu.mainriver import spec as S
from nakkhu.mainriver import surfaces as F


def _grid(width: int = 5, height: int = 1, res: float = 1000.0) -> GridSpec:
    return GridSpec("EPSG:32645", Affine(res, 0, 0, 0, -res, height * res), width, height)


def test_hand_mr_raster_follows_d8_to_the_main_river() -> None:
    grid = _grid()
    fdir = np.full((1, 5), 1, dtype=np.int64)          # every cell flows east (ESRI 1)
    fdir[0, 4] = 0                                     # the last cell is a pit
    acc = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])        # cells of 1 km2: area = acc km2
    z = np.array([[20.0, 15.0, 9.0, 5.0, 4.0]])
    line = gpd.GeoDataFrame(geometry=[LineString([(3100, 500), (3900, 500)])], crs=grid.crs)
    hand, info = F.hand_mr_raster(fdir, acc, z, np.ones((1, 5), dtype=bool), line, grid, 1.0)
    # M = cell 3 only (touched by the line, area >= 1 km2): HAND = z - z(cell 3)
    np.testing.assert_allclose(hand[0, :4], [15.0, 10.0, 4.0, 0.0])
    assert np.isnan(hand[0, 4])                        # downstream of M, the path never meets it
    assert info["main_cells"] == 1 and info["n_nan_in_domain"] == 1


def test_hand_mr_raster_respects_the_area_rule_and_domain() -> None:
    grid = _grid()
    fdir = np.full((1, 5), 1, dtype=np.int64)
    acc = np.array([[0.1, 0.2, 0.3, 0.5, 2.0]])        # only cell 4 reaches 1 km2
    z = np.array([[10.0, 8.0, 6.0, 4.0, 3.0]])
    line = gpd.GeoDataFrame(geometry=[LineString([(500, 500), (4500, 500)])], crs=grid.crs)
    dom = np.array([[False, True, True, True, True]])
    hand, info = F.hand_mr_raster(fdir, acc, z, dom, line, grid, 1.0)
    # not masked to the domain (as the sweep HAND): the cell outside it keeps its value
    np.testing.assert_allclose(hand[0], [7.0, 5.0, 3.0, 1.0, 0.0])
    assert info["main_cells"] == 1 and info["n_domain_cells"] == 4


def _sweep(tmp: Path, ref: str, rows: list[dict]) -> None:
    d = tmp / "runs" / ref
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "burn_check.csv", index=False)


def test_surfaces_apply_the_routing_gate(project) -> None:
    root = project.paths.root
    _sweep(root, "hand/a", [
        {"dem": "glo30", "conditioning": "fill_burn", "gate_pass": True, "gate_reason": ""},
        {"dem": "glo30", "conditioning": "breach_burn", "gate_pass": True, "gate_reason": ""},
        {"dem": "gedtm30", "conditioning": "breach_burn", "gate_pass": False,
         "gate_reason": "channel share 0.9867 < 0.99"}])
    _sweep(root, "hand-dems/b", [
        {"dem": "nasadem", "conditioning": "fill_burn", "gate_pass": True, "gate_reason": ""}])
    cfg = {"sources": {"canonical": "hand/a", "dem_extension": "hand-dems/b"},
           "ensemble": {"source": "canonical", "dems": ["glo30", "gedtm30"],
                        "conditioning": ["breach_burn"]},
           "extended": {"source": "dem_extension", "dems": ["nasadem"],
                        "conditioning": ["fill_burn"]},
           "primary": {"dem": "glo30", "conditioning": "breach_burn", "statistic": "centroid"}}
    out = {s.cfg_id: s for s in S.surfaces(project, cfg)}
    assert set(out) == {"glo30__breach_burn__mr", "gedtm30__breach_burn__mr",
                        "nasadem__fill_burn__mr"}
    assert out["glo30__breach_burn__mr"].is_primary
    assert out["glo30__breach_burn__mr"].in_sensitivity_range
    assert not out["gedtm30__breach_burn__mr"].computed
    assert out["nasadem__fill_burn__mr"].computed
    assert not out["nasadem__fill_burn__mr"].in_sensitivity_range   # extended: descriptive
    cfg["ensemble"]["conditioning"] = ["breach_noburn"]
    with pytest.raises(S.MainRiverError):
        S.surfaces(project, cfg)


def test_tier_ranges_keep_extended_members_out_of_the_primary_range() -> None:
    labels = ["on_drainage", "0-3", "3-6", "6-9", "9-12", ">12"]
    rows = []
    for cid, n0, inr, gate in (("a__breach_burn__mr", 10, True, True),
                               ("b__fill_burn__mr", 30, True, True),
                               ("c__fill_burn__mr", 99, False, True)):
        rows.append({"inventory": "microsoft", "cfg_id": cid, "dem": cid[0],
                     "conditioning": cid.split("__")[1], "threshold": "mr",
                     "statistic": "centroid", "is_primary": cid.startswith("a"),
                     "in_sensitivity_range": inr, "routing_gate_pass": gate,
                     "kappa_linear_vs_primary": 1.0,
                     **{f"n_{lab}": (n0 if lab == "0-3" else 1) for lab in labels}})
    out = F.tier_ranges(pd.DataFrame(rows), labels).set_index("group")
    prim = out.loc["all surfaces x statistics"]
    assert (prim["n_members"], prim["n_0-3_min"], prim["n_0-3_max"]) == (2, 10, 30)
    ext = out.loc["extended: all gate-passing DEMs x statistics"]
    assert (ext["n_members"], ext["n_0-3_max"]) == (3, 99)


def test_ece_and_h6_skill() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    # two equal-count bins: |0.1 - 0.5| and |0.9 - 1.0| -> mean 0.25
    assert A.ece(p, y, None, 2) == pytest.approx(0.25)
    rng = np.random.default_rng(0)
    n = 600
    hand = rng.uniform(0, 20, n)
    y = (rng.random(n) < 1 / (1 + np.exp(hand - 4))).astype(int)
    sub = pd.DataFrame({"h": hand, "block_id": np.repeat(np.arange(6), n // 6).astype(str)})
    rows = {r["method"]: r for r in A.h6_skill(sub, y, None, "h", 10)}
    for r in rows.values():
        assert 0 <= r["ece"] < 0.2
        assert r["bss"] == pytest.approx(1 - r["brier"] / (r["prevalence_scored"]
                                                            * (1 - r["prevalence_scored"])))
        assert r["bss"] > 0


def test_team_rows_score_each_member_on_its_own_coverage() -> None:
    rng = np.random.default_rng(1)
    n = 400
    d = rng.uniform(0, 1000, n)
    y = (rng.random(n) < np.exp(-d / 150)).astype(int)
    sub = pd.DataFrame({"building_uid": [f"m:{i}" for i in range(n)], "dist_channel_m": d,
                        "h": d / 50, "block_id": np.repeat(np.arange(10), n // 10).astype(str)})
    team = pd.DataFrame({"building_uid": sub["building_uid"],
                         "hand_team_acc__glo30__breach_burn":
                             np.where(np.arange(n) < 300, d / 40, np.nan)})
    ev = {"n_boot": 50, "seed": 0, "min_cell": 5}
    rows = A.team_rows(sub, y, None, team, "h", ev, inventory="microsoft", reference="R1")
    assert {r["comparator"] for r in rows} == {"distance_to_channel", "hand_mr"}
    assert all(r["n"] == 300 and r["n_reference_rows"] == n for r in rows)


def _receipt(lines: list[str], index: int, note: str, retry: bool = False) -> dict:
    entry = json.loads(lines[index - 1])
    return {"run": entry["run"], "run_cfghash": entry["eval_cfghash"], "retry": retry,
            "journal": {"index": index, "entry": {"note": note},
                        "line_sha256": R.sha256_bytes(lines[index - 1].encode("utf-8"))}}


def test_journal_line_check_is_one_shot() -> None:
    line = lambda run, note: json.dumps({"run": run, "eval_cfghash": "h", "note": note},  # noqa: E731
                                        sort_keys=True)
    lines = [line("a/1", "x"), line("mainriver/1", "main_river_v1:event")]
    assert R._line_problems(lines, _receipt(lines, 2, "main_river_v1:event"), "j") == []
    twice = [*lines, line("mainriver/2", "main_river_v1:event")]
    assert R._line_problems(twice, _receipt(twice, 3, "main_river_v1:event"), "j")
    altered = [lines[0], lines[1].replace("main_river", "main-river")]
    assert R._line_problems(altered, _receipt(lines, 2, "main_river_v1:event"), "j")


def test_outputs_and_notes() -> None:
    rc = {"version": "main_river_v1"}
    assert R.note_of(rc, "exposure") == "main_river_v1:exposure"
    assert R.note_of(rc, "event", retry=True) == "main_river_v1:event retry"
    assert "ppi_tier_counts.csv" in R.outputs_of("exposure")
    assert "h6_skill.csv" in R.outputs_of("event")
    assert not set(R.LOCAL_ONLY) & set(R.outputs_of("event"))
    assert R.note_of(rc, "roc") == "main_river_v1:roc"
    assert not set(R.LOCAL_ONLY) & set(R.outputs_of("roc"))


def test_shipped_configs_parse(project) -> None:
    real = Path(__file__).resolve().parents[1]
    from nakkhu.config import load_yaml

    ens = load_yaml(real / S.ENSEMBLE_CONFIG)
    rc = load_yaml(real / S.RECOMPUTE_CONFIG)
    assert S.primary_column(ens) == "glo30__breach_burn__mr__centroid"
    assert S.statistics(ens) == ["centroid", "footprint_min", "buffer_min"]
    assert set(rc["analyses"]) == {"exposure", "event", "roc"}


def test_roc_predictor_curves_match_auc_and_thin_per_predictor():
    from nakkhu.flood import curves as C
    from nakkhu.flood import validate as V

    rng = np.random.default_rng(3)
    n = 400
    ens = {"primary": {"dem": "glo30", "conditioning": "breach_burn", "statistic": "centroid"}}
    pcol = S.primary_column(ens)
    hand = rng.gamma(2.0, 4.0, n)
    hand[:5] = np.nan                                   # not on the primary surface: dropped
    dist = hand * 20 + rng.normal(0, 30, n)
    dist[5:8] = np.nan                                  # distance not finite: dropped for it
    y = (rng.random(n) < 1 / (1 + np.exp(0.6 * (np.nan_to_num(hand, nan=50) - 4)))).astype(int)
    uid = [f"b{i}" for i in range(n)]
    pred = pd.DataFrame({"building_uid": uid, "block_id": rng.integers(0, 9, n),
                         "dist_channel_m": dist, pcol: hand})
    obs = np.ones(n, dtype=bool)
    obs[-20:] = False                                   # unobservable: not in R1
    change = pd.DataFrame({"building_id": uid, "observable": obs,
                           "class": np.where(y == 1, "positive", "negative")})
    est = {"estimates": {"ppv": {"wilson_lo": 0.7}, "npv": {}}}
    rc = {"analyses": {"roc": {"inventories": ["microsoft"]}}}
    full, summary = A.roc_predictor_curves(None, rc, ens, Path("."), {"microsoft": pred},
                                           {"microsoft": change}, est)
    assert sorted(full["predictor"].unique()) == ["distance_to_channel", "hand_mr"]
    by = {c["predictor"]: c for c in summary["curves"]}
    assert by["hand_mr"]["n"] == n - 5 - 20 and by["distance_to_channel"]["n"] == n - 8 - 20
    for name, c in by.items():
        g = full.loc[full["predictor"] == name]
        assert C.roc_auc_trapezoid(g["fpr"], g["tpr"]) == pytest.approx(c["auc"], abs=1e-12)
    keep = (obs & np.isfinite(hand))
    assert by["hand_mr"]["auc"] == pytest.approx(V.roc_auc(-hand[keep], y[keep]))
    thin, recs = C.thin_table(full, "fpr", "tpr", cap=50,
                              group=["inventory", "reference", "predictor"])
    assert {r["predictor"] for r in recs} == {"hand_mr", "distance_to_channel"}
    assert all(r["kept"] <= 50 and abs(r["auc_full"] - r["auc_thin"]) < 0.01 for r in recs)
    assert set(thin.columns) == set(full.columns)
    est_drop = {"estimates": {"ppv": {"wilson_lo": 0.4}, "npv": {}}}
    with pytest.raises(S.MainRiverError):
        A.roc_predictor_curves(None, rc, ens, Path("."), {"microsoft": pred},
                               {"microsoft": change}, est_drop)
