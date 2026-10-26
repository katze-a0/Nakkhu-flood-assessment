"""``nakkhu hand tier-export``: per-building HAND tier table from an attribute run (synthetic)."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import box

from nakkhu import hazard
from nakkhu.hydro import tier_export as T

EDGES = [0, 3, 6, 9, 12]
LABELS = hazard.tier_labels(EDGES)
CFG = "glo30__breach_burn__tdrop"
COLUMN = f"{CFG}__centroid"
# one building per tier plus an exact-edge value (3.0 m is in 0-3) and one missing value
HAND = np.array([0.0, 1.5, 3.0, 4.0, 7.0, 10.0, 12.0000005, 30.0, np.nan], dtype=np.float32)


def _sweep_yaml(root, *, tiers=EDGES):
    extent = gpd.GeoDataFrame(geometry=[box(330_000, 3_050_000, 331_000, 3_051_000)],
                              crs="EPSG:32645")
    (root / "configs" / "aoi").mkdir(parents=True, exist_ok=True)
    extent.to_file(root / "configs" / "aoi" / "extent.geojson", driver="GeoJSON")
    cfg = {"grid": {"crs": "EPSG:32645"}, "analysis_extent": "configs/aoi/extent.geojson",
           "tiers_m": list(tiers), "statistics": ["centroid"], "buffer_m": 10,
           "greenhouse_filter": {"enabled": False},
           "primary": {"dem": "glo30", "conditioning": "breach_burn", "threshold": "drop",
                       "threshold_km2": 0.2, "statistic": "centroid"},
           "inventories": {"fp": {"path": "data/fp.parquet"},
                           "other": {"path": "data/fp.parquet"}}}
    (root / "configs" / "hand").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "hand" / "sweep.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _inventory(root):
    # 9 buildings inside the extent and one outside (dropped by the extent filter)
    geoms = [box(330_010 + 100 * i, 3_050_100, 330_030 + 100 * i, 3_050_112) for i in range(9)]
    geoms.append(box(340_000, 3_060_000, 340_010, 3_060_010))
    frame = gpd.GeoDataFrame({"source_id": list(range(10))}, geometry=geoms, crs="EPSG:32645")
    (root / "data").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "data" / "fp.parquet")
    return frame.iloc[:9]


def _attribute_run(root, inside, *, run_id="20261012T000000Z-aaaaaaaa", hand=HAND):
    run = root / "runs" / "hand-attribute" / run_id
    run.mkdir(parents=True)
    tiers = hazard.tier_index(hand, EDGES)
    counts = hazard.tier_counts(tiers, len(LABELS))
    pd.DataFrame({"building_uid": [f"fp:{i}" for i in range(len(inside))],
                  "source_id": inside["source_id"].to_numpy(),
                  "area_m2": inside.geometry.area.astype(np.float32).to_numpy(),
                  COLUMN: hand}).to_parquet(run / "building_hand_fp.parquet", index=False)
    pd.DataFrame([{"inventory": "fp", "cfg_id": CFG, "statistic": "centroid",
                   "is_primary": True, **{f"n_{lab}": int(n) for lab, n in
                                          zip(LABELS, counts, strict=True)}}]).to_csv(
        run / "count_stability.csv", index=False)
    (run / "summary.json").write_text(json.dumps({
        "primary": COLUMN, "inventories": {"fp": {"n_in_extent": 9, "n_final": 9,
                                                  "n_buildings": 9, "role": "primary"}}}),
        encoding="utf-8")
    (run / "journal.jsonl").write_text(
        '{"event": "start", "stage": "attribute-fp", "utc": "x"}\n'
        '{"event": "end", "returncode": 0, "stage": "attribute-fp", "utc": "x"}\n',
        encoding="utf-8")
    return run, counts


def _ppi(root, counts, inventory="fp"):
    rows = [{"coding": "primary", "inventory": inventory, "level": "tier", "group": 0,
             "tiers": lab, "estimator": "naive", "count_unit": "objects",
             "role": "raw_count", "estimate": float(n)} for lab, n in
            zip(LABELS, counts, strict=True)]
    rows.append({"coding": "primary", "inventory": inventory, "level": "tier", "group": 0,
                 "tiers": "0-3", "estimator": "precision_only", "count_unit": "objects",
                 "role": "primary", "estimate": 0.5})
    (root / "artifacts" / "uncertainty").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "artifacts" / "uncertainty" / "ppi_tier_counts.csv",
                              index=False)


@pytest.fixture
def setup(project):
    root = project.paths.root
    _sweep_yaml(root)
    inside = _inventory(root)
    run, counts = _attribute_run(root, inside)
    return project, inside, run, counts


def test_tier_table_tiers_ids_and_centroids(setup):
    _, inside, run, _ = setup
    frame = inside.reset_index(drop=True).copy()
    frame.insert(0, "building_uid", [f"fp:{i}" for i in range(len(frame))])
    hand = pd.read_parquet(run / "building_hand_fp.parquet")
    out = T.tier_table(frame, hand, COLUMN, EDGES)
    assert list(out.columns) == T.COLUMNS
    assert out["tier"].tolist()[:8] == ["on_drainage", "0-3", "0-3", "3-6", "6-9", "9-12",
                                        ">12", ">12"]
    assert pd.isna(out["tier"].iloc[8])
    assert out.loc[0, "x"] == pytest.approx(330_020.0) and out.loc[0, "y"] == pytest.approx(
        3_050_106.0)
    # a CSV round trip keeps every tier when the HAND values are re-tiered
    back = pd.read_csv(pd.io.common.StringIO(out.to_csv(index=False)))
    np.testing.assert_array_equal(hazard.tier_index(back["hand_m"].to_numpy(), EDGES),
                                  hazard.tier_index(HAND, EDGES))


def test_tier_table_refuses_mismatched_ids_areas_and_crs(setup):
    _, inside, run, _ = setup
    frame = inside.reset_index(drop=True).copy()
    frame.insert(0, "building_uid", [f"fp:{i}" for i in range(len(frame))])
    hand = pd.read_parquet(run / "building_hand_fp.parquet")
    with pytest.raises(T.TierExportError, match="order"):
        T.tier_table(frame, hand.iloc[::-1].reset_index(drop=True), COLUMN, EDGES)
    with pytest.raises(T.TierExportError, match="areas"):
        T.tier_table(frame, hand.assign(area_m2=hand["area_m2"] + 5), COLUMN, EDGES)
    with pytest.raises(T.TierExportError, match="buildings"):
        T.tier_table(frame.iloc[:-1], hand, COLUMN, EDGES)
    with pytest.raises(T.TierExportError, match="column"):
        T.tier_table(frame, hand, "other__centroid", EDGES)
    with pytest.raises(T.TierExportError, match="CRS"):
        T.tier_table(frame.to_crs("EPSG:4326"), hand, COLUMN, EDGES)


def test_promoted_counts_reads_only_primary_naive_rows():
    ppi = pd.DataFrame([{"coding": c, "inventory": "fp", "level": "tier", "tiers": lab,
                         "estimator": e, "estimate": 7.0 if e == "naive" else 0.3}
                        for c in ("primary", "sensitivity") for e in ("naive", "ratio")
                        for lab in LABELS])
    assert T.promoted_counts(ppi, "fp", LABELS) == dict.fromkeys(LABELS, 7)
    assert T.promoted_counts(ppi, "missing", LABELS) is None
    with pytest.raises(T.TierExportError, match="tiers"):
        T.promoted_counts(ppi[ppi["tiers"] != ">12"], "fp", LABELS)


def test_export_run_matches_counts_and_records_hashes(setup):
    project, _, run_dir, counts = setup
    _ppi(project.paths.root, counts)
    run, summary = T.export_run(project, ["fp"])
    info = summary["inventories"]["fp"]
    assert info["attribute_run"] == f"hand-attribute/{run_dir.name}"
    assert info["counts_match"] == {"attribute_run": True, "promoted": True}
    assert info["rows"] == 9 and info["n_missing_hand"] == 1
    assert info["counts"] == dict(zip(LABELS, map(int, counts), strict=True))
    for key in ("hand_table_sha256", "inventory_sha256", "count_stability_sha256",
                "output_sha256"):
        assert len(info[key]) == 64
    assert summary["promoted_counts"]["sha256"] is not None
    assert summary["tiers_m"] == EDGES and summary["hand_column"] == COLUMN
    assert summary["test_access"] is False and summary["reads_audit_answers"] is False
    out = pd.read_csv(run.path / "building_tiers_fp.csv")
    assert list(out.columns) == T.COLUMNS and len(out) == 9
    assert run.status == "complete"
    assert not (project.paths.runs / "eval" / "test_journal.jsonl").exists()


def test_export_run_uses_config_edges(project):
    root = project.paths.root
    _sweep_yaml(root, tiers=[0, 5])
    _attribute_run(root, _inventory(root))
    with pytest.raises(T.TierExportError, match="count_stability"):
        T.export_run(project, ["fp"])   # the attribute run's table has the 6-tier columns


def test_export_run_stops_on_count_mismatch(setup):
    project, _, _, counts = setup
    wrong = counts.copy()
    wrong[1] += 1
    _ppi(project.paths.root, wrong)
    with pytest.raises(T.TierExportError, match="promoted"):
        T.export_run(project, ["fp"])
    run_dir = sorted((project.paths.runs / "hand-tiers").iterdir())[-1]
    check = pd.read_csv(run_dir / "count_check_fp.csv")
    assert check.loc[check["tier"] == "0-3", "diff_promoted"].item() == -1
    assert not (run_dir / "building_tiers_fp.csv").exists()
    assert json.loads((run_dir / "summary.json").read_text())["inventories"]["fp"][
        "counts_match"]["promoted"] is False


def test_export_run_without_promoted_rows_reports_not_available(setup):
    project, _, _, counts = setup
    _ppi(project.paths.root, counts, inventory="someone_else")
    _, summary = T.export_run(project, ["fp"])
    assert summary["inventories"]["fp"]["counts_match"] == {"attribute_run": True,
                                                            "promoted": "not_available"}


def test_export_run_checks_ppi_against_index(setup):
    project, _, _, counts = setup
    _ppi(project.paths.root, counts)
    (project.paths.artifacts / "INDEX.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "artifacts": [{"path": T.PPI_ARTIFACT, "sha256": "0" * 64}]}),
        encoding="utf-8")
    with pytest.raises(T.TierExportError, match="INDEX"):
        T.export_run(project, ["fp"])


def test_find_attribute_run_picks_newest_complete_run(setup):
    project, inside, first, _ = setup
    newer, _ = _attribute_run(project.paths.root, inside, run_id="20261013T000000Z-bbbbbbbb")
    assert T.find_attribute_run(project, "fp").run_id == newer.name
    assert T.find_attribute_run(project, "fp", [f"hand-attribute/{first.name}"]).run_id == \
        first.name
    (newer / "journal.jsonl").write_text(
        '{"event": "start", "stage": "attribute-fp", "utc": "x"}\n', encoding="utf-8")
    assert T.find_attribute_run(project, "fp").run_id == first.name
    with pytest.raises(T.TierExportError, match="no complete"):
        T.find_attribute_run(project, "other")
    with pytest.raises(T.TierExportError, match="not a"):
        T.find_attribute_run(project, "fp", ["hand/x"])


def test_cli_registers_tier_export():
    from nakkhu.cli import build_parser

    args = build_parser().parse_args(["hand", "tier-export", "--inventory", "microsoft",
                                      "detector_final"])
    assert args.inventory == ["microsoft", "detector_final"] and args.attribute is None
    assert args.hand_config == "configs/hand/sweep.yaml"
