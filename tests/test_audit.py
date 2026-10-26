"""Plan 006: audit sampling, queue, resumable click server, chips and estimators."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
import yaml
from PIL import Image
from rasterio.transform import from_origin

from nakkhu.audit import core, estimators, frames, sample
from nakkhu.audit.core import AuditError, build_queue, replay

CFG = {
    "schema_version": 1, "seed": 0,
    "scenes": {"pre": "img_pre"},
    "chips": {"context_m": 64.0, "upscale": 2, "jpeg_quality": 85, "prefetch": 3},
    "time": {"cap_min": 120, "sitting_max_min": 35, "sitting_budget_min": 30, "pilot_items": 6,
             "floor_per_task": 10, "repeat_frac": 0.1, "repeat_min_gap": 3},
    "tasks": {
        "recall_points": {"question": "Roof?", "keys": {"yes": "Yes", "no": "No",
                                                         "unsure": "Unsure"},
                          "design_n": 30, "min_per_stratum": 2, "sources": ["google"],
                          "auditor": "teammate_1"},
        "recall_nearmiss": {"question": "Roof in circle?", "auditor": "author",
                            "parent": "recall_points",
                            "keys": {"yes": "Yes", "no": "No", "unsure": "Unsure"},
                            "include_answers": ["no", "unsure"], "exclude_strata": ["outside"],
                            "overlap_n": 4, "radius_m": 8.0, "sensitivity_radius_m": 4.0},
        "precision_pool": {"question": "Box?", "keys": {"yes": "Yes", "no": "No",
                                                         "unsure": "Unsure"},
                           "design_n": 30, "min_per_stratum": 2},
        "flood_trace": {"question": "Trace?", "keys": {"yes": "Yes", "no": "No",
                                                        "unsure": "Not observable"},
                        "design_n": 30, "min_per_stratum": 2, "followup": "damage",
                        "time_factor": 2.0, "auditor": "author"},
        "damage": {"question": "Damage?", "keys": {"yes": "Yes", "no": "No",
                                                    "unsure": "Unsure"}},
    },
}


def _frame(sizes: dict[str, int]) -> pd.DataFrame:
    rows, unit = [], 0
    for stratum, n in sizes.items():
        for _ in range(n):
            rows.append({"unit": unit, "stratum": stratum, "x": 1000.0 + unit, "y": 2000.0})
            unit += 1
    return pd.DataFrame(rows)


def _items(task: str = "recall_points", sizes: dict[str, int] | None = None,
           total: int = 30, pilot: int = 6) -> pd.DataFrame:
    frame = _frame(sizes or {"A": 400, "B": 120, "C": 30, "D": 2})
    design = sample.design_sizes(frame.groupby("stratum").size().to_dict(), total, 2)
    drawn = sample.draw(frame, design, 0, task)
    return sample.finish_items(drawn, task, 0, pilot if task == "recall_points" else 0)


# --- sampling ---------------------------------------------------------------------------------


def test_design_sizes_minimum_take_all_and_total():
    sizes = sample.design_sizes({"A": 400, "B": 100, "C": 30, "D": 2, "E": 0}, 40, 5)
    assert sum(sizes.values()) == 40 and "E" not in sizes
    assert sizes["D"] == 2 and all(v >= 5 for k, v in sizes.items() if k != "D")
    assert sizes["A"] > sizes["B"] > sizes["C"]          # sqrt allocation keeps the order
    assert sample.design_sizes({"A": 3, "B": 2}, 100, 5) == {"A": 3, "B": 2}


def test_draw_is_deterministic_and_pilot_is_lowest_ranks():
    a, b = _items(), _items()
    pd.testing.assert_frame_equal(a, b)
    assert set(a["stratum"]) == {"A", "B", "C", "D"}
    assert a["pilot"].sum() == 6
    assert (a.loc[a["pilot"], "rank"] < a.loc[a["pilot"], "design_n"]).all()
    for _, g in a.groupby("stratum"):
        assert sorted(g["rank"]) == list(range(len(g)))
        if g["pilot"].any():
            assert g.loc[g["pilot"], "rank"].max() < g.loc[~g["pilot"], "rank"].min()
    assert a["item_id"].is_unique and a["item_id"].str.match(r"^rp\d{4}$").all()
    np.testing.assert_allclose(a["pi_design"], a["design_n"] / a["frame_n"])


def test_stratum_sizes_keep_pilot_and_cap_design():
    items = _items()
    sizes = core.stratum_sizes(items, 12)
    pilot = items.loc[items["pilot"]].groupby("stratum").size()
    assert sum(sizes.values()) == 12
    assert all(sizes[k] >= v for k, v in pilot.items())
    design = items.groupby("stratum")["design_n"].first()
    assert core.stratum_sizes(items, 10_000) == design.to_dict()


def test_queue_pilot_first_repeats_after_gap_and_never_in_pilot():
    items = _items()
    pilot_only = build_queue(items, None, "recall_points", CFG)
    assert len(pilot_only) == 6 and not any(s.repeat for s in pilot_only)
    alloc = {"tasks": {"recall_points": {"n": 20}}}
    queue = build_queue(items, alloc, "recall_points", CFG)
    assert [s.slot for s in queue[:6]] == [s.slot for s in pilot_only]
    primary = [s.item_id for s in queue if not s.repeat]
    assert len(primary) == 20 and len(set(primary)) == 20
    repeats = [(i, s) for i, s in enumerate(queue) if s.repeat]
    assert len(repeats) == 2
    for pos, slot in repeats:
        orig = next(i for i, s in enumerate(queue) if s.slot == slot.item_id)
        assert pos > orig and pos >= 6
    assert [s.slot for s in build_queue(items, alloc, "recall_points", CFG)] == \
        [s.slot for s in queue]


def test_replay_undo_and_torn_line(tmp_path):
    log = tmp_path / "r.jsonl"
    core.append_event(log, {"event": "answer", "slot": "a", "question": "q", "answer": "yes",
                            "ms": 900})
    core.append_event(log, {"event": "answer", "slot": "b", "question": "q", "answer": "no",
                            "ms": 1100})
    core.append_event(log, {"event": "undo", "slot": "b", "question": "q"})
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "answer", "slot"')        # torn by a crash
    events = core.read_events(log)
    state = replay(events)
    assert state.stack == [("a", "q")]
    assert core.logged_ms(events) == 2000                  # undone time still counts


# --- server (resume, blindness, pilot) ------------------------------------------------------


@pytest.fixture
def audit_project(project):
    root = project.paths.root
    (root / "configs" / "audit").mkdir(parents=True)
    (root / "configs" / "audit" / "audit.yaml").write_text(yaml.safe_dump(CFG),
                                                           encoding="utf-8")
    items = _items()
    path = core.items_path(project, "recall_points")
    path.parent.mkdir(parents=True)
    items.to_parquet(path, index=False)
    chips = core.chip_dir(project, "recall_points")
    chips.mkdir(parents=True)
    for item in items["item_id"]:
        Image.new("RGB", (8, 8)).save(chips / core.chip_name("recall_points", item))
    return project


def _session(project):
    from nakkhu.audit.server import Session

    return Session(project, "recall_points", core.load_audit_config(project))


def test_kill_and_restart_loses_and_duplicates_nothing(audit_project):
    s = _session(audit_project)
    st = s.state()
    assert st["phase"] == "run" and st["total"] == 6
    assert not core.responses_path(audit_project, "recall_points").exists()   # reads only
    for answer in ("yes", "no", "unsure"):
        st = s.answer(st["pos"], st["qid"], answer, 1500)
    s.answer(0, "main", "yes", 1500)                        # stale double submit: ignored
    del s                                                    # "kill" mid-sitting
    s2 = _session(audit_project)
    st = s2.state()
    assert st["done"] == 3 and st["pos"] == 3
    events = core.read_events(core.responses_path(audit_project, "recall_points"))
    answers = [e for e in events if e["event"] == "answer"]
    assert len(answers) == 3 and len({e["slot"] for e in answers}) == 3
    st = s2.undo()
    assert st["pos"] == 2 and st["done"] == 2


def test_pilot_completion_writes_allocation_and_continues(audit_project):
    s = _session(audit_project)
    st = s.state()
    while st["phase"] == "run":
        st = s.answer(st["pos"], st["qid"], "yes", 3000)
    assert st["phase"] == "pilot_done"
    alloc = core.load_allocation(audit_project)
    assert alloc is not None
    assert alloc["pilot"]["n"] == 6 and alloc["pilot"]["median_s"] == pytest.approx(3.0)
    n = alloc["tasks"]["recall_points"]["n"]
    assert 10 <= n <= 30
    st = s.state()
    assert st["phase"] == "run" and st["total"] > 6 and st["done"] == 6
    with pytest.raises(AuditError):
        from nakkhu.audit.workflow import write_allocation

        write_allocation(audit_project, CFG)                 # written once


def test_http_page_is_blind_and_local(audit_project):
    from nakkhu.audit.server import make_server

    s = _session(audit_project)
    server = make_server(s, 0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = urllib.request.urlopen(base + "/api/state").read().decode()
        items = core.load_items(audit_project, "recall_points")
        for leak in [*items["item_id"], *set(items["stratum"]), "stratum", "item_id", "slot",
                     "repeat", "pi_design"]:
            assert f'"{leak}"' not in body
        assert "#r" not in body
        state = json.loads(body)
        req = urllib.request.Request(base + "/api/answer", method="POST", data=json.dumps(
            {"pos": state["pos"], "qid": state["qid"], "answer": "no", "ms": 800}).encode())
        after = json.loads(urllib.request.urlopen(req).read())
        assert after["done"] == 1
        assert urllib.request.urlopen(base + state["chip"]).status == 200
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/chip/..%2F..%2Fitems.parquet")
        assert b"<title>" in urllib.request.urlopen(base + "/").read()
    finally:
        server.shutdown()
        server.server_close()


# --- chips ------------------------------------------------------------------------------------


def test_chips_render_twenty_items_only_under_interim(project):
    from nakkhu.audit.chips import render_task

    root = project.paths.root
    raw = root / "data" / "raw" / "img_pre"
    raw.mkdir(parents=True)
    tif = raw / "img_pre.tif"
    rng = np.random.default_rng(0)
    with rasterio.open(tif, "w", driver="GTiff", width=600, height=600, count=3, dtype="uint8",
                       crs="EPSG:32645", transform=from_origin(1000, 2200, 0.30517578125,
                                                               0.30517578125)) as ds:
        ds.write(rng.integers(30, 200, (3, 600, 600), dtype=np.uint8))
    (root / "configs" / "data" / "manifest.yaml").write_text(yaml.safe_dump(
        {"schema_version": 1, "entries": [{"id": "img_pre", "local_path": "data/raw/img_pre",
                                           "status": "ok"}]}), encoding="utf-8")
    pts = np.column_stack([rng.uniform(1040, 1140, 20), rng.uniform(2060, 2160, 20)])
    items = pd.DataFrame({"item_id": [f"rp{i:04d}" for i in range(20)], "x": pts[:, 0],
                          "y": pts[:, 1]})
    before = {p for p in root.rglob("*") if p.is_file()}
    out = core.chip_dir(project, "recall_points")
    manifest = render_task(project, "recall_points", items, CFG, out)
    new = {p for p in root.rglob("*") if p.is_file()} - before
    interim = (root / "data" / "interim").resolve()
    assert len(manifest) == 20 and all(interim in p.resolve().parents for p in new)
    img = np.asarray(Image.open(out / manifest["rp0000"]["file"]))
    assert img.shape == (420, 420, 3)
    assert img[210 - 30:210 - 8, 209:212].min() < 30         # crosshair arm drawn
    with pytest.raises(AuditError):
        render_task(project, "recall_points", items, CFG, root / "artifacts" / "x")


# --- frames -----------------------------------------------------------------------------------


def test_dedup_merges_only_across_sources():
    pts = shapely.points([[0, 0], [2, 0], [1, 0], [10, 0]])
    source = np.array([0, 0, 1, 1])
    kept, owner = frames.dedup_cross_source(pts, source, 3.0)
    assert kept.tolist() == [True, True, False, True]          # two Google units both kept
    assert owner.tolist() == [0, 1, 0, 3]


def test_shift_points_clusters_and_bands():
    index = pd.DataFrame({"tile_id": ["t0"], "bounds": [np.array([0, 0, 320, 320])],
                          "affine": [np.array([0.5, 0, 0, 0, -0.5, 320])]})
    shifts = pd.DataFrame({"dx": [4.0], "dy": [2.0], "reg_ok": [True]}, index=["t0"])
    moved = frames.apply_shifts(np.array([shapely.box(10, 10, 20, 20)]), index, shifts)
    assert moved[0].bounds == pytest.approx((12, 9, 22, 19))  # +2 m east, -1 m north
    l_shape = shapely.Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])
    rep = frames.representative_points(np.array([l_shape]))
    assert l_shape.contains(rep[0])
    labels = frames.cluster_labels(4, np.array([0, 2]), np.array([1, 3]))
    assert labels[0] == labels[1] != labels[2] == labels[3]
    assert frames.band(np.array([50, 150, 900.0]), [0, 100, 300, np.inf], "c").tolist() == \
        ["c0", "c1", "c2"]
    area = shapely.box(0, 0, 100, 100).difference(shapely.box(0, 0, 50, 100))
    out = frames.outside_points(area, 25, np.random.default_rng(0))
    assert len(out) == 25 and all(area.contains(p) for p in out)


# --- estimators -------------------------------------------------------------------------------


def test_unsure_is_excluded_not_coded_as_no_and_kappa():
    items = pd.DataFrame({"item_id": list("abcd"), "stratum": "A", "frame_n": 100})
    answers = {"a": "yes", "b": "unsure", "c": "no", "d": "yes"}
    frame = estimators.answered(items, answers)
    assert len(frame) == 3 and frame["w"].iloc[0] == pytest.approx(100 / 3)
    table = estimators.stratum_table(items, answers)
    assert table.loc[0, "unsure"] == 1 and table.loc[0, "p_yes"] == pytest.approx(2 / 3)
    k = estimators.cohen_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "yes"])
    assert k["agreement"] == pytest.approx(0.75) and k["kappa"] == pytest.approx(0.5)


def test_take_all_strata_add_no_bootstrap_variance():
    frame = pd.DataFrame({"stratum": ["A"] * 5, "frame_n": 5, "w": 1.0,
                          "y": [1.0, 0, 1, 1, 0]})
    est = estimators.proportion(frame, n_boot=200)
    assert est["ci_lo"] == pytest.approx(0.6) and est["ci_hi"] == pytest.approx(0.6)


def test_simulation_unbiased_and_covers():
    """Synthetic population with known truth: stratified HT proportion and its Rao-Wu
    bootstrap 95 % interval over 1,000 replicates."""
    rng = np.random.default_rng(1)
    big_n = {"A": 6000, "B": 2500, "C": 800, "D": 300}
    p = {"A": 0.88, "B": 0.7, "C": 0.45, "D": 0.25}
    n = {"A": 90, "B": 80, "C": 70, "D": 60}
    pop = {k: (rng.random(big_n[k]) < p[k]).astype(float) for k in big_n}
    truth = sum(pop[k].sum() for k in pop) / sum(big_n.values())
    est, cover = [], 0
    for r in range(1000):
        parts = []
        for k in sorted(big_n):
            y = rng.choice(pop[k], size=n[k], replace=False)
            parts.append(pd.DataFrame({"stratum": k, "frame_n": big_n[k],
                                       "w": big_n[k] / n[k], "y": y}))
        frame = pd.concat(parts, ignore_index=True)
        e = estimators.proportion(frame, n_boot=400, seed=r)
        est.append(e["estimate"])
        cover += e["ci_lo"] <= truth <= e["ci_hi"]
    est = np.asarray(est)
    assert abs(est.mean() - truth) < 3 * est.std(ddof=1) / np.sqrt(len(est))
    assert 0.93 <= cover / 1000 <= 0.97


def test_estimate_refuses_without_frozen_prereg(audit_project):
    from nakkhu.audit.workflow import estimate
    from nakkhu.evaluate.guard import GuardError

    with pytest.raises(GuardError):
        estimate(audit_project, "recall_points", n_boot=10)


# --- sittings 2 and 3 on synthetic inputs ------------------------------------------------------


def _geojson(path: Path, geom: object) -> None:
    import geopandas as gpd

    gpd.GeoDataFrame(geometry=[geom], crs="EPSG:32645").to_file(path, driver="GeoJSON")


def _raster(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=1400, height=1400, count=3,
                       dtype="uint8", crs="EPSG:32645",
                       transform=from_origin(900, 2300, 0.30517578125, 0.30517578125)) as ds:
        ds.write(rng.integers(30, 200, (3, 1400, 1400), dtype=np.uint8))


@pytest.fixture
def pending_project(root):
    import geopandas as gpd

    from nakkhu.config import load_project

    (root / "configs" / "aoi").mkdir()
    _geojson(root / "configs" / "aoi" / "extent.geojson", shapely.box(950, 1950, 1250, 2250))
    _geojson(root / "configs" / "aoi" / "river.geojson",
             shapely.LineString([(950, 2000), (1250, 2000)]))
    text = (root / "configs" / "project.yaml").read_text(encoding="utf-8")
    (root / "configs" / "project.yaml").write_text(
        text + "aoi:\n  analysis_extent: configs/aoi/extent.geojson\n", encoding="utf-8")
    cfg = json.loads(json.dumps(CFG))
    cfg["region"] = "analysis_extent"
    cfg["channel_lines"] = ["configs/aoi/river.geojson"]
    cfg["scenes"] = {"pre": "img_pre", "post": ["img_post"], "oam": []}
    rng = np.random.default_rng(5)
    centres = np.column_stack([rng.uniform(980, 1220, 60), rng.uniform(1980, 2220, 60)])
    runs = []
    for k in range(3):
        run_dir = root / "runs" / "infer" / f"r{k}"
        run_dir.mkdir(parents=True)
        boxes = [shapely.box(x - 4 + 0.3 * k, y - 3, x + 4 + 0.3 * k, y + 3)
                 for x, y in centres[k * 10: k * 10 + 40]]
        gpd.GeoDataFrame({"conf": np.linspace(0.2, 0.9, len(boxes))}, geometry=boxes,
                         crs="EPSG:32645").to_file(run_dir / "detections.gpkg", layer="obb")
        runs.append({"run": f"infer/r{k}", "model": "m" if k < 2 else "n", "threshold": 0.3})
    cfg["tasks"]["precision_pool"].update(runs=runs, cluster_iou=0.5,
                                          run_share_bands=[0.0, 0.34, 0.67, 1.0],
                                          channel_bands_m=[0, 100, float("inf")])
    cmap = gpd.GeoDataFrame({"building_id": np.arange(60),
                             "class": np.where(np.arange(60) % 3 == 0, "positive", "negative"),
                             "observable": np.arange(60) % 10 != 9},
                            geometry=[shapely.box(x - 4, y - 3, x + 4, y + 3) for x, y in centres],
                            crs="EPSG:32645")
    (root / "artifacts" / "flood").mkdir(parents=True)
    cmap.to_parquet(root / "artifacts" / "flood" / "change_map_buildings.parquet")
    cfg["tasks"]["flood_trace"].update(change_map="artifacts/flood/change_map_buildings.parquet",
                                       near_channel_m=100)
    (root / "configs" / "audit").mkdir()
    (root / "configs" / "audit" / "audit.yaml").write_text(yaml.safe_dump(cfg),
                                                           encoding="utf-8")
    for sid, seed in (("img_pre", 1), ("img_post", 2)):
        _raster(root / "data" / "raw" / sid / f"{sid}.tif", seed)
    (root / "configs" / "data" / "manifest.yaml").write_text(yaml.safe_dump(
        {"schema_version": 1, "entries": [
            {"id": s, "local_path": f"data/raw/{s}", "status": "ok"}
            for s in ("img_pre", "img_post")]}), encoding="utf-8")
    return load_project(root)


def test_precision_pool_and_flood_trace_end_to_end(pending_project):
    from nakkhu.audit.chips import render_task
    from nakkhu.audit.server import Session
    from nakkhu.audit.workflow import build_items

    project = pending_project
    cfg = core.load_audit_config(project)
    items, frame, extra = build_items(project, cfg, "precision_pool")
    assert set(frame["stratum"]) and frame["stratum"].str.match(r"^share\d_chan\d$").all()
    assert (frame["run_share"] > 0).all() and (frame["run_share"] <= 1).all()
    assert set(extra["members"]["run"]) == {"infer/r0", "infer/r1", "infer/r2"}
    assert not items["pilot"].any()
    again, _, _ = build_items(project, cfg, "precision_pool")
    pd.testing.assert_frame_equal(items, again)
    pool = estimators.precision_estimates(
        items, dict.fromkeys(items["item_id"], "yes"), extra["members"], n_boot=50)
    assert pool["models"]["m"]["runs"] == 2
    assert all(r["estimate"] == pytest.approx(1.0) for r in pool["runs"].values())

    ft, ft_frame, _ = build_items(project, cfg, "flood_trace")
    assert set(ft_frame["stratum"]) <= {"positive", "negative_near", "negative_far"}
    assert len(ft_frame) == 54                       # unobservable buildings dropped
    path = core.items_path(project, "flood_trace")
    path.parent.mkdir(parents=True)
    ft.to_parquet(path, index=False)
    manifest = render_task(project, "flood_trace", ft, cfg, core.chip_dir(project,
                                                                          "flood_trace"))
    first = Image.open(core.chip_dir(project, "flood_trace") / manifest[ft["item_id"][0]]["file"])
    assert first.size[0] > 2 * 420 and manifest[ft["item_id"][0]]["post"] == "img_post"

    with pytest.raises(AuditError):                   # no pilot, no allocation yet
        Session(project, "flood_trace", cfg)
    core.allocation_path(project).parent.mkdir(parents=True, exist_ok=True)
    core.allocation_path(project).write_text(yaml.safe_dump(
        {"pilot": {"median_s": 2.0}, "tasks": {"flood_trace": {"n": 20}}}), encoding="utf-8")
    s = Session(project, "flood_trace", cfg)
    st = s.state()
    assert st["qid"] == "main" and st["keys"]["unsure"] == "Not observable"
    st = s.answer(st["pos"], "main", "yes", 2000)
    assert st["qid"] == "followup" and st["pos"] == 0 and st["question"] == "Damage?"
    st = s.answer(0, "followup", "no", 900)
    assert st["qid"] == "main" and st["pos"] == 1


def test_greenhouse_footprints_dropped_and_overture_refused(project):
    import geopandas as gpd

    gh = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 20, 20)], crs="EPSG:32645")
    polys = {"google": gpd.GeoDataFrame(geometry=[shapely.box(2, 2, 8, 8),
                                                  shapely.box(50, 50, 60, 60)],
                                        crs="EPSG:32645")}
    kept, dropped = frames.drop_greenhouses(polys, gh, 0.3)
    assert dropped == {"google": 1} and len(kept["google"]) == 1
    area = frames.outside_area(shapely.box(0, 0, 100, 100), np.array([shapely.box(50, 50, 60, 60)]),
                               5.0, np.array([shapely.box(0, 0, 20, 20)]))
    assert not area.intersects(shapely.Point(10, 10)) and area.contains(shapely.Point(30, 80))
    cfg = json.loads(json.dumps(CFG))
    cfg["tasks"]["recall_points"]["sources"] = ["google", "overture"]
    with pytest.raises(AuditError, match="Overture"):
        frames.recall_frame(project, cfg)


def test_examples_come_from_the_caption_manifest(audit_project):
    directory = core.examples_dir(audit_project, "recall_points")
    directory.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(directory / "no_1.jpg")
    (directory / "examples.json").write_text(json.dumps(
        [{"file": "no_1.jpg", "caption": "No: greenhouse"},
         {"file": "../x.jpg", "caption": "bad"}]), encoding="utf-8")
    state = _session(audit_project).state()
    assert state["examples"] == [{"src": "/example/no_1.jpg", "answer": "No: greenhouse"}]
    assert any("greenhouse" in line and "= No" in line for line in state["help"])


def test_rebalance_keeps_total_and_caps():
    sizes = {"positive": 40, "negative_near": 118, "negative_far": 142}
    frame_n = {"positive": 900, "negative_near": 5000, "negative_far": 9000}
    out = sample.rebalance(sizes, frame_n, "positive", 100, "negative_far")
    assert out == {"positive": 100, "negative_near": 118, "negative_far": 82}
    small = sample.rebalance(sizes, {**frame_n, "positive": 55}, "positive", 100, "negative_far")
    assert small["positive"] == 55 and sum(small.values()) == 300
    assert sample.rebalance(sizes, frame_n, "missing", 100, "negative_far") == sizes


# --- sitting 1b: near-miss follow-up --------------------------------------------------------


def _parent_answers(items: pd.DataFrame) -> dict[str, str]:
    cycle = ["yes", "no", "unsure"]
    return {i: cycle[k % 3] for k, i in enumerate(sorted(items["item_id"]))}


def _with_outside(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items.loc[items["stratum"] == "D", "stratum"] = "outside"
    return items


def test_nearmiss_items_select_no_unsure_outside_excluded_and_overlap():
    parent = _with_outside(_items())
    answers = _parent_answers(parent)
    tcfg = CFG["tasks"]["recall_nearmiss"]
    out = sample.nearmiss_items(parent, answers, tcfg, 0)
    near = out.loc[out["kind"] == "nearmiss"]
    outside = set(parent.loc[parent["stratum"] == "outside", "item_id"])
    want = {i for i, a in answers.items() if a in ("no", "unsure")} - outside
    assert set(near["parent_item"]) == want
    assert (near["question"] == "recall_nearmiss").all() and (near["radius_m"] == 8.0).all()
    over = out.loc[out["kind"] == "overlap"]
    assert len(over) == 4 and (over["question"] == "recall_points").all()
    assert over["radius_m"].isna().all()
    assert out["item_id"].str.match(r"^nm\d{4}$").all() and out["item_id"].is_unique
    pd.testing.assert_frame_equal(out, sample.nearmiss_items(parent, answers, tcfg, 0))
    queue = build_queue(out, None, "recall_nearmiss", CFG)
    assert len(queue) == len(out) and not any(s.repeat for s in queue)
    kinds = out.set_index("item_id").loc[[s.item_id for s in queue], "kind"].tolist()
    assert kinds != sorted(kinds)                              # interleaved, not blocked
    with pytest.raises(AuditError, match="no answer"):
        sample.nearmiss_items(parent, dict(list(answers.items())[:-1]), tcfg, 0)
    with pytest.raises(AuditError, match="quote"):           # YAML `no` parses as False
        sample.nearmiss_items(parent, answers, {**tcfg, "include_answers": [False, "unsure"]}, 0)


@pytest.fixture
def nearmiss_project(audit_project):
    parent = core.load_items(audit_project, "recall_points")
    out = sample.nearmiss_items(parent, _parent_answers(parent),
                                CFG["tasks"]["recall_nearmiss"], 0)
    path = core.items_path(audit_project, "recall_nearmiss")
    path.parent.mkdir(parents=True)
    out.to_parquet(path, index=False)
    chips = core.chip_dir(audit_project, "recall_nearmiss")
    chips.mkdir(parents=True)
    for item in out["item_id"]:
        Image.new("RGB", (420, 420)).save(chips / core.chip_name("recall_nearmiss", item))
    return audit_project


def test_nearmiss_session_click_offsets_auditor_and_blindness(nearmiss_project):
    from nakkhu.audit.server import Session

    project = nearmiss_project
    items = core.load_items(project, "recall_nearmiss").set_index("item_id")
    s = Session(project, "recall_nearmiss", core.load_audit_config(project))
    seen = {"recall_nearmiss": 0, "recall_points": 0}
    st = s.state()
    while st["phase"] == "run":
        body = json.dumps(st)
        for leak in ["parent_item", "nearmiss", "overlap", "stratum", *items["parent_item"]]:
            assert f'"{leak}"' not in body
        if st["click_on_yes"]:
            assert st["question"] == "Roof in circle?"
            seen["recall_nearmiss"] += 1
            refused = s.answer(st["pos"], "main", "yes", 1000)          # Yes needs a click
            assert refused.get("click_refused") and refused["pos"] == st["pos"]
            far = s.answer(st["pos"], "main", "yes", 1000, {"x_px": 210 + 60, "y_px": 210})
            assert far.get("click_refused")                         # 60 px = 9.2 m > 8 m
            st = s.answer(st["pos"], "main", "yes", 1500, {"x_px": 210 + 20, "y_px": 210 - 10})
        else:
            assert st["question"] == "Roof?"
            seen["recall_points"] += 1
            st = s.answer(st["pos"], "main", "no", 900, {"x_px": 1, "y_px": 1})
    assert st["phase"] == "done" and seen["recall_points"] == 4 and seen["recall_nearmiss"] > 0
    events = core.read_events(core.responses_path(project, "recall_nearmiss"))
    answers = [e for e in events if e["event"] == "answer"]
    assert all(e["auditor"] == "author" for e in events)
    clicked = [e for e in answers if e["answer"] == "yes"]
    assert len(clicked) == seen["recall_nearmiss"]
    mpp = 0.30517578125 / 2
    for e in clicked:
        assert e["click_dx_m"] == pytest.approx(20 * mpp, abs=1e-3)
        assert e["click_dy_m"] == pytest.approx(10 * mpp, abs=1e-3)
        assert e["click_m"] == pytest.approx(np.hypot(20, 10) * mpp, abs=1e-3)
    assert not any("click_m" in e for e in answers if e["question"] == "recall_points")
    assert core.auditors_of(events) == ["author"]


def test_author_time_cap_counts_the_author_only(nearmiss_project):
    project = nearmiss_project
    rp = core.responses_path(project, "recall_points")
    core.append_event(rp, {"event": "answer", "slot": "a", "question": "recall_points",
                           "answer": "yes", "ms": 60_000, "sitting": "s1"})
    core.append_event(rp, {"event": "auditor", "sitting": "s1", "auditor": "teammate_1"})
    nm = core.responses_path(project, "recall_nearmiss")
    core.append_event(nm, {"event": "answer", "slot": "b", "question": "recall_nearmiss",
                           "answer": "no", "ms": 2_000, "sitting": "s2", "auditor": "author"})
    assert core.cumulative_ms(project) == 2_000
    assert core.cumulative_ms(project, "teammate_1") == 60_000
    events = core.read_events(rp)
    assert core.auditors_of(events) == ["teammate_1"]
    assert core.auditors_of(events[:1], "teammate_1") == ["teammate_1"]     # config fallback


def test_combine_recall_protocol_v2_and_unsure_bounds():
    s1 = {"a": "yes", "b": "no", "c": "unsure", "d": "no", "e": "unsure", "f": "no", "o": "no"}
    near = {"b": {"answer": "yes", "click_m": 3.0}, "c": {"answer": "yes", "click_m": 7.9},
            "d": {"answer": "no"}, "e": {"answer": "unsure"},
            "f": {"answer": "yes", "click_m": 8.5}}
    v2 = estimators.combine_recall(s1, near, 8.0)
    assert v2 == {"a": "yes", "b": "yes", "c": "yes", "d": "no", "e": "unsure", "f": "no",
                  "o": "no"}
    assert estimators.combine_recall(s1, near, 4.0)["c"] == "no"
    assert estimators.unsure_as(v2, "yes")["e"] == "yes"
    assert estimators.unsure_as(v2, "no")["e"] == "no"
    items = pd.DataFrame({"item_id": list(s1), "stratum": ["G"] * 6 + ["outside"],
                          "frame_n": [600] * 6 + [0], "in_google": True})
    res = estimators.recall_with_bounds(items, v2, ["google"], None, 50, 0)
    assert res["answers"] == {"yes": 3, "no": 3, "unsure": 1}
    share = {k: res[k]["roof_share_of_union_points"]["estimate"]
             for k in ("unsure_excluded", "unsure_as_yes", "unsure_as_no")}
    assert share == pytest.approx({"unsure_excluded": 3 / 5, "unsure_as_yes": 4 / 6,
                                   "unsure_as_no": 3 / 6})
    assert res["unsure_excluded"]["roof_units_total"]["estimate"] == pytest.approx(360.0)
    agree = estimators.agreement(["yes", "no", "unsure", "yes"], ["yes", "no", "no", "no"])
    assert agree["all"]["n"] == 4 and agree["all"]["agreement"] == pytest.approx(0.5)
    assert agree["yes_no"]["n"] == 3


def test_nearmiss_chip_draws_circle_at_radius(project):
    from nakkhu.audit.chips import render_task

    root = project.paths.root
    raw = root / "data" / "raw" / "img_pre"
    raw.mkdir(parents=True)
    with rasterio.open(raw / "img_pre.tif", "w", driver="GTiff", width=600, height=600, count=3,
                       dtype="uint8", crs="EPSG:32645",
                       transform=from_origin(1000, 2200, 0.30517578125, 0.30517578125)) as ds:
        ds.write(np.full((3, 600, 600), 120, dtype=np.uint8))
    (root / "configs" / "data" / "manifest.yaml").write_text(yaml.safe_dump(
        {"schema_version": 1, "entries": [{"id": "img_pre", "local_path": "data/raw/img_pre",
                                           "status": "ok"}]}), encoding="utf-8")
    items = pd.DataFrame({"item_id": ["nm0001", "nm0002"], "kind": ["nearmiss", "overlap"],
                          "x": [1090.0, 1090.0], "y": [2110.0, 2110.0],
                          "radius_m": [8.0, np.nan]})
    out = core.chip_dir(project, "recall_nearmiss")
    manifest = render_task(project, "recall_nearmiss", items, CFG, out)
    near = np.asarray(Image.open(out / manifest["nm0001"]["file"])).astype(int)
    over = np.asarray(Image.open(out / manifest["nm0002"]["file"])).astype(int)
    r_px = int(round(8.0 / (0.30517578125 / 2)))              # 52 px at 2x
    assert np.abs(near[210, 210 + r_px - 2: 210 + r_px + 3] - 120).max() > 60   # ring at R
    assert np.abs(over[210, 210 + r_px] - 120).max() < 20     # overlap chip: no circle
    assert np.abs(over[210 - 30:210 - 8, 210] - 120).max() > 60   # but the crosshair arm
    assert np.abs(near[210 - 30:210 - 14, 210] - 120).max() < 20  # small mark only
    assert manifest["nm0001"]["radius_m"] == 8.0


def test_recall_protocols_end_to_end_carry_auditors_and_agreement(nearmiss_project):
    from nakkhu.audit.server import Session
    from nakkhu.audit.workflow import _recall_protocols, require_complete

    project = nearmiss_project
    cfg = core.load_audit_config(project)
    parent = core.load_items(project, "recall_points")
    s1 = _parent_answers(parent)
    rp = core.responses_path(project, "recall_points")
    for item, answer in s1.items():           # sitting 1 as logged before the auditor field
        core.append_event(rp, {"event": "answer", "slot": item, "item_id": item,
                               "question": "recall_points", "answer": answer, "ms": 1000,
                               "repeat": False, "sitting": "s1"})
    core.append_event(rp, {"event": "auditor", "sitting": "s1", "auditor": "teammate_1"})
    with pytest.raises(AuditError, match="not complete"):
        require_complete(project, cfg, "recall_nearmiss")
    s = Session(project, "recall_nearmiss", cfg)
    st = s.state()
    while st["phase"] == "run":
        if st["click_on_yes"]:
            st = s.answer(st["pos"], "main", "yes", 1500, {"x_px": 230, "y_px": 210})
        else:
            st = s.answer(st["pos"], "main", "yes", 900)
    require_complete(project, cfg, "recall_nearmiss")
    out = _recall_protocols(project, cfg, parent, s1, 20, 0)
    assert out["auditors"] == {"recall_points": ["teammate_1"], "recall_nearmiss": ["author"]}
    assert out["inter_rater_overlap"]["raters"] == ["teammate_1", "author"]
    assert out["inter_rater_overlap"]["all"]["n"] == 4
    near_n = sum(a in ("no", "unsure") for a in s1.values())
    assert out["nearmiss"]["items"] == near_n and out["nearmiss"]["answers"]["yes"] == near_n
    v1 = out["protocol_v1_sitting1_only"]["answers"]
    v2 = out["protocol_v2_primary"]["answers"]
    assert v1["yes"] + near_n == v2["yes"] == len(s1)            # every near-miss Yes in R
    assert out["protocol_v2_sensitivity_radius"]["roof_share_of_union_points"]["estimate"] == 1.0
    strata = out["protocol_v2_primary"]["unsure_excluded"]["strata"]
    assert sum(r["yes"] for r in strata) == v2["yes"]
    assert all(r["wilson_lo"] <= r["p_yes"] <= r["wilson_hi"] for r in strata if r["yes"] + r["no"])
    # recall_v2 (reused by G1 confirmatory) codes the answers exactly like the estimate.
    from nakkhu.audit.workflow import recall_v2

    items, v2_answers, clicked, meta = recall_v2(project, cfg)
    assert len(items) == len(parent) and meta["radius_m"] == 8.0
    assert {a: sum(x == a for x in v2_answers.values()) for a in ("yes", "no", "unsure")} == v2
    assert set(clicked) == {i for i, a in s1.items() if a != "yes"}
    xy = parent.set_index("item_id")
    some = next(iter(clicked))
    assert clicked[some][0] != xy.loc[some, "x"] or clicked[some][1] != xy.loc[some, "y"]
    bad = json.loads(json.dumps(cfg))
    bad["tasks"]["recall_points"]["auditor"] = "author"
    with pytest.raises(AuditError, match="audit.yaml says"):
        _recall_protocols(project, bad, parent, s1, 20, 0)


@pytest.mark.parametrize("allocation", [None, {"pilot": {"median_s": 2.0},
                                               "tasks": {"recall_points": {"n": 20}}}])
def test_nearmiss_serves_every_item_without_an_allocation_entry(nearmiss_project, allocation):
    """Sitting 1b is sized by design: with no allocation file, or one without a
    recall_nearmiss entry, the page serves all items in the pre-registered order."""
    from nakkhu.audit.server import Session, make_server

    project = nearmiss_project
    if allocation is not None:
        core.allocation_path(project).parent.mkdir(parents=True, exist_ok=True)
        core.allocation_path(project).write_text(yaml.safe_dump(allocation), encoding="utf-8")
    items = core.load_items(project, "recall_nearmiss")
    first = items.sort_values("order_key")["item_id"].iloc[0]
    server = make_server(Session(project, "recall_nearmiss", core.load_audit_config(project)), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/state"
        state = json.loads(urllib.request.urlopen(url).read())
    finally:
        server.shutdown()
        server.server_close()
    assert state["phase"] == "run" and state["done"] == 0 and state["total"] == len(items)
    assert state["pos"] == 0 and state["task"] == "recall_nearmiss"
    assert state["chip"] == f"/chip/{core.chip_name('recall_nearmiss', first)}"
    assert not core.responses_path(project, "recall_nearmiss").exists()


def test_second_server_on_a_busy_port_is_refused(audit_project):
    """A stale sitting's server must not keep answering the browser on a shared port."""
    from nakkhu.audit.server import make_server

    first = make_server(_session(audit_project), 0)
    try:
        with pytest.raises(OSError, match="already in use"):
            make_server(_session(audit_project), first.server_address[1])
    finally:
        first.server_close()


# --- plan 013: Microsoft polygon precision (sitting 2) ---------------------------------------


def test_new_task_is_appended_so_existing_streams_do_not_move():
    assert core.TASKS[:3] == ("recall_points", "precision_pool", "flood_trace")
    assert core.TASKS.index("precision_microsoft") == 3
    assert core.ALLOCATED_TASKS == core.TASKS[:3]
    assert sample.PREFIX["precision_microsoft"] == "pm"
    from nakkhu.audit.server import HELP
    from nakkhu.audit_cli import TASK_CHOICES

    assert "precision_microsoft" in TASK_CHOICES and "precision_microsoft" in core.SERVE_TASKS
    assert any("shifted a few metres" in line for line in HELP["precision_microsoft"])


def test_task_absent_from_allocation_keeps_design_size_and_gets_repeats():
    frame = _frame({"agree_chan0": 300, "ms_only_chan0": 80, "agree_chan2": 900})
    design = sample.design_sizes(frame.groupby("stratum").size().to_dict(), 150, 10)
    items = sample.finish_items(sample.draw(frame, design, 0, "precision_microsoft"),
                                "precision_microsoft", 0, 0)
    assert len(items) == 150 and items["item_id"].str.match(r"^pm\d{4}$").all()
    alloc = {"pilot": {"median_s": 3.0}, "tasks": {"precision_pool": {"n": 100}}}
    cfg = json.loads(json.dumps(CFG))
    cfg["time"]["repeat_frac"] = 0.05
    assert len(core.active_items(items, alloc, "precision_microsoft")) == 150
    queue = build_queue(items, alloc, "precision_microsoft", cfg)
    assert sum(not s.repeat for s in queue) == 150 and sum(s.repeat for s in queue) == 8
    with pytest.raises(AuditError):               # no allocation and no pilot: refused as before
        build_queue(items, None, "precision_microsoft", cfg)


def test_polygon_strata_agree_and_channel_bands():
    polys = np.array([shapely.box(0, 0, 10, 10), shapely.box(0, 150, 10, 160),
                      shapely.box(0, 500, 10, 510)], dtype=object)
    boxes = np.array([shapely.box(2, 2, 6, 6), shapely.box(8, 158, 14, 164)], dtype=object)
    agree, dist, stratum = frames.polygon_strata(
        polys, boxes, shapely.LineString([(-50, -5), (50, -5)]), [0, 100, 300, float("inf")])
    # box 2's centroid (11, 161) lies outside polygon 2
    assert agree.tolist() == [True, False, False]
    assert stratum.tolist() == ["agree_chan0", "ms_only_chan1", "ms_only_chan2"]
    assert dist[0] == pytest.approx(10.0)


def test_polygon_precision_estimates_strata_and_unsure_bounds():
    frame = _frame({"agree_chan0": 40, "ms_only_chan0": 40})
    items = sample.finish_items(sample.draw(frame, {"agree_chan0": 10, "ms_only_chan0": 10}, 0,
                                            "precision_microsoft"), "precision_microsoft", 0, 0)
    ans = {i: ("yes" if s == "agree_chan0" else "no")
           for i, s in zip(items["item_id"], items["stratum"], strict=True)}
    first = items.loc[items["stratum"] == "ms_only_chan0", "item_id"].iloc[0]
    ans[first] = "unsure"
    out = estimators.polygon_precision_estimates(items, ans, n_boot=50)
    assert out["by_stratum"]["agree_chan0"]["estimate"] == pytest.approx(1.0)
    assert out["by_stratum"]["ms_only_chan0"]["estimate"] == pytest.approx(0.0)
    assert out["overall"]["estimate"] == pytest.approx(0.5)       # equal N_h
    assert out["unsure"] == 1
    assert out["unsure_as_no"]["estimate"] <= out["overall"]["estimate"] \
        <= out["unsure_as_yes"]["estimate"]


def test_microsoft_frame_items_and_outline_chips(pending_project, monkeypatch):
    """The frame takes the sweep's Microsoft exposure units (ids kept), moves each by the
    footprint shift, and strata come from detector_final box centroids inside them."""
    import geopandas as gpd

    from nakkhu.audit.chips import render_task
    from nakkhu.audit.workflow import build_items
    from nakkhu.hydro import sweep

    project = pending_project
    rng = np.random.default_rng(3)
    centres = np.column_stack([rng.uniform(990, 1210, 80), rng.uniform(1990, 2210, 80)])
    polys = gpd.GeoDataFrame(
        {"building_uid": [f"microsoft:{i}" for i in range(80)]},
        geometry=[shapely.box(x - 5, y - 4, x + 5, y + 4) for x, y in centres], crs="EPSG:32645")
    boxes = gpd.GeoDataFrame({"building_uid": [f"detector_final:{i}" for i in range(30)]},
                             geometry=[shapely.box(x - 2, y - 2, x + 2, y + 2)
                                       for x, y in centres[:30]], crs="EPSG:32645")

    class Settings:
        v = {"analysis_extent": "configs/aoi/extent.geojson", "inventories": {
            "detector_final": {"path": "artifacts/none.parquet"}}}

        def read_polygon(self, _):
            return shapely.box(950, 1950, 1250, 2250)

        def resolve(self, p):
            return project.paths.resolve(p)

    monkeypatch.setattr(sweep, "load_settings", lambda _p: (Settings(), None))
    monkeypatch.setattr(sweep, "load_greenhouses", lambda _s: None)
    monkeypatch.setattr(sweep, "load_inventory", lambda _s, name, _e, _g: (
        polys.copy() if name == "microsoft" else boxes.copy(), {"n_final": 80}))
    index = pd.DataFrame({"tile_id": ["t0"], "bounds": [[900.0, 1900.0, 1300.0, 2300.0]],
                          "affine": [[0.5, 0.0, 900.0, 0.0, -0.5, 2300.0]]})
    shifts = pd.DataFrame({"dx": [4.0], "dy": [-2.0], "reg_ok": [True]},
                          index=pd.Index(["t0"], name="tile_id"))
    monkeypatch.setattr(frames, "tile_index", lambda _p: index)
    monkeypatch.setattr(frames, "load_shifts", lambda _p, _c: {"footprints": shifts,
                                                               "osm": shifts})
    cfg = core.load_audit_config(project)
    cfg["tasks"]["precision_microsoft"] = {
        "question": "Outline?", "keys": {"yes": "Yes", "no": "No", "unsure": "Unsure"},
        "auditor": "author", "inventory": "microsoft", "agree_with": "detector_final",
        "channel_bands_m": [0, 100, float("inf")], "design_n": 40, "min_per_stratum": 5}
    items, frame, extra = build_items(project, cfg, "precision_microsoft")
    assert len(frame) == 80 and extra["stats"]["agree"] >= 30   # overlapping outlines
    assert frame["stratum"].str.match(r"^(agree|ms_only)_chan\d$").all()
    assert len(items) == 40 and items["item_id"].str.match(r"^pm\d{4}$").all()
    row = items.iloc[0]
    uid = int(row["building_uid"].split(":")[1])
    # shown moved by (dx * 0.5, dy * -0.5) = (+2, +1) m; the HAND point is the unmoved centroid
    assert row["x"] - row["hand_x"] == pytest.approx(2.0)
    assert row["y"] - row["hand_y"] == pytest.approx(1.0)
    assert row["hand_x"] == pytest.approx(centres[uid, 0])
    again, _, _ = build_items(project, cfg, "precision_microsoft")
    pd.testing.assert_frame_equal(items, again)
    manifest = render_task(project, "precision_microsoft", items, cfg,
                           core.chip_dir(project, "precision_microsoft"))
    assert len(manifest) == 40 and all(isinstance(v["valid"], float) for v in manifest.values())


# --- redo of a whole sitting (serve --redo) --------------------------------------------------


def _redo_project(project, task: str = "precision_pool"):
    root = project.paths.root
    cfg = json.loads(json.dumps(CFG))
    cfg["tasks"][task]["auditor"] = "author"
    (root / "configs" / "audit").mkdir(parents=True)
    (root / "configs" / "audit" / "audit.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    items = _items(task, pilot=0)
    path = core.items_path(project, task)
    path.parent.mkdir(parents=True)
    items.to_parquet(path, index=False)
    chips = core.chip_dir(project, task)
    chips.mkdir(parents=True)
    for item in items["item_id"]:
        Image.new("RGB", (8, 8)).save(chips / core.chip_name(task, item))
    core.allocation_path(project).write_text(yaml.safe_dump(
        {"pilot": {"median_s": 2.0}, "tasks": {task: {"n": 20}}}), encoding="utf-8")
    return project


def _serve_session(project, task: str = "precision_pool"):
    from nakkhu.audit.server import Session

    return Session(project, task, core.load_audit_config(project))


def _first_pass(project, answers, task: str = "precision_pool"):
    s = _serve_session(project, task)
    st = s.state()
    for answer in answers:
        st = s.answer(st["pos"], st["qid"], answer, 1000)
    return s


def test_redo_archives_once_and_resumes_the_same_queue(project):
    project = _redo_project(project)
    first = _first_pass(project, ["yes", "no", "unsure"])
    log, archive = core.responses_path(project, "precision_pool"), \
        core.archive_path(project, "precision_pool")
    assert archive.name == "responses_pass1.jsonl" and archive.parent == log.parent
    pass1 = log.read_bytes()
    header = core.start_redo(project, "precision_pool", "author")
    assert archive.read_bytes() == pass1                     # moved unchanged
    assert header["redo_of"] == "pass1" and header["prior_answer_shown"] is True
    assert header["archive_sha256"] == core.sha256_file(archive)
    assert core.read_events(log) == [header]
    s = _serve_session(project)
    assert [q.slot for q in s.queue] == [q.slot for q in first.queue]   # same order
    st = s.state()
    assert st["done"] == 0 and st["pos"] == 0 and st["redo"] == {"of": "pass1"}
    assert st["chip"] == f"/chip/{core.chip_name('precision_pool', first.queue[0].item_id)}"
    assert st["prior"] == [{"label": "Previous answer", "value": "yes", "text": "Yes"}]
    st = s.answer(0, "main", "no", 900)
    assert st["prior"][0]["value"] == "no"
    events = core.read_events(log)
    start, answer = events[1], events[2]
    assert start["event"] == "start" and start["redo_of"] == "pass1"
    assert answer["redo_of"] == "pass1" and answer["prior_answer_shown"] is True
    assert answer["prior_answer"] == "yes" and answer["answer"] == "no"
    del s                                                    # "kill" mid-redo, then resume
    assert core.start_redo(project, "precision_pool", "author") == header
    assert archive.read_bytes() == pass1
    s2 = _serve_session(project)
    st = s2.state()
    assert st["done"] == 1 and st["pos"] == 1 and st["prior"][0]["value"] == "no"
    st = s2.answer(1, "main", "yes", 900)
    st = s2.answer(2, "main", "yes", 900)
    assert st["pos"] == 3 and st["prior"] == [{"label": "Previous answer", "value": None,
                                                "text": "none"}]
    answers = [e for e in core.read_events(log) if e["event"] == "answer"]
    assert len(answers) == 3 and all(e["prior_answer_shown"] for e in answers)
    assert core.cumulative_ms(project) == 3000 + 2700        # pass-1 time still counts


def test_redo_refuses_to_clobber_and_half_made_states(project):
    project = _redo_project(project)
    with pytest.raises(AuditError, match="nothing to redo"):
        core.start_redo(project, "precision_pool", "author")
    _first_pass(project, ["yes", "no"])
    log = core.responses_path(project, "precision_pool")
    archive = core.archive_path(project, "precision_pool")
    archive.write_text('{"event": "answer"}\n', encoding="utf-8")   # an archive already there
    before = archive.read_bytes()
    with pytest.raises(AuditError, match="refusing to overwrite"):
        core.start_redo(project, "precision_pool", "author")
    assert archive.read_bytes() == before and log.is_file()
    with pytest.raises(AuditError, match="--redo"):              # plain serve refuses too
        _serve_session(project)
    log.unlink()                                                 # crash between move and header
    header = core.start_redo(project, "precision_pool", "author")
    assert archive.read_bytes() == before and core.read_events(log) == [header]
    assert core.any_answers(project, "precision_pool")
    archive.unlink()
    with pytest.raises(AuditError, match="missing"):
        core.redo_state(project, "precision_pool")


def test_redo_prior_answer_in_page_payload_and_followup(project):
    from nakkhu.audit.server import make_server

    project = _redo_project(project, "flood_trace")
    _first_pass(project, ["yes", "no", "no"], "flood_trace")      # yes, damage no, then no
    core.start_redo(project, "flood_trace", "author")
    s = _serve_session(project, "flood_trace")
    server = make_server(s, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = urllib.request.urlopen(base + "/api/state").read().decode()
        page = urllib.request.urlopen(base + "/").read().decode()
    finally:
        server.shutdown()
        server.server_close()
    state = json.loads(body)
    assert state["prior"] == [
        {"label": "Previous answer", "value": "yes", "text": "Yes"},
        {"label": "Previous follow-up answer", "value": "no", "text": "No"}]
    items = core.load_items(project, "flood_trace")
    for leak in [*items["item_id"], "item_id", "slot", "repeat", "stratum"]:
        assert f'"{leak}"' not in body
    assert 'id="prior"' in page and "Previous" not in page.split("<script>")[0]
    st = s.answer(0, "main", "yes", 900)
    assert st["qid"] == "followup"
    assert st["prior"] == [{"label": "Previous answer", "value": "no", "text": "No"}]
    st = s.answer(0, "followup", "yes", 900)
    assert st["prior"] == [{"label": "Previous answer", "value": "no", "text": "No"}]
    last = core.read_events(core.responses_path(project, "flood_trace"))[-1]
    assert last["question"] == "damage" and last["prior_answer"] == "no"


def test_status_reports_redo_progress_and_archive(project):
    from nakkhu.audit.workflow import status

    project = _redo_project(project)
    _first_pass(project, ["yes", "no", "unsure"])
    plain = status(project)["tasks"]["precision_pool"]
    assert plain["answered"] == 3 and "redo" not in plain
    core.start_redo(project, "precision_pool", "author")
    s = _serve_session(project)
    s.answer(0, "main", "no", 900)
    task = status(project)["tasks"]["precision_pool"]
    assert task["answered"] == 1
    redo = task["redo"]
    assert redo["archive"] == "audit/precision_pool/responses_pass1.jsonl"
    assert redo["archive_answered"] == 3 and redo["redo_of"] == "pass1"
    assert redo["prior_answer_shown"] is True and redo["state"] == "in progress"
    assert not any(k in redo for k in ("yes", "no", "unsure", "answers"))


def test_estimate_uses_the_complete_redo_only(project, monkeypatch):
    from nakkhu.audit import workflow

    project = _redo_project(project)
    monkeypatch.setattr(workflow, "require_frozen",
                        lambda _p: {"prereg_sha256": "p", "split_sha256": "s"})
    monkeypatch.setattr(workflow, "record_test_access", lambda *a, **k: {"flagged": False})
    monkeypatch.setattr(estimators, "precision_estimates", lambda *a, **k: {})
    s = _first_pass(project, [])
    st = s.state()
    while st["phase"] == "run":
        st = s.answer(st["pos"], st["qid"], "yes", 500)
    core.start_redo(project, "precision_pool", "author")
    with pytest.raises(AuditError, match="not complete"):       # a redo is estimated whole
        workflow.estimate(project, "precision_pool", n_boot=10)
    s = _serve_session(project)
    st = s.state()
    while st["phase"] == "run":
        st = s.answer(st["pos"], st["qid"], "no", 500)
    pd.DataFrame({"run": ["r"], "cluster": [0], "model": ["m"]}).to_parquet(
        core.task_dir(project, "precision_pool") / "members.parquet")
    workflow.estimate(project, "precision_pool", n_boot=10)
    out = json.loads((project.paths.artifacts / "audit" / "precision_pool_estimates.json")
                     .read_text(encoding="utf-8"))
    redo = out["answers_pass"]["precision_pool"]
    assert redo["redo_of"] == "pass1" and redo["prior_answer_shown"] is True
    assert redo["archive_sha256"] == core.sha256_file(
        core.archive_path(project, "precision_pool"))
    assert sum(r["yes"] for r in out["strata"]) == 0             # pass-1 answers never used


# --- blind restart of a whole sitting (serve --restart) --------------------------------------

REASON = "pass 1 void: damage criterion misapplied; drone panel mislabelled pre-flood imagery"


def _no_archive_reads(monkeypatch):
    """A blind restart must never read the archived answers."""
    def boom(*_a, **_k):
        raise AssertionError("archived pass-1 answers were read")
    monkeypatch.setattr(core, "prior_answers", boom)


def test_restart_archives_once_and_resumes_blind(project, monkeypatch):
    project = _redo_project(project)
    first = _first_pass(project, ["yes", "no", "unsure"])
    log, archive = core.responses_path(project, "precision_pool"), \
        core.archive_path(project, "precision_pool")
    pass1 = log.read_bytes()
    header = core.start_restart(project, "precision_pool", "author", REASON)
    assert archive.read_bytes() == pass1                     # moved unchanged
    assert header["event"] == "restart" and header["restart_of"] == "pass1"
    assert header["prior_answer_shown"] is False and header["reason"] == REASON
    assert "redo_of" not in header and header["archive_sha256"] == core.sha256_file(archive)
    assert core.read_events(log) == [header]
    _no_archive_reads(monkeypatch)
    s = _serve_session(project)
    assert [q.slot for q in s.queue] == [q.slot for q in first.queue]   # same queue
    st = s.state()
    assert st["done"] == 0 and st["pos"] == 0 and st["redo"] is None and "prior" not in st
    st = s.answer(0, "main", "no", 900)
    assert "prior" not in st
    start, answer = core.read_events(log)[1:3]
    assert start["event"] == "start" and start["restart_of"] == "pass1"
    assert start["prior_answer_shown"] is False
    assert answer["restart_of"] == "pass1" and answer["prior_answer_shown"] is False
    assert "prior_answer" not in answer and "redo_of" not in answer
    del s                                                    # "kill" mid-restart, then resume
    assert core.start_restart(project, "precision_pool", "author", None) == header
    assert archive.read_bytes() == pass1
    st = _serve_session(project).state()
    assert st["done"] == 1 and st["pos"] == 1 and "prior" not in st
    assert core.cumulative_ms(project) == 3000 + 900         # pass-1 time still counts


def test_restart_refuses_clobber_missing_reason_and_mode_switch(project):
    project = _redo_project(project)
    with pytest.raises(AuditError, match="nothing to restart"):
        core.start_restart(project, "precision_pool", "author", REASON)
    _first_pass(project, ["yes", "no"])
    log = core.responses_path(project, "precision_pool")
    archive = core.archive_path(project, "precision_pool")
    before_log = log.read_bytes()
    for reason in (None, "  "):
        with pytest.raises(AuditError, match="--reason"):
            core.start_restart(project, "precision_pool", "author", reason)
    assert log.read_bytes() == before_log and not archive.exists()
    archive.write_text('{"event": "answer"}\n', encoding="utf-8")   # an archive already there
    before = archive.read_bytes()
    with pytest.raises(AuditError, match="refusing to overwrite"):
        core.start_restart(project, "precision_pool", "author", REASON)
    assert archive.read_bytes() == before and log.read_bytes() == before_log
    with pytest.raises(AuditError, match="--restart"):           # plain serve refuses too
        _serve_session(project)
    log.unlink()                                                 # crash between move and header
    header = core.start_restart(project, "precision_pool", "author", REASON)
    assert archive.read_bytes() == before and core.read_events(log) == [header]
    with pytest.raises(AuditError, match="restart is in progress"):   # never switch to non-blind
        core.start_redo(project, "precision_pool", "author")
    assert core.read_events(log) == [header]


def test_redo_in_progress_refuses_restart(project):
    project = _redo_project(project)
    _first_pass(project, ["yes"])
    header = core.start_redo(project, "precision_pool", "author")
    with pytest.raises(AuditError, match="redo is in progress"):
        core.start_restart(project, "precision_pool", "author", REASON)
    assert core.read_events(core.responses_path(project, "precision_pool")) == [header]


def test_restart_page_payload_is_blind_with_followup(project, monkeypatch):
    from nakkhu.audit.server import make_server

    project = _redo_project(project, "flood_trace")
    _first_pass(project, ["yes", "no", "no"], "flood_trace")      # yes, damage no, then no
    core.start_restart(project, "flood_trace", "author", REASON)
    _no_archive_reads(monkeypatch)
    s = _serve_session(project, "flood_trace")
    server = make_server(s, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = urllib.request.urlopen(base + "/api/state").read().decode()
    finally:
        server.shutdown()
        server.server_close()
    state = json.loads(body)
    assert "prior" not in state and state["redo"] is None
    for leak in ("prior", "Previous", "pass1", "restart", REASON):
        assert leak not in body
    st = s.answer(0, "main", "yes", 900)
    assert st["qid"] == "followup" and "prior" not in st
    st = s.answer(0, "followup", "no", 900)
    assert st["qid"] == "main" and st["pos"] == 1 and "prior" not in st
    last = core.read_events(core.responses_path(project, "flood_trace"))[-1]
    assert last["question"] == "damage" and last["restart_of"] == "pass1"
    assert last["prior_answer_shown"] is False and "prior_answer" not in last


def test_status_reports_restart_block(project):
    from nakkhu.audit.workflow import status

    project = _redo_project(project)
    _first_pass(project, ["yes", "no", "unsure"])
    core.start_restart(project, "precision_pool", "author", REASON)
    _serve_session(project).answer(0, "main", "no", 900)
    task = status(project)["tasks"]["precision_pool"]
    assert task["answered"] == 1 and "redo" not in task
    block = task["restart"]
    assert block["archive"] == "audit/precision_pool/responses_pass1.jsonl"
    assert block["archive_answered"] == 3 and block["restart_of"] == "pass1"
    assert block["prior_answer_shown"] is False and block["reason"] == REASON
    assert block["state"] == "in progress" and "redo_of" not in block
    assert not any(k in block for k in ("yes", "no", "unsure", "answers"))


def test_estimate_uses_the_complete_restart_only(project, monkeypatch):
    from nakkhu.audit import workflow

    project = _redo_project(project)
    monkeypatch.setattr(workflow, "require_frozen",
                        lambda _p: {"prereg_sha256": "p", "split_sha256": "s"})
    monkeypatch.setattr(workflow, "record_test_access", lambda *a, **k: {"flagged": False})
    monkeypatch.setattr(estimators, "precision_estimates", lambda *a, **k: {})
    s = _first_pass(project, [])
    st = s.state()
    while st["phase"] == "run":
        st = s.answer(st["pos"], st["qid"], "yes", 500)
    core.start_restart(project, "precision_pool", "author", REASON)
    with pytest.raises(AuditError, match="not complete"):       # estimated once, whole
        workflow.estimate(project, "precision_pool", n_boot=10)
    s = _serve_session(project)
    st = s.state()
    while st["phase"] == "run":
        st = s.answer(st["pos"], st["qid"], "no", 500)
    pd.DataFrame({"run": ["r"], "cluster": [0], "model": ["m"]}).to_parquet(
        core.task_dir(project, "precision_pool") / "members.parquet")
    workflow.estimate(project, "precision_pool", n_boot=10)
    out = json.loads((project.paths.artifacts / "audit" / "precision_pool_estimates.json")
                     .read_text(encoding="utf-8"))
    disclosed = out["answers_pass"]["precision_pool"]
    assert disclosed["mode"] == "restart" and disclosed["restart_of"] == "pass1"
    assert disclosed["prior_answer_shown"] is False and disclosed["reason"] == REASON
    assert disclosed["archive_sha256"] == core.sha256_file(
        core.archive_path(project, "precision_pool"))
    assert sum(r["yes"] for r in out["strata"]) == 0             # pass-1 answers never used
    assert sum(r["no"] for r in out["strata"]) > 0


def test_serve_restart_claims_the_port_before_moving_files(project, monkeypatch):
    import argparse
    import socket

    from nakkhu import audit_cli
    from nakkhu.evaluate import guard

    project = _redo_project(project)
    _first_pass(project, ["yes", "no"])
    log = core.responses_path(project, "precision_pool")
    before = log.read_bytes()
    monkeypatch.setattr(guard, "require_frozen", lambda _p: {})
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        args = argparse.Namespace(task="precision_pool", port=busy.getsockname()[1],
                                  redo=False, restart=True, reason=REASON)
        assert audit_cli._serve(project, args) == 1              # port taken: nothing moved
    finally:
        busy.close()
    assert log.read_bytes() == before
    assert not core.archive_path(project, "precision_pool").exists()
    args = argparse.Namespace(task="precision_pool", port=0, redo=False, restart=False,
                              reason=REASON)
    assert audit_cli._serve(project, args) == 1                  # --reason without --restart
    assert log.read_bytes() == before


# --- chip validity (raster mask + no-data stored as 0) ---------------------------------------


def test_valid_share_uses_the_mask_and_rejects_zero_stored_nodata(root):
    from nakkhu.audit.chips import ChipStyle, Reader, render_single, valid_share

    rng = np.random.default_rng(3)
    data = rng.integers(30, 200, (3, 600, 600), dtype=np.uint8)
    data[:, 250:350, 250:350] = 1          # interior hole: 0/1 under an all-valid mask (OAM)
    transform = from_origin(1000, 2200, 0.30517578125, 0.30517578125)
    holes, masked = root / "holes.tif", root / "masked.tif"
    with rasterio.open(holes, "w", driver="GTiff", width=600, height=600, count=3,
                       dtype="uint8", crs="EPSG:32645", transform=transform) as ds:
        ds.write(data)
    bright = rng.integers(30, 200, (3, 600, 600), dtype=np.uint8)
    mask = np.full((600, 600), 255, dtype=np.uint8)
    mask[250:350, 250:350] = 0             # masked out although its pixels are bright
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), \
            rasterio.open(masked, "w", driver="GTiff", width=600, height=600, count=3,
                          dtype="uint8", crs="EPSG:32645", transform=transform) as ds:
        ds.write(bright)
        ds.write_mask(mask)
    style = ChipStyle()
    # pixel (300, 300) centre: x = 1000 + 300.5 * 0.305, y = 2200 - 300.5 * 0.305
    cx, cy = 1000 + 300.5 * 0.30517578125, 2200 - 300.5 * 0.30517578125
    reader = Reader()
    try:
        for path in (holes, masked):
            _, share = render_single(reader, path, cx, cy, style)
            assert share == 0.0, path.name
            _, share = render_single(reader, path, 1000 + 100 * 0.30517578125,
                                     2200 - 100 * 0.30517578125, style)
            assert share == 1.0, path.name
        rgb = reader.read(holes, (cx - 32, cy - 32, cx + 32, cy + 32), style.size_px)
        old_rule = float((rgb[200:220, 200:220].max(axis=-1) > 0).mean())
        assert old_rule == 1.0                 # the blank-panel bug: max > 0 passed the hole
        c = style.size_px / 2
        assert valid_share(rgb, (c, c)) == 0.0
    finally:
        reader.close()
