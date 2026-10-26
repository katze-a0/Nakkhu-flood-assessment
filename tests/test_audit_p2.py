"""Plan 014 Phase-2 audit task (flood_trace_p2): label-blind order, per-site design and
repeats, two-strip pre chips, pace for every rater, the local and rater pages, the follow-up
mapping, and the P2-audit estimate (synthetic data)."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
import yaml
from rasterio.transform import from_origin

from nakkhu.audit import core, frames, raters
from nakkhu.audit.core import P2_TASK, AuditError

CFG = {
    "schema_version": 1, "seed": 0,
    "scenes": {"pre": "img_pre_a", "post": ["img_post"], "oam": []},
    "chips": {"context_m": 32.0, "upscale": 1, "jpeg_quality": 80, "prefetch": 3},
    "time": {"cap_min": 120, "sitting_max_min": 35, "sitting_budget_min": 30, "pilot_items": 6,
             "floor_per_task": 10, "repeat_frac": 0.05, "repeat_min_gap": 3},
    "tasks": {
        P2_TASK: {"question": "Trace?", "keys": {"yes": "Yes", "no": "No",
                                                 "unsure": "Not observable"},
                  "auditor": "author", "followup": "damage", "requires_frozen": "prereg_v9_a4",
                  "sites": ["A", "B"], "change_maps": "configs/followup/phase2/p2_change_maps.yaml",
                  "site_lines": "artifacts/followup/phase2/site_{site}_lines.geojson",
                  "site_extent": "artifacts/followup/phase2/site_{site}_extent.geojson",
                  "near_channel_m": 30, "design_n": 24, "min_per_stratum": 2,
                  "rebalance": {"stratum": "positive", "n": 8, "from": "negative_far"},
                  "repeats_per_site": 3, "pace_all_raters": True,
                  "scenes": {"pre": {"A": ["img_pre_a", "img_pre_b"], "B": ["img_pre_a"]},
                             "post": ["img_post"]}},
        "damage": {"question": "Damage?", "keys": {"yes": "Yes", "no": "No",
                                                    "unsure": "Unsure"}},
    },
}
# site A: x 1000-1200 (inside img_pre_a) and x 1300-1500 (only img_pre_b); site B: y below
SITES = {"A": (shapely.box(1000, 2000, 1500, 2200), 2100.0),
         "B": (shapely.box(1000, 1500, 1200, 1700), 1600.0)}


def _raster(path, seed, west, width_m):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    px = int(width_m / 0.30517578125)
    with rasterio.open(path, "w", driver="GTiff", width=px, height=3000, count=3, dtype="uint8",
                       crs="EPSG:32645",
                       transform=from_origin(west, 2300, 0.30517578125, 0.30517578125)) as ds:
        ds.write(rng.integers(40, 200, (3, 3000, px), dtype=np.uint8))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def p2(project, monkeypatch):
    root = project.paths.root
    art = root / "artifacts" / "followup" / "phase2"
    art.mkdir(parents=True)
    rng = np.random.default_rng(3)
    maps = {}
    for site, (box, yline) in SITES.items():
        x0, y0, x1, y1 = box.bounds
        gpd.GeoDataFrame({"site": [site]}, geometry=[box], crs="EPSG:32645").to_file(
            art / f"site_{site}_extent.geojson", driver="GeoJSON")
        gpd.GeoDataFrame({"part": ["site"]}, geometry=[shapely.LineString([(x0, yline),
                                                                          (x1, yline)])],
                         crs="EPSG:32645").to_file(art / f"site_{site}_lines.geojson",
                                                   driver="GeoJSON")
        n = 120
        xs, ys = rng.uniform(x0 + 20, x1 - 20, n), rng.uniform(y0 + 20, y1 - 20, n)
        cls = np.where(rng.random(n) < 0.25, "positive", "negative")
        cmap = gpd.GeoDataFrame({"building_id": [f"{site}/microsoft:{i}" for i in range(n)],
                                 "class": cls, "observable": rng.random(n) > 0.1},
                                geometry=[shapely.box(x - 4, y - 3, x + 4, y + 3)
                                          for x, y in zip(xs, ys, strict=True)],
                                crs="EPSG:32645")
        f = root / "runs" / "followup-p2" / f"chg{site}" / "change_map_buildings_microsoft.parquet"
        f.parent.mkdir(parents=True)
        cmap.to_parquet(f)
        maps[site] = {"file": f.relative_to(root).as_posix(), "sha256": _sha(f)}
    (root / "configs" / "followup" / "phase2").mkdir(parents=True)
    (root / CFG["tasks"][P2_TASK]["change_maps"]).write_text(yaml.safe_dump({"sites": maps}),
                                                             encoding="utf-8")
    (root / "configs" / "audit").mkdir(parents=True)
    (root / "configs" / "audit" / "audit.yaml").write_text(yaml.safe_dump(CFG), encoding="utf-8")
    _raster(root / "data" / "raw" / "img_pre_a" / "a.tif", 1, 900, 350)     # x 900-1250
    _raster(root / "data" / "raw" / "img_pre_b" / "b.tif", 2, 900, 700)     # x 900-1600
    _raster(root / "data" / "raw" / "img_post" / "p.tif", 3, 900, 700)
    (root / "configs" / "data" / "manifest.yaml").write_text(yaml.safe_dump(
        {"schema_version": 1, "entries": [{"id": s, "local_path": f"data/raw/{s}", "status": "ok"}
                                          for s in ("img_pre_a", "img_pre_b", "img_post")]}),
        encoding="utf-8")
    core.allocation_path(project).parent.mkdir(parents=True, exist_ok=True)
    core.allocation_path(project).write_text(yaml.safe_dump(
        {"pilot": {"median_s": 2.0}, "tasks": {}}), encoding="utf-8")
    monkeypatch.setattr(frames, "p2_require_freeze", lambda project, tcfg: None)
    return project


def _sampled(project):
    from nakkhu.audit.chips import render_task
    from nakkhu.audit.workflow import build_items

    cfg = core.load_audit_config(project)
    items, frame, extra = build_items(project, cfg, P2_TASK)
    path = core.items_path(project, P2_TASK)
    path.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(path, index=False)
    manifest = render_task(project, P2_TASK, items, cfg, core.chip_dir(project, P2_TASK))
    return cfg, items, frame, extra, manifest


def test_drawing_refuses_before_the_phase2_freeze(project):
    with pytest.raises(AuditError, match="prereg_v9_a4 is not frozen"):
        frames.p2_require_freeze(project, CFG["tasks"][P2_TASK])


def test_items_are_per_site_strata_with_the_v9_rule_and_deterministic(p2):
    from nakkhu.audit.workflow import build_items

    cfg, items, frame, extra, _ = _sampled(p2)
    assert set(frame["stratum"]) <= {f"{s}:{k}" for s in "AB"
                                     for k in ("positive", "negative_near", "negative_far")}
    for site in "AB":
        sub = items.loc[items["site"] == site]
        assert len(sub) == 24
        assert (sub["stratum_in_site"] == "positive").sum() == min(
            8, int(((frame["site"] == site) & (frame["stratum_in_site"] == "positive")).sum()))
    assert items["item_id"].str.match(r"^fq\d{4}$").all()
    again, _, _ = build_items(p2, cfg, P2_TASK)
    pd.testing.assert_frame_equal(items.reset_index(drop=True), again.reset_index(drop=True))


def test_queue_has_three_repeats_per_site_for_author_and_rater(p2):
    cfg, items, *_ = _sampled(p2)
    site = dict(zip(items["item_id"], items["site"], strict=True))
    for queue in (core.build_queue(items, core.load_allocation(p2), P2_TASK, cfg),
                  raters.rater_queue(items, core.load_allocation(p2), P2_TASK, cfg, "rater_2")):
        reps = [s for s in queue if s.repeat]
        assert sorted(site[s.item_id] for s in reps) == ["A"] * 3 + ["B"] * 3
        pos = {s.slot: i for i, s in enumerate(queue)}
        n = len(queue)
        assert all(pos[f"{s.item_id}#r"] - pos[s.item_id] >= 3 or pos[f"{s.item_id}#r"] >= n - 7
                   for s in reps)
        assert len([s for s in queue if not s.repeat]) == 48


def test_chips_take_the_pre_strip_that_covers_the_building(p2):
    _, items, _, _, manifest = _sampled(p2)
    for row in items.itertuples():
        entry = manifest[row.item_id]
        if row.site == "A" and 1225 <= row.x <= 1260:
            continue                                 # near the strip edge: either is right
        want = "img_pre_a" if row.site == "B" or row.x < 1225 else "img_pre_b"
        assert entry["pre"] == want and entry["post"] == "img_post", (row.site, row.x, entry)


def _fast(session, n):
    st, warned = session.state(), False
    for _ in range(n):
        st = session.answer(st["pos"], st["qid"], "no", 500)
        warned = warned or bool(st.get("pace_warning"))
    return {**st, "pace_warning": warned}


def test_author_page_is_paced_and_the_damage_followup_follows_a_yes(p2):
    from nakkhu.audit.server import Session

    cfg, *_ = _sampled(p2)
    s = Session(p2, P2_TASK, cfg)
    st = s.state()
    assert st["pace"]["target_s"] == [10, 15] and st["pace_warning"] is False
    assert any("10-15 seconds" in line for line in st["help"])
    st = s.answer(st["pos"], "main", "yes", 12000)
    assert st["qid"] == "followup" and st["question"] == "Damage?"
    st = s.answer(st["pos"], "followup", "no", 4000)
    st = _fast(s, raters.FAST_WINDOW)
    assert st["pace_warning"] is True
    events = core.read_events(core.responses_path(p2, P2_TASK))
    assert any(e["event"] == "pace_warning" for e in events)


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_local_and_rater_pages_render_with_chips(p2):
    from nakkhu.audit.server import RaterHub, Session, make_rater_server, make_server

    cfg, items, *_ = _sampled(p2)
    server = make_server(Session(p2, P2_TASK, cfg), 0)
    base = _serve(server)
    try:
        page = urllib.request.urlopen(base + "/").read()
        assert b"<title>" in page
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert state["phase"] == "run" and state["pace"]
        chip = urllib.request.urlopen(base + state["chip"])
        assert chip.status == 200 and chip.headers["Content-Type"] == "image/jpeg"
    finally:
        server.shutdown()
        server.server_close()
    token, _ = raters.invite(p2, cfg, P2_TASK, "rater_2")
    hub = RaterHub(p2, P2_TASK, cfg, refresh_s=0.0)
    rserver = make_rater_server(hub, 0)
    rbase = _serve(rserver) + f"/r/{token}/"
    try:
        assert b"<title>" in urllib.request.urlopen(rbase).read()
        body = urllib.request.urlopen(rbase + "api/state").read().decode()
        state = json.loads(body)
        assert state["remote"] and state["pace"] and state["phase"] == "run"
        for secret in ('"site"', '"stratum"', "positive", "negative_", '"A"', "change_class"):
            assert secret not in body
        assert urllib.request.urlopen(rbase + state["chip"]).status == 200
    finally:
        rserver.shutdown()
        rserver.server_close()


def test_followup_names_map_to_the_p2_log(p2):
    from nakkhu.audit import workflow as W

    assert W._log_task("damage_p2") == P2_TASK and W._question("damage_p2") == "damage"
    assert W._log_task("damage") == "flood_trace" and W._question(P2_TASK) == P2_TASK
    assert P2_TASK in W.STRICT_TASKS and "damage_p2" in W.STRICT_TASKS
    from nakkhu.audit_cli import SUMMARY_CHOICES, TASK_CHOICES

    assert P2_TASK in TASK_CHOICES and "damage_p2" in SUMMARY_CHOICES
    assert P2_TASK in core.SERVE_TASKS and core.TASKS[-1] == P2_TASK


def _answer_all(project, cfg, positive_yes=0.8):
    from nakkhu.audit.server import Session

    items = core.load_items(project, P2_TASK)
    pos = dict(zip(items["item_id"], items["stratum_in_site"] == "positive", strict=True))
    s = Session(project, P2_TASK, cfg)
    rng = np.random.default_rng(0)
    st = s.state()
    while st["phase"] == "run":
        item = s.queue[st["pos"]].item_id
        if st["qid"] == "followup":
            ans = "no"
        else:
            ans = "yes" if pos[item] and rng.random() < positive_yes else "no"
        st = s.answer(st["pos"], st["qid"], ans, 11000)
    return st


def test_p2_audit_estimate_reads_the_primary_rater_per_site(p2):
    from nakkhu.followup import p2audit

    cfg, *_ = _sampled(p2)
    (core.task_dir(p2, P2_TASK) / "sample.json").write_text(json.dumps({"inputs": {}}),
                                                           encoding="utf-8")
    assert any("not complete" in p for p in p2audit.prerequisites(p2))
    assert _answer_all(p2, cfg)["phase"] == "done"
    assert p2audit.prerequisites(p2) == []
    out, info = p2audit.score(p2, {}, n_boot=30, seed=0)
    doc = out["p2_audit_estimates.json"]
    assert doc["primary_rater"] == "author" and set(doc["sites"]) == {"A", "B"}
    for site in "AB":
        ppv = doc["sites"][site]["ppv"]
        assert ppv["n"] > 0 and 0 <= ppv["wilson_lo"] <= ppv["estimate"] <= 1
        assert doc["sites"][site]["drop_change_map"] == (ppv["wilson_lo"] < 0.5)
    assert set(info) == {"A", "B"}
    assert len(p2audit.input_files(p2)) == 6


def test_estimate_command_refuses_the_p2_task(p2):
    from nakkhu.audit.workflow import estimate

    with pytest.raises(Exception, match="P2-audit"):
        estimate(p2, P2_TASK)


def test_p2_audit_reads_a_remote_primary_rater_log(p2):
    from nakkhu.audit.server import Session
    from nakkhu.followup import p2audit

    cfg, *_ = _sampled(p2)
    cfg["tasks"][P2_TASK]["primary_rater"] = "rater_1"
    (p2.paths.root / "configs" / "audit" / "audit.yaml").write_text(yaml.safe_dump(cfg),
                                                                   encoding="utf-8")
    (core.task_dir(p2, P2_TASK) / "sample.json").write_text(json.dumps({"inputs": {}}),
                                                           encoding="utf-8")
    raters.invite(p2, cfg, P2_TASK, "rater_1")
    inv = raters.load_invites(p2, P2_TASK)[(P2_TASK, "rater_1")]
    assert any("unanswered by the primary rater" in p for p in p2audit.prerequisites(p2))
    s = Session(p2, P2_TASK, cfg, inv)
    st = s.state()
    while st["phase"] == "run":
        st = s.answer(st["pos"], st["qid"], "no" if st["qid"] == "main" else "no", 11000)
    assert st["phase"] == "done"
    assert not core.responses_path(p2, P2_TASK).exists()
    assert p2audit.prerequisites(p2) == []
    assert p2audit.primary_log(p2, cfg) == raters.rater_log(p2, P2_TASK, "rater_1")
    out, _ = p2audit.score(p2, {}, n_boot=20, seed=0)
    doc = out["p2_audit_estimates.json"]
    assert doc["primary_rater"] == "rater_1" and doc["primary_remote"] is True
    assert doc["primary_log"].endswith("raters/rater_1/responses.jsonl")
    assert raters.rater_log(p2, P2_TASK, "rater_1").relative_to(p2.paths.root).as_posix() in [
        f.relative_to(p2.paths.root).as_posix() for f in p2audit.input_files(p2)]
