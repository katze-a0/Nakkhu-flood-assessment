"""``nakkhu audit`` actions: sample, chips, allocation, summarise, estimate, status.

Test-region rules (prereg ``test_access``): sampling and chips read reference footprints,
registration shifts and imagery of the test region but score nothing, so they need the frozen
prereg (:func:`require_frozen`) and are not journalled. ``summarise`` reports only blind process
numbers (counts, timing, self-agreement), never a per-source or per-model figure. ``estimate``
turns answers into source/model/change-map estimates: that is a test access, journalled in
``runs/eval/test_journal.jsonl`` before anything is computed.

``recall_nearmiss`` (sitting 1b) is drawn from the finished sitting 1 (its items and effective
answers, hashed into ``sample.json``); ``estimate --task recall_points`` then reports recall
protocol v2 (primary, needs sitting 1b complete) and v1 (sitting 1 only), each carrying the
auditor of every sitting.

A redone or restarted sitting (``serve --redo`` / ``--restart``) is estimated from the current
pass only (the current ``responses.jsonl``), once complete; the archived pass 1 is hashed into
the material, never read for answers, and the result carries the disclosure (``redo_of`` or
``restart_of``, ``prior_answer_shown``, ``mode``, the restart ``reason``). Passes never mix.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import shapely
import yaml

from nakkhu.audit import core, frames
from nakkhu.audit import sample as S
from nakkhu.audit.core import AuditError
from nakkhu.config import Project, ResolvedConfig
from nakkhu.evaluate.guard import record_test_access, require_frozen
from nakkhu.runs import atomic_write_bytes, atomic_write_text, create_run, utc_now

ITEM_COLUMNS = ["item_id", "task", "stratum", "frame_n", "design_n", "rank", "pi_design",
                "pilot", "order_key", "unit", "x", "y"]


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _input_hashes(project: Project, cfg: dict[str, Any], task: str) -> dict[str, Any]:
    entries = frames.manifest_entries(project)
    out: dict[str, Any] = {}
    if task == "recall_points":
        for name in cfg["tasks"][task]["sources"]:
            out[frames.SOURCE_IDS[name]] = entries[frames.SOURCE_IDS[name]].get("sha256")
        settings = project.paths.resolve(frames.GREENHOUSE_SETTINGS)
        out["greenhouse_settings"] = core.sha256_file(settings) if settings.is_file() else None
        for kind, rel in (cfg.get("registration") or {}).items():
            path = project.paths.artifacts / str(rel)
            out[f"registration_{kind}"] = core.sha256_file(path) if path.is_file() else None
    elif task in ("flood_trace", "damage"):
        out.update(frames.flood_hashes(project, cfg["tasks"]["flood_trace"]))
    elif task in (core.P2_TASK, "damage_p2"):
        out.update(frames.p2_hashes(project, cfg["tasks"][core.P2_TASK]))
    elif task == "precision_microsoft":
        out[frames.SOURCE_IDS["microsoft"]] = entries[frames.SOURCE_IDS["microsoft"]].get("sha256")
        tcfg = cfg["tasks"][task]
        from nakkhu.hydro.sweep import load_settings

        settings, _ = load_settings(project)
        agree = settings.resolve(settings.v["inventories"][str(tcfg.get(
            "agree_with", "detector_final"))]["path"])
        out["agree_inventory"] = core.sha256_file(agree) if agree.is_file() else None
        reg = project.paths.artifacts / str((cfg.get("registration") or {})["footprints"])
        out["registration_footprints"] = core.sha256_file(reg) if reg.is_file() else None
    elif task == core.NEARMISS_TASK:
        parent = str(cfg["tasks"][task].get("parent", core.PILOT_TASK))
        for kind, path in (("parent_items", core.items_path(project, parent)),
                           ("parent_responses", core.responses_path(project, parent))):
            out[kind] = core.sha256_file(path) if path.is_file() else None
        reg = project.paths.artifacts / str((cfg.get("registration") or {})["footprints"])
        out["registration_footprints"] = core.sha256_file(reg) if reg.is_file() else None
    return out


# serve / estimate refuse on any input drift
STRICT_TASKS = ("flood_trace", "damage", core.NEARMISS_TASK, core.P2_TASK, "damage_p2")


def primary_answers(project: Project, task: str, question: str | None = None
                    ) -> dict[str, str]:
    """Effective answers of the primary (non-repeat) slots of a task, undo events honoured."""
    state = core.replay(core.read_events(core.responses_path(project, task)))
    q = question or task
    return {e["item_id"]: e["answer"] for e in state.effective
            if e["question"] == q and not e.get("repeat")}


def require_complete(project: Project, cfg: dict[str, Any], task: str) -> None:
    """Refuse unless every queue slot of ``task`` has an effective answer."""
    items = core.load_items(project, task)
    queue = core.build_queue(items, core.load_allocation(project), task, cfg)
    state = core.replay(core.read_events(core.responses_path(project, task)))
    open_slots = [s.slot for s in queue if (s.slot, s.question or task) not in state.answers]
    if open_slots:
        raise AuditError(f"{task} is not complete: {len(open_slots)} of {len(queue)} slots "
                         "unanswered")


def require_followups(project: Project, cfg: dict[str, Any], task: str) -> None:
    """Refuse unless every effective Yes of ``task`` has its follow-up answer (sitting 3:
    the damage question asked right after a flood_trace Yes)."""
    follow = core.task_config(cfg, task).get("followup")
    if not follow:
        return
    state = core.replay(core.read_events(core.responses_path(project, task)))
    missing = [s for (s, q), e in state.answers.items()
               if q == task and e.get("answer") == "yes" and (s, str(follow)) not in state.answers]
    if missing:
        raise AuditError(f"{task} is not complete: {len(missing)} Yes answer(s) without the "
                         f"{follow} follow-up")


def build_nearmiss(project: Project, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    tcfg = core.task_config(cfg, core.NEARMISS_TASK)
    parent = str(tcfg.get("parent", core.PILOT_TASK))
    require_complete(project, cfg, parent)
    items = core.load_items(project, parent)
    answers = primary_answers(project, parent)
    out = S.nearmiss_items(items, answers, tcfg, int(cfg.get("seed", 0)), parent)
    kinds = out["kind"].value_counts().to_dict()
    near = out.loc[out["kind"] == "nearmiss"]
    stats = {"parent": parent, "radius_m": float(tcfg["radius_m"]),
             "auditor": core.task_auditor(cfg, core.NEARMISS_TASK),
             "parent_auditor": core.task_auditor(cfg, parent),
             "kinds": {k: int(v) for k, v in sorted(kinds.items())},
             "nearmiss_by_stratum": {k: int(v) for k, v in
                                     near["stratum"].value_counts().sort_index().items()}}
    return out, stats


def _sample_nearmiss(project: Project, cfg: dict[str, Any], guard: dict[str, Any], *,
                     replace: bool, check: bool) -> dict[str, Any]:
    task = core.NEARMISS_TASK
    path = core.items_path(project, task)
    if check:
        if not path.is_file():
            raise AuditError(f"{path} does not exist; nothing to check")
        items, _ = build_nearmiss(project, cfg)
        same = pd.read_parquet(path).reset_index(drop=True).equals(items.reset_index(drop=True))
        return {"task": task, "identical": bool(same), "items": len(items)}
    if path.is_file() and not replace:
        raise AuditError(f"{path} exists (the frozen sample); pass --replace to redraw")
    if core.any_answers(project, task):
        raise AuditError(f"{task} already has answers; the sample is final")
    items, stats = build_nearmiss(project, cfg)
    run = create_run(project, "audit", ResolvedConfig(
        values={"command": "audit-sample", "task": task, "audit": cfg}, source=core.AUDIT_CONFIG))
    with run.step(f"sample-{task}"):
        data = _items_bytes(items)
        atomic_write_bytes(path, data)
        summary = {"task": task, "items": len(items), "items_sha256":
                   hashlib.sha256(data).hexdigest(), "seed": int(cfg.get("seed", 0)),
                   "prereg_sha256": guard["prereg_sha256"],
                   "split_sha256": guard["split_sha256"],
                   "inputs": _input_hashes(project, cfg, task), **stats}
        run.write_summary(summary)
        atomic_write_text(core.task_dir(project, task) / "sample.json",
                          json.dumps({"run": run.ref, **summary}, indent=1, sort_keys=True))
    return {"run": run.ref, **summary}


def build_items(project: Project, cfg: dict[str, Any], task: str
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(items, frame, extra) for a task; deterministic for a fixed seed and inputs."""
    seed = int(cfg.get("seed", 0))
    tcfg = core.task_config(cfg, task)
    if task == "recall_points":
        frame, extra = frames.recall_frame(project, cfg)
    elif task == "precision_pool":
        frame, extra = frames.precision_frame(project, cfg)
    elif task == "flood_trace":
        frame, extra = frames.flood_frame(project, cfg)
    elif task == "precision_microsoft":
        frame, extra = frames.microsoft_frame(project, cfg)
    elif task == core.P2_TASK:
        frame, extra = frames.p2_frame(project, cfg)
    else:
        raise AuditError(f"{task} has no frame of its own")
    frame_n = frame.groupby("stratum").size().to_dict()
    if task == core.P2_TASK:
        sizes = p2_sizes(frame, tcfg)
    else:
        sizes = S.design_sizes(frame_n, int(tcfg["design_n"]), int(tcfg["min_per_stratum"]))
        if tcfg.get("rebalance"):
            rb = tcfg["rebalance"]
            sizes = S.rebalance(sizes, frame_n, str(rb["stratum"]), int(rb["n"]),
                                str(rb["from"]))
    drawn = S.draw(pd.DataFrame(frame.drop(columns="geometry")).assign(
        wkb=shapely.to_wkb(np.asarray(frame.geometry.values, dtype=object))),
        sizes, seed, task)
    extra_stats = dict(extra.get("stats") or {})
    if task == "recall_points" and int(tcfg.get("outside_n", 0)) > 0:
        area = frames.outside_area(extra["region"], extra["all_polygons"],
                                   float(tcfg["outside_buffer_m"]), extra.get("greenhouses"))
        rng = np.random.default_rng([seed, 3, 0])
        pts = frames.outside_points(area, int(tcfg["outside_n"]), rng)
        out = pd.DataFrame({"unit": -1 - np.arange(len(pts)), "stratum": frames.OUTSIDE,
                            "x": shapely.get_x(pts), "y": shapely.get_y(pts),
                            "wkb": shapely.to_wkb(pts), "rank": np.arange(len(pts)),
                            "frame_n": 0, "design_n": len(pts)})
        for name in tcfg["sources"]:
            out[f"in_{name}"] = False
            out[f"origin_{name}"] = False
        out["in_google_t90"] = False
        drawn = pd.concat([drawn, out], ignore_index=True)
        extra_stats["outside_area_m2"] = float(area.area)
        sizes[frames.OUTSIDE] = len(pts)
        frame_n[frames.OUTSIDE] = 0
    pilot = int((cfg.get("time") or {}).get("pilot_items", 0))
    items = S.finish_items(drawn, task, seed, pilot)
    items["frame_area_m2"] = np.where(items["stratum"] == frames.OUTSIDE,
                                      extra_stats.get("outside_area_m2", np.nan), np.nan)
    rest = [c for c in items.columns if c not in ITEM_COLUMNS]
    items = items[ITEM_COLUMNS + rest]
    extra_stats["strata"] = {k: {"frame_n": int(frame_n[k]), "design_n": int(sizes.get(k, 0))}
                             for k in sorted(frame_n)}
    extra_stats["pilot_items"] = int(items["pilot"].sum())
    return items, frame, {**extra, "stats": extra_stats}


def p2_sizes(frame: pd.DataFrame, tcfg: dict[str, Any]) -> dict[str, int]:
    """Design sizes of the Phase-2 audit: per site the v9 flood_trace rule (sqrt allocation of
    ``design_n`` with ``min_per_stratum``, then the ``rebalance``), keys ``<site>:<stratum>``."""
    out: dict[str, int] = {}
    rb = tcfg.get("rebalance")
    for site in tcfg["sites"]:
        sub = frame.loc[frame["site"].astype(str) == str(site)]
        n_h = sub.groupby("stratum_in_site").size().to_dict()
        sizes = S.design_sizes(n_h, int(tcfg["design_n"]), int(tcfg["min_per_stratum"]))
        if rb:
            sizes = S.rebalance(sizes, n_h, str(rb["stratum"]), int(rb["n"]), str(rb["from"]))
        out.update({f"{site}:{k}": int(v) for k, v in sizes.items()})
    return out


def _items_bytes(items: pd.DataFrame) -> bytes:
    import io

    buffer = io.BytesIO()
    items.to_parquet(buffer, index=False)
    return buffer.getvalue()


def sample_task(project: Project, task: str, *, replace: bool = False,
                check: bool = False) -> dict[str, Any]:
    cfg = core.load_audit_config(project)
    guard = require_frozen(project)
    if task == core.NEARMISS_TASK:
        return _sample_nearmiss(project, cfg, guard, replace=replace, check=check)
    path = core.items_path(project, task)
    if check:
        if not path.is_file():
            raise AuditError(f"{path} does not exist; nothing to check")
        items, _, extra = build_items(project, cfg, task)
        old = pd.read_parquet(path)
        same = old.reset_index(drop=True).equals(items.reset_index(drop=True))
        return {"task": task, "identical": bool(same), "items": len(items)}
    if path.is_file() and not replace:
        raise AuditError(f"{path} exists (the frozen sample); pass --replace to redraw")
    if core.any_answers(project, task):
        raise AuditError(f"{task} already has answers (or an archived pass); the sample is "
                         "final")
    items, frame, extra = build_items(project, cfg, task)
    run = create_run(project, "audit", ResolvedConfig(
        values={"command": "audit-sample", "task": task, "audit": cfg}, source=core.AUDIT_CONFIG))
    with run.step(f"sample-{task}"):
        interim = core.interim_dir(project, task)
        interim.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(interim / "frame.parquet")
        data = _items_bytes(items)
        atomic_write_bytes(path, data)
        if "members" in extra:
            extra["members"].to_parquet(core.task_dir(project, task) / "members.parquet",
                                        index=False)
        summary = {"task": task, "items": len(items), "items_sha256":
                   hashlib.sha256(data).hexdigest(), "seed": int(cfg.get("seed", 0)),
                   "prereg_sha256": guard["prereg_sha256"],
                   "split_sha256": guard["split_sha256"],
                   "inputs": _input_hashes(project, cfg, task), **extra["stats"]}
        run.write_summary(summary)
        atomic_write_text(core.task_dir(project, task) / "sample.json",
                          json.dumps({"run": run.ref, **summary}, indent=1, sort_keys=True))
    return {"run": run.ref, **summary}


def input_drift(project: Project, task: str) -> list[str]:
    """Inputs whose hash differs from the one recorded when the sample was drawn."""
    path = core.task_dir(project, task) / "sample.json"
    if not path.is_file():
        return []
    recorded = json.loads(path.read_text(encoding="utf-8")).get("inputs") or {}
    live = _input_hashes(project, core.load_audit_config(project), task)
    return sorted(k for k, v in recorded.items() if live.get(k) != v)


# --- chips ------------------------------------------------------------------------------------


def chips_task(project: Project, task: str, *, force: bool = False,
               examples: bool = True) -> dict[str, Any]:
    from nakkhu.audit.chips import render_task

    cfg = core.load_audit_config(project)
    require_frozen(project)
    items = core.load_items(project, task)
    manifest = render_task(project, task, items, cfg, core.chip_dir(project, task), force=force)
    out: dict[str, Any] = {"task": task, "chips": len(manifest),
                           "rendered": sum(not v.get("reused") for v in manifest.values()),
                           "dir": str(core.chip_dir(project, task))}
    low = [k for k, v in manifest.items()
           if isinstance(v.get("valid"), float) and v["valid"] < 0.5]
    out["low_valid_chips"] = len(low)
    if examples and task == "recall_points":
        out["examples"] = render_examples(project, cfg, force=force)
    return out


EXAMPLES_FILE = "examples.json"          # audit/recall_points/examples.json (tracked)
CANDIDATE_DIR = "example_candidates"     # under {data_root}/interim/audit/recall_points/
EXAMPLE_CAPTIONS = {"yes": "Yes", "no": "No", "greenhouse": "No: greenhouse"}


def example_candidates(project: Project, cfg: dict[str, Any], per_kind: int = 8
                       ) -> list[dict[str, Any]]:
    """Rule-based instruction-card candidates from usable *train* tiles (never test items),
    rendered locally for a visual check:

    - ``yes``: Google >= t90 polygon of 80-400 m2 whose point lies inside a Microsoft polygon
      (footprint-registered tiles);
    - ``no``: a random point farther than 15 m from every Google, Microsoft and OSM footprint;
    - ``greenhouse``: a point on an OSM greenhouse area of >= 100 m2 (OSM-shifted).
    """
    from nakkhu.audit.chips import ChipStyle, Reader, render_single, scene_raster, write_jpeg
    from nakkhu.splits import split_tile_ids

    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng([seed, 4, 0])
    index = frames.tile_index(project)
    train = set(split_tile_ids(project, "train", usable_only=True))
    reg = pd.read_csv(project.paths.artifacts / cfg["registration"]["footprints"])
    ok = set(reg.loc[reg["reg_ok"].astype(bool), "tile_id"])
    pool = index.loc[index["tile_id"].isin(train & ok)].reset_index(drop=True)
    tiles = pool.iloc[rng.choice(len(pool), size=min(16, len(pool)), replace=False)]
    clip = shapely.union_all([shapely.box(*b) for b in tiles["bounds"]])
    inner = clip.buffer(-40)
    polys = frames.load_sources(project, cfg, clip, ["google", "microsoft", "osm"])
    g, m = polys["google"], polys["microsoft"]
    gg = np.asarray(g.geometry.values, dtype=object)
    good = (g["confidence"].to_numpy() >= frames.google_t90(project)) \
        & (shapely.area(gg) >= 80) & (shapely.area(gg) <= 400)
    rep = frames.representative_points(gg[good])
    rep = rep[shapely.within(rep, inner)]
    mi, _ = shapely.STRtree(np.asarray(m.geometry.values, dtype=object)).query(
        rep, predicate="within")
    yes_pts = rep[np.unique(mi)]
    allp = np.concatenate([np.asarray(p.geometry.values, dtype=object) for p in polys.values()])
    tree = shapely.STRtree(allp)
    no_pts: list[Any] = []
    for _ in range(50):
        if len(no_pts) >= 20 * per_kind:
            break
        xy = rng.uniform(inner.bounds[:2], inner.bounds[2:], size=(500, 2))
        cand = shapely.points(xy)
        cand = cand[shapely.within(cand, inner)]
        near, _ = tree.query(cand, predicate="dwithin", distance=15.0)
        no_pts.extend(np.delete(cand, np.unique(near)))
    gh, _ = frames.greenhouse_areas(project)
    train_area = shapely.union_all([shapely.box(*b) for b in
                                    index.loc[index["tile_id"].isin(train), "bounds"]])
    gh_geoms = np.asarray(gh.geometry.values, dtype=object)
    gh_geoms = gh_geoms[(shapely.area(gh_geoms) >= 100)
                        & shapely.within(shapely.point_on_surface(gh_geoms),
                                         train_area.buffer(-40))]
    gh_geoms = frames.apply_shifts(gh_geoms, index, frames.load_shifts(project, cfg)["osm"])
    gh_pts = shapely.point_on_surface(gh_geoms)

    out_dir = core.interim_dir(project, "recall_points") / CANDIDATE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    style = ChipStyle.from_config(cfg)
    pre = scene_raster(project, str(cfg["scenes"]["pre"]))
    reader = Reader()
    out: list[dict[str, Any]] = []
    try:
        for kind, pts in (("yes", yes_pts), ("no", np.asarray(no_pts, dtype=object)),
                          ("greenhouse", gh_pts)):
            n = 0
            for k in rng.permutation(len(pts)):
                if n >= per_kind:
                    break
                x, y = float(shapely.get_x(pts[k])), float(shapely.get_y(pts[k]))
                img, share = render_single(reader, pre, x, y, style)
                if share < 1.0:
                    continue
                n += 1
                name = f"{kind}_{n}.jpg"
                write_jpeg(img, out_dir / name, style.quality)
                out.append({"kind": kind, "file": name, "x": x, "y": y})
    finally:
        reader.close()
    (out_dir / "candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def render_examples(project: Project, cfg: dict[str, Any], *, force: bool = False
                    ) -> dict[str, Any]:
    """Render the chosen examples (``audit/recall_points/examples.json``: kind, x, y and who
    checked them) into the examples directory with a caption manifest. Without that file the
    card shows no examples."""
    from nakkhu.audit.chips import ChipStyle, Reader, render_single, scene_raster, write_jpeg

    chosen_path = core.task_dir(project, "recall_points") / EXAMPLES_FILE
    out_dir = core.examples_dir(project, "recall_points")
    if not chosen_path.is_file():
        return {"examples": 0, "note": f"{chosen_path} missing: no examples on the card"}
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))["examples"]
    manifest = out_dir / "examples.json"
    if not force and manifest.is_file() and \
            len(json.loads(manifest.read_text(encoding="utf-8"))) == len(chosen):
        return {"examples": len(chosen), "reused": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    style = ChipStyle.from_config(cfg)
    pre = scene_raster(project, str(cfg["scenes"]["pre"]))
    reader = Reader()
    entries = []
    try:
        for i, ex in enumerate(chosen, start=1):
            img, _ = render_single(reader, pre, float(ex["x"]), float(ex["y"]), style)
            answer = "yes" if ex["kind"] == "yes" else "no"
            name = f"{answer}_{i}.jpg"
            write_jpeg(img, out_dir / name, style.quality)
            entries.append({"file": name, "caption": EXAMPLE_CAPTIONS[ex["kind"]]})
    finally:
        reader.close()
    manifest.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return {"examples": len(entries)}


# --- pilot allocation -------------------------------------------------------------------------


def design_totals(project: Project, cfg: dict[str, Any]) -> dict[str, int]:
    out = {}
    for task in core.ALLOCATED_TASKS:
        path = core.items_path(project, task)
        if path.is_file():
            items = pd.read_parquet(path)
            out[task] = int(items.drop_duplicates("stratum")["design_n"].to_numpy().sum())
        else:
            tcfg = cfg["tasks"][task]
            out[task] = int(tcfg["design_n"]) + int(tcfg.get("outside_n", 0))
    return out


def write_allocation(project: Project, cfg: dict[str, Any], *, replace: bool = False
                     ) -> dict[str, Any]:
    path = core.allocation_path(project)
    if path.is_file() and not replace:
        raise AuditError(f"{path} exists; the allocation is written once")
    items = core.load_items(project, core.PILOT_TASK)
    events = core.read_events(core.responses_path(project, core.PILOT_TASK))
    pilot_ids = set(items.loc[items["pilot"], "item_id"])
    if replace:
        later = [e for e in core.replay(events).effective if e["slot"] not in pilot_ids]
        if later:
            raise AuditError("answers beyond the pilot exist; the allocation cannot change")
    timing = S.pilot_timing(events, pilot_ids)
    alloc = S.allocation_from_pilot(cfg, timing, design_totals(project, cfg))
    alloc = {"schema_version": 1, "written_utc": utc_now(),
             "items_sha256": core.sha256_file(core.items_path(project, core.PILOT_TASK)),
             **alloc}
    atomic_write_text(path, "# Final audit sizes from the timed pilot (plan 006 Step 5). "
                      "Written once by the click page or `nakkhu audit allocate`.\n"
                      + yaml.safe_dump(alloc, sort_keys=False))
    return alloc


# --- summaries --------------------------------------------------------------------------------


def _log_task(task: str) -> str:
    return core.FOLLOWUPS[task][0] if task in core.FOLLOWUPS else task


def _question(task: str) -> str:
    """The question a summary / agreement of ``task`` reads in its log."""
    return core.FOLLOWUPS[task][1] if task in core.FOLLOWUPS else task


def summarise(project: Project, task: str) -> dict[str, Any]:
    """Blind process summary: counts, timing, self-agreement. Promotes
    ``artifacts/audit/<task>_summary.json``."""
    from nakkhu.audit.estimators import cohen_kappa

    cfg = core.load_audit_config(project)
    log_task = _log_task(task)
    items = core.load_items(project, log_task)
    events = core.read_events(core.responses_path(project, log_task))
    state = core.replay(events)
    if task == core.NEARMISS_TASK:
        return _summarise_nearmiss(project, cfg, items, events, state)
    question = _question(task)
    eff = [e for e in state.effective if e["question"] == question]
    primary = [e for e in eff if not e.get("repeat")]
    repeats = {e["item_id"]: e["answer"] for e in eff if e.get("repeat")}
    first = {e["item_id"]: e["answer"] for e in primary}
    pairs = [(first[i], a) for i, a in repeats.items() if i in first]
    yes_no = [p for p in pairs if "unsure" not in p]
    ms = np.array([int(e["ms"]) for e in eff], dtype=float)
    all_answers = [e for e in events if e.get("event") == "answer" and e["question"] == question]
    sittings: dict[str, float] = {}
    for e in all_answers:
        sittings[e["sitting"]] = sittings.get(e["sitting"], 0.0) + int(e["ms"]) / 60_000
    queue = core.build_queue(items, core.load_allocation(project), log_task, cfg) \
        if task not in core.FOLLOWUPS else []
    summary = {
        "task": task, "items_sha256": core.sha256_file(core.items_path(project, log_task)),
        "queue_slots": len(queue),
        "answered": len(primary), "repeats_answered": len(repeats),
        "answers": {a: sum(e["answer"] == a for e in primary) for a in core.ANSWERS},
        "undone": sum(e.get("event") == "undo" and e.get("question") == question
                      for e in events),
        "timing": {"median_s": float(np.median(ms) / 1000) if len(ms) else None,
                   "mean_s": float(ms.mean() / 1000) if len(ms) else None,
                   "logged_min": sum(int(e["ms"]) for e in all_answers) / 60_000,
                   "sittings_min": sittings},
        "self_agreement": cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs]),
        "self_agreement_yes_no": cohen_kappa([p[0] for p in yes_no], [p[1] for p in yes_no]),
        "auditors": core.auditors_of(events, core.task_auditor(cfg, log_task)),
        "author_total_min": core.cumulative_ms(project) / 60_000,
        "cap_min": float(cfg["time"]["cap_min"]),
        "redo": core.redo_state(project, log_task),
    }
    return _write_summary(project, task, summary)


def _write_summary(project: Project, task: str, summary: dict[str, Any]) -> dict[str, Any]:
    from nakkhu.artifacts import promote

    run = create_run(project, "audit", ResolvedConfig(
        values={"command": "audit-summarise", "task": task}, source=core.AUDIT_CONFIG))
    with run.step(f"summarise-{task}"):
        path = run.path / f"{task}_summary.json"
        atomic_write_text(path, json.dumps(summary, indent=1, sort_keys=True))
        run.write_summary(summary)
    promote(path, f"audit/{task}_summary.json", project.paths, replace=True)
    return {"run": run.ref, **summary}


def _summarise_nearmiss(project: Project, cfg: dict[str, Any], items: pd.DataFrame,
                        events: list[dict[str, Any]], state: core.Replay) -> dict[str, Any]:
    """Blind process summary of sitting 1b: counts per question, timing, auditor (no
    sitting-1 comparison, no estimate)."""
    task = core.NEARMISS_TASK
    eff = state.effective
    ms = np.array([int(e["ms"]) for e in eff], dtype=float)
    answers = [e for e in events if e.get("event") == "answer"]
    queue = core.build_queue(items, None, task, cfg)
    by_q: dict[str, dict[str, int]] = {}
    for e in eff:
        by_q.setdefault(e["question"], dict.fromkeys(core.ANSWERS, 0))[e["answer"]] += 1
    summary = {
        "task": task, "items_sha256": core.sha256_file(core.items_path(project, task)),
        "queue_slots": len(queue), "answered": len(eff), "answers_by_question": by_q,
        "clicks": sum(e.get("click_m") is not None for e in eff),
        "undone": sum(e.get("event") == "undo" for e in events),
        "timing": {"median_s": float(np.median(ms) / 1000) if len(ms) else None,
                   "mean_s": float(ms.mean() / 1000) if len(ms) else None,
                   "logged_min": sum(int(e["ms"]) for e in answers) / 60_000},
        "auditors": core.auditors_of(events, core.task_auditor(cfg, task)),
        "author_total_min": core.cumulative_ms(project) / 60_000,
        "cap_min": float(cfg["time"]["cap_min"]),
    }
    return _write_summary(project, task, summary)


def _detected_items(project: Project, cfg: dict[str, Any], items: pd.DataFrame,
                    points: dict[str, tuple[float, float]] | None = None
                    ) -> dict[str, set[str]]:
    """Item ids whose point (or its override in ``points``, e.g. the clicked roof of a
    near-miss Yes) lies within 1 m of a detection OBB, per run."""
    specs = cfg["tasks"]["precision_pool"].get("runs") or []
    if not specs:
        return {}
    det = frames.load_detections(project, cfg, frames.region(project, cfg))
    xy = items[["x", "y"]].to_numpy(dtype=float).copy()
    for i, item in enumerate(items["item_id"]):
        if points and item in points:
            xy[i] = points[item]
    pts = shapely.points(xy)
    out = {}
    for run, g in det.groupby("run"):
        tree = shapely.STRtree(np.asarray(g.geometry.values, dtype=object))
        hit, _ = tree.query(pts, predicate="dwithin", distance=1.0)
        out[str(run)] = set(items["item_id"].iloc[np.unique(hit)])
    return out


def estimate(project: Project, task: str, n_boot: int = 2000) -> dict[str, Any]:
    """Source / model / change-map estimates (a journalled test access)."""
    from nakkhu.artifacts import promote
    from nakkhu.audit import estimators as E

    cfg = core.load_audit_config(project)
    log_task = _log_task(task)
    if log_task == core.P2_TASK:
        raise AuditError("the Phase-2 audit is estimated once as prereg_v9_a3:P2-audit: "
                         "`nakkhu followup reserve --test P2-audit`, then `run` on a remote "
                         "host")
    guard = require_frozen(project)
    if task == core.NEARMISS_TASK:
        raise AuditError("recall_nearmiss has no estimate of its own: run `nakkhu audit "
                         "estimate --task recall_points` (protocols v1 and v2)")
    checked = [log_task] + ([core.NEARMISS_TASK] if task == core.PILOT_TASK else [])
    for t in checked:
        if t in STRICT_TASKS:
            drift = input_drift(project, t)
            if drift:
                raise AuditError(f"inputs changed since the {t} sample was drawn "
                                 f"({', '.join(drift)}); estimates refused")
    if task == core.PILOT_TASK:
        for t in checked:          # v2 is primary: both sittings must be complete first
            require_complete(project, cfg, t)
    if log_task == "flood_trace":  # sitting 3 (and its damage follow-ups) estimated once, whole
        require_complete(project, cfg, log_task)
        require_followups(project, cfg, log_task)
    if task == "precision_microsoft":   # estimated once, whole (plan 013)
        require_complete(project, cfg, task)
    redo = {t: core.redo_state(project, t) for t in checked}    # refuses a half-made redo
    for t, header in redo.items():
        if header is not None:          # a redo is estimated once, whole, from its own answers
            require_complete(project, cfg, t)
            require_followups(project, cfg, t)
    items = core.load_items(project, log_task)
    log = core.responses_path(project, log_task)
    material = {"task": task, "items": core.sha256_file(core.items_path(project, log_task)),
                "responses": core.sha256_file(log) if log.is_file() else None,
                "audit_config": _sha(cfg), "n_boot": n_boot}
    if any(redo.values()):
        material["redo"] = {t: {**core.pass_fields(h), "mode": h["event"],
                                **({"reason": h["reason"]} if h.get("reason") else {}),
                                "archive": h["archive"],
                                "archive_sha256": core.sha256_file(core.archive_path(project, t))}
                            for t, h in redo.items() if h}
    if task == core.PILOT_TASK:
        nm_log = core.responses_path(project, core.NEARMISS_TASK)
        material.update(nearmiss_items=core.sha256_file(core.items_path(project,
                                                                        core.NEARMISS_TASK)),
                        nearmiss_responses=core.sha256_file(nm_log))
    access = record_test_access(project, f"audit/{task}", _sha(material)[:12],
                                guard["prereg_sha256"], guard["split_sha256"],
                                note="nakkhu audit estimate")
    state = core.replay(core.read_events(log))
    answers = {e["item_id"]: e["answer"] for e in state.effective
               if e["question"] == task and not e.get("repeat")}
    seed = int(cfg.get("seed") or 0)
    result: dict[str, Any] = {"task": task, "test_access": access, "material": material,
                              "answers_pass": material.get("redo") or "first pass (blind)",
                              "strata": E.stratum_table(items, answers).to_dict("records")}
    if task == "recall_points":
        result.update(_recall_protocols(project, cfg, items, answers, n_boot, seed))
    elif task == "precision_pool":
        members = pd.read_parquet(core.task_dir(project, task) / "members.parquet")
        result["estimates"] = E.precision_estimates(items, answers, members, n_boot, seed)
    elif task == "flood_trace":
        result["estimates"] = E.change_map_estimates(items, answers, n_boot, seed)
    elif task == "precision_microsoft":
        result["estimates"] = E.polygon_precision_estimates(items, answers, n_boot, seed)
    else:   # damage: among flood_trace yes items, weighted by the flood_trace design
        ft = {e["item_id"]: e["answer"] for e in state.effective
              if e["question"] == "flood_trace" and not e.get("repeat")}
        yes_items = items.loc[items["item_id"].map(ft) == "yes"]
        frame = E.answered(yes_items, answers)
        result["estimates"] = {"damage_share_of_traced": E.proportion(frame, n_boot, seed)}
    run = create_run(project, "audit", ResolvedConfig(
        values={"command": "audit-estimate", "task": task, **material},
        source=core.AUDIT_CONFIG))
    with run.step(f"estimate-{task}"):
        path = run.path / f"{task}_estimates.json"
        atomic_write_text(path, json.dumps(result, indent=1, sort_keys=True, default=float))
        run.write_summary({"task": task, "test_access_flagged": access["flagged"]})
    promote(path, f"audit/{task}_estimates.json", project.paths, replace=True)
    return {"run": run.ref, "flagged": access["flagged"]}


def _nearmiss_answers(project: Project, s1: dict[str, str]
                      ) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]],
                                 list[dict[str, Any]]]:
    """Sitting-1b effective answers: (near-miss answer per parent item with its click,
    (sitting 1, sitting 1b) answer pairs of the overlap items, the sitting-1b events)."""
    nm_items = core.load_items(project, core.NEARMISS_TASK)
    nm_events = core.read_events(core.responses_path(project, core.NEARMISS_TASK))
    nm_eff = {e["item_id"]: e for e in core.replay(nm_events).effective}
    parent_of = dict(zip(nm_items["item_id"], nm_items["parent_item"], strict=True))
    kind_of = dict(zip(nm_items["item_id"], nm_items["kind"], strict=True))
    keys = ("answer", "click_m", "click_dx_m", "click_dy_m")
    near: dict[str, dict[str, Any]] = {parent_of[i]: {k: e.get(k) for k in keys}
                                       for i, e in nm_eff.items() if kind_of[i] == "nearmiss"}
    overlap = [(s1[parent_of[i]], e["answer"]) for i, e in sorted(nm_eff.items())
               if kind_of[i] == "overlap"]
    return near, overlap, nm_events


def clicked_points(items: pd.DataFrame, near: dict[str, dict[str, Any]], v2: dict[str, str],
                   s1: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Clicked roof point (map coordinates) of every near-miss Yes that protocol v2 counts
    (a sitting-1 No/Unsure turned Yes)."""
    xy = items.set_index("item_id")[["x", "y"]]
    return {p: (float(xy.loc[p, "x"]) + float(n["click_dx_m"]),
                float(xy.loc[p, "y"]) + float(n["click_dy_m"]))
            for p, n in near.items() if v2.get(p) == "yes" and s1.get(p) != "yes"}


def _recall_auditors(cfg: dict[str, Any], rp_events: list[dict[str, Any]],
                     nm_events: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Auditors of sittings 1 and 1b; refuses answers by anyone but audit.yaml's auditor."""
    auditors = {core.PILOT_TASK: core.auditors_of(rp_events,
                                                  core.task_auditor(cfg, core.PILOT_TASK)),
                core.NEARMISS_TASK: core.auditors_of(nm_events,
                                                     core.task_auditor(cfg, core.NEARMISS_TASK))}
    for t, seen in auditors.items():
        want = core.task_auditor(cfg, t)
        if want and seen != [want]:
            raise AuditError(f"{t}: answers by {seen}, audit.yaml says {want!r}; refused")
    return auditors


def recall_v2(project: Project, cfg: dict[str, Any]
              ) -> tuple[pd.DataFrame, dict[str, str], dict[str, tuple[float, float]],
                         dict[str, Any]]:
    """(recall_points items, protocol-v2 answers at the primary radius, clicked roof point of
    each near-miss Yes, auditors): the coding of ``estimate --task recall_points`` (sitting-1
    primary slots, undo events honoured), for analyses that reuse it (G1 confirmatory)."""
    from nakkhu.audit import estimators as E

    items = core.load_items(project, core.PILOT_TASK)
    s1 = primary_answers(project, core.PILOT_TASK)
    radius = float(core.task_config(cfg, core.NEARMISS_TASK)["radius_m"])
    near, _, nm_events = _nearmiss_answers(project, s1)
    rp_events = core.read_events(core.responses_path(project, core.PILOT_TASK))
    auditors = _recall_auditors(cfg, rp_events, nm_events)
    v2 = E.combine_recall(s1, near, radius)
    return items, v2, clicked_points(items, near, v2, s1), {"auditors": auditors,
                                                            "radius_m": radius}


def _recall_protocols(project: Project, cfg: dict[str, Any], items: pd.DataFrame,
                      s1: dict[str, str], n_boot: int, seed: int) -> dict[str, Any]:
    """Recall protocols v2 (primary) and v1 with Unsure bounds, agreement and auditors
    (prereg ``audit_recall``)."""
    from nakkhu.audit import estimators as E

    tcfg = core.task_config(cfg, core.NEARMISS_TASK)
    radius = float(tcfg["radius_m"])
    sens = float(tcfg["sensitivity_radius_m"])
    sources = list(cfg["tasks"][core.PILOT_TASK]["sources"])
    near, overlap, nm_events = _nearmiss_answers(project, s1)
    v2 = E.combine_recall(s1, near, radius)
    v2_sens = E.combine_recall(s1, near, sens)
    clicked = clicked_points(items, near, v2, s1)
    det_v1 = _detected_items(project, cfg, items)
    det_v2 = _detected_items(project, cfg, items, clicked)
    rp_events = core.read_events(core.responses_path(project, core.PILOT_TASK))
    rp_state = core.replay(rp_events)
    first = {e["item_id"]: e["answer"] for e in rp_state.effective if not e.get("repeat")}
    repeats = [(first[e["item_id"]], e["answer"]) for e in rp_state.effective
               if e.get("repeat") and e["item_id"] in first]
    clicks = np.array([float(n["click_m"]) for n in near.values()
                       if n["answer"] == "yes" and n.get("click_m") is not None])
    auditors = _recall_auditors(cfg, rp_events, nm_events)
    raters = [core.task_auditor(cfg, core.PILOT_TASK), core.task_auditor(cfg, core.NEARMISS_TASK)]
    return {
        "auditors": auditors,
        "radius_m": radius, "sensitivity_radius_m": sens,
        "nearmiss": {"items": len(near),
                     "answers": {a: sum(n["answer"] == a for n in near.values())
                                 for a in core.ANSWERS},
                     "click_m_q25_q50_q75_max": ([float(v) for v in np.quantile(
                         clicks, [0.25, 0.5, 0.75, 1.0])] if len(clicks) else None)},
        "protocol_v2_primary": E.recall_with_bounds(items, v2, sources, det_v2, n_boot, seed),
        "protocol_v1_sitting1_only": E.recall_with_bounds(items, s1, sources, det_v1, n_boot,
                                                          seed),
        "protocol_v2_sensitivity_radius": E.recall_estimates(items, v2_sens, sources, det_v2,
                                                             n_boot, seed),
        "inter_rater_overlap": {"raters": raters,
                                **E.agreement([a for a, _ in overlap], [b for _, b in overlap])},
        "self_agreement_sitting1": {"rater": raters[0],
                                    **E.agreement([a for a, _ in repeats],
                                                  [b for _, b in repeats])},
    }


def _rater_answers(events: list[dict[str, Any]], question: str
                   ) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """(primary answer per item, (first, repeat) pairs) of one log, undo events honoured."""
    eff = [e for e in core.replay(events).effective if e["question"] == question]
    first = {e["item_id"]: e["answer"] for e in eff if not e.get("repeat")}
    pairs = [(first[e["item_id"]], e["answer"]) for e in eff
             if e.get("repeat") and e["item_id"] in first]
    return first, pairs


def agreement(project: Project, task: str, n_boot: int = 2000) -> dict[str, Any]:
    """Inter-rater agreement between the task's pre-declared auditor (``responses.jsonl``) and
    every remote rater (``raters/<name>/responses.jsonl``): pairwise Cohen's kappa, Fleiss'
    kappa with three or more raters (both with bootstrap CIs), and each rater's self-agreement
    on their own repeats. Blind process figures like :func:`summarise` (no source, model or
    change-map score); the primary estimates still read the pre-declared auditor only.
    Promotes ``artifacts/audit/<task>_agreement.json``."""
    from nakkhu.artifacts import promote
    from nakkhu.audit import estimators as E
    from nakkhu.audit import raters as R

    cfg = core.load_audit_config(project)
    log_task = _log_task(task)
    items = core.load_items(project, log_task)
    alloc = core.load_allocation(project)
    seed = int(cfg.get("seed") or 0)
    declared = core.task_auditor(cfg, log_task) or core.AUTHOR
    logs = {declared: core.responses_path(project, log_task)}
    for name in R.rater_names(project, log_task):
        logs[name] = R.rater_log(project, log_task, name)
    answers: dict[str, dict[str, str]] = {}
    per_rater: dict[str, Any] = {}
    material: dict[str, Any] = {"task": task,
                                "items": core.sha256_file(core.items_path(project, log_task))}
    for name, log in logs.items():
        events = core.read_events(log)
        first, pairs = _rater_answers(events, _question(task))
        if not first:
            continue
        yes_no = [(a, b) for a, b in pairs if "unsure" not in (a, b)]
        answers[name] = first
        material[f"responses_{name}"] = core.sha256_file(log)
        queue = (core.build_queue(items, alloc, log_task, cfg) if name == declared
                 else R.rater_queue(items, alloc, log_task, cfg, name))
        state = core.replay(events)
        open_slots = sum((s.slot, s.question or log_task) not in state.answers for s in queue)
        ms = [int(e["ms"]) for e in state.effective if e["question"] == _question(task)]
        per_rater[name] = {
            "role": "pre-declared (primary estimates)" if name == declared else "remote rater",
            "log": log.relative_to(project.paths.root).as_posix(),
            "answered": len(first),
            "complete": None if task in core.FOLLOWUPS else open_slots == 0,
            "answers": {a: sum(v == a for v in first.values()) for a in core.ANSWERS},
            "median_s": float(np.median(ms) / 1000) if ms else None,
            "under_3s_share": float(np.mean([m < R.FAST_MS for m in ms])) if ms else None,
            "pace_warnings": sum(e.get("event") == "pace_warning" for e in events),
            "auditors_in_log": core.auditors_of(events, declared if name == declared else name),
            "self_agreement": {
                "repeats": len(pairs),
                "all": E.kappa_ci([a for a, _ in pairs], [b for _, b in pairs], n_boot, seed),
                "yes_no": E.kappa_ci([a for a, _ in yes_no], [b for _, b in yes_no], n_boot,
                                     seed)},
        }
    if declared not in answers:
        per_rater.setdefault(declared, {"role": "pre-declared (primary estimates)",
                                        "answered": 0})
    result = {"task": task, "pre_declared_rater": declared, "material": material,
              "n_boot": n_boot, "raters": per_rater,
              "inter_rater": E.multi_rater_agreement(answers, n_boot, seed),
              "note": ("unweighted sample agreement on items both (all) raters answered; the "
                       "primary estimators use the pre-declared rater only")}
    run = create_run(project, "audit", ResolvedConfig(
        values={"command": "audit-agreement", **material, "n_boot": n_boot},
        source=core.AUDIT_CONFIG))
    with run.step(f"agreement-{task}"):
        path = run.path / f"{task}_agreement.json"
        atomic_write_text(path, json.dumps(result, indent=1, sort_keys=True, default=float))
        run.write_summary({"task": task, "raters": sorted(answers)})
    promote(path, f"audit/{task}_agreement.json", project.paths, replace=True)
    return {"run": run.ref, **result}


def status(project: Project) -> dict[str, Any]:
    cfg = core.load_audit_config(project)
    alloc = core.load_allocation(project)
    out: dict[str, Any] = {"allocation": bool(alloc),
                           "author_total_min": round(core.cumulative_ms(project) / 60_000, 1),
                           "cap_min": cfg["time"]["cap_min"], "tasks": {}}
    for task in core.SERVE_TASKS:
        path = core.items_path(project, task)
        if not path.is_file():
            out["tasks"][task] = "not sampled"
            continue
        items = pd.read_parquet(path)
        try:
            queue = core.build_queue(items, alloc, task, cfg)
        except AuditError as error:
            out["tasks"][task] = f"sampled ({len(items)} items); {error}"
            continue
        state = core.replay(core.read_events(core.responses_path(project, task)))
        done = sum((s.slot, s.question or task) in state.answers for s in queue)
        chips = core.chip_dir(project, task)
        have = sum((chips / core.chip_name(task, s.item_id)).is_file() for s in queue)
        out["tasks"][task] = {"items": len(items), "queue": len(queue), "answered": done,
                              "chips_ready": have}
        remote = _rater_status(project, cfg, alloc, task, items)
        if remote:
            out["tasks"][task]["raters"] = remote
        archive = core.archive_path(project, task)
        if archive.is_file():
            block = _redo_status(project, task, queue, done)
            out["tasks"][task][block.pop("mode")] = block     # "redo" or "restart"
    return out


def _rater_status(project: Project, cfg: dict[str, Any], alloc: dict[str, Any] | None,
                  task: str, items: pd.DataFrame) -> dict[str, Any]:
    """Invited and answering remote raters of a task: token id, answered / queue slots."""
    from nakkhu.audit import raters as R

    out: dict[str, Any] = {}
    invites = {r: i for (t, r), i in R.load_invites(project, task).items()}
    for name in sorted(set(invites) | set(R.rater_names(project, task))):
        inv = invites.get(name)
        row: dict[str, Any] = {"token_id": inv.token_id if inv else None}
        try:
            queue = R.rater_queue(items, alloc, task, cfg, name)
            state = core.replay(core.read_events(R.rater_log(project, task, name)))
            row.update(queue=len(queue), answered=sum(
                (s.slot, s.question or task) in state.answers for s in queue))
        except AuditError as error:
            row["error"] = str(error)
        out[name] = row
    return out


def _redo_status(project: Project, task: str, queue: list[core.Slot], done: int
                 ) -> dict[str, Any]:
    """Redo/restart progress and the archived pass (answered slots only: no answer
    distribution). ``mode`` is the header event (``redo``/``restart``; ``redo`` if unknown)."""
    archive = core.archive_path(project, task)
    prior = core.replay(core.read_events(archive))
    out: dict[str, Any] = {
        "mode": "redo",
        "archive": str(archive.relative_to(project.paths.root).as_posix()),
        "archive_answered": sum((s.slot, s.question or task) in prior.answers for s in queue)}
    try:
        header = core.redo_state(project, task)
    except AuditError as error:
        return {**out, "state": f"inconsistent: {error}"}
    if header is None:
        return {**out, "state": "archive without a redo or restart log"}
    out.update(mode=header["event"], **core.pass_fields(header), started_utc=header["utc"],
               state="complete" if done == len(queue) else "in progress")
    if header.get("reason"):
        out["reason"] = header["reason"]
    return out
