"""``P2-audit`` of prereg_v9_a3: the one estimate of the Phase-2 audit (plan 014).

Run once through the receipt flow of :mod:`nakkhu.followup.remote` (reserved on the laptop as
``prereg_v9_a3:P2-audit``, executed remotely). It reads the ``flood_trace_p2`` items and the
primary rater's answers (``tasks.flood_trace_p2.primary_rater``: a remote rater's
``raters/<name>/responses.jsonl``; else the task's ``auditor`` and ``responses.jsonl``) and
writes, per site, the v9 change-map estimates (:func:`nakkhu.audit.estimators.
change_map_estimates`: PPV on the positive stratum with its Wilson interval, design-weighted NPV,
the drop rule PPV Wilson lower bound < 0.5) and the damage share of the traced items. Remote
raters' answers are never read here (they enter only ``nakkhu audit agreement``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nakkhu.audit import core

TASK = core.P2_TASK
OUTPUTS = ("p2_audit_estimates.json",)


def primary_rater(cfg: dict[str, Any]) -> tuple[str, bool]:
    """(name, remote) of the primary rater: ``tasks.flood_trace_p2.primary_rater`` (a remote
    rater, answers in ``raters/<name>/responses.jsonl``) or else the task's ``auditor``."""
    tcfg = core.task_config(cfg, TASK)
    remote = tcfg.get("primary_rater")
    if remote:
        return str(remote), True
    return str(core.task_auditor(cfg, TASK) or core.AUTHOR), False


def primary_log(project: Any, cfg: dict[str, Any]) -> Path:
    from nakkhu.audit import raters

    name, remote = primary_rater(cfg)
    return raters.rater_log(project, TASK, name) if remote else core.responses_path(project,
                                                                                    TASK)


def primary_queue(project: Any, cfg: dict[str, Any], items: pd.DataFrame) -> list[Any]:
    from nakkhu.audit import raters

    name, remote = primary_rater(cfg)
    alloc = core.load_allocation(project)
    return (raters.rater_queue(items, alloc, TASK, cfg, name) if remote
            else core.build_queue(items, alloc, TASK, cfg))


def completeness(project: Any, cfg: dict[str, Any]) -> list[str]:
    """Every slot of the primary rater's queue and every follow-up of a Yes answered."""
    items = core.load_items(project, TASK)
    queue = primary_queue(project, cfg, items)
    state = core.replay(core.read_events(primary_log(project, cfg)))
    follow = str(core.task_config(cfg, TASK).get("followup") or "")
    out = []
    open_slots = [s for s in queue if (s.slot, s.question or TASK) not in state.answers]
    if open_slots:
        out.append(f"{TASK} is not complete: {len(open_slots)} of {len(queue)} slots "
                   "unanswered by the primary rater")
    if follow:
        missing = [s for (s, q), e in state.answers.items()
                   if q == TASK and e.get("answer") == "yes" and (s, follow) not in state.answers]
        if missing:
            out.append(f"{TASK} is not complete: {len(missing)} Yes answer(s) without the "
                       f"{follow} follow-up")
    return out


def input_files(project: Any) -> list[Path]:
    from nakkhu.audit import frames

    cfg = core.load_audit_config(project)
    tcfg = core.task_config(cfg, TASK)
    files = [core.items_path(project, TASK), primary_log(project, cfg),
             core.task_dir(project, TASK) / "sample.json",
             project.paths.resolve(str(tcfg["change_maps"]))]
    files += [v["file"] for v in frames.p2_change_maps(project, tcfg).values()]
    return files


def prerequisites(project: Any) -> list[str]:
    """Label-free refusals before the journal entry: the Phase-2 freeze, the frozen items with
    no input drift, a complete sitting (every slot and every follow-up of a Yes answered) by the
    primary rater only."""
    from nakkhu.audit import frames
    from nakkhu.audit import workflow as W

    try:
        cfg = core.load_audit_config(project)
        tcfg = core.task_config(cfg, TASK)
        frames.p2_require_freeze(project, tcfg)
        core.load_items(project, TASK)
    except core.AuditError as error:
        return [str(error)]
    problems = []
    drift = W.input_drift(project, TASK)
    if drift:
        problems.append(f"inputs changed since the {TASK} sample ({', '.join(drift)})")
    problems += completeness(project, cfg)
    name, _ = primary_rater(cfg)
    log = primary_log(project, cfg)
    seen = core.auditors_of(core.read_events(log), name)
    if seen not in ([name], []):
        problems.append(f"{TASK}: {log.name} holds answers of {seen}, not only the primary "
                        f"rater {name}")
    return problems


def score(project: Any, spec: dict[str, Any], *, n_boot: int, seed: int
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    from nakkhu.audit import estimators as E

    cfg = core.load_audit_config(project)
    tcfg = core.task_config(cfg, TASK)
    items = core.load_items(project, TASK)
    log = primary_log(project, cfg)
    state = core.replay(core.read_events(log))
    answers = {e["item_id"]: e["answer"] for e in state.effective
               if e["question"] == TASK and not e.get("repeat")}
    damage = {e["item_id"]: e["answer"] for e in state.effective
              if e["question"] == str(tcfg.get("followup", "damage")) and not e.get("repeat")}
    sites: dict[str, Any] = {}
    for site in tcfg["sites"]:
        sub = items.loc[items["site"].astype(str) == str(site)].copy()
        sub["stratum"] = sub["stratum_in_site"].astype(str)
        est = E.change_map_estimates(sub, answers, n_boot, seed)
        traced = sub.loc[sub["item_id"].map(answers) == "yes"]
        frame = E.answered(traced, damage)
        sites[str(site)] = {**est,
                            "strata": E.stratum_table(sub, answers).to_dict("records"),
                            "damage_share_of_traced": E.proportion(frame, n_boot, seed)
                            if len(frame) else None,
                            "items": int(len(sub))}
    name, remote = primary_rater(cfg)
    out = {"task": TASK, "primary_rater": name, "primary_remote": remote,
           "primary_log": log.relative_to(project.paths.root).as_posix(),
           "items_sha256": core.sha256_file(core.items_path(project, TASK)),
           "responses_sha256": core.sha256_file(log),
           "rule": "per site, the change map is dropped if the PPV Wilson 95 % lower bound is "
                   "< 0.5 (v9 drop rule)",
           "sites": sites, "n_boot": int(n_boot), "seed": int(seed),
           "reporting": "prereg_v9_a3 P2-audit (primary rater only)"}
    info = {s: {"drop_change_map": v["drop_change_map"]} for s, v in sites.items()}
    return {"p2_audit_estimates.json": out}, info


def write_output(path: Path, value: Any) -> None:
    from nakkhu.flood.validate import jsonable

    if isinstance(value, pd.DataFrame):
        value.to_csv(path, index=False, lineterminator="\n")
    else:
        path.write_text(json.dumps(jsonable(value), indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")

