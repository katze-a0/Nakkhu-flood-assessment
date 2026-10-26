"""``nakkhu audit ...`` subcommands (plan 006)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project

TASK_CHOICES = ("recall_points", "precision_pool", "flood_trace", "recall_nearmiss",
                "precision_microsoft", "flood_trace_p2")
SUMMARY_CHOICES = (*TASK_CHOICES, "damage", "damage_p2")


def _errors() -> tuple[type[BaseException], ...]:
    from nakkhu.audit.core import AuditError
    from nakkhu.evaluate.guard import GuardError

    return AuditError, GuardError


def _print(value: Any) -> None:
    print(json.dumps(value, indent=1, default=str))


def _sample(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.workflow import sample_task

    try:
        result = sample_task(project, args.task, replace=args.replace, check=args.check)
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print(result)
    return 1 if args.check and not result.get("identical") else 0


def _chips(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.workflow import chips_task

    try:
        _print(chips_task(project, args.task, force=args.force))
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _serve(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit import core
    from nakkhu.audit.server import HOST, Session, attach, bind_server
    from nakkhu.evaluate.guard import require_frozen

    server = None
    try:
        require_frozen(project)
        if _drift_refused(project, args.task):
            return 1
        cfg = core.load_audit_config(project)
        if args.reason and not args.restart:
            raise core.AuditError("--reason goes with --restart")
        server = bind_server(args.port)       # before any file moves: a live server refuses
        if args.redo:
            core.start_redo(project, args.task, core.task_auditor(cfg, args.task))
        elif args.restart:
            core.start_restart(project, args.task, core.task_auditor(cfg, args.task),
                               args.reason)
        session = Session(project, args.task, cfg)
        attach(server, session)
    except (*_errors(), OSError) as error:
        if server is not None:
            server.server_close()
        print(f"error: {error}", file=sys.stderr)
        return 1
    state = session.state()
    if session.redo and session.show_prior:
        print(f"REDO of {core.pass_of(session.redo)}: the first pass is kept unchanged in "
              f"{core.archive_path(project, args.task)}; each image shows its previous "
              "answer.", flush=True)
    elif session.redo:
        print(f"RESTART of {core.pass_of(session.redo)} (blind): the first pass is kept "
              f"unchanged in {core.archive_path(project, args.task)} and is not shown or "
              "used; start again from the first image.", flush=True)
    print(f"Audit '{args.task}': {state['done']} of {state['total']} answered.", flush=True)
    print(f"Open http://{HOST}:{server.server_address[1]}/ in your browser.", flush=True)
    print("Answers are saved as you go. Press Ctrl+C here to stop; the same command resumes.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("stopped")
    return 0


def _invite(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit import core
    from nakkhu.audit.raters import invite

    try:
        token, rec = invite(project, core.load_audit_config(project), args.task, args.rater,
                            replace=args.replace, revoke=args.revoke)
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if token is None:
        print(f"Revoked the {args.task} invite of {args.rater} (token id {rec['supersedes']}).")
        return 0
    print(f"Invited {args.rater} to {args.task} (token id {rec['token_id']}; run {rec['run']}).")
    if rec["supersedes"]:
        print(f"The previous token (id {rec['supersedes']}) no longer works.")
    print("Token (shown once, stored only as a hash; do not paste it into logs or notes):")
    print(f"  {token}")
    print(f"Rater link: <public base URL from `nakkhu audit share --task {args.task}`>/r/{token}/")
    return 0


def _share(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.share import share
    from nakkhu.evaluate.guard import require_frozen

    try:
        require_frozen(project)
        if _drift_refused(project, args.task, strict=True):
            return 1
        if args.no_tunnel:
            from nakkhu.audit.share import serve_only

            return serve_only(project, args.task, args.port)
        return share(project, args.task, args.port, cloudflared=args.cloudflared)
    except (*_errors(), OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _drift_refused(project: Project, task: str, strict: bool = False) -> bool:
    """Print and return True if the task's inputs changed since sampling and serving must stop
    (any drift when ``strict``)."""
    from nakkhu.audit import core
    from nakkhu.audit.workflow import STRICT_TASKS, input_drift

    drift = input_drift(project, task)
    if not drift:
        return False
    if strict or not core.any_answers(project, task) or task in STRICT_TASKS:
        print(f"error: inputs changed since the sample was drawn ({', '.join(drift)}); "
              f"redraw with `nakkhu audit sample --task {task} --replace` and "
              f"`nakkhu audit chips --task {task} --force`", file=sys.stderr)
        return True
    print(f"warning: inputs changed after answers began ({', '.join(drift)}); the "
          "frozen sample is kept", file=sys.stderr)
    return False


def _agreement(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.workflow import agreement

    try:
        _print(agreement(project, args.task, n_boot=args.n_boot))
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _allocate(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.core import load_audit_config
    from nakkhu.audit.workflow import write_allocation

    try:
        _print(write_allocation(project, load_audit_config(project), replace=args.replace))
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _summarise(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.workflow import summarise

    try:
        _print(summarise(project, args.task))
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _estimate(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.workflow import estimate

    try:
        _print(estimate(project, args.task, n_boot=args.n_boot))
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _status(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.workflow import status

    try:
        _print(status(project))
    except _errors() as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _locations(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.audit.locations import locations_run

    try:
        run, summary = locations_run(project, args.task)
    except (*_errors(), ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print(summary)
    print(f"run {run.ref}")
    return 0


def register(commands: Any) -> None:
    audit = commands.add_parser("audit", help="author yes/no audit (plan 006; local only)")
    sub = audit.add_subparsers(dest="audit_command", required=True, metavar="ACTION")

    p = sub.add_parser("sample", help="build the frame and draw the frozen stratified sample")
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--replace", action="store_true", help="redraw (only before any answer)")
    p.add_argument("--check", action="store_true",
                   help="rebuild with the same seed and compare with audit/<task>/items.parquet")
    p.set_defaults(func=_sample)

    p = sub.add_parser("chips", help="render local chips under data/interim/audit/<task>/")
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--force", action="store_true", help="re-render existing chips")
    p.set_defaults(func=_chips)

    p = sub.add_parser("serve", help="serve the click page on 127.0.0.1 (resumes)")
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--port", type=int, default=8765)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--redo", action="store_true",
                      help="redo the whole sitting: move responses.jsonl to responses_pass1.jsonl "
                      "once (never overwritten), serve the same queue from scratch and show each "
                      "previous answer (non-blind, recorded); resumes a redo in progress")
    mode.add_argument("--restart", action="store_true",
                      help="restart the whole sitting blind (pass 1 void): move responses.jsonl "
                      "to responses_pass1.jsonl once (never overwritten), serve the same queue "
                      "from scratch with no previous answer shown; needs --reason; resumes a "
                      "restart in progress")
    p.add_argument("--reason", help="why pass 1 is void (--restart; recorded in the log header)")
    p.set_defaults(func=_serve)

    p = sub.add_parser("invite", help="issue a remote rater's secret token for a task "
                       "(printed once; stored hashed in a runs/audit-invite run)")
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--rater", required=True, help="rater id, e.g. rater_2 ([a-z][a-z0-9_])")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--replace", action="store_true",
                        help="issue a new token for an invited rater (the old one stops working; "
                        "order and answers are kept)")
    action.add_argument("--revoke", action="store_true", help="withdraw the rater's token")
    p.set_defaults(func=_invite)

    p = sub.add_parser("share", help="serve the invited raters on 127.0.0.1 through a "
                       "cloudflared quick tunnel (token required; Ctrl+C stops both)")
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--cloudflared", help="path to cloudflared (default: on PATH)")
    p.add_argument("--no-tunnel", action="store_true",
                   help="start only the token-gated rater server on 127.0.0.1 (you run your own "
                   "tunnel to it)")
    p.set_defaults(func=_share)

    p = sub.add_parser("agreement", help="inter-rater (Cohen, Fleiss) and per-rater self-"
                       "agreement kappa with CIs; promotes artifacts/audit/<task>_agreement.json")
    p.add_argument("--task", required=True, choices=SUMMARY_CHOICES)
    p.add_argument("--n-boot", type=int, default=2000)
    p.set_defaults(func=_agreement)

    p = sub.add_parser("allocate", help="write audit/allocation.yaml from the timed pilot")
    p.add_argument("--replace", action="store_true",
                   help="rewrite (only while no answer beyond the pilot exists)")
    p.set_defaults(func=_allocate)

    p = sub.add_parser("summarise", help="blind process summary; promotes "
                       "artifacts/audit/<task>_summary.json")
    p.add_argument("--task", required=True, choices=SUMMARY_CHOICES)
    p.set_defaults(func=_summarise)

    p = sub.add_parser("estimate", help="source/model/change-map estimates (test access, "
                       "journalled)")
    p.add_argument("--task", required=True, choices=SUMMARY_CHOICES)
    p.add_argument("--n-boot", type=int, default=2000)
    p.set_defaults(func=_estimate)

    p = sub.add_parser("status", help="items, queue, answers and author time per task")
    p.set_defaults(func=_status)

    p = sub.add_parser("locations", help="answer-free item locations (item_id, x, y, stratum) "
                       "of the frozen sample into an audit-locations run; never reads answers")
    p.add_argument("--task", required=True, choices=TASK_CHOICES)
    p.set_defaults(func=_locations)
