"""Answer-free item location layer of an audit task (``nakkhu audit locations``).

A column subset of the frozen sample ``audit/<task>/items.parquet``: item id, x, y (EPSG:32645
metres, the sample's work CRS) and stratum. It never opens ``responses*.jsonl``, so it holds no
answer and is not a test access. The run records the SHA-256 of the items file and the
``items_sha256`` of ``sample.json``; the file is promoted with ``nakkhu artifacts promote``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nakkhu.config import Project, ResolvedConfig

STAGE = "audit-locations"
COLUMNS = ["item_id", "x", "y", "stratum"]
CRS = "EPSG:32645"
#: columns that would carry an answer or an answer-derived value; refused in the output
ANSWER_WORDS = ("answer", "response", "yes", "unsure", "trace", "damage", "label")


def location_table(items: pd.DataFrame) -> pd.DataFrame:
    """The answer-free subset of an items table, sorted by item id."""
    missing = [c for c in COLUMNS if c not in items.columns]
    if missing:
        raise ValueError("items table lacks " + ", ".join(missing))
    if not items["item_id"].is_unique:
        raise ValueError("item ids are not unique")
    out = items[COLUMNS].sort_values("item_id").reset_index(drop=True)
    bad = [c for c in out.columns if any(w in c.lower() for w in ANSWER_WORDS)]
    if bad:
        raise ValueError("answer-like column(s) in the location layer: " + ", ".join(bad))
    return out


def locations_run(project: Project, task: str) -> tuple[Any, dict[str, Any]]:
    """Write ``<task>_locations.csv`` and a summary into a new ``audit-locations`` run."""
    import json

    from nakkhu.audit import core
    from nakkhu.evaluate.guard import sha256_file
    from nakkhu.runs import create_run

    items_path = core.items_path(project, task)
    sample_path = items_path.parent / "sample.json"
    items_sha = sha256_file(items_path)
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.is_file() else {}
    if sample.get("items_sha256") not in (None, items_sha):
        raise ValueError(f"{items_path} does not match sample.json items_sha256")
    rel = items_path.relative_to(project.paths.root).as_posix()
    run = create_run(project, STAGE, ResolvedConfig(
        values={"command": "audit-locations", "task": task, "source": rel,
                "source_sha256": items_sha, "columns": COLUMNS, "crs": CRS}, source="cli"))
    with run.step("locations"):
        table = location_table(pd.read_parquet(items_path))
        name = f"{task}_locations.csv"
        table.to_csv(run.path / name, index=False)
        summary = {"task": task, "source": rel, "source_sha256": items_sha,
                   "sample_items_sha256": sample.get("items_sha256"),
                   "sample_run": sample.get("run"), "columns": COLUMNS, "crs": CRS,
                   "rows": len(table), "by_stratum": table["stratum"].value_counts()
                   .sort_index().to_dict(), "file": name,
                   "output_sha256": sha256_file(run.path / name),
                   "answers_read": False}
        run.write_summary(summary)
    return run, summary
