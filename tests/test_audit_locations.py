"""``nakkhu audit locations``: answer-free column subset of a frozen sample (synthetic)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nakkhu.audit import locations as L
from nakkhu.evaluate.guard import sha256_file


def _items() -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": ["ft0003", "ft0001", "ft0002"], "task": "flood_trace",
        "stratum": ["positive", "negative_far", "negative_near"],
        "x": [331_000.5, 332_000.0, 333_000.25], "y": [3_055_000.0, 3_056_000.0, 3_057_000.0],
        "building_id": ["microsoft:1", "microsoft:2", "microsoft:3"],
        "change_class": ["positive", "negative", "negative"], "pi_design": [0.3, 0.02, 0.04]})


def test_location_table_keeps_only_id_xy_stratum():
    out = L.location_table(_items())
    assert list(out.columns) == ["item_id", "x", "y", "stratum"]
    assert out["item_id"].tolist() == ["ft0001", "ft0002", "ft0003"]
    assert out.loc[out["item_id"] == "ft0003", "x"].item() == 331_000.5


def test_location_table_refuses_duplicates_and_missing_columns():
    dup = pd.concat([_items(), _items().iloc[:1]])
    with pytest.raises(ValueError, match="not unique"):
        L.location_table(dup)
    with pytest.raises(ValueError, match="lacks"):
        L.location_table(_items().drop(columns="stratum"))


def test_locations_run_records_source_hash_and_reads_no_answers(project):
    task_dir = project.paths.root / "audit" / "flood_trace"
    task_dir.mkdir(parents=True)
    items = task_dir / "items.parquet"
    _items().to_parquet(items, index=False)
    sha = sha256_file(items)
    (task_dir / "sample.json").write_text(json.dumps({"items_sha256": sha, "run": "audit/x"}),
                                          encoding="utf-8")
    (task_dir / "responses.jsonl").write_text("not json: must never be opened\n",
                                              encoding="utf-8")
    run, summary = L.locations_run(project, "flood_trace")
    assert summary["source_sha256"] == sha and summary["sample_items_sha256"] == sha
    assert summary["rows"] == 3 and summary["answers_read"] is False
    assert summary["by_stratum"] == {"negative_far": 1, "negative_near": 1, "positive": 1}
    out = pd.read_csv(run.path / "flood_trace_locations.csv")
    assert list(out.columns) == ["item_id", "x", "y", "stratum"]
    assert summary["output_sha256"] == sha256_file(run.path / "flood_trace_locations.csv")
    assert run.status == "complete"
    # a changed items file no longer matches the frozen sample
    _items().iloc[:2].to_parquet(items, index=False)
    with pytest.raises(ValueError, match="does not match"):
        L.locations_run(project, "flood_trace")
