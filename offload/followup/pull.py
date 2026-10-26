"""Merge a pulled copy of remote run directories into nakkhu/runs/ (laptop). Run from nakkhu/:

    uv run python offload/followup/pull.py PULLED_DIR

PULLED_DIR holds ``runs/followup/<run>/...`` as tarred from a remote ``root/``. Only files that
are new are copied; a file that already exists locally must be byte-identical (the reserved
config, env and receipt), otherwise the pull stops. Nothing local is deleted or overwritten.
Shard JSONL files are the exception: a pulled shard file may extend the local one (append-only
growth while a shard is still running), never shrink or change it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(pulled: Path) -> int:
    src = pulled / "runs" / "followup"
    if not src.is_dir():
        sys.exit(f"{src} missing")
    new = grown = same = 0
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(pulled)
        t = ROOT / rel
        data = f.read_bytes()
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
            new += 1
            continue
        old = t.read_bytes()
        if old == data:
            same += 1
        elif "shards" in rel.parts and rel.suffix == ".jsonl" and data.startswith(old):
            t.write_bytes(data)
            grown += 1
        elif "shards" in rel.parts and rel.name.startswith("status_"):
            t.write_bytes(data)                  # shard status snapshots are refreshed
            grown += 1
        else:
            sys.exit(f"refused: {rel} differs from the local copy")
    print(f"new {new}, grown {grown}, identical {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
