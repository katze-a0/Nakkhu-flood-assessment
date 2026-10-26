"""D3 zero-shot baseline import (plan 007 Phase C): SAM3 "building" detections from Kaggle.

``nakkhu detector zeroshot <pulled dir>`` reads every ``**/out/<region>_<variant>/`` written by
``offload/sam3/sam3_infer.py`` (``detections_px.parquet`` + ``inputs.json``), copies it into a
new ``runs/infer/<id>/`` (SHA-256 recorded), georeferences it with the frozen post-processing
(centre ownership + cross-tile NMS, ``nakkhu georef``) and, for val only, scores it like a
detector (``nakkhu eval --split val``: F1-max threshold against aligned OSM). The variant with
the higher val F1@0.5 (ties: ``full``, the simpler one) is the D3 SAM3 row; only its test run
enters the guarded one-shot test plan. Test runs are inference only here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from nakkhu.config import Project, ResolvedConfig

FILES = ("detections_px.parquet", "inputs.json")
VARIANT_ORDER = ("full", "quad")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_dir(project: Project, src: Path) -> dict[str, Any]:
    from nakkhu.georef import georef_run
    from nakkhu.runs import atomic_write_json, create_run

    inputs = json.loads((src / "inputs.json").read_text(encoding="utf-8"))
    if not inputs.get("zeroshot"):
        raise ValueError(f"{src}: not a zero-shot output")
    shas = {f: _sha(src / f) for f in FILES}
    values = {"command": "detector zeroshot", "model": inputs["model"],
              "region": inputs["region"], "source": src.as_posix(), "sha256": shas}
    run = create_run(project, "infer", ResolvedConfig(values=values, source="nakkhu.zeroshot"))
    with run.step("import"):
        for f in FILES:
            shutil.copy2(src / f, run.path / f)
            if _sha(run.path / f) != shas[f]:
                raise RuntimeError(f"{f}: copy differs")
        atomic_write_json(run.path / "imported.json", values)
    georef_run(project, run.ref)
    run.write_summary({k: inputs.get(k) for k in ("model", "region", "detections",
                                                   "ms_per_tile", "variant", "prompt")})
    return {"run": run.ref, **inputs}


def cli_zeroshot(project: Project, args: Any) -> int:
    from nakkhu.detector import load_registry, save_registry
    from nakkhu.evaluate.score import evaluate_run

    reg = load_registry(project)
    zs = reg.setdefault("zeroshot", {})
    for source in args.source:
        for d in sorted(Path(source).rglob("out/*_*")):
            if not (d / "inputs.json").is_file() or not (d / "detections_px.parquet").is_file():
                continue
            inputs = json.loads((d / "inputs.json").read_text(encoding="utf-8"))
            model, region = inputs["model"], inputs["region"]
            entry = zs.setdefault(model, {"variant": inputs.get("variant")})
            if entry.get(f"infer_{region}"):
                continue
            res = import_dir(project, d)
            entry[f"infer_{region}"] = res["run"]
            entry[f"ms_per_tile_{region}"] = inputs.get("ms_per_tile")
            entry[f"capped_{region}"] = inputs.get("inputs_capped_at_num_queries")
            if region == "val":
                ev = evaluate_run(project, res["run"], "val")
                entry.update(eval_val=ev["eval_run"], val_f1_50=ev["f1_50"],
                             val_threshold=ev["threshold"])
            save_registry(project, reg)
            print(json.dumps({"model": model, "region": region, "run": res["run"],
                              "val_f1_50": entry.get("val_f1_50")}), flush=True)
    best = select_variant(zs)
    if best:
        zs["selected"] = {"model": best, "rule": "higher val F1@0.5 vs aligned OSM; "
                          "ties: full"}
        save_registry(project, reg)
        print(json.dumps({"selected": best}))
    return 0


def select_variant(zs: dict[str, Any]) -> str | None:
    """Model key with the highest val F1@0.5; ties go to the earlier of ``VARIANT_ORDER``."""
    def rank(v: dict[str, Any]) -> int:
        var = v.get("variant")
        return -VARIANT_ORDER.index(var) if var in VARIANT_ORDER else -len(VARIANT_ORDER)

    scored = [(float(v["val_f1_50"]), rank(v), k) for k, v in zs.items()
              if k != "selected" and v.get("val_f1_50") is not None]
    return max(scored)[2] if scored else None
