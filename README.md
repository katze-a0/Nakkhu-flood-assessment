# nakkhu

Run from this directory.

| Purpose | Command |
|---|---|
| Install | `uv sync` |
| Help | `uv run nakkhu --help` |
| Verify data manifest | `uv run nakkhu data verify` |
| Fetch input data (plan 002; sources in `configs/data/sources.yaml`) | `uv run nakkhu data fetch <id>... \| --all [--dry-run] [--force]` |
| Freeze inputs (snapshot, licences, coverage into `artifacts/data/`) | `uv run nakkhu data freeze` |
| Rebuild the derived AOI files into `runs/aoi/` and compare with `configs/aoi/` (`configs/aoi/build.yaml`; never overwrites) | `uv run nakkhu aoi build [--only <product>...] [--upstream rebuilt\|registered]` |
| Write the tile grid into `configs/tiling.yaml` (plan 003) | `uv run nakkhu tile --write-grid` |
| Mosaic (0.5 m) and tiles of a scene | `uv run nakkhu tile --scene <id> --purpose train\|infer [--rebuild-mosaic]` |
| Tile summary (promotes `artifacts/tiling/tile_summary.json`) | `uv run nakkhu tile --summary` |
| Georeference and de-duplicate an infer run | `uv run nakkhu georef --run <infer-run>` |
| HAND sweep (plan 008; `configs/hand/sweep.yaml`) | `uv run nakkhu hand sweep` |
| One HAND configuration | `uv run nakkhu hand run --config configs/hand/primary.yaml` |
| Building HAND + tier stability | `uv run nakkhu hand attribute --inventory <id>\|all [--sweep hand/<run_id>]` |
| Per-building HAND tier table (`building_id, x, y, hand_m, tier`, EPSG:32645 centroids; primary HAND column and `tiers_m` from the sweep config; read from a hand-attribute run, same ids as `hand attribute`; stops unless the tier counts equal the attribute run and the promoted raw counts; descriptive, no test access; promote with `artifacts promote`) | `uv run nakkhu hand tier-export --inventory <id>... [--attribute hand-attribute/<run_id>...]` |
| Build label arms (plan 004; promotes `artifacts/labels/*`; pending arms reported) | `uv run nakkhu labels build --arm all\|L1 L5 ... [--workers N]` |
| Register OSM per tile to the imagery (before any scoring; `labelqa.aligned_osm` for plan 005) | `uv run nakkhu labels osm-register` |
| Row-house convention vs aligned OSM (train domain only) | `uv run nakkhu labels convention` |
| L7 admit fraction on val-split tiles | `uv run nakkhu labels setpoint [--write]` |
| Tile gate scores | `uv run nakkhu labels gate` |
| G1: gate score vs label quality vs aligned OSM (dev: train split; `--splits test` needs the frozen prereg and is journalled) | `uv run nakkhu labels g1 --arm L5 --splits train` |
| G1 confirmatory rho vs the recall audit (prereg `falsifiers.G1.confirmatory`; reads the primary G1 run's tiles and gate scores; one journalled test access, refuses a second; reported, not gating) | `uv run nakkhu labels g1-confirmatory --primary labels/<g1-test-run> [--dry-run]` |
| SAM2 prompts / local run / merge (`sam` group) | `uv run nakkhu labels sam2-prompts`; `uv run --group sam nakkhu labels sam2-run --part 0/1 [--tiles-file F]`; `uv run nakkhu labels sam2-merge` |
| SAM2 offload bundle (Kaggle, 2x T4) | `uv run python offload/sam2_refine/make_bundle.py` |
| SAM2 delta bundle (tiles whose prompts changed since a bundle; merge prefers delta results) | `uv run nakkhu labels sam2-delta --base <bundle-name>` |
| Label contact sheet (local only) | `uv run nakkhu labels contact-sheet --arm L5 [--all-train]` |
| Spatial-block split (plan 005; writes `configs/splits/split_v1.yaml`, promotes `artifacts/splits/*`) | `uv run nakkhu split build [--check]` |
| Freeze / check the pre-registration (`configs/prereg.yaml` → `artifacts/prereg/<version>.sha256`, `history.json`, `changelog.md`) | `uv run nakkhu prereg freeze [--replace] [--changelog F]`; `uv run nakkhu prereg verify` |
| One audited amendment after test access (plan 013; previous file hash-checked, changed paths only under `--allow` and outside the guard's locked sections, refused once a pending analysis is journalled; changelog appended, history records the diff) | `uv run nakkhu prereg freeze --amend --previous-file F --allow PREFIX... --changelog F` |
| Score a run vs aligned OSM (val: picks the F1-max threshold; test: frozen prereg only, journalled in `runs/eval/test_journal.jsonl`) | `uv run nakkhu eval --run <stage>/<id> --split val\|test [--threshold-from eval/<id>] [--fixed]` |
| Paired block bootstrap of two eval runs | `uv run nakkhu eval --compare eval/<a> eval/<b>` |
| Author audit (plan 006; `configs/audit/audit.yaml`): frozen sample into `audit/<task>/items.parquet`, local chips (`recall_nearmiss` = sitting 1b, drawn from the finished sitting 1) | `uv run nakkhu audit sample --task recall_points\|precision_pool\|precision_microsoft\|flood_trace\|recall_nearmiss [--check]`; `uv run nakkhu audit chips --task <t>` |
| Audit click page on 127.0.0.1 (resumes; answers in `audit/<task>/responses.jsonl`; pilot writes `audit/allocation.yaml`) | `uv run nakkhu audit serve --task <t>` |
| Redo a whole sitting (first pass moved once to `audit/<task>/responses_pass1.jsonl`; same queue from scratch; each slot shows its previous answer, non-blind and recorded; resumes) | `uv run nakkhu audit serve --task <t> --redo` |
| Restart a whole sitting blind (pass 1 void: moved once to `audit/<task>/responses_pass1.jsonl`, never shown or estimated; same queue from scratch; header records `restart_of`, `prior_answer_shown: false` and the reason; resumes) | `uv run nakkhu audit serve --task <t> --restart --reason "<why pass 1 is void>"` |
| Audit status / blind summary (promotes `artifacts/audit/<task>_summary.json`) | `uv run nakkhu audit status`; `uv run nakkhu audit summarise --task <t>\|damage` |
| Answer-free item locations of a frozen sample (`item_id, x, y, stratum`, EPSG:32645; never reads answers; run records the items SHA-256; promote the CSV with `artifacts promote`) | `uv run nakkhu audit locations --task <t>` |
| Audit estimates (source/model/change map; a journalled test access; `recall_points` reports recall protocols v2 and v1 and needs sitting 1b complete) | `uv run nakkhu audit estimate --task <t>\|damage` |
| Flood-trace / damage change map (plan 009; `configs/flood/change.yaml`; promotes `artifacts/flood/*` per audit inventory with a hash sidecar; 0500 gated) | `uv run nakkhu flood change [--inventory <id>...] [--no-promote]` |
| Event validation V1-V4, H2-H6 (journalled test access; refuses until the change-map prereg, sitting 3 and the plan 007 inventory exist) | `uv run nakkhu flood validate --inventory final\|all\|<id>... [--check]` |
| Re-run only the V4 competitor table of a validate run (plans/competitor_auc_bug.md; same inventories, references, n_boot, seed; one journalled test access per source run, reason in the note; refuses if prereg/split hashes differ from the source's journal entry or an input changed after it; checks untouched rows are byte-identical; new `validation-competitors` run, promotes nothing) | `uv run nakkhu flood validate --rerun-competitors validation/<id> --reason "<why>" [--dry-run]` |
| Plan 014 follow-up tests (prereg addendum `prereg_v9_a3`; pins in `configs/followup/prep.yaml`): label-free prep (v9 HAND tables pinned, distance / join checks, HAND at the 16 thresholds, main-river HAND, P1-6 features, team HAND reproduction and coverage, UNOSAT integrity; promotes `artifacts/followup/prep_*`); one declared Phase-1 test, reserved on the laptop and executed remotely (`reserve` refuses unless prereg verify passes, `runs/eval/test_hold.yaml` is off and names the addendum, the test never ran, the prep is intact and the code is a clean commit; it journals `prereg_v9_a3:<test>` and writes a receipt; `run --receipt` executes only on a remote host from a bundle (`offload/followup/`), verifies the receipt against the shipped journal copy, never writes the journal; `promote` / `merge` (P1-6 shards) verify the pulled run against receipt and live journal, then promote `artifacts/followup/p1_*`) | `uv run nakkhu followup prep`; `uv run nakkhu followup reserve --test P1-3 [--dry-run]`; remote: `nakkhu followup run --test P1-3 --receipt <receipt.json>` (P1-6: `--shard i/n --jobs k`); `uv run nakkhu followup promote --test P1-3`; `uv run nakkhu followup merge --test P1-6` |
| Main-river HAND recompute (`configs/hand/main_river_v1.yaml`, `configs/mainriver/recompute_v1.yaml`; plans/recompute_main_river.md): label-free ensemble surfaces, building tables, tier ranges and tier export (remote host; promote on the laptop); label-free input check with answers withheld (remote); the exposure (plan 010 spec) and event (plan 009 V2-V4, H2-H6, UNOSAT, team recipe own coverage) analyses reserved on the laptop (journal `main_river_v1:<analysis>` + receipt), executed remotely from a bundle (`offload/mainriver/`), promoted to `artifacts/mainriver/` after verification | `nakkhu mainriver surfaces`; `uv run nakkhu mainriver promote-surfaces --run hand-mr/<id>`; `nakkhu mainriver check-inputs --hand-run hand-mr/<id>`; `uv run nakkhu mainriver reserve --analysis exposure\|event`; remote: `nakkhu mainriver run --analysis <a> --receipt <r>`; `uv run nakkhu mainriver promote --analysis <a>` |
| Thinned ROC / PR curves of a validate run (RDP, endpoints kept, <= 1,000 points per curve; reads only that run's curve files, no test access; promotes `artifacts/validation/*_thin.csv`) | `uv run nakkhu flood thin-curves --run validation/<id>` |
| Uncertainty U1-U3, D4 (plan 010; spec `configs/uncertainty/uncertainty.yaml`, frozen in prereg v5): prerequisites / spec (no test access); the one journalled run (refuses unless prereg verify passes, `runs/eval/test_hold.yaml` says `hold: false` + `by:`, it never ran and every input exists; promotes `artifacts/uncertainty/*`) | `uv run nakkhu uncertainty check`; `uv run nakkhu uncertainty spec`; `uv run nakkhu uncertainty run --dry-run`; `uv run nakkhu uncertainty run` |
| Detector training set from an arm (plan 007; train + val tiles only) | `uv run --group train nakkhu dataset build --arm L5 [--task obb\|detect]` |
| Train one (arm, model, seed) unit (offloaded GPU; `configs/train/`) | `uv run --group train nakkhu train --config configs/train/yolo11n-obb.yaml --arm L5 --seed 0` |
| Val inference windows (stride 512, CPU) / predict with a unit (offloaded GPU) | `uv run nakkhu infer --cut-val-windows`; `uv run --group train nakkhu infer --run train/L5/yolo11n-obb/seed0 --region val\|train\|test` |
| Offload bundle (private Kaggle data + code datasets) and launches | `uv run python offload/detector/make_bundle.py [--kind code --base B]`; `uv run python offload/detector/remote.py kaggle-dataset\|kaggle-push\|colab-start\|tick B ...` |
| Ingest pulled units (SHA-256 check, georef, val eval) / L8 round / val tables | `uv run nakkhu detector ingest <pulled dir>`; `uv run nakkhu detector selftrain --train-preds infer/<id> --val-eval eval/<id>`; `uv run nakkhu detector summarise [--promote]` |
| Model size and logged predict timings (params / GFLOPs at 640 from seed-0 best.pt on CPU; val / test-region ms per window, mean ± SD over seeds 0-2, read from infer runs; promotes `artifacts/detector/model_efficiency.csv`) | `uv run --group train nakkhu detector efficiency [--seeds 0 1 2] [--no-promote]` |
| List runs | `uv run nakkhu runs list [--stage STAGE]` |
| Promote a run output | `uv run nakkhu artifacts promote <stage>/<run_id>/<file> <artifact-path> [--replace]` |
| List artifacts | `uv run nakkhu artifacts list` |
| Override config | `uv run nakkhu --set paths.data_root=E:/nakkhu-data data verify` |
| Tests | `uv run pytest` |
| Lint | `uv run ruff check src tests` |
| Types | `uv run ty check` |

## Licence

Code: MIT (`LICENSE`). Trained model weights will be released under CC BY-NC 4.0.
