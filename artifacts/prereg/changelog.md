# Pre-registration changelog

| Version | SHA-256 of `configs/prereg.yaml` | Frozen | Test journal at freeze |
|---|---|---|---|
| prereg_v1 | `472c9213111d023aecb7becfae6fac675cdc8b22bd69a69b66be7f208b59fff6` | 2026-10-05, run `prereg/20261005T160809Z-61353c73` | 0 entries |
| prereg_v2 | `aac2bf59826a46827467e5116e1b32d6aa8e455be81cc05749db8a5864e53a95` | 2026-10-06, run `prereg/20261005T193247Z-51ed7e40` | 0 entries |
| prereg_v3 | `09685cbf703f61d2268e3ce256436afa7d0b6651b6734c4515aba280d68e709a` | 2026-10-06, run `prereg/20261005T205921Z-80a8cac6` | 0 entries |
| prereg_v4 | `52cc20e9a4d262a8d6d4a6a66bfd5142c81dc2e6152d5d7fc28f865265bcdde4` | 2026-10-12, run `prereg/20261012T164634Z-542b2335` | 0 entries |
| prereg_v5 | `65acaa4a4a9b034209fe1deeb62326fef9079931bcd38bf0a65b3d5aae2f8196` | 2026-10-13, run `prereg/20261012T182649Z-e2659dd5` | 0 entries |
| prereg_v6 | `56793d7f2b9eecccc71aa5b52a06c000463431d36ddf831f2cd726693ea1a1f3` | 2026-10-13, run `prereg/20261012T192301Z-4a470523` | 0 entries |
| prereg_v7 | see `prereg_v7.sha256` | 2026-10-13 (this freeze) | 0 entries |

prereg_v2 is the re-freeze of v1 that was planned before any test access ("G1 primary" in the
Decisions: "Refreeze prereg v1 -> `--replace` once plan 004's registration is final; record
both hashes"). No test tile, audit estimate or G1 test statistic was computed under v1:
`runs/eval/test_journal.jsonl` does not exist (0 entries). The split is unchanged
(`split_v1.yaml`, `52786e81…09ef`). The v1 hash file stays in `artifacts/prereg/prereg_v1.sha256`.

## Changes v1 -> v2 and why

1. **G1 primary** (`falsifiers.G1`). Primary rho is computed only on held-out test tiles whose
   OSM registration succeeded (`reg_ok`). Label quality is the per-tile F1 of the L6 (= L7)
   labels against aligned, greenhouse-free OSM as rotated rectangles, at rotated-polygon IoU 0.5,
   with IoU 0.3 reported too. The audit-based rho is confirmatory and does not gate. Pass is
   unchanged (rho >= 0.3). Why: Decisions "G1 primary" and decision log 2026-10-05: an
   unaligned reference measures shift, not label quality.
2. **L6 rule** (`labels.arms.L6`). Where mask/footprint IoU >= 0.5 the SAM2 geometry replaces
   the footprint. Otherwise the registered footprint is kept; it is no longer ignored. Why:
   decision log 2026-10-06. The strict ignore left 54 train tiles, and the median SAM2 vs
   footprint IoU of 0.456 does not tell which one is wrong.
3. **L7 = L6** (`labels.arms.L7`, `labels.gate` mirror). The set-point was chosen on val-split
   tiles only and resolved to admit fraction 1.0, so the gate admits every tile. This is
   recorded as a frozen fact, not as something to retune. Why: decision log 2026-10-06 (set-point
   bug fixed, still 1.0). The new mirror covers `gate.features`, `admit_candidates`,
   `admit_iou` and `admit_fraction` = 1. It replaces v1's `gate_features` mirror.
4. **Labels config mirrors** (`labels.split_section`, `labels.greenhouse_filter`). The new
   `split` and `greenhouse_filter` sections of `configs/labels/common.yaml` are mirrored. Why:
   plan 004 Outcome ("Configs changed since plan 005 froze the prereg").
5. **Greenhouse exclusion** (new `greenhouses` section; `references.r_osm.greenhouses`;
   `exposure_baseline.greenhouses`). The tags and overlap rule of
   `nakkhu.reference.greenhouse` are checked as Python constants (`GREENHOUSE_TAGS`,
   `DEFAULT_IOU` 0.3). The rule applies to labels, to the aligned-OSM reference and to exposure
   counts (`configs/hand/sweep.yaml#greenhouse_filter` is mirrored). The OSM PBF and
   `fp_osm.parquet` are pinned by SHA-256. Why: Decisions "Greenhouses are NOT buildings";
   plans 004 and 008 (Follow-up 2).
6. **HAND primary after the river-line update** (`hand`). This records the primary run
   `hand/20261005T191856Z-b3228d98` (sweep `hand/20261005T191226Z-dc9658f5`, 11 Nakkhu/Bagmati
   ways with OSM way 591536939 added). Inside the extent HAND is identical on all 63 surfaces.
   These mirrors are added: conditioning `breach_burn`, `breach`, `drop_analysis`, the full
   `burn` section (depth, all_touched, line files) and the routing-gate reference catchment.
   These files are pinned by SHA-256: both burn-line files, the analysis extent, the routing
   domain and the c4 reference catchment. Why: decision log 2026-10-06 (way 591536939, HAND
   rerun before the re-freeze); plan 008 Follow-up 3.
7. **Audit config mirrored** (new `audit` section). `configs/audit/audit.yaml` is mirrored
   except three keys:
   - `tasks.precision_pool.runs`: plan 007 detection runs, filled later.
   - `tasks.flood_trace.time_factor`: set from sitting-3 timing.
   - `tasks.recall_points.sources`: see item 8.

   `references.r_aud` now states that a greenhouse is answered No, and that estimates are
   computed only after the freeze, each as a journalled test access. Why: Decisions "Audit
   estimates wait until the final prereg refreeze; `configs/audit/audit.yaml` becomes a mirrored
   prereg section"; plan 006 Outcome.
8. **Overture** (`references.overture`, `audit.overture_rule`, `conf_threshold`,
   `exposure_baseline`). Overture is a descriptive baseline only. It is never a reference, never
   an audit stratum and never in a falsifier. Why: Decisions "Overture is a merge of
   OSM/Google/Microsoft here". Conflict: the live `audit.yaml` still lists `overture` in
   `recall_points.sources`, and `audit/recall_points/items.parquet` was drawn with Overture
   membership strata. Plan 006 must redraw before sitting 1. That is why `sources` is not
   mirrored here.
9. **Guard** (code, `src/nakkhu/evaluate/guard.py`). Each version is frozen to
   `artifacts/prereg/<version>.sha256`, and earlier versions are kept. `history.json` lists
   every freeze. This changelog is promoted with the freeze. Mirrors compare nested mappings
   recursively. The new `sha256_of` check pins files and the new `constant_of` check pins Python
   constants. A new freeze is refused once the test journal has any entry.

## Changes v2 -> v3 and why

The test journal is still empty (`runs/eval/` does not exist). The split is unchanged. Sources:
the plan 009 Outcome ("Proposed prereg v3 section"), plan 006 commit `cb77944` and the
Orchestrator decision log in `plans/README.md`.

1. **Change-map rules frozen** (`change_map`). The feature signs, the scoring rule and the class
   rule are frozen, along with not_observable as its own class and damage as evidence only.
   `configs/flood/change.yaml` is pinned by SHA-256. Mirrors are added for `score` (signs,
   required features, GMM settings), `coregistration`, `mask`, `normalise`, `buildings` and
   `sentinel2`. `nakkhu.flood.features:PIXEL_FEATURES` is checked as a constant. HAND, DEMs,
   channel lines or distances, and audit answers are forbidden as inputs. The drop rule is
   unchanged. Why: v1/v2 left the signs pending until plan 009 fixed them before any feature
   was computed. Decision log 2026-10-06: test scoring is held until the plan 009 items are
   frozen as v3.
2. **Strip order and the 0500 co-registration gate** (`change_map.strips`). GE01 0600 is used
   first, then 0500. A strip whose held-back median residual is above 1 px is not used in the
   primary, so its gap-fill buildings are not_observable there. The both-strip result
   (`*_all_strips`) is a sensitivity. On this data 0600 passes (0.38 px) and 0500 fails
   (1.03 px). Why: decision log 2026-10-06 ("GE-1 strip 0500 fails the pre-set co-registration
   gate (1.03 px) ... a sensitivity analysis includes them").
3. **Inventory** (`change_map.inventory`, `buildings` mirror). The primary is `detector_final`,
   the final detector inventory under the prereg selection rule (plan 007). The Google v3 t90
   cut is the secondary comparison. Microsoft, OSM and Google uncut are by-inventory
   sensitivity. Why: decision log 2026-10-06 ("flood validation and sitting 3 use the final
   detector inventory ... as primary; Google cut is secondary").
4. **Hash sidecars** (`change_map.hashes`). The sitting-3 sample records the inventory id,
   `change_map_sha256`, `inventory_sha256` and `buildings_sha256` from
   `change_map_buildings_<inv>.json`. Audit serve / estimate and flood validate refuse if any
   of them differs. Why: coordinator decision 2026-10-06 (plan 009 Outcome, item 4).
5. **Audit flood_trace** (`audit_flood_trace` mirror, and the v2 `audit.config` mirror updated
   to plan 006 commit `cb77944`). `inventory` is now an inventory id (`detector_final`).
   `change_map` / `change_map_meta` are per-inventory paths. The rebalance
   `{stratum: positive, n: 100, from: negative_far}` keeps 300 items in total (dry run on Google
   cut: 100 / 118 / 82). Why: decision log 2026-10-06 ("sitting 3 rebalanced within 300 items
   (positive stratum ~100, taken from far negatives) to narrow the PPV CI without adding author
   time"). Before this freeze, v2's `audit.config` mirror no longer matched the live
   `audit.yaml`, so `prereg verify` (and every guard-gated audit command) refused. v3 restores
   a clean state.

## Changes v3 -> v4 and why

The test journal is still empty (`runs/eval/test_journal.jsonl` does not exist). The split is
unchanged. No recall estimate has been run, and sitting 1b has no answer. Sources: plan 006
Outcome (follow-up 2026-10-12) and the coordinator brief of 2026-10-12.

1. **Auditor per sitting** (`audit_recall.auditors`, `auditor` keys in the `audit.config`
   mirror). Sitting 1 (`recall_points`, 350 items + 18 repeats, complete 2026-10-12) was answered
   by one teammate (`teammate_1`), not the author. Sitting 1b and sittings 2-3 are the author's.
   The auditor is set per task in `audit.yaml`, written into every response event (sitting 1 by
   one appended `auditor` event, so the raw lines stay untouched) and carried by every estimate.
   The 120 min cap counts author time only. Why: user, 2026-10-12.
2. **Sitting 1b, `recall_nearmiss`** (`audit_recall.sitting_1b`, new `tasks.recall_nearmiss`
   mirror). A sitting-1 point is a footprint's representative point. On the 71 of 180 test tiles
   where the Google/Microsoft footprints did not register they keep an offset of metres, so a
   real building's point often lands beside its roof: even in stratum GMO only 46 of 64 points
   were Yes. That biases the true-building count (contribution 2) and source accuracy low. Every
   sitting-1 item answered No or Unsure in a footprint stratum (136) is asked "Is there a
   building roof within the circle?" with a circle of radius R; a Yes needs a click on the roof,
   whose offset is logged in metres. 20 random sitting-1 items are asked the original question
   again for inter-rater agreement. All 156 are interleaved blind. Why: the recall estimates
   would otherwise measure footprint offset, not building existence.
3. **R = 8.0 m, chosen before any sitting-1b answer and from no audit answer**
   (`audit_recall.radius`). It is the 95th percentile of the Google/Microsoft footprint
   registration shift magnitude on the 109 registered test-region tiles (7.91 m, plan 004
   promoted shifts `5132e5ea...`), rounded up to 0.5 m. All 739 registered tiles give 7.65 m and
   train tiles only give 7.21 m. Cap: R is below the median nearest-neighbour spacing of Google
   split-unit points in the extent (8.24 m; Microsoft 13.78 m, OSM 12.89 m). Contiguous row
   houses (69 % of Google units touch another footprint) remain a stated limitation, with a
   4.0 m sensitivity radius (median shift 3.81 m, rounded up).
4. **Recall estimators** (`audit_recall.estimators`). Primary = protocol v2 (sitting-1 Yes, or
   sitting-1 No/Unsure and near-miss Yes within R). Secondary = protocol v1 (sitting 1 only).
   Both report Unsure excluded plus the all-Unsure-as-Yes and all-Unsure-as-No bounds. Also
   reported: v2 at 4.0 m, inter-rater Cohen's kappa and raw agreement on the 20 overlap items,
   and teammate_1 self-agreement on the 18 repeats. A near-miss Yes is tested for model recall
   at its clicked roof point. Estimates need sittings 1 and 1b complete.
5. **Pinned files** (`audit_recall.files`): sitting-1 items, sitting-1 responses (with the
   auditor event), `audit/allocation.yaml`, sitting-1b items and the footprint registration
   shifts R came from. Any later change fails `prereg verify`.

## Changes v4 -> v5 and why

The test journal is still empty (`runs/eval/test_journal.jsonl` does not exist). The split is
unchanged. No audit estimate has been run, and the sitting-2 and sitting-3 samples have not been
drawn. Why one more freeze: the plan 010 analysis spec was the last piece to pre-register before
test scoring (decision log 2026-10-12), and an audit of plans 007, 009 and 010 found other rules
that will read test tiles or test-region audit answers but were not fixed yet. All of them go
into v5, so that the test tiles are opened exactly once.

1. **Plan 010 uncertainty analysis** (new `uncertainty` section;
   `configs/uncertainty/uncertainty.yaml` pinned by SHA-256 and partly mirrored; code in
   `src/nakkhu/uncertainty/`, tested on synthetic data only).
   - U1 calibrates the detector_final existence probability from the ensemble conf, with
     temperature and isotonic fits. Fitting is leave-one-block-out over plan 005 blocks with
     design weights. The method with the lower weighted ECE (10 equal-count bins) is chosen.
     Pass: ECE <= 0.1.
   - U3 primary is a PPI++ total of real buildings in the recall frame, per tier group of the
     primary HAND surface. The answers use protocol v2 at R = 8 m. Secondary estimators: ratio
     D_t p_g / r_g, HT, naive and precision-only. Sensitivities: protocol v1, v2 at 4 m and both
     Unsure bounds. Intervals come from a Rao-Wu bootstrap (2000 replicates, seed 0). Tiers are
     pooled from answered counts only (at least 20 per group).
   - U2 is a balanced ANOVA of seed x 28 HAND surfaces x 200 audit replicates on ratio-corrected
     counts. The descriptive factorial is inventories x 84 HAND members.
   - D4 reports the tier counts per inventory and the event AUC.
2. **Split-sample miss rule** (falsifiers.C2.ppi). A miss is |A - B| > 1.96 sqrt(se_A^2 +
   se_B^2). The literal "A's CI misses B's point" rule ignores B's own sampling error. In the
   synthetic check (`tests/test_uncertainty.py`), exact intervals miss at that rule about 30 %
   of the time, so a 10 % limit would fail by construction. The combined rule misses about 6 %.
   The 10 % threshold is unchanged, and the literal rate is still reported. Fixed before any
   data.
3. **Final detector selection** (`detector_selection`). The selection rule, its val-only outcome
   (L5 + YOLO11n-OBB) and the source run are recorded, and the run file is pinned. Why:
   decision log 2026-10-12. The model rule had not been pre-registered before.
4. **detector_final** (`detector_final`). This is the plan 007 rule in `nakkhu.ensemble`
   (`nakkhu detector final`):
   - each seed at its own val threshold;
   - greedy clusters with at most one box per seed, at rotated IoU >= 0.5 (`IOU_MIN` pinned);
   - medoid geometry;
   - conf = sum of the seed confs / 5;
   - cut at the ensemble's own val F1-max threshold.

   Why: the rule decides every exposure result, sitting 3 and the V-metrics. Coordinator
   decision 2026-10-13 aligned the prereg text with the implemented code.
5. **Sitting-2 pool** (`audit_precision_pool`). The pool is six runs: the five seeds and the
   ensemble test run at its cut, each with `threshold_from` its own val eval. Why: U1, U3 and the
   U2 per-seed precision and recall all need these units. Consequence: R-AUD rows exist only for
   detector_final and the final seeds, so C1's R-AUD row is reported for L5 only.
6. **Sitting 1b responses pinned** (`audit_recall.files.recall_nearmiss_responses`). Sitting 1b
   is complete (156 of 156 answered).
7. **Event validation settings** (`validation`). The section fixes:
   - the primary reference (R1 if the drop rule keeps it, else R1-aud);
   - the primary inventory and surface;
   - plan 005 blocks as clusters (pinned);
   - the bootstrap, Moran's I, V2 and reliability defaults, the cumulative cuts and the
     likelihood edges, pinned as code constants;
   - H6: leave-one-block-out folds and the primary method by out-of-fold Brier (tie: logistic);
   - C3: evaluated once on detector_final x the primary reference x the primary surface.

   The guard now accepts dotted `constant_of` names (e.g. `f.__defaults__`), so function defaults
   can be pinned without editing plan 009 code.
8. **D5 cross-sensor protocol** (`cross_sensor`, falsifiers.C4). The scene is WV03 2023-04-04.
   The five final seeds run at their WV02 thresholds, against OSM registered to WV03 by the same
   method. The WV02 F1 is recomputed on the same tiles. Pass: seed-mean drop <= 20 pt. C4 reruns
   are defined as three CPU reruns that must stay within the five-seed 95 % interval.
9. **C5 and detector comparisons** (`falsifiers.C5`, `detector_comparisons`). L8 is deferred, so
   C5 compares the footprints with the final model. "Match" means the paired CI includes or lies
   below 0 for at least 3 of the 5 seeds. Every unit is scored once at its frozen threshold, and
   differences are claimed only under the C1 rule.
10. **G1 confirmatory rho** (`falsifiers.G1.confirmatory`). The audit-based tile quality is now
    defined, from recall units with protocol v2 within 1 m of an L6 label. It is reported and
    does not gate.
11. **Test access list** (`test_access.analyses`). This lists every analysis that reads test
    data.

## Changes v5 -> v6 and why

The test journal is still empty (`runs/eval/test_journal.jsonl` does not exist). The split is
unchanged. No audit estimate has been computed, the sitting-2 and sitting-3 samples have not been
drawn, and `nakkhu uncertainty run` has never run. **The change was motivated by synthetic
simulation only.** No audit answer, detection, test tile or estimate was used.

1. **C2.ppi split-sample check** (`falsifiers.C2.ppi`, `uncertainty.u3.settings.split_sample`,
   `configs/uncertainty/uncertainty.yaml#u3.split_sample`, re-pinned).
   - Halves are now formed at random within each audit stratum with a fixed seed (stream
     `split_sample`, seed 0; `nakkhu.uncertainty.ppi.halve`). This matches the stratified random
     design the audit was drawn under, so each half is itself a stratified random sample.
   - v5 halved the plan 005 blocks. Tier groups follow HAND, which is clustered in space, so a
     block half is a cluster subsample. The within-half SEs omit the block-assignment variance.
   - In a synthetic simulation with exact intervals (`tests/test_uncertainty_run.py` fixtures),
     the combined-SE rule missed **0.36 with block halves vs 0.07 with random halves**, so the
     frozen check would have failed by construction.
   - Unchanged: the combined-SE miss rule |A - B| > 1.96 sqrt(se_A^2 + se_B^2), the 10 % limit,
     1000 repeats, 2000 bootstrap replicates and min 2 answered units per stratum and half.
   - Newly reported: the simulated expected miss rate (same frozen settings, plug-in population)
     beside the observed one. The block split is kept as a secondary, descriptive row (observed
     and expected, never gating), so the spatial-clustering effect stays visible.
   - Why: coordinator decision 2026-10-13 (option 1 of the plan 010 Outcome).

## Changes v6 -> v7 and why

The test journal is still empty (`runs/eval/test_journal.jsonl` does not exist). The split is
unchanged. **v7 changes no analysis setting.**

1. **Duplicate top-level key fixed** (`uncertainty_propagation`).
   - v5 added the plan 010 section under the top-level key `uncertainty`, which the plan 005
     evaluation block (block bootstrap, seeds, Wilson, Wilcoxon) already used. v6 kept it.
   - YAML keeps only the last value of a duplicate key. So v5 and v6 **silently dropped the
     plan 005 `uncertainty.block_bootstrap` key** and the rest of that block from the parsed
     prereg. The plan 005 text in the file was never edited.
   - The effect: `nakkhu.evaluate.score.eval_params` raised KeyError, so every eval refused.
   - **No eval and no test access ran under the broken parse.** There is no `runs/eval/` run
     after the v5 freeze (2026-10-12T18:26Z), and the test journal does not exist.
   - v7 renames the plan 010 section to `uncertainty_propagation`. Its content is unchanged,
     apart from re-pinning `configs/uncertainty/uncertainty.yaml`, where only a comment line
     naming the section changed. The plan 005 `uncertainty` block is byte-identical to v1-v6 and
     parses again (`block_bootstrap.n_resamples` 2000, seed 0).
   - Readers were updated: `nakkhu.uncertainty.spec` / `run` and the tests.
2. **Guard** (`nakkhu.evaluate.guard`).
   - `load_prereg` now refuses a prereg with a duplicate mapping key at any level (new
     `duplicate_keys`). `prereg verify` reports it and `prereg freeze` refuses it.
   - Test: `tests/test_evaluate.py::test_duplicate_keys_refused_at_any_level`.
   - A scan of every `configs/**/*.yaml` and of this file finds no other duplicate key.

## Changes v7 -> v8 and why (audited amendment, plan 013)

| Version | SHA-256 of `configs/prereg.yaml` | Frozen | Test journal at freeze |
|---|---|---|---|
| prereg_v7 | `d31e45d69dafd709e105be03cd7782d0d51b7eebd180bd63664adbe4991d32e9` | 2026-10-13, run `prereg/20261012T193855Z-c6b09658` | 0 entries |
| prereg_v8 | `284669fc745263f32f86b0868cdff15f673618f79637510b0bfe5d5a638be059` | 2026-10-16, `nakkhu prereg freeze --amend` (run in `history.json`) | 42 entries |

**prereg_v8 is the only amendment made after test access.** It was frozen with
`nakkhu prereg freeze --amend`. The guard checked that the given copy of v7 hashes to
`d31e45d6…`, that every changed dotted path lies under an explicit `--allow` prefix, that none
lies under a locked section (`nakkhu.evaluate.guard.AMEND_LOCKED`: every section the 42 test
accesses used), and that the journal holds no pending analysis. The v7 file is kept as
`artifacts/prereg/prereg_v7.yaml`; `history.json` lists the changed paths and the allow list.

### Why

1. **C5 fallback.** C5 failed under v7 (journal entry 39, `detector/falsifiers:C1,C5`,
   `artifacts/detector/test_falsifiers.json`): Microsoft matches the final model on 5 of 5 seeds
   (Google v3 t90 on 1 of 5). v7 C5 says "adopt footprints as the inventory", but v7 still named
   detector_final as primary in its change-map, audit, validation and uncertainty sections. The
   user decided on 2026-10-16 to apply the fallback: **Microsoft footprints are the main
   inventory; detector_final is the secondary inventory.** Exposure is reported for both.
2. **Count unit** (user, 2026-10-16). A recall-frame unit is a footprint point, and one building
   can carry several points (`plans/building_unit_definition.md`). Exposure is now counted in
   real objects (Microsoft polygons, detector boxes). Totals over frame points (PPI++ / HT T_g)
   are reported only as upper bounds on buildings, labelled as such. A ratio of an object count
   to a point-level recall or completeness mixes units and is a sensitivity row, never primary.

### What had been seen when the amendment was decided

Journal entries 1-42, all under v7: the detector test scoring of 38 runs (entries 1-38), C1 +
C5 (39), C4 (40), the recall audit estimate (41) and the SAM3 zero-shot run (42). So the C5
outcome and
the audited recall and completeness were known: detector_final recall 0.548 [0.472, 0.622] and
Microsoft completeness 0.737 [0.704, 0.768] (protocol v2, Unsure excluded;
`artifacts/audit/recall_points_estimates.json`). Both are point-level quantities. **Not seen:**
no sitting-2 or sitting-3 answer exists, and no estimate of precision_pool, flood_trace or
damage, no `flood validate`, no `uncertainty run` and no G1 test run was ever journalled.

### Changes (41 dotted paths; nothing outside these)

1. `version`: prereg_v7 -> prereg_v8.
2. `change_map`: `config.sha256` re-pinned (configs/flood/change.yaml now says
   `primary_inventory: microsoft`, `audit_inventories: [microsoft, detector_final, google_v3]`;
   these keys only label and promote outputs, no computed value depends on them);
   `buildings.values` mirrors that; `inventory` prose (Microsoft primary, detector_final
   secondary, Google t90 comparison, Google uncut and OSM D4); new `detector_final_map`: the
   detector_final change map stays as built under v7 (`93c29513…`), and a rerun under the v8
   config reproduces it byte for byte (determinism check, plan 013 Step 4).
3. `audit.config.values.tasks.flood_trace.inventory` and `audit_flood_trace.values.inventory`:
   detector_final -> microsoft. New `audit_flood_trace.redraw`: the sitting-3 sample drawn on
   detector_final (items sha256 `0cf83866b1f201c0ef606a1a82eb393f7ae338a3efda04fe7e75b94c89633268`,
   run `audit/20261012T195240Z-5dbdead4`) was never answered and is replaced by a draw on the
   Microsoft change map with the same design (280 active items + 14 repeats).
4. New task `audit.config.values.tasks.precision_microsoft` and section
   `audit_precision_microsoft` (sitting 2, author): 150 Microsoft exposure polygons, strata
   agree / ms_only x channel band (6 strata), sqrt allocation, min 10, 8 repeats, about 8 extra
   author minutes. It gives the precision of a Microsoft polygon, the unit counted in Microsoft
   exposure. Appended to the task list, so no earlier sample moves; the sitting-2 detector pool
   (precision_pool) is unchanged.
5. `detector_final.uses`: secondary inventory (its event skill uses R1 under the
   Microsoft-audited drop rule, with no R1-aud and no V2); U1, the U2 primary and the detector U3
   row. The detector_final rule itself is unchanged (locked).
6. `validation.inventory` (Microsoft primary; one sitting-3 PPV and drop rule for every
   inventory) and `validation.c3` (evaluated once on Microsoft x the primary reference x the
   primary surface).
7. `uncertainty_propagation` (configs/uncertainty/uncertainty.yaml re-pinned):
   - U3 runs per inventory, `[microsoft, detector_final]`, one tier grouping for both (pooling
     now also needs the answered Microsoft precision units per tier).
   - **Primary exposure count** = the main inventory's objects per tier times their audited
     polygon precision (D_t p_g, Microsoft, precision_microsoft) with a Rao-Wu CI; the same for
     detector_final as the secondary inventory; raw D_t beside both.
   - PPI++ / HT T_g (frame points): upper bounds, labelled `upper_bound_frame_points`.
   - Ratio D_t p_g / r_g: **flagged mixed units, sensitivity only** (`sensitivity_mixed_units`).
   - U2 primary response: v7's ratio-corrected seed count D p / r mixed units, so the v8 primary
     is the unit-consistent D p (seed boxes x their audited precision). D p / r is decomposed the
     same way as a sensitivity row. Note: falsifiers.C2.decomposition refers to section u2 by
     reference, so its terrain-vs-detector finding now uses D p. It is a finding either way and
     never gates C2.
   - D4 delta % is taken against Microsoft (v7: detector_final).
   - New count_units section, mirrors of `inputs.main_inventory`, `u3.count_units`,
     `u2.primary.recall_correction` and `d4.baseline`.
   - **C2 unchanged and still on detector_final** (U1 ECE, the PPI++ split-sample check with f =
     detector_final). The same split-sample check with Microsoft as f is reported beside it and
     never gates (moving a falsifier target after C5 was seen would bend the freeze).
8. `falsifiers.C5.outcome` (new): C5 failed under v7; the fallback is applied in v8. C5's
   statement, match rule and reported rows are unchanged.
9. `exposure_baseline.microsoft` and `exposure_baseline.count_unit` (new).
10. `test_access.analyses` gains `audit estimate precision_microsoft`; new
    `test_access.amendment` states this amendment.

### Unchanged (locked by the guard)

split, seeds, references, matching, conf_threshold, detection_postprocess, metrics, the plan 005
`uncertainty` block, labels, greenhouses, hand, audit_recall (with the pinned sitting-1/1b files
and `audit/allocation.yaml`), detector_selection, the detector_final rule, audit_precision_pool,
cross_sensor, detector_comparisons and falsifiers C1, C2, C4, C5 (except `outcome`) and G1.
`falsifiers.C4.reruns` stays on detector_final: it tests the detector's reproducibility.

## Changes v8 -> v9 and why (audited amendment)

| Version | SHA-256 of `configs/prereg.yaml` | Frozen | Test journal at freeze |
|---|---|---|---|
| prereg_v8 | `284669fc745263f32f86b0868cdff15f673618f79637510b0bfe5d5a638be059` | 2026-10-16, run `prereg/20261016T092609Z-494f0549` | 42 entries |
| prereg_v9 | see `prereg_v9.sha256` and `history.json` | 2026-10-16, `nakkhu prereg freeze --amend` | 42 entries |

prereg_v9 was frozen with `nakkhu prereg freeze --amend` (user approval, 2026-10-16), with
`--allow version audit.config.values.scenes.oam`. The guard checked the v8 copy against
`284669fc…`, the locked sections and the journal (no pending analysis). The v8 file is kept as
`artifacts/prereg/prereg_v8.yaml`.

### Why

OAM Dukuchhap UAV mosaics are pre-flood imagery despite 2024-10-01 metadata
(plans/oam_panel_check.md); dropped from flood_trace chips before any flood_trace/damage
estimate; sitting 3 pass 1 void and restarted blind.

The OAM records give an acquisition of 2024-10-01 (a user-entered one-hour window). The image
content shows the pre-flood state where the 6 Oct GeoEye-1 strip shows the flood's sediment and
scour, so the drone panel was a second "before" view labelled "after". Four drone panels were also
blank (no-data stored as 0 under an all-valid mask). The decision rests on the imagery and its
metadata, not on any audit answer.

### What had been seen when the amendment was decided

Journal entries 1-42, unchanged since v8 (no entry was added under v8). Sitting-2 answers
(precision_pool, precision_microsoft) and a sitting-3 pass-1 response log (flood_trace with its
damage follow-up) exist on disk; none was estimated and no audit estimate of precision_pool, precision_microsoft, flood_trace
or damage, no `flood validate`, no `uncertainty run` and no G1 test run was journalled. Sitting-3
pass 1 was answered on the three-panel chips and is void: its responses are archived as
`responses_pass1.jsonl` by `nakkhu audit serve --restart` and never enter an estimate; sitting 3
is served again from scratch on the same queue with no prior answer shown.

### Changes (2 dotted paths; nothing outside these)

1. `version`: prereg_v8 -> prereg_v9.
2. `audit.config.values.scenes.oam`: `[oam_dukuchhap_1, oam_dukuchhap_2, oam_dukuchhap_3]` ->
   `[]` (mirrored by `configs/audit/audit.yaml` `scenes.oam`). flood_trace chips show WV02
   (before) and GE01 0600/0500 (after) only.

### Unchanged

Everything else, including every section locked by the guard, the flood_trace and damage design
(`audit_flood_trace`, its sample and allocation entry), the chip settings and the post scenes.
The chip `valid_share` check now reads the raster mask (a code fix, not a prereg setting).

## prereg_v9_a1: follow-up tests on the HAND result (addendum, plan 014), 2026-10-17

- **Base and timing.** Addendum to prereg_v9
  (89baaed3e19554db599b510219c94285880820d4b840a7a0e76478d2150253ba). Frozen after 51
  journalled test accesses, among them flood validate (journal 47, where C3 failed) and the
  competitor re-run (journal 51). configs/prereg.yaml is unchanged, and C3 stays the
  pre-registered result.
- **Why.** On 2026-10-17 the user decided to run diagnostic and improvement tests. The HAND
  investigations found a mechanism, not a bug: at the 0.2 km2 drop threshold, 41-47 % of
  buildings take their HAND from tributary heads. The amendment path is closed, because pending
  analyses are already in the journal.
- **Phase 1 (post hoc, on Nakkhu).** P1-1 to P1-8: main-river HAND, the threshold curve, the
  joint model and distance bands, the DEM comparison (GLO-30, FABDEM, GEDTM30, SRTM GL1,
  NASADEM, AW3D30, HMA 8 m), label diagnostics, the ML model, metric diagnostics, and the
  UNOSAT Spot-6 second reference. Each runs once and is journalled. The conditional P1-4-* and
  P1-5-ext follow the same rule. Open exploration is journalled as `explore`.
- **Phase 2 (confirmation).** Frozen candidates (the main-river HAND rule and the ML model) are
  tested on new sites: A = Manohara + Hanumante, B = Godawari + Kodku, pooled, Microsoft
  inventory, UNOSAT as a secondary reference. The candidates are frozen in prereg_v9_a2
  before any label exists, then tested once with Holm over H2a/H2b at one-sided alpha 0.05.
- **Mechanism.** nakkhu.evaluate.addendum (`nakkhu prereg addendum`); `prereg verify` now also
  checks addenda.

## prereg_v9_a2 supersedes prereg_v9_a1 (plan 014), 2026-10-17

- **Defect in a1.** prereg_v9_a1 (23f08ce415b72ea25db846b2ca0f43f5837125d3f5a0a22397a5a9278264cc80)
  pinned the whole of configs/hand/sweep.yaml by sha256. Minutes after the freeze, the
  label-free DEM work that a1's own P1-4 needs (SRTM GL1, NASADEM, AW3D30, HMA 8 m, geoid
  settings) edited that file, so `prereg verify` failed.
- **What a2 changes.** a2 is a1 with two changes:
  - the full-file pin is replaced by mirrors of the method keys (grid, burn, breach,
    breach_burn conditioning, statistics, buffer, tiers, drop analysis, thresholds), plus the
    routing-domain hash;
  - the ids are renumbered: the Phase-1/Phase-2 declaration becomes prereg_v9_a2, and the
    Phase-2 candidate record becomes prereg_v9_a3.
  No declared test, setting or rule changed.
- **Status of a1.** a1 had no journal entry. It stays frozen as history and is still
  integrity-checked, but its tests are refused. Journal entries at the a2 freeze: 51.
- **Tooling.** Supersession was added to nakkhu.evaluate.addendum. It is allowed only for an
  unused addendum, and only once.

## prereg_v9_a3 supersedes prereg_v9_a2 (plan 014), 2026-10-17

- **Trigger.** After prereg_v9_a2 (b14b52a1900570fff008e273e8d72ea8f7dc846f57206defc1b5767fbf26ab1e)
  was frozen, the user supplied the team's / supervisor's earlier ArcGIS HAND model of the
  Nakkhu. a2 had no journal entry. Journal entries at the a3 freeze: 51.
- **Additions in a3.**
  - P1-9: the team surface (`Hand Value.tif`) evaluated as-is. Coverage is reported first, then
    the test is scored on the covered rows, with distance, hand_v9 and hand_mr_d8 on the same
    rows.
  - P1-10: the team recipe reproduced on Dem12.5, GLO-30 and FABDEM × {fill, breach_burn}.
    Two drainage definitions: basin-clipped at a 0.78125 km2 threshold, and the digitized
    Nakkhu line. A label-free reproduction check comes first.
  - hand_team_acc (breach_burn) joins the Phase-2 HAND candidate set. Members are compared on
    the covered rows, with ties going to hand_mr_d8 first.
- **Ids and data.** The Phase-2 record is renumbered a3 -> a4. The team zips are copied
  unmodified to data/raw/ext_team_hand/ as reference-only data, with their sha256 recorded in
  the addendum.
- **Unchanged.** Every other a2 test, setting and rule.

## prereg_v9_a4 (2026-10-17): plan 014 Phase-2 record

Requires prereg_v9_a3 (709c7245...); prereg_v9 unchanged. Freezes the Phase-2 candidates (HAND rule hand_mr_d8 x GLO-30 breach_burn centroid, selected by the a3 rule on 3,920 common covered rows in explore 1, journal 64; M-C GAM refit, lambda 10, model sha256 227c1fca...), sites A (Manohara + Hanumante, 36 blocks), B (Godawari + Kodku, 23 blocks) and the negative control N (Bishnumati + Dhobi, 39 blocks) with their lines, extents, buildings, mosaics, registration and co-registration gate results (all pass), the v9 change map per site (resolved site configs pinned), the P2-audit design (task flood_trace_p2: 150 + 8 repeats per site, one remote primary rater rater_1, pace guidance for every rater, sessions <= 30 min), and the a3 hypotheses, test, Holm, size check and modes (mirrored). Frozen before any Phase-2 change-map class or audit answer exists.
