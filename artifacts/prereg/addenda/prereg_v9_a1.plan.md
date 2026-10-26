# Plan 014: Follow-up tests on the HAND result (prereg addendum prereg_v9_a1)

> **Executor instructions**: Read `AGENTS.md`, `plans/README.md` (decision log 2026-10-17),
> `plans/investigation_hand.md`, `plans/investigation_eval.md`, `plans/investigation_prior.md`,
> `plans/competitor_auc_bug.md`, plans 008 and 009, and this plan. Standing rules: `uv` only,
> never `pip`; never invent numbers; heavy runs (HAND sweeps, nested CV, bootstraps) go through
> the `offload-experiments` skill (VPS `vps.brihatech.com` is CPU only; Kaggle is optional);
> `archive/` is read-only; delete nothing; local commits in `nakkhu/` only, never push. Every
> command runs from `nakkhu/`.
>
> **The machine-readable declaration is `nakkhu/configs/prereg_addenda/prereg_v9_a1.yaml`.**
> Its frozen copy and this plan's frozen copy are in `nakkhu/artifacts/prereg/addenda/`. Where
> this text and the YAML differ, the YAML governs. Code reads every setting from the YAML.
>
> **Drift check (run first)**: `uv run nakkhu prereg verify` shows `problems: []`, prereg
> `89baaed3…` and `addenda: {prereg_v9_a1: <hash>}`; `runs/eval/test_journal.jsonl` has no entry
> whose note starts with `prereg_v9_a1:` other than the ones this plan's Outcome lists. If an
> undeclared entry exists, STOP.

## Status

- **State**: frozen, not run (2026-10-17). Addendum frozen with `nakkhu prereg addendum`
  (journal 51 at the freeze; prereg v9 unchanged). No Phase-1 or Phase-2 access has run.
- **Priority**: P0 (decides how the paper reports HAND)
- **Effort**: L. Phase 1: about 2-3 days code and tests, then CPU runs. Phase 2: one more
  pre-event strip for site A, then one author audit sitting (316 items for two sites).
- **Risk**: HIGH for over-claiming. Phase 1 re-uses labels that are already seen, so it can
  explain and develop, never confirm.
- **Depends on**: 005 (guard), 008 (HAND sweep), 009 (change map, validate), 013 (inventory)
- **Planned at**: 2026-10-17

## Why This Matters

The pre-registered test (C3, prereg v9) failed. On Microsoft R1 (n 6,769, 352 affected, 26
blocks), HAND scored 0.887 and distance to the channel 0.895: ΔAUC −0.008 [−0.044, 0.023]
(`artifacts/validation/competitors_auc_v2.csv`). HAND does beat plain elevation, ASF HAND and
the 100 m buffer. The user and supervisor expected HAND to clearly beat distance. On
2026-10-17 the user decided: "run any test that will let us know what is wrong and let us
improve", with ample compute.

The investigations found no bug, but a mechanism. At the 0.2 km² drop threshold, 41 % of
Microsoft buildings and 47 % of detector buildings take HAND from a tributary head, not from
the Nakkhu/Bagmati. Distance, by contrast, is measured to the river that flooded. In the
logged H3 grid, HAND rises to 0.908-0.915 at ≥ 0.5-2 km². That figure is a post-hoc reading
of the test set. Three other facts matter. The test has low power (26 blocks; a difference of
about ±0.035 cannot be detected). The labels mostly record deposition near the bank. The DSM
sits 2.7 m (median) above FABDEM at buildings.

C3 stays the reported pre-registered result. This plan declares, before any run:
- **Phase 1**: diagnostic and development tests on Nakkhu, all post hoc.
- **Phase 2**: one confirmatory test of frozen candidates on a new site, run once.

## What is frozen, and how

- The guard's amendment path (`prereg freeze --amend`) cannot be used. It refuses once a
  pending analysis is in the journal, and flood validate (journal 47) is. An amendment would
  also rewrite the rules already used. So this plan added a minimal **addendum** mechanism
  (`src/nakkhu/evaluate/addendum.py`, `nakkhu prereg addendum`):
  - an addendum is a separate YAML in `configs/prereg_addenda/`, named
    `prereg_v<N>_a<M>`;
  - it names its base prereg and that prereg's hash, and must match the frozen one;
  - it is frozen into `artifacts/prereg/addenda/` as `<id>.sha256`, a frozen YAML copy, the
    frozen plan text and `index.json`; the changelog is appended to `artifacts/prereg/changelog.md`;
  - it is immutable: a change needs a new id;
  - `prereg verify` checks every frozen addendum, including its pins, and test scoring then
    refuses as it does for the prereg.
- **Test access**: every Phase-1 or Phase-2 run that reads a flood label or an audit answer
  goes through `nakkhu.evaluate.addendum.record_addendum_access(project, "prereg_v9_a1", <test>,
  run_ref, cfghash)`. That call:
  - verifies the prereg and every addendum;
  - refuses a test that is not declared, or one that has already run;
  - writes one journal entry with the note `prereg_v9_a1:<test>` and `addendum_sha256`.
  Each test runs **once**. A single retry (`retry=True`, note `… retry`) is allowed only for a
  run that failed before writing any metric, and it is disclosed.
- **Label-free work** (new HAND surfaces, features, drain shares, site checks) is `access: none`
  and is not journalled.

## Shared settings (all tests; YAML `common`)

| Item | Setting |
|---|---|
| Headline cell | Microsoft × R1, the exact rows of the v9 primary cell (n 6,769, 352 affected, 26 blocks) |
| Secondary | detector_final × R1; Microsoft × R1-aud (design weights, as v9) |
| Orientation | lower HAND / distance / elevation = more exposed (score = −value); models give probabilities |
| Metric | ROC-AUC (`flood.validate.roc_auc`, ties ½, weighted for R1-aud); PR-AUC also reported |
| Paired test | ΔAUC on identical rows: DeLong (unweighted R1 only) and paired block bootstrap (`block_bootstrap_delta_auc`) |
| CI | 95 % percentile, 2,000 resamples of whole plan-005 blocks, seed 0 (the C3 method; defaults pinned) |
| Multiplicity | none in Phase 1 (all exploratory); Holm in Phase 2 |
| Distance competitor | distance from the building centroid to `osm_nakkhu_bagmati_lines.geojson` (the v9 `distance_to_channel`) |
| Small cells | a band or subgroup is scored only with ≥ 10 affected and ≥ 10 unaffected; otherwise counts only |
| Labelling | every Phase-1 number is reported as "post hoc (prereg_v9_a1 P1-n)" and never replaces C3 |

### Derived predictors (label-free, YAML `definitions`)

- **Main mask M**: cells of the sweep grid touched (`all_touched`) by the 11 OSM Nakkhu/Bagmati
  ways, with D8 contributing area ≥ 1 km² on the same DEM × conditioning. The area condition
  drops the bank cells of the 2-cell burn band.
- **hand_mr_d8** (main-river HAND): `max(0, z_ref(c) − z_ref(m(c)))`. Here m(c) is the first
  cell of M on the breach_burn D8 path from c, and z_ref is the conditioned DEM with the burn
  undone (as in the sweep). The value is NaN if the path never reaches M.
- **hand_mr_rem** (relative elevation model): `max(0, z_raw(c) − z_ch(c))`. Here z_ch is the
  minimum z_raw over the M cells within 60 m of the M cell nearest to c.
- **hand_any_t**: the sweep HAND at stream threshold t (nearest drainage of any kind).
  **dist_stream_t**: distance to the nearest stream cell at t. **drains_main_t**: whether the
  first stream cell on the D8 path at t lies in M.
- **dist_cell**: the distance competitor measured from the 30 m HAND cell centre
  (resolution-matched).

## Phase 1: diagnostics and development on Nakkhu (post hoc, exploratory)

Nakkhu is now development data. Every declared test below reads the R1 classes, and some read
the sitting-3 answers. Each is therefore **one journalled test access, run once**.

| ID | Question | Predictors / model | Main output (`artifacts/followup/`) |
|---|---|---|---|
| P1-1 | Does HAND referenced to the flooding river beat distance? | hand_mr_d8, hand_mr_rem × GLO-30, FABDEM vs distance and vs hand_v9; R1, R1-aud, R1-all-strips; tier hit rates + ORs of hand_mr_d8 (v9 tiers and monotone rule) | `p1_1_main_river.csv`, `p1_1_tiers.csv` |
| P1-2 | How does skill move with the stream threshold, and is the gap about height or about which channel? | 16 thresholds 0.05-5 km² (the drop candidates), GLO-30 breach_burn: AUC of hand_any_t and dist_stream_t; ΔAUC vs distance, vs dist_stream_t (drainage-matched), vs hand_mr_d8 (nearest any drainage vs the flooding river); AUCs within drains_main_t True/False | `p1_2_threshold_curve.csv` |
| P1-3 | Does HAND add information beyond distance? | Leave-one-block-out logistic: M0 = ns(log1p dist, df 4); M1 = M0 + ns(hand_v9, 4); M1mr = M0 + ns(hand_mr_d8, 4). OOF AUC, Δlog-loss, ΔBrier, paired block bootstrap; CR1 Wald χ² (df 4) of the HAND terms. Distance bands [0,100), [100,300), ≥ 300 m: AUC and ΔAUC of hand_v9 and hand_mr_d8, stratified AUC; mirror = distance inside HAND bands ≤ 3, (3,12], > 12 m | `p1_3_joint.csv`, `p1_3_bands.csv` |
| P1-4 | Does the DEM matter at matched thresholds? | GLO-30, FABDEM, GEDTM30 (routing-gate fail, flagged), SRTM GL1, NASADEM, AW3D30 and NASA HMA 8 m, each × {0.2, 0.5, 1, 2 km², the DEM's drop threshold, hand_mr_d8, hand_mr_rem}, breach_burn, EGM2008; ΔAUC vs distance and vs GLO-30 at the same rule. The four extra DEMs are being fetched and routed label-free by another agent. GLO-30, SRTM, NASADEM, AW3D30 and HMA 8 m are surface models (HMA 8 m is a 2002-2016 VHR-stereo DSM, buildings included); FABDEM and GEDTM30 are DTMs. HMA enters resampled onto the sweep grid, with a native-8 m surface as an extra row. Its no-data buildings are counted, and all predictors are also scored on its valid rows | `p1_4_dems.csv` |
| P1-4-* | The same protocol for a DEM that is not ready when P1-4 runs | `srtmgl1`, `nasadem`, `aw3d30`, `hma8m` (only if late), `tandemx12` (DLR proposal), `uav` (Khatiwada 2026, if shared; on its covered buildings only). Each runs once when available | `p1_4_<dem>.csv` |
| P1-5 | How much of the tie comes from the labels? | (a) On the answered sitting-3 Microsoft items: AUCs against the audit answer vs against the R1 class of the same items (unweighted and weighted). (b) Change-map misses: audit Yes vs No among audited negatives. (c) R1 share by distance band and HAND tier (Wilson). (d) Spearman of the continuous change-map score with HAND and with distance, overall and per band | `p1_5_labels.csv` |
| P1-8 | Is the tie an artefact of our labels? A second, independent reference | UNOSAT FL20240928NPL Spot-6, 30 Sep 2024 (HDX package 3996, CC BY-SA). UNOSAT says it is not field validated, so it is a second reference, not truth. R-UN = the footprint intersects the flood extent; R-UN-pts = an affected-building point lies inside or within 8 m of the footprint. Rows: all Microsoft buildings of the extent inside the UNOSAT analysis extent, and the v9 R1 rows. Output: AUC/ΔAUC of hand_v9, hand_mr_d8 and distance; κ and the 2×2 table with R1; AUCs against R1 within R-UN = 1 and R-UN = 0. Caveat: whether UNOSAT used a DEM mask is unverified | `p1_8_unosat.csv` |
| P1-5-ext | A further independent reference (conditional) | Only for a reference not derived from HAND, a DEM or a hydraulic model, whose licence permits analysis (e.g. TASA FORMOSAT-5 once its licence is confirmed). Sentinel-1 is excluded: the 27 Sep pass precedes the peak and the 2 Oct pass follows the recession. Protocol as P1-8 | `p1_5_ext.csv` |
| P1-6 | Can a learned exposure model beat distance? | See below | `p1_6_models.csv`, `p1_6_importance.csv` |
| P1-7 | Is the metric hiding skill? | Standardised pAUC (FPR ≤ 0.1, McClish), PR-AUC with paired bootstrap, resolution-matched distance (dist_cell), and TOST of hand_v9 vs distance at ±0.03 (equivalent if the 90 % bootstrap CI lies inside) | `p1_7_metrics.csv` |

**P1-6, the ML model** (YAML `tests.P1-6`):
- **Label**: Microsoft R1, binary and unweighted. R1-aud is never used for training.
- **Features** (label-free; 28 in total):
  - HAND: hand_v9; hand_any at 0.05, 0.1, 0.25, 0.5, 1 and 2 km²; hand_mr_d8 and hand_mr_rem
    (GLO-30); hand_mr_d8 on FABDEM.
  - Distance: log1p of the distance competitor, log1p dist_stream at 0.2 and 2 km², and the D8
    flow-path length to the main river.
  - Elevation: GLO-30, FABDEM, and GLO-30 − FABDEM.
  - Terrain: slope; plan and profile curvature; TPI at 150 m and 500 m; TWI.
  - Flow: log contributing area at the building cell, at the first drainage cell reached and
    at the main-river cell reached.
  - Building: footprint area and centroid density within 100 m.
  - Excluded: coordinates, block id, and any imagery, change-map or audit quantity.
- **Families**:
  - M-C: penalised logistic GAM (ns df 4 per feature, L2 penalty λ ∈ {0.001, 0.01, 0.1, 1,
    10, 100}).
  - M-D: scikit-learn `HistGradientBoostingClassifier`, early stopping off. Random search
    over 60 configurations (seed 0) on learning_rate × max_iter × max_leaf_nodes ×
    min_samples_leaf × l2_regularization × max_features (grid in the YAML).
  - M-E: M-D with monotone constraints (non-increasing in every HAND and distance feature).
- **CV**: outer leave-one-block-out (26 folds). Inner 5-fold `GroupKFold` by block, selected
  by mean inner log-loss, then refit on the outer-training rows. detector_final rows of a
  held-out block are predicted by that fold's Microsoft-trained model. No tuning outside this
  budget.
- **Report**: pooled OOF AUC, PR-AUC, log-loss, Brier and ECE; ΔAUC vs distance, vs hand_v9
  and vs hand_mr_d8 (paired block bootstrap, DeLong); grouped permutation importance within the
  held-out block (20 repeats, descriptive).

**P1-explore (not a declared test).** Nakkhu is development data, so open exploration is
allowed after P1-1 to P1-8. Each such run is journalled with
`guard.record_test_access(..., note="prereg_v9_a1:explore <n> <description>")`. Every one is
listed, with its count, in the Phase-1 report. None is ever called a declared test. None can
become a Phase-2 candidate unless prereg_v9_a2 declares it.

### Power note (Phase 1)

The logged C3 interval [−0.044, 0.023] (26 blocks) gives a block-bootstrap SE of about 0.017.
The following assumes the SE scales as 1/√(blocks); it is an approximation, not a guarantee.

- **Cannot detect**: any |ΔAUC| below about 0.035 (the 95 % half-width) is indistinguishable
  from 0 on Nakkhu. That includes the logged grid gaps of main-stem HAND over distance (about
  +0.013 to +0.020). 80 % power at two-sided 5 % needs |ΔAUC| ≳ 0.048.
- **Can detect**: large effects, such as a ΔAUC of ±0.05 or more; HAND vs elevation or vs ASF
  HAND (already shown); and label-free mechanisms (drain shares). The CR1 Wald test of P1-3
  uses the likelihood and may be more sensitive than ΔAUC. With only 26 clusters it can be
  anti-conservative, so it is reported with that caveat.
- **Weaker still**: within-band, subgroup and R1-aud analyses. R1-aud has 264 items, 22 blocks
  and a CI width of about 0.14. Only large differences show there.
- **What Phase 1 can deliver**: direction, magnitude with CIs, consistency across inventories
  and references, and a mechanism. It cannot deliver a claim that HAND beats distance.

## Phase 2: confirmation on a new site (the only confirmatory part)

**Candidates (selection rule declared now, YAML `phase2.candidates`):**
1. **Frozen HAND rule**: the main-river variant with the highest P1-1 AUC on Microsoft R1 among
   {hand_mr_d8, hand_mr_rem} × {GLO-30, FABDEM}. Any variant within 0.005 of the best counts as
   tied; ties prefer GLO-30, then hand_mr_d8.
2. **Frozen ML model**: the P1-6 family with the highest pooled OOF AUC; ties within 0.005
   prefer M-C, then M-E, then M-D. It is refit on all Nakkhu Microsoft R1 rows, with its
   configuration chosen by 5-fold GroupKFold under the same budget. It is never recalibrated.
3. Both go to Phase 2 whatever their Phase-1 result (no gate). Any extra candidate must be
   declared in prereg_v9_a2 and joins the Holm family.

**Freeze before any Phase-2 label.** `configs/prereg_addenda/prereg_v9_a2.yaml` has `requires:
[prereg_v9_a1]` and records:
- the site extent(s) and the main-line set;
- imagery ids, blocks and inventory;
- the HAND candidate's config and surface hashes;
- the ML model file and feature-spec hashes, and the nakkhu commit.

It is frozen with `nakkhu prereg addendum` before any change-map class or audit answer for the
site exists. A site with any earlier label is ineligible.

**Site-selection criteria (the site itself is not chosen here):**
- riverine flooding in the 26-28 Sept 2024 event is documented in a public source, and the
  source is recorded;
- the site lies outside the Nakkhu extent and its 320 m test buffer;
- pre (≤ 25 Sept 2024) and post (28 Sept-15 Oct 2024) VHR imagery exists at ≤ 0.75 m, under a
  licence that allows analysis, and it passes the v9 co-registration gate;
- Microsoft footprints cover the site, and its river is mapped in OSM as `waterway=river`;
- a routing domain built by the plan 008 rule passes the routing gate. Sites inside the
  upper-Bagmati domain reuse the frozen grid; sites outside it (e.g. Roshi) need a new domain
  built by the same rule.

Site definitions:
- **Extent**: every 960 m block of the plan-005 lattice (extended) whose centre lies within
  1,000 m of the main-river lines, along the whole river length inside the imagery overlap. It
  is never trimmed by visible flood traces.
- **Main lines**: the site river plus the river it joins inside the extent (the Nakkhu +
  Bagmati rule). The distance competitor uses the same lines.
- Every qualifying site is used and pooled. Candidates named by the user: Kodku, Godawari,
  Bagmati upstream, Hanumante, Roshi, Bishnumati.

**Declared sites** (from `plans/followup_data_check.md`, label-free metadata; each must still
pass every criterion before a2):
- GE01 `105001003E1F0600` (6 Oct 2024) is the only usable post image; 0500 stays gated out as
  in v9. Both sites lie inside the existing routing domain.
- **Site A: Manohara + Hanumante.** Pre-event images: WV02 `10300100F5AB8000`, plus the open
  east strip `10300100F2BCCF00`, which still has to be downloaded. About 14.8k Microsoft
  buildings lie within 500 m of the rivers, and the data check estimates about 290 change-map
  positives (uncertain by at least ±50 %). The Bhaktapur part is covered only by 0500, so it
  is out.
- **Site B: Godawari + Kodku.** All imagery is already held. About 15.0k buildings and about
  225 estimated positives.
- **Excluded.**
  - The Bagmati main stem is the development river. Its segments inside a site extent count
    only as that site's main lines.
  - Bishnumati/Dhobi had almost no mapped flood, so it is a negative control only: counts and
    predicted exposure are reported, an AUC only if min_cell holds, and no hypothesis is tested.
  - Roshi, Balkhu and Nallu lie outside the post image.
- **Inventory**: Microsoft only. 29 % / 58 % of the site corridors lie in the detector training
  domain, so detector_final is not scored in Phase 2.
- **Pooling**: the primary test pools every site that passes its gates. Blocks are nested in
  sites, and the bootstrap is stratified by site. Per-site rows are secondary and carry no
  decision, so the Holm family stays {H2a, H2b}.
- **A site that fails** co-registration, routing or its audit drop rule leaves the pool, with
  the reason reported. The size check then applies to the remaining site(s).

**Labels.**
- R1 comes from the v9 change map, unchanged (`configs/flood/change.yaml` is pinned in the
  YAML). Site parameters go in a separate site file frozen in a2.
- The change-map code imports no HAND, DEM or river module (test-enforced), so label
  construction is blind to the predictors.
- A V1 audit is required (test `P2-audit`, journalled once): the v9 flood_trace protocol and
  strata, 150 active items + 8 repeats per site, in one blind sitting, imagery only. For
  scale, sitting 3 took 7.1 min for 280 items. The v9 drop rule applies per site (a PPV Wilson
  lower bound < 0.5 drops that site's change map).
- Secondary reference: UNOSAT Spot-6 (R-UN and R-UN-pts, as in P1-8) on the same sites. It is
  reported, with no decision.

**Size check and mode.** These are computed inside the P2 run before any predictor is read.
- **Confirmatory**: ≥ 26 blocks with an observable building, ≥ 150 R1 affected, and the change
  map kept by the audit.
- **Descriptive**: otherwise. The run then reports point estimates and CIs as "underpowered /
  no valid reference, not confirmatory", and no hypothesis decision.
- The MDE of the actual set is computed from its block count and written before scoring.

**Hypotheses (one-sided), test and decision.**
- H2a: AUC(frozen HAND) − AUC(distance) > 0. H2b: AUC(frozen ML) − AUC(distance) > 0.
- Paired block bootstrap: 2,000 resamples of whole blocks stratified by site, seed 0,
  p = (1 + #{Δ* ≤ 0}) / 2,001.
- Holm at family-wise one-sided α = 0.05: reject the smaller p if p ≤ 0.025, then the other if
  p ≤ 0.05. A missing candidate counts as p = 1.
- Reported: ΔAUC with the 95 % CI and the one-sided 97.5 % lower bound, and DeLong.
  Secondary rows, with no decision: v9 hand_v9 vs distance (a C3 replication), ML vs frozen
  HAND, tier hit rates / ORs, per-site rows, both hypotheses against UNOSAT, and the
  negative-control river.

**Sample size** (Nakkhu SE about 0.017 at 26 blocks; one-sided α 0.025, power 0.8; SE ∝
1/√blocks; approximate):

| Blocks | 26 | 40 | 67 | 100 | 150 |
|---|---|---|---|---|---|
| Minimum detectable ΔAUC | 0.048 | 0.039 | 0.030 | 0.024 | 0.020 |

Detecting a gain of the size seen post hoc on Nakkhu (about 0.02) needs about 150 blocks. One
valley tributary will not supply that, which is why the qualifying sites are pooled.

A rough planning guess: sites A and B each hold about as many riverside buildings as the
Nakkhu extent. If they pool to about 52 blocks, the MDE is about 0.034, enough to detect only
a large gain. Below the confirmatory minimum, Phase 2 is reported as descriptive, and the
paper says so.

## Steps

| # | Command (to be built in Step 1 unless it exists) | Output | Test access |
|---|---|---|---|
| 0 | `uv run nakkhu prereg verify` | `problems: []`, addendum listed | no |
| 1 | Code: `nakkhu.followup` package (predictors per `definitions`, the P1 statistics reusing `flood.validate` / `hazard`, the runner calling `record_addendum_access` before any label is read), `uv add --group ml scikit-learn`, synthetic tests for every test; plus the HAND-table join check by centroid x/y (`investigation_eval.md` §1) | tests green | no |
| 2 | `uv run nakkhu followup prep` (VPS): HAND at the 16 thresholds (GLO-30, FABDEM, GEDTM30), hand_mr_d8 / hand_mr_rem, the P1-6 feature table, drain shares | `artifacts/followup/prep_*` (label-free) | no |
| 3 | Coordinator sets `runs/eval/test_hold.yaml` `hold: false` with `addendum: prereg_v9_a1` (user authorisation of 2026-10-17) | hold file | no |
| 4 | `uv run nakkhu followup run --test P1-1` … `P1-8` (P1-6 on the VPS), once each; the UNOSAT zip is registered in the manifest with its sha256 first | `artifacts/followup/p1_*` | **yes, 8 entries** |
| 5 | `uv run nakkhu followup run --test P1-4-<dem>` / `P1-5-ext` when the data exist | as above | **yes, one each** |
| 6 | Optional: `uv run nakkhu followup explore --note "<n> <description>"` | runs only | **yes, each journalled** |
| 7 | Site check of A and B (label-free; download WV02 `10300100F2BCCF00` for A), then write and freeze a2: `uv run nakkhu prereg addendum --file configs/prereg_addenda/prereg_v9_a2.yaml --plan ../plans/014-followup-tests.md --changelog <md>` | `artifacts/prereg/addenda/prereg_v9_a2.*` | no |
| 8 | Phase-2 labels: `flood change` per site (no access), then the V1 audit sitting (150 + 8 items per site) and its estimate | change map, audit estimate | **yes, `prereg_v9_a1:P2-audit`** |
| 9 | `uv run nakkhu followup confirm` (once) | `artifacts/followup/p2_confirm.json` | **yes, `prereg_v9_a1:P2`** |
| 10 | One FINDINGS line per result, tagged post hoc (Phase 1) or confirmatory/descriptive (Phase 2); README row updated | | no |

## Reporting rules

- C3 and every v9 number stay as reported. Phase-1 results are presented as post-hoc
  diagnostics and development, with this addendum cited and the order of events (C3 seen
  first) stated.
- No Phase-1 threshold, DEM or model is described as "the HAND result". Only the Phase-2
  decision can support "HAND (or the model) beats distance", and only in confirmatory mode.
- Every declared test is reported whatever its outcome. The explore count is reported.

## Done Criteria

- Phase 1: the 8 declared tests (P1-1 to P1-8) are journalled once each, their artifacts are promoted and
  listed in FINDINGS (post hoc), and `prereg verify` is clean.
- Phase 2: prereg_v9_a2 is frozen before any site label; P2 is journalled once; the mode
  (confirmatory/descriptive) and the Holm decision are written in `p2_confirm.json`.

## STOP Conditions

- `prereg verify` reports any problem, or the journal holds an undeclared `prereg_v9_a1:` entry.
- A Phase-1 test would need a setting that is not in the YAML. Do not improvise; declare a new
  addendum first.
- Any Phase-2 label exists before prereg_v9_a2 is frozen, or the change-map rule would change
  for the new site.
- Data a test needs (DEM, independent reference, imagery) needs an account or licence the user
  has not provided.

## Open Questions (user)

1. The V1 audit for Phase 2 needs one author sitting of 158 items per site, 316 for A + B.
   For scale, sitting 3 took 7.1 min for 280 items. Is that acceptable? Without it, Phase 2
   can only be descriptive.
2. The TanDEM-X 12 m proposal to DLR (P1-4-tandemx12) is the user's to file. SRTM GL1,
   NASADEM, AW3D30 and HMA 8 m are being fetched by another agent.
3. The Khatiwada et al. (2026) high-water marks / UAV DEM and the supervisor's earlier HAND
   basis are still being gathered. They join only as P1-4-uav or a new addendum.
