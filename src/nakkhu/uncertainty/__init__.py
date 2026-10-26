"""Detection-to-exposure uncertainty (plan 010; contribution C2: U1, U2, U3, D4).

The analysis is fixed in ``configs/uncertainty/uncertainty.yaml`` and frozen in prereg v5
(``uncertainty`` section). Modules:

- :mod:`.spec`: config loader, random streams, prerequisites (no test access).
- :mod:`.ensemble`: 5-seed clustering, raw existence score, ``detector_final`` membership (U1).
- :mod:`.calibrate`: temperature / isotonic calibration, block cross-fitting, weighted ECE (U1).
- :mod:`.ppi`: design weights, HT, PPI++ and ratio tier counts, tier pooling, split-sample
  coverage check and its simulation (U3).
- :mod:`.decompose`: balanced-factorial ANOVA shares, share CIs, factorial checks (U2).
- :mod:`.falsifiers`: C2 outcomes exactly as pre-registered.
"""
