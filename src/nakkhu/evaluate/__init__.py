"""Evaluation harness (plan 005): matching, metrics, block bootstrap, intervals, test guard.

- :mod:`.match`: rotated-polygon IoU pairs and greedy-by-confidence one-to-one matching.
- :mod:`.metrics`: P/R/F1, AP (COCO 101-point; VOC all-point), F1-max threshold, count errors.
- :mod:`.bootstrap`: block bootstrap of additive per-block statistics, AP and paired differences.
- :mod:`.ci`: Wilson intervals, seed mean +- SD, Wilcoxon signed-rank on per-block values.
- :mod:`.guard`: frozen ``configs/prereg.yaml`` / split check and the test access journal.
- :mod:`.score`: scoring a run's detections against aligned OSM on val or test blocks.
"""
