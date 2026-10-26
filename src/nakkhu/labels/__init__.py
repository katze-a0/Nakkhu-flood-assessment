"""Consensus weak labels from Google Open Buildings v3 and Microsoft footprints (plan 004).

Only Google v3 and Microsoft footprints are label sources. OpenStreetMap and Overture are the
held-out references and are read only by :mod:`nakkhu.labelqa`; a test enforces that no module
here references them. One exception, by user decision (2026-10-05): footprints overlapping an OSM
greenhouse area (shared rule ``nakkhu.reference.greenhouse``, also used for exposure counts) are
dropped, since greenhouses are not buildings; OSM never adds a label.
"""
