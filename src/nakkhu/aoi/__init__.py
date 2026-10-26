"""Derived AOI files of plan 002, rebuilt from registered raw inputs (``nakkhu aoi build``).

Modules: :mod:`.lines` (OSM river lines and the waterway=river set from the dated Geofabrik
PBF), :mod:`.basins` (GLO-30 catchment c4 with the OSM burn, and the upper-Bagmati routing
domain), :mod:`.extent` (analysis extent), :mod:`.sites` (named flood sites from a versioned
config), :mod:`.compare` (rebuilt vs registered geometry) and :mod:`.build` (the run).
Settings: ``configs/aoi/build.yaml``.
"""

BUILD_CONFIG = "configs/aoi/build.yaml"
PRODUCTS = ("waterways_rivers", "osm_lines", "catchment_c4", "routing_domain", "analysis_extent",
            "flood_sites")
