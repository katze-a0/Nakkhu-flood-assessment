// aoi_osm_waterways_rivers: OSM waterway=river ways crossing the Kathmandu-valley box.
// Reconstructed 2026-10-05: the research query was not saved. Against the Geofabrik snapshot
// nepal-261003.osm.pbf this selection returns exactly the 103 ways of the registered Overpass
// result (timestamp_osm_base 2026-10-05T10:16:34Z), with identical tags and node coordinates.
// Any box with S 27.445-27.45, W 85.17-85.23, N 27.745-27.76, E 85.445-85.45 gives the same
// set; this is the only one on a 0.05 degree grid. `nakkhu aoi build` applies the same rule to
// the PBF (configs/aoi/build.yaml, waterways_rivers); this file is for reference and live checks.
[out:json];
way["waterway"="river"](27.45,85.20,27.75,85.45);
out geom;
