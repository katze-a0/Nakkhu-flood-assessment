"""Build candidate study-area boundaries (a, b, c) for Nakkhu; write GeoJSON to .tmp/data."""
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio import features
import pyflwdir
from shapely.geometry import box, Point, shape, mapping
from shapely.ops import unary_union

D = r"D:\academia\flood\.tmp\data"
UTM = "EPSG:32645"

def save(gdf, name):
    gdf.to_crs(4326).to_file(rf"{D}\{name}.geojson", driver="GeoJSON")

def km2(g):
    return gpd.GeoSeries([g], crs=4326).to_crs(UTM).area.iloc[0] / 1e6

# (a) old stated area (Main.tex l.197): 27 38'-27 41' N, 85 17'-85 21' E
a = box(85 + 17 / 60, 27 + 38 / 60, 85 + 21 / 60, 27 + 41 / 60)
ga = gpd.GeoDataFrame({"name": ["a_old_stated_bbox"], "source": ["Main.tex l.197: 27d38'-27d41'N, 85d17'-85d21'E"]}, geometry=[a], crs=4326)
save(ga, "aoi_a_old_stated_bbox")
print("(a) area km2", round(km2(a), 2), a.bounds)

# (b) old mosaic extent, fitted in .tmp/prototype-inventory.md section 1.1 (EPSG:32645)
E0, N0 = 330323.102, 3061574.731
b_utm = box(E0, N0 - 0.5 * 28800, E0 + 0.5 * 17280, N0)
gb = gpd.GeoDataFrame({"name": ["b_old_mosaic_extent"], "source": ["affine fit of Paper01 CSVs: E 330323-338963, N 3047175-3061575 (EPSG:32645)"]}, geometry=[b_utm], crs=UTM).to_crs(4326)
save(gb, "aoi_b_old_mosaic_extent")
print("(b) area km2", round(b_utm.area / 1e6, 2), gb.geometry.iloc[0].bounds)

# confluence from OSM (way 518744798 'Nakhu River' end node meets Bagmati)
CONF = (85.296452, 27.6632326)

# (c1) HydroBASINS level 12 upstream of the confluence
hb = gpd.read_file(r"zip://D:\academia\flood\.tmp\data\raw\hybas_as_lev12_v1c.zip", bbox=(85.0, 27.3, 85.7, 27.9))
print("hybas cols", list(hb.columns))
# basin containing a point 300 m upstream on the Nakkhu (OSM node approx.)
lines = gpd.read_file(rf"{D}\osm_nakkhu_bagmati_lines.geojson")
nak = lines[lines["name"].isin(["Nakhu River", "Nakhhu"])].to_crs(UTM)
nak_line = unary_union(nak.geometry)
conf_utm = gpd.GeoSeries([Point(CONF)], crs=4326).to_crs(UTM).iloc[0]
print("nakkhu OSM main-stem length km (Nakhu River + Nakhhu ways)", round(nak_line.length / 1000, 2))
hb_utm = hb.to_crs(UTM)
cands = hb_utm[hb_utm.intersects(conf_utm.buffer(400))]
print(cands[["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]])
# pick the one intersecting the Nakkhu line within 1 km of the confluence and not the largest (Bagmati) UP_AREA
near = nak_line.intersection(conf_utm.buffer(1000))
c_hb = hb_utm[hb_utm.intersects(near)]
print("intersect nakkhu near confluence:\n", c_hb[["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]])

def upstream_ids(hid):
    ids = {hid}
    frontier = [hid]
    while frontier:
        nxt = hb.loc[hb["NEXT_DOWN"].isin(frontier), "HYBAS_ID"].tolist()
        nxt = [i for i in nxt if i not in ids]
        ids.update(nxt)
        frontier = nxt
    return ids

out = {}
for hid in c_hb["HYBAS_ID"]:
    ids = upstream_ids(hid)
    poly = unary_union(hb.loc[hb["HYBAS_ID"].isin(ids)].geometry)
    out[hid] = (ids, poly)
    print("HYBAS", hid, "n_sub", len(ids), "area km2", round(km2(poly), 1), "UP_AREA", hb.loc[hb.HYBAS_ID == hid, "UP_AREA"].iloc[0])

# (c2) MERIT-Hydro via mghydro API (downloaded separately)
mg = gpd.read_file(rf"{D}\mghydro_nakkhu_watershed.geojson")
print("mghydro cols", list(mg.columns), mg.drop(columns="geometry").head().to_dict())
print("(c2) mghydro area km2", round(km2(unary_union(mg.geometry)), 2), unary_union(mg.geometry).bounds)

# (c3) delineate from Copernicus GLO-30 with pyflwdir
with rasterio.open(r"D:\academia\flood\.tmp\data\raw\Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif") as src:
    win = from_bounds(85.1, 27.35, 85.6, 27.85, src.transform)
    dem = src.read(1, window=win).astype("float32")
    tr = src.window_transform(win)
    nod = src.nodata
print("dem shape", dem.shape, "nodata", nod)
flw = pyflwdir.from_dem(data=dem, nodata=nod if nod is not None else -9999, transform=tr, latlon=True, outlets="edge")
upa = flw.upstream_area(unit="km2")
# snap: candidate cells within 600 m of confluence lying on Nakkhu line buffer 60 m, choose max upa that is < 400 km2 (exclude Bagmati)
rows, cols = np.indices(dem.shape)
xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel())
xs = np.asarray(xs).reshape(dem.shape); ys = np.asarray(ys).reshape(dem.shape)
sel = (np.abs(xs - CONF[0]) < 0.01) & (np.abs(ys - CONF[1]) < 0.01)
r_i, c_i = np.where(sel)
pts = gpd.GeoSeries([Point(xs[r, c], ys[r, c]) for r, c in zip(r_i, c_i)], crs=4326).to_crs(UTM)
nb = nak_line.buffer(60)
ok = [(upa[r, c], r, c) for (r, c), p in zip(zip(r_i, c_i), pts) if nb.contains(p) and p.distance(conf_utm) < 700]
ok.sort(reverse=True)
print("top snap cands (upa km2,row,col)", [(round(u, 1), r, c) for u, r, c in ok[:8]])
results = {}
for u, r, c in ok[:1]:
    bas = flw.basins(idxs=np.array([r * dem.shape[1] + c]))
    shp = [shape(g) for g, v in features.shapes(bas.astype("int32"), mask=bas > 0, transform=tr)]
    poly = unary_union(shp)
    results["glo30"] = poly
    print("(c3) GLO-30 pyflwdir basin km2", round(km2(poly), 2), "upa at outlet", round(u, 2), "outlet", xs[r, c], ys[r, c], poly.bounds)

# save (c) candidates
rows_out = []
for hid, (ids, poly) in out.items():
    rows_out.append({"name": f"c1_hydrobasins_l12_up_of_{hid}", "source": "HydroBASINS v1c level 12, upstream of HYBAS_ID %d" % hid, "area_km2": round(km2(poly), 2), "geometry": poly})
rows_out.append({"name": "c2_merit_hydro_mghydro", "source": "mghydro.com watershed_api lat=27.6625 lng=85.2985 precision=high (MERIT-Hydro)", "area_km2": round(km2(unary_union(mg.geometry)), 2), "geometry": unary_union(mg.geometry)})
if "glo30" in results:
    rows_out.append({"name": "c3_glo30_pyflwdir", "source": "Copernicus GLO-30 N27E085, pyflwdir from_dem, outlet snapped on OSM Nakkhu <700 m above confluence", "area_km2": round(km2(results['glo30']), 2), "geometry": results["glo30"]})
gc = gpd.GeoDataFrame(rows_out, crs=4326)
save(gc, "aoi_c_nakkhu_catchment_candidates")
# upper Bagmati to Nakkhu confluence (for HAND routing domain), from GLO-30: snap to max upa on Bagmati near confluence
bag = lines[lines["name"].str.contains("Bagmati")].to_crs(UTM)
bline = unary_union(bag.geometry).buffer(60)
okb = [(upa[r, c], r, c) for (r, c), p in zip(zip(r_i, c_i), pts) if bline.contains(p) and p.distance(conf_utm) < 1500]
okb.sort(reverse=True)
u, r, c = okb[0]
bas = flw.basins(idxs=np.array([r * dem.shape[1] + c]))
poly = unary_union([shape(g) for g, v in features.shapes(bas.astype("int32"), mask=bas > 0, transform=tr)])
print("Bagmati incl. Nakkhu, below confluence: upa", round(u, 1), "basin km2", round(km2(poly), 1), "outlet", xs[r, c], ys[r, c], poly.bounds)
save(gpd.GeoDataFrame({"name": ["upper_bagmati_below_nakkhu_confluence_glo30"], "area_km2": [round(km2(poly), 1)]}, geometry=[poly], crs=4326), "routing_domain_upper_bagmati_glo30")
