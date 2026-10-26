"""GLO-30 Nakkhu catchment with OSM stream burning (Nakkhu main stem + Bagmati lowered 15 m), outlet at OSM confluence."""
import numpy as np, geopandas as gpd, rasterio, pyflwdir
from rasterio.windows import from_bounds
from rasterio import features
from shapely.geometry import Point, shape
from shapely.ops import unary_union

D = r"D:\academia\flood\.tmp\data"; UTM = "EPSG:32645"
km2 = lambda g: gpd.GeoSeries([g], crs=4326).to_crs(UTM).area.iloc[0] / 1e6
L = gpd.read_file(rf"{D}\osm_nakkhu_bagmati_lines.geojson")
with rasterio.open(rf"{D}\raw\Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif") as src:
    win = from_bounds(85.1, 27.35, 85.6, 27.85, src.transform)
    dem = src.read(1, window=win).astype("float32"); tr = src.window_transform(win)
burn = features.rasterize([(g, 1) for g in L.geometry], out_shape=dem.shape, transform=tr, all_touched=True).astype(bool)
demb = dem - 60.0 * burn
flw = pyflwdir.from_dem(data=demb, nodata=-9999, transform=tr, latlon=True, outlets="edge")
upa = flw.upstream_area(unit="km2")
nak = unary_union(L[L.name.isin(["Nakhu River", "Nakhhu"])].geometry)
# candidate burned Nakkhu cells within 300 m of the confluence, upa < 300 km2 -> take the largest (most downstream on the Nakkhu)
rows, cols = np.where(burn & (upa > 20) & (upa < 300))
xs, ys = rasterio.transform.xy(tr, rows, cols)
P = gpd.GeoSeries([Point(a, b) for a, b in zip(xs, ys)], crs=4326).to_crs(UTM)
conf = gpd.GeoSeries([Point(85.296452, 27.6632326)], crs=4326).to_crs(UTM).iloc[0]
cand = sorted([(upa[r, c], r, c, a, b) for p, r, c, a, b in zip(P, rows, cols, xs, ys) if p.distance(conf) < 300], reverse=True)
u, r, c, a, b = cand[0]
bas = flw.basins(idxs=np.array([r * dem.shape[1] + c]))
poly = unary_union([shape(g) for g, v in features.shapes(bas.astype("int32"), mask=bas > 0, transform=tr)])
print("burned GLO-30 Nakkhu basin: outlet", round(a, 5), round(b, 5), "upa", round(u, 2), "area km2", round(km2(poly), 2), "bounds", [round(v, 4) for v in poly.bounds])
pu = gpd.GeoSeries([poly], crs=4326).to_crs(UTM).iloc[0]
stem = unary_union(L[L.name.isin(["Nakhu River", "Nakhhu", "Nallu Khola"])].to_crs(UTM).geometry)
print("share of OSM stem inside", round(stem.intersection(pu).length / stem.length * 100, 1), "%")
old = gpd.read_file(rf"{D}\aoi_c_nakkhu_catchment.geojson")
mg = old[old.name == "c2_merit_hydro_mghydro"].geometry.iloc[0]; g3 = old[old.name == "c3_glo30_pyflwdir"].geometry.iloc[0]
for n, o in [("merit", mg), ("glo30 unburned", g3)]:
    print("IoU vs", n, round(km2(poly.intersection(o)) / km2(poly.union(o)), 3))
for nm, xy in {"Mediciti": (85.3027, 27.6624), "Nakhu Jail": (85.3050, 27.6634), "Nakhu Bridge": (85.3060, 27.6625), "Kantipur Colony": (85.3130, 27.6465), "Nakhipot": (85.3208, 27.6504), "Nakhudol": (85.3178, 27.6534), "Bagdol": (85.3035, 27.6697), "Bhaisepati": (85.3060, 27.6562)}.items():
    print("  ", nm, poly.contains(Point(xy)))
row = gpd.GeoDataFrame([{"name": "c4_glo30_osm_burned", "source": "Copernicus GLO-30 N27E085 with OSM Nakkhu/Nallu/Lele/Bagmati lines burned 60 m (all_touched), pyflwdir from_dem, outlet = burned Nakkhu cell <300 m from OSM confluence", "area_km2": round(km2(poly), 2), "geometry": poly}], crs=4326)
allc = gpd.GeoDataFrame(list(old.to_dict("records")) + list(row.to_dict("records")), crs=4326)
allc.to_file(rf"{D}\aoi_c_nakkhu_catchment.geojson", driver="GeoJSON")
