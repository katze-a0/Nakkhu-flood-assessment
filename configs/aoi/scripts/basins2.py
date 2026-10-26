"""Nakkhu catchment candidates (c): HydroBASINS L12 (cannot isolate), MERIT-Hydro (mghydro), GLO-30 pyflwdir."""
import numpy as np, geopandas as gpd, rasterio, pyflwdir
from rasterio.windows import from_bounds
from rasterio import features
from shapely.geometry import Point, shape
from shapely.ops import unary_union

D = r"D:\academia\flood\.tmp\data"; UTM = "EPSG:32645"
km2 = lambda g: gpd.GeoSeries([g], crs=4326).to_crs(UTM).area.iloc[0] / 1e6
CONF = (85.296452, 27.6632326)
lines = gpd.read_file(rf"{D}\osm_nakkhu_bagmati_lines.geojson")
nak_line = unary_union(lines[lines["name"].isin(["Nakhu River", "Nakhhu"])].to_crs(UTM).geometry)
conf_utm = gpd.GeoSeries([Point(CONF)], crs=4326).to_crs(UTM).iloc[0]

hb = gpd.read_file(r"zip://D:\academia\flood\.tmp\data\raw\hybas_as_lev12_v1c.zip", bbox=(85.0, 27.3, 85.7, 27.9))
# which L12 units does the Nakkhu main stem cross?
hbu = hb.to_crs(UTM)
x = hbu[hbu.intersects(nak_line)]
print("HydroBASINS L12 units crossed by OSM Nakkhu main stem:\n", x[["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]])

with rasterio.open(rf"{D}\raw\Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif") as src:
    win = from_bounds(85.1, 27.35, 85.6, 27.85, src.transform)
    dem = src.read(1, window=win).astype("float32"); tr = src.window_transform(win)
flw = pyflwdir.from_dem(data=dem, nodata=-9999, transform=tr, latlon=True, outlets="edge")
upa = flw.upstream_area(unit="km2")
rows, cols = np.where(upa > 20)
xs, ys = rasterio.transform.xy(tr, rows, cols)
pts = gpd.GeoSeries([Point(a, b) for a, b in zip(xs, ys)], crs=4326).to_crs(UTM)
nb = nak_line.buffer(90)
cand = [(p.distance(conf_utm), upa[r, c], r, c, a, b) for p, r, c, a, b in zip(pts, rows, cols, xs, ys) if nb.contains(p) and upa[r, c] < 300]
cand.sort()
print("closest GLO-30 stream cells on Nakkhu (dist m, upa km2):", [(round(d), round(u, 1)) for d, u, *_ in cand[:6]])
d, u, r, c, a, b = cand[0]
bas = flw.basins(idxs=np.array([r * dem.shape[1] + c]))
poly = unary_union([shape(g) for g, v in features.shapes(bas.astype("int32"), mask=bas > 0, transform=tr)])
print("GLO-30 Nakkhu basin: outlet", a, b, "dist to OSM confluence m", round(d), "area km2", round(km2(poly), 2), "bounds", poly.bounds)

mg = gpd.read_file(rf"{D}\mg_27.6590_lng_85.3040.geojson")
mgp = unary_union(mg.geometry)
print("MERIT mghydro area km2", round(km2(mgp), 2), "bounds", mgp.bounds)
iou = km2(poly.intersection(mgp)) / km2(poly.union(mgp))
print("IoU GLO30 vs MERIT", round(iou, 3))
out = gpd.GeoDataFrame([
    {"name": "c2_merit_hydro_mghydro", "source": "https://mghydro.com/app/watershed_api?lat=27.659&lng=85.304&precision=high (MERIT-Hydro/MERIT-Basins hybrid), outlet ~0.8 km above OSM confluence", "area_km2": round(km2(mgp), 2), "geometry": mgp},
    {"name": "c3_glo30_pyflwdir", "source": "Copernicus GLO-30 DSM tile N27E085 (AWS), pyflwdir 0.5 from_dem, outlet = GLO-30 stream cell on OSM Nakkhu nearest the OSM Nakkhu-Bagmati confluence (85.296452, 27.6632326)", "area_km2": round(km2(poly), 2), "geometry": poly},
], crs=4326)
out.to_file(rf"{D}\aoi_c_nakkhu_catchment.geojson", driver="GeoJSON")
