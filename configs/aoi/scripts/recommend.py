"""Recommended analysis extent = (Nakkhu catchment ∪ 1 km confluence zone) ∩ Maxar pre (WV02 2024-01-13) ∩ post (GE01 2024-10-06).
Checks which named flooded sites fall inside each extent; counts Microsoft footprints; GE01 clear fraction is computed separately."""
import gzip, json, os
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union

D = r"D:\academia\flood\.tmp\data"; UTM = "EPSG:32645"; SP = os.path.dirname(os.path.abspath(__file__))
it = gpd.read_file(rf"{D}\maxar_nepal_floods_2024_items.geojson").to_crs(UTM)
pre = unary_union(it[it.catalog_id == "10300100F5AB8000"].geometry)
post = unary_union(it[it.catalog_id.isin(["105001003E1F0500", "105001003E1F0600"])].geometry)
C = gpd.read_file(rf"{D}\aoi_c_nakkhu_catchment.geojson").to_crs(UTM)
catch = C[C.name == "c4_glo30_osm_burned"].geometry.iloc[0]
conf = gpd.GeoSeries([Point(85.296452, 27.6632326)], crs=4326).to_crs(UTM).iloc[0]
zone = conf.buffer(1000)
rec = catch.union(zone).intersection(pre).intersection(post)
rec = unary_union([g for g in getattr(rec, "geoms", [rec]) if g.area > 1e4])
d = gpd.read_file(rf"{D}\aoi_d_analysis_extent_lower_nakkhu.geojson").to_crs(UTM).geometry.iloc[0]
print("recommended extent km2", round(rec.area / 1e6, 2), "| extent d km2", round(d.area / 1e6, 2))
sites = {"Nepal Mediciti": (85.3027, 27.6624), "Nakhu Jail": (85.3050, 27.6634), "Nakhu Bridge": (85.3060, 27.6625), "Nakhu LMC-13": (85.3070, 27.6641),
         "Bagdol": (85.3035, 27.6697), "Kantipur Colony": (85.3130, 27.6465), "Nakhipot": (85.3208, 27.6504), "Nakhudol": (85.3178, 27.6534),
         "Bhaisepati": (85.3060, 27.6562), "Thecho": (85.3207, 27.6141), "Tikabhairab": (85.3121, 27.5746), "Dukuchhap": (85.2943, 27.5867),
         "Khatiwada domain centre": (85.300, 27.664), "UNOSAT AOI1 centre": (85.3008, 27.6628)}
P = gpd.GeoSeries([Point(v) for v in sites.values()], crs=4326).to_crs(UTM)
for (n, _), p in zip(sites.items(), P):
    print(f"  {n:24s} catchment {catch.contains(p)!s:5s} extent_d {d.contains(p)!s:5s} recommended {rec.contains(p)!s:5s} dist-to-catchment {p.distance(catch):6.0f} m")
kb = gpd.GeoSeries([box(85.295, 27.661, 85.305, 27.667)], crs=4326).to_crs(UTM).iloc[0]
print("Khatiwada domain bbox inside recommended %:", round(kb.intersection(rec).area / kb.area * 100, 1), "| inside catchment %:", round(kb.intersection(catch).area / kb.area * 100, 1))
g = gpd.GeoDataFrame({"name": ["recommended_analysis_extent"], "definition": ["(GLO-30 OSM-burned Nakkhu catchment c4 ∪ 1 km radius around OSM Nakkhu-Bagmati confluence 85.296452E 27.663233N) ∩ Maxar WV02 10300100F5AB8000 (2024-01-13) ∩ GE01 105001003E1F0500/0600 (2024-10-06) data footprints"],
                      "area_km2": [round(rec.area / 1e6, 2)]}, geometry=[rec], crs=UTM).to_crs(4326)
g.to_file(rf"{D}\aoi_recommended_analysis_extent.geojson", driver="GeoJSON")
r4 = g.geometry.iloc[0]
n = 0
with gzip.open(rf"{SP}\ms_q.csv.gz", "rt") as f:
    bb = r4.bounds
    for line in f:
        c = json.loads(line)["geometry"]["coordinates"][0]
        x = sum(p[0] for p in c) / len(c); y = sum(p[1] for p in c) / len(c)
        if bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3] and r4.contains(Point(x, y)):
            n += 1
print("Microsoft footprints (centroid) in recommended extent:", n, "| bounds", [round(v, 4) for v in r4.bounds])
