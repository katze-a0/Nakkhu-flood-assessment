import geopandas as gpd
from shapely.geometry import Point
D=r"D:\academia\flood\.tmp\data"
hb=gpd.read_file(r"zip://D:\academia\flood\.tmp\data\raw\hybas_as_lev12_v1c.zip", bbox=(85.0,27.3,85.7,27.9))
u=hb[hb.HYBAS_ID==4120843950].copy()
u["note"]="HydroBASINS v1c L12 unit containing the Nakkhu confluence; SUB_AREA 201.2 km2, UP_AREA 678.2 km2: lumps Nakkhu with Bagmati, cannot isolate Nakkhu"
u.to_file(rf"{D}\aoi_c_hydrobasins_l12_unit_4120843950.geojson",driver="GeoJSON")
sites={"Nakkhu-Bagmati confluence (OSM way 518744798 end)":(85.296452,27.6632326),"Tikabhairab (Lele-Nallu junction, head of main stem)":(85.3121,27.5746),
"Upstream end of UNOSAT-mapped water (map-read, +-100 m)":(85.309,27.588),"Khatiwada 2026 Nakkhu Breakpoint (map-read)":(85.3036,27.6630),
"Nepal Mediciti":(85.3027,27.6624),"Nakhu Jail":(85.3050,27.6634),"Nakhu Bridge":(85.3060,27.6625),"Nakhu LMC-13":(85.3070,27.6641),
"Bagdol LMC-4":(85.3035,27.6697),"Kantipur Colony, Nakhipot":(85.3130,27.6465),"Nakhipot LMC-14 (locality centroid)":(85.3208,27.6504),
"Nakhudol (locality centroid)":(85.3178,27.6534),"Bhaisepati":(85.3060,27.6562),"UNOSAT AOI1 centre":(85.3008,27.6628)}
gpd.GeoDataFrame({"name":list(sites),"coord_source":["OSM Overpass/Nominatim or map-read; see research-area-imagery.md"]*len(sites)},geometry=[Point(v) for v in sites.values()],crs=4326).to_file(rf"{D}\flood_reach_sites_2024.geojson",driver="GeoJSON")
