import json
from shapely.geometry import LineString, mapping
from shapely.ops import nearest_points, linemerge, unary_union

d = json.load(open(r'D:\academia\flood\.tmp\data\osm_waterways_rivers.json', encoding='utf-8'))
lines = {}
for e in d['elements']:
    if e['type'] != 'way':
        continue
    n = e['tags'].get('name') or ''
    g = LineString([(p['lon'], p['lat']) for p in e['geometry']])
    lines.setdefault(n, []).append((e['id'], g))

for key in ['Nakhu River', 'Nakhhu', 'Nallu Khola', 'Lele River']:
    for wid, g in lines.get(key, []):
        print(key, wid, 'len_pts', len(g.coords), 'start', g.coords[0], 'end', g.coords[-1], 'bounds', g.bounds)

nak = unary_union([g for k in ['Nakhu River', 'Nakhhu'] for _, g in lines.get(k, [])])
bag = unary_union([g for k in lines if 'Bagmati' in k for _, g in lines[k]])
p1, p2 = nearest_points(nak, bag)
print('nearest Nakhu-Bagmati', p1, p2, 'dist deg', p1.distance(p2))
feats = []
for k in ['Nakhu River', 'Nakhhu', 'Nallu Khola', 'Lele River']:
    for wid, g in lines.get(k, []):
        feats.append({'type': 'Feature', 'geometry': mapping(g), 'properties': {'name': k, 'osm_way': wid}})
for k in lines:
    if 'Bagmati' in k:
        for wid, g in lines[k]:
            feats.append({'type': 'Feature', 'geometry': mapping(g), 'properties': {'name': k, 'osm_way': wid}})
json.dump({'type': 'FeatureCollection', 'features': feats}, open(r'D:\academia\flood\.tmp\data\osm_nakkhu_bagmati_lines.geojson', 'w'))
