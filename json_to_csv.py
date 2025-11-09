import json
import csv

# Path to your GeoJSON file
geojson_file = '/media/eric/D/repos/caixaenginyers-heatmap-demo/data/citylocation.geojson'
# Output CSV file
csv_file = '/media/eric/D/repos/caixaenginyers-heatmap-demo/data/banks-by-population.csv'

# Load GeoJSON data
with open(geojson_file) as f:
    data = json.load(f)

# Open CSV file for writing
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['Municipality', 'Bank', 'Longitude', 'Latitude'])

    # Iterate over features
    for feature in data['features']:
        properties = feature['properties']
        geometry = feature['geometry']

        # New dataset (citylocation.geojson) lists municipalities/places, not bank amenities.
        # Derive municipality name from available keys.

        # Municipality candidates (best-effort from available address fields)
        municipality = (
            properties.get('idee:name')
            or properties.get('name')
            or properties.get('addr:city')
            or properties.get('addr:town')
            or properties.get('addr:village')
            or properties.get('addr:hamlet')
            or 'Unknown'
        )

        # Bank name is not present in this dataset; use municipality as a placeholder
        bank = municipality

        # Extract representative coordinates depending on geometry type
        longitude = 'Unknown'
        latitude = 'Unknown'
        try:
            gtype = geometry.get('type')
            coords = geometry.get('coordinates')
            if gtype == 'Point' and isinstance(coords, list) and len(coords) >= 2:
                longitude, latitude = coords[0], coords[1]
            elif gtype == 'LineString' and isinstance(coords, list) and len(coords) >= 1:
                first = coords[0]
                if isinstance(first, list) and len(first) >= 2:
                    longitude, latitude = first[0], first[1]
            elif gtype == 'Polygon' and isinstance(coords, list) and len(coords) >= 1:
                ring0 = coords[0]
                if isinstance(ring0, list) and len(ring0) >= 1 and isinstance(ring0[0], list) and len(ring0[0]) >= 2:
                    longitude, latitude = ring0[0][0], ring0[0][1]
            elif gtype == 'MultiLineString' and isinstance(coords, list) and len(coords) >= 1:
                line0 = coords[0]
                if isinstance(line0, list) and len(line0) >= 1 and isinstance(line0[0], list) and len(line0[0]) >= 2:
                    longitude, latitude = line0[0][0], line0[0][1]
            elif gtype == 'MultiPolygon' and isinstance(coords, list) and len(coords) >= 1:
                poly0 = coords[0]
                if isinstance(poly0, list) and len(poly0) >= 1:
                    ring0 = poly0[0]
                    if isinstance(ring0, list) and len(ring0) >= 1 and isinstance(ring0[0], list) and len(ring0[0]) >= 2:
                        longitude, latitude = ring0[0][0], ring0[0][1]
        except Exception:
            # Keep Unknowns if parsing fails
            pass
        
        # Skip rows with any Unknowns or non-numeric coordinates
        if (
            municipality == 'Unknown'
            or longitude == 'Unknown'
            or latitude == 'Unknown'
        ):
            continue
        if not (isinstance(longitude, (int, float)) and isinstance(latitude, (int, float))):
            continue

        # Write row
        writer.writerow([municipality, bank, longitude, latitude])
