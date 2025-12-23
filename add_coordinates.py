import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

print("Starting geocoding... This will take 10-20 minutes.")

df = pd.concat([pd.read_csv('pitchers.csv'), pd.read_csv('hitters.csv')], ignore_index=True)
places = df['hsplace'].dropna().unique()
print(f"Found {len(places)} unique hometowns")

geolocator = Nominatim(user_agent="baseball_map")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

coords = {}
for i, place in enumerate(places):
    try:
        location = geocode(place + ", USA")
        if location:
            coords[place] = (location.latitude, location.longitude)
            print(f"{i+1}/{len(places)} ✓ {place}")
        else:
            coords[place] = (None, None)
            print(f"{i+1}/{len(places)} ✗ {place}")
    except:
        coords[place] = (None, None)
        print(f"{i+1}/{len(places)} ✗ Error: {place}")
    time.sleep(1)

# Save coordinates
mapping = pd.DataFrame(coords.items(), columns=['hsplace', 'coords'])
mapping[['lat', 'lon']] = pd.DataFrame(mapping['coords'].tolist(), index=mapping.index)
mapping = mapping.drop('coords', axis=1)
mapping.to_csv('hsplace_coords.csv', index=False)

# Add lat/lon to your CSVs
coord_dict = dict(zip(mapping['hsplace'], zip(mapping['lat'], mapping['lon'])))
for file in ['pitchers.csv', 'hitters.csv']:
    data = pd.read_csv(file)
    data['lat'] = data['hsplace'].map(lambda x: coord_dict.get(x, (None, None))[0] if pd.notna(x) else None)
    data['lon'] = data['hsplace'].map(lambda x: coord_dict.get(x, (None, None))[1] if pd.notna(x) else None)
    data.to_csv(file, index=False)
    print(f"Updated {file} with coordinates")

print("ALL DONE! Now upload the new pitchers.csv and hitters.csv to GitHub.")