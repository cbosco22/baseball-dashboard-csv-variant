import pandas as pd
from rapidfuzz import process, fuzz

# Load SimpleMaps
cities = pd.read_csv('uscities.csv')
cities['city_state'] = cities['city'] + ', ' + cities['state_id']

# Normalize for fuzzy
cities['norm'] = cities['city_state'].str.upper().str.replace('.', '').str.replace('ST ', 'SAINT ').str.strip()

# Load your player data
df = pd.concat([pd.read_csv('pitchers.csv'), pd.read_csv('hitters.csv')], ignore_index=True)
df['hsplace_norm'] = df['hsplace'].str.upper().str.strip().str.replace('.', '')

# Build fuzzy dict
coord_dict = {}
for place in df['hsplace_norm'].dropna().unique():
    match = process.extractOne(place, cities['norm'], scorer=fuzz.ratio, score_cutoff=85)
    if match:
        matched_row = cities[cities['norm'] == match[0]].iloc[0]
        coord_dict[place] = (matched_row['lat'], matched_row['lng'])
    else:
        coord_dict[place] = (None, None)
    print(f"Matched: {place} → {match[0] if match else 'No match'}")

# Add to CSVs
for file in ['pitchers.csv', 'hitters.csv']:
    data = pd.read_csv(file)
    data['hsplace_norm'] = data['hsplace'].str.upper().str.strip().str.replace('.', '')
    data['lat'] = data['hsplace_norm'].map(coord_dict).apply(lambda x: x[0] if isinstance(x, tuple) else None)
    data['lon'] = data['hsplace_norm'].map(coord_dict).apply(lambda x: x[1] if isinstance(x, tuple) else None)
    data = data.drop(columns=['hsplace_norm'], errors='ignore')
    data.to_csv(file, index=False)
    matched = data['lat'].notna().sum()
    print(f"{file}: {matched}/{len(data)} matched")

print("DONE! Upload the new CSVs.")