import pandas as pd

# Load SimpleMaps
cities = pd.read_csv('uscities.csv')

# Normalize: "City, ST" upper, strip spaces
cities['city_state_norm'] = (cities['city'].str.strip() + ', ' + cities['state_id'].str.strip()).str.upper()

# Build dict
coord_dict = dict(zip(cities['city_state_norm'], zip(cities['lat'], cities['lng'])))

# Process your CSVs
for file in ['pitchers.csv', 'hitters.csv']:
    data = pd.read_csv(file)
    
    # Normalize hsplace exactly the same way
    data['hsplace_norm'] = data['hsplace'].str.strip().str.upper()
    
    # Map
    data['lat'] = data['hsplace_norm'].map(coord_dict).apply(lambda x: x[0] if isinstance(x, tuple) else None)
    data['lon'] = data['hsplace_norm'].map(coord_dict).apply(lambda x: x[1] if isinstance(x, tuple) else None)
    
    # Clean
    data = data.drop(columns=['hsplace_norm'], errors='ignore')
    data.to_csv(file, index=False)
    
    matched = data['lat'].notna().sum()
    total = len(data)
    print(f"{file}: {matched}/{total} matched ({matched/total*100:.1f}%)")

print("DONE! Upload the new CSVs.")