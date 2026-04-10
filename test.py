import numpy as np

fuel_mapping = {
    'Spain': {
        'GASOLINA': 'PETROL',
        'DIESEL': 'DIESEL'
    },
    'Italy': {
        'Benzina': 'PETROL',
        'Diesel': 'DIESEL'
    }
}

def map_power_category(row, country):
    if row['POWER CATEGORY'] == 'NHEV':
        return fuel_mapping.get(country, {}).get(row['FUEL_TYPE2'], None)
    return row['POWER CATEGORY']

df['POWER CATEGORY_2'] = df.apply(lambda row: map_power_category(row, COUNTRY), axis=1)






fuel_map_generic = {
    'PETROL': 'G',
    'GASOLINA': 'G',
    'Benzina': 'G',
    'DIESEL': 'D',
    'Diesel': 'D'
}

df['POWER CATEGORY_3'] = df['POWER CATEGORY']

fuel_suffix = df['FUEL_TYPE2'].map(fuel_map_generic)

mask_mhev = df['POWER CATEGORY'] == 'MHEV'
mask_phev = df['POWER CATEGORY'] == 'PLUG-IN HYBRID'

df.loc[mask_mhev, 'POWER CATEGORY_3'] = 'MHEV-' + fuel_suffix
df.loc[mask_phev, 'POWER CATEGORY_3'] = 'PHEV-' + fuel_suffix