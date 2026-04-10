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