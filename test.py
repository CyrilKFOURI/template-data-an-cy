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

fuel_suffix_map = {
    'GASOLINA': 'G',
    'Benzina': 'G',
    'PETROL': 'G',
    'DIESEL': 'D',
    'Diesel': 'D'
}

mapping = fuel_mapping.get(COUNTRY, {})

# -------- CATEGORY 2 --------
df['POWER CATEGORY_2'] = np.where(
    df['POWER CATEGORY'].eq('NHEV'),
    df['FUEL_TYPE2'].map(mapping),
    df['POWER CATEGORY']
)

# -------- CATEGORY 3 --------
fuel_suffix = df['FUEL_TYPE2'].map(fuel_suffix_map)

df['POWER CATEGORY_3'] = np.select(
    [
        df['POWER CATEGORY'].eq('MHEV'),
        df['POWER CATEGORY'].eq('PLUG-IN HYBRID')
    ],
    [
        'MHEV-' + fuel_suffix,
        'PHEV-' + fuel_suffix
    ],
    default=df['POWER CATEGORY']
)