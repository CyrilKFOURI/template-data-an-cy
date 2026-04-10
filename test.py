import numpy as np

fuel_mapping = {
    'Spain': {
        'GASOLINA': 'PETROL',
        'DIESEL': 'DIESEL'
    },
    'Italy': {
        'BENZINA': 'PETROL',
        'DIESEL': 'DIESEL'
    }
}

fuel_suffix_map = {
    'GASOLINA': 'G',
    'BENZINA': 'G',
    'PETROL': 'G',
    'DIESEL': 'D'
}

mapping = fuel_mapping.get(COUNTRY.strip().title(), {})

# -------- CATEGORY 2 --------
df['POWER CATEGORY_2'] = np.where(
    df['POWER CATEGORY'].astype(str).str.strip().str.upper().eq('NHEV'),
    df['FUEL_TYPE2'].astype(str).str.strip().str.upper().map(mapping),
    df['POWER CATEGORY']
)

# -------- CATEGORY 3 --------
fuel_clean = df['FUEL_TYPE2'].astype(str).str.strip().str.upper().map(fuel_suffix_map)

df['POWER CATEGORY_3'] = np.select(
    [
        df['POWER CATEGORY'].astype(str).str.strip().str.upper().eq('MHEV'),
        df['POWER CATEGORY'].astype(str).str.strip().str.upper().eq('PLUG-IN HYBRID')
    ],
    [
        'MHEV-' + fuel_clean,
        'PHEV-' + fuel_clean
    ],
    default=df['POWER CATEGORY']
)

# fallback
df['POWER CATEGORY_2'] = df['POWER CATEGORY_2'].fillna(df['POWER CATEGORY'])
df['POWER CATEGORY_3'] = df['POWER CATEGORY_3'].fillna(df['POWER CATEGORY'])