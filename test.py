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

# =========================
# NORMALISATION INLINE
# =========================
pc = df['POWER_CATEGORY'].astype(str).str.strip().str.upper()
f2 = df['FUEL_TYPE2'].astype(str).str.strip().str.upper()
f1 = df['FUEL_TYPE'].astype(str).str.strip().str.upper()

# =========================
# CATEGORY 2
# =========================
fuel2 = f2.map(mapping)
fuel1 = f1.map(mapping)

df['POWER_CATEGORY_2'] = np.where(
    pc.eq('MHEV'),
    np.where(
        fuel2.notna(),
        fuel2,
        np.where(
            fuel1.notna(),
            fuel1,
            'MHEV'
        )
    ),
    df['POWER_CATEGORY']
)

# =========================
# CATEGORY 3
# =========================
suffix2 = f2.map(fuel_suffix_map)
suffix1 = f1.map(fuel_suffix_map)

df['POWER_CATEGORY_3'] = np.select(
    [
        pc.eq('MHEV') & suffix2.notna(),
        pc.eq('MHEV') & suffix2.isna() & suffix1.notna(),
        pc.eq('PLUG-IN HYBRID') & suffix2.notna(),
        pc.eq('PLUG-IN HYBRID') & suffix2.isna() & suffix1.notna()
    ],
    [
        'MHEV-' + suffix2,
        'MHEV-' + suffix1,
        'PHEV-' + suffix2,
        'PHEV-' + suffix1
    ],
    default=df['POWER_CATEGORY']
)

df['POWER_CATEGORY_3'] = df['POWER_CATEGORY_3'].fillna(df['POWER_CATEGORY'])