import numpy as np

def add_power_category_2(df):
    df['POWER_CATEGORY_2'] = np.where(
        df['POWER_CATEGORY'] == 'MHEV',
        np.where(df['FUEL_TYPE2'] == 'GASOLINA', 'PETROL',
                 np.where(df['FUEL_TYPE2'] == 'DIESELO', 'DIESEL', 'MHEV')),
        df['POWER_CATEGORY']
    )
    return df
