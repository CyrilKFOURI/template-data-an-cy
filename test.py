nova['CLS_VEHICLE_TYPE_clean'] = (
    pd.to_numeric(nova['CLS_VEHICLE_TYPE'], errors='coerce')
      .astype('Int64')
)

pct_by_country = (
    pd.crosstab(
        nova['COUNTRY'],
        nova['CLS_VEHICLE_TYPE_clean'],
        normalize='index'
    ) * 100
).round(1)
