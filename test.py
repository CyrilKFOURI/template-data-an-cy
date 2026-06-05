nova['CLS_VEHICLE_TYPE_clean'] = (
    nova['CLS_VEHICLE_TYPE']
      .astype(str)
      .str.strip()
      .replace(r'^(\d+)\.0$', r'\1', regex=True)
)

pct_by_country = (
    pd.crosstab(
        nova['COUNTRY'],
        nova['CLS_VEHICLE_TYPE_clean'],
        normalize='index'
    ) * 100
).round(1)
