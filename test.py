nova['CLS_VEHICLE_TYPE_clean'] = (
    pd.to_numeric(nova['CLS_VEHICLE_TYPE'], errors='coerce')  # convertit en numérique
      .round()                                               # enlève les .0 résiduels
      .astype('Int64')                                       # entier nullable pandas
      .astype(str)                                           # optionnel si tu veux des labels propres
)

pct_by_country = (
    pd.crosstab(
        nova['COUNTRY'],
        nova['CLS_VEHICLE_TYPE_clean'],
        normalize='index'
    ) * 100
).round(1)
