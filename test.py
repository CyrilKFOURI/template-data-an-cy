# Si vous voulez voir les valeurs de BIKE_OR_CAR dans le tableau :
a = pd.pivot_table(
    nova, 
    index='COUNTRY', 
    columns='CDN_CLF_SEGMENT', 
    values='BIKE_OR_CAR', 
    aggfunc='first' # Affiche la première valeur trouvée pour ce croisement
)
