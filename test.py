# On met COUNTRY en index (lignes)
# On met les deux dimensions dans une liste pour les colonnes (hiérarchie)
a = pd.crosstab(
    index=nova['COUNTRY'], 
    columns=[nova['CDN_CLF_SEGMENT'], nova['BIKE_OR_CAR']]
)

# On affiche pour vérifier
print(a)

# On exporte avec la fonction
exporter_excel_debug(a, "analyse_finale")
