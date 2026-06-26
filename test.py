# On ajoute BIKE_OR_CAR dans la liste des colonnes
a = pd.crosstab(
    index=nova['COUNTRY'], 
    columns=[nova['CDN_CLF_SEGMENT'], nova['BIKE_OR_CAR']],
    margins=True,       # Ajoute les totaux pour la lisibilité
    margins_name="Total"
)

# Puis on exporte avec la fonction que nous avons créée
exporter_excel_debug(a, "analyse_pays_segment_vehicule")
