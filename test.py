# Définition du dictionnaire de correspondance
mapping_segments = {
    "A": "City",
    "B": "Mid Small",
    "C": "Mid Large",
    "D": "Mid Large",
    "E": "High / Luxury",
    "F": "High / Luxury"
}

# Remplacement direct dans la colonne source
nova["CON_CLF_SEGMENT"] = nova["CON_CLF_SEGMENT"].map(mapping_segments)

# Ensuite, vous pouvez exécuter votre crosstab normalement
df_1 = pd.crosstab(
    index=[nova["NORMALISED_VEHICLE_TYPE"], nova["CON_CLF_SEGMENT"]], 
    columns=nova["COUNTRY"]
)

