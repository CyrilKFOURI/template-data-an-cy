# Mettez à jour votre dictionnaire avec les valeurs manquantes si nécessaire
# ou gardez-le tel quel pour laisser les autres valeurs tranquilles
mapping_segments = {
    "A": "City",
    "B": "Mid Small",
    "C": "Mid Large",
    "D": "Mid Large",
    "E": "High / Luxury",
    "F": "High / Luxury"
}

# Utilisez replace() au lieu de map()
nova["CDN_CLF_SEGMENT"] = nova["CDN_CLF_SEGMENT"].replace(mapping_segments)
