import numpy as np

# init colonne
spain_202512["MODEL_2"] = np.nan

for brand in a["BRAND"].dropna().unique():

    # subset brand
    nova_brand = spain_202512[spain_202512["BRAND"] == brand]
    a_brand = a[a["BRAND"] == brand]

    for model_a in a_brand["MODEL"].dropna().unique():

        mask = (
            (spain_202512["BRAND"] == brand) &
            (spain_202512["MODEL"].str.contains(model_a, na=False))
        )

        spain_202512.loc[mask, "MODEL_2"] = model_a