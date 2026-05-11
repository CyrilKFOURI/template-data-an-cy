import pandas as pd

# force types
spain_202512_models["MODEL"] = spain_202512_models["MODEL"].astype("string")
a["MODEL"] = a["MODEL"].astype("string")

# IMPORTANT: init propre en string
spain_202512_models["MODEL_2"] = None

for brand in a["BRAND"].dropna().unique():

    a_brand = a[a["BRAND"] == brand]

    nova_mask_brand = spain_202512_models["BRAND_UPDATE"] == brand

    for model_a in a_brand["MODEL"].dropna().unique():

        mask = (
            nova_mask_brand &
            spain_202512_models["MODEL"].str.contains(
                model_a,
                na=False,
                regex=False
            )
        )

        spain_202512_models.loc[mask, "MODEL_2"] = model_a