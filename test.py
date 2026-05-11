import numpy as np

spain_202512_models["MODEL_2"] = np.nan

for brand in a["BRAND"].dropna().unique():

    # dataset a
    a_brand = a[a["BRAND"] == brand]

    # dataset nova (spain)
    nova_brand = spain_202512_models[
        spain_202512_models["BRAND_UPDATE"] == brand
    ]

    for model_a in a_brand["MODEL"].dropna().astype(str).unique():

        mask = (
            (spain_202512_models["BRAND_UPDATE"] == brand) &
            (spain_202512_models["MODEL"].astype(str).str.contains(model_a, na=False))
        )

        spain_202512_models.loc[mask, "MODEL_2"] = model_a