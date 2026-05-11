import numpy as np

spain_202512_models["MODEL_2"] = np.nan

# force proprement les types une seule fois (IMPORTANT)
spain_202512_models["MODEL"] = spain_202512_models["MODEL"].fillna("").astype(str)
a["MODEL"] = a["MODEL"].fillna("").astype(str)

for brand in a["BRAND"].dropna().unique():

    a_brand = a[a["BRAND"] == brand]
    nova_mask_brand = spain_202512_models["BRAND_UPDATE"] == brand

    for model_a in a_brand["MODEL"].unique():

        if model_a == "":
            continue

        mask = (
            nova_mask_brand &
            spain_202512_models["MODEL"].str.contains(
                model_a,
                na=False,
                regex=False   # 🔥 IMPORTANT (évite bugs regex + accélère)
            )
        )

        spain_202512_models.loc[mask, "MODEL_2"] = model_a