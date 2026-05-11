import pandas as pd

# =========================
# 1. CLEAN / NORMALISATION
# =========================
def norm(s):
    return (
        s.fillna("")
         .astype(str)
         .str.upper()
         .str.strip()
    )

spain_202512_models["MODEL"] = norm(spain_202512_models["MODEL"])
a["MODEL"] = norm(a["MODEL"])
spain_202512_models["BRAND_UPDATE"] = norm(spain_202512_models["BRAND_UPDATE"])
a["BRAND"] = norm(a["BRAND"])

# =========================
# 2. INIT OUTPUT COL
# =========================
spain_202512_models["MODEL_2"] = None

# =========================
# 3. MATCHING LOGIC
# =========================
for brand in a["BRAND"].dropna().unique():

    a_brand = a[a["BRAND"] == brand]

    nova_mask_brand = spain_202512_models["BRAND_UPDATE"] == brand

    for model_a in a_brand["MODEL"].dropna().unique():

        if model_a == "":
            continue

        mask = (
            nova_mask_brand &
            spain_202512_models["MODEL"].str.contains(
                model_a,
                na=False,
                regex=False
            )
        )

        spain_202512_models.loc[mask, "MODEL_2"] = model_a