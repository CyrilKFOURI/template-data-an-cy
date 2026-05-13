import pandas as pd

# =========================
# 1. SAFETY COPY
# =========================
nova = nova.copy()

# =========================
# 2. CLEAN RIGHT TABLE (NO COLLISIONS EVER)
# =========================
right = (
    remarketing_classification_p[
        ["BRAND", "MODEL_NOVA", "CDN_CLF_SEGMENT"]
    ]
    .drop_duplicates()
    .rename(columns={
        "BRAND": "BRAND_RIGHT",
        "MODEL_NOVA": "MODEL_RIGHT"
    })
)

# =========================
# 3. ENSURE TYPE CONSISTENCY
# =========================
nova["BRAND_UPDATE"] = nova["BRAND_UPDATE"].astype(str)
nova["MARKET_MODEL"] = nova["MARKET_MODEL"].astype(str)

right["BRAND_RIGHT"] = right["BRAND_RIGHT"].astype(str)
right["MODEL_RIGHT"] = right["MODEL_RIGHT"].astype(str)

# =========================
# 4. MERGE (SAFE + LEFT JOIN)
# =========================
nova = nova.merge(
    right,
    how="left",
    left_on=["BRAND_UPDATE", "MARKET_MODEL"],
    right_on=["BRAND_RIGHT", "MODEL_RIGHT"]
)

# =========================
# 5. CLEAN UP (KEEP NOVA CLEAN)
# =========================
nova.drop(
    columns=["BRAND_RIGHT", "MODEL_RIGHT"],
    inplace=True
)

# =========================
# 6. FINAL COLUMN CLEANUP
# =========================
nova.rename(
    columns={"CDN_CLF_SEGMENT": "CDN_CLF_SEGMENT"},
    inplace=True
)

# =========================
# 7. OPTIONAL CHECKS
# =========================
print("Rows:", len(nova))
print("Missing segment:", nova["CDN_CLF_SEGMENT"].isna().sum())