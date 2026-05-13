import pandas as pd

# =========================
# 0. COPY SAFE
# =========================
nova = nova.copy()
nova_models = nova_models.copy()
market_models = market_models.copy()

# =========================
# 1. NETTOYER DOUBLONS CLÉS (CRITIQUE)
# =========================
nova_models = nova_models.drop_duplicates(
    subset=["BRAND_UPDATE", "MODEL"]
)

market_models = market_models.drop_duplicates(
    subset=["Make", "Sub Model Short"]
)

# =========================
# 2. STEP 1 : MODEL MATCH
# =========================
nova = nova.merge(
    nova_models[["BRAND_UPDATE", "MODEL", "MODEL_2"]],
    how="left",
    on=["BRAND_UPDATE", "MODEL"]
)

nova["MARKET_MODEL"] = nova["MODEL_2"]
nova.drop(columns=["MODEL_2"], inplace=True)

# =========================
# 3. STEP 2 : BODY GROUP MATCH
# =========================
nova = nova.merge(
    market_models[["Make", "Sub Model Short", "Body Group"]],
    how="left",
    left_on=["BRAND_UPDATE", "MARKET_MODEL"],
    right_on=["Make", "Sub Model Short"],
    how="left"
)

nova.drop(columns=["Make", "Sub Model Short"], inplace=True)

# =========================
# 4. (OPTIONNEL) CLEAN FINAL
# =========================
nova.rename(columns={"Body Group": "BODY_GROUP"}, inplace=True)

# =========================
# 5. CHECKS (IMPORTANT)
# =========================
print("Rows final:", len(nova))
print("Missing MARKET_MODEL:", nova["MARKET_MODEL"].isna().sum())
print("Missing BODY_GROUP:", nova["BODY_GROUP"].isna().sum())