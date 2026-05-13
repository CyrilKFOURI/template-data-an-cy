import pandas as pd

print("===== 1. CHECK SHAPE =====")
print("nova:", nova.shape)
print("remarketing:", remarketing_classification_p.shape)


# =========================
# 2. CHECK DUPLICATES (CRUCIAL)
# =========================
print("\n===== 2. DUPLICATES CHECK =====")

dup_right = remarketing_classification_p.duplicated(
    subset=["BRAND", "MODEL_NOVA"]
).sum()

print("Duplicates (BRAND, MODEL_NOVA) in remarketing:", dup_right)


# =========================
# 3. CLEAN RIGHT TABLE (SAFE MERGE)
# =========================
remarketing_clean = remarketing_classification_p.drop_duplicates(
    subset=["BRAND", "MODEL_NOVA"]
)

print("Right table after dedup:", remarketing_clean.shape)


# =========================
# 4. CHECK KEY OVERLAP
# =========================
print("\n===== 3. KEY OVERLAP CHECK =====")

nova_keys = set(zip(nova["BRAND_UPDATE"], nova["MARKET_MODEL"]))
right_keys = set(zip(remarketing_clean["BRAND"], remarketing_clean["MODEL_NOVA"]))

print("Unique nova keys:", len(nova_keys))
print("Unique right keys:", len(right_keys))
print("Intersection:", len(nova_keys & right_keys))


# =========================
# 5. TEST INNER JOIN RATE
# =========================
print("\n===== 4. INNER JOIN TEST =====")

test_join = nova.merge(
    remarketing_clean,
    how="inner",
    left_on=["BRAND_UPDATE", "MARKET_MODEL"],
    right_on=["BRAND", "MODEL_NOVA"]
)

print("Inner join rows:", len(test_join))
print("Match rate:", len(test_join) / len(nova))


# =========================
# 6. REAL MERGE (SAFE)
# =========================
print("\n===== 5. FINAL MERGE =====")

nova = nova.merge(
    remarketing_clean[["BRAND", "MODEL_NOVA", "CDN_CLF_SEGMENT"]],
    how="left",
    left_on=["BRAND_UPDATE", "MARKET_MODEL"],
    right_on=["BRAND", "MODEL_NOVA"]
)

nova.drop(columns=["BRAND", "MODEL_NOVA"], inplace=True)


# =========================
# 7. FINAL CHECKS
# =========================
print("\n===== 6. FINAL CHECK =====")

print("Final rows:", len(nova))
print("CDN NULL %:", nova["CDN_CLF_SEGMENT"].isna().mean())
print("CDN filled:", nova["CDN_CLF_SEGMENT"].notna().sum())