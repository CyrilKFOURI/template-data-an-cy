# sécurité types
nova["BRAND_UPDATE"] = nova["BRAND_UPDATE"].astype(str)
nova["MARKET_MODEL"] = nova["MARKET_MODEL"].astype(str)

remarketing_classification_p["BRAND"] = remarketing_classification_p["BRAND"].astype(str)
remarketing_classification_p["MODEL_NOVA"] = remarketing_classification_p["MODEL_NOVA"].astype(str)

# éviter doublons côté mapping
mapping = remarketing_classification_p[["BRAND", "MODEL_NOVA", "CDN_CLF_SEGMENT"]].drop_duplicates()

# merge
nova = nova.merge(
    mapping,
    how="left",
    left_on=["BRAND_UPDATE", "MARKET_MODEL"],
    right_on=["BRAND", "MODEL_NOVA"]
)

# clean
nova.drop(columns=["BRAND", "MODEL_NOVA"], inplace=True)