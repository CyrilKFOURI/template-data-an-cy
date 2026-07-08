top10 = (
    nova.groupby(["COUNTRY", "BRAND_UPDATE", "MARKET_MODEL"])
        .size()
        .reset_index(name="Count")
        .sort_values(["COUNTRY", "Count"], ascending=[True, False])
        .groupby("COUNTRY")
        .head(10)
)

print(top10)