top10 = (
    nova.groupby(["COUNTRY", "BRAND_UPDATE", "MARKET_MODEL"])
        .size()
        .groupby(level=0)
        .nlargest(10)
        .reset_index(name="Count")
)

print(top10)