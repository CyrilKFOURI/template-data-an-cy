counts = (
    nova
    .groupby(["COUNTRY", "MARKET_MODEL"])
    .size()
    .reset_index(name="Volume")
)

top10_par_pays = (
    counts
    .sort_values(["COUNTRY", "Volume"], ascending=[True, False])
    .groupby("COUNTRY")
    .head(10)
    .reset_index(drop=True)
)

top10_par_pays["Share (%)"] = (
    top10_par_pays
    .groupby("COUNTRY")["Volume"]
    .transform(lambda x: (x / x.sum() * 100).round(1))
)

brand_map = (
    nova
    .drop_duplicates(["COUNTRY", "MARKET_MODEL"])
    .set_index(["COUNTRY", "MARKET_MODEL"])["BRAND_UPDATE"]
)

top10_par_pays["BRAND_UPDATE"] = (
    top10_par_pays
    .set_index(["COUNTRY", "MARKET_MODEL"])
    .index
    .map(brand_map)
)
