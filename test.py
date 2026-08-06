counts = (
    nova
    .groupby(["COUNTRY", "MARKET_MODEL"])
    .size()
    .reset_index(name="Count")
)

top10_par_pays = (
    counts
    .sort_values(["COUNTRY", "Count"], ascending=[True, False])
    .groupby("COUNTRY")
    .head(10)
    .drop(columns="Count")
    .reset_index(drop=True)
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

# Renommer les colonnes
top10_par_pays = top10_par_pays.rename(columns={
    "MARKET_MODEL": "MODEL",
    "BRAND_UPDATE": "BRAND"
})

# Export Excel
top10_par_pays.to_excel(
    "Top10_Models_By_Country.xlsx",
    index=False
)




counts = (
    nova
    .groupby("MARKET_MODEL")
    .size()
    .reset_index(name="Count")
)

top10_global = (
    counts
    .sort_values("Count", ascending=False)
    .head(10)
    .drop(columns="Count")
    .reset_index(drop=True)
)

brand_map = (
    nova
    .drop_duplicates("MARKET_MODEL")
    .set_index("MARKET_MODEL")["BRAND_UPDATE"]
)

top10_global["BRAND_UPDATE"] = (
    top10_global["MARKET_MODEL"].map(brand_map)
)

# Renommer les colonnes
top10_global = top10_global.rename(columns={
    "MARKET_MODEL": "MODEL",
    "BRAND_UPDATE": "BRAND"
})

# Export Excel
top10_global.to_excel(
    "Top10_Models_Global.xlsx",
    index=False
)