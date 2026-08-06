counts = (
    nova
    .groupby(["COUNTRY", "MODEL"])
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
    .drop_duplicates(["COUNTRY", "MODEL"])
    .set_index(["COUNTRY", "MODEL"])["BRAND"]
)

top10_par_pays["BRAND"] = (
    top10_par_pays
    .set_index(["COUNTRY", "MODEL"])
    .index
    .map(brand_map)
)



counts = (
    nova
    .groupby("MODEL")
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
    .drop_duplicates("MODEL")
    .set_index("MODEL")["BRAND"]
)

top10_global["BRAND"] = (
    top10_global["MODEL"].map(brand_map)
)

