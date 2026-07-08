
top10 = (
    df["VEHICLE_MODEL"]
    .value_counts()
    .head(10)
    .rename_axis("VEHICLE_MODEL")
    .reset_index(name="Count")
)

# Ajouter la marque correspondante
top10["BRAND"] = top10["VEHICLE_MODEL"].map(
    df.drop_duplicates("VEHICLE_MODEL").set_index("VEHICLE_MODEL")["BRAND"]
)

print(top10)