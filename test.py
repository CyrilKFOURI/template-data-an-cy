def kp19_1_type_share_quarter(df, year, vehicle_type, country, asset_status, bike_or_car='CAR'):
    df = df[(df["YEAR"] == year) & (df["NOVA_ASSET_STATUS"] == asset_status) & (df["MARKET_BODY_GROUP"] == vehicle_type) & (df["COUNTRY"] == country)].copy()

    df["Quarter"] = ((df["MONTH"] - 1) // 3 + 1).apply(lambda x: f"Q{x}")

    key_cols = ["ID_QUOTATION", "ID_CONTRACT", "VEHICLE_ID"]
    df = df.drop_duplicates(subset=key_cols)

    grouped = df.groupby(["POWER_CATEGORY", "Quarter"]).size().reset_index(name="COUNT")

    pivot = grouped.pivot(index="POWER_CATEGORY", columns="Quarter", values="COUNT").fillna(0)

    share = pivot.div(pivot.sum(axis=0), axis=1) * 100

    return share.round(2)