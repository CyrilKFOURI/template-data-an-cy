def kp19_1_type_share_quarter(df, year, vehicle_type, country, asset_status, bike_or_car='CAR'):
    df = df[(df["YEAR"] == year) & (df["NOVA_ASSET_STATUS"] == asset_status) & (df["MARKET_BODY_GROUP"] == vehicle_type) & (df["COUNTRY"] == country)].copy()

    df["Quarter"] = ((df["MONTH"] - 1) // 3 + 1).apply(lambda x: f"Q{x}")

    key_cols = ["ID_QUOTATION", "ID_CONTRACT", "VEHICLE_ID"]
    df = df.drop_duplicates(subset=key_cols)

    grouped = df.groupby("Quarter")["VEHICLE_ID"].nunique().reset_index(name="COUNT")

    total = grouped["COUNT"].sum()

    grouped["SHARE"] = (grouped["COUNT"] / total) * 100

    return grouped[["Quarter", "SHARE"]].round(2)