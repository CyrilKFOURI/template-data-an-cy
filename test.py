def kp19_1_type_share_quarter(df, year, vehicle_type, country, asset_status, bike_or_car='CAR'):
    df = df[(df["YEAR"] == year) & (df["NOVA_ASSET_STATUS"] == asset_status) & (df["COUNTRY"] == country)].copy()

    key_cols = ["ID_QUOTATION", "ID_CONTRACT", "VEHICLE_ID"]
    df = df.drop_duplicates(subset=key_cols)

    df["Quarter"] = ((df["MONTH"] - 1) // 3 + 1).apply(lambda x: f"Q{x}")

    total = df.groupby("Quarter")["VEHICLE_ID"].count()
    type_count = df[df["MARKET_BODY_GROUP"] == vehicle_type].groupby("Quarter")["VEHICLE_ID"].count()

    pct = (type_count / total * 100).fillna(0).round(2)

    return pct.reset_index(name="PCT")