def kp19_1_type_share_quarter(df, year, vehicle_type, country, asset_status, bike_or_car='CAR'):
    df = df[(df["YEAR"] == year) & (df["NOVA_ASSET_STATUS"] == asset_status) & (df["COUNTRY"] == country)].copy()

    key_cols = ["ID_QUOTATION", "ID_CONTRACT", "VEHICLE_ID"]
    df = df.drop_duplicates(subset=key_cols)

    total = len(df)

    type_count = (df["MARKET_BODY_GROUP"] == vehicle_type).sum()

    pct = (type_count / total) * 100 if total > 0 else 0

    return round(pct, 2)