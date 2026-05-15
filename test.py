def kp19_2_power_category_per_type_quarter(
    df,
    year,
    vehicle_type,
    country,
    asset_status,
    bike_or_car='CAR'
):

    # Filtre
    df = df[
        (df["YEAR"] == year)
        & (df["NOVA_ASSET_STATUS"] == asset_status)
        & (df["MARKET_BODY_GROUP"] == vehicle_type)
        & (df["COUNTRY"] == country)
    ].copy()

    # Quarter
    df["Quarter"] = (
        ((df["MONTH"] - 1) // 3) + 1
    ).apply(lambda x: f"Q{x}")

    # Drop duplicates sur la clé
    key_cols = [
        "ID_QUOTATION",
        "ID_CONTRACT",
        "VEHICLE_ID"
    ]

    df = df.drop_duplicates(subset=key_cols)

    # Groupby
    grouped = (
        df.groupby(["POWER_CATEGORY", "Quarter"])
        .size()
        .reset_index(name="COUNT")
    )

    # Pivot
    pivot = grouped.pivot(
        index="Quarter",
        columns="POWER_CATEGORY",
        values="COUNT"
    ).fillna(0)

    return pivot