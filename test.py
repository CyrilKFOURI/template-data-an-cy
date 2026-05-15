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

    # Date
    df["COB_DATE"] = pd.to_datetime(df["COB_DATE"])

    # Quarter
    df["Quarter"] = (
        ((df["COB_DATE"].dt.month - 1) // 3) + 1
    ).apply(lambda x: f"Q{x}")

    # Clé unique
    key_cols = [
        "ID_QUOTATION",
        "ID_CONTRACT",
        "VEHICLE_ID"
    ]

    # Une seule ligne par clé et COB_DATE
    df = df.drop_duplicates(
        subset=key_cols + ["COB_DATE"]
    )

    # Dernier snapshot du quarter
    latest_cob_per_quarter = (
        df.groupby("Quarter")["COB_DATE"]
        .max()
        .reset_index()
    )

    df = df.merge(
        latest_cob_per_quarter,
        on=["Quarter", "COB_DATE"]
    )

    # Comptage
    grouped = (
        df.groupby(["Quarter", "POWER_CATEGORY"])
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