def kpi10_volume_per_model_quarter_EOC(
    df,
    NOVA_ASSET_STATUS,
    start_year,
    end_year,
    country,
    bike_or_car="CAR"
):
    df = df[
        (df["NOVA_ASSET_STATUS"] == NOVA_ASSET_STATUS)
        & (df["COUNTRY"] == country)
    ].copy()

    key_cols = ["ID_QUOTATION", "ID_CONTRACT", "VEHICLE_ID"]
    df = df.drop_duplicates(subset=key_cols)

    df = df.dropna(subset=["CONTRACT_FINAL_END"])
    df["CONTRACT_FINAL_END"] = pd.to_datetime(df["CONTRACT_FINAL_END"])

    df = df[
        (df["CONTRACT_FINAL_END"].dt.year >= start_year)
        & (df["CONTRACT_FINAL_END"].dt.year <= end_year)
    ]

    df["Quarter"] = df["CONTRACT_FINAL_END"].dt.to_period("Q")

    pivot = df.pivot_table(
        index="Quarter",
        columns="VEHICLE_MODEL_MAPED",
        values="VEHICLE_ID",
        aggfunc="nunique",
        fill_value=0
    )

    total_row = pd.DataFrame(pivot.sum()).T
    total_row.index = ["Total"]

    pivot = pd.concat([pivot, total_row])

    col_order = pivot.loc["Total"].sort_values(ascending=False).index
    pivot = pivot[col_order]

    pivot = pivot.reset_index()

    return pivot