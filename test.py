def kp18_production_ytd(df: pd.DataFrame,
                        country: str,
                        year: int,
                        asset_status: str = "ALL",
                        metric_mode: str = "volume") -> pd.DataFrame:

    # 🔹 on garde TON filter_base
    subset = filter_base(df, country, year,
                         status="ALL",   # important
                         bike_or_car="CAR")

    # 🔹 on applique le status APRES (comme ton KPI initial)
    if asset_status != "ALL":
        subset = subset[subset["NOVA_ASSET_STATUS"] == asset_status]

    # 🔹 filtre date EXACT comme ton premier KPI
    subset = subset[
        (subset["CONTRACT_START_DATE"].dt.year == year) &
        (subset["CONTRACT_START_DATE"].dt.month == subset["MONTH"])
    ].copy()

    # 🔹 clean power category (très important)
    subset["POWER_CATEGORY"] = subset["POWER_CATEGORY"].where(
        subset["POWER_CATEGORY"].isin([
            "DIESEL", "PETROL", "FULL HYBRID",
            "PLUG-IN HYBRID", "ELECTRIC"
        ]),
        "Others"
    )

    # 🔹 groupby
    grouped = subset.groupby(["YEAR", "MONTH", "POWER_CATEGORY"]) \
                    .size() \
                    .reset_index(name="VOLUME")

    # 🔹 pivot
    table = grouped.pivot(index=["YEAR", "MONTH"],
                          columns="POWER_CATEGORY",
                          values="VOLUME") \
                   .fillna(0) \
                   .reset_index()

    # 🔹 share mode
    metric_cols = [c for c in table.columns if c not in ["YEAR", "MONTH"]]

    if metric_mode.lower() == "share":
        totals = table[metric_cols].sum(axis=1)
        table[metric_cols] = table[metric_cols].div(
            totals.replace(0, 1), axis=0
        ) * 100

    return table.round(3)