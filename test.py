def kpi_top_brand_vs_market(df_portfolio, df_market, var_col_portfolio, var_col_market):

    # --- PORTFOLIO ---
    df_portfolio["QUARTER"] = df_portfolio["COB_DATE"].dt.to_period('Q')

    port = (
        df_portfolio
        .groupby(["COUNTRY", "QUARTER", var_col_portfolio])["VEHICLE_ID"]
        .count()
        .reset_index(name="volume_portfolio")
    )

    port["total_portfolio"] = port.groupby(["COUNTRY", "QUARTER"])["volume_portfolio"].transform("sum")
    port["share_portfolio"] = port["volume_portfolio"] / port["total_portfolio"]

    port = port.rename(columns={var_col_portfolio: "BRAND"})

    # TOP BRAND PAR COUNTRY / QUARTER
    port_top = (
        port.sort_values("volume_portfolio", ascending=False)
        .groupby(["COUNTRY", "QUARTER"])
        .head(1)
    )


    # --- MARKET ---
    df_market["QUARTER"] = df_market["date"].dt.to_period('Q')

    market = (
        df_market
        .groupby(["Country/Territory-Number", "QUARTER", var_col_market])["volume"]
        .sum()
        .reset_index(name="volume_market")
    )

    market["total_market"] = market.groupby(["Country/Territory-Number", "QUARTER"])["volume_market"].transform("sum")
    market["share_market"] = market["volume_market"] / market["total_market"]

    market = market.rename(columns={
        "Country/Territory-Number": "COUNTRY",
        var_col_market: "BRAND"
    })


    # --- MERGE sur la top brand ---
    df_final = port_top.merge(
        market,
        on=["COUNTRY", "QUARTER", "BRAND"],
        how="left"
    )

    # --- RATIO ---
    df_final["share_ratio"] = df_final["share_portfolio"] / df_final["share_market"]

    return df_final
