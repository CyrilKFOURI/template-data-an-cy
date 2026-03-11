def kpi8_top10_eoc_quarter(self, asset_status):
    df = self.df[self.df["ASSET_STATUS"] == asset_status].copy()
    df = df.dropna(subset=["PLANNED_RETURN_DATE"])

    quarters = pd.date_range(start="2026-01-01", end="2029-12-31", freq="QE")
    results = []

    for q in quarters:
        start_q = q - pd.offsets.QuarterEnd(startingMonth=q.month)
        mask = (df["PLANNED_RETURN_DATE"] > start_q) & (df["PLANNED_RETURN_DATE"] <= q)
        top10 = df.loc[mask, "VEHICLE_MODEL_2"].value_counts().head(10).reset_index()
        top10["QUARTER"] = q
        results.append(top10)

    return pd.concat(results, ignore_index=True).rename(
        columns={"index": "VEHICLE_MODEL_2", "VEHICLE_MODEL_2": "VOLUME"}
    )
