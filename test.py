def kpi8_volume_per_model_quarter(self, asset_status):
    df = self.df[self.df["ASSET_STATUS"] == asset_status].copy()
    df = df.dropna(subset=["PLANNED_RETURN_DATE"])
    df["PLANNED_RETURN_DATE"] = pd.to_datetime(df["PLANNED_RETURN_DATE"])
    
    df["Quarter"] = df["PLANNED_RETURN_DATE"].dt.to_period("Q")
    
    pivot = df.pivot_table(
        index="Quarter",
        columns="VEHICLE_MODEL_2",
        values="VEHICLE_ID",
        aggfunc="count",
        fill_value=0
    )
    
    total_row = pivot.sum()
    total_row.name = "Total"
    pivot = pivot.append(total_row)
    
    pivot.reset_index(inplace=True)
    return pivot



def kpi13_top_brand_per_quarter(self, asset_status, year):
    df = self.df[(self.df["ASSET_STATUS"]==asset_status) & (self.df["YEAR"]==year)].copy()
    
    # Crée la colonne Quarter à partir du mois
    df["Quarter"] = ((df["MONTH"]-1)//3 + 1).apply(lambda x: f"Q{x}")
    
    countries = df["COUNTRY"].unique()
    results = []

    for country in countries:
        df_country = df[df["COUNTRY"]==country]
        row_data = {"COUNTRY": country}
        quarters = sorted(df_country["Quarter"].unique())
        for i, q in enumerate(quarters, 1):
            df_q = df_country[df_country["Quarter"]==q]
            brand_counts = df_q.groupby("BRAND")["VEHICLE_ID"].count()
            if len(brand_counts) == 0:
                top_brand = None
                top_volume = 0
            else:
                top_brand = brand_counts.idxmax()
                top_volume = brand_counts.max()
            row_data[f"{q}_VOLUME"] = top_volume
            row_data[f"{q}_BRAND"] = top_brand
        results.append(row_data)
    
    return pd.DataFrame(results)
