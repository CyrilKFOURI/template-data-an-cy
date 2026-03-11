def kpi8_volume_per_model_quarter(self, asset_status, end_year):
    df = self.df[self.df["ASSET_STATUS"] == asset_status].copy()
    df = df.dropna(subset=["PLANNED_RETURN_DATE"])
    df["PLANNED_RETURN_DATE"] = pd.to_datetime(df["PLANNED_RETURN_DATE"])
    
    df = df[df["PLANNED_RETURN_DATE"].dt.year <= end_year]
    
    df["Quarter"] = df["PLANNED_RETURN_DATE"].dt.to_period("Q")
    
    pivot = df.pivot_table(
        index="Quarter",
        columns="VEHICLE_MODEL_2",
        values="VEHICLE_ID",
        aggfunc="count",
        fill_value=0
    )
    
    total_row = pd.DataFrame(pivot.sum()).T
    total_row.index = ["Total"]
    
    pivot = pd.concat([pivot, total_row])
    
    # Trier les colonnes par le total décroissant
    col_order = pivot.loc["Total"].sort_values(ascending=False).index
    pivot = pivot[col_order]
    
    pivot.reset_index(inplace=True)
    
    return pivot
