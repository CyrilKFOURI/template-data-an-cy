import pandas as pd

def kpi_Count_Share_YTD_by_quarter_fixed(df, asset_status, var, bike_or_car='CAR'):
    df = df[
        (df["NOVA_ASSET_STATUS"].str.contains(asset_status, na=False)) &
        (df["BIKE_OR_CAR"] == bike_or_car)
    ]
    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month
    df = df.drop_duplicates(subset=['ID_CONTRACT', 'VEHICLE_ID', 'ID_QUOTATION'])
    df["Quarter"] = ((df["MONTH"] - 1) // 3) + 1
    df["Quarter"] = 'Q' + df["Quarter"].astype(str)
    df['VOLUME_MONTH'] = 1
    df = df.sort_values(['COUNTRY', var, 'YEAR', 'MONTH'])
    df['VOLUME_YTD'] = df.groupby(['COUNTRY', var, 'YEAR'])['VOLUME_MONTH'].cumsum()
    grouped = df.groupby(['COUNTRY', var, 'YEAR', 'Quarter']).agg(
        VOLUME_YTD=('VOLUME_YTD', 'max')
    ).reset_index()
    grouped["TOTAL_YTD"] = grouped.groupby(['COUNTRY', 'YEAR', 'Quarter'])['VOLUME_YTD'].transform('sum')
    grouped["SHARE_YTD"] = grouped["VOLUME_YTD"] / grouped["TOTAL_YTD"] * 100
    return grouped


import pandas as pd

def kpi_top_per_quarter_YTD_with_share_fixed(df, asset_status, var_col, bike_or_car='CAR'):
    """
    KPI Top par quarter, YTD (cumulé depuis début d'année)
    var_col : colonne dynamique à analyser (ex: BRAND_UPDATE)
    """
    df = df[(df["NOVA_ASSET_STATUS"].str.contains(asset_status, na=False)) &
            (df["BIKE_OR_CAR"] == bike_or_car)]
    df = df.drop_duplicates(subset=['VEHICLE_ID'])

    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month

    results = []
    countries = df["COUNTRY"].unique()

    for country in countries:
        df_country = df[df["COUNTRY"] == country]
        row_data = {"COUNTRY": country}

        for year in sorted(df_country["YEAR"].unique()):
            df_year = df_country[df_country["YEAR"] == year].sort_values("MONTH")

            # cumuler YTD par var_col
            df_year['MONTH_RANK'] = df_year.groupby(var_col)['MONTH'].rank(method='first')
            df_year['VOLUME_YTD'] = df_year.groupby(var_col).cumcount() + 1  # nombre cumulatif par var_col

            for q, df_q in df_year.groupby(((df_year["MONTH"]-1)//3)+1):  # Q1=Jan-Mar
                quarter_label = f"Q{q}"
                counts = df_q.groupby(var_col)['VOLUME_YTD'].max()
                total_volume = counts.sum()

                if counts.empty:
                    top_var = None
                    top_volume = 0
                    top_share = 0
                else:
                    top_var = counts.idxmax()
                    top_volume = counts.max()
                    top_share = round((top_volume / total_volume) * 100, 2)

                row_data[f"{year}{quarter_label}_VAR"] = top_var
                row_data[f"{year}{quarter_label}_VOLUME_YTD"] = top_volume
                row_data[f"{year}{quarter_label}_SHARE_YTD"] = top_share

        results.append(row_data)

    return pd.DataFrame(results)
