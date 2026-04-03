
import pandas as pd

def kpi_Count_Share_YTD_by_quarter(df, asset_status, var, bike_or_car='CAR'):
    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month

    df = df[
        (df["NOVA_ASSET_STATUS"].str.contains(asset_status, na=False)) &
        (df["BIKE_OR_CAR"] == bike_or_car)
    ]

    df = df.drop_duplicates(subset=['ID_CONTRACT', 'VEHICLE_ID', 'ID_QUOTATION'])

    # Quarter simple
    df["Quarter"] = ((df["MONTH"] - 1) // 3) + 1
    df["Quarter"] = 'Q' + df["Quarter"].astype(str)

    # on cumule par COUNTRY + var + YEAR + mois croissant
    df = df.sort_values(['COUNTRY', var, 'YEAR', 'MONTH'])
    df['VOLUME_CUM'] = df.groupby(['COUNTRY', var, 'YEAR'])['VEHICLE_ID'].cumsum()

    # on regroupe par COUNTRY + var + YEAR + Quarter
    grouped = df.groupby(['COUNTRY', var, 'YEAR', 'Quarter']).agg(
        VOLUME_YTD=('VOLUME_CUM', 'max')
    ).reset_index()

    # total YTD par COUNTRY + YEAR + Quarter pour le share
    grouped["TOTAL_YTD"] = grouped.groupby(['COUNTRY', 'YEAR', 'Quarter'])['VOLUME_YTD'].transform('sum')

    # share YTD
    grouped["SHARE_YTD"] = grouped["VOLUME_YTD"] / grouped["TOTAL_YTD"] * 100

    return grouped


import pandas as pd

def kpi_top_per_quarter_with_share(df, asset_status, var_col, bike_or_car='CAR'):
    """
    df : dataframe source
    asset_status : str ou regex pour filtrer NOVA_ASSET_STATUS
    var_col : colonne à analyser (ex: 'BRAND_UPDATE', 'MODEL', etc.)
    bike_or_car : 'CAR' ou 'BIKE'
    """
    df = df[(df["NOVA_ASSET_STATUS"].str.contains(asset_status, na=False)) &
            (df["BIKE_OR_CAR"] == bike_or_car)]
    df = df.drop_duplicates(subset=['VEHICLE_ID'])

    # année et quarter
    df["YEAR"] = df["COB_DATE"].dt.year
    df["QUARTER"] = df["COB_DATE"].dt.to_period('Q')

    results = []

    countries = df["COUNTRY"].unique()
    for country in countries:
        df_country = df[df["COUNTRY"] == country]
        row_data = {"COUNTRY": country}  # une seule ligne par country

        for year in sorted(df_country["YEAR"].unique()):
            df_year = df_country[df_country["YEAR"] == year]

            for q, df_q in df_year.groupby("QUARTER"):
                counts = df_q.groupby(var_col)["VEHICLE_ID"].count()
                total_volume = counts.sum()

                if counts.empty:
                    top_var = None
                    top_volume = 0
                    top_share = 0
                else:
                    top_var = counts.idxmax()
                    top_volume = counts.max()
                    top_share = round((top_volume / total_volume) * 100, 2)

                # colonnes = année+quarter
                row_data[f"{q}_VAR"] = top_var
                row_data[f"{q}_VOLUME"] = top_volume
                row_data[f"{q}_SHARE"] = top_share

        results.append(row_data)

    return pd.DataFrame(results)




import pandas as pd

def kpi_top_per_quarter_YTD_with_share(df, asset_status, var_col, bike_or_car='CAR'):
    """
    KPI Top X par quarter, YTD (cumulé depuis début d'année)
    df : dataframe source
    asset_status : str ou regex pour filtrer NOVA_ASSET_STATUS
    var_col : colonne à analyser (ex: 'BRAND_UPDATE', 'MODEL', etc.)
    bike_or_car : 'CAR' ou 'BIKE'
    """
    df = df[(df["NOVA_ASSET_STATUS"].str.contains(asset_status, na=False)) &
            (df["BIKE_OR_CAR"] == bike_or_car)]
    df = df.drop_duplicates(subset=['VEHICLE_ID'])

    # année et quarter
    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month
    df = df.sort_values(['COUNTRY', var_col, 'YEAR', 'MONTH'])

    results = []
    countries = df["COUNTRY"].unique()

    for country in countries:
        df_country = df[df["COUNTRY"] == country]
        row_data = {"COUNTRY": country}

        for year in sorted(df_country["YEAR"].unique()):
            df_year = df_country[df_country["YEAR"] == year]

            # cumuler VOLUME pour YTD
            df_year['VOLUME_CUM'] = df_year.groupby(var_col)['VEHICLE_ID'].cumsum()

            for q, df_q in df_year.groupby(((df_year["MONTH"]-1)//3)+1):  # Q1=Jan-Mar etc
                quarter_label = f"Q{q}"
                counts = df_q.groupby(var_col)['VOLUME_CUM'].max()
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




import plotly.graph_objects as go

def plot_top_var_kpi(df_kpi):
    """
    df_kpi : sortie de kpi_top_per_quarter_with_share ou YTD
    Colonnes attendues : COUNTRY, <YEAR>Q<quarter>_VAR, _VOLUME, _SHARE
    Chaque ligne = 1 COUNTRY
    """

    fig = go.Figure()

    for idx, row in df_kpi.iterrows():  # 1 ligne par country
        country = row["COUNTRY"]

        # Toutes les colonnes *_VOLUME
        volume_cols = [c for c in df_kpi.columns if "_VOLUME" in c]

        x_labels = []
        volumes = []
        shares = []

        for col in volume_cols:
            var_col = col.replace("_VOLUME", "_VAR")
            share_col = col.replace("_VOLUME", "_SHARE")

            var_value = row.get(var_col, "No Value")
            volume = row.get(col, 0)
            share = row.get(share_col, 0)

            if pd.isna(var_value) or volume == 0:
                continue

            x_labels.append(f"{var_value} ({col[:6]})")  # label = Value (2023Q1)
            volumes.append(volume)
            shares.append(share)

        # Bar = volume
        fig.add_trace(go.Bar(
            x=x_labels,
            y=volumes,
            name=f"{country} Volume",
            legendgroup=country
        ))

        # Scatter = share %
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=shares,
            mode="lines+markers",
            name=f"{country} Share",
            legendgroup=country,
            yaxis="y2"
        ))

    # Layout avec 2 axes y
    fig.update_layout(
        title="Top Variable Volume & Share per Quarter",
        barmode="group",
        yaxis=dict(title="Volume"),
        yaxis2=dict(title="Share (%)", overlaying='y', side='right'),
        height=600
    )

    return fig
