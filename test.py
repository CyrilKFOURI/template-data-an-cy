
import pandas as pd

def kpi_top_brand_per_quarter_with_share(df, asset_status, bike_or_car='CAR'):
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
                brand_counts = df_q.groupby("BRAND_UPDATE")["VEHICLE_ID"].count()
                total_volume = brand_counts.sum()

                if brand_counts.empty:
                    top_brand = None
                    top_volume = 0
                    top_share = 0
                else:
                    top_brand = brand_counts.idxmax()
                    top_volume = brand_counts.max()
                    top_share = round((top_volume / total_volume) * 100, 2)

                # colonnes = année+quarter
                row_data[f"{q}_BRAND"] = top_brand
                row_data[f"{q}_VOLUME"] = top_volume
                row_data[f"{q}_SHARE"] = top_share

        results.append(row_data)

    return pd.DataFrame(results)




import plotly.graph_objects as go

def plot_top_brand_kpi(df_kpi):
    """
    df_kpi : sortie de kpi_top_brand_per_quarter_with_share
    Colonnes attendues : COUNTRY, <YEAR>Q<quarter>_BRAND, _VOLUME, _SHARE
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
            brand_col = col.replace("_VOLUME","_BRAND")
            share_col = col.replace("_VOLUME","_SHARE")

            brand = row.get(brand_col, "No Brand")
            volume = row.get(col, 0)
            share = row.get(share_col, 0)

            if pd.isna(brand) or volume == 0:
                continue

            x_labels.append(f"{brand} ({col[:6]})")  # label = Brand (2023Q1)
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
        title="Top Brand Volume & Share per Quarter",
        barmode="group",
        yaxis=dict(title="Volume"),
        yaxis2=dict(title="Share (%)", overlaying='y', side='right'),
        height=600
    )

    return fig
