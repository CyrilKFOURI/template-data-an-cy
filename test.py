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

        for year in df_country["YEAR"].unique():
            df_year = df_country[df_country["YEAR"] == year]

            row_data = {"COUNTRY": country, "YEAR": year}

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
                    top_share = round((top_volume / total_volume) * 100, 2)  # en %

                row_data[f"{q}_BRAND"] = top_brand
                row_data[f"{q}_VOLUME"] = top_volume
                row_data[f"{q}_SHARE"] = top_share

            results.append(row_data)

    return pd.DataFrame(results)




import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_top_brand(df_plot):
    """
    df_plot = output de kpi_top_brand_per_quarter_with_share
    Colonnes attendues : COUNTRY, YEAR, <QUARTER>_BRAND, <QUARTER>_VOLUME, <QUARTER>_SHARE
    """

    countries = df_plot["COUNTRY"].unique()
    n_cols = len(countries)
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=countries, specs=[[{"secondary_y": True}]*n_cols])

    for i, country in enumerate(countries, 1):
        df_c = df_plot[df_plot["COUNTRY"] == country]

        # Pour chaque quarter présent
        quarters = sorted([c for c in df_c.columns if "_VOLUME" in c])
        for q in quarters:
            brand_col = q.replace("_VOLUME", "_BRAND")
            share_col = q.replace("_VOLUME", "_SHARE")

            df_c_q = df_c[[brand_col, q, share_col]].dropna()
            if df_c_q.empty:
                continue

            # Bar = volume
            fig.add_trace(go.Bar(
                x=df_c_q[brand_col],
                y=df_c_q[q],
                name=f"{q} Volume",
                legendgroup=f"{q}",
                showlegend=(i==1)
            ), row=1, col=i, secondary_y=False)

            # Scatter = share %
            fig.add_trace(go.Scatter(
                x=df_c_q[brand_col],
                y=df_c_q[share_col],
                mode="lines+markers",
                name=f"{q} Share",
                legendgroup=f"{q}",
                showlegend=(i==1)
            ), row=1, col=i, secondary_y=True)

        fig.update_yaxes(title_text="Volume", secondary_y=False, row=1, col=i)
        fig.update_yaxes(title_text="Share (%)", secondary_y=True, row=1, col=i)

    fig.update_layout(
        title="Top Brand Volume & Share per Quarter",
        barmode="group",
        height=500,
        width=300*n_cols
    )

    return fig
