import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display


countries = sorted(nova["COUNTRY"].dropna().unique())

country_dropdown = widgets.Dropdown(
    options=countries,
    description="Country:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width="300px")
)


def plot_top_models(country):

    df_country = nova[nova["COUNTRY"] == country]

    # Top 10 modèles
    top10 = (
        df_country["MARKET_MODEL"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top10.columns = ["MARKET_MODEL", "Volume"]

    # Pourcentage
    top10["Share (%)"] = (
        top10["Volume"] / top10["Volume"].sum() * 100
    ).round(1)

    # Marque associée
    brand_map = (
        df_country[["MARKET_MODEL", "BRAND_UPDATE"]]
        .drop_duplicates("MARKET_MODEL")
        .set_index("MARKET_MODEL")["BRAND_UPDATE"]
    )

    top10["BRAND_UPDATE"] = top10["MARKET_MODEL"].map(brand_map)


    fig = px.bar(
        top10,
        x="MARKET_MODEL",
        y="Volume",
        text="Volume",
        title=f"Top 10 Vehicle Models - {country}",
        custom_data=["BRAND_UPDATE", "Share (%)"]
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=
        "<b>%{x}</b><br>" +
        "Brand: %{customdata[0]}<br>" +
        "Volume: %{y}<br>" +
        "Share: %{customdata[1]}%<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Vehicle Model",
        yaxis_title="Number of Vehicles",
        xaxis_tickangle=-45
    )

    fig.show()


display(
    widgets.interactive(
        plot_top_models,
        country=country_dropdown
    )
)