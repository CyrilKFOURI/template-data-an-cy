import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display


# Liste des pays
countries = sorted(nova["COUNTRY"].dropna().unique())

country_dropdown = widgets.Dropdown(
    options=countries,
    description="Country:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width="300px")
)


def plot_top_models(country):
    
    # Filtrer le pays choisi
    df_country = nova[nova["COUNTRY"] == country]

    # Top 10 modèles
    top10 = (
        df_country["MARKET_MODEL"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top10.columns = ["MARKET_MODEL", "Volume"]

    # Ajouter le pourcentage
    top10["Share (%)"] = (
        top10["Volume"] / top10["Volume"].sum() * 100
    ).round(1)

    # Ajouter la marque
    top10["BRAND_UPDATE"] = top10["MARKET_MODEL"].map(
        df_country.drop_duplicates("MARKET_MODEL")
        .set_index("MARKET_MODEL")["BRAND_UPDATE"]
    )

    # Affichage
    fig = px.pie(
        top10,
        values="Volume",
        names="MARKET_MODEL",
        title=f"Top 10 Vehicle Models - {country}",
        hole=0.45,
        hover_data=["BRAND_UPDATE", "Volume", "Share (%)"]
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate=
        "<b>%{label}</b><br>" +
        "Brand: %{customdata[0]}<br>" +
        "Volume: %{customdata[1]}<br>" +
        "Share: %{customdata[2]}%"
    )

    fig.show()


# Interaction avec le filtre
widgets.interactive(
    plot_top_models,
    country=country_dropdown
)