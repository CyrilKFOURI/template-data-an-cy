import plotly.graph_objects as go
import pandas as pd

def plot_kpi8(df):
    df_plot = df.copy()
    df_plot.set_index("Quarter", inplace=True)
    df_plot = df_plot.drop("Total", errors="ignore")

    fig = go.Figure()
    for col in df_plot.columns:
        fig.add_bar(
            x=df_plot.index.astype(str),
            y=df_plot[col],
            name=col
        )

    fig.update_layout(
        title="KPI 8 - Vehicle model volumes per quarter",
        xaxis_title="Quarter",
        yaxis_title="Volume",
        barmode="stack",
        height=500
    )
    fig.show()


def plot_kpi13(df_vol, df_brand):
    df_vol_plot = df_vol.copy()
    df_vol_plot.set_index("Country", inplace=True)

    quarters = [c for c in df_vol_plot.columns if c != "Country"]
    
    fig = go.Figure()
    
    for q in quarters:
        fig.add_bar(
            x=[q],
            y=[df_vol_plot.loc[df_vol_plot.index[0], q]],
            name=f"Vol - {q}"
        )
        fig.add_scatter(
            x=[q],
            y=[df_vol_plot.loc[df_vol_plot.index[0], q]],
            mode="text",
            text=[df_brand.loc[df_brand.index[0], q]],
            textposition="top center",
            showlegend=False
        )
    
    fig.update_layout(
        title="KPI 13 - Highest brand volume per quarter",
        xaxis_title="Quarter",
        yaxis_title="Volume",
        height=500
    )
    
    fig.show()



import plotly.express as px

def plot_kpi_over_time(df, kpi_col):
    df_plot = df.copy()
    df_plot["Month-Year"] = df_plot["MONTH"].astype(str).str.zfill(2) + "-" + df_plot["YEAR"].astype(str)
    fig = px.line(df_plot, x="Month-Year", y=kpi_col, markers=True, text=kpi_col)
    fig.update_traces(textposition="top center")
    fig.update_layout(title=f"{kpi_col} evolution over time", xaxis_title="Month-Year", yaxis_title=kpi_col)
    fig.show()



df_kpi1 = fleet.kpi1_lease_under_25(asset_status="IN_FLEET", year=2025)
plot_kpi_over_time(df_kpi1, "<25_months_%")
