import plotly.graph_objects as go
import pandas as pd

def plot_kpi8(pivot_df):
    pivot_melt = pivot_df.melt(id_vars=["Quarter"], var_name="Vehicle_Model", value_name="Volume")
    fig = go.Figure()
    for model in pivot_melt["Vehicle_Model"].unique():
        sub = pivot_melt[pivot_melt["Vehicle_Model"]==model]
        fig.add_bar(x=sub["Quarter"].astype(str), y=sub["Volume"], name=model)
    fig.update_layout(title="KPI 8 - Vehicle Volume per Quarter", xaxis_title="Quarter", yaxis_title="Volume", barmode="group")
    fig.show()

def plot_kpi13(pivot_df):
    country_col = pivot_df.columns[0]
    pivot_melt = pivot_df.melt(id_vars=[country_col], var_name="Quarter_Brand", value_name="Volume")
    fig = go.Figure()
    for qb in pivot_melt["Quarter_Brand"].unique():
        sub = pivot_melt[pivot_melt["Quarter_Brand"]==qb]
        fig.add_bar(x=sub[country_col], y=sub["Volume"], name=qb)
    fig.update_layout(title="KPI 13 - Top Brand Volume per Quarter per Country", xaxis_title="Country", yaxis_title="Volume", barmode="group")
    fig.show()
