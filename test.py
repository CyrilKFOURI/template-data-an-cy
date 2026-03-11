import plotly.express as px
import pandas as pd

def plot_kpi8(df):
    df_plot = df.copy()
    if "Total" in df_plot.index: df_plot = df_plot.drop("Total")
    df_plot = df_plot.melt(id_vars=["Quarter"], var_name="Vehicle Model", value_name="Volume")
    fig = px.bar(df_plot, x="Quarter", y="Volume", color="Vehicle Model", text="Volume")
    fig.update_traces(textposition="outside")
    fig.update_layout(title="KPI 8: Volume per Vehicle Model per Quarter", barmode="group")
    fig.show()

def plot_kpi13(df):
    df_plot = df.copy()
    volume_cols = [c for c in df_plot.columns if "_vol" in c]
    df_plot = df_plot.reset_index().melt(id_vars=["COUNTRY"], value_vars=volume_cols, var_name="Quarter", value_name="Volume")
    fig = px.bar(df_plot, x="Quarter", y="Volume", color="COUNTRY", text="Volume")
    fig.update_traces(textposition="outside")
    fig.update_layout(title="KPI 13: Top Brand Volume per Quarter per Country", barmode="group")
    fig.show()
