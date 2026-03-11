import plotly.graph_objects as go

def plot_kpi8(pivot_df):
    df = pivot_df.copy()
    df.rename(columns={df.columns[0]:"Period"}, inplace=True)
    fig = go.Figure()
    for model in df.columns[1:]:
        fig.add_bar(x=df["Period"].astype(str), y=df[model], name=model)
    fig.update_layout(title="KPI 8 - Vehicle Volume per Model per Quarter", xaxis_title="Period", yaxis_title="Volume", barmode="group")
    fig.show()
