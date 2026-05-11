import plotly.express as px

unique_df = df[["BRAND", "MODEL"]].drop_duplicates()

fig = px.scatter(
    unique_df,
    x="BRAND",
    y="MODEL",
    title="Models by Brand"
)

fig.update_layout(
    xaxis_tickangle=-45,
    height=800
)