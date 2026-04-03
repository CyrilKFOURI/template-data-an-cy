import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_kpi_share(df):

    df=df.copy()
    df["TIME"]=df["YEAR"].astype(str)+"-"+df["Quarter"]

    countries=sorted(df["COUNTRY"].unique())

    fig=make_subplots(rows=1,cols=len(countries),subplot_titles=countries,specs=[[{"secondary_y":True}]*len(countries)])

    for i,country in enumerate(countries,1):

        df_c=df[df["COUNTRY"]==country]

        total=df_c.groupby("TIME")["VOLUME"].sum().reset_index()

        # top brands global sur le pays
        brand_rank=df_c.groupby("BRAND")["VOLUME"].sum().sort_values(ascending=False)
        brands=brand_rank.index.tolist()

        if len(brands)>5:
            brands=brands[:5]
            mode="group"
        else:
            mode="stack"

        for brand in brands:
            d=df_c[df_c["BRAND"]==brand]
            if d.empty:
                continue
            fig.add_trace(go.Bar(x=d["TIME"],y=d["SHARE"],name=brand,legendgroup=brand,showlegend=(i==1)),row=1,col=i,secondary_y=False)

        fig.add_trace(go.Scatter(x=total["TIME"],y=total["VOLUME"],mode='lines+markers',name="TOTAL",legendgroup="TOTAL",showlegend=(i==1)),row=1,col=i,secondary_y=True)

        fig.update_layout(barmode=mode)

    fig.update_layout(title="Market Share by Brand (Quarterly)",height=500)
    fig.update_yaxes(title_text="Share (%)",secondary_y=False)
    fig.update_yaxes(title_text="Volume",secondary_y=True)

    return fig
