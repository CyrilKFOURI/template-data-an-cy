import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_kpi_share(df,col):

    df=df.copy()
    df["TIME"]=df["YEAR"].astype(str)+"-"+df["Quarter"]

    countries=sorted(df["COUNTRY"].unique())

    fig=make_subplots(rows=1,cols=len(countries),subplot_titles=countries,specs=[[{"secondary_y":True}]*len(countries)])

    for i,country in enumerate(countries,1):

        df_c=df[df["COUNTRY"]==country]

        total=df_c.groupby("TIME")["VOLUME"].sum().reset_index()

        rank=df_c.groupby(col)["VOLUME"].sum().sort_values(ascending=False)
        cats=rank.index.tolist()

        if len(cats)>5:
            cats=cats[:5]
            mode="group"
        else:
            mode="stack"

        df_c=df_c[df_c[col].isin(cats)]

        for c in cats:
            d=df_c[df_c[col]==c]
            if d.empty:
                continue
            fig.add_trace(go.Bar(x=d["TIME"],y=d["SHARE"],name=c,legendgroup=c,showlegend=(i==1)),row=1,col=i,secondary_y=False)

        fig.add_trace(go.Scatter(x=total["TIME"],y=total["VOLUME"],mode='lines+markers',name="TOTAL",legendgroup="TOTAL",showlegend=(i==1)),row=1,col=i,secondary_y=True)

        fig.update_layout(barmode=mode)

    fig.update_layout(title=f"Market Share by {col} (Quarterly)",height=500)
    fig.update_yaxes(title_text="Share (%)",secondary_y=False)
    fig.update_yaxes(title_text="Volume",secondary_y=True)

    return fig
