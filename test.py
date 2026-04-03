
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_kpi_share(df,var="BRAND",asset_status=""):

    df=df.copy()
    df["TIME"]=df["YEAR"].astype(str)+"-"+df["Quarter"]

    countries=sorted(df["COUNTRY"].unique())
    brands=sorted(df[var].unique())

    fig=make_subplots(rows=1,cols=len(countries),subplot_titles=countries,specs=[[{"secondary_y":True}]*len(countries)])

    for i,country in enumerate(countries,1):

        df_c=df[df["COUNTRY"]==country]

        total=df_c.groupby("TIME")["VOLUME"].sum().reset_index()

        for brand in brands:
            d=df_c[df_c[var]==brand]
            if d.empty:
                continue
            fig.add_trace(go.Bar(x=d["TIME"],y=d["SHARE"],name=brand,legendgroup=brand,showlegend=(i==1)),row=1,col=i,secondary_y=False)

        fig.add_trace(go.Scatter(x=total["TIME"],y=total["VOLUME"],mode='lines+markers',name="TOTAL",legendgroup="TOTAL",showlegend=(i==1)),row=1,col=i,secondary_y=True)

    title=f"Share of {var} by Country (Quarterly)"
    if asset_status:
        title+=f" | Asset Status: {asset_status}"

    fig.update_layout(barmode='stack',title=title,height=500)
    fig.update_yaxes(title_text="Share (%)",secondary_y=False)
    fig.update_yaxes(title_text="Volume",secondary_y=True)

    return fig
