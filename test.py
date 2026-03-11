
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def plot_categorical(cols, data):
    rows = math.ceil(len(cols)/2)
    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols)

    for i, col in enumerate(cols):
        counts = data[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        counts["percent"] = (counts["count"]/counts["count"].sum()*100).round(1)
        counts["label"] = counts["percent"].astype(str) + "%"
        n_categories = counts.shape[0]
        counts = counts.head(25)

        r = i//2 + 1
        c = i%2 + 1

        fig.add_bar(x=counts[col], y=counts["count"], text=counts["label"], name=col, row=r, col=c)

    fig.update_layout(height=350*rows, width=900, showlegend=False, title="Categorical distributions")
    fig.show()


def kpi9_power_category_per_segment_quarter(self, year):

    df = self.df.copy()

    df = df[df["YEAR"] == year]

    df = df.dropna(subset=["INITIAL_POWER", "SEGMENT"])

    df["Quarter"] = pd.to_datetime(df["MONTH"], format="%m").apply(lambda m: f"Q{((m.month-1)//3)+1}")

    pivot = df.pivot_table(
        index="Quarter",
        columns=["SEGMENT", "INITIAL_POWER"],
        values="VEHICLE_ID",
        aggfunc="count",
        fill_value=0
    )

    pivot.reset_index(inplace=True)

    return pivot


def kpi9_1_segment_share_quarter(self, segment, year):

    df = self.df.copy()

    df = df[df["YEAR"] == year]

    df = df.dropna(subset=["SEGMENT"])

    df["Quarter"] = pd.to_datetime(df["MONTH"], format="%m").dt.to_period("Q")

    total_per_quarter = df.groupby("Quarter")["VEHICLE_ID"].count()

    segment_per_quarter = df[df["SEGMENT"] == segment].groupby("Quarter")["VEHICLE_ID"].count()

    result = pd.concat([total_per_quarter, segment_per_quarter], axis=1).fillna(0)

    result.columns = ["TOTAL_VOLUME", "SEGMENT_VOLUME"]

    result["SEGMENT_SHARE_%"] = (result["SEGMENT_VOLUME"] / result["TOTAL_VOLUME"] * 100).round(2)

    result.reset_index(inplace=True)

    return result



import plotly.express as px

def plot_kpi9_1(df):
    fig = px.line(df, x="Quarter", y="SEGMENT_SHARE_%", markers=True)
    fig.update_layout(title="KPI 9_1 - Segment share per quarter", xaxis_title="Quarter", yaxis_title="Segment Share (%)")
    fig.show()


import plotly.graph_objects as go

def plot_kpi9(df):
    fig = go.Figure()
    for col in df.columns[1:]:
        fig.add_bar(x=df.iloc[:,0].astype(str), y=df[col], name=str(col))
    fig.update_layout(title="KPI 9 - Power category volume per segment per quarter", xaxis_title="Quarter", yaxis_title="Volume", barmode="stack")
    fig.show()
