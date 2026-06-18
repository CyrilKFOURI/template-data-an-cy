import pandas as pd
import plotly.express as px

tmp = pd.json_normalize(df["data"].explode())

fig = px.line(
    tmp.sort_values("QTR"),
    x="QTR",
    y="WLTP_Q_MEAN_VEH_NEW",
    markers=True,
    title="Evolution de WLTP_Q_MEAN_VEH_NEW"
)

fig.show()
