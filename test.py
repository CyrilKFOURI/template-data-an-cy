import pandas as pd
import plotly.express as px

tmp = pd.DataFrame(wltp["data_json"].iloc[0])

fig = px.line(
    tmp,
    x="QTR",
    y="WLTP_Q_MEAN_VEH_NEW",
    markers=True
)

fig.show()