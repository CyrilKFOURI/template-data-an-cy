import pandas as pd
import plotly.express as px

pd.set_option('display.max_columns', None)

pct_by_country = (
    pd.crosstab(nova['COUNTRY'], nova['CLS_VEHICLE_TYPE'], normalize='index') * 100
).round(1)

pct_by_country
