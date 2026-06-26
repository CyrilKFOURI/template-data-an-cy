a = pd.crosstab(
    index=nova['COUNTRY'], 
    columns=nova['CDN_CLF_SEGMENT'], 
    values=nova['BIKE_OR_CAR'],  # La colonne que vous voulez agréger
    aggfunc='count'              # 'count', 'sum', 'mean', 'max', etc.
)
