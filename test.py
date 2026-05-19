def build_kpi6_table(df, metric='SHARE'):
    
    # copie
    temp = df.copy()

    # créer le nom des futures colonnes
    temp['COL_NAME'] = (
        temp['YEAR'].astype(str)
        + '_'
        + temp['PERIOD'].astype(str)
        + '_'
        + metric.upper()
    )

    # pivot pour les valeurs SHARE ou VOLUME
    pivot_metric = temp.pivot_table(
        index='COUNTRY',
        columns='COL_NAME',
        values=metric.upper(),
        aggfunc='first'
    ).reset_index()

    # pivot pour la variable (marque/oem/etc)
    pivot_var = temp.pivot_table(
        index='COUNTRY',
        columns='COL_NAME',
        values='BRAND_UPDATE',
        aggfunc='first'
    ).reset_index()

    # rename colonnes variable
    pivot_var.columns = [
        col.replace(metric.upper(), 'VAR')
        if col != 'COUNTRY' else col
        for col in pivot_var.columns
    ]

    # merge des deux
    final = pivot_metric.merge(
        pivot_var,
        on='COUNTRY',
        how='left'
    )

    return final

kpi11 = get_kpi5(
    view5,
    'portfolio',
    'IN FLEET',
    'ES',
    SOURCE_FILTER[0],
    'quarterly',
    top_n='1'
)

tableau = build_kpi6_table(kpi11, metric='SHARE')

tableau





def build_kpi6_table(df, variable, metric='SHARE'):
    
    temp = df.copy()

    temp['COL_NAME'] = (
        temp['YEAR'].astype(str)
        + '_'
        + temp['PERIOD'].astype(str)
        + '_'
        + metric.upper()
    )

    pivot_metric = temp.pivot_table(
        index='COUNTRY',
        columns='COL_NAME',
        values=metric.upper(),
        aggfunc='first'
    ).reset_index()

    pivot_var = temp.pivot_table(
        index='COUNTRY',
        columns='COL_NAME',
        values=variable,
        aggfunc='first'
    ).reset_index()

    pivot_var.columns = [
        col.replace(metric.upper(), 'VAR')
        if col != 'COUNTRY' else col
        for col in pivot_var.columns
    ]

    final = pivot_metric.merge(
        pivot_var,
        on='COUNTRY',
        how='left'
    )

    return final


tableau = build_kpi6_table(
    kpi11,
    variable=SOURCE_FILTER[0],
    metric='SHARE'
)