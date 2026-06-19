import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np

def get_grouped_data(df, y_col, x_cols, metric='volume'):
    if metric == 'concentration_financiere':
        agg_col = 'EXPOSURE_AMOUNT_TOT'
        agg_func = 'sum'
    elif metric == 'intensite_risk_asset':
        agg_col = 'VEHICLE_PRICE_EUR'
        agg_func = 'sum'
    else:
        agg_col = y_col
        agg_func = 'size'

    if agg_func == 'size':
        df_grouped = df.groupby([y_col] + x_cols).size().reset_index(name='count')
    else:
        df_grouped = df.groupby([y_col] + x_cols)[agg_col].sum().reset_index(name='count')
    
    df_grouped['x_combined'] = df_grouped[x_cols].astype(str).agg(' - '.join, axis=1)
    return df_grouped.pivot_table(index=y_col, columns='x_combined', values='count', aggfunc='sum', fill_value=0)

def plot_heatmap(df_pivot, title, metric='volume', page=0):
    rows_per_page = 30
    start_row = page * rows_per_page
    end_row = start_row + rows_per_page
    df_subset = df_pivot.iloc[start_row:end_row]
    
    height = max(8, len(df_subset) * 0.3)
    fmt_val = "d" if metric == 'volume' else ".0f"
    
    plt.figure(figsize=(16, height))
    sns.heatmap(df_subset, annot=True, fmt=fmt_val, cmap="YlGnBu")
    plt.title(f"{title} (Page {page + 1})")
    plt.show()

def interactive_heatmap(df):
    y_options = ['MARKET_MODEL', 'BRAND_UPDATE', 'POWER_CATEGORY', 'VA_CO2_EMSS_REAL']
    x_options = ['CLS_GROUP_RATING', 'ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION']
    metric_options = [('Volume', 'volume'), ('Concentration Financière', 'concentration_financiere'), ('Intensité Risk Asset', 'intensite_risk_asset')]
    
    def update(y_val, x_val, metric_val, page_val):
        data = get_grouped_data(df, y_val, list(x_val), metric=metric_val)
        plot_heatmap(data, f'{y_val} vs {list(x_val)} ({metric_val})', metric=metric_val, page=page_val)
    
    interact(update, 
             y_val=widgets.Dropdown(options=y_options, description='Axe Y:'), 
             x_val=widgets.SelectMultiple(options=x_options, value=[x_options[0]], description='Axe X:'),
             metric_val=widgets.Dropdown(options=metric_options, description='Mode:'),
             page_val=widgets.IntSlider(min=0, max=10, step=1, description='Page:'))

interactive_heatmap(nova)
