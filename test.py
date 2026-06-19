import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive
import numpy as np

def format_number(val, pos):
    if val >= 1_000_000:
        return f'{val/1_000_000:.1f}M'
    elif val >= 1_000:
        return f'{val/1_000:.1f}k'
    return f'{int(val)}'

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
    total_rows = len(df_pivot)
    
    if total_rows > rows_per_page:
        start_row = page * rows_per_page
        end_row = min(start_row + rows_per_page, total_rows)
        df_subset = df_pivot.iloc[start_row:end_row]
    else:
        df_subset = df_pivot

    height = max(8, len(df_subset) * 0.4)
    plt.figure(figsize=(20, height))
    
    annot_data = df_subset.applymap(lambda x: format_number(x, 0))
    
    sns.heatmap(df_subset, annot=annot_data, fmt="", cmap="YlGnBu", annot_kws={"size": 10})
    plt.title(f"{title} - Total: {total_rows} lignes")
    plt.show()

def interactive_heatmap(df):
    y_options = ['MARKET_MODEL', 'BRAND_UPDATE', 'POWER_CATEGORY', 'VA_CO2_EMSS_REAL']
    x_options = ['CLS_GROUP_RATING', 'ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION']
    metric_options = [('Volume', 'volume'), ('Concentration Financière', 'concentration_financiere'), ('Intensité Risk Asset', 'intensite_risk_asset')]
    
    def ui_handler(y_val, x_val, metric_val, page_val):
        data = get_grouped_data(df, y_val, list(x_val), metric=metric_val)
        
        if len(data) > 30:
            plot_heatmap(data, f'{y_val} vs {list(x_val)} ({metric_val})', metric=metric_val, page=page_val)
        else:
            plot_heatmap(data, f'{y_val} vs {list(x_val)} ({metric_val})', metric=metric_val, page=0)

    y_w = widgets.Dropdown(options=y_options, description='Axe Y:')
    x_w = widgets.SelectMultiple(options=x_options, value=[x_options[0]], description='Axe X:')
    m_w = widgets.Dropdown(options=metric_options, description='Mode:')
    p_w = widgets.IntSlider(min=0, max=10, step=1, description='Page:', continuous_update=False)

    def show_or_hide_page(change):
        data = get_grouped_data(df, y_w.value, list(x_w.value), metric=m_w.value)
        p_w.layout.display = 'flex' if len(data) > 30 else 'none'
        
    y_w.observe(show_or_hide_page, 'value')
    x_w.observe(show_or_hide_page, 'value')
    m_w.observe(show_or_hide_page, 'value')
    
    interact(ui_handler, y_val=y_w, x_val=x_w, metric_val=m_w, page_val=p_w)

interactive_heatmap(nova)
