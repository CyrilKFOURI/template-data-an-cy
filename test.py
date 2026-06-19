import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

def get_grouped_data(df, y_col, x_cols):
    df_grouped = df.groupby([y_col] + x_cols).size().reset_index(name='count')
    df_grouped['x_combined'] = df_grouped[x_cols].astype(str).agg(' - '.join, axis=1)
    return df_grouped.pivot_table(index=y_col, columns='x_combined', values='count', aggfunc='sum', fill_value=0)

def plot_heatmap(df_pivot, title):
    plt.figure(figsize=(16, 10))
    sns.heatmap(df_pivot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(title)
    plt.show()

# La 3e fonction mise à jour
def interactive_heatmap(df):
    y_options = ['MARKET_MODEL', 'BRAND_UPDATE', 'POWER_CATEGORY', 'VA_CO2_EMSS_REAL']
    x_options = ['CLS_GROUP_RATING', 'ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION']
    
    def update(y_val, x_val):
        # On s'assure que x_val est bien une liste
        data = get_grouped_data(df, y_val, list(x_val))
        plot_heatmap(data, f'{y_val} vs {list(x_val)}')
    
    interact(update, 
             y_val=widgets.Dropdown(options=y_options, description='Axe Y:'), 
             x_val=widgets.SelectMultiple(options=x_options, value=[x_options[0]], description='Axe X:'))

# Lancement
interactive_heatmap(nova)
