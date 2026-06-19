import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Tes deux fonctions de base
def get_grouped_data(df, y_col, x_cols):
    df_grouped = df.groupby([y_col] + x_cols).size().reset_index(name='count')
    df_grouped['x_combined'] = df_grouped[x_cols].astype(str).agg(' - '.join, axis=1)
    return df_grouped.pivot_table(index=y_col, columns='x_combined', values='count', aggfunc='sum', fill_value=0)

def plot_heatmap(df_pivot, title):
    plt.figure(figsize=(16, 10))
    sns.heatmap(df_pivot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(title)
    plt.show()

# La 3e fonction avec widget
def interactive_heatmap(df):
    # Liste des colonnes dispos
    y_options = ['MARKET_MODEL', 'BRAND_UPDATE', 'POWER_CATEGORY', 'VA_CO2_EMSS_REAL']
    x_options = ['CLS_GROUP_RATING', 'ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION']
    
    y_dropdown = widgets.Dropdown(options=y_options, description='Axe Y:')
    x_select = widgets.SelectMultiple(options=x_options, value=[x_options[0]], description='Axe X:')
    
    def on_change(change):
        # Efface la sortie précédente
        plt.close('all')
        # Calcule et affiche
        data = get_grouped_data(df, y_dropdown.value, list(x_select.value))
        plot_heatmap(data, f'{y_dropdown.value} vs {list(x_select.value)}')
        
    # Lier le changement des widgets à l'affichage
    widgets.interactive_output(on_change, {'y_dropdown': y_dropdown, 'x_select': x_select})
    display(y_dropdown, x_select)

# Pour lancer l'outil :
interactive_heatmap(nova)
