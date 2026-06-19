import seaborn as sns
import matplotlib.pyplot as plt

def get_grouped_data(df, y_col):
    # Prépare le dataframe agrégé pour la heatmap
    df_grouped = df.groupby([y_col, 'ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION', 'CLS_GROUP_RATING']).size().reset_index(name='count')
    return df_grouped.pivot_table(index=y_col, columns='ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION', values='count', aggfunc='sum', fill_value=0)

def plot_arval_heatmap(df, y_col):
    # Génère et affiche la heatmap
    data = get_grouped_data(df, y_col)
    plt.figure(figsize=(14, 10))
    sns.heatmap(data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f'Distribution: {y_col} par Segment')
    plt.show()
