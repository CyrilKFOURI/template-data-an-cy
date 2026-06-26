# Pour avoir Pays en ligne, puis Variable_A, et en colonnes Variable_B
df_crosstab = pd.crosstab(
    index=[df['Pays'], df['Variable_A']], 
    columns=df['Variable_B']
)
