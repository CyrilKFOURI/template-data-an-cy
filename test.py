import pandas as pd
import xlsxwriter

def exporter_excel_joli(df, nom_fichier):
    """
    Exporte un DataFrame vers Excel avec un design simple et efficace.
    """
    # Vérification de sécurité : évite de planter si le DF est vide
    if df.empty:
        print("Le dataframe est vide, rien à exporter.")
        return

    # Création du fichier
    writer = pd.ExcelWriter(f"{nom_fichier}.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Données', index=True)
    
    workbook  = writer.book
    worksheet = writer.sheets['Données']
    
    # Formats
    header_format = workbook.add_format({
        'bold': True,
        'fg_color': '#2C3E50',
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    
    cell_format = workbook.add_format({'border': 1, 'align': 'center'})
    alt_format = workbook.add_format({'bg_color': '#F4F6F7', 'border': 1, 'align': 'center'})

    # Appliquer le format aux en-têtes
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num + 1, value, header_format)

    # Appliquer le format aux données avec gestion d'alternance
    for row_num in range(len(df)):
        fmt = alt_format if row_num % 2 == 0 else cell_format
        # Appliquer le format sur toute la ligne
        worksheet.set_row(row_num + 1, None, fmt)

    # Ajustement auto des colonnes
    for i, col in enumerate(df.columns):
        # Utilisation de .any() ou .all() si vous faites des calculs ici
        max_length = max(df[col].astype(str).map(len).max(), len(col))
        worksheet.set_column(i + 1, i + 1, max_length + 2)

    writer.close()
    print(f"Fichier '{nom_fichier}.xlsx' créé avec succès.")

# --- Exemple d'utilisation sécurisée ---
# Supposons que 'df' est votre crosstab
# Si vous aviez une condition avant, utilisez .any() ou .all()
# Par exemple : if (df > 0).any().any():
exporter_excel_joli(df, "resultat_analyse")
