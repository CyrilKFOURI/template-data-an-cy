import pandas as pd

def exporter_excel_joli(df, nom_fichier):
    """
    Exporte un DataFrame vers Excel avec un design simple et efficace.
    """
    # Création du writer avec le moteur xlsxwriter
    writer = pd.ExcelWriter(f"{nom_fichier}.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Données', index=True)
    
    workbook  = writer.book
    worksheet = writer.sheets['Données']
    
    # Définition des formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#2C3E50', # Bleu marine profond
        'font_color': 'white',
        'border': 1
    })
    
    cell_format = workbook.add_format({'border': 1})
    alt_format = workbook.add_format({'bg_color': '#F4F6F7', 'border': 1}) # Gris très clair

    # Appliquer le format aux en-têtes
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num + 1, value, header_format)

    # Appliquer le format aux lignes (avec alternance de couleurs pour le "joli" design)
    for row_num in range(len(df)):
        fmt = alt_format if row_num % 2 == 0 else cell_format
        worksheet.set_row(row_num + 1, None, fmt)

    # Ajustement auto des colonnes
    for i, col in enumerate(df.columns):
        column_len = max(df[col].astype(str).map(len).max(), len(col))
        worksheet.set_column(i + 1, i + 1, column_len + 2)

    writer.close()
    print(f"Fichier {nom_fichier}.xlsx créé avec succès.")

# Exemple d'utilisation :
# exporter_excel_joli(df, "mon_rapport_finance")
