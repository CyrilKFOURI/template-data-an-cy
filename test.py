import pandas as pd
import xlsxwriter

def exporter_excel_joli(df, nom_fichier):
    # Sécurité absolue : on ne travaille que si le DF n'est pas vide
    if df.empty:
        print("Le dataframe est vide.")
        return

    # Utilisation explicite du moteur xlsxwriter
    writer = pd.ExcelWriter(f"{nom_fichier}.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Données', index=True)
    
    workbook  = writer.book
    worksheet = writer.sheets['Données']
    
    # Définition des styles
    header_fmt = workbook.add_format({'bold': True, 'fg_color': '#2C3E50', 'font_color': 'white', 'border': 1, 'align': 'center'})
    cell_fmt = workbook.add_format({'border': 1, 'align': 'center'})
    alt_fmt = workbook.add_format({'bg_color': '#F4F6F7', 'border': 1, 'align': 'center'})

    # Appliquer styles en-têtes
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num + 1, value, header_fmt)

    # Appliquer styles cellules
    for row_num in range(len(df)):
        fmt = alt_fmt if row_num % 2 == 0 else cell_fmt
        for col_num in range(len(df.columns)):
            val = df.iloc[row_num, col_num]
            worksheet.write(row_num + 1, col_num + 1, val, fmt)

    # Ajustement largeur
    for i, col in enumerate(df.columns):
        worksheet.set_column(i + 1, i + 1, 15)

    writer.close()
    print("Export réussi.")

# --- APPEL DE LA FONCTION ---
# Assurez-vous que 'df' est bien défini ici avant l'appel
# Ne mettez AUCUN 'if' autour de 'df' avant cette ligne
exporter_excel_joli(df, "mon_rapport")
