import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

def export_kpi8_to_excel(pivot_df, year, filename="KPI8_Excel.xlsx"):
    """
    Export KPI8 final pivot DataFrame vers Excel avec style :
    - Titre jaune/rouge
    - Quarter et mois texte
    - Gradient vert/orange/rouge sur valeurs numériques
    """
    df = pivot_df.copy()  # ne jamais modifier le df original
    
    # Assure que MONTH est int pour le calcul de Quarter
    df["MONTH"] = df["MONTH"].astype(int)
    
    # Calcul Quarter
    df["Quarter"] = ((df["MONTH"]-1)//3 + 1).apply(lambda x: f"Q{x}")
    
    # Mapping mois int -> texte
    month_mapping = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    df["MONTH_TEXT"] = df["MONTH"].map(month_mapping)
    
    # Réorganiser colonnes : Quarter | Mois texte | reste
    cols_order = ["Quarter", "MONTH_TEXT"] + [c for c in df.columns if c not in ["Quarter","MONTH","MONTH_TEXT","YEAR"]]
    df_excel = df[cols_order]
    
    # Créer workbook
    wb = Workbook()
    ws = wb.active
    ws.title = f"KPI8_{year}"
    
    # Écrire titre
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df_excel.columns))
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = f"Production Lease Start Date {year}"
    title_cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # jaune
    title_cell.font = Font(color="FF0000", bold=True)  # rouge
    title_cell.alignment = Alignment(horizontal="center")
    
    # Écrire DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Gradient sur valeurs numériques (vert/orange/rouge)
    for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=3, max_col=ws.max_column):
        for cell in r:
            if isinstance(cell.value, (int,float)):
                if cell.value > 80:
                    cell.fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")
                elif cell.value > 50:
                    cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
    
    wb.save(filename)
    print(f"KPI8 exporté avec succès dans {filename}")




def export_kpi10_to_excel(pivot_df, filename="KPI10_Excel.xlsx"):
    """
    Export KPI10 final pivot DataFrame vers Excel avec style :
    - Titre gris
    - Gradient gris sur valeurs numériques
    - Quarter converti en str si besoin
    """
    df = pivot_df.copy()  # ne jamais modifier le df original
    
    # Convertir index Quarter en str si c'est un Period
    df.index = df.index.astype(str)
    
    # Créer workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Concentration"
    
    # Écrire titre
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns)+1)
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = "Concentration par modèle et trimestre"
    title_cell.fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")  # gris
    title_cell.alignment = Alignment(horizontal="center")
    
    # Ajouter index comme première colonne
    df_excel = df.reset_index()
    
    # Écrire DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Gradient gris sur les valeurs numériques
    # On calcule le max pour normaliser
    max_val = df_excel.iloc[:, 1:].select_dtypes(include=[int,float]).values.max()
    
    for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=2, max_col=ws.max_column):
        for cell in r:
            if isinstance(cell.value, (int,float)):
                intensity = int(200 - (cell.value/max_val)*150)  # 200=clair, 50=foncé
                hex_color = f"{intensity:02X}{intensity:02X}{intensity:02X}"
                cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
    
    wb.save(filename)
    print(f"KPI10 exporté avec succès dans {filename}")
