import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils.dataframe import dataframe_to_rows

def export_kpi10_to_excel(pivot_df, filename="KPI10.xlsx"):
    """
    Exporte le pivot KPI10 en Excel avec styles simples
    pivot_df : DataFrame issu de KPI10 (index Quarter, colonnes = VEHICLE_MODEL_3)
    """
    # 1️⃣ Reset index pour que Quarter devienne une colonne normale
    df = pivot_df.reset_index()

    # 2️⃣ Convertir Quarter en str si c'est un Period
    df[df.columns[0]] = df[df.columns[0]].astype(str)

    # 3️⃣ Convertir les autres colonnes en str si elles sont Period (sécurité)
    for col in df.columns[1:]:
        if isinstance(df[col].dtype, pd.PeriodDtype):
            df[col] = df[col].astype(str)

    # 4️⃣ Tri des colonnes par volume total (sauf la première colonne qui est Quarter)
    col_order = [df.columns[0]] + list(df[df.columns[1:]].sum().sort_values(ascending=False).index)
    df = df[col_order]

    # 5️⃣ Créer le workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "KPI10"

    # 6️⃣ Titre du tableau
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns))
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = "Concentration per Vehicle Model (KPI10)"
    title_cell.font = Font(color="FFFFFF", bold=True)
    title_cell.fill = PatternFill(start_color="333333", end_color="333333", fill_type="solid")
    title_cell.alignment = Alignment(horizontal="center")

    # 7️⃣ Écrire les données
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.alignment = Alignment(horizontal="center")
            # 8️⃣ Style : intensité de gris selon valeur (pour les colonnes numériques)
            if r_idx > 2 and c_idx > 1 and isinstance(value, (int, float)):
                # gris clair pour faible volume, gris foncé pour grand volume
                max_val = df.iloc[:, 1:].max().max()
                intensity = int(255 - (value / max_val * 150))  # 105->255
                hex_color = f"{intensity:02X}{intensity:02X}{intensity:02X}"
                cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")

    # 9️⃣ Sauvegarder
    wb.save(filename)
    print(f"KPI10 exporté avec succès dans {filename}")










import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

def export_kpi8_to_excel(kpi8_pivot, filename="KPI8_Excel.xlsx"):
    """
    Exporte KPI8 en Excel avec style :
    - Mapping mois int -> texte
    - Titre avec background jaune et texte rouge
    - Quarters à gauche, mois à droite
    - Gradient vert/orange/rouge selon valeur
    """
    # --- 1️⃣ Copier le df pour ne rien modifier à l'extérieur ---
    df = kpi8_pivot.copy()
    
    # --- 2️⃣ Mapping mois int -> texte ---
    month_mapping = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    df["MONTH_TEXT"] = df["MONTH"].map(month_mapping)
    
    # --- 3️⃣ Créer colonne Quarter ---
    df["Quarter"] = ((df["MONTH"] - 1) // 3 + 1).apply(lambda x: f"Q{x}")
    
    # --- 4️⃣ Réorganiser colonnes pour export ---
    cols_order = ["Quarter", "MONTH_TEXT"] + [c for c in df.columns if c not in ["Quarter","MONTH","MONTH_TEXT","YEAR"]]
    df_excel = df[cols_order]
    
    # --- 5️⃣ Créer le workbook ---
    wb = Workbook()
    ws = wb.active
    ws.title = f"KPI8_{df['YEAR'].iloc[0]}"
    
    # --- 6️⃣ Écrire le titre ---
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df_excel.columns))
    cell = ws.cell(row=1, column=1)
    cell.value = f"Production Lease Start Date {df['YEAR'].iloc[0]}"
    cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    cell.font = Font(color="FF0000", bold=True)
    cell.alignment = Alignment(horizontal="center")
    
    # --- 7️⃣ Écrire le DataFrame ligne par ligne ---
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # --- 8️⃣ Appliquer style selon colonne ---
    for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=3, max_col=ws.max_column):
        for cell in r:
            val = cell.value
            if isinstance(val, (int, float)):
                # Gradient vert/orange/rouge selon valeur
                if val > 80:
                    cell.fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")
                elif val > 50:
                    cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
    
    # --- 9️⃣ Sauvegarder ---
    wb.save(filename)
    print(f"KPI8 exporté avec succès dans {filename}")
