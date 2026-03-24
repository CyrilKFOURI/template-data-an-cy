import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

def export_kpi10_fixed(pivot_df, filename="KPI10_Excel.xlsx"):
    """
    Exporte KPI10 final pivot DataFrame vers Excel :
    - Convertit PeriodIndex ou tout index en string
    - Gère les colonnes numériques
    - Applique gradient gris
    """
    # Copier pour ne pas toucher l'original
    df = pivot_df.copy()

    # Si l'index est un PeriodIndex ou autre, convertir en str
    if isinstance(df.index, pd.PeriodIndex) or isinstance(df.index, pd.Index):
        df.index = df.index.astype(str)

    # Reset index pour mettre l'index en première colonne
    df_excel = df.reset_index()

    # Créer Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "KPI10_Concentration"

    # Titre
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df_excel.columns))
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = "Concentration par modèle et trimestre"
    title_cell.fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
    title_cell.alignment = Alignment(horizontal="center")

    # Écrire DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Gradient gris sur les colonnes numériques
    numeric_cols = df_excel.select_dtypes(include=[int, float]).columns
    if len(numeric_cols) > 0:
        max_val = df_excel[numeric_cols].values.max()
        for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=2, max_col=ws.max_column):
            for cell in r:
                if isinstance(cell.value, (int, float)):
                    intensity = int(200 - (cell.value / max_val) * 150)  # 200 clair, 50 foncé
                    hex_color = f"{intensity:02X}{intensity:02X}{intensity:02X}"
                    cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
                cell.alignment = Alignment(horizontal="center")

    # Sauvegarder
    wb.save(filename)
    print(f"KPI10 exporté avec succès dans {filename}")













import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils.dataframe import dataframe_to_rows

def export_kpi8_colored(kpi8_df, filename="KPI8_Excel.xlsx"):
    """
    Exporte KPI8 pivot final vers Excel avec gradient vert-jaune-rouge
    """
    df = kpi8_df.copy()

    # Mapping mois
    month_mapping = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    df["MONTH_TEXT"] = df["MONTH"].map(month_mapping)
    df["Quarter"] = ((df["MONTH"]-1)//3 +1).apply(lambda x: f"Q{x}")

    # Réorganiser colonnes
    cols_order = ["Quarter", "MONTH_TEXT"] + [c for c in df.columns if c not in ["Quarter","MONTH","MONTH_TEXT","YEAR"]]
    df_excel = df[cols_order].reset_index(drop=True)

    # Créer workbook
    wb = Workbook()
    ws = wb.active
    ws.title = f"KPI8_{df['YEAR'].iloc[0]}"

    # Titre
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df_excel.columns))
    cell = ws.cell(row=1, column=1)
    cell.value = f"Production Lease Start Date {df['YEAR'].iloc[0]}"
    cell.fill = PatternFill(start_color="00A651", end_color="00A651", fill_type="solid")  # vert foncé
    cell.font = Font(color="FFFFFF", bold=True)
    cell.alignment = Alignment(horizontal="center")

    # Écrire DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)
            ws.cell(row=r_idx, column=c_idx).alignment = Alignment(horizontal="center")

    # Appliquer gradient vert→jaune→rouge sur les valeurs numériques
    numeric_cols = df_excel.select_dtypes(include=[int, float]).columns
    if len(numeric_cols) > 0:
        min_val = df_excel[numeric_cols].min().min()
        max_val = df_excel[numeric_cols].max().max()
        mid_val = (max_val + min_val)/2

        for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=3, max_col=ws.max_column):
            for cell in r:
                if isinstance(cell.value, (int, float)):
                    val = cell.value
                    if val >= mid_val:  # Vert → jaune
                        green = 200
                        red = int(255 - (val - mid_val)/(max_val - mid_val)*200)
                    else:  # Jaune → rouge
                        red = 255
                        green = int(100 + (val - min_val)/(mid_val - min_val)*155)
                    blue = 0
                    hex_color = f"{red:02X}{green:02X}{blue:02X}"
                    cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")

    wb.save(filename)
    print(f"KPI8 exporté avec succès dans {filename}")
