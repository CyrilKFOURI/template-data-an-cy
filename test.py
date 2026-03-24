import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils.dataframe import dataframe_to_rows

def export_kpi10(pivot_df, filename="KPI10_Excel.xlsx"):
    """
    Exporte KPI10 pivot final vers Excel avec gradient gris.
    - Supprime la colonne VEHICLE_MODEL_3 si elle existe
    - Convertit l'index PeriodIndex en string
    """
    df = pivot_df.copy()

    # Supprimer la colonne VEHICLE_MODEL_3 si elle existe
    if 'VEHICLE_MODEL_3' in df.columns:
        df = df.drop(columns=['VEHICLE_MODEL_3'])

    # Convertir l'index PeriodIndex ou DatetimeIndex en str
    if isinstance(df.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        df.index = df.index.astype(str)

    # Reset index pour mettre les quarters en première colonne
    df_excel = df.reset_index()

    # Créer workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "KPI10_Concentration"

    # Titre
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df_excel.columns))
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = "Concentration par modèle et trimestre"
    title_cell.fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
    title_cell.font = Font(color="000000", bold=True)
    title_cell.alignment = Alignment(horizontal="center")

    # Écrire le DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)
            ws.cell(row=r_idx, column=c_idx).alignment = Alignment(horizontal="center")

    # Gradient gris pour toutes les valeurs numériques
    numeric_cols = df_excel.select_dtypes(include=[int, float]).columns
    if len(numeric_cols) > 0:
        max_val = df_excel[numeric_cols].values.max()
        min_val = df_excel[numeric_cols].values.min()
        # éviter division par zéro si toutes les valeurs identiques
        denom = max_val - min_val if max_val != min_val else 1
        for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=2, max_col=ws.max_column):
            for cell in r:
                if isinstance(cell.value, (int, float)):
                    # intensité entre 200 clair et 50 foncé
                    intensity = int(200 - (cell.value - min_val) / denom * 150)
                    hex_color = f"{intensity:02X}{intensity:02X}{intensity:02X}"
                    cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")

    # Sauvegarder
    wb.save(filename)
    print(f"KPI10 exporté avec succès dans {filename}")
