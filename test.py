def export_kpi10_to_excel_fixed(pivot_df, filename="KPI10_Excel.xlsx"):
    """
    Export KPI10 final pivot DataFrame vers Excel avec :
    - Conversion Period -> str pour Excel
    - Gradient gris
    - Titre
    """
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows

    # Copier df pour ne pas modifier l'original
    df = pivot_df.copy()
    
    # Si l'index est un Period, le convertir en string
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.astype(str)
    
    # Créer workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Concentration"

    # Titre
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns)+1)
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = "Concentration par modèle et trimestre"
    title_cell.fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
    title_cell.alignment = Alignment(horizontal="center")

    # Ajouter index comme première colonne
    df_excel = df.reset_index()

    # Écrire DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Gradient gris sur valeurs numériques
    numeric_cols = df_excel.select_dtypes(include=[int, float]).columns
    max_val = df_excel[numeric_cols].values.max()

    for r in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=2, max_col=ws.max_column):
        for cell in r:
            if isinstance(cell.value, (int, float)):
                intensity = int(200 - (cell.value / max_val) * 150)  # 200 clair, 50 foncé
                hex_color = f"{intensity:02X}{intensity:02X}{intensity:02X}"
                cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
            cell.alignment = Alignment(horizontal="center")

    wb.save(filename)
    print(f"KPI10 exporté avec succès dans {filename}")
