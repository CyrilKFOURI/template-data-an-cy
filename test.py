
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

# Exemple de df_kpis_summary
df_kpis_summary = pd.DataFrame({
    "Asset Risk, Financed Fleet": [
        'LTR < 25 (production)',
        '25m <= LTR <= 30m (production)',
        'DI vs Non-DI',
        'Hybrid share',
        'EV share',
        'PC vs LCV'
    ],
    "Period": ['2025', '2025', 'Dec-2025', 'Dec-2025', 'Dec-2025', 'Dec-2025'],
    "Result": [12.5, 27.3, "15 & 85", 30.0, 25.0, "40 & 60"],
    "Unit": ['x', '%', '%', '%', '%', '%'],
    "Comment": [
        "Limit is 5%",
        "Limit is 10%",
        "Total current fleet",
        "Hyb is within Non-DI share. Total current fleet",
        "EV is within Non-DI share. Total current fleet",
        "Total current fleet"
    ]
})

# Création du workbook
wb = Workbook()
ws = wb.active
ws.title = "KPI Summary"

# Styles
header_fill_gray = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
header_fill_green = PatternFill(start_color="006400", end_color="006400", fill_type="solid")  # vert foncé
font_white = Font(color="FFFFFF", bold=True)
align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)

# Écrire l'entête
for col_num, col_name in enumerate(df_kpis_summary.columns, 1):
    cell = ws.cell(row=1, column=col_num, value=col_name)
    cell.alignment = align_center
    if col_num == 1:
        cell.fill = header_fill_green
        cell.font = font_white
    else:
        cell.fill = header_fill_gray

# Écrire les données
for r_idx, row in enumerate(df_kpis_summary.itertuples(index=False), 2):
    for c_idx, value in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=value)
        cell.alignment = align_center

# Ajuster la largeur des colonnes (optionnel)
for column_cells in ws.columns:
    length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
    ws.column_dimensions[column_cells[0].column_letter].width = length + 5

# Sauvegarde
wb.save("KPI_Summary.xlsx")





























import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

def export_kpi7_to_excel(share_df, year, filename="KPI7_Fuel_Share.xlsx"):
    """
    Export share_of_fuel_type_per_quarter en Excel avec design
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Fuel Share"

    # Styles
    title_fill = PatternFill(start_color="006400", end_color="006400", fill_type="solid")  # vert foncé
    title_font = Font(color="FFFFFF", bold=True)
    align_center = Alignment(horizontal="center", vertical="center")

    yellow_fill = PatternFill(start_color="FFFACD", end_color="FFFACD", fill_type="solid")  # jaune clair
    green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # vert clair

    # Ligne titre (fusionnée)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=share_df.shape[1]+1)
    cell = ws.cell(row=1, column=1, value="Share of Fuel Type")
    cell.fill = title_fill
    cell.font = title_font
    cell.alignment = align_center

    # Ligne Quarters (Q1 2025, Q2 2025...)
    ws.cell(row=2, column=1, value="")  # première colonne vide
    for i, col in enumerate(share_df.columns, start=2):
        ws.cell(row=2, column=i, value=f"{col} {year}").alignment = align_center

    # Écriture des données
    for r_idx, (pc, row) in enumerate(share_df.iterrows(), start=3):
        # 1ère colonne = power category
        ws.cell(row=r_idx, column=1, value=pc).alignment = align_center

        # Colonnes des valeurs
        for c_idx, value in enumerate(row, start=2):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.alignment = align_center

            # couleur du background selon le quarter
            if c_idx <= 4:  # 3 premiers quarters (Q1-Q3)
                cell.fill = yellow_fill
            elif c_idx == 5:  # 4e quarter (Q4)
                cell.fill = green_fill

    # Ajuster largeur des colonnes
    for column_cells in ws.columns:
        length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length + 5

    # Sauvegarde
    wb.save(filename)
    print(f"Exported {filename} successfully.")




























import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import calendar

def export_kpi8_to_excel(kpi_df, year, filename="KPI8_Volume.xlsx"):
    """
    Export KPI8 en Excel avec:
    - Titre fusionné 'Production Lease Start Date' (jaune, Production noir, Lease Start Date rouge)
    - Première colonne = Quarters / Mois
    - Valeurs colorées selon intensité
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Volume by Power"

    # Styles
    title_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # jaune
    font_black = Font(color="000000", bold=True)
    font_red = Font(color="FF0000", bold=True)
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")

    # Ligne titre (fusionnée)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=kpi_df.shape[1]+1)
    cell = ws.cell(row=1, column=1, value="Production Lease Start Date")
    cell.fill = title_fill
    # mettre "Production" noir et "Lease Start Date" rouge → openpyxl ne supporte pas texte partiel, on met tout noir ou rouge pour l'instant
    cell.font = font_black
    cell.alignment = align_center

    # Mapping mois
    month_mapping = {i: calendar.month_name[i] for i in range(1,13)}

    # Préparer la première colonne : Quarters et Mois
    quarter_map = {1:"Q1", 2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}

    row_idx = 2
    for q in range(1,5):
        months_in_q = [m for m in range(1,13) if quarter_map[m]==f"Q{q}"]
        # Fusion verticale pour Quarter
        ws.merge_cells(start_row=row_idx, start_column=1, end_row=row_idx+len(months_in_q)-1, end_column=1)
        ws.cell(row=row_idx, column=1, value=f"Q{q}").alignment = align_center

        # Écrire mois et valeurs
        for m in months_in_q:
            ws.cell(row=row_idx, column=2, value=month_mapping[m]).alignment = align_left

            # Écrire les valeurs des Power Categories
            if m in kpi_df.index.get_level_values("MONTH"):
                values = kpi_df.loc[(year,m)].values
                for c_idx, val in enumerate(values, start=3):
                    cell = ws.cell(row=row_idx, column=c_idx, value=val)
                    # Couleur selon valeur (exemple simple : faible vert, moyen orange, fort rouge)
                    if val < 10:
                        cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # vert
                    elif val < 20:
                        cell.fill = PatternFill(start_color="FFD580", end_color="FFD580", fill_type="solid")  # orange
                    else:
                        cell.fill = PatternFill(start_color="FF7F7F", end_color="FF7F7F", fill_type="solid")  # rouge
                    cell.alignment = align_center
            row_idx +=1

    # Écrire les titres des Power Categories en 1ère ligne de valeurs
    for c_idx, pc in enumerate(kpi_df.columns, start=3):
        ws.cell(row=2, column=c_idx, value=pc).alignment = align_center

    # Ajuster largeur des colonnes
    for col in range(1, kpi_df.shape[1]+3):
        ws.column_dimensions[get_column_letter(col)].width = 15

    wb.save(filename)
    print(f"Exported {filename} successfully.")



















import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

def export_kpi10_to_excel(kpi_df, title="Concentration by Model", filename="KPI10_Concentration.xlsx"):
    """
    Export KPI10 en Excel avec:
    - Titre du tableau
    - Fond gris pour toutes les cellules
    - Intensité du gris pour les valeurs
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Concentration"

    # Styles
    align_center = Alignment(horizontal="center", vertical="center")
    title_fill = PatternFill(start_color="A9A9A9", end_color="A9A9A9", fill_type="solid")  # gris foncé
    title_font = Font(color="FFFFFF", bold=True)
    gray_fill_base = 200  # valeur de base pour fond gris clair (RGB)

    # Ligne titre (fusionnée sur toutes les colonnes)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=kpi_df.shape[1]+1)
    cell = ws.cell(row=1, column=1, value=title)
    cell.fill = title_fill
    cell.font = title_font
    cell.alignment = align_center

    # Écrire l’en-tête
    for col_idx, col_name in enumerate(kpi_df.columns, start=1):
        cell = ws.cell(row=2, column=col_idx, value=col_name)
        cell.fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")  # gris clair
        cell.alignment = align_center
        cell.font = Font(bold=True)

    # Écrire les données
    for r_idx, row in enumerate(kpi_df.itertuples(index=False), start=3):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.alignment = align_center

            # Intensité de gris selon la valeur (0 → clair, max → foncé)
            if isinstance(value, (int, float)):
                max_val = kpi_df.iloc[:, c_idx-1].max()
                if max_val == 0:
                    intensity = 230  # gris très clair
                else:
                    # calculer intensité inversée (plus la valeur est grande, plus foncé)
                    intensity = int(230 - (value / max_val) * 130)  # 230 → 100
                hex_color = f"{intensity:02X}{intensity:02X}{intensity:02X}"
                cell.fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
            else:
                # texte → gris clair de base
                cell.fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")

    # Ajuster largeur des colonnes
    for col in range(1, kpi_df.shape[1]+1):
        ws.column_dimensions[get_column_letter(col)].width = 15

    wb.save(filename)
    print(f"Exported {filename} successfully.")
