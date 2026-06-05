import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

output_file = "pct_by_country.xlsx"

# Export
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    pct_by_country.to_excel(writer, sheet_name="pct_by_country")

# Reopen
wb = load_workbook(output_file)
ws = wb["pct_by_country"]

# Palette light / élégante
header_fill = PatternFill("solid", fgColor="DCEAF7")   # bleu très clair
header_font = Font(color="1F3B5B", bold=True)
index_fill = PatternFill("solid", fgColor="F5F7FA")    # gris bleuté très light
index_font = Font(color="2F2F2F", bold=True)
thin_border = Border(
    bottom=Side(style="thin", color="E6ECF2")
)
center = Alignment(horizontal="center", vertical="center")
left = Alignment(horizontal="left", vertical="center")

# Header row
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center
    cell.border = thin_border

# Index column (pays)
for row in range(2, ws.max_row + 1):
    cell = ws.cell(row=row, column=1)
    cell.fill = index_fill
    cell.font = index_font
    cell.alignment = left

# Align data + subtle number format
for row in range(2, ws.max_row + 1):
    for col in range(2, ws.max_column + 1):
        cell = ws.cell(row=row, column=col)
        cell.alignment = center
        cell.number_format = '0.0'

# Conditional formatting on numeric area
if ws.max_row >= 2 and ws.max_column >= 2:
    data_range = f"B2:{get_column_letter(ws.max_column)}{ws.max_row}"
    color_scale = ColorScaleRule(
        start_type='min', start_color='FFFFFF',      # blanc
        mid_type='percentile', mid_value=50, mid_color='DCEAF7',  # bleu très clair
        end_type='max', end_color='8FB8DE'           # bleu doux
    )
    ws.conditional_formatting.add(data_range, color_scale)

# Freeze panes
ws.freeze_panes = "B2"

# Auto width
for col in range(1, ws.max_column + 1):
    col_letter = get_column_letter(col)
    max_length = 0
    for cell in ws[col_letter]:
        val = "" if cell.value is None else str(cell.value)
        max_length = max(max_length, len(val))
    ws.column_dimensions[col_letter].width = min(max_length + 2, 18)

# Optional: make header row a bit taller
ws.row_dimensions[1].height = 22

wb.save(output_file)
