import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo

# export dataframe to excel
output_file = "pct_by_country.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    pct_by_country.to_excel(writer, sheet_name="pct_by_country")

# reopen workbook to style
wb = load_workbook(output_file)
ws = wb["pct_by_country"]

# basic styling
header_fill = PatternFill("solid", fgColor="1F4E78")  # blue
header_font = Font(color="FFFFFF", bold=True)
center = Alignment(horizontal="center", vertical="center")

# style header row
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center

# style index column
for cell in ws["A"][1:]:
    cell.font = Font(bold=True)

# freeze panes
ws.freeze_panes = "B2"

# add Excel table
max_row = ws.max_row
max_col = ws.max_column
table_ref = f"A1:{chr(64 + max_col)}{max_row}"
tab = Table(displayName="PctByCountryTable", ref=table_ref)
style = TableStyleInfo(
    name="TableStyleMedium2",
    showFirstColumn=False,
    showLastColumn=False,
    showRowStripes=True,
    showColumnStripes=False,
)
tab.tableStyleInfo = style
ws.add_table(tab)

wb.save(output_file)
