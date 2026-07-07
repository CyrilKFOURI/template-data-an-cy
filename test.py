import pandas as pd
import matplotlib.pyplot as plt

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter


def generate_power_category_report(nova, output_file="Power_Category_Report.xlsx"):

    # ==========================
    # Create summary table
    # ==========================
    counts = nova["POWER_CATEGORY"].value_counts(dropna=False)

    summary = pd.DataFrame({
        "Energy": counts.index.astype(str),
        "Share (%)": ((counts / counts.sum()) * 100).round(2),
        "Nb of Units": counts.values
    })

    total_row = pd.DataFrame({
        "Energy": ["Total"],
        "Share (%)": [100.00],
        "Nb of Units": [counts.sum()]
    })

    summary = pd.concat([summary, total_row], ignore_index=True)


    # ==========================
    # Create full pie chart
    # ==========================
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.pie(
        counts,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1
        }
    )

    ax.set_title(
        "Distribution of Energy Categories - IN FLEET",
        fontsize=14,
        fontweight="bold"
    )

    ax.axis("equal")

    chart_file = "power_category_pie.png"

    plt.savefig(
        chart_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==========================
    # Export Excel
    # ==========================
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        summary.to_excel(
            writer,
            sheet_name="Energy Summary",
            index=False,
            startrow=2
        )

        workbook = writer.book
        ws = writer.sheets["Energy Summary"]

        # Title
        ws["A1"] = "POWER CATEGORY SUMMARY"
        ws["A1"].font = Font(
            size=16,
            bold=True
        )


        # ==========================
        # Styling
        # ==========================
        header_fill = PatternFill(
            fill_type="solid",
            fgColor="1F4E78"
        )

        header_font = Font(
            color="FFFFFF",
            bold=True
        )

        border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin")
        )


        # Header row
        for cell in ws[3]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = border


        # Body
        for row in ws.iter_rows(
            min_row=4,
            max_row=ws.max_row,
            min_col=1,
            max_col=3
        ):
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal="center")


        # Highlight Total row
        for cell in ws[ws.max_row]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="D9EAD3"
            )


        # Adjust column width
        for col in ws.columns:
            max_length = max(
                len(str(cell.value)) if cell.value else 0
                for cell in col
            )

            ws.column_dimensions[
                get_column_letter(col[0].column)
            ].width = max_length + 5


        # ==========================
        # Add chart
        # ==========================
        img = Image(chart_file)

        img.width = 500
        img.height = 500

        ws.add_image(img, "E2")


    return summary