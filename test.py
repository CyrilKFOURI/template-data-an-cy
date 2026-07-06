import pandas as pd
import matplotlib.pyplot as plt

def power_category_summary(df):
    """
    Create a pie chart and a summary table for POWER_CATEGORY.

    Returns
    -------
    summary : pd.DataFrame
        Columns:
        - Energy
        - Share (%)
        - Nb of Units
    """

    # Count occurrences
    counts = df["POWER_CATEGORY"].value_counts(dropna=False)

    # Summary table
    summary = pd.DataFrame({
        "Energy": counts.index.astype(str),
        "Share (%)": (counts / counts.sum() * 100).round(2),
        "Nb of Units": counts.values
    })

    # Add total row
    total_row = pd.DataFrame({
        "Energy": ["Total"],
        "Share (%)": [100.00],
        "Nb of Units": [counts.sum()]
    })

    summary = pd.concat([summary, total_row], ignore_index=True)

    # Pie chart (without Total row)
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Distribution of Energy Categories")
    plt.axis("equal")
    plt.show()

    return summary