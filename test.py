import pandas as pd
import plotly.express as px

# =========================
# 1. AGRÉGATION
# =========================
grouped = (
    nova
    .groupby(["BRAND_UPDATE", "BODY_GROUP"])
    .size()
    .reset_index(name="count")
)

# total par marque
totals = (
    nova
    .groupby("BRAND_UPDATE")
    .size()
    .reset_index(name="total")
)

# merge
df = grouped.merge(totals, on="BRAND_UPDATE")

# % calcul
df["pct"] = df["count"] / df["total"] * 100


# =========================
# 2. PLOTLY (STACKED BAR %)
# =========================
fig = px.bar(
    df,
    x="BRAND_UPDATE",
    y="pct",
    color="BODY_GROUP",
    title="BODY GROUP distribution per brand (%)",
    labels={"pct": "% share", "BRAND_UPDATE": "Brand"},
)

fig.update_layout(
    barmode="stack",
    xaxis_tickangle=-45,
    yaxis_title="% within brand"
)

fig.show()


# =========================
# 3. (OPTIONNEL) TABLE SUV ONLY
# =========================
suv = df[df["BODY_GROUP"] == "SUV"].sort_values("pct", ascending=False)

fig2 = px.bar(
    suv,
    x="BRAND_UPDATE",
    y="pct",
    title="SUV share per brand (%)",
    labels={"pct": "SUV %", "BRAND_UPDATE": "Brand"}
)

fig2.update_layout(xaxis_tickangle=-45)
fig2.show()