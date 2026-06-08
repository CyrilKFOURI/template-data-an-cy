import pandas as pd
import os

BASE_DATA = "precomputed_fast/data"

COUNTRIES = [
    "BELGIUM", "FRANCE", "GERMANY", "ITALY",
    "LUXEMBOURG", "NETHERLANDS", "SPAIN", "UNITED KINGDOM",
]

# ── Charger kpi9_1 (a COUNTRY_FILTER + YEAR_FILTER + PERIOD en YYYY-MM si monthly)
# ou fallback sur v1 (COUNTRY + YEAR + MONTH)
k91_path = os.path.join(BASE_DATA, "view3", "kpi9_1.parquet")
v1_path  = os.path.join(BASE_DATA, "merged_v1.parquet")

if os.path.exists(k91_path):
    df = pd.read_parquet(k91_path, columns=["COUNTRY_FILTER", "YEAR_FILTER", "PERIOD", "PERIOD_MODE_FILTER"])
    df = df[df["PERIOD_MODE_FILTER"] == "monthly"]   # PERIOD = "YYYY-MM"
    country_col = "COUNTRY_FILTER"
    df["_last_date"] = pd.to_datetime(
        df["YEAR_FILTER"].astype(str) + "-" + df["PERIOD"].str[-2:] + "-01",
        errors="coerce"
    )
elif os.path.exists(v1_path):
    df = pd.read_parquet(v1_path, columns=["COUNTRY", "YEAR", "MONTH"])
    df = df[df["MONTH"] != "ALL"]
    country_col = "COUNTRY"
    df["_last_date"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
else:
    raise FileNotFoundError("Aucun parquet trouvé — lance d'abord generate_all_precomputed.py")

# ── 1. Dernière date disponible par pays ──────────────────────────────────────
last = (
    df.groupby(country_col)["_last_date"]
    .max()
    .reset_index()
    .rename(columns={country_col: "COUNTRY", "_last_date": "DERNIERE_DATE"})
    .sort_values("DERNIERE_DATE", ascending=False)
)
last["DERNIERE_DATE"] = last["DERNIERE_DATE"].dt.strftime("%Y-%m")
print("── Dernière période disponible par pays ───────────────────────────")
print(last.to_string(index=False))

# ── 2. Pays manquants vs liste COUNTRIES ─────────────────────────────────────
existing = set(df[country_col].dropna().str.upper().unique())
expected = set(c.upper() for c in COUNTRIES)
missing  = sorted(expected - existing)

print("\n── Pays présents ──────────────────────────────────────────────────")
print(sorted(expected & existing))

print("\n── Pays manquants (dans COUNTRIES mais pas dans les parquets) ─────")
print(missing if missing else "Aucun — tous présents ✓")
