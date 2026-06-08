# Dernière période par pays
tmp = df_kpis[df_kpis["MONTH"] != "ALL"].copy()
tmp["_date"] = pd.to_datetime(
    tmp["YEAR"].astype(str) + "-" + tmp["MONTH"].astype(str).str.zfill(2),
    format="%Y-%m", errors="coerce"
)
last = (
    tmp.groupby("COUNTRY")["_date"]
    .max()
    .dt.strftime("%Y-%m")
    .reset_index()
    .rename(columns={"_date": "DERNIERE_PERIODE"})
    .sort_values("DERNIERE_PERIODE", ascending=False)
)
print(last.to_string(index=False))

# Pays manquants vs COUNTRIES
existing = set(df_kpis["COUNTRY"].dropna().str.upper().unique())
missing  = sorted(set(c.upper() for c in COUNTRIES) - existing)
print("\nManquants :", missing if missing else "Aucun ✓")
