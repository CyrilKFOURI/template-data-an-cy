tmp = df_kpis[df_kpis["MONTH"] != "ALL"].copy()
tmp["YEAR"]  = tmp["YEAR"].astype(int)
tmp["MONTH"] = pd.to_numeric(tmp["MONTH"], errors="coerce")

# Pour chaque pays : dernière année, puis dans cette année le dernier mois
last_year  = tmp.groupby("COUNTRY")["YEAR"].max().reset_index(name="LAST_YEAR")
tmp2       = tmp.merge(last_year, on="COUNTRY")
tmp2       = tmp2[tmp2["YEAR"] == tmp2["LAST_YEAR"]]
last_month = tmp2.groupby("COUNTRY")["MONTH"].max().reset_index(name="LAST_MONTH")

result = last_year.merge(last_month, on="COUNTRY")
result["DERNIERE_PERIODE"] = result["LAST_YEAR"].astype(str) + "-" + result["LAST_MONTH"].astype(int).astype(str).str.zfill(2)
print(result[["COUNTRY", "DERNIERE_PERIODE"]].sort_values("DERNIERE_PERIODE", ascending=False).to_string(index=False))

# Pays manquants
existing = set(df_kpis["COUNTRY"].dropna().str.upper().unique())
missing  = sorted(set(c.upper() for c in COUNTRIES) - existing)
print("\nManquants :", missing if missing else "Aucun ✓")
