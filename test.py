VUE 4 — KPI 10 (Top modèles à l'EOC)
======================================
Fichier : data/view4/kpi10.parquet


── Colonnes brutes du parquet ──────────────────────────────────────────────────

  PERIOD              — label de la période dérivé de CONTRACT_FINAL_END :
                          "2026Q1"  si period_mode = "quarterly"
                          "2026-03" si period_mode = "monthly"
                          "2026"    si period_mode = "yearly"
  EOC_YEAR            — année entière (int, ex. 2026) → filtre de plage
  MARKET_MODEL        — libellé du modèle (ex. "PEUGEOT 208", "RENAULT CLIO")
  COUNT               — nombre de contrats uniques (int)
  COUNTRY_FILTER      — code pays (ex. "ES", "FR", "ALL")
  ASSET_STATUS_FILTER — statut (ex. "IN FLEET", "ORDER", "DEHIRE", "ALL")
  BIKE_OR_CAR_FILTER  — "ALL" | "CAR" | "MOTO"
  PERIOD_MODE_FILTER  — "monthly" | "quarterly" | "yearly"

Format : LONG — une ligne par (PERIOD, MARKET_MODEL, status, boc, period_mode).
Le pivot + top-10 + Total se font dans le reader (voir fonction ci-dessous).


── Filtre de lookup ─────────────────────────────────────────────────────────────

  COUNTRY_FILTER      == country          (exact)
  ASSET_STATUS_FILTER == asset_status     (exact)
  BIKE_OR_CAR_FILTER  == bike_or_car      (exact)
  PERIOD_MODE_FILTER  == period_mode      (exact)
  EOC_YEAR            >= start_year       (plage)
  EOC_YEAR            <= end_year         (plage)


── Output après pivot ───────────────────────────────────────────────────────────

  Period  | <modèle 1> | <modèle 2> | … | <modèle 10>
  --------|-----------|-----------|---|------------
  2026Q1  |    590    |    409    | … |     110
  2026Q2  |    655    |    371    | … |     119
  …
  Total   |   4 800   |   3 500   | … |     800

  - Lignes  : une par période dans la plage (triées par PERIOD)
  - Colonnes : top-10 modèles par volume total sur la plage, triés décroissant
  - Ligne "Total" : somme de chaque colonne sur toutes les périodes


── Comment peupler les dropdowns ───────────────────────────────────────────────

  COUNTRY      : valeurs uniques de COUNTRY_FILTER (exclure "ALL" si désiré)
  START_YEAR   : valeurs uniques de EOC_YEAR (int), tri croissant
  END_YEAR     : idem → forcer END_YEAR >= START_YEAR dans l'UI
  ASSET_STATUS : valeurs uniques de ASSET_STATUS_FILTER
  BIKE_OR_CAR  : valeurs uniques de BIKE_OR_CAR_FILTER
  PERIOD_MODE  : ["monthly", "quarterly", "yearly"] — fixe


── Code de lookup ──────────────────────────────────────────────────────────────

import pandas as pd

def get_kpi10(df, country, start_year, end_year, asset_status, bike_or_car,
              period_mode="quarterly"):
    """
    Retourne un DataFrame pivoté :
      - lignes  = périodes (PERIOD) dans [start_year, end_year]
      - colonnes = top-10 modèles (MARKET_MODEL) par volume total, triés décroissant
      - dernière ligne = "Total" : somme de chaque colonne sur toutes les périodes

    Paramètres
    ----------
    country      : str   ex. "ES", "FR"
    start_year   : int   ex. 2026
    end_year     : int   ex. 2029
    asset_status : str   "IN FLEET" | "ORDER" | "DEHIRE" | "ALL"
    bike_or_car  : str   "ALL" | "CAR" | "MOTO"
    period_mode  : str   "monthly" | "quarterly" | "yearly"
    """
    mask = (
        (df["COUNTRY_FILTER"]      == country) &
        (df["ASSET_STATUS_FILTER"] == asset_status) &
        (df["BIKE_OR_CAR_FILTER"]  == bike_or_car) &
        (df["PERIOD_MODE_FILTER"]  == period_mode) &
        (df["EOC_YEAR"]            >= int(start_year)) &
        (df["EOC_YEAR"]            <= int(end_year))
    )
    rows = df[mask]
    if rows.empty:
        return pd.DataFrame()

    # Pivot : PERIOD en index, MARKET_MODEL en colonnes, COUNT agrégé
    pivot = rows.pivot_table(
        index="PERIOD", columns="MARKET_MODEL", values="COUNT",
        aggfunc="sum", fill_value=0,
    )

    # Top-10 modèles triés par volume total décroissant
    top10 = pivot.sum().nlargest(10).index
    pivot = pivot[top10]

    # Ligne Total : somme de chaque colonne sur toutes les périodes
    total_row = pd.DataFrame(
        [pivot.sum()],
        index=pd.Index(["Total"], name="PERIOD"),
    )
    pivot = pd.concat([pivot, total_row])

    result = pivot.reset_index().rename(columns={"PERIOD": "Period"})
    result["Period"] = result["Period"].astype(str)
    return result


── Exemple d'appel ─────────────────────────────────────────────────────────────

df = pd.read_parquet("precomputed_fast/data/view4/kpi10.parquet")

result = get_kpi10(
    df,
    country      = "ES",
    start_year   = 2026,
    end_year     = 2029,
    asset_status = "IN FLEET",
    bike_or_car  = "ALL",
    period_mode  = "quarterly",
)

# result :
# Period  | TUCSON | QASHQAI | XC40 | SPORTAGE | … (top 10)
# 2026Q1  |   590  |   409   |  451 |   275    | …
# 2026Q2  |   655  |   371   |  445 |   382    | …
# …
# Total   |  4800  |  3500   | 3200 |  2900    | …

import pandas as pd

def get_kpi10(df, country, start_year, end_year, asset_status, bike_or_car,
              period_mode="quarterly"):
    mask = (
        (df["COUNTRY_FILTER"]      == country) &
        (df["ASSET_STATUS_FILTER"] == asset_status) &
        (df["BIKE_OR_CAR_FILTER"]  == bike_or_car) &
        (df["PERIOD_MODE_FILTER"]  == period_mode) &
        (df["EOC_YEAR"]            >= int(start_year)) &
        (df["EOC_YEAR"]            <= int(end_year))
    )
    rows = df[mask]
    if rows.empty:
        return pd.DataFrame()

    pivot = rows.pivot_table(
        index="PERIOD", columns="MARKET_MODEL", values="COUNT",
        aggfunc="sum", fill_value=0,
    )
    top10 = pivot.sum().nlargest(10).index
    pivot = pivot[top10]
    total_row = pd.DataFrame(
        [pivot.sum()],
        index=pd.Index(["Total"], name="PERIOD"),
    )
    pivot = pd.concat([pivot, total_row])
    result = pivot.reset_index().rename(columns={"PERIOD": "Period"})
    result["Period"] = result["Period"].astype(str)
    return result










