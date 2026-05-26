VUE 3 — KPI 9_1 et KPI 9_2
============================
Fichiers : data/view3/kpi9_1.parquet
           data/view3/kpi9_2.parquet


── KPI 9_1 (part de type de véhicule par période) ──────────────────────────

Colonnes de données (après filtrage) :
  PERIOD        — label de la période (ex. "01", "Q1", "2023" selon period_mode)
  PCT           — part en % (float)
  ASSET_STATUS  — statut du contrat correspondant à cette ligne

Colonnes de filtre (à utiliser pour la lookup, à retirer ensuite) :
  COUNTRY_FILTER        — code pays (ex. "ES", "FR", "ALL")
  YEAR_FILTER           — année (int, ex. 2023)
  STATUS_SELECTION      — statuts sélectionnés, pipe-séparés et ORDONNÉS selon l'ordre de
                          asset_status_options (ex. "IN FLEET|ORDER", "IN FLEET", "ALL")
                          → si l'utilisateur sélectionne plusieurs statuts, les trier dans
                          l'ordre : ALL > IN FLEET > ORDER > DEHIRE > SOLD, puis joindre avec |
  PERIOD_MODE_FILTER    — "monthly" | "quarterly" | "yearly"
  BIKE_OR_CAR_FILTER    — "ALL" | "CAR" | "MOTO" (valeurs uniques de la colonne dans les données)
  VEHICLE_CLASS_FILTER  — "ALL" ou valeur unique de CLS_VEHICLE_TYPE (ex. "PV", "LCV")
  VEHICLE_BODY_FILTER   — "ALL" ou valeur unique de MARKET_BODY_GROUP (ex. "SEDAN", "SUV")

Lookup = AND exact sur les 7 colonnes de filtre.
Résultat : une ou plusieurs lignes [PERIOD, PCT, ASSET_STATUS], une par statut sélectionné.
Affichage : bar + line chart groupé par ASSET_STATUS, axe X = PERIOD, axe Y = PCT (%).


── KPI 9_2 (mix de catégories de puissance par type de véhicule, par période) ──

Colonnes de données (après filtrage) :
  Period          — label de la période
  [POWER_CATEGORY columns] — une colonne par catégorie (ex. "BEV", "HEV", "ICE", "PHEV")
                             valeurs = volumes (int/float)

Colonnes de filtre :
  COUNTRY_FILTER        — code pays
  YEAR_FILTER           — année (int)
  STATUS_FILTER         — un seul statut (ex. "IN FLEET", "ORDER") — pas de multi-sélection
  PERIOD_MODE_FILTER    — "monthly" | "quarterly" | "yearly"
  BIKE_OR_CAR_FILTER    — "ALL" | "CAR" | "MOTO"
  VEHICLE_CLASS_FILTER  — "ALL" ou valeur unique de CLS_VEHICLE_TYPE
  VEHICLE_BODY_FILTER   — "ALL" ou valeur unique de MARKET_BODY_GROUP

Lookup = AND exact sur les 7 colonnes de filtre.
Résultat : tableau pivot — index = Period, colonnes = catégories de puissance, valeurs = volumes.
Affichage : line chart multi-traces (une trace par catégorie de puissance).


── Comment peupler les dropdowns ───────────────────────────────────────────

COUNTRY      : valeurs uniques de COUNTRY_FILTER (exclure "ALL")
YEAR         : valeurs uniques de YEAR_FILTER (int), tri croissant
PERIOD_MODE  : ["monthly", "quarterly", "yearly"]
BIKE_OR_CAR  : valeurs uniques de BIKE_OR_CAR_FILTER
VEHICLE_CLASS: valeurs uniques de VEHICLE_CLASS_FILTER (inclure "ALL")
VEHICLE_BODY : valeurs uniques de VEHICLE_BODY_FILTER (inclure "ALL")
STATUS (9_1) : multi-select — valeurs uniques de l'union de tous les tokens pipe de STATUS_SELECTION
STATUS (9_2) : valeurs uniques de STATUS_FILTER

















VUE 4 — KPI 10 (Top modèles à l'EOC)
======================================
Fichier : data/view4/kpi10.parquet


── Colonnes de données (après filtrage) ───────────────────────────────────

  Period          — année EOC (ex. "2023", "2024") ou "Total" pour la ligne de totaux
  [MODEL columns] — jusqu'à 10 colonnes, une par modèle (valeur = MARKET_MODEL)
                    ex. "PEUGEOT 208", "RENAULT CLIO"
                    valeurs = volumes (float)

Note : la ligne avec Period == "Total" contient la somme de tous les véhicules
       sur la plage de dates sélectionnée. Les lignes restantes sont une par année.


── Colonnes de filtre ──────────────────────────────────────────────────────

  COUNTRY_FILTER       — code pays (ex. "ES", "FR", "ALL")
  START_YEAR_FILTER    — première année de la plage EOC (int, ex. 2022)
  END_YEAR_FILTER      — dernière année de la plage EOC (int, ex. 2025)
  ASSET_STATUS_FILTER  — statut unique (ex. "IN FLEET", "ORDER", "DEHIRE", "ALL")
  BIKE_OR_CAR_FILTER   — "ALL" | "CAR" | "MOTO"

Lookup = AND exact sur les 5 colonnes de filtre.
END_YEAR_FILTER >= START_YEAR_FILTER toujours vrai dans les données.


── Affichage ────────────────────────────────────────────────────────────────

Graphe : bar chart groupé — axe X = années (lignes hors "Total"),
         une barre par modèle + une trace Scatter (axe Y secondaire) pour le total.
Tableau : tableau brut Period + colonnes modèles (incluant la ligne "Total").


── Comment peupler les dropdowns ───────────────────────────────────────────

COUNTRY      : valeurs uniques de COUNTRY_FILTER (exclure "ALL" si désiré)
START_YEAR   : valeurs uniques de START_YEAR_FILTER (int), tri croissant
END_YEAR     : valeurs uniques de END_YEAR_FILTER (int), tri croissant
               → forcer END_YEAR >= START_YEAR dans l'UI
ASSET_STATUS : valeurs uniques de ASSET_STATUS_FILTER
BIKE_OR_CAR  : valeurs uniques de BIKE_OR_CAR_FILTER









VUE 6 — KPI 13 (Portfolio vs Marché)
======================================
Fichier : data/view6/kpi13.parquet


── Colonnes de données (après filtrage) ───────────────────────────────────

  COUNTRY           — nom du pays (ex. "Spain", "France")
  PERIOD            — label de la période (ex. "2023-01", "Q1", "2023")
  BRAND             — valeur de la variable sélectionnée en majuscules
                      (ex. marque, OEM, bucket CO2, BEV flag)
  volume_portfolio  — volume du parc Arval (float)
  share_portfolio   — part du parc Arval en % (float, arrondi 2 décimales)
  volume_market     — volume marché (float)
  share_market      — part marché en % (float, arrondi 2 décimales)
  ratio             — share_portfolio / share_market (float, arrondi 2 décimales)


── Colonnes de filtre ──────────────────────────────────────────────────────

  STATUS_FILTER       — statut unique (ex. "IN FLEET", "ORDER", "DEHIRE", "ORDER YTD")
  PORT_REG_FILTER     — type véhicule côté portfolio, pipe-séparé
                        valeurs possibles : "PV" | "LCV" | "PV|LCV"
  MKT_REG_FILTER      — type enregistrement côté marché, pipe-séparé
                        valeurs possibles (combinaisons) :
                          "Passenger Cars"
                          "Light Commercial Vehicle"
                          "Heavy Commercial Vehicle"
                          "Passenger Cars|Light Commercial Vehicle"
                          "Passenger Cars|Heavy Commercial Vehicle"
                          "Light Commercial Vehicle|Heavy Commercial Vehicle"
                          "Passenger Cars|Light Commercial Vehicle|Heavy Commercial Vehicle"
  VARIABLE_FILTER     — variable de concentration :
                          "BRAND_UPDATE"  → BRAND = marque
                          "OEM_UPDATE"    → BRAND = groupe OEM
                          "HIGHEST_BEV"  → BRAND = flag BEV
                          "CO2_BUCKET"   → BRAND = bucket CO2
  PERIOD_MODE_FILTER  — "monthly" | "quarterly" | "yearly"
  TOP_N_FILTER        — "3" | "5" | "10" | "ALL"
  OWNER_FILTER        — type de propriétaire marché : "ALL" ou valeur unique de Ownertype

Lookup = AND exact sur les 7 colonnes de filtre.

Important pour PORT_REG_FILTER et MKT_REG_FILTER :
  Si l'utilisateur sélectionne une liste, trier selon l'ordre original et joindre avec "|"
  Exemple : ["Light Commercial Vehicle", "Passenger Cars"] → "Passenger Cars|Light Commercial Vehicle"
  (l'ordre dans le parquet suit toujours l'ordre de définition dans le générateur)


── Affichage ────────────────────────────────────────────────────────────────

Graphe : scatter/bar comparant share_portfolio vs share_market par BRAND et PERIOD,
         groupé par COUNTRY. Le ratio indique la sur/sous-représentation dans le parc.
Tableau : colonnes [COUNTRY, PERIOD, BRAND, volume_portfolio, share_portfolio,
                    volume_market, share_market, ratio]


── Comment peupler les dropdowns ───────────────────────────────────────────

STATUS          : valeurs uniques de STATUS_FILTER
PORT_REG        : ["PV", "LCV", "PV|LCV"] — fixe, multi-select
MKT_REG         : 7 combinaisons listées ci-dessus — fixe, multi-select
VARIABLE        : ["BRAND_UPDATE", "OEM_UPDATE", "HIGHEST_BEV", "CO2_BUCKET"] — fixe
PERIOD_MODE     : ["monthly", "quarterly", "yearly"] — fixe
TOP_N           : ["3", "5", "10", "ALL"] — fixe
OWNER           : valeurs uniques de OWNER_FILTER
