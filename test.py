
                         POWER_CATEGORY
                               │
              ┌────────────────┴────────────────┐
              │                                  │
            MHEV                          NOT IDENTIFIED
              │                                  │
              ▼                                  ▼
     FUEL_TYPE2 reconnu ? ──────────────  FUEL_TYPE2 reconnu ?
        │            │                       │            │
       OUI           NON                    OUI           NON
        │            │                       │            │
        ▼            ▼                       ▼            ▼
  résultat =   FUEL_TYPE (1)           résultat =   FUEL_TYPE (1)
  FUEL_TYPE2    reconnu ?              FUEL_TYPE2    reconnu ?
                │      │                             │      │
               OUI    NON                           OUI    NON
                │      │                             │      │
                ▼      ▼                             ▼      ▼
          résultat =  "MHEV NOT               résultat =  "NOT
          FUEL_TYPE   IDENTIFIED"             FUEL_TYPE   IDENTIFIED"




# =============================================================================
# Nouveau mapping FUEL_TYPE2 / FUEL_TYPE -> catégorie normalisée
# Construit à partir des valeurs vues dans les pivots (MHEV et NOT IDENTIFIED),
# indépendant du pays (une valeur comme "DIESEL" signifie la même chose
# partout). Remplace/complète fuel_mapping + fuel_suffix_map existants.
# =============================================================================

FUEL_VALUE_MAPPING = {
    # --- Essence / Petrol ---
    "UNLEADED":            "PETROL",
    "BENZINA":              "PETROL",
    "BENZINA (MILDHYBRID)": "PETROL",
    "GASOLINA":              "PETROL",
    "ESSENCE":               "PETROL",
    "MHEV-G":                "PETROL",   # déjà taggé mild-hybrid essence

    # --- Diesel ---
    "DIESEL":               "DIESEL",
    "GASOIL":                "DIESEL",   # FR/BE : gasoil = diesel -- à confirmer
    "MHEV DIESEL":           "DIESEL",
    "MHEV-D":                "DIESEL",

    # --- Electrique ---
    "ELECTRIQUE":            "ELECTRIC",
    "ELETTRICO":             "ELECTRIC",
    "ELECTRICITY":           "ELECTRIC",
    "ELECTRIC":              "ELECTRIC",
    "BEV":                   "ELECTRIC",

    # --- Hybride (non rechargeable) ---
    "HEV":                   "HYBRID",
    "HYBRID-D":              "HYBRID",     # hybride diesel
    "HYBRIDE-ESSENCE":       "HYBRID",     # hybride essence

    # --- Carburants alternatifs ---
    "BIO-ETHANOL":           "BIOETHANOL",
    "GLP":                   "LPG",
    "HYDROGENE":             "HYDROGEN",
    "FLEX":                  "FLEX",       # carburant flex (Brésil) -- à confirmer

    # --- Valeurs "pas bonnes" -> doivent tomber au fallback suivant ---
    "<NA>":  None,
    "NAN":   None,
    "":      None,
    "NONE":  None,
}


def _map_fuel_value(value) -> str | None:
    """Retourne la catégorie normalisée pour une valeur brute FUEL_TYPE/FUEL_TYPE2,
    ou None si la valeur est absente/inconnue (doit déclencher le fallback suivant)."""
    v = str(value).strip().upper()
    return FUEL_VALUE_MAPPING.get(v)


def apply_new_power_category_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Réaffecte POWER_CATEGORY quand il vaut MHEV ou NOT IDENTIFIED, en
    utilisant FUEL_TYPE2 puis FUEL_TYPE(1) comme sources de vérité.
      - MHEV            -> FUEL_TYPE2 si reconnu, sinon FUEL_TYPE, sinon "MHEV NOT IDENTIFIED"
      - NOT IDENTIFIED  -> FUEL_TYPE2 si reconnu, sinon FUEL_TYPE, sinon "NOT IDENTIFIED"
      - tout autre POWER_CATEGORY -> inchangé
    """
    power_category = df["POWER_CATEGORY"].astype(str).str.strip().str.upper()
    fuel_type2 = df.get("FUEL_TYPE2", pd.Series(index=df.index, dtype="object"))
    fuel_type1 = df.get("FUEL_TYPE", pd.Series(index=df.index, dtype="object"))

    mapped_fuel2 = fuel_type2.map(_map_fuel_value)
    mapped_fuel1 = fuel_type1.map(_map_fuel_value)
    resolved = mapped_fuel2.fillna(mapped_fuel1)

    is_mhev = power_category.eq("MHEV")
    is_not_identified = power_category.eq("NOT IDENTIFIED")

    df["POWER_CATEGORY_NEW"] = df["POWER_CATEGORY"]
    df.loc[is_mhev, "POWER_CATEGORY_NEW"] = resolved[is_mhev].fillna("MHEV NOT IDENTIFIED")
    df.loc[is_not_identified, "POWER_CATEGORY_NEW"] = resolved[is_not_identified].fillna("NOT IDENTIFIED")

    return df
