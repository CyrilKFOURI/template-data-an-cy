def compute_average_exposure(
    df: pd.DataFrame,
    country: str,
    month: int,
    year: int,
    asset_status: str,
    country_col: str = "COUNTRY",
    cob_col: str = "COB_DATE",
    status_col: str = "NOVA_ASSET_STATUS",
    customer_col: str = "ID_CUSTOMER",
) -> float:
    """
    Exposition moyenne pour un pays / mois / année de COB_DATE / statut d'actif donnés.

    Logique :
      1. Filtre sur country, year, month (COB_DATE) et asset_status.
      2. Calcule l'exposition ligne par ligne = EXPOSURE_AMOUNT_LTR + PENDING_ORDERS.
      3. Dédup sur ID_CUSTOMER unique (garde le dernier snapshot COB_DATE par client)
         — APRÈS le calcul de l'exposition ligne par ligne.
      4. Moyenne de l'exposition sur les clients uniques restants.
    """
    d = df[
        (df[country_col] == country)
        & (df[cob_col].dt.year == year)
        & (df[cob_col].dt.month == month)
        & (df[status_col] == asset_status)
    ].copy()

    if d.empty:
        return 0.0

    # 1 exposition par ligne = somme des 2 champs
    d["EXPOSURE"] = d["EXPOSURE_AMOUNT_LTR"].fillna(0) + d["PENDING_ORDERS"].fillna(0)

    # Dédup sur ID_CUSTOMER unique APRÈS le calcul (garde le dernier snapshot)
    d = d.sort_values(cob_col).drop_duplicates(subset=[customer_col], keep="last")

    # Moyenne sur les clients uniques
    return float(d["EXPOSURE"].mean())
