def compute_average_exposure(
    df: pd.DataFrame,
    country: str,
    month: int,
    year: int,
    asset_status: str,
    field_aggregation: str = "sum",  # "sum" ou "mean" sur EXPOSURE_AMOUNT_LTR + PENDING_ORDERS
    country_col: str = "COUNTRY",
    cob_col: str = "COB_DATE",
    status_col: str = "NOVA_ASSET_STATUS",
    customer_col: str = "ID_CUSTOMER",
) -> float:
    """
    Exposition moyenne pour un pays / mois / année de COB_DATE / statut d'actif donnés.

    Logique :
      1. Filtre sur country, year, month (COB_DATE) et asset_status.
      2. Calcule l'exposition ligne par ligne = agrégation (sum ou mean) de
         EXPOSURE_AMOUNT_LTR et PENDING_ORDERS.
      3. Dédup sur ID_CUSTOMER unique (garde le dernier snapshot COB_DATE par client)
         — APRÈS le calcul de l'exposition ligne par ligne.
      4. Moyenne de l'exposition sur les clients uniques restants.
    """
    if field_aggregation not in ("sum", "mean"):
        raise ValueError('field_aggregation doit être "sum" ou "mean"')

    d = df[
        (df[country_col] == country)
        & (df[cob_col].dt.year == year)
        & (df[cob_col].dt.month == month)
        & (df[status_col] == asset_status)
    ].copy()

    if d.empty:
        return 0.0

    # 1 exposition par ligne = sum ou mean des 2 champs, au choix
    fields = d[["EXPOSURE_AMOUNT_LTR", "PENDING_ORDERS"]].fillna(0)
    d["EXPOSURE"] = fields.sum(axis=1) if field_aggregation == "sum" else fields.mean(axis=1)

    # Dédup sur ID_CUSTOMER unique APRÈS le calcul (garde le dernier snapshot)
    d = d.sort_values(cob_col).drop_duplicates(subset=[customer_col], keep="last")

    # Moyenne sur les clients uniques
    return float(d["EXPOSURE"].mean())


avg_exposure = compute_average_exposure(
    df=UC1_DF,
    country="LU",
    month=12,
    year=2025,
    asset_status="IN FLEET",
    field_aggregation="sum",  # ou "mean" selon ce que tu veux tester
)
