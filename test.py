import pandas as pd

def compute_exposure(
    df,
    month,
    year,
    asset_status,
    aggregation="sum",
    country_col="COUNTRY",
    cob_col="COB_DATE",
    status_col="NOVA_ASSET_STATUS",
    customer_col="ID_CUSTOMER",
):
    # Filtre
    mask = (
        (df[cob_col].dt.year == year)
        & (df[cob_col].dt.month == month)
    )

    if str(asset_status).upper() != "ALL":
        mask &= df[status_col] == asset_status

    d = df.loc[mask].copy()

    if d.empty:
        return pd.DataFrame(columns=[country_col, "EXPOSURE"])

    # Calcul de l'exposition
    d["EXPOSURE"] = (
        d["EXPOSURE_AMOUNT_LTR"].fillna(0)
        + d["PENDING_ORDERS"].fillna(0)
    )

    # Dernière ligne par client
    d = (
        d.sort_values(cob_col)
         .drop_duplicates(subset=[customer_col], keep="last")
    )

    # Agrégation par pays
    if aggregation == "sum":
        result = d.groupby(country_col, as_index=False)["EXPOSURE"].sum()
    else:
        result = d.groupby(country_col, as_index=False)["EXPOSURE"].mean()

    return result


# Calcul
exposure_df = compute_exposure(
    n,
    month=12,
    year=2025,
    asset_status="ALL",
    aggregation="sum",
)

# Export Excel
exposure_df.to_excel("Exposure_by_Country_2025_12.xlsx", index=False)

print(exposure_df)