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
    # Filtre date
    mask = (
        (df[cob_col].dt.year == year)
        & (df[cob_col].dt.month == month)
    )

    # Filtre statut
    if str(asset_status).upper() != "ALL":
        mask &= (df[status_col] == asset_status)

    d = df.loc[mask].copy()

    if d.empty:
        return pd.DataFrame(columns=[country_col, "EXPOSURE_MILLIONS"])

    # Calcul de l'exposition
    d["EXPOSURE"] = (
        d["EXPOSURE_AMOUNT_LTR"].fillna(0)
        + d["PENDING_ORDERS"].fillna(0)
    )

    # Même logique que ton code d'origine :
    # dernière ligne par client ET par pays
    d = (
        d.sort_values(cob_col)
         .drop_duplicates(subset=[country_col, customer_col], keep="last")
    )

    # Group by pays
    if aggregation == "sum":
        result = (
            d.groupby(country_col, as_index=False)["EXPOSURE"]
             .sum()
        )
    else:
        result = (
            d.groupby(country_col, as_index=False)["EXPOSURE"]
             .mean()
        )

    # Affichage en millions
    result["EXPOSURE_MILLIONS"] = (
        result["EXPOSURE"] / 1_000_000
    ).round(4)

    return result[[country_col, "EXPOSURE_MILLIONS"]]


# Calcul
exposure_df = compute_exposure(
    df,              # remplace par le nom de ton DataFrame
    month=12,
    year=2025,
    asset_status="ALL",
    aggregation="sum",
)

# Export Excel
exposure_df.to_excel("Exposure_by_Country.xlsx", index=False)

# Affichage
print(exposure_df)