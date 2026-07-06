def compute_exposure(df, country, month, year, asset_status, aggregation="sum",
                      country_col="COUNTRY", cob_col="COB_DATE",
                      status_col="NOVA_ASSET_STATUS", customer_col="ID_CUSTOMER"):
    mask = (df[country_col] == country) & (df[cob_col].dt.year == year) & (df[cob_col].dt.month == month)
    if str(asset_status).upper() != "ALL":
        mask &= df[status_col] == asset_status
    d = df[mask].copy()
    if d.empty:
        return 0.0
    d["EXPOSURE"] = d["EXPOSURE_AMOUNT_LTR"].fillna(0) + d["PENDING_ORDERS"].fillna(0)
    d = d.sort_values(cob_col).drop_duplicates(subset=[customer_col], keep="last")
    return float(d["EXPOSURE"].sum() if aggregation == "sum" else d["EXPOSURE"].mean())


def format_millions(value):
    s = f"{value / 1_000_000:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


exposure = compute_exposure(UC1_DF, "LU", 12, 2025, "IN FLEET", aggregation="sum")
print(f"L'exposition est de {format_millions(exposure)} millions")



exposure = compute_exposure(UC1_DF, "LU", 12, 2025, "ALL", aggregation="sum")
print(f"L'exposition est de {format_millions(exposure)} millions")
