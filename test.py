
ARRS_BTWN_BUCKETS = [
    ("ARRS_BTWN_0_30D", "0-30 days"),
    ("ARRS_BTWN_31_60D", "31-60 days"),
    ("ARRS_BTWN_61_90D", "61-90 days"),
    ("ARRS_BTWN_91_180D", "91-180 days"),
    ("ARRS_BTWN_181_270D", "181-270 days"),
]


def _arrears_summary(row) -> dict:
    """Total overdue amount + worst (latest) bucket with money in it, from one
    snapshot row. Total = sum of every BTWN_* bucket + the 270D+ tail."""
    present = [(c, label) for c, label in ARRS_BTWN_BUCKETS if c in ARRS_COLUMNS_PRESENT]
    total = sum(row.get(c) or 0 for c, _ in present)
    if "ARRS_MORE_270D" in ARRS_COLUMNS_PRESENT:
        total += row.get("ARRS_MORE_270D") or 0

    worst = "Current (0 days)"
    for c, label in present:
        v = row.get(c)
        if pd.notna(v) and v > 0:
            worst = label
    if "ARRS_MORE_270D" in ARRS_COLUMNS_PRESENT:
        v = row.get("ARRS_MORE_270D")
        if pd.notna(v) and v > 0:
            worst = "270+ days"

    return {"TOTAL": total, "WORST_BUCKET": worst}


def _arrears_at_date(customer_country_rows: pd.DataFrame, asof_date) -> dict | None:
    """Latest known arrears snapshot at/before asof_date — same 'as of' logic
    already used for _rating_at_date."""
    if not ARRS_COLUMNS_PRESENT or pd.isna(asof_date):
        return None
    d = customer_country_rows.dropna(subset=["COB_DATE"])
    d = d[d["COB_DATE"] <= asof_date].sort_values("COB_DATE")
    if d.empty:
        return None
    return _arrears_summary(d.iloc[-1])


def build_arrears_comparison(customer_id: str, country: str | None) -> list[dict]:
    """Per vehicle: arrears total + worst bucket at contract start vs. today."""
    d = GLOBAL_DF[GLOBAL_DF["ID_CUSTOMER"].astype(str).str.strip() == customer_id]
    if country and "COUNTRY" in d.columns:
        d = d[d["COUNTRY"] == country]
    if d.empty:
        return []

    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    veh = d.sort_values("COB_DATE") if "COB_DATE" in d.columns else d
    if keys:
        veh = veh.drop_duplicates(subset=keys, keep="last")
    if "CONTRACT_START_DATE" in veh.columns:
        veh = veh.sort_values("CONTRACT_START_DATE")

    current = _arrears_summary(d.sort_values("COB_DATE").iloc[-1]) if not d.empty else None

    rows = []
    for v in veh.to_dict("records"):
        start = v.get("CONTRACT_START_DATE")
        at_start = _arrears_at_date(d, start)
        rows.append({
            "VEHICLE_ID": v.get("VEHICLE_ID"),
            "CONTRACT_START_DATE": start.strftime("%Y-%m-%d") if pd.notna(start) else "—",
            "ARREARS_AT_ORIGINATION_TOTAL": at_start["TOTAL"] if at_start else None,
            "ARREARS_AT_ORIGINATION_WORST": at_start["WORST_BUCKET"] if at_start else "No data",
            "ARREARS_CURRENT_TOTAL": current["TOTAL"] if current else None,
            "ARREARS_CURRENT_WORST": current["WORST_BUCKET"] if current else "No data",
        })
    return rows


def print_arrears_report(rating_col: str, rating_val: str, limit: int = 10):
    """For every client currently at rating_val on rating_col, print arrears
    (total + worst bucket) at contract start vs. today."""
    disp_col = f"{rating_col}_DISP"
    matches = CURRENT_STATE_DF[CURRENT_STATE_DF[disp_col] == rating_val].head(limit)

    for _, c in matches.iterrows():
        cust_id, country = c["ID_CUSTOMER"], c["COUNTRY"]
        rows = build_arrears_comparison(cust_id, country)
        print(f"\n=== {cust_id} ({country}) ===")
        for r in rows:
            print(f"  Vehicle {r['VEHICLE_ID']} | start {r['CONTRACT_START_DATE']} | "
                  f"at origination: {r['ARREARS_AT_ORIGINATION_TOTAL']} ({r['ARREARS_AT_ORIGINATION_WORST']}) | "
                  f"current: {r['ARREARS_CURRENT_TOTAL']} ({r['ARREARS_CURRENT_WORST']})")


print_arrears_report("CLS_GROUP_RATING", "11")
