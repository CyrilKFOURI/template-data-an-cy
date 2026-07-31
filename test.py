import pandas as pd

# df = ton DataFrame NOVA déjà chargé. Colonnes attendues :
# ID_CUSTOMER, COUNTRY, COB_DATE, CONTRACT_START_DATE, ID_CONTRACT, VEHICLE_ID, ID_QUOTATION,
# GROUP_RATING, COUNTERPARTY_RATING, CLS_GROUP_RATING,
# ARRS_BTWN_0_30D, ARRS_BTWN_31_60D, ARRS_BTWN_61_90D, ARRS_BTWN_91_180D, ARRS_BTWN_181_270D, ARRS_MORE_270D

UNIQUE_KEY_COLS = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]

RATING_COLUMNS = [
    {"col": "GROUP_RATING", "numeric": False},
    {"col": "COUNTERPARTY_RATING", "numeric": False},
    {"col": "CLS_GROUP_RATING", "numeric": True},
]
RATING_COL_NUMERIC = {r["col"]: r["numeric"] for r in RATING_COLUMNS}

WANTED_ARRS_COLUMNS = [
    "ARRS_BTWN_0_30D", "ARRS_BTWN_31_60D", "ARRS_BTWN_61_90D",
    "ARRS_BTWN_91_180D", "ARRS_BTWN_181_270D",
    "ARRS_MORE_30D", "ARRS_MORE_60D", "ARRS_MORE_90D", "ARRS_MORE_180D", "ARRS_MORE_270D",
]
ARRS_COLUMNS_PRESENT = [c for c in WANTED_ARRS_COLUMNS if c in df.columns]

ARRS_BTWN_BUCKETS = [
    ("ARRS_BTWN_0_30D", "0-30 days"),
    ("ARRS_BTWN_31_60D", "31-60 days"),
    ("ARRS_BTWN_61_90D", "61-90 days"),
    ("ARRS_BTWN_91_180D", "91-180 days"),
    ("ARRS_BTWN_181_270D", "181-270 days"),
]


def _fmt_rating(col, v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if RATING_COL_NUMERIC.get(col):
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else f"{f:g}"
        except (TypeError, ValueError):
            return str(v)
    return str(v).strip()


def _build_current_state(df):
    rating_cols = [r["col"] for r in RATING_COLUMNS if r["col"] in df.columns]
    keep_cols = [c for c in (["ID_CUSTOMER", "COUNTRY", "COB_DATE"] + rating_cols) if c in df.columns]
    d = df[keep_cols].dropna(subset=["ID_CUSTOMER"]).copy()
    d["ID_CUSTOMER"] = d["ID_CUSTOMER"].astype(str).str.strip()
    d = d.sort_values("COB_DATE")
    fill_cols = [c for c in keep_cols if c not in ("ID_CUSTOMER", "COB_DATE")]
    if fill_cols:
        d[fill_cols] = d.groupby("ID_CUSTOMER")[fill_cols].ffill()
    d = d.drop_duplicates(subset=["ID_CUSTOMER"], keep="last")
    for col in rating_cols:
        d[f"{col}_DISP"] = d[col].apply(lambda v, c=col: _fmt_rating(c, v))
    return d


CURRENT_STATE_DF = _build_current_state(df)


def _arrears_summary(row):
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


def _arrears_at_date(customer_country_rows, asof_date):
    if not ARRS_COLUMNS_PRESENT or pd.isna(asof_date):
        return None
    d = customer_country_rows.dropna(subset=["COB_DATE"])
    d = d[d["COB_DATE"] <= asof_date].sort_values("COB_DATE")
    if d.empty:
        return None
    return _arrears_summary(d.iloc[-1])


def build_arrears_comparison(customer_id, country):
    d = df[df["ID_CUSTOMER"].astype(str).str.strip() == customer_id]
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


def print_arrears_report(rating_col, rating_val, limit=10):
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
