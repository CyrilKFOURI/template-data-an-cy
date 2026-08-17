
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, callback_context, dcc, html, no_update

# =============================================================================
# Config — adjust to your environmenta
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"                                          # NOVA data
CREDIT_VERIFICATION_DATA_PATH = BASE_DIR / "credit_verification_data.parquet"  # CLS/obligor extract

# If CREDIT_VERIFICATION_DATA_PATH doesn't exist, every section that depends
# on it degrades gracefully and says so, exactly like models.parquet
# elsewhere in this project — nothing crashes, NOVA-only checks still run.

COUNTRIES_TO_READ: list[str] = ["SPAIN"]
START_YYYYMM = "202001"
END_YYYYMM   = "202512"

UNIQUE_KEY_COLS = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]

NOVA_COLUMNS = [
    "COB_DATE", "ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION",
    "COUNTRY", "NOVA_ASSET_STATUS", "CONTRACT_START_DATE",
    "ID_CUSTOMER", "GROUP_RATING", "COUNTERPARTY_RATING", "CLS_GROUP_RATING",
    "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION", "SHARED_CLIENT_FLAG",
    "EXPOSURE_AMOUNT_LTR", "EXPOSURE_AMOUNT_MTR", "PENDING_ORDERS",
]
WANTED_ARRS_COLUMNS = [
    "ARRS_BTWN_0_30D", "ARRS_BTWN_31_60D", "ARRS_BTWN_61_90D",
    "ARRS_BTWN_91_180D", "ARRS_BTWN_181_270D",
    "ARRS_MORE_30D", "ARRS_MORE_60D", "ARRS_MORE_90D", "ARRS_MORE_180D", "ARRS_MORE_270D",
]

RATING_COLUMNS = [
    {"col": "GROUP_RATING", "label": "Group Rating", "numeric": False},
    {"col": "COUNTERPARTY_RATING", "label": "Counterparty Rating", "numeric": False},
    {"col": "CLS_GROUP_RATING", "label": "CLS Rating", "numeric": True},
]
RATING_COL_NUMERIC = {r["col"]: r["numeric"] for r in RATING_COLUMNS}

ARRS_BTWN_BUCKETS = [
    ("ARRS_BTWN_0_30D", "0-30 days"),
    ("ARRS_BTWN_31_60D", "31-60 days"),
    ("ARRS_BTWN_61_90D", "61-90 days"),
    ("ARRS_BTWN_91_180D", "91-180 days"),
    ("ARRS_BTWN_181_270D", "181-270 days"),
]

# The full column list of the CLS obligor extract, as given — matched
# case/whitespace-insensitively against whatever the real file actually has,
# so trailing spaces or minor casing differences in a real export don't
# silently break every lookup below.
CLS_EXPECTED_COLUMNS = [
    "Period Begin", "Period End", "Physical Person Flag", "Code Source System", "Code Country",
    "Partner (CLS)", "Sales Channel (CLS)", "Local Management Entity", "Flag Managing Entity",
    "Id Customer", "Id Client Number", "Obligor National ID", "National ID Type", "Main NAC",
    "Risk Profile", "Country Of Business (CLS)", "Country Of Incorporation (CLS)",
    "Segmentation Code", "Segmentation Description", "Segmentation Normalized Code",
    "Type Of Client (RMPM)", "BNPP Industry Code", "BNP Industry Code",
    "BNPP Industry Sector Description (CLS)", "Legal Status (CLS)", "Currency ISO (CLS)",
    "Currency System", "Limit Value Amount (LTR)", "Limit Value Expiration Date",
    "Limit Value Amount (MTR)", "Limit Available", "Exposure Amount (LTR)", "Exposure Date (LTR)",
    "Exposure Amount (MTR)", "Exposure Date (MTR)", "Pending Orders Amount",
    "Total Exposure (incl. Pending Ord) (LTR)", "Obligor Rating (CLS)", "Obligor Rating Date (CLS)",
    "Credit Watch Indicator (CLS)", "Credit Watch Indicator Date (CLS)", "Arval Fleet Size",
    "Event Of Default", "Date Event Of Default", "Real Payment Term", "Litigation Status",
    "Litigation Status Code", "Legal Situation Date", "Total Arrears",
    "Arrears Between 0 And 30 Days", "Arrears Between 31 And 60 Days",
    "Arrears Between 61 And 90 Days", "Arrears Between 91 And 180 Days",
    "Arrears Between 181 And 270 Days", "Arrears Above 30 Days", "Arrears Above 60 Days",
    "Arrears Above 90 Days", "Arrears Above 180 Days", "Arrears Above 270 Days",
]
CLS_ARREARS_BTWN = [
    "Arrears Between 0 And 30 Days", "Arrears Between 31 And 60 Days",
    "Arrears Between 61 And 90 Days", "Arrears Between 91 And 180 Days",
    "Arrears Between 181 And 270 Days",
]
CLS_DATE_COLUMNS = [
    "Period Begin", "Period End", "Limit Value Expiration Date", "Exposure Date (LTR)",
    "Exposure Date (MTR)", "Obligor Rating Date (CLS)", "Credit Watch Indicator Date (CLS)",
    "Date Event Of Default", "Legal Situation Date",
]
CLS_AMOUNT_COLUMNS = [
    "Limit Value Amount (LTR)", "Limit Value Amount (MTR)", "Limit Available",
    "Exposure Amount (LTR)", "Exposure Amount (MTR)", "Pending Orders Amount",
    "Total Exposure (incl. Pending Ord) (LTR)", "Total Arrears",
] + CLS_ARREARS_BTWN + ["Arrears Above 30 Days", "Arrears Above 60 Days", "Arrears Above 90 Days",
                         "Arrears Above 180 Days", "Arrears Above 270 Days"]

# =============================================================================
# NOVA loading — same filename convention as the other dashboards in this repo
# =============================================================================

def _detect_available_columns(folder_path: Path, wanted_cols: list[str]) -> list[str]:
    files = glob.glob(f"{folder_path}/*.parquet")
    if not files:
        return list(wanted_cols)
    try:
        import pyarrow.parquet as pq
        schema_cols = set(pq.read_schema(files[0]).names)
    except Exception:
        return list(wanted_cols)
    return [c for c in wanted_cols if c in schema_cols]


def load_country_monthly_data(folder_path, countries, start_yyyymm, end_yyyymm, cols=None) -> pd.DataFrame:
    files = glob.glob(f"{folder_path}/*.parquet")
    dfs: list[pd.DataFrame] = []
    start_int, end_int = int(start_yyyymm), int(end_yyyymm)
    countries_upper = {c.strip().upper() for c in countries}

    for fp in files:
        name = os.path.basename(fp).replace(".parquet", "")
        parts = [p.strip() for p in name.split("-")]
        if len(parts) < 3:
            continue
        file_country = parts[1].upper()
        try:
            file_yyyymm = int(parts[2])
        except ValueError:
            continue
        if file_country in countries_upper and start_int <= file_yyyymm <= end_int:
            dfs.append(pd.read_parquet(fp, columns=cols))

    if not dfs:
        return pd.DataFrame(columns=cols or NOVA_COLUMNS)
    return pd.concat(dfs, ignore_index=True)


def prepare_nova(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ("COB_DATE", "CONTRACT_START_DATE"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("COUNTRY",):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


_wanted_nova_cols = NOVA_COLUMNS + WANTED_ARRS_COLUMNS
_nova_cols_to_read = _detect_available_columns(DATA_FOLDER, _wanted_nova_cols)
ARRS_COLUMNS_PRESENT = [c for c in WANTED_ARRS_COLUMNS if c in _nova_cols_to_read]

NOVA_DF = load_country_monthly_data(DATA_FOLDER, COUNTRIES_TO_READ, START_YYYYMM, END_YYYYMM,
                                     cols=_nova_cols_to_read)
NOVA_DF = prepare_nova(NOVA_DF)


# =============================================================================
# CLS loading — optional second dataset (obligor-level risk extract)
# =============================================================================

def _load_cls_data() -> pd.DataFrame:
    if not os.path.exists(CREDIT_VERIFICATION_DATA_PATH):
        return pd.DataFrame()
    try:
        return pd.read_parquet(CREDIT_VERIFICATION_DATA_PATH)
    except Exception:
        return pd.DataFrame()


def _resolve_columns(df: pd.DataFrame, expected: list[str]) -> dict:
    """Maps each expected CLS column label to the actual column name found in
    df, case/whitespace-insensitively — so a real export's trailing spaces or
    casing quirks don't silently break every lookup that follows."""
    norm = {str(c).strip().lower(): c for c in df.columns}
    return {label: norm.get(label.strip().lower()) for label in expected}


def prepare_cls(df: pd.DataFrame, col: dict) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for label in CLS_DATE_COLUMNS:
        c = col.get(label)
        if c and c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    for label in CLS_AMOUNT_COLUMNS:
        c = col.get(label)
        if c and c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    c = col.get("Code Country")
    if c and c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    c = col.get("Id Customer")
    if c and c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    return d


def _as_bool_flag(series: pd.Series) -> pd.Series:
    """Normalizes a Y/N, 1/0, True/False-style flag column (real-world export
    encoding is unknown ahead of time) into a clean nullable boolean."""
    if series is None:
        return series
    mapped = series.astype(str).str.strip().str.upper().map({
        "Y": True, "YES": True, "TRUE": True, "1": True, "1.0": True,
        "N": False, "NO": False, "FALSE": False, "0": False, "0.0": False,
    })
    return mapped


CLS_RAW = _load_cls_data()
CLS_COL = _resolve_columns(CLS_RAW, CLS_EXPECTED_COLUMNS) if not CLS_RAW.empty else {}
CLS_DF = prepare_cls(CLS_RAW, CLS_COL)
CLS_AVAILABLE = not CLS_DF.empty
CLS_MISSING_MSG = (
    f"Credit verification data not found at {os.path.basename(CREDIT_VERIFICATION_DATA_PATH)}. "
    "Drop your export there (next to this script) to enable data-quality checks, "
    "cross-dataset comparison, and arrears-onset analysis on it. "
    "NOVA-only sections below still work."
)


# =============================================================================
# Rating formatting (same convention as client_rating_at_origination.py)
# =============================================================================

def _fmt_rating(col: str, v) -> str | None:
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


RATING_ORDER_HINT = {  # coarse "good..bad" ordering for the "good rating but arrears" check
    "01": 1, "02-": 2, "02": 3, "02+": 4, "03-": 5, "03": 6, "03+": 7,
}


def _rating_is_top_band(col: str, disp: str | None) -> bool | None:
    if disp is None:
        return None
    if RATING_COL_NUMERIC.get(col):
        try:
            return float(disp) <= 3
        except (TypeError, ValueError):
            return None
    return RATING_ORDER_HINT.get(disp, 99) <= 3


# =============================================================================
# NOVA customer-level state — latest known rating/exposure/arrears per
# customer, plus the validated Total Exposure formula (dedup per contract,
# summed) from client_rating_at_origination.py / customer_vehicle_explorer.py
# =============================================================================

def _build_nova_customer_state(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ID_CUSTOMER" not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=["ID_CUSTOMER"]).copy()
    d["ID_CUSTOMER"] = d["ID_CUSTOMER"].astype(str).str.strip()

    rating_cols = [r["col"] for r in RATING_COLUMNS if r["col"] in d.columns]
    profile_cols = [c for c in (["COUNTRY"] + rating_cols + ARRS_COLUMNS_PRESENT) if c in d.columns]
    ds = d.sort_values("COB_DATE") if "COB_DATE" in d.columns else d
    if profile_cols:
        ds = ds.copy()
        ds[profile_cols] = ds.groupby("ID_CUSTOMER")[profile_cols].ffill()
    latest = ds.drop_duplicates(subset=["ID_CUSTOMER"], keep="last")

    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    exp = d.copy()
    ltr = exp["EXPOSURE_AMOUNT_LTR"] if "EXPOSURE_AMOUNT_LTR" in exp.columns else 0.0
    pending = exp["PENDING_ORDERS"] if "PENDING_ORDERS" in exp.columns else 0.0
    exp["_EXPOSURE"] = (ltr.fillna(0) if hasattr(ltr, "fillna") else ltr) + \
                        (pending.fillna(0) if hasattr(pending, "fillna") else pending)
    if "COB_DATE" in exp.columns:
        exp = exp.sort_values("COB_DATE")
    if keys:
        exp = exp.drop_duplicates(subset=keys + ["ID_CUSTOMER"], keep="last")
    total_exposure = exp.groupby("ID_CUSTOMER")["_EXPOSURE"].sum().rename("TOTAL_EXPOSURE")

    out = latest.merge(total_exposure, on="ID_CUSTOMER", how="left")
    for col in rating_cols:
        out[f"{col}_DISP"] = out[col].apply(lambda v, c=col: _fmt_rating(c, v))

    if ARRS_COLUMNS_PRESENT:
        present = [(c, l) for c, l in ARRS_BTWN_BUCKETS if c in ARRS_COLUMNS_PRESENT]
        out["TOTAL_ARREARS"] = sum(out[c].fillna(0) for c, _ in present) + \
            (out["ARRS_MORE_270D"].fillna(0) if "ARRS_MORE_270D" in ARRS_COLUMNS_PRESENT else 0.0)
    else:
        out["TOTAL_ARREARS"] = np.nan

    n_vehicles = (d.drop_duplicates(subset=keys + ["ID_CUSTOMER"]) if keys else d) \
        .groupby("ID_CUSTOMER").size().rename("N_VEHICLES")
    out = out.merge(n_vehicles, on="ID_CUSTOMER", how="left")
    out["N_VEHICLES"] = out["N_VEHICLES"].fillna(0).astype(int)
    return out


NOVA_CUSTOMER_STATE = _build_nova_customer_state(NOVA_DF)


# =============================================================================
# CLS customer-level state — latest period per customer
# =============================================================================

def _build_cls_customer_state(df: pd.DataFrame, col: dict) -> pd.DataFrame:
    id_col, date_col = col.get("Id Customer"), col.get("Period End")
    if df.empty or not id_col or id_col not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[id_col]).copy()
    if date_col and date_col in d.columns:
        d = d.sort_values(date_col)
    return d.drop_duplicates(subset=[id_col], keep="last")


CLS_CUSTOMER_STATE = _build_cls_customer_state(CLS_DF, CLS_COL)


# =============================================================================
# Check framework — every check returns a small dict; rendered as a colored
# checklist (good / warning / critical / info) in the UI.
# =============================================================================

def _check(check_id, title, status, detail, metric=None):
    return {"id": check_id, "title": title, "status": status, "detail": detail, "metric": metric}


def _rate(n_bad, n_total, warn_pct=0.01, crit_pct=0.05):
    if n_total == 0:
        return "info"
    pct = n_bad / n_total
    if pct > crit_pct:
        return "critical"
    if pct > warn_pct:
        return "warning"
    return "good"


def run_nova_quality_checks(df: pd.DataFrame, state: pd.DataFrame) -> list[dict]:
    """Credit/arrears-focused checks on NOVA — same spirit as the CLS checks
    below, not generic structural QA (that lives in the source pipeline, not
    a credit-risk control center)."""
    if df.empty:
        return [_check("nova-empty", "NOVA data loaded", "critical", "No NOVA rows loaded — check DATA_FOLDER / COUNTRIES_TO_READ.")]
    checks = []
    n = len(df)
    n_state = len(state) if not state.empty else 0

    if "EXPOSURE_AMOUNT_LTR" in df.columns:
        n_neg = int((df["EXPOSURE_AMOUNT_LTR"] < 0).sum())
        checks.append(_check("nova-neg-exp", "Negative EXPOSURE_AMOUNT_LTR",
                              _rate(n_neg, n), f"{n_neg:,} rows have a negative LTR exposure.", f"{n_neg:,}"))

    if n_state and "CLS_GROUP_RATING_DISP" in state.columns:
        n_nr = int(state["CLS_GROUP_RATING_DISP"].isna().sum())
        checks.append(_check("nova-nr-rating", "Customers with a known CLS rating",
                              _rate(n_nr, n_state, warn_pct=0.10, crit_pct=0.30),
                              f"{n_nr:,} / {n_state:,} customers have no CLS_GROUP_RATING (shown as NR).",
                              f"{100 * (1 - n_nr / n_state):.1f}% rated"))

    if not ARRS_COLUMNS_PRESENT:
        checks.append(_check("nova-arrs-missing", "Arrears columns present in NOVA", "info",
                              "None of the ARRS_* columns exist in this NOVA export, so arrears-based credit "
                              "checks on NOVA are skipped here — see the CLS checks below, and the Adjusted "
                              "Exposure tab (which falls back to CLS arrears matched by NOVA's own COB_DATE)."))
        return checks

    present = [c for c, _ in ARRS_BTWN_BUCKETS if c in ARRS_COLUMNS_PRESENT]
    neg_mask = pd.Series(False, index=df.index)
    for c in present:
        neg_mask = neg_mask | (df[c] < 0)
    n_neg_arr = int(neg_mask.sum())
    checks.append(_check("nova-neg-arrears", "Negative arrears bucket values",
                          _rate(n_neg_arr, n), f"{n_neg_arr:,} rows have a negative value in an arrears bucket.",
                          f"{n_neg_arr:,}"))

    if n_state:
        top_band = state["CLS_GROUP_RATING_DISP"].apply(lambda v: _rating_is_top_band("CLS_GROUP_RATING", v))
        n_bad = int(((top_band == True) & (state["TOTAL_ARREARS"].fillna(0) > 0)).sum())  # noqa: E712
        checks.append(_check("nova-good-rating-arrears", "Top-band rating but has arrears",
                              _rate(n_bad, n_state, warn_pct=0.005, crit_pct=0.02),
                              f"{n_bad:,} customers carry a top-3 CLS rating band yet have Total Arrears > 0 — "
                              f"rating may not reflect real payment behavior.", f"{n_bad:,}"))

        n_odd = int(((state["TOTAL_ARREARS"].fillna(0) > 0) & (state["TOTAL_EXPOSURE"].fillna(0) <= 0)).sum())
        checks.append(_check("nova-arrears-no-exposure", "Arrears reported with zero exposure",
                              _rate(n_odd, n_state, warn_pct=0.005, crit_pct=0.02),
                              f"{n_odd:,} customers show arrears but zero total exposure — financially odd "
                              f"(money owed with nothing currently at risk).", f"{n_odd:,}"))

    return checks


def run_cls_quality_checks(df: pd.DataFrame, col: dict) -> list[dict]:
    if not CLS_AVAILABLE:
        return [_check("cls-missing", "CLS data loaded", "info", CLS_MISSING_MSG)]
    checks = []
    n = len(df)

    total_c, btwn_cs, tail_c = col.get("Total Arrears"), [col.get(l) for l in CLS_ARREARS_BTWN], col.get("Arrears Above 270 Days")
    if total_c and all(btwn_cs) and tail_c and all(c in df.columns for c in [total_c, tail_c] + btwn_cs):
        computed = sum(df[c].fillna(0) for c in btwn_cs) + df[tail_c].fillna(0)
        diff = (df[total_c].fillna(0) - computed).abs()
        n_mismatch = int((diff > 1).sum())
        checks.append(_check("cls-arrears-formula", "'Total Arrears' = sum of buckets",
                              _rate(n_mismatch, n),
                              f"{n_mismatch:,} / {n:,} rows where Total Arrears differs from the bucket "
                              f"sum by more than 1 unit — validates the formula used everywhere else in "
                              f"this project.", f"{100 * (1 - n_mismatch / n):.1f}% match"))

    exp_ltr_c, pending_c, total_exp_c = col.get("Exposure Amount (LTR)"), col.get("Pending Orders Amount"), \
        col.get("Total Exposure (incl. Pending Ord) (LTR)")
    if exp_ltr_c and pending_c and total_exp_c and all(c in df.columns for c in [exp_ltr_c, pending_c, total_exp_c]):
        computed_exp = df[exp_ltr_c].fillna(0) + df[pending_c].fillna(0)
        diff = (df[total_exp_c].fillna(0) - computed_exp).abs()
        n_mismatch = int((diff > 1).sum())
        checks.append(_check("cls-exposure-formula", "'Total Exposure' = LTR + Pending Orders",
                              _rate(n_mismatch, n),
                              f"{n_mismatch:,} / {n:,} rows differ by more than 1 unit.",
                              f"{100 * (1 - n_mismatch / n):.1f}% match"))

    pb_c, pe_c = col.get("Period Begin"), col.get("Period End")
    if pb_c and pe_c and pb_c in df.columns and pe_c in df.columns:
        both = df.dropna(subset=[pb_c, pe_c])
        n_bad = int((both[pb_c] > both[pe_c]).sum())
        checks.append(_check("cls-period-order", "Period Begin ≤ Period End",
                              _rate(n_bad, n), f"{n_bad:,} rows have Period Begin after Period End.", f"{n_bad:,}"))

    id_c = col.get("Id Customer")
    if id_c and pe_c and id_c in df.columns and pe_c in df.columns:
        n_dupe = int(df.duplicated(subset=[id_c, pe_c]).sum())
        checks.append(_check("cls-dupe-period", "Duplicate (customer, period) rows",
                              _rate(n_dupe, n), f"{n_dupe:,} duplicate (Id Customer, Period End) rows.", f"{n_dupe:,}"))

    eod_c, eod_date_c = col.get("Event Of Default"), col.get("Date Event Of Default")
    if eod_c and eod_date_c and eod_c in df.columns and eod_date_c in df.columns:
        flag = _as_bool_flag(df[eod_c])
        n_bad = int(((flag == True) & df[eod_date_c].isna()).sum())  # noqa: E712
        checks.append(_check("cls-eod-consistency", "Event Of Default has a default date",
                              _rate(n_bad, n), f"{n_bad:,} rows flagged in default with no Date Event Of Default.",
                              f"{n_bad:,}"))

    exp_c, limit_c = col.get("Exposure Amount (LTR)"), col.get("Limit Value Amount (LTR)")
    if exp_c and limit_c and exp_c in df.columns and limit_c in df.columns:
        both = df.dropna(subset=[exp_c, limit_c])
        n_over = int((both[exp_c] > both[limit_c]).sum())
        checks.append(_check("cls-over-limit", "Customers within their credit limit",
                              _rate(n_over, n, warn_pct=0.02, crit_pct=0.08),
                              f"{n_over:,} rows have Exposure Amount (LTR) above Limit Value Amount (LTR).",
                              f"{n_over:,} over-limit"))

    rating_c, rating_date_c, pe_c2 = col.get("Obligor Rating (CLS)"), col.get("Obligor Rating Date (CLS)"), col.get("Period End")
    if rating_date_c and pe_c2 and rating_date_c in df.columns and pe_c2 in df.columns:
        both = df.dropna(subset=[rating_date_c, pe_c2])
        stale_days = (both[pe_c2] - both[rating_date_c]).dt.days
        n_stale = int((stale_days > 365).sum())
        checks.append(_check("cls-stale-rating", "Rating refreshed within the last 12 months",
                              _rate(n_stale, n, warn_pct=0.05, crit_pct=0.15),
                              f"{n_stale:,} rows have a rating older than 365 days vs. their Period End.",
                              f"{n_stale:,} stale"))

    # "Invented" cross-checks — internally inconsistent risk signals worth a look.
    if total_c and rating_c and total_c in df.columns and rating_c in df.columns:
        rat_disp = df[rating_c].apply(lambda v: _fmt_rating("CLS_GROUP_RATING", v))
        top_band = rat_disp.apply(lambda v: _rating_is_top_band("CLS_GROUP_RATING", v))
        n_bad = int(((top_band == True) & (df[total_c].fillna(0) > 0)).sum())  # noqa: E712
        checks.append(_check("cls-good-rating-arrears", "Top-band rating but has arrears",
                              _rate(n_bad, n, warn_pct=0.005, crit_pct=0.02),
                              f"{n_bad:,} rows carry a top-3 rating band yet report Total Arrears > 0 — "
                              f"rating may not reflect real payment behavior.", f"{n_bad:,}"))

    watch_c = col.get("Credit Watch Indicator (CLS)")
    if watch_c and rating_c and watch_c in df.columns and rating_c in df.columns:
        watch_flag = _as_bool_flag(df[watch_c])
        rat_disp = df[rating_c].apply(lambda v: _fmt_rating("CLS_GROUP_RATING", v))
        top_band = rat_disp.apply(lambda v: _rating_is_top_band("CLS_GROUP_RATING", v))
        n_bad = int(((watch_flag == True) & (top_band == True)).sum())  # noqa: E712
        checks.append(_check("cls-watch-vs-rating", "Credit Watch flagged with a top-band rating",
                              _rate(n_bad, n, warn_pct=0.005, crit_pct=0.02),
                              f"{n_bad:,} rows are on Credit Watch despite a top-3 rating band.", f"{n_bad:,}"))

    return checks


# =============================================================================
# Cross-dataset comparison — matched customers, NOVA-computed vs CLS-reported
# =============================================================================

def build_comparison() -> pd.DataFrame:
    if not CLS_AVAILABLE or NOVA_CUSTOMER_STATE.empty:
        return pd.DataFrame()
    id_c, rating_c, total_exp_c, total_arr_c = (
        CLS_COL.get("Id Customer"), CLS_COL.get("Obligor Rating (CLS)"),
        CLS_COL.get("Total Exposure (incl. Pending Ord) (LTR)"), CLS_COL.get("Total Arrears"),
    )
    if not id_c:
        return pd.DataFrame()

    right = CLS_CUSTOMER_STATE.copy()
    right["_ID"] = right[id_c].astype(str).str.strip()
    left = NOVA_CUSTOMER_STATE.copy()
    left["_ID"] = left["ID_CUSTOMER"]

    merged = left.merge(right, on="_ID", how="inner", suffixes=("_NOVA", "_CLS"))
    if merged.empty:
        return merged

    # NaN-aware on purpose: a customer missing a value on one side is a
    # "no data to compare" case, not a fake mismatch against an assumed 0 —
    # filling with 0 here used to make missing-CLS-exposure customers look
    # like the biggest "mismatches" when they're really just unrated/unreported.
    if rating_c and rating_c in merged.columns:
        merged["CLS_RATING_DISP"] = merged[rating_c].apply(lambda v: _fmt_rating("CLS_GROUP_RATING", v))
        both_rated = merged["CLS_GROUP_RATING_DISP"].notna() & merged["CLS_RATING_DISP"].notna()
        merged["RATING_MATCH"] = np.where(both_rated, merged["CLS_GROUP_RATING_DISP"] == merged["CLS_RATING_DISP"], np.nan)
    if total_exp_c and total_exp_c in merged.columns:
        merged["EXPOSURE_DIFF"] = merged["TOTAL_EXPOSURE"] - merged[total_exp_c]
        merged["BOTH_HAVE_EXPOSURE"] = merged["TOTAL_EXPOSURE"].notna() & merged[total_exp_c].notna()
    if total_arr_c and total_arr_c in merged.columns:
        merged["ARREARS_DIFF"] = merged["TOTAL_ARREARS"] - merged[total_arr_c]
        merged["BOTH_HAVE_ARREARS"] = merged["TOTAL_ARREARS"].notna() & merged[total_arr_c].notna()
    return merged


COMPARISON_DF = build_comparison()


# =============================================================================
# Arrears onset — first period a customer shows Total Arrears > 0, vs. their
# eventual Event Of Default date (if any). Answers "how much warning did the
# arrears give before the client actually defaulted?"
# =============================================================================

def build_arrears_onset() -> pd.DataFrame:
    if not CLS_AVAILABLE:
        return pd.DataFrame()
    id_c, pe_c, total_c, eod_c, eod_date_c = (
        CLS_COL.get("Id Customer"), CLS_COL.get("Period End"), CLS_COL.get("Total Arrears"),
        CLS_COL.get("Event Of Default"), CLS_COL.get("Date Event Of Default"),
    )
    if not (id_c and pe_c and total_c):
        return pd.DataFrame()

    d = CLS_DF.dropna(subset=[id_c, pe_c]).copy()
    with_arrears = d[d[total_c].fillna(0) > 0]
    onset = with_arrears.sort_values(pe_c).drop_duplicates(subset=[id_c], keep="first")
    onset = onset[[id_c, pe_c]].rename(columns={id_c: "ID_CUSTOMER", pe_c: "ONSET_DATE"})

    ever_default = pd.DataFrame()
    if eod_c and eod_c in d.columns:
        flag = _as_bool_flag(d[eod_c])
        defaulted = d[flag == True]  # noqa: E712
        if eod_date_c and eod_date_c in defaulted.columns:
            ever_default = defaulted.dropna(subset=[eod_date_c]).sort_values(eod_date_c) \
                .drop_duplicates(subset=[id_c], keep="first")[[id_c, eod_date_c]] \
                .rename(columns={id_c: "ID_CUSTOMER", eod_date_c: "DEFAULT_DATE"})

    out = onset.merge(ever_default, on="ID_CUSTOMER", how="left") if not ever_default.empty else onset.assign(DEFAULT_DATE=pd.NaT)
    out["DAYS_TO_DEFAULT"] = (out["DEFAULT_DATE"] - out["ONSET_DATE"]).dt.days
    return out


ARREARS_ONSET_DF = build_arrears_onset()


# =============================================================================
# Adjusted exposure — the way larger financial institutions size real credit
# exposure folds unpaid arrears into it, not just the notional/contracted
# amount: money that's overdue but not collected is still money at risk, on
# top of what's already outstanding. Adjusted Exposure = Total Exposure
# (LTR + Pending, dedup per contract) + arrears.
#
# The arrears figure always tracks NOVA's own COB_DATE (never CLS's Period
# End) per the user's requirement: if NOVA carries its own ARRS_* columns,
# those are used directly; otherwise CLS's Total Arrears is looked up "as of"
# — the latest CLS Period End at or before that customer's own NOVA COB_DATE
# — via merge_asof, the same backward-looking join used for rating-at-date
# elsewhere in this project, just vectorized here for the whole portfolio.
# =============================================================================

ADJUSTED_EXPOSURE_ARREARS_SOURCE = "unavailable"


def build_adjusted_exposure() -> pd.DataFrame:
    global ADJUSTED_EXPOSURE_ARREARS_SOURCE
    if NOVA_CUSTOMER_STATE.empty:
        return pd.DataFrame()

    d = NOVA_CUSTOMER_STATE.dropna(subset=["TOTAL_EXPOSURE"]).copy()

    if ARRS_COLUMNS_PRESENT:
        d["ARREARS_USED"] = d["TOTAL_ARREARS"].fillna(0)
        ADJUSTED_EXPOSURE_ARREARS_SOURCE = "NOVA's own arrears columns"
    elif CLS_AVAILABLE and CLS_COL.get("Id Customer") and CLS_COL.get("Period End") and CLS_COL.get("Total Arrears"):
        id_c, pe_c, total_c = CLS_COL["Id Customer"], CLS_COL["Period End"], CLS_COL["Total Arrears"]
        left = d[["ID_CUSTOMER", "COB_DATE"]].dropna(subset=["COB_DATE"]).sort_values("COB_DATE")
        right = (CLS_DF[[id_c, pe_c, total_c]].dropna(subset=[pe_c])
                 .rename(columns={id_c: "ID_CUSTOMER", pe_c: "COB_DATE", total_c: "ARREARS_USED"})
                 .sort_values("COB_DATE"))
        # merge_asof requires an exact datetime64 dtype match on the join key —
        # NOVA's COB_DATE and CLS's Period End can come out at different
        # (us/ns) resolutions after independent pd.to_datetime() parsing.
        left["COB_DATE"] = left["COB_DATE"].astype("datetime64[ns]")
        right["COB_DATE"] = right["COB_DATE"].astype("datetime64[ns]")
        matched = pd.merge_asof(left, right, on="COB_DATE", by="ID_CUSTOMER", direction="backward")
        d = d.merge(matched[["ID_CUSTOMER", "ARREARS_USED"]], on="ID_CUSTOMER", how="left")
        d["ARREARS_USED"] = d["ARREARS_USED"].fillna(0)
        ADJUSTED_EXPOSURE_ARREARS_SOURCE = "CLS Total Arrears, as of each customer's own latest NOVA COB_DATE"
    else:
        d["ARREARS_USED"] = 0.0
        ADJUSTED_EXPOSURE_ARREARS_SOURCE = "unavailable — no arrears source on either side"

    d["ADJUSTED_EXPOSURE"] = d["TOTAL_EXPOSURE"] + d["ARREARS_USED"]
    # NaN (not 0%) when raw exposure is 0 but arrears exist — that's a jump
    # from nothing to something at risk, not "no change", and must sort/read
    # as the most dramatic case rather than disappear at the bottom.
    d["UPLIFT_PCT"] = np.where(d["TOTAL_EXPOSURE"] > 0, 100 * d["ARREARS_USED"] / d["TOTAL_EXPOSURE"], np.nan)
    return d


ADJUSTED_EXPOSURE_DF = build_adjusted_exposure()


# =============================================================================
# Dash app
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Credit Risk Control Center"

PAGE_STYLE = {"fontFamily": "Inter, -apple-system, sans-serif", "minHeight": "100vh", "background": "#f7fafc"}
CARD_STYLE = {"background": "#ffffff", "borderRadius": "10px", "border": "1px solid #e2e8f0", "padding": "18px"}
PANEL_TITLE_STYLE = {"fontWeight": "700", "fontSize": "14px", "color": "#1a1a2e", "marginBottom": "12px"}
TABLE_STYLE = {"width": "100%", "borderCollapse": "collapse", "fontSize": "12.5px"}
TH_STYLE = {"textAlign": "left", "padding": "8px 10px", "borderBottom": "2px solid #e2e8f0",
            "color": "#718096", "fontSize": "11px", "fontWeight": "700", "textTransform": "uppercase"}
TD_STYLE = {"padding": "8px 10px", "borderBottom": "1px solid #f0f2f5", "color": "#1a1a2e"}

STATUS_COLOR = {"good": "#2f855a", "warning": "#d98943", "critical": "#9b2c2c", "info": "#3182ce"}
STATUS_BG = {"good": "#f0fff4", "warning": "#fffaf0", "critical": "#fff5f5", "info": "#ebf8ff"}
STATUS_ICON = {"good": "✓", "warning": "!", "critical": "✕", "info": "i"}

DETAILS_BTN_STYLE = {
    "padding": "5px 12px", "fontWeight": "700", "fontSize": "11px",
    "background": "#ffffff", "color": "#3182ce", "border": "1px solid #bee3f8",
    "borderRadius": "6px", "cursor": "pointer",
}
SECONDARY_BTN_STYLE = {
    "padding": "9px 20px", "fontWeight": "600", "fontSize": "13px",
    "background": "#f7fafc", "color": "#718096", "border": "1px solid #cbd5e0",
    "borderRadius": "6px", "cursor": "pointer",
}
MODAL_OVERLAY_STYLE = {
    "display": "none", "position": "fixed", "top": "0", "left": "0",
    "width": "100%", "height": "100%", "backgroundColor": "rgba(15, 23, 42, 0.55)",
    "zIndex": "9999", "justifyContent": "center", "alignItems": "center",
}
MODAL_PANEL_STYLE = {
    "background": "#ffffff", "borderRadius": "12px", "width": "900px",
    "maxWidth": "94vw", "maxHeight": "90vh", "overflowY": "auto",
    "padding": "30px", "boxShadow": "0 20px 60px rgba(0,0,0,0.25)",
}


def _panel_title(text: str):
    return html.Div(text, style=PANEL_TITLE_STYLE)


def _table(headers: list[str], rows: list[list]):
    return html.Table(
        [html.Thead(html.Tr([html.Th(h, style=TH_STYLE) for h in headers]))] +
        [html.Tbody([html.Tr([html.Td(c, style=TD_STYLE) for c in r]) for r in rows])],
        style=TABLE_STYLE,
    )


def _stat_tile(label: str, value: str, sub: str | None = None, accent: str = "#3182ce"):
    return html.Div(
        [
            html.Div(label, style={"fontSize": "11px", "color": "#718096", "fontWeight": "700",
                                    "textTransform": "uppercase", "letterSpacing": "0.04em", "marginBottom": "8px"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "800", "color": accent, "lineHeight": "1.1"}),
            html.Div(sub or "", style={"fontSize": "12px", "color": "#a0aec0", "marginTop": "6px"}),
        ],
        style={**CARD_STYLE, "padding": "16px 18px"},
    )


def _check_row(check: dict):
    color = STATUS_COLOR[check["status"]]
    bg = STATUS_BG[check["status"]]
    return html.Div(
        [
            html.Div(STATUS_ICON[check["status"]], style={
                "width": "22px", "height": "22px", "borderRadius": "50%", "background": color,
                "color": "#fff", "fontSize": "12px", "fontWeight": "800", "display": "flex",
                "alignItems": "center", "justifyContent": "center", "flexShrink": "0",
            }),
            html.Div([
                html.Div(check["title"], style={"fontWeight": "700", "fontSize": "13px", "color": "#1a1a2e"}),
                html.Div(check["detail"], style={"fontSize": "12px", "color": "#718096", "marginTop": "2px"}),
            ], style={"flex": "1"}),
            html.Div(check["metric"] or "", style={"fontWeight": "700", "fontSize": "13px", "color": color,
                                                     "whiteSpace": "nowrap", "marginLeft": "10px"}),
        ],
        style={"display": "flex", "alignItems": "flex-start", "gap": "12px", "padding": "12px 14px",
               "background": bg, "borderRadius": "8px", "marginBottom": "8px", "border": f"1px solid {color}22"},
    )


def _checklist_card(title: str, checks: list[dict]):
    return html.Div([_panel_title(title), html.Div([_check_row(c) for c in checks])],
                     style={**CARD_STYLE, "marginBottom": "16px"})


def _missing_cls_banner():
    return html.Div(CLS_MISSING_MSG, style={
        "background": "#fffaf0", "border": "1px solid #d98943", "color": "#8a5a1f",
        "borderRadius": "8px", "padding": "12px 16px", "fontSize": "13px", "marginBottom": "16px",
    })


# ── Tab renderers ────────────────────────────────────────────────────────────

def render_overview():
    n_nova_cust = NOVA_CUSTOMER_STATE["ID_CUSTOMER"].nunique() if not NOVA_CUSTOMER_STATE.empty else 0
    nova_total_exp = NOVA_CUSTOMER_STATE["TOTAL_EXPOSURE"].sum() if not NOVA_CUSTOMER_STATE.empty else 0
    n_cls_cust = CLS_CUSTOMER_STATE[CLS_COL.get("Id Customer")].nunique() if CLS_AVAILABLE and CLS_COL.get("Id Customer") else 0
    cls_total_exp = (CLS_CUSTOMER_STATE[CLS_COL["Total Exposure (incl. Pending Ord) (LTR)"]].sum()
                      if CLS_AVAILABLE and CLS_COL.get("Total Exposure (incl. Pending Ord) (LTR)") else None)
    n_matched = COMPARISON_DF["_ID"].nunique() if not COMPARISON_DF.empty else 0
    n_default = 0
    if CLS_AVAILABLE and CLS_COL.get("Event Of Default"):
        n_default = int((_as_bool_flag(CLS_CUSTOMER_STATE[CLS_COL["Event Of Default"]]) == True).sum())  # noqa: E712

    tiles = [
        _stat_tile("NOVA customers", f"{n_nova_cust:,}", f"{COUNTRIES_TO_READ}"),
        _stat_tile("NOVA total exposure", f"€{nova_total_exp / 1_000_000:,.2f}M", "LTR + pending, dedup per contract"),
        _stat_tile("CLS customers", f"{n_cls_cust:,}" if CLS_AVAILABLE else "—",
                    "latest period per customer" if CLS_AVAILABLE else "file not found"),
        _stat_tile("CLS total exposure",
                    f"€{cls_total_exp / 1_000_000:,.2f}M" if cls_total_exp is not None else "—",
                    "as reported by CLS" if CLS_AVAILABLE else "file not found"),
        _stat_tile("Matched customers", f"{n_matched:,}" if CLS_AVAILABLE else "—",
                    "present in both NOVA and CLS" if CLS_AVAILABLE else "file not found"),
        _stat_tile("Customers in default", f"{n_default:,}" if CLS_AVAILABLE else "—",
                    "Event Of Default flag" if CLS_AVAILABLE else "file not found",
                    accent="#9b2c2c" if n_default else "#3182ce"),
    ]

    children = [] if CLS_AVAILABLE else [_missing_cls_banner()]
    children.append(html.Div(tiles, style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                                            "gap": "14px"}))
    return html.Div(children)


def render_data_quality():
    nova_checks = run_nova_quality_checks(NOVA_DF, NOVA_CUSTOMER_STATE)
    cls_checks = run_cls_quality_checks(CLS_DF, CLS_COL)
    return html.Div([
        _checklist_card("NOVA — data quality", nova_checks),
        _checklist_card("CLS — data quality", cls_checks),
    ])


def render_comparison():
    if not CLS_AVAILABLE:
        return _missing_cls_banner()
    if COMPARISON_DF.empty:
        return html.P("No matching customers found between NOVA and CLS.", style={"color": "#a0aec0"})

    n = len(COMPARISON_DF)
    # .mean() over an all-NaN column (e.g. NOVA has no arrears data at all
    # locally) returns float NaN, not None — `is not None` doesn't catch that,
    # so the display check below must be pd.notna(), not an identity check.
    rating_match_pct = 100 * COMPARISON_DF["RATING_MATCH"].mean() if "RATING_MATCH" in COMPARISON_DF else None
    exp_mae = COMPARISON_DF["EXPOSURE_DIFF"].abs().mean() if "EXPOSURE_DIFF" in COMPARISON_DF else None
    arr_mae = COMPARISON_DF["ARREARS_DIFF"].abs().mean() if "ARREARS_DIFF" in COMPARISON_DF else None

    tiles = [
        _stat_tile("Matched customers", f"{n:,}"),
        _stat_tile("Rating match rate", f"{rating_match_pct:.1f}%" if pd.notna(rating_match_pct) else "—",
                    "NOVA CLS_GROUP_RATING vs. CLS Obligor Rating"),
        _stat_tile("Exposure — mean abs diff", f"€{exp_mae:,.0f}" if pd.notna(exp_mae) else "—",
                    "NOVA computed vs. CLS Total Exposure"),
        _stat_tile("Arrears — mean abs diff", f"€{arr_mae:,.0f}" if pd.notna(arr_mae) else "—",
                    "NOVA computed vs. CLS Total Arrears" if ARRS_COLUMNS_PRESENT else
                    "NOVA has no arrears data locally to compare"),
    ]

    fig = None
    if "EXPOSURE_DIFF" in COMPARISON_DF.columns and CLS_COL.get("Total Exposure (incl. Pending Ord) (LTR)"):
        cls_exp_col = CLS_COL["Total Exposure (incl. Pending Ord) (LTR)"]
        fig = go.Figure(go.Scatter(
            x=COMPARISON_DF[cls_exp_col], y=COMPARISON_DF["TOTAL_EXPOSURE"],
            mode="markers", marker=dict(size=7, color="#3182ce", opacity=0.6),
            hovertemplate="CLS: €%{x:,.0f}<br>NOVA: €%{y:,.0f}<extra></extra>",
        ))
        max_v = float(max(COMPARISON_DF[cls_exp_col].max(), COMPARISON_DF["TOTAL_EXPOSURE"].max()) or 1)
        fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                                  line=dict(color="#a0aec0", width=1, dash="dash"), hoverinfo="skip"))
        fig.update_layout(
            height=340, margin=dict(l=50, r=20, t=20, b=40), showlegend=False,
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#4a5568"),
            xaxis=dict(title="CLS Total Exposure (LTR)", gridcolor="#f0f2f5"),
            yaxis=dict(title="NOVA computed Total Exposure", gridcolor="#f0f2f5"),
        )

    mismatch_table = html.P("No exposure column found on the CLS side.", style={"color": "#a0aec0"})
    excluded_note = None
    if "EXPOSURE_DIFF" in COMPARISON_DF.columns and "BOTH_HAVE_EXPOSURE" in COMPARISON_DF.columns:
        cls_exp_col = CLS_COL.get("Total Exposure (incl. Pending Ord) (LTR)")
        # Only rank pairs where BOTH sides actually have a value — a customer
        # missing CLS exposure isn't a "€X mismatch" against an assumed 0,
        # it's missing data, and used to swamp this table with false positives.
        comparable = COMPARISON_DF[COMPARISON_DF["BOTH_HAVE_EXPOSURE"]]
        n_excluded = len(COMPARISON_DF) - len(comparable)
        top = comparable.reindex(comparable["EXPOSURE_DIFF"].abs().sort_values(ascending=False).index).head(15)
        rows = [[
            r["_ID"], r.get("CLS_GROUP_RATING_DISP") or "NR", r.get("CLS_RATING_DISP") or "NR",
            f"€{r['TOTAL_EXPOSURE']:,.0f}", f"€{r[cls_exp_col]:,.0f}" if cls_exp_col else "—",
            f"€{r['EXPOSURE_DIFF']:,.0f}",
            html.Button("Details", id={"type": "cmp-details", "index": str(r["_ID"])}, n_clicks=0,
                        className="crcc-details-btn", style=DETAILS_BTN_STYLE),
        ] for _, r in top.iterrows()]
        mismatch_table = _table(
            ["Customer", "NOVA rating", "CLS rating", "NOVA exposure", "CLS exposure", "Diff", ""], rows,
        ) if rows else html.P("No comparable pair (both sides populated) to rank.", style={"color": "#a0aec0"})
        if n_excluded:
            excluded_note = html.Div(
                f"{n_excluded:,} customer(s) excluded from this ranking — either NOVA or CLS has no exposure "
                f"value for them (shown as missing, not treated as a 0-vs-real-value mismatch). Click a row's "
                f"Details to see the raw data on both sides.",
                style={"fontSize": "11px", "color": "#a0aec0", "marginTop": "8px"},
            )

    return html.Div([
        html.Div(tiles, style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "14px", "marginBottom": "16px"}),
        html.Div([_panel_title("Exposure — NOVA (computed) vs. CLS (reported)"),
                  dcc.Graph(figure=fig, config={"displayModeBar": False}) if fig else
                  html.P("No exposure column found on the CLS side.", style={"color": "#a0aec0"})],
                 style={**CARD_STYLE, "marginBottom": "16px"}),
        html.Div([_panel_title("Top 15 exposure mismatches"), mismatch_table, excluded_note], style=CARD_STYLE),
    ])


def _fmt_money_or_dash(v) -> str:
    return f"€{v:,.0f}" if pd.notna(v) else "—"


def _render_comparison_detail(customer_id: str):
    """Side-by-side raw data from both datasets for one customer, so a
    flagged mismatch can actually be diagnosed rather than just observed."""
    nova_match = NOVA_CUSTOMER_STATE[NOVA_CUSTOMER_STATE["ID_CUSTOMER"] == customer_id]
    nova_row = nova_match.iloc[0] if not nova_match.empty else None

    cls_row = None
    cls_exp_col = cls_arr_col = None
    if CLS_AVAILABLE and CLS_COL.get("Id Customer"):
        id_c = CLS_COL["Id Customer"]
        m = CLS_CUSTOMER_STATE[CLS_CUSTOMER_STATE[id_c].astype(str).str.strip() == customer_id]
        cls_row = m.iloc[0] if not m.empty else None
        cls_exp_col = CLS_COL.get("Total Exposure (incl. Pending Ord) (LTR)")
        cls_arr_col = CLS_COL.get("Total Arrears")

    if nova_row is None:
        nova_panel = html.P("No NOVA data for this customer.", style={"color": "#a0aec0"})
    else:
        cob = nova_row.get("COB_DATE")
        rows = [
            ["Country", nova_row.get("COUNTRY") or "—"],
            ["Latest COB Date", cob.strftime("%Y-%m-%d") if pd.notna(cob) else "—"],
            ["Group Rating", nova_row.get("GROUP_RATING_DISP") or "NR"],
            ["Counterparty Rating", nova_row.get("COUNTERPARTY_RATING_DISP") or "NR"],
            ["CLS Rating", nova_row.get("CLS_GROUP_RATING_DISP") or "NR"],
            ["Vehicles", nova_row.get("N_VEHICLES", 0)],
            ["Exposure Amount (LTR)", _fmt_money_or_dash(nova_row.get("EXPOSURE_AMOUNT_LTR"))],
            ["Pending Orders", _fmt_money_or_dash(nova_row.get("PENDING_ORDERS"))],
            ["Total Exposure (computed)", _fmt_money_or_dash(nova_row.get("TOTAL_EXPOSURE"))],
            ["Total Arrears (computed)", _fmt_money_or_dash(nova_row.get("TOTAL_ARREARS"))
                if ARRS_COLUMNS_PRESENT else "not available in this NOVA export"],
        ]
        nova_panel = _table(["Field", "Value"], rows)

    if cls_row is None:
        cls_panel = html.P("No CLS data for this customer.", style={"color": "#a0aec0"})
    else:
        def g(label):
            c = CLS_COL.get(label)
            return cls_row.get(c) if c else None

        rating_date = g("Obligor Rating Date (CLS)")
        period_end = g("Period End")
        rows = [
            ["Country", g("Code Country") or "—"],
            ["Period End", period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else "—"],
            ["Obligor Rating", _fmt_rating("CLS_GROUP_RATING", g("Obligor Rating (CLS)")) or "NR"],
            ["Rating Date", rating_date.strftime("%Y-%m-%d") if pd.notna(rating_date) else "—"],
            ["Exposure Amount (LTR)", _fmt_money_or_dash(g("Exposure Amount (LTR)"))],
            ["Pending Orders Amount", _fmt_money_or_dash(g("Pending Orders Amount"))],
            ["Total Exposure (reported)", _fmt_money_or_dash(g("Total Exposure (incl. Pending Ord) (LTR)"))],
            ["Total Arrears (reported)", _fmt_money_or_dash(g("Total Arrears"))],
            ["Event Of Default", "Yes" if _as_bool_flag(pd.Series([g("Event Of Default")])).iloc[0] else "No"],
            ["Credit Watch", "Yes" if _as_bool_flag(pd.Series([g("Credit Watch Indicator (CLS)")])).iloc[0] else "No"],
            ["Credit Limit (LTR)", _fmt_money_or_dash(g("Limit Value Amount (LTR)"))],
            ["Litigation Status", g("Litigation Status") or "—"],
        ]
        cls_panel = _table(["Field", "Value"], rows)

    diag = []
    if nova_row is not None and cls_row is not None:
        nova_exp, cls_exp = nova_row.get("TOTAL_EXPOSURE"), (cls_row.get(cls_exp_col) if cls_exp_col else None)
        if pd.notna(nova_exp) and pd.notna(cls_exp):
            d = nova_exp - cls_exp
            pct = f" ({100 * d / cls_exp:+.1f}% vs CLS)" if cls_exp else ""
            diag.append(f"Exposure diff: €{d:,.0f}{pct}")
        elif pd.isna(cls_exp):
            diag.append("CLS has no exposure value for this customer.")
        elif pd.isna(nova_exp):
            diag.append("NOVA has no computed exposure for this customer.")

        nova_arr = nova_row.get("TOTAL_ARREARS") if ARRS_COLUMNS_PRESENT else None
        cls_arr = cls_row.get(cls_arr_col) if cls_arr_col else None
        if pd.notna(nova_arr) and pd.notna(cls_arr):
            diag.append(f"Arrears diff: €{nova_arr - cls_arr:,.0f}")

    return html.Div([
        html.Div(customer_id, style={"fontWeight": "700", "fontSize": "18px", "color": "#1a1a2e", "marginBottom": "4px"}),
        html.Div(" · ".join(diag) or "Not enough data on both sides to compute a diagnosis.",
                 style={"fontSize": "12px", "color": "#718096", "marginBottom": "16px"}),
        html.Div([
            html.Div([_panel_title("NOVA"), nova_panel], style={**CARD_STYLE, "flex": "1", "minWidth": "0"}),
            html.Div([_panel_title("CLS"), cls_panel], style={**CARD_STYLE, "flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
    ])


def render_exposure_risk():
    if NOVA_CUSTOMER_STATE.empty:
        return html.P("No NOVA customer data available.", style={"color": "#a0aec0"})

    d = NOVA_CUSTOMER_STATE.dropna(subset=["TOTAL_EXPOSURE"]).copy()
    total_exp = d["TOTAL_EXPOSURE"].sum()
    top10 = d.sort_values("TOTAL_EXPOSURE", ascending=False).head(10)
    top10_share = 100 * top10["TOTAL_EXPOSURE"].sum() / total_exp if total_exp else 0

    d["_BAND"] = d["CLS_GROUP_RATING_DISP"].fillna("NR")
    by_band = d.groupby("_BAND")["TOTAL_EXPOSURE"].sum().sort_index(
        key=lambda idx: [_rating_sort_key_local(v) for v in idx])
    fig = go.Figure(go.Bar(x=list(by_band.index), y=list(by_band.values), marker_color="#3182ce"))
    fig.update_layout(
        height=320, margin=dict(l=50, r=20, t=20, b=40), showlegend=False,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#4a5568"),
        xaxis=dict(title="CLS Rating band", gridcolor="#f0f2f5"),
        yaxis=dict(title="Total exposure (€)", gridcolor="#f0f2f5"),
    )

    over_limit_block = html.P("CLS data not available for over-limit check.", style={"color": "#a0aec0"})
    if CLS_AVAILABLE and CLS_COL.get("Exposure Amount (LTR)") and CLS_COL.get("Limit Value Amount (LTR)"):
        exp_c, limit_c, id_c = CLS_COL["Exposure Amount (LTR)"], CLS_COL["Limit Value Amount (LTR)"], CLS_COL["Id Customer"]
        over = CLS_CUSTOMER_STATE.dropna(subset=[exp_c, limit_c])
        over = over[over[exp_c] > over[limit_c]].sort_values(exp_c, ascending=False).head(15)
        if over.empty:
            over_limit_block = html.P("No customer currently exceeds their credit limit.", style={"color": "#2f855a", "fontWeight": "600"})
        else:
            rows = [[r[id_c], f"€{r[exp_c]:,.0f}", f"€{r[limit_c]:,.0f}", f"€{r[exp_c] - r[limit_c]:,.0f}"]
                    for _, r in over.iterrows()]
            over_limit_block = _table(["Customer", "Exposure", "Limit", "Over by"], rows)

    return html.Div([
        html.Div([
            _stat_tile("Total NOVA exposure", f"€{total_exp / 1_000_000:,.2f}M"),
            _stat_tile("Top 10 concentration", f"{top10_share:.1f}%", "share of total exposure held by the 10 largest customers",
                       accent="#d98943" if top10_share > 30 else "#3182ce"),
            _stat_tile("Customers with exposure", f"{len(d):,}"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px", "marginBottom": "16px"}),
        html.Div([_panel_title("Exposure by rating band"), dcc.Graph(figure=fig, config={"displayModeBar": False})],
                 style={**CARD_STYLE, "marginBottom": "16px"}),
        html.Div([_panel_title("Customers over their credit limit"), over_limit_block], style=CARD_STYLE),
    ])


def render_arrears_onset():
    if not CLS_AVAILABLE:
        return _missing_cls_banner()
    if ARREARS_ONSET_DF.empty:
        return html.P("No customer currently shows Total Arrears > 0 in CLS.", style={"color": "#2f855a", "fontWeight": "600"})

    n_ever_arrears = len(ARREARS_ONSET_DF)
    n_defaulted = int(ARREARS_ONSET_DF["DEFAULT_DATE"].notna().sum())
    default_rate = 100 * n_defaulted / n_ever_arrears if n_ever_arrears else 0
    lead_times = ARREARS_ONSET_DF["DAYS_TO_DEFAULT"].dropna()

    tiles = [
        _stat_tile("Customers who ever had arrears", f"{n_ever_arrears:,}"),
        _stat_tile("...who later defaulted", f"{n_defaulted:,}", f"{default_rate:.1f}% of the above",
                   accent="#9b2c2c" if n_defaulted else "#2f855a"),
        _stat_tile("Median lead time to default", f"{lead_times.median():.0f} days" if len(lead_times) else "—",
                   "from first arrears to Event Of Default"),
    ]

    fig = None
    if len(lead_times) > 0:
        fig = go.Figure(go.Histogram(x=lead_times, marker_color="#3182ce", nbinsx=20))
        fig.update_layout(
            height=300, margin=dict(l=50, r=20, t=20, b=40), showlegend=False,
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#4a5568"),
            xaxis=dict(title="Days from first arrears to default", gridcolor="#f0f2f5"),
            yaxis=dict(title="Customers", gridcolor="#f0f2f5"),
        )

    fastest = ARREARS_ONSET_DF.dropna(subset=["DAYS_TO_DEFAULT"]).sort_values("DAYS_TO_DEFAULT").head(15)
    rows = [[r["ID_CUSTOMER"], r["ONSET_DATE"].strftime("%Y-%m-%d"), r["DEFAULT_DATE"].strftime("%Y-%m-%d"),
             f"{int(r['DAYS_TO_DEFAULT'])} days"] for _, r in fastest.iterrows()]
    fastest_table = _table(["Customer", "Arrears onset", "Default date", "Lead time"], rows) if rows else \
        html.P("No customer with both an arrears onset and a default date.", style={"color": "#a0aec0"})

    children = [html.Div(tiles, style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px", "marginBottom": "16px"})]
    if fig:
        children.append(html.Div([_panel_title("Lead time from first arrears to default"),
                                   dcc.Graph(figure=fig, config={"displayModeBar": False})],
                                  style={**CARD_STYLE, "marginBottom": "16px"}))
    children.append(html.Div([_panel_title("Fastest arrears-to-default cases"), fastest_table], style=CARD_STYLE))
    return html.Div(children)


def render_adjusted_exposure():
    if ADJUSTED_EXPOSURE_DF.empty:
        return html.P("No NOVA customer data available.", style={"color": "#a0aec0"})

    d = ADJUSTED_EXPOSURE_DF
    raw_total = d["TOTAL_EXPOSURE"].sum()
    adj_total = d["ADJUSTED_EXPOSURE"].sum()
    uplift_pct = 100 * (adj_total - raw_total) / raw_total if raw_total else 0
    # NaN uplift (0 raw exposure, but arrears now add real exposure) counts as
    # material too — it's the most dramatic case, not a non-event.
    n_material = int(((d["UPLIFT_PCT"] > 5) | (d["UPLIFT_PCT"].isna() & (d["ARREARS_USED"] > 0))).sum())

    tiles = [
        _stat_tile("Raw exposure", f"€{raw_total / 1_000_000:,.2f}M", "LTR + pending only"),
        _stat_tile("Adjusted exposure", f"€{adj_total / 1_000_000:,.2f}M", "raw + arrears",
                    accent="#d98943" if uplift_pct > 0 else "#3182ce"),
        _stat_tile("Portfolio uplift", f"{uplift_pct:+.2f}%", "adjusted vs. raw, whole portfolio"),
        _stat_tile("Customers with >5% uplift", f"{n_material:,}", "arrears meaningfully change their risk picture"),
    ]

    d = d.copy()
    d["_BAND"] = d["CLS_GROUP_RATING_DISP"].fillna("NR")
    by_band = d.groupby("_BAND")[["TOTAL_EXPOSURE", "ADJUSTED_EXPOSURE"]].sum()
    by_band = by_band.reindex(sorted(by_band.index, key=_rating_sort_key_local))
    fig = go.Figure([
        go.Bar(name="Raw", x=list(by_band.index), y=list(by_band["TOTAL_EXPOSURE"]), marker_color="#a0aec0"),
        go.Bar(name="Adjusted", x=list(by_band.index), y=list(by_band["ADJUSTED_EXPOSURE"]), marker_color="#3182ce"),
    ])
    fig.update_layout(
        height=340, margin=dict(l=50, r=20, t=20, b=40), barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#4a5568"),
        xaxis=dict(title="CLS Rating band", gridcolor="#f0f2f5"),
        yaxis=dict(title="Exposure (€)", gridcolor="#f0f2f5"),
    )

    # NaN (0 -> something) sorts as "worse than any finite %", so it leads the table
    # instead of dropping to the bottom under pandas' default NaN-last sort.
    top_uplift = d[d["ARREARS_USED"] > 0].copy()
    top_uplift["_SORT_KEY"] = top_uplift["UPLIFT_PCT"].fillna(np.inf)
    top_uplift = top_uplift.sort_values("_SORT_KEY", ascending=False).head(15)
    rows = [[
        r["ID_CUSTOMER"], r.get("CLS_GROUP_RATING_DISP") or "NR",
        f"€{r['TOTAL_EXPOSURE']:,.0f}", f"€{r['ARREARS_USED']:,.0f}",
        f"€{r['ADJUSTED_EXPOSURE']:,.0f}", f"{r['UPLIFT_PCT']:+.1f}%" if pd.notna(r["UPLIFT_PCT"]) else "New",
    ] for _, r in top_uplift.iterrows()]
    top_table = _table(["Customer", "Rating", "Raw exposure", "Arrears added", "Adjusted exposure", "Uplift"], rows) \
        if rows else html.P("No customer currently has arrears folded into their exposure.", style={"color": "#2f855a", "fontWeight": "600"})

    return html.Div([
        html.Div([
            html.Div("Methodology", style={"fontWeight": "700", "fontSize": "13px", "color": "#1a1a2e", "marginBottom": "4px"}),
            html.Div("Adjusted Exposure = Total Exposure (EXPOSURE_AMOUNT_LTR + PENDING_ORDERS, dedup per "
                     "contract) + Arrears. Money that's overdue but unpaid is still money at risk on top of "
                     "what's already outstanding — this is closer to how exposure is actually sized at larger "
                     "financial institutions than the raw contracted amount alone. Every figure here is anchored "
                     "to each customer's own latest NOVA COB_DATE, never a CLS reporting date.",
                     style={"fontSize": "12px", "color": "#718096", "lineHeight": "1.5", "marginBottom": "8px"}),
            html.Div(f"Arrears source used: {ADJUSTED_EXPOSURE_ARREARS_SOURCE}.",
                     style={"fontSize": "11px", "color": "#3182ce", "fontWeight": "600"}),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),
        html.Div(tiles, style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "14px", "marginBottom": "16px"}),
        html.Div([_panel_title("Exposure by rating band — raw vs. adjusted"),
                  dcc.Graph(figure=fig, config={"displayModeBar": False})],
                 style={**CARD_STYLE, "marginBottom": "16px"}),
        html.Div([_panel_title("Top 15 customers by uplift"), top_table], style=CARD_STYLE),
    ])


def _rating_sort_key_local(disp: str):
    if disp == "NR":
        return (-1, 0)
    try:
        return (0, float(disp))
    except (TypeError, ValueError):
        s, suffix = disp, 0
        if s.endswith("+"):
            suffix, s = 1, s[:-1]
        elif s.endswith("-"):
            suffix, s = -1, s[:-1]
        try:
            return (0, int(s), suffix)
        except ValueError:
            return (0, 999, 0)


TABS = [
    ("overview", "Overview", render_overview),
    ("quality", "Data Quality", render_data_quality),
    ("comparison", "NOVA ↔ CLS Comparison", render_comparison),
    ("adjusted", "Adjusted Exposure", render_adjusted_exposure),
    ("exposure", "Exposure Risk", render_exposure_risk),
    ("onset", "Arrears Onset", render_arrears_onset),
]

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Credit Risk Control Center", style={"margin": "0 0 4px", "fontSize": "22px",
                                                               "fontWeight": "700", "color": "#1a1a2e"}),
                html.Div("NOVA fleet data cross-checked against the CLS obligor extract — data quality, "
                         "exposure risk, and arrears-to-default analysis.",
                         style={"fontSize": "12px", "color": "#718096"}),
            ],
            style={"padding": "20px 28px", "borderBottom": "1px solid #e2e8f0", "background": "#ffffff"},
        ),
        dcc.Tabs(id="crcc-tabs", value="overview",
                  children=[dcc.Tab(label=label, value=key) for key, label, _ in TABS],
                  style={"padding": "0 28px"}),
        html.Div(id="crcc-tab-content", style={"padding": "24px 28px"}),

        html.Div(
            id="crcc-modal",
            style=MODAL_OVERLAY_STYLE,
            children=[
                html.Div(
                    style=MODAL_PANEL_STYLE,
                    children=[
                        html.Div(
                            html.Button("Close", id="crcc-modal-close", n_clicks=0,
                                        className="crcc-secondary-btn", style=SECONDARY_BTN_STYLE),
                            style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"},
                        ),
                        html.Div(id="crcc-modal-body"),
                    ],
                ),
            ],
        ),
    ],
    style=PAGE_STYLE,
)


@app.callback(Output("crcc-tab-content", "children"), Input("crcc-tabs", "value"))
def _render_tab(tab_value):
    for key, _, renderer in TABS:
        if key == tab_value:
            return renderer()
    return html.P("Unknown tab.")


@app.callback(
    Output("crcc-modal", "style"),
    Output("crcc-modal-body", "children"),
    Input({"type": "cmp-details", "index": ALL}, "n_clicks"),
    Input("crcc-modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def _toggle_comparison_modal(card_clicks, _close_clicks):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if "crcc-modal-close" in trig:
        return {**MODAL_OVERLAY_STYLE, "display": "none"}, no_update

    if not any(card_clicks or []):
        return no_update, no_update

    prop_id = trig.rsplit(".", 1)[0]
    try:
        cust_id = json.loads(prop_id)["index"]
    except Exception:
        return no_update, no_update

    return {**MODAL_OVERLAY_STYLE, "display": "flex"}, _render_comparison_detail(cust_id)


if __name__ == "__main__":
    app.run(debug=True, port=8056)
