from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update, ALL, callback_context
import dash

# =============================================================================
# Config — adjust to your environment
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
MODELS_PATH = BASE_DIR / "models.parquet"
LOGO_PATH   = BASE_DIR / "a.jpg"

COUNTRIES_TO_READ: list[str] = ["SPAIN"]   # e.g. ["SPAIN", "ITALY", "FRANCE"]
START_YYYYMM = "202301"
END_YYYYMM   = "202512"

COLUMNS_TO_READ = [
    "COB_DATE", "ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION",
    "COUNTRY", "NOVA_ASSET_STATUS", "BIKE_OR_CAR",
    "CLASS_CATALOG", "BRAND_UPDATE", "VEHICLE_CLASS",
    "POWER_CATEGORY", "FUEL_TYPE2", "FUEL_TYPE",
    "FINAL_CONTRACT_DURATION", "VA_CO2_EMSS_REAL",
    "OEM_UPDATE",
    # Synthetic fields (added by add_new_fields_to_parquets.py)
    "GROUP_RATING", "COUNTERPARTY_RATING", "CLS_GROUP_RATING",
    "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION", "SHARED_CLIENT_FLAG",
    "VEHICLE_PRICE_EUR", "EXPOSURE_AMOUNT_LTR", "EXPOSURE_AMOUNT_MTR",
    "OBLIGOR_IDENTIFIER",
    # Temporary stand-ins backfilled by scripts/add_pending_orders_and_id_customer.py
    "PENDING_ORDERS", "ID_CUSTOMER",
]

UNIQUE_KEY_COLS = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]

# Columns expected/desired from models.parquet
_MODELS_ENRICH_COLS = [
    "BRAND_UPDATE", "MODEL",
    "MARKET_MODEL", "MARKET_BODY_GROUP",
    "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE",
]

PAGE_SIZE = 15


# =============================================================================
# Data loading — same filename convention as key_refactor
# Filename pattern:  <prefix>-<COUNTRY>-<YYYYMM>.parquet
# =============================================================================

def load_country_monthly_data(
    folder_path: Path | str,
    countries: list[str],
    start_yyyymm: str,
    end_yyyymm: str,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reads parquet files whose names follow  *-<COUNTRY>-<YYYYMM>.parquet.
    Country is decoded from parts[1], period from parts[2].
    """
    files = glob.glob(f"{folder_path}/*.parquet")
    dfs: list[pd.DataFrame] = []

    start_int = int(start_yyyymm)
    end_int   = int(end_yyyymm)
    countries_upper = {c.strip().upper() for c in countries}

    for fp in files:
        name  = os.path.basename(fp).replace(".parquet", "")
        parts = [p.strip() for p in name.split("-")]
        if len(parts) < 3:
            continue
        file_country = parts[1].upper()
        try:
            file_yyyymm = int(parts[2])
        except ValueError:
            continue
        if file_country in countries_upper and start_int <= file_yyyymm <= end_int:
            chunk = pd.read_parquet(fp, columns=cols)
            dfs.append(chunk)

    if not dfs:
        return pd.DataFrame(columns=cols or COLUMNS_TO_READ)
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Models enrichment — same logic as key_refactor
# =============================================================================

def _load_models_df() -> pd.DataFrame:
    path = str(MODELS_PATH)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(path).names
        cols_to_read = [c for c in _MODELS_ENRICH_COLS if c in schema_cols]
        mdf = pd.read_parquet(path, columns=cols_to_read)
        mdf["MODEL"]        = mdf["MODEL"].astype(str).str.strip()
        mdf["BRAND_UPDATE"] = mdf["BRAND_UPDATE"].astype(str).str.strip()
        return mdf.drop_duplicates(subset=["BRAND_UPDATE", "MODEL"])
    except Exception:
        return pd.DataFrame()


MODELS_DF = _load_models_df()


def enrich_with_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts MODEL from CLASS_CATALOG (split on '/'), then left-joins models.parquet
    to add MARKET_MODEL, MARKET_BODY_GROUP (and CDN columns if present).
    """
    enrich_cols = ["MARKET_MODEL", "MARKET_BODY_GROUP", "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE"]

    if "CLASS_CATALOG" not in df.columns or "BRAND_UPDATE" not in df.columns:
        return df

    out = df.copy()
    out["MODEL"]        = out["CLASS_CATALOG"].astype(str).str.split("/").str[0].str.strip()
    out["BRAND_UPDATE"] = out["BRAND_UPDATE"].astype(str).str.strip()

    if MODELS_DF.empty:
        for c in enrich_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    # Skip if already fully enriched
    if all(c in out.columns and out[c].notna().any() for c in ["MARKET_MODEL", "MARKET_BODY_GROUP"]):
        return out

    out = out.drop(columns=[c for c in enrich_cols if c in out.columns], errors="ignore")
    merge_cols = [c for c in _MODELS_ENRICH_COLS if c in MODELS_DF.columns]
    out = out.merge(MODELS_DF[merge_cols], how="left", on=["BRAND_UPDATE", "MODEL"])
    return out


# =============================================================================
# Data preparation
# =============================================================================

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "COB_DATE" in df.columns:
        df["COB_DATE"] = pd.to_datetime(df["COB_DATE"], errors="coerce")
    df["YEAR"]  = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month
    for col in ["COUNTRY", "NOVA_ASSET_STATUS", "BIKE_OR_CAR", "BRAND_UPDATE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # CO2_BUCKET — same bucketing as notebook ([0-9], [10-19], ...)
    if "VA_CO2_EMSS_REAL" in df.columns:
        def _co2_bucket(v):
            try:
                n = int(float(v)); lo = (n // 10) * 10
                return f"[{lo}-{lo+9}]"
            except Exception:
                return "UNK"
        df["CO2_BUCKET"] = df["VA_CO2_EMSS_REAL"].apply(_co2_bucket)
    # EXPOSURE_AMOUNT_TOT
    if "EXPOSURE_AMOUNT_LTR" in df.columns and "EXPOSURE_AMOUNT_MTR" in df.columns:
        df["EXPOSURE_AMOUNT_TOT"] = df["EXPOSURE_AMOUNT_LTR"].fillna(0) + df["EXPOSURE_AMOUNT_MTR"].fillna(0)
    return df


def prepare_uc1(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to CAR (any NOVA_ASSET_STATUS — that's a live dashboard filter now),
    enrich with market model/body type."""
    if df.empty:
        return df
    d = df[df["BIKE_OR_CAR"] == "CAR"].copy()
    d = enrich_with_models(d)
    return d


# =============================================================================
# Startup: load + prepare
# =============================================================================

_raw = load_country_monthly_data(
    DATA_FOLDER, COUNTRIES_TO_READ, START_YYYYMM, END_YYYYMM,
    cols=[c for c in COLUMNS_TO_READ if c],
)
_raw = prepare_dataset(_raw)
UC1_DF = prepare_uc1(_raw)

# Pre-compute filter option lists (used to populate dropdowns at layout time)
_countries_avail  = sorted(UC1_DF["COUNTRY"].dropna().unique().tolist()) if not UC1_DF.empty else []
_brands_avail     = sorted(UC1_DF["BRAND_UPDATE"].dropna().unique().tolist()) if not UC1_DF.empty else []
_models_avail     = sorted(UC1_DF["MARKET_MODEL"].dropna().unique().tolist()) if "MARKET_MODEL" in UC1_DF.columns and not UC1_DF.empty else []
_btypes_avail     = sorted(UC1_DF["MARKET_BODY_GROUP"].dropna().unique().tolist()) if "MARKET_BODY_GROUP" in UC1_DF.columns and not UC1_DF.empty else []
_asset_statuses_avail = sorted(UC1_DF["NOVA_ASSET_STATUS"].dropna().unique().tolist()) if not UC1_DF.empty else []

# COB period lists for the global time filter
_cob_monthly_vals   = sorted(UC1_DF["COB_DATE"].dropna().dt.to_period("M").astype(str).unique().tolist()) if not UC1_DF.empty else []
_cob_quarterly_vals = sorted(set(
    f"{p[:4]}-Q{(int(p[5:7]) - 1) // 3 + 1}" for p in _cob_monthly_vals
))
_cob_yearly_vals    = sorted(set(p[:4] for p in _cob_monthly_vals))
_default_cob_period = _cob_monthly_vals[-1] if _cob_monthly_vals else None


# =============================================================================
# Heatmap axis / metric options
# =============================================================================

# Every categorical dimension is available on both axes — user picks freely,
# mutual exclusion (can't pick the same field on X and Y) is enforced by callback.
ALL_AXIS_OPTIONS = [
    {"label": "Brand",                      "value": "BRAND_UPDATE"},
    {"label": "Power Category",             "value": "POWER_CATEGORY"},
    {"label": "CO2 Bucket",                 "value": "CO2_BUCKET"},
    {"label": "Industry Type",              "value": "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"},
    {"label": "Group Rating",               "value": "GROUP_RATING"},
    {"label": "Counterparty Rating",        "value": "COUNTERPARTY_RATING"},
    {"label": "CLS Group Rating",           "value": "CLS_GROUP_RATING"},
]
Y_OPTIONS = ALL_AXIS_OPTIONS
X_OPTIONS = ALL_AXIS_OPTIONS

METRIC_OPTIONS = [
    {"label": "Volume (Contracts)",         "value": "volume"},
    {"label": "Exposure",                   "value": "exposure"},
    {"label": "Vehicle Price",              "value": "intensite_risk_asset"},
]

ASSET_STATUS_OPTIONS = [{"label": "ALL", "value": "ALL"}] + [
    {"label": s, "value": s} for s in _asset_statuses_avail
]


# =============================================================================
# Core helpers: filter → COB filter → pivot → figure
# =============================================================================

def apply_filters(country, brands, models, bodytypes, asset_status=None) -> pd.DataFrame:
    d = UC1_DF.copy()
    if country and country != "ALL" and "COUNTRY" in d.columns:
        d = d[d["COUNTRY"] == country]
    if brands:
        d = d[d["BRAND_UPDATE"].isin(brands)]
    if models and "MARKET_MODEL" in d.columns:
        d = d[d["MARKET_MODEL"].isin(models)]
    if bodytypes and "MARKET_BODY_GROUP" in d.columns:
        d = d[d["MARKET_BODY_GROUP"].isin(bodytypes)]
    if asset_status and asset_status != "ALL" and "NOVA_ASSET_STATUS" in d.columns:
        d = d[d["NOVA_ASSET_STATUS"] == asset_status]
    return d


def apply_cob_filter(d: pd.DataFrame, granularity: str, period: str) -> pd.DataFrame:
    if not period or "COB_DATE" not in d.columns:
        return d
    if granularity == "monthly":
        return d[d["COB_DATE"].dt.to_period("M").astype(str) == period]
    if granularity == "quarterly":
        year_s, q_s = period.split("-Q")
        months = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}[int(q_s)]
        return d[(d["COB_DATE"].dt.year == int(year_s)) & (d["COB_DATE"].dt.month.isin(months))]
    if granularity == "yearly":
        return d[d["COB_DATE"].dt.year == int(period)]
    return d


def compute_row_exposure(d: pd.DataFrame) -> pd.Series:
    return d["EXPOSURE_AMOUNT_LTR"].fillna(0) + d["PENDING_ORDERS"].fillna(0)


def _fill_rating_na(d: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Missing GROUP_RATING/COUNTERPARTY_RATING become the literal "NR" category
    instead of being dropped, so unrated contracts still show up on the heatmap."""
    d = d.copy()
    for c in cols:
        if c in RATING_ORDER_FIELDS and c in d.columns:
            d[c] = d[c].fillna(RATING_NR_LABEL)
    return d


def build_exposure_pivot(d: pd.DataFrame, y_col: str, x_col: str, aggregation: str = "sum") -> pd.DataFrame:
    needed = {y_col, x_col, "ID_CUSTOMER", "EXPOSURE_AMOUNT_LTR", "PENDING_ORDERS", "COB_DATE"}
    cols = [c for c in needed if c in d.columns]
    d = _fill_rating_na(d[cols], y_col, x_col).dropna(subset=[y_col, x_col])
    if d.empty or "ID_CUSTOMER" not in d.columns:
        return pd.DataFrame()

    d["EXPOSURE"] = compute_row_exposure(d)

    if "COB_DATE" in d.columns:
        d = d.sort_values("COB_DATE")
    d = d.drop_duplicates(subset=[y_col, x_col, "ID_CUSTOMER"], keep="last")

    agg = "mean" if aggregation == "mean" else "sum"
    return d.groupby([y_col, x_col])["EXPOSURE"].agg(agg).unstack(x_col, fill_value=0)


def build_pivot(d: pd.DataFrame, y_col: str, x_col: str, metric: str, aggregation: str = "sum") -> pd.DataFrame:
    if metric == "exposure":
        return _apply_rating_order(build_exposure_pivot(d, y_col, x_col, aggregation), y_col, x_col)

    keep = list({y_col, x_col} | {c for c in UNIQUE_KEY_COLS if c in d.columns} | {"COB_DATE", "VEHICLE_PRICE_EUR"} & set(d.columns))
    d = _fill_rating_na(d[[c for c in keep if c in d.columns]], y_col, x_col).dropna(subset=[y_col, x_col])
    if d.empty:
        return pd.DataFrame()

    # Dedup: latest COB snapshot per unique contract key
    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    if keys:
        if "COB_DATE" in d.columns:
            d = d.sort_values("COB_DATE")
        d = d.drop_duplicates(subset=keys, keep="last")

    if metric == "intensite_risk_asset" and "VEHICLE_PRICE_EUR" in d.columns:
        piv = d.groupby([y_col, x_col])["VEHICLE_PRICE_EUR"].sum().unstack(x_col, fill_value=0)
    else:  # "volume" or fallback
        piv = d.groupby([y_col, x_col]).size().unstack(x_col, fill_value=0)

    return _apply_rating_order(piv, y_col, x_col)


def compute_total_exposure(d: pd.DataFrame, aggregation: str = "sum") -> float:
    if d.empty or "ID_CUSTOMER" not in d.columns:
        return 0.0
    dd = d.copy()
    dd["EXPOSURE"] = compute_row_exposure(dd)
    if "COB_DATE" in dd.columns:
        dd = dd.sort_values("COB_DATE")
    dd = dd.drop_duplicates(subset=["ID_CUSTOMER"], keep="last")
    return float(dd["EXPOSURE"].sum()) if aggregation != "mean" else float(dd["EXPOSURE"].mean())


def compute_total_metric(d: pd.DataFrame, metric: str, aggregation: str = "sum") -> float:
    if metric == "exposure":
        return compute_total_exposure(d, aggregation)
    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    dd = d.copy()
    if keys:
        if "COB_DATE" in dd.columns:
            dd = dd.sort_values("COB_DATE")
        dd = dd.drop_duplicates(subset=keys, keep="last")
    if metric == "intensite_risk_asset" and "VEHICLE_PRICE_EUR" in dd.columns:
        return float(dd["VEHICLE_PRICE_EUR"].sum())
    return float(len(dd))


def format_millions(value: float) -> str:
    s = f"{value / 1_000_000:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def make_heatmap_fig(
    piv: pd.DataFrame, title: str, colorscale: str = "Blues",
    page: int = 0, zmid=None, zmin=None, zmax=None, show_totals: bool = False,
) -> go.Figure:
    if piv.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font={"size": 14, "color": "#aaa"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220)
        return fig

    paged = piv.iloc[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]
    z = paged.values.astype(float)
    x_lbl = [str(c) for c in paged.columns]
    y_lbl = [str(r) for r in paged.index]

    text = [[_fmt_val(v) for v in row] for row in z]

    if show_totals and not paged.empty:
        row_totals = paged.sum(axis=1).values
        col_totals = piv.reindex(columns=paged.columns).sum(axis=0).values
        grand_total = float(piv.values.sum())

        z = np.hstack([z, np.full((z.shape[0], 1), np.nan)])
        for i, v in enumerate(row_totals):
            text[i] = text[i] + [_fmt_val(v)]
        x_lbl = x_lbl + ["Total"]

        # Prepended (not appended) — Plotly's categorical y-axis renders the last
        # array entry at the top, so putting "Total" first here puts it at the
        # bottom of the chart, matching a conventional totals row.
        z = np.vstack([np.full((1, z.shape[1]), np.nan), z])
        text = [[_fmt_val(v) for v in col_totals] + [_fmt_val(grand_total)]] + text
        y_lbl = ["Total"] + y_lbl

    hm_kwargs: dict = dict(
        z=z.tolist(), x=x_lbl, y=y_lbl,
        text=text, texttemplate="%{text}",
        colorscale=colorscale, hoverongaps=False,
        hovertemplate="%{y} × %{x}: %{z}<extra></extra>",
    )
    if zmid is not None: hm_kwargs["zmid"] = zmid
    if zmin  is not None: hm_kwargs["zmin"] = zmin
    if zmax  is not None: hm_kwargs["zmax"] = zmax

    fig = go.Figure(go.Heatmap(**hm_kwargs))
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": "#1a1a2e"}, "x": 0.01},
        margin={"l": 10, "r": 10, "t": 50, "b": 85},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "size": 11},
        height=max(300, 60 + 28 * len(y_lbl)),
        xaxis={"tickangle": -40, "tickfont": {"size": 10}},
        yaxis={"tickfont": {"size": 10}},
    )
    return fig


# =============================================================================
# Simulation domain helpers — obligor "customers", brand/model maps, templates
# =============================================================================

RATING_ORDER_FIELDS = {"GROUP_RATING", "COUNTERPARTY_RATING"}
RATING_NR_LABEL = "NR"


def _rating_sort_key(value) -> tuple:
    """Sort key for GROUP_RATING/COUNTERPARTY_RATING grades, parsed algorithmically
    (numeric grade, then -/plain/+ sub-order: e.g. "2-" < "2" < "2+") instead of a
    fixed list, so it keeps working for any grade the data has (1 up to 12+ ...).
    Missing/blank ratings ("NR") always sort first."""
    v = str(value).strip().upper()
    if v in ("", RATING_NR_LABEL, "NAN", "NONE"):
        return (-1, 0)
    suffix_rank, core = 1, v
    if v.endswith("+"):
        suffix_rank, core = 2, v[:-1]
    elif v.endswith("-"):
        suffix_rank, core = 0, v[:-1]
    try:
        grade = int(core)
    except ValueError:
        return (10 ** 6, 0)
    return (grade, suffix_rank)


def _ordered_axis(values, col: str, is_y: bool = False) -> list:
    """Order axis category values: rating grade order for rating fields (reversed
    on the y-axis, since Plotly renders a heatmap's last y entry at the top and
    the first at the bottom — opposite of the x-axis), plain sort otherwise.
    Used both to reorder a single pivot's axis and to build the union index/columns
    when aligning two pivots (e.g. Original vs Simulated) — any reindex done after
    this must reuse it instead of a plain sorted(), or the rating order is lost."""
    if col in RATING_ORDER_FIELDS:
        ordered = sorted(values, key=_rating_sort_key)
        return list(reversed(ordered)) if is_y else ordered
    return sorted(values)


def _apply_rating_order(piv: pd.DataFrame, y_col: str, x_col: str) -> pd.DataFrame:
    if piv.empty:
        return piv
    if y_col in RATING_ORDER_FIELDS:
        piv = piv.reindex(index=_ordered_axis(piv.index.tolist(), y_col, is_y=True))
    if x_col in RATING_ORDER_FIELDS:
        piv = piv.reindex(columns=_ordered_axis(piv.columns.tolist(), x_col, is_y=False))
    return piv


def _co2_bucket_low(bucket: str) -> int:
    try:
        return int(str(bucket).split("-")[0].strip("[]"))
    except Exception:
        return 10 ** 9


def co2_bucket_midpoint(bucket: str) -> float:
    try:
        lo, hi = str(bucket).strip("[]").split("-")
        return (int(lo) + int(hi)) / 2
    except Exception:
        return float("nan")


def _mode_or(series: pd.Series, default=""):
    s = series.dropna()
    if s.empty:
        return default
    m = s.mode()
    return m.iloc[0] if not m.empty else default


def _cls_rating_label(r) -> str:
    """CLS_GROUP_RATING is numeric on most datasets (1-11) but some snapshots store
    it as a non-numeric string — fall back to the raw value instead of crashing."""
    try:
        f = float(r)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(r)


# One row per OBLIGOR_IDENTIFIER ("customer") — latest snapshot — backs the
# "existing customer" search in the Add Vehicles wizard.
if not UC1_DF.empty and "OBLIGOR_IDENTIFIER" in UC1_DF.columns:
    _ob_cols = [c for c in [
        "OBLIGOR_IDENTIFIER", "COB_DATE", "COUNTRY", "GROUP_RATING",
        "COUNTERPARTY_RATING", "CLS_GROUP_RATING",
        "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION", "SHARED_CLIENT_FLAG",
    ] if c in UC1_DF.columns]
    OBLIGOR_PROFILES = (
        UC1_DF[_ob_cols].dropna(subset=["OBLIGOR_IDENTIFIER"])
        .sort_values("COB_DATE")
        .drop_duplicates(subset=["OBLIGOR_IDENTIFIER"], keep="last")
        .set_index("OBLIGOR_IDENTIFIER")
    )
else:
    OBLIGOR_PROFILES = pd.DataFrame()

# Brand -> models / body types actually present in the data (cascading dropdowns)
BRAND_MODEL_MAP: dict[str, list[str]] = {}
BRAND_BODYTYPE_MAP: dict[str, list[str]] = {}
if not UC1_DF.empty and "MARKET_MODEL" in UC1_DF.columns:
    for _b, _grp in UC1_DF.dropna(subset=["MARKET_MODEL"]).groupby("BRAND_UPDATE"):
        BRAND_MODEL_MAP[_b] = sorted(_grp["MARKET_MODEL"].dropna().unique().tolist())
if not UC1_DF.empty and "MARKET_BODY_GROUP" in UC1_DF.columns:
    for _b, _grp in UC1_DF.dropna(subset=["MARKET_BODY_GROUP"]).groupby("BRAND_UPDATE"):
        BRAND_BODYTYPE_MAP[_b] = sorted(_grp["MARKET_BODY_GROUP"].dropna().unique().tolist())

# Reverse lookup — market model -> brands that have it (used to auto-select the
# Brand filter when a Market Model is picked in the Filters panel)
MODEL_BRAND_MAP: dict[str, list[str]] = {}
if not UC1_DF.empty and "MARKET_MODEL" in UC1_DF.columns:
    for _m, _grp in UC1_DF.dropna(subset=["MARKET_MODEL"]).groupby("MARKET_MODEL"):
        MODEL_BRAND_MAP[_m] = sorted(_grp["BRAND_UPDATE"].dropna().unique().tolist())

_TEMPLATE_COLS = ["FUEL_TYPE", "FUEL_TYPE2", "VEHICLE_CLASS", "OEM_UPDATE",
                   "FINAL_CONTRACT_DURATION", "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE"]


def get_brand_model_template(brand: str, model: str | None) -> dict:
    """Most-frequent (mode) value of incidental, non-risk columns for this brand/model.
    These columns are never used as a heatmap axis or metric, so filling them
    deterministically (not randomly) just keeps the synthetic row schema-complete."""
    sub = UC1_DF[UC1_DF["BRAND_UPDATE"] == brand]
    if model and "MARKET_MODEL" in UC1_DF.columns:
        model_sub = sub[sub["MARKET_MODEL"] == model]
        if len(model_sub) >= 3:
            sub = model_sub
    out = {}
    for c in _TEMPLATE_COLS:
        if c in sub.columns:
            fallback = _mode_or(UC1_DF[c]) if c in UC1_DF.columns else ""
            out[c] = _mode_or(sub[c], default=fallback)
        else:
            out[c] = ""
    return out


def median_price_for(brand: str, model: str | None) -> float:
    if UC1_DF.empty or "VEHICLE_PRICE_EUR" not in UC1_DF.columns:
        return 20000.0
    sub = UC1_DF[UC1_DF["BRAND_UPDATE"] == brand]
    if model and "MARKET_MODEL" in UC1_DF.columns:
        model_sub = sub[sub["MARKET_MODEL"] == model]
        if len(model_sub) >= 3:
            sub = model_sub
    prices = sub["VEHICLE_PRICE_EUR"].dropna()
    if prices.empty:
        prices = UC1_DF["VEHICLE_PRICE_EUR"].dropna()
    return float(prices.median()) if not prices.empty else 20000.0


# Canonical dropdown option lists, sourced from real data (not hardcoded)
INDUSTRY_OPTIONS = (
    sorted(UC1_DF["ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"].dropna().unique().tolist())
    if "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION" in UC1_DF.columns and not UC1_DF.empty else []
)
SHARED_FLAG_OPTIONS = (
    sorted(UC1_DF["SHARED_CLIENT_FLAG"].dropna().unique().tolist())
    if "SHARED_CLIENT_FLAG" in UC1_DF.columns and not UC1_DF.empty else []
)
GROUP_RATING_OPTIONS = (
    sorted(UC1_DF["GROUP_RATING"].dropna().unique().tolist(), key=_rating_sort_key)
    if "GROUP_RATING" in UC1_DF.columns and not UC1_DF.empty else []
)
COUNTERPARTY_RATING_OPTIONS = (
    sorted(UC1_DF["COUNTERPARTY_RATING"].dropna().unique().tolist(), key=_rating_sort_key)
    if "COUNTERPARTY_RATING" in UC1_DF.columns and not UC1_DF.empty else []
)
def _cls_sort_key(r):
    try:
        return (0, float(r))
    except (TypeError, ValueError):
        return (1, str(r))


CLS_RATING_OPTIONS = (
    sorted(UC1_DF["CLS_GROUP_RATING"].dropna().unique().tolist(), key=_cls_sort_key)
    if "CLS_GROUP_RATING" in UC1_DF.columns and not UC1_DF.empty else []
)
POWER_CATEGORY_OPTIONS = (
    sorted(UC1_DF["POWER_CATEGORY"].dropna().unique().tolist())
    if "POWER_CATEGORY" in UC1_DF.columns and not UC1_DF.empty else []
)
CO2_BUCKET_OPTIONS = (
    sorted(UC1_DF["CO2_BUCKET"].dropna().unique().tolist(), key=_co2_bucket_low)
    if "CO2_BUCKET" in UC1_DF.columns and not UC1_DF.empty else []
)

_NEW_OBLIGOR_COUNTER = {"n": 0}


def resolve_cob_date(cob_store: dict) -> pd.Timestamp:
    """A concrete date inside the currently selected COB period, so newly added
    vehicles actually show up when the simulation is run under that period."""
    cob = cob_store or {}
    gran = cob.get("granularity", "monthly")
    period = cob.get("period") or _default_cob_period
    try:
        if gran == "monthly" and period:
            return pd.Period(period, freq="M").end_time.normalize()
        if gran == "quarterly" and period:
            year_s, q_s = period.split("-Q")
            month = {1: 3, 2: 6, 3: 9, 4: 12}[int(q_s)]
            return (pd.Timestamp(year=int(year_s), month=month, day=1) + pd.offsets.MonthEnd(0)).normalize()
        if gran == "yearly" and period:
            return pd.Timestamp(year=int(period), month=12, day=31)
    except Exception:
        pass
    return pd.Timestamp.now().normalize()


def _narrow_by_removal_filters(d: pd.DataFrame, brand, model=None, bodytype=None, power=None, co2=None) -> pd.DataFrame:
    """Apply the same brand / model / body type / power category / CO2 bucket
    filters used by the Add Vehicles wizard, but to narrow an existing population
    instead of specifying a new one. Only fields that are actually set constrain
    the result, so this works the same whether the user has picked one field or
    all of them."""
    out = d
    if brand and "BRAND_UPDATE" in out.columns:
        out = out[out["BRAND_UPDATE"] == brand]
    if model and "MARKET_MODEL" in out.columns:
        out = out[out["MARKET_MODEL"] == model]
    if bodytype and "MARKET_BODY_GROUP" in out.columns:
        out = out[out["MARKET_BODY_GROUP"] == bodytype]
    if power and "POWER_CATEGORY" in out.columns:
        out = out[out["POWER_CATEGORY"] == power]
    if co2 and "CO2_BUCKET" in out.columns:
        out = out[out["CO2_BUCKET"] == co2]
    return out


def _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store,
                     brand=None, model=None, bodytype=None, power=None, co2=None) -> pd.DataFrame:
    """Dashboard-filtered, COB-filtered, contract-deduped vehicles narrowed by
    whichever Remove Vehicles wizard fields are already picked. Shared by the
    live counter and by each step's option list, so a dropdown never offers a
    choice that would leave zero matching vehicles."""
    cob = cob_store or {}
    d = apply_filters(country, brands_f, models_f, bodytypes_f, asset_status)
    d = apply_cob_filter(d, cob.get("granularity", "monthly"), cob.get("period") or "")
    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    if keys:
        if "COB_DATE" in d.columns:
            d = d.sort_values("COB_DATE")
        d = d.drop_duplicates(subset=keys, keep="last")
    return _narrow_by_removal_filters(d, brand, model, bodytype, power, co2)


def count_matching_for_removal(country, brands_f, models_f, bodytypes_f, asset_status, cob_store,
                                brand, model=None, bodytype=None, power=None, co2=None) -> int:
    """Number of distinct vehicles (same contract-key dedup as the rest of the tool)
    that match the current dashboard filters plus the Remove Vehicles wizard's own
    brand/model/body type/power/CO2 picks. Drives the live counter shown throughout
    the wizard."""
    d = _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store,
                         brand, model, bodytype, power, co2)
    return int(len(d))


_RWIZ_FIELD_CHAIN = ["BRAND_UPDATE", "MARKET_MODEL", "MARKET_BODY_GROUP", "POWER_CATEGORY", "CO2_BUCKET"]


def _rwiz_valid_options(d: pd.DataFrame, field: str, remaining_fields: list[str]) -> list:
    """Values of `field` worth offering in a Remove Vehicles wizard dropdown: not
    just values with a matching vehicle, but values for which at least one vehicle
    also has every field still to come (`remaining_fields`) populated. Without this,
    a field could show a value whose only matching vehicles are missing one of the
    later fields, dead-ending the wizard on an empty next dropdown instead of a
    genuine "0 vehicles" message."""
    if field not in d.columns:
        return []
    needed = [field] + [f for f in remaining_fields if f in d.columns]
    valid = d.dropna(subset=needed)
    return sorted(valid[field].dropna().unique().tolist())


def apply_removal_batches(d: pd.DataFrame, removal_entries: list[dict]) -> pd.DataFrame:
    """Remove N rows per queued removal spec, matched on the exact same brand / model
    / body type / power category / CO2 bucket combination the Remove Vehicles wizard
    used to build that spec — the removal counterpart of build_synthetic_vehicles.
    Rows are picked arbitrarily among the matching set; nothing about the filters or
    metrics themselves is randomised."""
    rng = np.random.default_rng(42)
    out = d.copy()
    for entry in removal_entries:
        n = int(entry.get("qty") or 0)
        if n <= 0:
            continue
        matching = _narrow_by_removal_filters(
            out, entry.get("brand"), entry.get("model"), entry.get("body_type"),
            entry.get("power_category"), entry.get("co2_bucket"),
        )
        idx = matching.index.tolist()
        if idx:
            drop_idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
            out = out.drop(index=drop_idx)
    return out


def build_synthetic_vehicles(batch: dict, cob_date: pd.Timestamp, start_id: int) -> pd.DataFrame:
    """Expand one queued batch entry into `qty` fully-populated rows, matching the
    real dataset schema so they aggregate identically to real vehicles in any pivot."""
    qty = int(batch["qty"])
    brand = batch["brand"]
    model = batch.get("model")
    body_type = batch.get("body_type")
    tmpl = get_brand_model_template(brand, model)

    co2_mid = co2_bucket_midpoint(batch["co2_bucket"])
    ltr = float(batch.get("exposure_ltr") or 0.0)
    mtr = float(batch.get("exposure_mtr") or 0.0)
    ids = list(range(start_id, start_id + qty))

    rows = {
        "ID_CONTRACT": ids,
        "VEHICLE_ID": ids,
        "ID_QUOTATION": ids,
        "COB_DATE": [cob_date] * qty,
        "COUNTRY": [batch.get("country")] * qty,
        "NOVA_ASSET_STATUS": ["IN FLEET"] * qty,
        "BIKE_OR_CAR": ["CAR"] * qty,
        "BRAND_UPDATE": [brand] * qty,
        "MODEL": [model] * qty,
        "MARKET_MODEL": [model] * qty,
        "MARKET_BODY_GROUP": [body_type] * qty,
        "CLASS_CATALOG": [f"{model}/{model}"] * qty,
        "POWER_CATEGORY": [batch["power_category"]] * qty,
        "CO2_BUCKET": [batch["co2_bucket"]] * qty,
        "VA_CO2_EMSS_REAL": [co2_mid] * qty,
        "OBLIGOR_IDENTIFIER": [batch.get("obligor_id")] * qty,
        "ID_CUSTOMER": [batch.get("obligor_id")] * qty,
        "GROUP_RATING": [batch.get("group_rating")] * qty,
        "COUNTERPARTY_RATING": [batch.get("counterparty_rating")] * qty,
        "CLS_GROUP_RATING": [batch.get("cls_rating")] * qty,
        "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION": [batch.get("industry")] * qty,
        "SHARED_CLIENT_FLAG": [batch.get("shared_flag")] * qty,
        "VEHICLE_PRICE_EUR": [float(batch["price"])] * qty,
        "EXPOSURE_AMOUNT_LTR": [ltr] * qty,
        "EXPOSURE_AMOUNT_MTR": [mtr] * qty,
        "EXPOSURE_AMOUNT_TOT": [ltr + mtr] * qty,
        "PENDING_ORDERS": [mtr] * qty,
        "FUEL_TYPE": [tmpl["FUEL_TYPE"]] * qty,
        "FUEL_TYPE2": [tmpl["FUEL_TYPE2"]] * qty,
        "VEHICLE_CLASS": [tmpl["VEHICLE_CLASS"]] * qty,
        "OEM_UPDATE": [tmpl["OEM_UPDATE"]] * qty,
        "FINAL_CONTRACT_DURATION": [tmpl["FINAL_CONTRACT_DURATION"]] * qty,
        "CDN_CLF_SEGMENT": [tmpl["CDN_CLF_SEGMENT"]] * qty,
        "CDN_CLF_BODY_TYPE": [tmpl["CDN_CLF_BODY_TYPE"]] * qty,
    }
    df = pd.DataFrame(rows)
    df["COB_DATE"] = pd.to_datetime(df["COB_DATE"])
    df["YEAR"]  = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month
    return df


# =============================================================================
# Dash app
# =============================================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "fleet_assets"),
)
app.title = "Use Case 1 — Portfolio Heatmap"


# ── Style tokens ─────────────────────────────────────────────────────────────

MODAL_OVERLAY_STYLE = {
    "display": "none", "position": "fixed", "top": "0", "left": "0",
    "width": "100%", "height": "100%", "backgroundColor": "rgba(15, 23, 42, 0.55)",
    "zIndex": "9999", "justifyContent": "center", "alignItems": "center",
}
MODAL_PANEL_STYLE = {
    "background": "#ffffff", "borderRadius": "12px", "width": "560px",
    "maxWidth": "92vw", "maxHeight": "88vh", "overflowY": "auto",
    "padding": "28px", "boxShadow": "0 20px 60px rgba(0,0,0,0.25)",
}
FIELD_LABEL_STYLE = {
    "fontSize": "11px", "color": "#718096", "fontWeight": "700",
    "marginBottom": "5px", "display": "block",
    "textTransform": "uppercase", "letterSpacing": "0.04em",
}
FIELD_WRAP_STYLE = {"marginBottom": "16px"}
PRIMARY_BTN_STYLE = {
    "padding": "9px 20px", "fontWeight": "700", "fontSize": "13px",
    "background": "#3182ce", "color": "#fff", "border": "none",
    "borderRadius": "6px", "cursor": "pointer",
}
SECONDARY_BTN_STYLE = {
    "padding": "9px 20px", "fontWeight": "600", "fontSize": "13px",
    "background": "#f7fafc", "color": "#718096", "border": "1px solid #cbd5e0",
    "borderRadius": "6px", "cursor": "pointer",
}
DANGER_BTN_STYLE = {
    "padding": "4px 10px", "fontWeight": "700", "fontSize": "12px",
    "background": "#fff5f5", "color": "#9b2c2c", "border": "1px solid #fed7d7",
    "borderRadius": "4px", "cursor": "pointer",
}
REMOVE_PRIMARY_BTN_STYLE = {
    "padding": "9px 20px", "fontWeight": "700", "fontSize": "13px",
    "background": "#9b2c2c", "color": "#fff", "border": "none",
    "borderRadius": "6px", "cursor": "pointer",
}
NUMBER_INPUT_STYLE = {"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #cbd5e0"}

WIZ_STEP_LABELS = ["Quantity & Brand", "Model & Body Type", "Characteristics", "Customer", "Price & Review"]
REMOVE_WIZ_STEP_LABELS = ["Brand", "Model & Body Type", "Characteristics", "Quantity & Review"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _lbl(options: list[dict], value: str) -> str:
    return next((o["label"] for o in options if o["value"] == value), value)


def _fmt_val(v, sign: bool = False) -> str:
    if isinstance(v, float) and np.isnan(v):
        return ""
    prefix = "+" if sign and v > 0 else ""
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{prefix}{v / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{prefix}{v / 1_000:.1f}K"
    if isinstance(v, float):
        return f"{prefix}{v:.1f}"
    return f"{prefix}{int(v)}"


def _panel_title(text: str, **extra_style):
    style = {"fontWeight": "700", "fontSize": "13px", "color": "#1a1a2e",
              "marginBottom": "12px", "letterSpacing": "0.02em"}
    style.update(extra_style)
    return html.Div(text, style=style)


def _wiz_field(label: str, component):
    return html.Div([html.Label(label, style=FIELD_LABEL_STYLE), component], style=FIELD_WRAP_STYLE)


# ── Wizard modal ─────────────────────────────────────────────────────────────

wizard_modal = html.Div(
    id="wiz-modal",
    style=MODAL_OVERLAY_STYLE,
    children=[
        html.Div(
            style=MODAL_PANEL_STYLE,
            children=[
                html.Div(
                    [
                        html.Div("Add Vehicles", style={"fontWeight": "700", "fontSize": "17px", "color": "#1a1a2e"}),
                        html.Div(id="wiz-step-indicator"),
                    ],
                    style={"marginBottom": "20px"},
                ),

                # Step 0 — Quantity & Brand
                html.Div(
                    id="wiz-step-0",
                    children=[
                        _wiz_field("Number of vehicles", dcc.Input(
                            id="wiz-qty", type="number", min=1, step=1, value=100, style=NUMBER_INPUT_STYLE)),
                        _wiz_field("Brand", dcc.Dropdown(
                            id="wiz-brand", options=[{"label": b, "value": b} for b in _brands_avail],
                            placeholder="Select a brand", clearable=False)),
                    ],
                ),

                # Step 1 — Model & Body Type
                html.Div(
                    id="wiz-step-1", style={"display": "none"},
                    children=[
                        _wiz_field("Model", dcc.Dropdown(
                            id="wiz-model", options=[], placeholder="Select a model", clearable=False)),
                        _wiz_field("Body Type", dcc.Dropdown(
                            id="wiz-bodytype", options=[], placeholder="Select a body type", clearable=False)),
                    ],
                ),

                # Step 2 — Characteristics
                html.Div(
                    id="wiz-step-2", style={"display": "none"},
                    children=[
                        _wiz_field("Power Category", dcc.Dropdown(
                            id="wiz-power", options=[], placeholder="Select a power category", clearable=False)),
                        _wiz_field("CO2 Bucket", dcc.Dropdown(
                            id="wiz-co2", options=[], placeholder="Select a CO2 bucket", clearable=False)),
                    ],
                ),

                # Step 3 — Customer
                html.Div(
                    id="wiz-step-3", style={"display": "none"},
                    children=[
                        _wiz_field("Customer", dcc.RadioItems(
                            id="wiz-customer-mode",
                            options=[{"label": " Existing customer", "value": "existing"},
                                     {"label": " New customer",      "value": "new"}],
                            value="existing", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "13px", "cursor": "pointer"},
                        )),
                        html.Div(
                            id="wiz-existing-panel",
                            children=[
                                _wiz_field("Search Customer (Obligor ID or Industry)", dcc.Dropdown(
                                    id="wiz-customer-search", options=[], searchable=True, clearable=True,
                                    placeholder="Type to search an existing customer...")),
                                html.Div(id="wiz-existing-info"),
                            ],
                        ),
                        html.Div(
                            id="wiz-new-panel", style={"display": "none"},
                            children=[
                                _wiz_field("Industry", dcc.Dropdown(
                                    id="wiz-new-industry",
                                    options=[{"label": i, "value": i} for i in INDUSTRY_OPTIONS],
                                    placeholder="Select an industry")),
                                _wiz_field("Group Rating", dcc.Dropdown(
                                    id="wiz-new-group-rating",
                                    options=[{"label": r, "value": r} for r in GROUP_RATING_OPTIONS],
                                    placeholder="Select a group rating")),
                                _wiz_field("Counterparty Rating", dcc.Dropdown(
                                    id="wiz-new-cp-rating",
                                    options=[{"label": r, "value": r} for r in COUNTERPARTY_RATING_OPTIONS],
                                    placeholder="Select a counterparty rating")),
                                _wiz_field("CLS Rating (optional)", dcc.Dropdown(
                                    id="wiz-new-cls-rating",
                                    options=[{"label": _cls_rating_label(r), "value": r}
                                             for r in CLS_RATING_OPTIONS],
                                    placeholder="Select a CLS rating")),
                                _wiz_field("Shared Client Flag", dcc.Dropdown(
                                    id="wiz-new-shared-flag",
                                    options=[{"label": s, "value": s} for s in SHARED_FLAG_OPTIONS],
                                    placeholder="Select a shared client flag")),
                            ],
                        ),
                    ],
                ),

                # Step 4 — Price & Review
                html.Div(
                    id="wiz-step-4", style={"display": "none"},
                    children=[
                        _wiz_field("Vehicle Price (EUR)", dcc.Input(
                            id="wiz-price", type="number", min=0, step=100, style=NUMBER_INPUT_STYLE)),
                        _wiz_field("Exposure Amount LTR (EUR)", dcc.Input(
                            id="wiz-exposure-ltr", type="number", min=0, step=100, style=NUMBER_INPUT_STYLE)),
                        _wiz_field("Exposure Amount MTR (EUR)", dcc.Input(
                            id="wiz-exposure-mtr", type="number", min=0, step=100, style=NUMBER_INPUT_STYLE)),
                        html.Div("Summary", style={
                            "fontSize": "11px", "color": "#718096", "fontWeight": "700",
                            "textTransform": "uppercase", "letterSpacing": "0.04em", "margin": "18px 0 8px",
                        }),
                        html.Div(id="wiz-summary"),
                    ],
                ),

                html.Div(id="wiz-validation-msg", style={
                    "color": "#c53030", "fontSize": "12px", "fontWeight": "600",
                    "margin": "14px 0 0", "minHeight": "16px",
                }),

                html.Div(
                    [
                        html.Button("Cancel", id="wiz-btn-cancel", n_clicks=0, style=SECONDARY_BTN_STYLE),
                        html.Div(
                            [
                                html.Button("Back", id="wiz-btn-back", n_clicks=0, style=SECONDARY_BTN_STYLE),
                                html.Button("Next", id="wiz-btn-primary", n_clicks=0, style=PRIMARY_BTN_STYLE),
                            ],
                            style={"display": "flex", "gap": "10px"},
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "marginTop": "18px",
                           "borderTop": "1px solid #e2e8f0", "paddingTop": "16px"},
                ),
            ],
        ),
    ],
)


# ── Remove Vehicles wizard modal ────────────────────────────────────────────
# Mirrors wizard_modal step for step (same cascading Brand -> Model & Body Type ->
# Characteristics flow) so the two wizards behave identically. It stops one step
# short since Customer and Price don't apply to vehicles that already exist, and
# ends on a Quantity & Review step: the quantity to remove is asked last, once
# the full filter combination — and therefore the true maximum available — is
# known, instead of asking for a number before the ceiling is even computed.

remove_wizard_modal = html.Div(
    id="rwiz-modal",
    style=MODAL_OVERLAY_STYLE,
    children=[
        html.Div(
            style=MODAL_PANEL_STYLE,
            children=[
                html.Div(
                    [
                        html.Div("Remove Vehicles", style={"fontWeight": "700", "fontSize": "17px", "color": "#1a1a2e"}),
                        html.Div(id="rwiz-step-indicator"),
                    ],
                    style={"marginBottom": "12px"},
                ),

                html.Div(id="rwiz-count", style={
                    "fontSize": "12px", "color": "#4a5568", "fontWeight": "600",
                    "padding": "8px 10px", "background": "#f7fafc", "borderRadius": "6px",
                    "marginBottom": "16px",
                }),

                # Step 0 — Brand
                html.Div(
                    id="rwiz-step-0",
                    children=[
                        _wiz_field("Brand", dcc.Dropdown(
                            id="rwiz-brand", options=[{"label": b, "value": b} for b in _brands_avail],
                            placeholder="Select a brand", clearable=False)),
                    ],
                ),

                # Step 1 — Model & Body Type
                html.Div(
                    id="rwiz-step-1", style={"display": "none"},
                    children=[
                        _wiz_field("Model", dcc.Dropdown(
                            id="rwiz-model", options=[], placeholder="Select a model", clearable=False)),
                        _wiz_field("Body Type", dcc.Dropdown(
                            id="rwiz-bodytype", options=[], placeholder="Select a body type", clearable=False)),
                    ],
                ),

                # Step 2 — Characteristics
                html.Div(
                    id="rwiz-step-2", style={"display": "none"},
                    children=[
                        _wiz_field("Power Category", dcc.Dropdown(
                            id="rwiz-power", options=[], placeholder="Select a power category", clearable=False)),
                        _wiz_field("CO2 Bucket", dcc.Dropdown(
                            id="rwiz-co2", options=[], placeholder="Select a CO2 bucket", clearable=False)),
                    ],
                ),

                # Step 3 — Quantity & Review
                html.Div(
                    id="rwiz-step-3", style={"display": "none"},
                    children=[
                        _wiz_field("Number of vehicles to remove", dcc.Input(
                            id="rwiz-qty", type="number", min=1, step=1, style=NUMBER_INPUT_STYLE)),
                        html.Div("Summary", style={
                            "fontSize": "11px", "color": "#718096", "fontWeight": "700",
                            "textTransform": "uppercase", "letterSpacing": "0.04em", "margin": "0 0 8px",
                        }),
                        html.Div(id="rwiz-summary"),
                    ],
                ),

                html.Div(id="rwiz-validation-msg", style={
                    "color": "#c53030", "fontSize": "12px", "fontWeight": "600",
                    "margin": "14px 0 0", "minHeight": "16px",
                }),

                html.Div(
                    [
                        html.Button("Cancel", id="rwiz-btn-cancel", n_clicks=0, style=SECONDARY_BTN_STYLE),
                        html.Div(
                            [
                                html.Button("Back", id="rwiz-btn-back", n_clicks=0, style=SECONDARY_BTN_STYLE),
                                html.Button("Next", id="rwiz-btn-primary", n_clicks=0, style=REMOVE_PRIMARY_BTN_STYLE),
                            ],
                            style={"display": "flex", "gap": "10px"},
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "marginTop": "18px",
                           "borderTop": "1px solid #e2e8f0", "paddingTop": "16px"},
                ),
            ],
        ),
    ],
)


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Store(id="page-store",     data=0),
        dcc.Store(id="npages-store",   data=1),
        dcc.Store(id="refresh-ts",     data=0),
        dcc.Store(id="cob-store",      data={"granularity": "monthly", "period": _default_cob_period}),
        dcc.Store(id="batch-store",    data=[]),
        dcc.Store(id="wiz-step-store", data=0),
        dcc.Store(id="remove-batch-store", data=[]),
        dcc.Store(id="rwiz-step-store",    data=0),

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Use Case 1 — Portfolio Heatmap",
                                style={"margin": "0", "fontSize": "22px", "fontWeight": "700", "color": "#1a1a2e"}),
                    ]
                ),
            ],
            style={"padding": "20px 28px 16px", "borderBottom": "1px solid #e2e8f0",
                   "background": "#ffffff"},
        ),

        # ── Global Period bar ────────────────────────────────────────────────
        html.Div(
            [
                html.Span("Period",
                          style={"fontWeight": "700", "fontSize": "11px", "color": "#718096",
                                 "letterSpacing": "0.06em", "textTransform": "uppercase",
                                 "marginRight": "14px", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="cob-granularity",
                    options=[
                        {"label": "Monthly",   "value": "monthly"},
                        {"label": "Quarterly", "value": "quarterly"},
                        {"label": "Yearly",    "value": "yearly"},
                    ],
                    value="monthly",
                    clearable=False,
                    style={"width": "130px", "fontSize": "13px"},
                ),
                dcc.Dropdown(
                    id="cob-period",
                    options=[{"label": p, "value": p} for p in _cob_monthly_vals],
                    value=_default_cob_period,
                    clearable=False,
                    style={"width": "150px", "fontSize": "13px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "6px",
                   "padding": "8px 28px", "background": "#f7fafc",
                   "borderBottom": "1px solid #e2e8f0"},
        ),

        # ── Main content ─────────────────────────────────────────────────────
        html.Div(
            [
                # ── Filters panel ────────────────────────────────────────────
                html.Div(
                    [
                        _panel_title("Filters"),
                        html.Div("Country", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="f-country",
                            options=[{"label": "ALL", "value": "ALL"}] + [{"label": c, "value": c} for c in _countries_avail],
                            value="ALL", clearable=False,
                            style={"marginBottom": "12px"},
                        ),
                        html.Div("Asset Status", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="f-asset-status",
                            options=ASSET_STATUS_OPTIONS,
                            value="IN FLEET", clearable=False,
                            style={"marginBottom": "12px"},
                        ),
                        html.Div("Brand", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(id="f-brand", options=[{"label": b, "value": b} for b in _brands_avail],
                                     multi=True, placeholder="All brands", style={"marginBottom": "12px"}),
                        html.Div("Market Model", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(id="f-model", options=[{"label": m, "value": m} for m in _models_avail],
                                     multi=True, placeholder="All models", style={"marginBottom": "12px"}),
                        html.Div("Body Type", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(id="f-bodytype", options=[{"label": bt, "value": bt} for bt in _btypes_avail],
                                     multi=True, placeholder="All body types", style={"marginBottom": "20px"}),

                        html.Hr(style={"borderColor": "#e2e8f0", "margin": "0 0 16px"}),
                        _panel_title("Axes & Metric"),
                        html.Div("Y Axis", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(id="f-y", options=Y_OPTIONS, value="BRAND_UPDATE",
                                     clearable=False, style={"marginBottom": "12px"}),
                        html.Div("X Axis", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(id="f-x", options=X_OPTIONS, value="GROUP_RATING",
                                     clearable=False, style={"marginBottom": "12px"}),
                        html.Div("Metric", style={"fontSize": "11px", "color": "#718096", "fontWeight": "600", "marginBottom": "4px"}),
                        dcc.Dropdown(id="f-metric", options=METRIC_OPTIONS, value="volume",
                                     clearable=False, style={"marginBottom": "20px"}),

                        html.Button("Refresh", id="btn-refresh", n_clicks=0,
                                    style={"width": "100%", "padding": "8px", "fontWeight": "600",
                                           "fontSize": "13px", "background": "#3182ce",
                                           "color": "#fff", "border": "none", "borderRadius": "6px",
                                           "cursor": "pointer", "marginBottom": "8px"}),
                        html.Button("Simulation", id="btn-sim-toggle", n_clicks=0,
                                    style={"width": "100%", "padding": "8px", "fontWeight": "600",
                                           "fontSize": "13px", "background": "#f7fafc",
                                           "color": "#3182ce", "border": "1px solid #3182ce",
                                           "borderRadius": "6px", "cursor": "pointer"}),
                    ],
                    style={"width": "240px", "minWidth": "240px", "padding": "20px",
                           "background": "#ffffff", "borderRight": "1px solid #e2e8f0",
                           "overflowY": "auto"},
                ),

                # ── Right: heatmap + simulation ──────────────────────────────
                html.Div(
                    [
                        # Heatmap panel
                        html.Div(
                            [
                                html.Div(id="kpi-headline", style={
                                    "fontSize": "13px", "fontWeight": "700", "color": "#1a1a2e",
                                    "marginBottom": "12px", "padding": "10px 14px",
                                    "background": "#f0f7ff", "border": "1px solid #bee3f8",
                                    "borderRadius": "6px",
                                }),
                                html.Div(
                                    [
                                        html.Div(id="heatmap-title",
                                                 style={"fontWeight": "700", "fontSize": "14px",
                                                        "color": "#1a1a2e", "flex": "1"}),
                                        html.Div(
                                            [
                                                html.Button("◀", id="btn-prev", n_clicks=0,
                                                            style={"padding": "4px 12px", "fontSize": "13px",
                                                                   "background": "#edf2f7", "border": "1px solid #cbd5e0",
                                                                   "borderRadius": "4px", "cursor": "pointer"}),
                                                html.Span(id="page-info", children="Page 1 / 1",
                                                          style={"margin": "0 12px", "fontWeight": "600",
                                                                 "fontSize": "12px", "color": "#718096",
                                                                 "whiteSpace": "nowrap"}),
                                                html.Button("▶", id="btn-next", n_clicks=0,
                                                            style={"padding": "4px 12px", "fontSize": "13px",
                                                                   "background": "#edf2f7", "border": "1px solid #cbd5e0",
                                                                   "borderRadius": "4px", "cursor": "pointer"}),
                                            ],
                                            style={"display": "flex", "alignItems": "center"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "12px"},
                                ),
                                dcc.Graph(id="heatmap", config={"displayModeBar": False}),
                            ],
                            style={"background": "#ffffff", "borderRadius": "8px",
                                   "border": "1px solid #e2e8f0", "padding": "20px",
                                   "marginBottom": "16px"},
                        ),

                        # Simulation panel
                        html.Div(
                            id="sim-panel",
                            style={"display": "none"},
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                _panel_title("Add Vehicles"),
                                                html.P(
                                                    "Define a fully-specified batch of new vehicles — brand, model, "
                                                    "power category, CO2 bucket, customer, ratings and price — via a "
                                                    "guided wizard. Every field is chosen explicitly, nothing is "
                                                    "randomly sampled.",
                                                    style={"fontSize": "12px", "color": "#718096", "margin": "0 0 14px"},
                                                ),
                                                html.Button("+ Add Vehicles", id="btn-open-wizard", n_clicks=0,
                                                            style=PRIMARY_BTN_STYLE),
                                            ],
                                            style={"background": "#ffffff", "borderRadius": "8px",
                                                   "border": "1px solid #e2e8f0", "padding": "20px", "flex": "1"},
                                        ),
                                        html.Div(
                                            [
                                                _panel_title("Remove Vehicles"),
                                                html.P(
                                                    "Pick an existing group of vehicles the same way — brand, model, "
                                                    "body type, power category and CO2 bucket — via the same guided "
                                                    "wizard. The number of matching vehicles is shown live at every "
                                                    "step.",
                                                    style={"fontSize": "12px", "color": "#718096", "margin": "0 0 14px"},
                                                ),
                                                html.Button("− Remove Vehicles", id="btn-open-remove-wizard", n_clicks=0,
                                                            style=REMOVE_PRIMARY_BTN_STYLE),
                                            ],
                                            style={"background": "#ffffff", "borderRadius": "8px",
                                                   "border": "1px solid #e2e8f0", "padding": "20px", "flex": "1"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                                ),

                                html.Div(
                                    [
                                        _panel_title("Pending Batch — Vehicles to be Added"),
                                        html.Div(id="pending-batch-list", children=[
                                            html.P("No vehicles queued yet. Use “+ Add Vehicles” to define a batch.",
                                                   style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"}),
                                        ]),
                                    ],
                                    style={"background": "#ffffff", "borderRadius": "8px",
                                           "border": "1px solid #e2e8f0", "padding": "20px", "marginBottom": "16px"},
                                ),

                                html.Div(
                                    [
                                        _panel_title("Pending Removals — Vehicles to be Removed"),
                                        html.Div(id="pending-removal-list", children=[
                                            html.P("No removals queued yet. Use “− Remove Vehicles” to define one.",
                                                   style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"}),
                                        ]),
                                    ],
                                    style={"background": "#ffffff", "borderRadius": "8px",
                                           "border": "1px solid #e2e8f0", "padding": "20px", "marginBottom": "16px"},
                                ),

                                html.Div(
                                    [
                                        html.Button("Run Simulation", id="btn-sim-run", n_clicks=0,
                                                    style={"padding": "8px 20px", "fontWeight": "600",
                                                           "fontSize": "13px", "background": "#276749",
                                                           "color": "#fff", "border": "none",
                                                           "borderRadius": "6px", "cursor": "pointer"}),
                                        html.Button("Reset", id="btn-sim-reset", n_clicks=0,
                                                    style={"padding": "8px 20px", "fontWeight": "600",
                                                           "fontSize": "13px", "background": "#f7fafc",
                                                           "color": "#718096", "border": "1px solid #cbd5e0",
                                                           "borderRadius": "6px", "cursor": "pointer"}),
                                    ],
                                    style={"display": "flex", "gap": "10px", "marginBottom": "16px"},
                                ),
                                html.Div(id="sim-result"),
                            ],
                        ),
                    ],
                    style={"flex": "1", "padding": "20px", "overflowY": "auto",
                           "background": "#f7fafc"},
                ),
            ],
            style={"display": "flex", "height": "calc(100vh - 115px)", "overflow": "hidden"},
        ),

        wizard_modal,
        remove_wizard_modal,
    ],
    style={"fontFamily": "Inter, -apple-system, sans-serif", "height": "100vh",
           "display": "flex", "flexDirection": "column", "background": "#f7fafc"},
)


# =============================================================================
# Callbacks — Global COB filter
# =============================================================================

@app.callback(
    Output("cob-period", "options"),
    Output("cob-period", "value"),
    Input("cob-granularity", "value"),
)
def _cob_period_opts(gran):
    if gran == "quarterly":
        opts = [{"label": q, "value": q} for q in _cob_quarterly_vals]
        default = _cob_quarterly_vals[-1] if _cob_quarterly_vals else None
    elif gran == "yearly":
        opts = [{"label": y, "value": y} for y in _cob_yearly_vals]
        default = _cob_yearly_vals[-1] if _cob_yearly_vals else None
    else:
        opts = [{"label": p, "value": p} for p in _cob_monthly_vals]
        default = _cob_monthly_vals[-1] if _cob_monthly_vals else None
    return opts, default


@app.callback(
    Output("cob-store", "data"),
    Input("cob-granularity", "value"),
    Input("cob-period", "value"),
)
def _cob_store(gran, period):
    return {"granularity": gran or "monthly", "period": period}


# =============================================================================
# Callbacks — Filter option cascading
# =============================================================================

@app.callback(
    Output("f-brand",   "options"),
    Output("f-model",   "options"),
    Output("f-bodytype","options"),
    Input("f-country", "value"),
    Input("f-asset-status", "value"),
    Input("f-brand", "value"),
    State("cob-store",  "data"),
)
def _filter_opts(country, asset_status, brands_selected, cob_store):
    cob = cob_store or {}
    d = apply_filters(country, None, None, None, asset_status)
    d = apply_cob_filter(d, cob.get("granularity", "monthly"), cob.get("period") or "")
    brands  = [{"label": b,  "value": b}  for b  in sorted(d["BRAND_UPDATE"].dropna().unique())]   if "BRAND_UPDATE"      in d.columns else []

    d_scoped = d[d["BRAND_UPDATE"].isin(brands_selected)] if brands_selected else d
    models  = [{"label": m,  "value": m}  for m  in sorted(d_scoped["MARKET_MODEL"].dropna().unique())]   if "MARKET_MODEL"      in d_scoped.columns else []
    btypes  = [{"label": bt, "value": bt} for bt in sorted(d_scoped["MARKET_BODY_GROUP"].dropna().unique())] if "MARKET_BODY_GROUP" in d_scoped.columns else []
    return brands, models, btypes


@app.callback(
    Output("f-brand", "value"),
    Input("f-model", "value"),
    prevent_initial_call=True,
)
def _model_selects_brand(models_selected):
    if not models_selected:
        return no_update
    brands = sorted({b for m in models_selected for b in MODEL_BRAND_MAP.get(m, [])})
    return brands if brands else no_update


@app.callback(
    Output("f-y", "options"),
    Output("f-x", "options"),
    Input("f-y", "value"),
    Input("f-x", "value"),
)
def _axis_mutual_exclusion(y_val, x_val):
    y_opts = [o for o in ALL_AXIS_OPTIONS if o["value"] != x_val]
    x_opts = [o for o in ALL_AXIS_OPTIONS if o["value"] != y_val]
    return y_opts, x_opts


# =============================================================================
# Callbacks — Pagination
# =============================================================================

@app.callback(
    Output("page-store", "data"),
    Input("btn-prev",    "n_clicks"),
    Input("btn-next",    "n_clicks"),
    Input("btn-refresh", "n_clicks"),
    State("page-store",  "data"),
    State("npages-store","data"),
    prevent_initial_call=True,
)
def _update_page(prev, nxt, ref, page, n_pages):
    trig   = callback_context.triggered[0]["prop_id"]
    page   = page or 0
    n_pages = max(1, n_pages or 1)
    if "prev"    in trig: return max(0, page - 1)
    if "next"    in trig: return min(n_pages - 1, page + 1)
    return 0  # refresh → back to page 0


@app.callback(
    Output("refresh-ts", "data"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def _refresh_ts(n):
    return n


# =============================================================================
# Callbacks — Main heatmap
# =============================================================================

def _kpi_text(metric: str, total: float) -> str:
    m_lbl = _lbl(METRIC_OPTIONS, metric)
    if metric == "volume":
        return f"Total {m_lbl}: {int(total):,}"
    return f"{m_lbl} (total): {format_millions(total)} million"


@app.callback(
    Output("heatmap",       "figure"),
    Output("heatmap-title", "children"),
    Output("page-info",     "children"),
    Output("npages-store",  "data"),
    Output("kpi-headline",  "children"),
    Input("page-store",  "data"),
    Input("refresh-ts",  "data"),
    State("f-country",   "value"),
    State("f-asset-status", "value"),
    State("f-brand",     "value"),
    State("f-model",     "value"),
    State("f-bodytype",  "value"),
    State("f-y",         "value"),
    State("f-x",         "value"),
    State("f-metric",    "value"),
    State("cob-store",   "data"),
)
def _heatmap(page, _ts, country, asset_status, brands, models, bodytypes, y_col, x_col,
             metric, cob_store):
    page   = page or 0
    y_col  = y_col  or "BRAND_UPDATE"
    x_col  = x_col  or "GROUP_RATING"
    metric = metric or "volume"
    cob    = cob_store or {}

    d = apply_filters(country, brands, models, bodytypes, asset_status)
    d = apply_cob_filter(d, cob.get("granularity", "monthly"), cob.get("period") or "")

    total_value  = compute_total_metric(d, metric)
    kpi_text     = _kpi_text(metric, total_value)
    period_label = cob.get("period") or "All"
    title_text   = f"{_lbl(METRIC_OPTIONS, metric)}  ·  {_lbl(Y_OPTIONS, y_col)} × {_lbl(X_OPTIONS, x_col)}  [{period_label}]"

    piv     = build_pivot(d, y_col, x_col, metric)
    n_rows  = len(piv)
    n_pages = max(1, (n_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    page    = max(0, min(page, n_pages - 1))

    fig  = make_heatmap_fig(piv, title_text, "Blues", page, show_totals=True)
    info = f"Page {page + 1} / {n_pages}  ({n_rows} categories)"
    return fig, title_text, info, n_pages, kpi_text


# =============================================================================
# Callbacks — Simulation panel toggle
# =============================================================================

@app.callback(
    Output("sim-panel", "style"),
    Input("btn-sim-toggle", "n_clicks"),
    State("sim-panel", "style"),
    prevent_initial_call=True,
)
def _toggle_sim(_, style):
    if style and style.get("display") == "none":
        return {"display": "block"}
    return {"display": "none"}


# =============================================================================
# Callbacks — Add Vehicles wizard: cascading options
# =============================================================================

@app.callback(
    Output("wiz-model", "options"),
    Output("wiz-model", "value"),
    Output("wiz-bodytype", "options"),
    Output("wiz-bodytype", "value"),
    Input("wiz-brand", "value"),
    prevent_initial_call=True,
)
def _wiz_brand_change(brand):
    models  = BRAND_MODEL_MAP.get(brand, [])    if brand else []
    btypes  = BRAND_BODYTYPE_MAP.get(brand, []) if brand else []
    return (
        [{"label": m, "value": m} for m in models], None,
        [{"label": b, "value": b} for b in btypes], None,
    )


@app.callback(
    Output("wiz-power", "options"),
    Output("wiz-power", "value"),
    Output("wiz-co2", "options"),
    Output("wiz-co2", "value"),
    Input("wiz-brand", "value"),
    Input("wiz-model", "value"),
    prevent_initial_call=True,
)
def _wiz_characteristics_options(brand, model):
    if not brand:
        return [], None, [], None
    sub = UC1_DF[UC1_DF["BRAND_UPDATE"] == brand]
    if model and "MARKET_MODEL" in UC1_DF.columns:
        model_sub = sub[sub["MARKET_MODEL"] == model]
        if not model_sub.empty:
            sub = model_sub
    powers = sorted(sub["POWER_CATEGORY"].dropna().unique().tolist()) if "POWER_CATEGORY" in sub.columns else []
    if not powers:
        powers = POWER_CATEGORY_OPTIONS
    co2s = sorted(sub["CO2_BUCKET"].dropna().unique().tolist(), key=_co2_bucket_low) if "CO2_BUCKET" in sub.columns else []
    if not co2s:
        co2s = CO2_BUCKET_OPTIONS
    return (
        [{"label": p, "value": p} for p in powers], None,
        [{"label": c, "value": c} for c in co2s], None,
    )


@app.callback(
    Output("wiz-price", "value"),
    Output("wiz-exposure-ltr", "value"),
    Output("wiz-exposure-mtr", "value"),
    Input("wiz-brand", "value"),
    Input("wiz-model", "value"),
    prevent_initial_call=True,
)
def _wiz_price_defaults(brand, model):
    if not brand:
        return None, None, None
    price = round(median_price_for(brand, model), 2)
    return price, round(price * 0.55, 2), round(price * 0.10, 2)


@app.callback(
    Output("wiz-existing-panel", "style"),
    Output("wiz-new-panel", "style"),
    Input("wiz-customer-mode", "value"),
)
def _wiz_customer_mode(mode):
    if mode == "new":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("wiz-customer-search", "options"),
    Input("wiz-customer-search", "search_value"),
    State("wiz-customer-search", "value"),
)
def _wiz_customer_search(search_value, current_value):
    if OBLIGOR_PROFILES.empty:
        return []
    if not search_value:
        idx = list(OBLIGOR_PROFILES.index[:25])
    else:
        s = str(search_value).strip().upper()
        mask = OBLIGOR_PROFILES.index.astype(str).str.upper().str.contains(s, na=False, regex=False)
        if "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION" in OBLIGOR_PROFILES.columns:
            mask = mask | OBLIGOR_PROFILES["ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"].astype(str).str.upper().str.contains(s, na=False, regex=False)
        idx = list(OBLIGOR_PROFILES.index[mask][:25])
    if current_value and current_value not in idx:
        idx = idx + [current_value]
    opts = []
    for oid in idx:
        row = OBLIGOR_PROFILES.loc[oid]
        opts.append({
            "label": f"{oid} — {row.get('ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION')} — rating {row.get('GROUP_RATING')}",
            "value": oid,
        })
    return opts


@app.callback(
    Output("wiz-existing-info", "children"),
    Input("wiz-customer-search", "value"),
)
def _wiz_existing_info(oid):
    if not oid or oid not in OBLIGOR_PROFILES.index:
        return html.P("Select a customer to see their profile.",
                       style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"})
    row = OBLIGOR_PROFILES.loc[oid]

    def _field(label, val):
        return html.Div([
            html.Span(label, style={"fontSize": "11px", "color": "#718096", "fontWeight": "700", "display": "block"}),
            html.Span(str(val) if pd.notna(val) else "—", style={"fontSize": "13px", "color": "#1a1a2e", "fontWeight": "600"}),
        ], style={"minWidth": "140px"})

    return html.Div([
        _field("Country", row.get("COUNTRY")),
        _field("Industry", row.get("ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION")),
        _field("Group Rating", row.get("GROUP_RATING")),
        _field("Counterparty Rating", row.get("COUNTERPARTY_RATING")),
        _field("CLS Rating", row.get("CLS_GROUP_RATING")),
        _field("Shared Client Flag", row.get("SHARED_CLIENT_FLAG")),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "marginTop": "10px",
              "padding": "12px", "background": "#f7fafc", "borderRadius": "6px"})


@app.callback(
    Output("wiz-summary", "children"),
    Input("wiz-qty", "value"),
    Input("wiz-brand", "value"),
    Input("wiz-model", "value"),
    Input("wiz-bodytype", "value"),
    Input("wiz-power", "value"),
    Input("wiz-co2", "value"),
    Input("wiz-customer-mode", "value"),
    Input("wiz-customer-search", "value"),
    Input("wiz-new-industry", "value"),
    Input("wiz-new-group-rating", "value"),
    Input("wiz-new-cp-rating", "value"),
    Input("wiz-new-cls-rating", "value"),
    Input("wiz-new-shared-flag", "value"),
    Input("wiz-price", "value"),
)
def _wiz_summary(qty, brand, model, bodytype, power, co2, cust_mode, cust_search,
                  new_ind, new_gr, new_cr, new_cls, new_sf, price):
    if not brand:
        return html.P("Fill in the previous steps to see a summary here.",
                       style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"})

    if cust_mode == "existing":
        if cust_search and cust_search in OBLIGOR_PROFILES.index:
            row = OBLIGOR_PROFILES.loc[cust_search]
            industry, gr, cr, cls_r, sf = (
                row.get("ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"), row.get("GROUP_RATING"),
                row.get("COUNTERPARTY_RATING"), row.get("CLS_GROUP_RATING"), row.get("SHARED_CLIENT_FLAG"),
            )
            cust_label = f"existing customer {cust_search}"
        else:
            industry = gr = cr = cls_r = sf = None
            cust_label = "existing customer (not selected yet)"
    else:
        industry, gr, cr, cls_r, sf = new_ind, new_gr, new_cr, new_cls, new_sf
        cust_label = "new customer"

    rating_txt = f"{gr or '—'} / {cr or '—'}"
    if cls_r not in (None, ""):
        rating_txt += f" / CLS {cls_r}"

    parts = [
        f"Power: {power or '—'}",
        f"CO2 bucket: {co2 or '—'}",
        f"Industry: {industry or '—'}",
        f"Rating: {rating_txt}",
        f"Shared flag: {sf or '—'}",
        cust_label,
        f"€{float(price):,.0f}/vehicle" if price else "price not set",
    ]
    headline = f"{int(qty) if qty else 0} × {brand}{' ' + model if model else ''}{' (' + bodytype + ')' if bodytype else ''}"
    return html.Div([
        html.Div(headline, style={"fontWeight": "700", "fontSize": "14px", "color": "#1a1a2e", "marginBottom": "6px"}),
        html.Div(" · ".join(parts), style={"fontSize": "12px", "color": "#4a5568"}),
    ], style={"padding": "14px", "background": "#f0fff4", "border": "1px solid #c6f6d5", "borderRadius": "6px"})


# =============================================================================
# Callbacks — Add Vehicles wizard: step visibility / navigation controller
# =============================================================================

@app.callback(
    Output("wiz-step-0", "style"),
    Output("wiz-step-1", "style"),
    Output("wiz-step-2", "style"),
    Output("wiz-step-3", "style"),
    Output("wiz-step-4", "style"),
    Output("wiz-step-indicator", "children"),
    Output("wiz-btn-primary", "children"),
    Output("wiz-btn-back", "style"),
    Input("wiz-step-store", "data"),
)
def _wiz_step_visibility(step):
    step = step or 0
    styles = [{"display": "none"}] * 5
    styles[step] = {"display": "block"}

    dots = []
    for i, label in enumerate(WIZ_STEP_LABELS):
        active = i == step
        done = i < step
        color = "#3182ce" if active else ("#276749" if done else "#a0aec0")
        dots.append(html.Span(
            f"{i + 1}. {label}",
            style={"fontSize": "11px", "fontWeight": "700" if active else "500",
                   "color": color, "marginRight": "14px"},
        ))
    indicator = html.Div(dots, style={"display": "flex", "flexWrap": "wrap", "marginTop": "6px"})

    primary_label = "Add to Batch" if step == 4 else "Next"
    back_style = dict(SECONDARY_BTN_STYLE)
    if step == 0:
        back_style["visibility"] = "hidden"
    return (*styles, indicator, primary_label, back_style)


@app.callback(
    Output("wiz-modal", "style"),
    Output("wiz-step-store", "data"),
    Output("batch-store", "data"),
    Output("wiz-validation-msg", "children"),
    Output("wiz-qty", "value"),
    Output("wiz-brand", "value"),
    Output("wiz-customer-mode", "value"),
    Output("wiz-customer-search", "value"),
    Output("wiz-new-industry", "value"),
    Output("wiz-new-group-rating", "value"),
    Output("wiz-new-cp-rating", "value"),
    Output("wiz-new-cls-rating", "value"),
    Output("wiz-new-shared-flag", "value"),
    Input("btn-open-wizard", "n_clicks"),
    Input("wiz-btn-back", "n_clicks"),
    Input("wiz-btn-primary", "n_clicks"),
    Input("wiz-btn-cancel", "n_clicks"),
    State("wiz-step-store", "data"),
    State("wiz-qty", "value"),
    State("wiz-brand", "value"),
    State("wiz-model", "value"),
    State("wiz-bodytype", "value"),
    State("wiz-power", "value"),
    State("wiz-co2", "value"),
    State("wiz-customer-mode", "value"),
    State("wiz-customer-search", "value"),
    State("wiz-new-industry", "value"),
    State("wiz-new-group-rating", "value"),
    State("wiz-new-cp-rating", "value"),
    State("wiz-new-cls-rating", "value"),
    State("wiz-new-shared-flag", "value"),
    State("wiz-price", "value"),
    State("wiz-exposure-ltr", "value"),
    State("wiz-exposure-mtr", "value"),
    State("f-country", "value"),
    State("batch-store", "data"),
    prevent_initial_call=True,
)
def _wiz_controller(open_n, back_n, primary_n, cancel_n, step,
                     qty, brand, model, bodytype, power, co2,
                     cust_mode, cust_search, new_ind, new_gr, new_cr, new_cls, new_sf,
                     price, ltr, mtr, f_country, batch_data):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    step = step or 0
    batch_data = batch_data or []

    no_reset = (no_update,) * 9
    RESET_VALUES = (100, None, "existing", None, None, None, None, None, None)

    if "btn-open-wizard" in trig:
        return ({**MODAL_OVERLAY_STYLE, "display": "flex"}, 0, no_update, "", *RESET_VALUES)

    if "wiz-btn-cancel" in trig:
        return ({**MODAL_OVERLAY_STYLE, "display": "none"}, no_update, no_update, "", *no_reset)

    if "wiz-btn-back" in trig:
        return (no_update, max(0, step - 1), no_update, "", *no_reset)

    if "wiz-btn-primary" in trig:
        if step == 0:
            if not qty or int(qty) <= 0 or not brand:
                return (no_update, step, no_update,
                         "Please enter a quantity greater than 0 and select a brand.", *no_reset)
            return (no_update, 1, no_update, "", *no_reset)

        if step == 1:
            if not model or not bodytype:
                return (no_update, step, no_update, "Please select a model and a body type.", *no_reset)
            return (no_update, 2, no_update, "", *no_reset)

        if step == 2:
            if not power or not co2:
                return (no_update, step, no_update, "Please select a power category and a CO2 bucket.", *no_reset)
            return (no_update, 3, no_update, "", *no_reset)

        if step == 3:
            if cust_mode == "existing":
                if not cust_search or cust_search not in OBLIGOR_PROFILES.index:
                    return (no_update, step, no_update, "Please search and select an existing customer.", *no_reset)
            else:
                if not new_ind or not new_gr or not new_cr or not new_sf:
                    return (no_update, step, no_update,
                             "Please fill in industry, ratings and shared client flag for the new customer.", *no_reset)
            return (no_update, 4, no_update, "", *no_reset)

        if step == 4:
            if not price or float(price) <= 0:
                return (no_update, step, no_update, "Please enter a vehicle price greater than 0.", *no_reset)

            ltr_v = float(ltr) if ltr not in (None, "") else 0.0
            mtr_v = float(mtr) if mtr not in (None, "") else 0.0

            if cust_mode == "existing":
                row = OBLIGOR_PROFILES.loc[cust_search]
                obligor_id   = cust_search
                industry     = row.get("ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION")
                group_rating = row.get("GROUP_RATING")
                cp_rating    = row.get("COUNTERPARTY_RATING")
                cls_rating   = row.get("CLS_GROUP_RATING")
                shared_flag  = row.get("SHARED_CLIENT_FLAG")
                country      = row.get("COUNTRY")
                cust_label   = f"existing customer {obligor_id}"
            else:
                _NEW_OBLIGOR_COUNTER["n"] += 1
                base_country = f_country if f_country and f_country != "ALL" else (_countries_avail[0] if _countries_avail else "SPAIN")
                obligor_id   = f"NEW-{str(base_country)[:3]}-{_NEW_OBLIGOR_COUNTER['n']:05d}"
                industry, group_rating, cp_rating, shared_flag = new_ind, new_gr, new_cr, new_sf
                cls_rating   = new_cls if new_cls not in (None, "") else None
                country      = base_country
                cust_label   = f"new customer {obligor_id}"

            cls_display = f" / CLS {cls_rating}" if cls_rating not in (None, "") else ""
            summary = (
                f"{int(qty)} × {brand} {model} ({bodytype}) — {power}, CO2 {co2}, "
                f"{industry}, rating {group_rating}/{cp_rating}{cls_display}, "
                f"{cust_label}, €{float(price):,.0f}/vehicle"
            )
            entry = {
                "qty": int(qty), "brand": brand, "model": model, "body_type": bodytype,
                "power_category": power, "co2_bucket": co2,
                "obligor_id": obligor_id, "industry": industry,
                "group_rating": group_rating, "counterparty_rating": cp_rating,
                "cls_rating": cls_rating, "shared_flag": shared_flag, "country": country,
                "price": float(price), "exposure_ltr": ltr_v, "exposure_mtr": mtr_v,
                "summary": summary,
            }
            new_batch = batch_data + [entry]
            return ({**MODAL_OVERLAY_STYLE, "display": "none"}, 0, new_batch, "", *RESET_VALUES)

    return (no_update,) * 13


# =============================================================================
# Callbacks — Pending batch list
# =============================================================================

@app.callback(
    Output("pending-batch-list", "children"),
    Input("batch-store", "data"),
)
def _render_pending_batch(batch_data):
    batch_data = batch_data or []
    if not batch_data:
        return html.P("No vehicles queued yet. Use “+ Add Vehicles” to define a batch.",
                       style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"})
    rows = []
    for i, entry in enumerate(batch_data):
        rows.append(html.Div(
            [
                html.Div(entry["summary"], style={"fontSize": "13px", "color": "#1a1a2e", "flex": "1"}),
                html.Button("Remove", id={"type": "batch-remove-btn", "index": i}, n_clicks=0, style=DANGER_BTN_STYLE),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "12px", "padding": "10px 12px",
                   "background": "#f7fafc", "borderRadius": "6px", "marginBottom": "8px",
                   "border": "1px solid #e2e8f0"},
        ))
    return rows


@app.callback(
    Output("batch-store", "data", allow_duplicate=True),
    Input({"type": "batch-remove-btn", "index": ALL}, "n_clicks"),
    State("batch-store", "data"),
    prevent_initial_call=True,
)
def _remove_batch_item(n_clicks_list, batch_data):
    if not callback_context.triggered or not any(n_clicks_list or []):
        return no_update
    prop_id = callback_context.triggered[0]["prop_id"].rsplit(".", 1)[0]
    try:
        idx = json.loads(prop_id)["index"]
    except Exception:
        return no_update
    batch_data = batch_data or []
    return [e for i, e in enumerate(batch_data) if i != idx]


# =============================================================================
# Callbacks — Remove Vehicles wizard: cascading options
# Each dropdown is scoped to the current dashboard filters/COB period plus every
# field already picked earlier in the wizard, so it only ever offers choices that
# leave at least one matching vehicle — never a combination showing "0 vehicles".
# =============================================================================

@app.callback(
    Output("rwiz-brand", "options"),
    Input("btn-open-remove-wizard", "n_clicks"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_brand_options(_open, country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    d = _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store)
    brands = _rwiz_valid_options(d, "BRAND_UPDATE", _RWIZ_FIELD_CHAIN[1:])
    return [{"label": b, "value": b} for b in brands]


@app.callback(
    Output("rwiz-model", "options"),
    Output("rwiz-model", "value"),
    Input("rwiz-brand", "value"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_model_options(brand, country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    if not brand:
        return [], None
    d = _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store, brand)
    models = _rwiz_valid_options(d, "MARKET_MODEL", _RWIZ_FIELD_CHAIN[2:])
    return [{"label": m, "value": m} for m in models], None


@app.callback(
    Output("rwiz-bodytype", "options"),
    Output("rwiz-bodytype", "value"),
    Input("rwiz-brand", "value"),
    Input("rwiz-model", "value"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_bodytype_options(brand, model, country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    if not brand:
        return [], None
    d = _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store, brand, model)
    btypes = _rwiz_valid_options(d, "MARKET_BODY_GROUP", _RWIZ_FIELD_CHAIN[3:])
    return [{"label": b, "value": b} for b in btypes], None


@app.callback(
    Output("rwiz-power", "options"),
    Output("rwiz-power", "value"),
    Input("rwiz-brand", "value"),
    Input("rwiz-model", "value"),
    Input("rwiz-bodytype", "value"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_power_options(brand, model, bodytype, country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    if not brand:
        return [], None
    d = _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store, brand, model, bodytype)
    powers = _rwiz_valid_options(d, "POWER_CATEGORY", _RWIZ_FIELD_CHAIN[4:])
    return [{"label": p, "value": p} for p in powers], None


@app.callback(
    Output("rwiz-co2", "options"),
    Output("rwiz-co2", "value"),
    Input("rwiz-brand", "value"),
    Input("rwiz-model", "value"),
    Input("rwiz-bodytype", "value"),
    Input("rwiz-power", "value"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_co2_options(brand, model, bodytype, power, country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    if not brand:
        return [], None
    d = _rwiz_scoped_df(country, brands_f, models_f, bodytypes_f, asset_status, cob_store, brand, model, bodytype, power)
    co2s = sorted(d["CO2_BUCKET"].dropna().unique().tolist(), key=_co2_bucket_low) if "CO2_BUCKET" in d.columns else []
    return [{"label": c, "value": c} for c in co2s], None


@app.callback(
    Output("rwiz-count", "children"),
    Input("rwiz-brand", "value"),
    Input("rwiz-model", "value"),
    Input("rwiz-bodytype", "value"),
    Input("rwiz-power", "value"),
    Input("rwiz-co2", "value"),
    Input("btn-open-remove-wizard", "n_clicks"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
)
def _rwiz_count(brand, model, bodytype, power, co2, _open,
                country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    if not brand:
        return "Select a brand to see how many vehicles match your selection."
    n = count_matching_for_removal(country, brands_f, models_f, bodytypes_f, asset_status, cob_store,
                                    brand, model, bodytype, power, co2)
    return f"{n:,} vehicle{'s' if n != 1 else ''} match this selection."


@app.callback(
    Output("rwiz-qty", "value", allow_duplicate=True),
    Output("rwiz-qty", "max"),
    Input("rwiz-brand", "value"),
    Input("rwiz-model", "value"),
    Input("rwiz-bodytype", "value"),
    Input("rwiz-power", "value"),
    Input("rwiz-co2", "value"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_qty_bounds(brand, model, bodytype, power, co2,
                      country, asset_status, brands_f, models_f, bodytypes_f, cob_store):
    """The quantity field is only asked once the full filter combination is set —
    it defaults to (and is capped at) the number of vehicles actually available for
    that combination, so the user can never type more than what exists."""
    if not brand:
        return None, None
    available = count_matching_for_removal(country, brands_f, models_f, bodytypes_f, asset_status, cob_store,
                                            brand, model, bodytype, power, co2)
    return (available, available) if available > 0 else (None, None)


# =============================================================================
# Callbacks — Remove Vehicles wizard: step visibility / navigation controller
# =============================================================================

@app.callback(
    Output("rwiz-step-0", "style"),
    Output("rwiz-step-1", "style"),
    Output("rwiz-step-2", "style"),
    Output("rwiz-step-3", "style"),
    Output("rwiz-step-indicator", "children"),
    Output("rwiz-btn-primary", "children"),
    Output("rwiz-btn-back", "style"),
    Input("rwiz-step-store", "data"),
)
def _rwiz_step_visibility(step):
    step = step or 0
    styles = [{"display": "none"}] * 4
    styles[step] = {"display": "block"}

    dots = []
    for i, label in enumerate(REMOVE_WIZ_STEP_LABELS):
        active = i == step
        done = i < step
        color = "#c53030" if active else ("#276749" if done else "#a0aec0")
        dots.append(html.Span(
            f"{i + 1}. {label}",
            style={"fontSize": "11px", "fontWeight": "700" if active else "500",
                   "color": color, "marginRight": "14px"},
        ))
    indicator = html.Div(dots, style={"display": "flex", "flexWrap": "wrap", "marginTop": "6px"})

    primary_label = "Remove" if step == 3 else "Next"
    back_style = dict(SECONDARY_BTN_STYLE)
    if step == 0:
        back_style["visibility"] = "hidden"
    return (*styles, indicator, primary_label, back_style)


@app.callback(
    Output("rwiz-modal", "style"),
    Output("rwiz-step-store", "data"),
    Output("remove-batch-store", "data"),
    Output("rwiz-validation-msg", "children"),
    Output("rwiz-qty", "value"),
    Output("rwiz-brand", "value"),
    Input("btn-open-remove-wizard", "n_clicks"),
    Input("rwiz-btn-back", "n_clicks"),
    Input("rwiz-btn-primary", "n_clicks"),
    Input("rwiz-btn-cancel", "n_clicks"),
    State("rwiz-step-store", "data"),
    State("rwiz-qty", "value"),
    State("rwiz-brand", "value"),
    State("rwiz-model", "value"),
    State("rwiz-bodytype", "value"),
    State("rwiz-power", "value"),
    State("rwiz-co2", "value"),
    State("f-country", "value"),
    State("f-asset-status", "value"),
    State("f-brand", "value"),
    State("f-model", "value"),
    State("f-bodytype", "value"),
    State("cob-store", "data"),
    State("remove-batch-store", "data"),
    prevent_initial_call=True,
)
def _rwiz_controller(open_n, back_n, primary_n, cancel_n, step,
                      qty, brand, model, bodytype, power, co2,
                      country, asset_status, brands_f, models_f, bodytypes_f, cob_store, removal_data):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    step = step or 0
    removal_data = removal_data or []

    no_reset = (no_update, no_update)
    RESET_VALUES = (None, None)

    if "btn-open-remove-wizard" in trig:
        return ({**MODAL_OVERLAY_STYLE, "display": "flex"}, 0, no_update, "", *RESET_VALUES)

    if "rwiz-btn-cancel" in trig:
        return ({**MODAL_OVERLAY_STYLE, "display": "none"}, no_update, no_update, "", *no_reset)

    if "rwiz-btn-back" in trig:
        return (no_update, max(0, step - 1), no_update, "", *no_reset)

    if "rwiz-btn-primary" in trig:
        if step == 0:
            if not brand:
                return (no_update, step, no_update, "Please select a brand.", *no_reset)
            return (no_update, 1, no_update, "", *no_reset)

        if step == 1:
            if not model or not bodytype:
                return (no_update, step, no_update, "Please select a model and a body type.", *no_reset)
            return (no_update, 2, no_update, "", *no_reset)

        if step == 2:
            if not power or not co2:
                return (no_update, step, no_update, "Please select a power category and a CO2 bucket.", *no_reset)
            return (no_update, 3, no_update, "", *no_reset)

        if step == 3:
            if not qty or int(qty) <= 0:
                return (no_update, step, no_update, "Please enter a quantity greater than 0.", *no_reset)
            available = count_matching_for_removal(country, brands_f, models_f, bodytypes_f, asset_status, cob_store,
                                                     brand, model, bodytype, power, co2)
            if int(qty) > available:
                return (no_update, step, no_update,
                         f"Only {available:,} vehicle{'s' if available != 1 else ''} match this selection.", *no_reset)

            summary = f"−{int(qty)} × {brand} {model} ({bodytype}) — {power}, CO2 {co2}"
            entry = {
                "qty": int(qty), "brand": brand, "model": model, "body_type": bodytype,
                "power_category": power, "co2_bucket": co2, "summary": summary,
            }
            new_removals = removal_data + [entry]
            return ({**MODAL_OVERLAY_STYLE, "display": "none"}, 0, new_removals, "", *RESET_VALUES)

    return (no_update,) * 6


@app.callback(
    Output("rwiz-summary", "children"),
    Input("rwiz-qty", "value"),
    Input("rwiz-brand", "value"),
    Input("rwiz-model", "value"),
    Input("rwiz-bodytype", "value"),
    Input("rwiz-power", "value"),
    Input("rwiz-co2", "value"),
)
def _rwiz_summary(qty, brand, model, bodytype, power, co2):
    if not brand:
        return html.P("Fill in the previous steps to see a summary here.",
                       style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"})
    headline = f"{int(qty) if qty else 0} × {brand}{' ' + model if model else ''}{' (' + bodytype + ')' if bodytype else ''}"
    parts = [f"Power: {power or '—'}", f"CO2 bucket: {co2 or '—'}"]
    return html.Div([
        html.Div(headline, style={"fontWeight": "700", "fontSize": "14px", "color": "#1a1a2e", "marginBottom": "6px"}),
        html.Div(" · ".join(parts), style={"fontSize": "12px", "color": "#4a5568"}),
    ], style={"padding": "14px", "background": "#fff5f5", "border": "1px solid #fed7d7", "borderRadius": "6px"})


# =============================================================================
# Callbacks — Pending removals list
# =============================================================================

@app.callback(
    Output("pending-removal-list", "children"),
    Input("remove-batch-store", "data"),
)
def _render_pending_removals(removal_data):
    removal_data = removal_data or []
    if not removal_data:
        return html.P("No removals queued yet. Use “− Remove Vehicles” to define one.",
                       style={"fontSize": "12px", "color": "#a0aec0", "fontStyle": "italic"})
    rows = []
    for i, entry in enumerate(removal_data):
        rows.append(html.Div(
            [
                html.Div(entry["summary"], style={"fontSize": "13px", "color": "#1a1a2e", "flex": "1"}),
                html.Button("Remove", id={"type": "removal-remove-btn", "index": i}, n_clicks=0, style=DANGER_BTN_STYLE),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "12px", "padding": "10px 12px",
                   "background": "#f7fafc", "borderRadius": "6px", "marginBottom": "8px",
                   "border": "1px solid #e2e8f0"},
        ))
    return rows


@app.callback(
    Output("remove-batch-store", "data", allow_duplicate=True),
    Input({"type": "removal-remove-btn", "index": ALL}, "n_clicks"),
    State("remove-batch-store", "data"),
    prevent_initial_call=True,
)
def _remove_removal_item(n_clicks_list, removal_data):
    if not callback_context.triggered or not any(n_clicks_list or []):
        return no_update
    prop_id = callback_context.triggered[0]["prop_id"].rsplit(".", 1)[0]
    try:
        idx = json.loads(prop_id)["index"]
    except Exception:
        return no_update
    removal_data = removal_data or []
    return [e for i, e in enumerate(removal_data) if i != idx]


# =============================================================================
# Callbacks — Run / Reset simulation
# =============================================================================

@app.callback(
    Output("sim-result", "children"),
    Output("batch-store", "data", allow_duplicate=True),
    Output("remove-batch-store", "data", allow_duplicate=True),
    Input("btn-sim-run",   "n_clicks"),
    Input("btn-sim-reset", "n_clicks"),
    State("batch-store",  "data"),
    State("remove-batch-store", "data"),
    State("f-country",    "value"),
    State("f-asset-status", "value"),
    State("f-brand",      "value"),
    State("f-model",      "value"),
    State("f-bodytype",   "value"),
    State("f-y",          "value"),
    State("f-x",          "value"),
    State("f-metric",     "value"),
    State("page-store",   "data"),
    State("cob-store",    "data"),
    prevent_initial_call=True,
)
def _sim_result(run, reset, batch_data, removal_data,
                country, asset_status, brands, models, bodytypes, y_col, x_col, metric,
                page, cob_store):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    if "reset" in trig:
        return [], [], []

    batch_data = batch_data or []
    removal_data = removal_data or []

    y_col  = y_col  or "BRAND_UPDATE"
    x_col  = x_col  or "GROUP_RATING"
    metric = metric or "volume"
    page   = page or 0
    cob    = cob_store or {}

    d_orig = apply_filters(country, brands, models, bodytypes, asset_status)
    d_orig = apply_cob_filter(d_orig, cob.get("granularity", "monthly"), cob.get("period") or "")

    if d_orig.empty and not batch_data:
        return html.P("No data.", style={"color": "#999", "padding": "20px", "textAlign": "center"}), no_update, no_update

    d_sim = apply_removal_batches(d_orig, removal_data)

    cob_date = resolve_cob_date(cob)
    if "ID_CONTRACT" in UC1_DF.columns and not UC1_DF.empty:
        start_id = int(pd.to_numeric(UC1_DF["ID_CONTRACT"], errors="coerce").max() or 0) + 1
    else:
        start_id = 900000

    synth_parts = []
    for entry in batch_data:
        part = build_synthetic_vehicles(entry, cob_date, start_id)
        start_id += entry["qty"]
        synth_parts.append(part)
    if synth_parts:
        d_sim = pd.concat([d_sim] + synth_parts, ignore_index=True)

    piv_orig  = build_pivot(d_orig, y_col, x_col, metric)
    piv_sim   = build_pivot(d_sim,  y_col, x_col, metric)
    all_cols  = _ordered_axis(set(piv_orig.columns) | set(piv_sim.columns), x_col, is_y=False)
    all_idx   = _ordered_axis(set(piv_orig.index)   | set(piv_sim.index),   y_col, is_y=True)
    piv_orig  = piv_orig.reindex(index=all_idx, columns=all_cols, fill_value=0)
    piv_sim   = piv_sim.reindex( index=all_idx, columns=all_cols, fill_value=0)
    piv_delta = piv_sim - piv_orig

    m_lbl     = _lbl(METRIC_OPTIONS, metric)
    fig_orig  = make_heatmap_fig(piv_orig,  f"Original — {m_lbl}",  "Blues", page, show_totals=True)
    fig_sim   = make_heatmap_fig(piv_sim,   f"Simulated — {m_lbl}", "BuGn",  page, show_totals=True)

    paged_d = piv_delta.iloc[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]
    if not paged_d.empty:
        # Normalize color contrast to what's actually visible on this page — using the
        # full matrix's max here would wash out real on-page changes when a much larger
        # change exists elsewhere (off-page).
        abs_max = max(abs(float(paged_d.values.max())), abs(float(paged_d.values.min())), 1.0)
        text_d  = [[_fmt_val(v, sign=True) for v in row] for row in paged_d.values]
        fig_delta = go.Figure(go.Heatmap(
            z=paged_d.values.tolist(),
            x=[str(c) for c in paged_d.columns],
            y=[str(r) for r in paged_d.index],
            text=text_d, texttemplate="%{text}",
            colorscale="RdYlGn", zmid=0, zmin=-abs_max, zmax=abs_max,
            hoverongaps=False,
            hovertemplate="%{y} × %{x}: %{z:+.1f}<extra></extra>",
        ))
        fig_delta.update_layout(
            title={"text": f"Delta (Simulated − Original) — {m_lbl}",
                   "font": {"size": 13, "color": "#1a1a2e"}, "x": 0.01},
            margin={"l": 10, "r": 10, "t": 50, "b": 85},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif", "size": 11},
            height=max(300, 60 + 28 * len(paged_d)),
            xaxis={"tickangle": -40, "tickfont": {"size": 10}},
            yaxis={"tickfont": {"size": 10}},
        )
    else:
        fig_delta = go.Figure()
        fig_delta.add_annotation(text="No delta", showarrow=False)
        fig_delta.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)")

    CARD = {"background": "#ffffff", "borderRadius": "8px",
            "border": "1px solid #e2e8f0", "padding": "16px",
            "flex": "1", "minWidth": "0"}

    caption_parts = [e["summary"] for e in batch_data]
    removed_parts = [e["summary"] for e in removal_data]
    caption_bits  = caption_parts + removed_parts
    caption = " | ".join(caption_bits) if caption_bits else "No changes queued — showing current portfolio only."

    result = html.Div([
        html.Div("Simulation Results",
                 style={"fontWeight": "700", "fontSize": "14px", "color": "#1a1a2e",
                        "margin": "0 0 6px"}),
        html.Div(caption, style={"fontSize": "12px", "color": "#4a5568", "margin": "0 0 12px"}),
        html.Div(
            [
                html.Div([dcc.Graph(figure=fig_orig,  config={"displayModeBar": False})], style=CARD),
                html.Div([dcc.Graph(figure=fig_sim,   config={"displayModeBar": False})], style=CARD),
            ],
            style={"display": "flex", "gap": "14px", "marginBottom": "14px"},
        ),
        html.Div(
            [dcc.Graph(figure=fig_delta, config={"displayModeBar": False})],
            style={**CARD, "flex": "unset"},
        ),
    ])
    return result, no_update, no_update


# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8051)
