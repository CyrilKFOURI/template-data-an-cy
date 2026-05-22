
from __future__ import annotations

import os
import sys
import calendar
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots as _make_subplots
from dash import Dash, Input, Output, State, dcc, html, no_update
import dash

# ── Paths ─────────────────────────────────────────────────────────────────────
FAST_DIR  = r"C:\Users\cyril\Documents\Internships\Arval\New\precomputed_fast"
BASE_DATA = os.path.join(FAST_DIR, "data")

# ── Skip raw data loading in fmd – read only precomputed parquets ──────────────
os.environ["FAST_READER_MODE"] = "1"

sys.path.insert(0, os.path.dirname(FAST_DIR))
import fleet_monitoring_dashboard_key_refactor as fmd

# =============================================================================
# 1.  LOAD ALL PRECOMPUTED DATA AT STARTUP
# =============================================================================

def _read_parquet_tree(root: str, filename_suffix: str) -> pd.DataFrame:
    """Recursively read all Parquet files matching a suffix under root."""
    dfs: list[pd.DataFrame] = []
    if not os.path.exists(root):
        return pd.DataFrame()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(filename_suffix):
                try:
                    dfs.append(pd.read_parquet(os.path.join(dirpath, f)))
                except Exception:
                    pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _normalise_cache(df: pd.DataFrame, cache_type: str) -> pd.DataFrame:
    """Rename columns from old lowercase/underscore format to new UPPERCASE format.

    Works with both old-style parquets (pre-generator rewrite) and new-style.
    Rename rules per cache type:
      v1  : country→COUNTRY, year→YEAR, status→ASSET_STATUS, date_mode→DATE_MODE, kpi1→KPI1 …
      kpi7: _country→COUNTRY, _status→ASSET_STATUS, _metric_mode→METRIC_MODE …
      kpi8: same underscore-prefixed mapping plus _date_mode→DATE_MODE
    """
    if df.empty:
        return df

    renames = {}
    if cache_type == "v1":
        pairs = [
            ("country", "COUNTRY"), ("year", "YEAR"), ("month", "MONTH"),
            ("bike_or_car", "BIKE_OR_CAR"), ("status", "ASSET_STATUS"),
            ("date_mode", "DATE_MODE"), ("kpi1", "KPI1"), ("kpi2", "KPI2"),
            ("diesel", "DIESEL"), ("non_diesel", "NON_DIESEL"),
            ("hybrid", "HYBRID"), ("ev", "EV"), ("pv", "PV"),
            ("lcv", "LCV"), ("volume", "VOLUME"),
        ]
    elif cache_type == "kpi7":
        pairs = [
            ("_country", "COUNTRY"), ("_status", "ASSET_STATUS"),
            ("_metric_mode", "METRIC_MODE"), ("_period_mode", "PERIOD_MODE"),
            ("_bike_or_car", "BIKE_OR_CAR"),
        ]
    elif cache_type == "kpi8":
        pairs = [
            ("_country", "COUNTRY"), ("_status", "ASSET_STATUS"),
            ("_metric_mode", "METRIC_MODE"), ("_bike_or_car", "BIKE_OR_CAR"),
            ("_date_mode", "DATE_MODE"), ("_period_mode", "PERIOD_MODE"),
        ]
    else:
        pairs = []

    for old, new in pairs:
        if old in df.columns and new not in df.columns:
            renames[old] = new

    return df.rename(columns=renames) if renames else df


def _load_cache(merged_path: str, tree_root: str, suffix: str) -> pd.DataFrame:
    """Load from a pre-merged parquet if it exists, otherwise walk the tree."""
    if os.path.exists(merged_path):
        try:
            return pd.read_parquet(merged_path)
        except Exception:
            pass
    return _read_parquet_tree(tree_root, suffix)


_raw_v1   = _load_cache(os.path.join(BASE_DATA, "merged_v1.parquet"),   os.path.join(BASE_DATA, "view1", "kpis"), ".parquet")
_raw_kpi7 = _load_cache(os.path.join(BASE_DATA, "merged_kpi7.parquet"), os.path.join(BASE_DATA, "view1", "kpi7"), ".parquet")
_raw_kpi8 = _load_cache(os.path.join(BASE_DATA, "merged_kpi8.parquet"), os.path.join(BASE_DATA, "view2", "kpi8"), ".parquet")

CACHE_V1   = _normalise_cache(_raw_v1,   "v1")
CACHE_KPI7 = _normalise_cache(_raw_kpi7, "kpi7")
CACHE_KPI8 = _normalise_cache(_raw_kpi8, "kpi8")

# Ensure YEAR is int where present
for _cache in (CACHE_V1, CACHE_KPI7, CACHE_KPI8):
    if not _cache.empty and "YEAR" in _cache.columns:
        _cache["YEAR"] = _cache["YEAR"].astype(int)


def _load_flat_parquet(path: str) -> pd.DataFrame:
    """Load a single flat parquet file (not partitioned)."""
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


CACHE_KPI91 = _load_flat_parquet(os.path.join(BASE_DATA, "view3", "kpi9_1.parquet"))
CACHE_KPI92 = _load_flat_parquet(os.path.join(BASE_DATA, "view3", "kpi9_2.parquet"))
CACHE_KPI10 = _load_flat_parquet(os.path.join(BASE_DATA, "view4", "kpi10.parquet"))
CACHE_KPI11 = _load_flat_parquet(os.path.join(BASE_DATA, "view5", "kpi11.parquet"))
CACHE_KPI13 = _load_flat_parquet(os.path.join(BASE_DATA, "view6", "kpi13.parquet"))


# =============================================================================
# 2.  LOOKUP HELPERS
# =============================================================================

def _normalise_month(month_raw) -> str:
    """Convert UI month values to parquet MONTH keys.

    UI sends:  None | 'ALL' | 'Q1'-'Q4' | '01'-'12'  (zero-padded strings)
    Parquet:   'ALL' | '1'-'12'  (no padding, as stored by generator)
    Quarterly: resolve Q-n → last month of quarter (e.g. Q1 → 3)
    """
    if month_raw in (None, "ALL"):
        return "ALL"
    if isinstance(month_raw, str) and month_raw.upper().startswith("Q"):
        try:
            q = int(month_raw[1])
            return str(q * 3)
        except Exception:
            return "ALL"
    try:
        return str(int(month_raw))     # strips leading zero: "01" → "1"
    except Exception:
        return "ALL"


def lookup_kpi_row(country, year, month, boc, status, dmode):
    """Return the precomputed parquet row for one KPI lookup, or None."""
    if CACHE_V1.empty:
        return None
    c = str(country) if country else "ALL"
    y = int(year) if year and year != "ALL" else None
    m = _normalise_month(month)
    b = str(boc) if boc else "ALL"
    s = str(status) if status else "IN FLEET"
    d = str(dmode) if dmode else "NONE"

    if y is None:
        return None

    mask = (
        (CACHE_V1["COUNTRY"]      == c) &
        (CACHE_V1["YEAR"]         == y) &
        (CACHE_V1["MONTH"]        == m) &
        (CACHE_V1["BIKE_OR_CAR"]  == b) &
        (CACHE_V1["ASSET_STATUS"] == s) &
        (CACHE_V1["DATE_MODE"]    == d)
    )
    rows = CACHE_V1[mask]
    return rows.iloc[0] if not rows.empty else None


def lookup_kpi7(country, year, status, metric_mode, period_mode, boc):
    """Return filtered KPI-7 pivot from cache, indexed by FUEL_TYPE."""
    required_meta = {"COUNTRY", "ASSET_STATUS", "METRIC_MODE", "PERIOD_MODE", "BIKE_OR_CAR"}
    if CACHE_KPI7.empty or "FUEL_TYPE" not in CACHE_KPI7.columns:
        return None
    if not required_meta.issubset(CACHE_KPI7.columns):
        return None

    c  = str(country)     if country     else "ALL"
    y  = int(year)        if year and year != "ALL" else None
    s  = str(status)      if status      else "IN FLEET"
    mm = str(metric_mode) if metric_mode else "share"
    pm = str(period_mode) if period_mode else "monthly"
    b  = str(boc)         if boc         else "ALL"

    mask = (
        (CACHE_KPI7["COUNTRY"]      == c) &
        (CACHE_KPI7["ASSET_STATUS"] == s) &
        (CACHE_KPI7["METRIC_MODE"]  == mm) &
        (CACHE_KPI7["PERIOD_MODE"]  == pm) &
        (CACHE_KPI7["BIKE_OR_CAR"]  == b)
    )

    meta_cols = ["COUNTRY", "YEAR", "ASSET_STATUS", "METRIC_MODE", "PERIOD_MODE", "BIKE_OR_CAR"]

    if pm == "yearly":
        # Yearly mode: show all years — do NOT filter by YEAR, merge rows per fuel type
        res = CACHE_KPI7[mask].copy()
        if res.empty:
            return None
        res = res.set_index("FUEL_TYPE")
        res = res.drop(columns=[col for col in meta_cols if col in res.columns], errors="ignore")
        # Each fuel type has one row per year with values in its year column — combine into one row
        res = res.groupby(res.index).first()
        res = res.dropna(axis=1, how="all")
    else:
        # Monthly / quarterly: must have a specific year selected
        if y is None or "YEAR" not in CACHE_KPI7.columns:
            return None
        mask = mask & (CACHE_KPI7["YEAR"] == y)
        res = CACHE_KPI7[mask].copy()
        if res.empty:
            return None
        res = res.set_index("FUEL_TYPE")
        res = res.drop(columns=[col for col in meta_cols if col in res.columns], errors="ignore")
        res = res.dropna(axis=1, how="all")

    return res if not res.empty else None


def lookup_kpi8(country, year, status, metric_mode, boc, dmode, period_mode):
    """Return filtered KPI-8 pivot from cache."""
    required_meta = {"COUNTRY", "YEAR", "ASSET_STATUS", "METRIC_MODE", "BIKE_OR_CAR", "DATE_MODE", "PERIOD_MODE"}
    if CACHE_KPI8.empty:
        return pd.DataFrame()
    if not required_meta.issubset(CACHE_KPI8.columns):
        return pd.DataFrame()

    c  = str(country)     if country     else "ALL"
    y  = int(year)        if year and year != "ALL" else None
    s  = str(status)      if status      else "IN FLEET"
    mm = str(metric_mode) if metric_mode else "volume"
    b  = str(boc)         if boc         else "ALL"
    d  = str(dmode)       if dmode       else "CONTRACT_START_DATE"
    pm = str(period_mode) if period_mode else "monthly"

    if y is None:
        return pd.DataFrame()

    mask = (
        (CACHE_KPI8["COUNTRY"]      == c) &
        (CACHE_KPI8["YEAR"]         == y) &
        (CACHE_KPI8["ASSET_STATUS"] == s) &
        (CACHE_KPI8["METRIC_MODE"]  == mm) &
        (CACHE_KPI8["BIKE_OR_CAR"]  == b) &
        (CACHE_KPI8["DATE_MODE"]    == d) &
        (CACHE_KPI8["PERIOD_MODE"]  == pm)
    )
    res = CACHE_KPI8[mask].copy()

    meta_cols = ["COUNTRY", "YEAR", "ASSET_STATUS", "METRIC_MODE", "BIKE_OR_CAR", "DATE_MODE", "PERIOD_MODE"]
    return res.drop(columns=[col for col in meta_cols if col in res.columns], errors="ignore")


# ── Helpers for Views 3-6 lookups ─────────────────────────────────────────────

def _serialize_value(value: object) -> str:
    """Mirror the generator's serialize_value: list → pipe-separated string."""
    if isinstance(value, (list, tuple, set)):
        parts = [_serialize_value(item) for item in value]
        return "|".join(p for p in parts if p)
    if value is None:
        return "ALL"
    if isinstance(value, float) and pd.isna(value):
        return "ALL"
    return str(value)


def _ordered_statuses(selected: list) -> list:
    """Return statuses ordered by fmd.asset_status_options (matches generator order)."""
    known = [item["value"] for item in fmd.asset_status_options]
    return [s for s in known if s in selected] + [s for s in selected if s not in known]


_KPI91_FILTER_COLS = {
    "YEAR_FILTER", "COUNTRY_FILTER", "VEHICLE_CLASS_FILTER",
    "VEHICLE_BODY_FILTER", "BIKE_OR_CAR_FILTER", "PERIOD_MODE_FILTER", "STATUS_SELECTION",
}
_KPI92_FILTER_COLS = {
    "YEAR_FILTER", "COUNTRY_FILTER", "VEHICLE_CLASS_FILTER",
    "VEHICLE_BODY_FILTER", "BIKE_OR_CAR_FILTER", "PERIOD_MODE_FILTER", "STATUS_FILTER",
}
_KPI10_FILTER_COLS = {
    "COUNTRY_FILTER", "YEAR_FILTER", "ASSET_STATUS_FILTER", "PERIOD_MODE_FILTER", "BIKE_OR_CAR_FILTER",
}
_KPI11_FILTER_COLS = {
    "SOURCE_FILTER", "COUNTRY_FILTER", "STATUS_FILTER", "REG_TYPE_FILTER",
    "OWNER_FILTER", "VARIABLE_FILTER", "TOP_N_FILTER", "PERIOD_MODE_FILTER",
}
_KPI13_FILTER_COLS = {
    "COUNTRY_FILTER", "STATUS_FILTER", "PORT_REG_FILTER", "MKT_REG_FILTER",
    "VARIABLE_FILTER", "PERIOD_MODE_FILTER", "TOP_N_FILTER", "OWNER_FILTER",
}


def lookup_kpi91(country, year, status_list, period_mode, bike_or_car, vehicle_class, vehicle_body):
    """Return KPI9_1 rows matching the given filters (all statuses in status_list)."""
    if CACHE_KPI91.empty:
        return pd.DataFrame()
    if not year or year == "ALL":
        return pd.DataFrame()
    ordered = _ordered_statuses(status_list if isinstance(status_list, list) else [status_list])
    status_key = _serialize_value(ordered)
    c  = str(country)      if country      else "ALL"
    pm = str(period_mode)  if period_mode  else "monthly"
    b  = str(bike_or_car)  if bike_or_car  else "ALL"
    vc = str(vehicle_class) if vehicle_class else "ALL"
    vb = str(vehicle_body)  if vehicle_body  else "ALL"
    try:
        y = int(year)
    except (TypeError, ValueError):
        return pd.DataFrame()
    mask = (
        (CACHE_KPI91["COUNTRY_FILTER"]       == c) &
        (CACHE_KPI91["YEAR_FILTER"]          == y) &
        (CACHE_KPI91["STATUS_SELECTION"]     == status_key) &
        (CACHE_KPI91["PERIOD_MODE_FILTER"]   == pm) &
        (CACHE_KPI91["BIKE_OR_CAR_FILTER"]   == b) &
        (CACHE_KPI91["VEHICLE_CLASS_FILTER"] == vc) &
        (CACHE_KPI91["VEHICLE_BODY_FILTER"]  == vb)
    )
    res = CACHE_KPI91[mask].copy()
    return res.drop(columns=[col for col in _KPI91_FILTER_COLS if col in res.columns], errors="ignore")


def lookup_kpi92(country, year, status, period_mode, bike_or_car, vehicle_class, vehicle_body):
    """Return KPI9_2 rows matching the given filters."""
    if CACHE_KPI92.empty:
        return pd.DataFrame()
    if not year or year == "ALL":
        return pd.DataFrame()
    c  = str(country)      if country      else "ALL"
    s  = str(status)       if status       else "IN FLEET"
    pm = str(period_mode)  if period_mode  else "monthly"
    b  = str(bike_or_car)  if bike_or_car  else "ALL"
    vc = str(vehicle_class) if vehicle_class else "ALL"
    vb = str(vehicle_body)  if vehicle_body  else "ALL"
    try:
        y = int(year)
    except (TypeError, ValueError):
        return pd.DataFrame()
    mask = (
        (CACHE_KPI92["COUNTRY_FILTER"]       == c) &
        (CACHE_KPI92["YEAR_FILTER"]          == y) &
        (CACHE_KPI92["STATUS_FILTER"]        == s) &
        (CACHE_KPI92["PERIOD_MODE_FILTER"]   == pm) &
        (CACHE_KPI92["BIKE_OR_CAR_FILTER"]   == b) &
        (CACHE_KPI92["VEHICLE_CLASS_FILTER"] == vc) &
        (CACHE_KPI92["VEHICLE_BODY_FILTER"]  == vb)
    )
    res = CACHE_KPI92[mask].copy()
    return res.drop(columns=[col for col in _KPI92_FILTER_COLS if col in res.columns], errors="ignore")


def lookup_kpi10(country, year, status, period_mode, bike_or_car):
    """Return KPI10 table matching the given filters."""
    if CACHE_KPI10.empty:
        return pd.DataFrame()
    if not year or year == "ALL":
        return pd.DataFrame()
    c  = str(country)     if country     else "ALL"
    s  = str(status)      if status      else "IN FLEET"
    pm = str(period_mode) if period_mode else "monthly"
    b  = str(bike_or_car) if bike_or_car else "ALL"
    try:
        y = int(year)
    except (TypeError, ValueError):
        return pd.DataFrame()
    mask = (
        (CACHE_KPI10["COUNTRY_FILTER"]      == c) &
        (CACHE_KPI10["YEAR_FILTER"]         == y) &
        (CACHE_KPI10["ASSET_STATUS_FILTER"] == s) &
        (CACHE_KPI10["PERIOD_MODE_FILTER"]  == pm) &
        (CACHE_KPI10["BIKE_OR_CAR_FILTER"]  == b)
    )
    res = CACHE_KPI10[mask].copy()
    return res.drop(columns=[col for col in _KPI10_FILTER_COLS if col in res.columns], errors="ignore")


_PORTFOLIO_VAR_COL: dict[str, str] = {
    "BRAND_UPDATE": "BRAND_UPDATE",
    "OEM_UPDATE":   "OEM_UPDATE",
    "CO2_BUCKET":   "CO2_BUCKET",
    "HIGHEST_BEV":  "BEV",
}
_MARKET_VAR_COL: dict[str, str] = {
    "BRAND_UPDATE": "Make",
    "OEM_UPDATE":   "Make Group",
    "CO2_BUCKET":   "CO2_BUCKET",
    "HIGHEST_BEV":  "BEV",
}


def _kpi11_var_col(source: str, variable: str) -> str:
    """Return the actual parquet column name for a KPI11 variable."""
    if source == "market":
        return _MARKET_VAR_COL.get(variable, variable)
    return _PORTFOLIO_VAR_COL.get(variable, variable)


def lookup_kpi11(source, status=None, reg_type=None, owner=None, variable=None, top_n=None, period_mode=None):
    """Return KPI11 rows for portfolio or market source."""
    if CACHE_KPI11.empty:
        return pd.DataFrame()
    src       = str(source)      if source      else "portfolio"
    pm        = str(period_mode) if period_mode else "monthly"
    tn        = str(top_n)       if top_n       else "ALL"
    actual_col = _kpi11_var_col(src, str(variable) if variable else "")

    if actual_col not in CACHE_KPI11.columns:
        return pd.DataFrame()

    mask = (
        (CACHE_KPI11["SOURCE_FILTER"]      == src) &
        (CACHE_KPI11["TOP_N_FILTER"]       == tn) &
        (CACHE_KPI11["PERIOD_MODE_FILTER"] == pm) &
        CACHE_KPI11[actual_col].notna()
    )
    if src == "portfolio":
        s    = str(status) if status else "IN FLEET"
        mask = mask & (CACHE_KPI11["STATUS_FILTER"] == s)
    else:
        rt = _serialize_value(reg_type) if reg_type else "ALL"
        ow = str(owner) if owner else "ALL"
        mask = mask & (CACHE_KPI11["REG_TYPE_FILTER"] == rt)
        mask = mask & (CACHE_KPI11["OWNER_FILTER"]    == ow)

    res = CACHE_KPI11[mask].copy()
    keep = ["COUNTRY", "YEAR", "PERIOD", actual_col, "VOLUME", "TOTAL", "SHARE"]
    if "VOLUME_YTD" in res.columns:
        keep += ["VOLUME_YTD", "SHARE_YTD"]
    return res[[c for c in keep if c in res.columns]]


def lookup_kpi13(status, port_reg, mkt_reg, variable, period_mode, top_n, owner):
    """Return KPI13 rows matching all filters."""
    if CACHE_KPI13.empty:
        return pd.DataFrame()
    s   = str(status)      if status      else "IN FLEET"
    pr  = _serialize_value(port_reg)      if port_reg   else "ALL"
    mr  = _serialize_value(mkt_reg)       if mkt_reg    else "ALL"
    var = str(variable)    if variable    else ""
    pm  = str(period_mode) if period_mode else "monthly"
    tn  = str(top_n)       if top_n       else "ALL"
    ow  = str(owner)       if owner       else "ALL"
    mask = (
        (CACHE_KPI13["STATUS_FILTER"]      == s) &
        (CACHE_KPI13["PORT_REG_FILTER"]    == pr) &
        (CACHE_KPI13["MKT_REG_FILTER"]     == mr) &
        (CACHE_KPI13["VARIABLE_FILTER"]    == var) &
        (CACHE_KPI13["PERIOD_MODE_FILTER"] == pm) &
        (CACHE_KPI13["TOP_N_FILTER"]       == tn) &
        (CACHE_KPI13["OWNER_FILTER"]       == ow)
    )
    res = CACHE_KPI13[mask].copy()
    return res.drop(columns=[col for col in _KPI13_FILTER_COLS if col in res.columns], errors="ignore")


# =============================================================================
# 3.  PATCH FMD OPTIONS FROM PARQUET (only show what was actually generated)
# =============================================================================

_avail_years:     list[int] = []
_avail_countries: list[str] = []

if not CACHE_V1.empty:
    _avail_years     = sorted(CACHE_V1["YEAR"].astype(int).unique())
    _avail_countries = sorted(str(c) for c in CACHE_V1["COUNTRY"].unique() if str(c) != "ALL")
    _avail_boc       = sorted(str(b) for b in CACHE_V1["BIKE_OR_CAR"].unique())
    _avail_statuses  = sorted(str(s) for s in CACHE_V1["ASSET_STATUS"].unique())
    _avail_dmodes    = sorted(str(d) for d in CACHE_V1["DATE_MODE"].unique())

    _avail_years = [int(y) for y in _avail_years]   # ensure plain Python int

    fmd.years           = _avail_years
    fmd.countries       = _avail_countries
    fmd.country_options = _avail_countries + ["ALL"]
    fmd.year_options    = [{"label": "ALL", "value": "ALL"}] + [
        {"label": str(y), "value": y} for y in _avail_years
    ]
    fmd.default_year    = _avail_years[-1] if _avail_years else None

    fmd.bike_or_car_options  = ["ALL"] + [b for b in _avail_boc if b != "ALL"]
    fmd.asset_status_options = [{"label": s, "value": s} for s in _avail_statuses]
    fmd.date_mode_options    = [
        {"label": d.replace("_", " ").title(), "value": d} for d in _avail_dmodes
    ]

if not CACHE_KPI7.empty:
    _meta_kpi7 = {"FUEL_TYPE", "COUNTRY", "YEAR", "ASSET_STATUS", "METRIC_MODE", "PERIOD_MODE", "BIKE_OR_CAR"}
    _period_cols = sorted(c for c in CACHE_KPI7.columns if c not in _meta_kpi7)
    fmd.cob_date_options = _period_cols

# Fallback: derive years/countries from V3-V6 caches when V1 was not generated
if not _avail_years:
    for _fc in (CACHE_KPI91, CACHE_KPI92, CACHE_KPI10, CACHE_KPI11, CACHE_KPI13):
        if _fc.empty:
            continue
        for _col in ("YEAR_FILTER", "YEAR"):
            if _col in _fc.columns:
                try:
                    _avail_years = sorted(
                        int(x) for x in _fc[_col].dropna().unique()
                        if str(x) not in ("ALL", "")
                    )
                except Exception:
                    pass
                if _avail_years:
                    break
        if _avail_years:
            break

if not _avail_countries:
    for _fc in (CACHE_KPI91, CACHE_KPI92, CACHE_KPI10, CACHE_KPI13):
        if _fc.empty:
            continue
        for _col in ("COUNTRY_FILTER", "COUNTRY"):
            if _col in _fc.columns:
                _avail_countries = sorted(
                    str(x) for x in _fc[_col].dropna().unique()
                    if str(x) not in ("ALL", "")
                )
                if _avail_countries:
                    break
        if _avail_countries:
            break


# =============================================================================
# 4.  DASH APP
# =============================================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(fmd.BASE_DIR / "fleet_assets"),
)
app.title = "Fleet Monitoring – Fast Reader"
app.layout = fmd.app.layout  # identical UI, same CSS, same filters

# Patch layout defaults that don't match precomputed filter values
def _patch_layout_value(component: object, target_id: str, new_value: object) -> bool:
    if getattr(component, "id", None) == target_id:
        component.value = new_value  # type: ignore[attr-defined]
        return True
    children = getattr(component, "children", None)
    if isinstance(children, list):
        return any(_patch_layout_value(c, target_id, new_value) for c in children)
    if children is not None:
        return _patch_layout_value(children, target_id, new_value)
    return False

# fmd default is "BRAND" but the parquet stores "BRAND_UPDATE"
_patch_layout_value(app.layout, "v6-kpi13-variable-filter", "BRAND_UPDATE")

# ── Inject View-2 complete-report modal into the shared layout ─────────────────
_modal_base_style = {
    "display": "none", "position": "fixed", "top": "0", "left": "0",
    "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)",
    "zIndex": "9999", "justifyContent": "center", "alignItems": "center",
}
_v2_complete_modal = html.Div(
    id="v2-complete-report-modal",
    style=dict(_modal_base_style),
    children=[
        html.Div(
            className="panel",
            style={"width": "480px", "maxWidth": "90%", "padding": "24px"},
            children=[
                html.H2("Generate View 2 Complete Report", style={"marginTop": "0"}),
                html.P(
                    "Select the years and granularities for your interactive KPI 8 report.",
                    className="small-note", style={"marginBottom": "16px"},
                ),
                html.Div("Country", className="filter-label",
                         style={"marginTop": "10px", "fontWeight": "bold"}),
                dcc.Dropdown(
                    id="v2-report-modal-country", options=[], value=None,
                    clearable=False, style={"width": "100%"},
                ),
                html.Div("Years to include", className="filter-label",
                         style={"marginTop": "16px", "fontWeight": "bold"}),
                dcc.Checklist(
                    id="v2-report-modal-years", options=[], value=[],
                    labelStyle={"display": "inline-block", "marginRight": "12px", "padding": "4px 0"},
                ),
                html.Div("Periods to include", className="filter-label",
                         style={"marginTop": "16px", "fontWeight": "bold"}),
                dcc.Checklist(
                    id="v2-report-modal-periods",
                    options=[
                        {"label": "Monthly",   "value": "monthly"},
                        {"label": "Quarterly", "value": "quarterly"},
                        {"label": "Annually",  "value": "yearly"},
                    ],
                    value=["yearly"],
                    labelStyle={"display": "inline-block", "marginRight": "12px", "padding": "4px 0"},
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "marginTop": "24px"},
                    children=[
                        html.Button("Cancel",          id="v2-close-complete-report-modal",
                                    n_clicks=0, className="secondary-button"),
                        html.Button("Download Report", id="v2-download-complete-report-btn",
                                    n_clicks=0, className="primary-button"),
                    ],
                ),
                dcc.Download(id="v2-complete-report-download"),
            ],
        )
    ],
)
app.layout.children.append(_v2_complete_modal)


# =============================================================================
# 5.  ROUTING & UI-ONLY CALLBACKS
# =============================================================================

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page(pathname: str):
    if pathname == "/vue-2":   return fmd.view2_layout()
    if pathname == "/vue-3":
        if not fmd.has_vehicle_type_temp: return fmd.view1_layout()
        return fmd.view3_layout()
    if pathname == "/vue-4":   return fmd.view4_layout()
    if pathname == "/vue-5":   return fmd.view5_layout()
    if pathname == "/vue-6":   return fmd.view6_layout()
    return fmd.view1_layout()


# Override unified_time_and_card_sync to avoid calling available_months(empty_df, ...)
# which would make the month dropdown show only "ALL" when a specific year is selected.
@app.callback(
    [
        Output("month-filter", "options"),
        Output("month-filter", "value"),
        Output("month-filter-label", "children"),
        Output("period-mode-filter", "value"),
    ] + [Output(f"kpi{i}-month-filter", "options") for i in range(1, 7)]
      + [Output(f"kpi{i}-month-filter", "value")   for i in range(1, 7)]
      + [Output(f"kpi{i}-country-filter", "value") for i in range(1, 7)]
      + [Output(f"kpi{i}-year-filter", "value")    for i in range(1, 7)]
      + [Output(f"kpi{i}-date-mode-filter", "value") for i in range(1, 7)]
      + [Output(f"kpi{i}-month-label", "children") for i in range(1, 7)]
      + [Output(f"kpi{i}-status-filter", "value")  for i in range(1, 7)]
      + [Output(f"kpi{i}-vehicle-filter", "value") for i in range(1, 7)],
    [
        Input("granularity-filter", "value"),
        Input("country-filter", "value"),
        Input("year-filter", "value"),
        Input("month-filter", "value"),
        Input("kpi-date-mode-filter", "value"),
        Input("status-filter", "value"),
        Input("kpi-common-bike-or-car-filter", "value"),
    ],
    prevent_initial_call=True,
)
def unified_time_and_card_sync(gran, country, year, month, date_mode, status, vehicle_type):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    month_opts   = [{"label": "ALL", "value": "ALL"}] + [
        {"label": calendar.month_name[i], "value": str(i).zfill(2)} for i in range(1, 13)
    ]
    quarter_opts = [{"label": "ALL", "value": "ALL"}] + [
        {"label": f"Q{i}", "value": f"Q{i}"} for i in range(1, 5)
    ]
    yearly_opts  = [{"label": "ALL", "value": "ALL"}]

    if gran == "yearly":
        opts  = yearly_opts
        label = "Month"
        if triggered_id == "granularity-filter":
            month = "ALL"
    elif gran == "quarterly":
        opts  = quarter_opts
        label = "Quarter"
        if triggered_id == "granularity-filter":
            month = month if any(o["value"] == month for o in quarter_opts) else "ALL"
    else:
        # Monthly mode: always show all months (no raw df available in fast reader)
        opts  = month_opts
        label = "Month"
        if triggered_id == "granularity-filter":
            month = month if any(o["value"] == month for o in opts) else "ALL"

    return (
        [opts, month, label, gran]
        + [opts] * 6
        + [month] * 6
        + [country] * 6
        + [year] * 6
        + [date_mode] * 6
        + [label] * 6
        + [status] * 6
        + [vehicle_type] * 6
    )


# ── Init: inject correct options from parquet into all dropdowns on page load ──
@app.callback(
    [
        Output("year-filter",          "options"),
        Output("year-filter",          "value"),
        Output("kpi7-year-filter",     "options"),
        Output("kpi7-year-filter",     "value"),
        Output("country-filter",       "options"),
        Output("country-filter",       "value"),
        Output("kpi7-country-filter",  "options"),
        Output("kpi7-country-filter",  "value"),
    ] + [Output(f"kpi{i}-year-filter", "options") for i in range(1, 7)],
    Input("url", "pathname"),
)
def init_filter_options(_):
    year_opts         = [{"label": str(y), "value": y} for y in _avail_years]
    country_all_opts  = [{"label": "ALL", "value": "ALL"}] + [{"label": c, "value": c} for c in _avail_countries]
    country_kpi7_opts = [{"label": c, "value": c} for c in _avail_countries]
    default_year      = _avail_years[-1] if _avail_years else None
    default_country   = _avail_countries[0] if _avail_countries else None
    return (
        [
            year_opts, default_year,               # year-filter
            year_opts, None,                       # kpi7-year-filter (user must pick)
            country_all_opts, default_country,     # country-filter
            country_kpi7_opts, default_country,    # kpi7-country-filter (ISO codes, no ALL)
        ] + [year_opts] * 6                        # kpi1-6 year-filter options
    )


# Modal toggle – handles both View-1 and View-2 complete-report modals
@app.callback(
    Output("complete-report-modal",    "style"),
    Output("v2-complete-report-modal", "style"),
    [
        Input({"type": "open-complete-report", "index": dash.ALL}, "n_clicks"),
        Input("close-complete-report-modal",    "n_clicks"),
        Input("download-complete-report-btn",   "n_clicks"),
        Input("v2-close-complete-report-modal", "n_clicks"),
        Input("v2-download-complete-report-btn","n_clicks"),
    ],
    [
        State("complete-report-modal",    "style"),
        State("v2-complete-report-modal", "style"),
    ],
    prevent_initial_call=True,
)
def toggle_modal(open_clicks_list, close_clicks, dl_clicks, _v2_close, _v2_dl, _v1_style, _v2_style):
    import json as _json
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update
    triggered_id    = ctx.triggered[0]["prop_id"]
    triggered_value = ctx.triggered[0].get("value", 0) or 0
    hidden = {"display": "none", "position": "fixed", "top": "0", "left": "0",
              "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)",
              "zIndex": "9999", "justifyContent": "center", "alignItems": "center"}
    shown  = {**hidden, "display": "flex"}

    if "open-complete-report" in triggered_id and triggered_value > 0:
        try:
            idx = _json.loads(triggered_id.split(".")[0])["index"]
        except Exception:
            idx = 1
        if idx == 2:
            return hidden, shown
        return shown, hidden

    if "close-complete-report-modal" in triggered_id or triggered_id.startswith("download-complete-report-btn"):
        return hidden, no_update

    if "v2-close-complete-report-modal" in triggered_id or triggered_id.startswith("v2-download-complete-report-btn"):
        return no_update, hidden

    return no_update, no_update


# =============================================================================
# 6.  VIEW 1 – KPI CARDS  (parquet lookup per card)
# =============================================================================

@app.callback(
    Output("kpi-1-card", "children"),
    Output("kpi-2-card", "children"),
    Output("kpi-3-card", "children"),
    Output("kpi-4-card", "children"),
    Output("kpi-5-card", "children"),
    Output("kpi-6-card", "children"),
    Input("view1-refresh-button", "n_clicks"),
    State("kpi1-country-filter", "value"),
    State("kpi1-year-filter", "value"),
    State("kpi1-month-filter", "value"),
    State("kpi2-country-filter", "value"),
    State("kpi2-year-filter", "value"),
    State("kpi2-month-filter", "value"),
    State("kpi3-country-filter", "value"),
    State("kpi3-year-filter", "value"),
    State("kpi3-month-filter", "value"),
    State("kpi4-country-filter", "value"),
    State("kpi4-year-filter", "value"),
    State("kpi4-month-filter", "value"),
    State("kpi5-country-filter", "value"),
    State("kpi5-year-filter", "value"),
    State("kpi5-month-filter", "value"),
    State("kpi6-country-filter", "value"),
    State("kpi6-year-filter", "value"),
    State("kpi6-month-filter", "value"),
    State("kpi1-date-mode-filter", "value"),
    State("kpi2-date-mode-filter", "value"),
    State("kpi3-date-mode-filter", "value"),
    State("kpi4-date-mode-filter", "value"),
    State("kpi5-date-mode-filter", "value"),
    State("kpi6-date-mode-filter", "value"),
    State("kpi-common-bike-or-car-filter", "value"),
    State("kpi-date-mode-filter", "value"),
    State("kpi1-status-filter", "value"),
    State("kpi2-status-filter", "value"),
    State("kpi3-status-filter", "value"),
    State("kpi4-status-filter", "value"),
    State("kpi5-status-filter", "value"),
    State("kpi6-status-filter", "value"),
    State("kpi1-vehicle-filter", "value"),
    State("kpi2-vehicle-filter", "value"),
    State("kpi3-vehicle-filter", "value"),
    State("kpi4-vehicle-filter", "value"),
    State("kpi5-vehicle-filter", "value"),
    State("kpi6-vehicle-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_kpi_cards(
    _n,
    c1, y1, m1, c2, y2, m2, c3, y3, m3, c4, y4, m4, c5, y5, m5, c6, y6, m6,
    dm1, dm2, dm3, dm4, dm5, dm6,
    global_boc, global_dm,
    s1, s2, s3, s4, s5, s6,
    v1, v2, v3, v4, v5, v6,
):
    waiting = fmd.build_card_body("--", "Select the filters then click Refresh.", "#9aa5b1")

    required = [c1, y1, c2, y2, c3, y3, m3, c4, y4, m4, c5, y5, m5, c6, y6, m6, global_boc]
    if fmd.has_missing_filters(*required):
        return waiting, waiting, waiting, waiting, waiting, waiting

    if CACHE_V1.empty:
        no_data = fmd.build_card_body("--", "No precomputed data. Run generate_all_precomputed.py first.", "#d64545")
        return no_data, no_data, no_data, no_data, no_data, no_data

    boc = [v1 or global_boc, v2 or global_boc, v3 or global_boc,
           v4 or global_boc, v5 or global_boc, v6 or global_boc]
    sta = [s1 or "IN FLEET", s2 or "IN FLEET", s3 or "IN FLEET",
           s4 or "IN FLEET", s5 or "IN FLEET", s6 or "IN FLEET"]
    dms = [dm1 or global_dm or "NONE", dm2 or global_dm or "NONE",
           dm3 or global_dm or "NONE", dm4 or global_dm or "NONE",
           dm5 or global_dm or "NONE", dm6 or global_dm or "NONE"]
    countries = [c1, c2, c3, c4, c5, c6]
    years     = [y1, y2, y3, y4, y5, y6]
    months    = [m1, m2, m3, m4, m5, m6]

    rows = [
        lookup_kpi_row(countries[i], years[i], months[i], boc[i], sta[i], dms[i])
        for i in range(6)
    ]

    # Field names match UPPERCASE parquet column names from generator
    kpi_fields = ["KPI1", "KPI2", None, "HYBRID", "EV", None]
    tuple_keys = [None, None, ("DIESEL", "NON_DIESEL"), None, None, ("PV", "LCV")]
    configs    = list(fmd.KPI_REGISTRY.items())

    cards = []
    for i, (kid, config) in enumerate(configs):
        row = rows[i]
        if row is None:
            display = "N/A"
        elif tuple_keys[i]:
            k1, k2 = tuple_keys[i]
            v_a = row.get(k1)
            v_b = row.get(k2)
            if v_a is not None and pd.notna(v_a):
                display = f"{fmd.percent_or_na_precision(float(v_a), 2)} / {fmd.percent_or_na_precision(float(v_b), 2)}"
            else:
                display = "N/A"
        else:
            field = kpi_fields[i]
            val   = row.get(field)
            display = fmd.percent_or_na_precision(float(val), 2) if val is not None and pd.notna(val) else "N/A"

        status_lbl = sta[i]
        cards.append(fmd.build_card_body(display, f"{config['label']} | {status_lbl}", config["color"]))

    return tuple(cards)


# =============================================================================
# 7.  VIEW 1 – KPI SUMMARY TABLE  (parquet lookup)
# =============================================================================

@app.callback(
    Output("kpi-summary-wrap", "children"),
    Input("view1-refresh-button", "n_clicks"),
    State("kpi1-country-filter", "value"),
    State("kpi1-year-filter", "value"),
    State("kpi1-month-filter", "value"),
    State("kpi2-country-filter", "value"),
    State("kpi2-year-filter", "value"),
    State("kpi2-month-filter", "value"),
    State("kpi3-country-filter", "value"),
    State("kpi3-year-filter", "value"),
    State("kpi3-month-filter", "value"),
    State("kpi4-country-filter", "value"),
    State("kpi4-year-filter", "value"),
    State("kpi4-month-filter", "value"),
    State("kpi5-country-filter", "value"),
    State("kpi5-year-filter", "value"),
    State("kpi5-month-filter", "value"),
    State("kpi6-country-filter", "value"),
    State("kpi6-year-filter", "value"),
    State("kpi6-month-filter", "value"),
    State("kpi1-date-mode-filter", "value"),
    State("kpi2-date-mode-filter", "value"),
    State("kpi3-date-mode-filter", "value"),
    State("kpi4-date-mode-filter", "value"),
    State("kpi5-date-mode-filter", "value"),
    State("kpi6-date-mode-filter", "value"),
    State("kpi-common-bike-or-car-filter", "value"),
    State("kpi-date-mode-filter", "value"),
    State("summary-kpi1-limit-filter", "value"),
    State("summary-kpi2-limit-filter", "value"),
    State("granularity-filter", "value"),
    State("kpi1-status-filter", "value"),
    State("kpi2-status-filter", "value"),
    State("kpi3-status-filter", "value"),
    State("kpi4-status-filter", "value"),
    State("kpi5-status-filter", "value"),
    State("kpi6-status-filter", "value"),
    State("kpi1-vehicle-filter", "value"),
    State("kpi2-vehicle-filter", "value"),
    State("kpi3-vehicle-filter", "value"),
    State("kpi4-vehicle-filter", "value"),
    State("kpi5-vehicle-filter", "value"),
    State("kpi6-vehicle-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_kpi_summary(
    _n,
    c1, y1, m1, c2, y2, m2, c3, y3, m3, c4, y4, m4, c5, y5, m5, c6, y6, m6,
    dm1, dm2, dm3, dm4, dm5, dm6,
    global_boc, global_dm,
    kpi1_limit, kpi2_limit, granularity,
    s1, s2, s3, s4, s5, s6,
    v1, v2, v3, v4, v5, v6,
):
    required = [c1, y1, c2, y2, c3, y3, m3, c4, y4, m4, c5, y5, m5, c6, y6, m6, global_boc]
    if fmd.has_missing_filters(*required):
        return html.Div("Select the filters then click Refresh.", className="small-note")

    if CACHE_V1.empty:
        return html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    boc = [v1 or global_boc, v2 or global_boc, v3 or global_boc,
           v4 or global_boc, v5 or global_boc, v6 or global_boc]
    sta = [s1 or "IN FLEET", s2 or "IN FLEET", s3 or "IN FLEET",
           s4 or "IN FLEET", s5 or "IN FLEET", s6 or "IN FLEET"]
    dms = [dm1 or global_dm or "NONE", dm2 or global_dm or "NONE",
           dm3 or global_dm or "NONE", dm4 or global_dm or "NONE",
           dm5 or global_dm or "NONE", dm6 or global_dm or "NONE"]

    rows = [
        lookup_kpi_row([c1,c2,c3,c4,c5,c6][i], [y1,y2,y3,y4,y5,y6][i],
                       [m1,m2,m3,m4,m5,m6][i], boc[i], sta[i], dms[i])
        for i in range(6)
    ]

    def _get(row, key, default=None):
        if row is None: return default
        v = row.get(key)
        return float(v) if v is not None and pd.notna(v) else default

    kpi1_val   = _get(rows[0], "KPI1")
    kpi2_val   = _get(rows[1], "KPI2")
    diesel_val = _get(rows[2], "DIESEL")
    non_di_val = _get(rows[2], "NON_DIESEL")
    hybrid_val = _get(rows[3], "HYBRID")
    ev_val     = _get(rows[4], "EV")
    pv_val     = _get(rows[5], "PV")
    lcv_val    = _get(rows[5], "LCV")

    vol1 = _get(rows[0], "VOLUME", 0)
    vol2 = _get(rows[1], "VOLUME", 0)
    vol3 = _get(rows[2], "VOLUME", 0)
    vol4 = _get(rows[3], "VOLUME", 0)
    vol5 = _get(rows[4], "VOLUME", 0)
    vol6 = _get(rows[5], "VOLUME", 0)

    import calendar as _cal
    def period_label(year, month_raw):
        if month_raw in (None, "ALL"): return str(year)
        if isinstance(month_raw, str) and month_raw.startswith("Q"): return f"{month_raw}-{year}"
        try:
            return f"{_cal.month_abbr[int(month_raw)]}-{int(year)}"
        except Exception:
            return f"{month_raw}-{year}"

    lim1 = float(kpi1_limit) if kpi1_limit is not None else None
    lim2 = float(kpi2_limit) if kpi2_limit is not None else None

    summary_rows = [
        {"label": "LTR < 25",           "country": c1, "period": period_label(y1, m1),
         "result_text": fmd.percent_or_na_precision(kpi1_val, 2),
         "signal": fmd.kpi_limit_status(kpi1_val, lim1), "volume": int(vol1 or 0),
         "unit": "%", "comment": f"LTR <25 share, Limit {lim1 or 'N/A'}%"},

        {"label": "LTR [25-30]",         "country": c2, "period": period_label(y2, m2),
         "result_text": fmd.percent_or_na_precision(kpi2_val, 2),
         "signal": fmd.kpi_limit_status(kpi2_val, lim2), "volume": int(vol2 or 0),
         "unit": "%", "comment": f"LTR [25-30] share, Limit {lim2 or 'N/A'}%"},

        {"label": "Diesel vs non-diesel", "country": c3, "period": period_label(y3, m3),
         "result_text": (f"{fmd.percent_or_na_precision(diesel_val, 2)} DI & {fmd.percent_or_na_precision(non_di_val, 2)} Non DI"
                         if diesel_val is not None else "N/A"),
         "signal": "neutral", "volume": int(vol3 or 0), "unit": "%", "comment": "Diesel Share"},

        {"label": "Hybrid share",         "country": c4, "period": period_label(y4, m4),
         "result_text": fmd.percent_or_na_precision(hybrid_val, 2),
         "signal": "neutral", "volume": int(vol4 or 0), "unit": "%", "comment": "Hyb (HEV+PHEV)"},

        {"label": "EV share",             "country": c5, "period": period_label(y5, m5),
         "result_text": fmd.percent_or_na_precision(ev_val, 2),
         "signal": "neutral", "volume": int(vol5 or 0), "unit": "%", "comment": "EV share"},

        {"label": "Passenger car vs LCV", "country": c6, "period": period_label(y6, m6),
         "result_text": (f"{fmd.percent_or_na_precision(pv_val, 2)} PV & {fmd.percent_or_na_precision(lcv_val, 2)} LCV"
                         if pv_val is not None else "N/A"),
         "signal": "neutral", "volume": int(vol6 or 0), "unit": "%", "comment": "PV vs LCV"},
    ]

    return fmd.render_kpi_summary_table(summary_rows)


@app.callback(
    Output("kpi-period-note", "children"),
    Input("view1-refresh-button", "n_clicks"),
    State("year-filter", "value"),
    State("month-filter", "value"),
    prevent_initial_call=True,
)
def update_period_note(_n, year, month):
    return ""


# =============================================================================
# 8.  VIEW 1 – KPI 7  (parquet lookup)
# =============================================================================

def _build_kpi7_table_from_cache(
    share_pivot: pd.DataFrame,
    metric_mode: str,
    country: str,
    year,
    status: str,
    period_mode: str,
    boc: str,
) -> pd.DataFrame:
    """Build the KPI7 display table entirely from precomputed cache (no raw df access)."""
    table_df    = share_pivot.copy()
    metric_cols = table_df.columns.tolist()

    if metric_mode == "share":
        vol_pivot = lookup_kpi7(country, year, status, "volume", period_mode, boc)
        if vol_pivot is not None and not vol_pivot.empty:
            volume_totals = vol_pivot.sum(axis=0).reindex(metric_cols, fill_value=0)
            total_row = pd.DataFrame([volume_totals], index=["TOTAL"])
        else:
            total_row = pd.DataFrame([{c: 0 for c in metric_cols}], index=["TOTAL"])
        table_df = pd.concat([table_df, total_row])
    else:
        total_row = pd.DataFrame([table_df[metric_cols].sum(axis=0)], index=["TOTAL"])
        table_df  = pd.concat([table_df, total_row])

    table_df.index.name = "Fuel type"
    table_df = table_df.reset_index()
    total_mask = table_df["Fuel type"].astype(str) == "TOTAL"

    for column in [c for c in table_df.columns if c != "Fuel type"]:
        display_col = table_df[column].astype(object)
        if metric_mode == "share":
            display_col.loc[~total_mask] = table_df.loc[~total_mask, column].map(
                lambda v: fmd.percent_or_na_precision(float(v), 1) if pd.notna(v) else "N/A"
            )
            display_col.loc[total_mask] = table_df.loc[total_mask, column].map(
                lambda v: fmd.format_volume(v)
            )
        else:
            display_col = table_df[column].map(lambda v: fmd.format_volume(v))
        table_df[column] = display_col

    return table_df


@app.callback(
    Output("kpi-7-graph", "figure"),
    Output("kpi-7-table-wrap", "children"),
    Input("view1-kpi7-refresh-button", "n_clicks"),
    State("kpi7-country-filter", "value"),
    State("kpi7-year-filter", "value"),
    State("status-group-filter", "value"),
    State("metric-mode-filter", "value"),
    State("period-mode-filter", "value"),
    State("kpi7-bike-or-car-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_kpi7(_n, country, year, status, metric_mode, period_mode, boc):
    empty_fig = go.Figure()

    if CACHE_KPI7.empty:
        empty_fig.update_layout(title="No precomputed data. Run generate_all_precomputed.py first.")
        return empty_fig, html.Div("No precomputed data.", className="small-note")

    country     = country     or "ALL"
    status      = status      or "IN FLEET"
    metric_mode = metric_mode or "share"
    period_mode = period_mode or "monthly"
    boc         = boc         or "ALL"

    if period_mode != "yearly" and (not year or year == "ALL"):
        empty_fig.update_layout(title="Select a year to display KPI 7.")
        return empty_fig, html.Div("Select a year.", className="small-note")

    pivot = lookup_kpi7(country, year, status, metric_mode, period_mode, boc)
    if pivot is None or (isinstance(pivot, pd.DataFrame) and pivot.empty):
        empty_fig.update_layout(title="No data for the selected filters.")
        return empty_fig, html.Div("No data for selected filters.", className="small-note")

    y_title      = "Share (%)" if metric_mode == "share" else "Volume"
    x_title      = {"monthly": "Month", "quarterly": "Quarter", "yearly": "Year"}.get(period_mode, "Period")
    title_mode   = "share" if metric_mode == "share" else "volume"
    period_title = {"monthly": "by month", "quarterly": "by quarter"}.get(period_mode, "yearly")
    year_label   = "All years" if period_mode == "yearly" else str(year)
    title        = f"Fuel Type {title_mode} {period_title} – {year_label}<br><sup>{country} | {status} | {boc}</sup>"
    fig          = fmd.figure_from_pivot(pivot, y_title, x_title, title)

    table_df = _build_kpi7_table_from_cache(pivot, metric_mode, country, year, status, period_mode, boc)
    table    = fmd.build_table(table_df, page_size=15, table_id="v1-kpi7-table")
    cap      = "Share" if metric_mode == "share" else "Volume"
    table_title = f"Fuel Type {cap} Table – {status} – {cap} – {period_mode.capitalize()}"

    return fig, html.Div([html.H4(table_title, className="panel-title"), table])


# =============================================================================
# 9.  VIEW 2 – KPI 8  (parquet lookup)
# =============================================================================

@app.callback(
    Output("v2-kpi8-graph", "figure"),
    Output("v2-kpi8-table-wrap", "children"),
    Output("v2-applied-filters", "data"),
    Input("view2-refresh-button", "n_clicks"),
    State("v2-country-filter", "value"),
    State("v2-year-filter", "value"),
    State("v2-metric-mode-filter", "value"),
    State("v2-bike-or-car-filter", "value"),
    State("v2-date-mode-filter", "value"),
    State("v2-period-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_kpi8(_n, country, year, metric_mode, boc, date_mode, period_mode):
    asset_status = "IN FLEET"
    country     = country     or "ALL"
    year        = year        or (fmd.default_year if fmd.default_year else None)
    metric_mode = metric_mode or "volume"
    boc         = boc         or "ALL"
    date_mode   = date_mode   or "CONTRACT_START_DATE"
    period_mode = period_mode or "monthly"
    applied = {
        "country": country, "year": year, "metric_mode": metric_mode,
        "bike_or_car": boc, "v2_date_mode_filter": date_mode,
        "period_mode": period_mode, "asset_status": asset_status,
    }
    empty_fig = go.Figure()

    if CACHE_KPI8.empty:
        empty_fig.update_layout(title="No precomputed data. Run generate_all_precomputed.py first.")
        return empty_fig, html.Div("No precomputed data.", className="small-note"), applied

    table_kpi8 = lookup_kpi8(country, year, asset_status, metric_mode, boc, date_mode, period_mode)
    if table_kpi8 is None or table_kpi8.empty:
        empty_fig.update_layout(title=f"No data for {country} – {year}")
        return empty_fig, html.Div("No data available.", className="small-note"), applied

    y_title = "Share (%)" if metric_mode == "share" else "Volume"
    x_title = {"monthly": "Month", "quarterly": "Quarter", "yearly": "Year"}.get(period_mode, "Period")

    period_col  = "PERIOD" if "PERIOD" in table_kpi8.columns else table_kpi8.columns[0]
    metric_cols = [c for c in table_kpi8.columns if c not in [period_col, "YEAR", "MONTH"]]
    color_map   = fmd.fuel_color_map(metric_cols)
    ordered     = sorted(metric_cols, key=lambda n: (fmd.fuel_label_priority(n), metric_cols.index(n)))

    title = f"Fuel Type {metric_mode} {x_title.lower()}<br><sup>{country} | {year} | {boc}</sup>"
    fig   = go.Figure()
    for col in ordered:
        color = color_map[col]
        fig.add_trace(go.Scatter(
            x=table_kpi8[period_col].tolist(),
            y=table_kpi8[col].tolist(),
            mode="lines+markers",
            name=str(col),
            line={"color": color, "width": 2.2},
            marker={"color": color, "size": 7},
        ))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      template="plotly_white", height=520)

    display_df  = fmd.build_view2_table_with_total(table_kpi8, metric_mode)
    display_df  = fmd.format_view2_table_for_display(display_df)
    table       = fmd.build_table(display_df, page_size=15, table_id="v2-kpi8-table")
    cap         = "Share" if metric_mode == "share" else "Volume"
    table_title = f"Production by Fuel Type Table – IN FLEET – {cap} – {x_title}"

    return fig, html.Div([html.H4(table_title, className="panel-title"), table]), applied


# =============================================================================
# 10.  DOWNLOAD CALLBACKS  (delegate to fmd – they build HTML from stored state)
# =============================================================================

@app.callback(
    Output("view1-html-download", "data"),
    Input("view1-download-button", "n_clicks"),
    State("country-filter", "value"),
    State("year-filter", "value"),
    State("month-filter", "value"),
    State("kpi1-country-filter", "value"),
    State("kpi1-year-filter", "value"),
    State("kpi1-month-filter", "value"),
    State("kpi2-country-filter", "value"),
    State("kpi2-year-filter", "value"),
    State("kpi2-month-filter", "value"),
    State("kpi3-country-filter", "value"),
    State("kpi3-year-filter", "value"),
    State("kpi3-month-filter", "value"),
    State("kpi4-country-filter", "value"),
    State("kpi4-year-filter", "value"),
    State("kpi4-month-filter", "value"),
    State("kpi5-country-filter", "value"),
    State("kpi5-year-filter", "value"),
    State("kpi5-month-filter", "value"),
    State("kpi6-country-filter", "value"),
    State("kpi6-year-filter", "value"),
    State("kpi6-month-filter", "value"),
    State("kpi1-date-mode-filter", "value"),
    State("kpi2-date-mode-filter", "value"),
    State("kpi3-date-mode-filter", "value"),
    State("kpi4-date-mode-filter", "value"),
    State("kpi5-date-mode-filter", "value"),
    State("kpi6-date-mode-filter", "value"),
    State("kpi-common-bike-or-car-filter", "value"),
    State("kpi-date-mode-filter", "value"),
    State("summary-kpi1-limit-filter", "value"),
    State("summary-kpi2-limit-filter", "value"),
    State("kpi7-country-filter", "value"),
    State("kpi7-year-filter", "value"),
    State("status-group-filter", "value"),
    State("metric-mode-filter", "value"),
    State("period-mode-filter", "value"),
    State("kpi7-bike-or-car-filter", "value"),
    State("granularity-filter", "value"),
    State("kpi1-status-filter", "value"),
    State("kpi2-status-filter", "value"),
    State("kpi3-status-filter", "value"),
    State("kpi4-status-filter", "value"),
    State("kpi5-status-filter", "value"),
    State("kpi6-status-filter", "value"),
    State("kpi1-vehicle-filter", "value"),
    State("kpi2-vehicle-filter", "value"),
    State("kpi3-vehicle-filter", "value"),
    State("kpi4-vehicle-filter", "value"),
    State("kpi5-vehicle-filter", "value"),
    State("kpi6-vehicle-filter", "value"),
    State("kpi-1-card", "children"),
    State("kpi-2-card", "children"),
    State("kpi-3-card", "children"),
    State("kpi-4-card", "children"),
    State("kpi-5-card", "children"),
    State("kpi-6-card", "children"),
    State("v1-summary-table", "data"),
    State("kpi-7-graph", "figure"),
    State("v1-kpi7-table", "data"),
    State("v1-kpi7-table", "columns"),
    prevent_initial_call=True,
)
def download_view1_html(
    n_clicks, country, year, month,
    c1, y1, m1, c2, y2, m2, c3, y3, m3,
    c4, y4, m4, c5, y5, m5, c6, y6, m6,
    dm1, dm2, dm3, dm4, dm5, dm6,
    global_boc, global_dm,
    kpi1_limit, kpi2_limit,
    kpi7_country, kpi7_year,
    kpi7_status, kpi7_metric, kpi7_period, kpi7_boc, granularity,
    s1, s2, s3, s4, s5, s6,
    v1, v2, v3, v4, v5, v6,
    kpi1_ch, kpi2_ch, kpi3_ch, kpi4_ch, kpi5_ch, kpi6_ch,
    summary_data, kpi7_fig, kpi7_tdata, kpi7_tcols,
):
    return fmd.download_view1_html(
        n_clicks, country, year, month,
        c1, y1, m1, c2, y2, m2, c3, y3, m3,
        c4, y4, m4, c5, y5, m5, c6, y6, m6,
        dm1, dm2, dm3, dm4, dm5, dm6,
        global_boc, global_dm,
        kpi1_limit, kpi2_limit,
        kpi7_country, kpi7_year,          # fmd now expects (kpi7_country, kpi7_year, ...)
        kpi7_status, kpi7_metric, kpi7_period, kpi7_boc, granularity,
        s1, s2, s3, s4, s5, s6,
        v1, v2, v3, v4, v5, v6,
        kpi1_ch, kpi2_ch, kpi3_ch, kpi4_ch, kpi5_ch, kpi6_ch,
        summary_data, kpi7_fig, kpi7_tdata, kpi7_tcols,
    )


@app.callback(
    Output("view2-html-download", "data"),
    Input("view2-download-button", "n_clicks"),
    State("v2-applied-filters", "data"),
    State("v2-kpi8-graph", "figure"),
    State("v2-kpi8-table", "data"),
    State("v2-kpi8-table", "columns"),
    prevent_initial_call=True,
)
def download_view2_html(*args):
    return fmd.download_view2_html(*args)


# ── Populate modal options when it opens ──────────────────────────────────────
@app.callback(
    Output("report-modal-country", "options"),
    Output("report-modal-country", "value"),
    Output("report-modal-years",   "options"),
    Output("report-modal-years",   "value"),
    Input({"type": "open-complete-report", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def populate_report_modal(_):
    country_opts = [{"label": c, "value": c} for c in _avail_countries]
    year_opts    = [{"label": str(y), "value": str(y)} for y in sorted(_avail_years, reverse=True)]
    default_c    = _avail_countries[0] if _avail_countries else None
    default_y    = [str(y) for y in _avail_years]
    return country_opts, default_c, year_opts, default_y


@app.callback(
    Output("v2-report-modal-country", "options"),
    Output("v2-report-modal-country", "value"),
    Output("v2-report-modal-years",   "options"),
    Output("v2-report-modal-years",   "value"),
    Input({"type": "open-complete-report", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def populate_v2_report_modal(_):
    country_opts = [{"label": c, "value": c} for c in _avail_countries]
    year_opts    = [{"label": str(y), "value": str(y)} for y in sorted(_avail_years, reverse=True)]
    default_c    = _avail_countries[0] if _avail_countries else None
    default_y    = [str(y) for y in _avail_years]
    return country_opts, default_c, year_opts, default_y


def _kpi7_table_from_cache(
    country: str, year, status: str, metric_mode: str, gran: str, boc: str
) -> pd.DataFrame:
    """Build formatted KPI7 table from cache (no raw df needed)."""
    pivot = lookup_kpi7(country, year, status, metric_mode, gran, boc)
    if pivot is None or pivot.empty:
        return pd.DataFrame()
    table_df    = pivot.copy()
    metric_cols = table_df.columns.tolist()
    if metric_mode == "share":
        vol = lookup_kpi7(country, year, status, "volume", gran, boc)
        totals = (vol.reindex(columns=metric_cols, fill_value=0).sum(axis=0)
                  if vol is not None and not vol.empty
                  else pd.Series(0, index=metric_cols))
    else:
        totals = table_df[metric_cols].sum(axis=0)
    table_df         = pd.concat([table_df, pd.DataFrame([totals], index=["TOTAL"])])
    table_df.index.name = "Fuel type"
    table_df         = table_df.reset_index()
    is_total         = table_df["Fuel type"].astype(str) == "TOTAL"
    for col in [c for c in table_df.columns if c != "Fuel type"]:
        table_df[col] = [
            fmd.format_volume(v) if t else (
                (fmd.percent_or_na_precision(float(v), 1) if pd.notna(v) else "N/A")
                if metric_mode == "share" else fmd.format_volume(v)
            )
            for v, t in zip(table_df[col], is_total)
        ]
    return table_df


def _build_complete_report_from_cache(
    country: str,
    selected_years: list[str],
    selected_periods: list[str],
) -> str:
    import calendar as _cal

    DATE_MODES = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]
    STATUSES   = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    BOCS       = _avail_boc if _avail_boc else ["CAR", "BIKE", "ALL"]
    METRICS    = ["share", "volume"]

    dmode_labels = {"NONE": "None", "CONTRACT_START_DATE": "Contract Start", "DELIVERY_DATE": "Delivery Date"}
    kpi_fields   = ["KPI1", "KPI2", None, "HYBRID", "EV", None]
    tuple_keys   = [None, None, ("DIESEL", "NON_DIESEL"), None, None, ("PV", "LCV")]
    kpi_configs  = list(fmd.KPI_REGISTRY.items())

    # ── Controls bar ──────────────────────────────────────────────────────────
    cs  = "padding:6px 10px;border-radius:8px;border:1px solid #d9e2ec;font-size:12px;color:#102a43;"
    ls  = "color:#627d98;font-size:12px;font-weight:600;margin-left:10px;"

    def sel(id_, opts_html):
        return f'<select id="{id_}" onchange="upd()" style="{cs}">{opts_html}</select>'

    y_opts  = "".join(f'<option value="{y}">{y}</option>' for y in selected_years)
    g_opts  = "".join(f'<option value="{p}">{p.capitalize()}</option>' for p in selected_periods)
    dm_opts = "".join(f'<option value="{d}">{dmode_labels[d]}</option>' for d in DATE_MODES)
    st_opts = "".join(f'<option value="{s}">{s}</option>' for s in STATUSES)
    boc_opt = "".join(f'<option value="{b}">{b}</option>' for b in BOCS)
    mt_opts = '<option value="share">Share (%)</option><option value="volume">Volume</option>'

    # Main KPI 1-6 sticky controls
    controls_html = (
        '<div style="padding:12px 24px;background:#fff;border-bottom:2px solid #d9e2ec;'
        'position:sticky;top:0;z-index:1000;display:flex;flex-wrap:wrap;gap:6px 10px;'
        'align-items:center;box-shadow:0 4px 14px rgba(16,42,67,.07);">'
        f'<span style="{ls}">Year:</span>{sel("r-year", y_opts)}'
        f'<span style="{ls}">Granularity:</span>{sel("r-gran", g_opts)}'
        f'<span style="{ls}">Period:</span><select id="r-sub" onchange="upd()" style="{cs}"></select>'
        f'<span style="{ls}">Date Rule:</span>{sel("r-dmode", dm_opts)}'
        f'<span style="{ls}">Asset Status:</span>{sel("r-status", st_opts)}'
        f'<span style="{ls}">Vehicle Type:</span>{sel("r-boc", boc_opt)}'
        '</div>'
    )

    # KPI 7 separate filter bar (non-sticky, placed above kpi7 blocks)
    kpi7_controls_html = (
        '<div style="margin:32px 24px 0;padding:14px 20px;background:#f7f9fc;'
        'border:1px solid #d9e2ec;border-radius:12px;display:flex;flex-wrap:wrap;'
        'gap:6px 10px;align-items:center;">'
        f'<span style="{ls}">Asset Status:</span>{sel("r-k7status", st_opts)}'
        f'<span style="{ls}">Vehicle Type:</span>{sel("r-k7boc", boc_opt)}'
        f'<span style="{ls}">Metric:</span>{sel("r-k7metric", mt_opts)}'
        '</div>'
    )

    # ── JS ────────────────────────────────────────────────────────────────────
    js = """<script>
function upd(){
    var year=document.getElementById('r-year').value;
    var gran=document.getElementById('r-gran').value;
    var ss=document.getElementById('r-sub');var cur=ss.value;
    if(gran==='monthly'){if(ss.options.length!==12){ss.innerHTML='';for(var i=1;i<=12;i++){var o=document.createElement('option');o.value=i;o.textContent=i;ss.appendChild(o);}}}
    else if(gran==='quarterly'){if(ss.options.length!==4){ss.innerHTML='';['Q1','Q2','Q3','Q4'].forEach(function(q){var o=document.createElement('option');o.value=q;o.textContent=q;ss.appendChild(o);});}}
    else{ss.innerHTML='<option value="ALL">Annual</option>';}
    var found=false;for(var i=0;i<ss.options.length;i++){if(String(ss.options[i].value)===String(cur)){ss.value=cur;found=true;break;}}
    if(!found&&ss.options.length>0)ss.selectedIndex=0;
    var sub=ss.value;
    var dmode=document.getElementById('r-dmode').value;
    var status=document.getElementById('r-status').value;
    var boc=document.getElementById('r-boc').value;
    var k7status=document.getElementById('r-k7status').value;
    var k7boc=document.getElementById('r-k7boc').value;
    var metric=document.getElementById('r-k7metric').value;
    var kpi7y=gran==='yearly'?'ALL':year;
    document.querySelectorAll('.rck').forEach(function(el){
        var m=el.dataset.year===year&&el.dataset.gran===gran&&String(el.dataset.sub)===String(sub)&&el.dataset.dmode===dmode&&el.dataset.boc===boc&&el.dataset.status===status;
        el.style.display=m?'block':'none';
    });
    document.querySelectorAll('.rk7').forEach(function(el){
        var m=el.dataset.year===kpi7y&&el.dataset.gran===gran&&el.dataset.boc===k7boc&&el.dataset.status===k7status&&el.dataset.metric===metric;
        el.style.display=m?'block':'none';
    });
    window.dispatchEvent(new Event('resize'));
}
window.onload=upd;
</script>"""

    # ── KPI cards blocks ──────────────────────────────────────────────────────
    card_blocks: list[str] = []
    for year_str in selected_years:
        y = int(year_str)
        for gran in selected_periods:
            subs = ([str(i) for i in range(1, 13)] if gran == "monthly" else
                    ["Q1", "Q2", "Q3", "Q4"]       if gran == "quarterly" else ["ALL"])
            for sub in subs:
                if gran == "quarterly":
                    month_val      = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}[sub]
                    period_display = f"{sub} {year_str}"
                elif gran == "monthly":
                    month_val      = int(sub)
                    period_display = f"{_cal.month_abbr[month_val]}-{year_str}"
                else:
                    month_val      = "ALL"
                    period_display = year_str

                for dmode in DATE_MODES:
                    for boc in BOCS:
                        for status in STATUSES:
                            row = lookup_kpi_row(country, y, month_val, boc, status, dmode)
                            vol = 0.0
                            if row is not None:
                                v = row.get("VOLUME")
                                if v is not None and pd.notna(v):
                                    vol = float(v)

                            def _get(field, _row=row):
                                if _row is None: return None
                                v = _row.get(field)
                                return float(v) if v is not None and pd.notna(v) else None

                            kpi_cards: list[dict] = []
                            kpi_rows:  list[dict] = []
                            for i, (_, cfg) in enumerate(kpi_configs):
                                if tuple_keys[i]:
                                    k1, k2 = tuple_keys[i]
                                    v1, v2 = _get(k1), _get(k2)
                                    disp = (f"{fmd.percent_or_na_precision(v1,2)} / "
                                            f"{fmd.percent_or_na_precision(v2,2)}") if v1 is not None else "N/A"
                                    sig  = "neutral"
                                else:
                                    val  = _get(kpi_fields[i])
                                    disp = fmd.percent_or_na_precision(val, 2) if val is not None else "N/A"
                                    sig  = fmd.kpi_limit_status(val, cfg.get("limit"))

                                kpi_cards.append({
                                    "label": cfg["label"], "result": disp,
                                    "status": sig, "show_legend": True,
                                    "params": [
                                        ("Country",      country),
                                        ("Period",       period_display),
                                        ("Vehicle type", boc),
                                        ("Asset status", status),
                                        ("Volume",       fmd.format_volume(vol)),
                                    ],
                                })
                                limit_desc = f", Limit {cfg['limit']}%" if cfg.get("limit") else ""
                                kpi_rows.append({
                                    "label":       cfg["label"].split(" - ")[-1],
                                    "description": cfg["description"] + limit_desc,
                                    "country":     country,
                                    "period":      period_display,
                                    "result":      disp,
                                    "volume":      fmd.format_volume(vol),
                                    "signal":      sig,
                                })

                            cards_html  = fmd.build_kpi_cards_section("Main KPIs", kpi_cards)
                            summary_df  = pd.DataFrame(kpi_rows)[
                                ["label", "country", "period", "result", "volume", "description", "signal"]
                            ]
                            tbl_html    = fmd.build_html_table_from_df(summary_df, "KPI Summary")
                            data_sub    = str(int(sub)) if gran == "monthly" else sub
                            card_blocks.append(
                                f'<div class="rck" data-year="{year_str}" data-gran="{gran}" '
                                f'data-sub="{data_sub}" data-dmode="{dmode}" '
                                f'data-boc="{boc}" data-status="{status}" '
                                f'style="display:none;padding:24px;">'
                                + cards_html + tbl_html + "</div>"
                            )

    # ── KPI-7 blocks ──────────────────────────────────────────────────────────
    kpi7_blocks: list[str] = []
    for gran in selected_periods:
        year_vals = ["ALL"] if gran == "yearly" else selected_years
        for year_val in year_vals:
            y     = int(year_val) if year_val != "ALL" else None
            sd    = f"{selected_years[0]}-01-01" if gran == "yearly" else f"{year_val}-01-01"
            ed    = f"{selected_years[-1]}-12-31" if gran == "yearly" else f"{year_val}-12-31"
            for boc in BOCS:
                for status in STATUSES:
                    for metric in METRICS:
                        pivot = lookup_kpi7(country, y, status, metric, gran, boc)
                        chart_html = ""
                        if pivot is not None and not pivot.empty:
                            y_label  = "All years" if gran == "yearly" else str(year_val)
                            fig      = fmd.figure_from_pivot(
                                pivot,
                                "Share (%)" if metric == "share" else "Volume",
                                gran.capitalize(),
                                f"Fuel Type {metric} – {y_label} | {country} | {status} | {boc}",
                            )
                            chart_html = fmd.build_kpi7_metadata_section(
                                country, status, metric, gran, boc, sd, ed, fig
                            )
                        tbl_df   = _kpi7_table_from_cache(country, y, status, metric, gran, boc)
                        tbl_html = fmd.build_html_table_from_df(
                            tbl_df,
                            f"Fuel Type {metric.capitalize()} Table – {gran.capitalize()}"
                        )
                        kpi7_blocks.append(
                            f'<div class="rk7" data-year="{year_val}" data-gran="{gran}" '
                            f'data-boc="{boc}" data-status="{status}" data-metric="{metric}" '
                            f'style="display:none;padding:0 24px 24px;">'
                            + chart_html + tbl_html + "</div>"
                        )

    return fmd.html_report_document(
        f"Fleet Monitoring - View 1 - Complete Report",
        [controls_html] + card_blocks + [kpi7_controls_html] + kpi7_blocks + [js],
    )


@app.callback(
    Output("complete-report-download", "data"),
    Input("download-complete-report-btn", "n_clicks"),
    State("report-modal-years",   "value"),
    State("report-modal-periods", "value"),
    State("report-modal-country", "value"),
    prevent_initial_call=True,
)
def download_complete_report(n_clicks, selected_years, selected_periods, country):
    if not n_clicks or not selected_years or not selected_periods:
        return no_update
    country = country or (_avail_countries[0] if _avail_countries else "ALL")
    html_content = _build_complete_report_from_cache(country, selected_years, selected_periods)
    return dcc.send_string(html_content, filename=f"Complete_Report_{country}.html")


def _build_complete_report_v2_from_cache(
    country: str,
    selected_years: list[str],
    selected_periods: list[str],
) -> str:
    ASSET_STATUS = "IN FLEET"
    METRICS = ["share", "volume"]
    BOCS    = _avail_boc if _avail_boc else ["ALL", "CAR"]
    DMODES  = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]

    dmode_labels = {
        "NONE": "None",
        "CONTRACT_START_DATE": "Contract Start",
        "DELIVERY_DATE": "Delivery Date",
    }

    cs = "padding:6px 10px;border-radius:8px;border:1px solid #d9e2ec;font-size:12px;color:#102a43;"
    ls = "color:#627d98;font-size:12px;font-weight:600;margin-left:10px;"

    def sel(id_, opts_html):
        return f'<select id="{id_}" onchange="upd8()" style="{cs}">{opts_html}</select>'

    y_opts   = "".join(f'<option value="{y}">{y}</option>' for y in selected_years)
    g_opts   = "".join(f'<option value="{p}">{p.capitalize()}</option>' for p in selected_periods)
    mt_opts  = '<option value="share">Share (%)</option><option value="volume">Volume</option>'
    boc_opts = "".join(f'<option value="{b}">{b}</option>' for b in BOCS)
    dm_opts  = "".join(f'<option value="{d}">{dmode_labels[d]}</option>' for d in DMODES)

    controls_html = (
        '<div style="padding:12px 24px;background:#fff;border-bottom:2px solid #d9e2ec;'
        'position:sticky;top:0;z-index:1000;display:flex;flex-wrap:wrap;gap:6px 10px;'
        'align-items:center;box-shadow:0 4px 14px rgba(16,42,67,.07);">'
        f'<span style="{ls}">Year:</span>{sel("r8-year", y_opts)}'
        f'<span style="{ls}">Granularity:</span>{sel("r8-gran", g_opts)}'
        f'<span style="{ls}">Metric:</span>{sel("r8-metric", mt_opts)}'
        f'<span style="{ls}">Vehicle Type:</span>{sel("r8-boc", boc_opts)}'
        f'<span style="{ls}">Date Rule:</span>{sel("r8-dmode", dm_opts)}'
        '</div>'
    )

    js = """<script>
function upd8(){
    var year=document.getElementById('r8-year').value;
    var gran=document.getElementById('r8-gran').value;
    var metric=document.getElementById('r8-metric').value;
    var boc=document.getElementById('r8-boc').value;
    var dmode=document.getElementById('r8-dmode').value;
    document.querySelectorAll('.rk8').forEach(function(el){
        var m=el.dataset.year===year&&el.dataset.gran===gran&&el.dataset.metric===metric&&el.dataset.boc===boc&&el.dataset.dmode===dmode;
        el.style.display=m?'block':'none';
    });
    window.dispatchEvent(new Event('resize'));
}
window.onload=upd8;
</script>"""

    blocks: list[str] = []
    for year_str in selected_years:
        y = int(year_str)
        for gran in selected_periods:
            for metric in METRICS:
                for boc in BOCS:
                    for dmode in DMODES:
                        raw = lookup_kpi8(country, y, ASSET_STATUS, metric, boc, dmode, gran)
                        chart_html = ""
                        tbl_html   = ""

                        if raw is not None and not raw.empty:
                            period_col  = "PERIOD" if "PERIOD" in raw.columns else raw.columns[0]
                            metric_cols = [c for c in raw.columns if c not in [period_col, "YEAR", "MONTH"]]
                            color_map   = fmd.fuel_color_map(metric_cols)  # type: ignore[arg-type]
                            ordered     = sorted(
                                metric_cols,
                                key=lambda n: (fmd.fuel_label_priority(n), metric_cols.index(n)),
                            )

                            fig = go.Figure()
                            for col in ordered:
                                color = color_map[col]
                                fig.add_trace(go.Scatter(
                                    x=raw[period_col].tolist(),
                                    y=raw[col].tolist(),
                                    mode="lines+markers",
                                    name=str(col),
                                    line={"color": color, "width": 2.2},
                                    marker={"color": color, "size": 7},
                                ))
                            cap     = "Share (%)" if metric == "share" else "Volume"
                            x_label = {"monthly": "Month", "quarterly": "Quarter", "yearly": "Year"}.get(gran, "Period")
                            fig.update_layout(
                                title=None,
                                xaxis_title=x_label, yaxis_title=cap,
                                template="plotly_white", height=460,
                                margin=dict(t=20, b=50),
                            )
                            section_title = (
                                f"Production Fuel Type {metric.capitalize()} – {gran.capitalize()}"
                            )
                            subtitle = (
                                f"{country} | {year_str} | {boc} | "
                                f"{dmode_labels[dmode]} | {ASSET_STATUS}"
                            )
                            chart_html = (
                                f'<div class="section"><h2>{section_title}</h2>'
                                f'<p class="muted">{subtitle}</p>'
                                + fmd.figure_to_html_block(fig)
                                + "</div>"
                            )

                            display_df = fmd.build_view2_table_with_total(raw, metric)
                            display_df = fmd.format_view2_table_for_display(display_df)
                            tbl_html   = fmd.build_html_table_from_df(
                                display_df,
                                f"Production by Fuel Type – {metric.capitalize()} – {gran.capitalize()}",
                            )

                        blocks.append(
                            f'<div class="rk8" data-year="{year_str}" data-gran="{gran}" '
                            f'data-metric="{metric}" data-boc="{boc}" data-dmode="{dmode}" '
                            f'style="display:none;padding:24px;">'
                            + chart_html + tbl_html + "</div>"
                        )

    return fmd.html_report_document(
        "Fleet Monitoring - View 2 - Complete Report",
        [controls_html] + blocks + [js],
    )


@app.callback(
    Output("v2-complete-report-download", "data"),
    Input("v2-download-complete-report-btn", "n_clicks"),
    State("v2-report-modal-years",   "value"),
    State("v2-report-modal-periods", "value"),
    State("v2-report-modal-country", "value"),
    prevent_initial_call=True,
)
def download_v2_complete_report(n_clicks, selected_years, selected_periods, country):
    if not n_clicks or not selected_years or not selected_periods:
        return no_update
    country = country or (_avail_countries[0] if _avail_countries else "ALL")
    html_content = _build_complete_report_v2_from_cache(country, selected_years, selected_periods)
    return dcc.send_string(html_content, filename=f"Complete_Report_V2_{country}.html")


# =============================================================================
# 11.  VIEWS 3-6 CALLBACKS  (fast parquet lookups)
# =============================================================================

# ── V3 KPI 9_1 ────────────────────────────────────────────────────────────────

@app.callback(
    Output("v3-kpi91-graph", "figure"),
    Output("v3-kpi91-table-wrap", "children"),
    Input("view3-refresh-button-91", "n_clicks"),
    State("v3-country-filter-91", "value"),
    State("v3-year-filter-91", "value"),
    State("v3-status-filter-91", "value"),
    State("v3-period-filter-91", "value"),
    State("v3-bike-or-car-filter-91", "value"),
    State("v3-vehicle-class-filter-91", "value"),
    State("v3-vehicle-body-filter-91", "value"),
    prevent_initial_call=True,
)
def fast_update_view3_kpi91(_n, country, year, status_group, period_mode, bike_or_car, vehicle_class, vehicle_body):
    empty_fig = go.Figure()

    if fmd.has_missing_filters(country, year, status_group, period_mode, bike_or_car):
        empty_fig.update_layout(title="KPI 9_1 – Select filters then click Reload")
        return empty_fig, html.Div("Select filters then click Reload.", className="small-note")

    if CACHE_KPI91.empty:
        empty_fig.update_layout(title="KPI 9_1 – No precomputed data")
        return empty_fig, html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    if isinstance(status_group, list):
        selected_statuses = [str(v).strip() for v in status_group if v is not None and str(v).strip()]
    else:
        selected_statuses = [str(status_group).strip()] if status_group else []

    if not selected_statuses:
        empty_fig.update_layout(title="KPI 9_1 – Select at least one Asset Status")
        return empty_fig, html.Div("Select at least one Asset Status.", className="small-note")

    res = lookup_kpi91(
        country or "ALL", year, selected_statuses,
        period_mode or "monthly", bike_or_car or "ALL",
        vehicle_class or "ALL", vehicle_body or "ALL",
    )

    if res is None or res.empty:
        empty_fig.update_layout(title="KPI 9_1 – No data for the selected filters")
        return empty_fig, html.Div("No data for selected filters.", className="small-note")

    fig = go.Figure()
    table_frames: list = []
    status_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf", "#8c564b"]

    for idx, sv in enumerate(selected_statuses):
        sub = res[res["ASSET_STATUS"] == sv].copy() if "ASSET_STATUS" in res.columns else res.copy()
        if sub.empty:
            continue
        color   = status_colors[idx % len(status_colors)]
        periods = sub["PERIOD"].tolist()
        pcts    = sub["PCT"].tolist()
        fig.add_trace(go.Bar(x=periods, y=pcts, name=f"{sv} (bar)",
                             marker_color=color, opacity=0.6, legendgroup=sv))
        fig.add_trace(go.Scatter(x=periods, y=pcts, mode="lines+markers",
                                 name=f"{sv} (line)",
                                 line={"dash": "dot", "color": color, "width": 2},
                                 marker={"color": color, "size": 7}, legendgroup=sv))
        sub_display = sub.copy()
        sub_display["ASSET_STATUS"] = sv
        table_frames.append(sub_display)

    vc_label = vehicle_class or "ALL"
    fig.update_layout(
        title=f"KPI 9_1 – {vc_label} share ({year}, {', '.join(selected_statuses)}, {bike_or_car})",
        xaxis_title="Period", yaxis_title="Share (%)",
        template="plotly_white", height=420, barmode="group",
    )

    if not table_frames:
        wrap = html.Div("No data for KPI 9_1.", className="small-note")
    else:
        tbl = pd.concat(table_frames, ignore_index=True)
        tbl = tbl.rename(columns={"PERIOD": "Period", "PCT": "Share (%)", "ASSET_STATUS": "Asset Status"})
        pivot = tbl.pivot_table(index="Asset Status", columns="Period", values="Share (%)", aggfunc="first")
        pivot = pivot.reindex(index=selected_statuses)
        pivot = pivot.reset_index()
        pivot.columns.name = None
        wrap = html.Div([html.H4("KPI 9_1 table", className="panel-title"),
                         fmd.build_table(pivot, page_size=15)])

    return fig, wrap


# ── V3 KPI 9_2 ────────────────────────────────────────────────────────────────

@app.callback(
    Output("v3-kpi92-graph", "figure"),
    Output("v3-kpi92-table-wrap", "children"),
    Input("view3-refresh-button-92", "n_clicks"),
    State("v3-country-filter-92", "value"),
    State("v3-year-filter-92", "value"),
    State("v3-status-filter-92", "value"),
    State("v3-period-filter-92", "value"),
    State("v3-bike-or-car-filter-92", "value"),
    State("v3-vehicle-class-filter-92", "value"),
    State("v3-vehicle-body-filter-92", "value"),
    prevent_initial_call=True,
)
def fast_update_view3_kpi92(_n, country, year, status, period_mode, bike_or_car, vehicle_class, vehicle_body):
    empty_fig = go.Figure()

    if fmd.has_missing_filters(country, year, status, period_mode, bike_or_car):
        empty_fig.update_layout(title="KPI 9_2 – Select filters then click Reload")
        return empty_fig, html.Div("Select filters then click Reload.", className="small-note")

    if CACHE_KPI92.empty:
        empty_fig.update_layout(title="KPI 9_2 – No precomputed data")
        return empty_fig, html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    res = lookup_kpi92(
        country or "ALL", year, status or "IN FLEET",
        period_mode or "monthly", bike_or_car or "ALL",
        vehicle_class or "ALL", vehicle_body or "ALL",
    )

    fig = go.Figure()

    if res is None or res.empty:
        empty_fig.update_layout(title="KPI 9_2 – No data for selected filters")
        return empty_fig, html.Div("No data for selected filters.", className="small-note")

    period_col = "Period"
    power_cols = [c for c in res.columns if c != period_col]
    for col in power_cols:
        fig.add_trace(go.Scatter(
            x=res[period_col].tolist(), y=res[col].tolist(),
            mode="lines+markers", name=str(col),
        ))

    vc_label = vehicle_class or "ALL"
    fig.update_layout(
        title=f"KPI 9_2 – {vc_label} power mix ({year}, {status}, {bike_or_car})",
        xaxis_title="Period", yaxis_title="Volume",
        template="plotly_white", height=420,
    )

    wrap = html.Div([html.H4("KPI 9_2 table (pivot)", className="panel-title"),
                     fmd.build_table(res.copy(), page_size=15)])
    return fig, wrap


# ── V4 KPI 10 ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("v4-kpi10-graph", "figure"),
    Output("v4-kpi10-table-wrap", "children"),
    Input("view4-refresh-button", "n_clicks"),
    State("v4-country-filter", "value"),
    State("v4-year-filter", "value"),
    State("v4-status-filter", "value"),
    State("v4-period-filter", "value"),
    State("v4-bike-or-car-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view4_kpi10(_n, country, year, status, period_mode, bike_or_car):
    empty_fig = go.Figure()

    if fmd.has_missing_filters(country, year, status, period_mode, bike_or_car):
        empty_fig.update_layout(title="KPI 10 – Select filters then click Reload")
        return empty_fig, html.Div("Select filters then click Reload.", className="small-note")

    if CACHE_KPI10.empty:
        empty_fig.update_layout(title="KPI 10 – No precomputed data")
        return empty_fig, html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    table = lookup_kpi10(
        country or "ALL", year,
        status or "IN FLEET", period_mode or "monthly", bike_or_car or "ALL",
    )

    if table is None or table.empty:
        empty_fig.update_layout(title="KPI 10 – No data for selected filters")
        return empty_fig, html.Div("No data for selected filters.", className="small-note")

    period_col = "Period"
    model_cols = [c for c in table.columns if c != period_col]
    is_total   = table[period_col].astype(str) == "Total"
    quarters   = table.loc[~is_total, period_col].tolist()

    if not quarters or not model_cols:
        empty_fig.update_layout(title="KPI 10 – No data")
        return empty_fig, fmd.build_table(table, page_size=15)

    fig = _make_subplots(specs=[[{"secondary_y": True}]])
    for model in model_cols:
        fig.add_trace(
            go.Bar(x=quarters, y=table.loc[~is_total, model].astype(float).tolist(), name=str(model)),
            secondary_y=False,
        )
    quarter_totals = table.loc[~is_total, model_cols].sum(axis=1)
    fig.add_trace(
        go.Scatter(x=quarters, y=quarter_totals.tolist(), mode="lines+markers", name="TOTAL"),
        secondary_y=True,
    )
    x_label = {"monthly": "Month", "quarterly": "Quarter", "yearly": "Year"}.get(period_mode or "monthly", "Period")
    fig.update_layout(
        title=f"KPI 10 – Top models at EOC ({country}, {year}, {status}, {bike_or_car})",
        barmode="group", template="plotly_white", height=460, xaxis_title=x_label,
    )
    fig.update_yaxes(title_text="Volume by model", secondary_y=False)
    fig.update_yaxes(title_text="Total volume", secondary_y=True)

    return fig, fmd.build_table(table, page_size=15)


# ── V5 UI sync (source toggle) ─────────────────────────────────────────────────

@app.callback(
    Output("v5-status-container", "style"),
    Output("v5-market-reg-container", "style"),
    Output("v5-variable-filter-label", "children"),
    Output("v5-variable-filter", "options"),
    Output("v5-variable-filter", "value"),
    Output("v5-market-owner-container", "style"),
    Input("v5-source-filter", "value"),
)
def sync_v5_source_filters(source):
    if source == "market":
        status_style     = {"display": "none"}
        market_reg_style = {"display": "block"}
        var_label        = "Market Variable"
        var_options      = fmd.CONCENTRATION_VARIABLE_OPTIONS_MARKET
        owner_style      = {"display": "block"}
    else:
        status_style     = {"display": "block"}
        market_reg_style = {"display": "none"}
        var_label        = "Portfolio Variable"
        var_options      = fmd.CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO
        owner_style      = {"display": "none"}
    var_value = var_options[0]["value"] if var_options else None
    return status_style, market_reg_style, var_label, var_options, var_value, owner_style


# ── V5 KPI 11 – Global Concentration ──────────────────────────────────────────

@app.callback(
    Output("v5-kpi11-graph", "figure"),
    Output("v5-kpi11-table-wrap", "children"),
    Input("view5-refresh-button", "n_clicks"),
    State("v5-status-filter", "value"),
    State("v5-market-reg-filter", "value"),
    State("v5-source-filter", "value"),
    State("v5-variable-filter", "value"),
    State("v5-top-n-filter", "value"),
    State("v5-period-filter", "value"),
    State("v5-market-owner-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view5_kpi11(_n, status, market_reg, source, variable, top_n, period_mode, market_owner):
    empty_fig = go.Figure()
    active_filter = market_reg if source == "market" else status

    if fmd.has_missing_filters(active_filter, source, variable):
        empty_fig.update_layout(title="KPI 11 – Select filters then click Reload")
        return empty_fig, html.Div("Select filters then click Reload.", className="small-note")

    if CACHE_KPI11.empty:
        empty_fig.update_layout(title="KPI 11 – No precomputed data")
        return empty_fig, html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    top_n = top_n or "ALL"
    if source == "market":
        df11 = lookup_kpi11(
            source="market",
            reg_type=market_reg,
            owner=market_owner or "ALL",
            variable=variable,
            top_n=top_n,
            period_mode=period_mode or "monthly",
        )
    else:
        df11 = lookup_kpi11(
            source="portfolio",
            status=status or "IN FLEET",
            variable=variable,
            top_n=top_n,
            period_mode=period_mode or "monthly",
        )

    if df11 is None or df11.empty:
        empty_fig.update_layout(title="KPI 11 – No data for selected filters")
        return empty_fig, html.Div("No data for selected filters.", className="small-note")

    var_col = _kpi11_var_col(source, variable)

    # Normalise YTD columns so plot and table always see VOLUME / SHARE
    if "VOLUME_YTD" in df11.columns:
        df11 = df11.rename(columns={"VOLUME_YTD": "VOLUME", "SHARE_YTD": "SHARE"})

    try:
        fig = fmd.plot_kpi_share(df11, var_col, period_mode or "monthly")
    except Exception as _e:
        empty_fig.update_layout(title=f"KPI 11 – Erreur rendu: {str(_e)[:120]}")
        fig = empty_fig

    # Table: drop TOTAL (internal) for clean display
    table_df = df11.drop(columns=["TOTAL", "TOTAL_YTD"], errors="ignore")
    table = fmd.build_table(table_df, page_size=15, heatmap=True, table_id="v5-kpi11-table")
    return fig, html.Div([html.H4("KPI 11 – Concentration", className="panel-title"), table])


# ── V5 Country Breakdown (shared section on vue-5) ────────────────────────────

def _build_kpi12_table(df: pd.DataFrame, metric: str, variable: str) -> pd.DataFrame:
    temp = df.copy()
    m = metric.upper()
    if m not in temp.columns:
        return temp
    temp["COL_NAME"] = temp["YEAR"].astype(str) + "_" + temp["PERIOD"].astype(str) + "_" + m
    pivot_metric = temp.pivot_table(index="COUNTRY", columns="COL_NAME", values=m, aggfunc="first").reset_index()
    pivot_metric.columns.name = None
    if variable in temp.columns:
        pivot_var = temp.pivot_table(index="COUNTRY", columns="COL_NAME", values=variable, aggfunc="first").reset_index()
        pivot_var.columns.name = None
        pivot_var.columns = [col.replace(m, "VAR") if col != "COUNTRY" else col for col in pivot_var.columns]
        return pivot_metric.merge(pivot_var, on="COUNTRY", how="left")
    return pivot_metric


@app.callback(
    Output("v6-kpi6-table-wrap", "children"),
    Input("view6-refresh-button", "n_clicks"),
    State("v6-status-filter", "value"),
    State("v6-source-filter", "value"),
    State("v6-variable-filter", "value"),
    State("v6-metric-mode-filter", "value"),
    State("v6-top-n-filter", "value"),
    State("v6-period-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view5_country_breakdown(_n, status, source, variable, metric_mode, top_n, period_mode):
    if fmd.has_missing_filters(status, source, variable, metric_mode):
        return html.Div("Select filters then click Reload.", className="small-note")

    if CACHE_KPI11.empty:
        return html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    df_cb = lookup_kpi11(
        source=source or "portfolio",
        status=status or "IN FLEET",
        variable=variable,
        top_n=top_n or "ALL",
        period_mode=period_mode or "monthly",
    )

    if df_cb is None or df_cb.empty:
        return html.Div("No data for selected filters.", className="small-note")

    var_col_cb = _kpi11_var_col(source or "portfolio", variable)
    metric = metric_mode or "SHARE"
    kpi12_df = _build_kpi12_table(df_cb, metric, var_col_cb)
    table = fmd.build_table(kpi12_df, page_size=15, heatmap=True, table_id="v5-kpi12-table")
    return html.Div([html.H4("KPI 12 – Concentration par pays", className="panel-title"), table])


# ── V6 UI sync ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("v6-variable-filter", "options"),
    Input("v6-source-filter", "value"),
)
def update_v6_variable_options(source):
    if source == "market":
        return fmd.CONCENTRATION_VARIABLE_OPTIONS_MARKET
    return fmd.CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO


# ── V6 KPI 13 – Portfolio vs Market ───────────────────────────────────────────

_KPI13_VAR_NORMALIZE = {
    "BRAND": "BRAND_UPDATE",
    "OEM":   "OEM_UPDATE",
    "BEV":   "HIGHEST_BEV",
    "CO2":   "CO2_BUCKET",
}

@app.callback(
    Output("v6-kpi13-graph", "figure"),
    Output("v6-kpi13-table-wrap", "children"),
    Input("view6-kpi13-refresh-button", "n_clicks"),
    State("v6-kpi13-status-filter", "value"),
    State("v6-kpi13-port-regtype-filter", "value"),
    State("v6-kpi13-mkt-regtype-filter", "value"),
    State("v6-kpi13-variable-filter", "value"),
    State("v6-kpi13-period-filter", "value"),
    State("v6-kpi13-top-n-filter", "value"),
    State("v6-kpi13-owner-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view6_kpi13(_n, status, port_reg, mkt_reg, variable, period_mode, top_n, owner):
    empty_fig = go.Figure()
    variable = _KPI13_VAR_NORMALIZE.get(str(variable), variable) if variable else variable

    if fmd.has_missing_filters(status, port_reg, mkt_reg, variable):
        empty_fig.update_layout(title="KPI 13 – Select filters then click Reload")
        return empty_fig, html.Div("Select filters then click Reload.", className="small-note")

    if CACHE_KPI13.empty:
        empty_fig.update_layout(title="KPI 13 – No precomputed data")
        return empty_fig, html.Div("No precomputed data. Run generate_all_precomputed.py first.", className="small-note")

    df13 = lookup_kpi13(
        status=status or "IN FLEET",
        port_reg=port_reg or ["PV"],
        mkt_reg=mkt_reg or ["Passenger Cars"],
        variable=variable,
        period_mode=period_mode or "monthly",
        top_n=top_n or "ALL",
        owner=owner or "ALL",
    )

    if df13 is None or df13.empty:
        empty_fig.update_layout(title="KPI 13 – No data for selected filters")
        return empty_fig, html.Div("No data for selected filters.", className="small-note")

    reg_p = port_reg if isinstance(port_reg, str) else ", ".join(map(str, port_reg or []))
    reg_m = mkt_reg  if isinstance(mkt_reg,  str) else ", ".join(map(str, mkt_reg  or []))
    title = f"KPI 13 – {variable} Portfolio vs Market ({status} / P:{reg_p} / M:{reg_m})"

    try:
        fig = fmd.figure_top_brand_vs_market(df13, title)
    except Exception:
        fig = empty_fig

    columns  = ["COUNTRY", "PERIOD", "BRAND", "volume_portfolio", "share_portfolio",
                 "volume_market", "share_market", "ratio"]
    table_df = df13[[c for c in columns if c in df13.columns]].copy()
    table    = fmd.build_table(table_df, page_size=15)
    return fig, html.Div([html.H4("KPI 13 table", className="panel-title"), table])


# =============================================================================
# 12.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\nStarting Fast Dashboard on http://127.0.0.1:8052/")
    app.run(debug=True, port=8052)
