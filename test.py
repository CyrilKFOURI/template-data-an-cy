
"""
Fast Fleet Dashboard – Read-Only Parquet Reader
================================================
Run generate_all_precomputed.py first to build the Parquet cache.
Then run this file to get instant dashboard responses.

  python precomputed_fast/fast_dashboard_reader.py
"""

from __future__ import annotations

import os
import sys
import calendar
import pandas as pd
import plotly.graph_objects as go
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
                except Exception as e:
                    print(f"  [warn] Could not read {f}: {e}")
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
        print(f"  [cache] reading merged file: {os.path.basename(merged_path)}")
        try:
            return pd.read_parquet(merged_path)
        except Exception as e:
            print(f"  [warn] merged file failed ({e}), falling back to tree walk")
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

print(f"  CACHE_V1   : {len(CACHE_V1):,} rows | cols: {list(CACHE_V1.columns)[:8]}")
print(f"  CACHE_KPI7 : {len(CACHE_KPI7):,} rows | cols: {list(CACHE_KPI7.columns)[:8]}")
print(f"  CACHE_KPI8 : {len(CACHE_KPI8):,} rows | cols: {list(CACHE_KPI8.columns)[:8]}")


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


# Modal toggle
@app.callback(
    Output("complete-report-modal", "style"),
    [
        Input({"type": "open-complete-report", "index": dash.ALL}, "n_clicks"),
        Input("close-complete-report-modal", "n_clicks"),
        Input("download-complete-report-btn", "n_clicks"),
    ],
    [State("complete-report-modal", "style")],
    prevent_initial_call=True,
)
def toggle_modal(open_clicks_list, close_clicks, dl_clicks, current_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    triggered_id    = ctx.triggered[0]["prop_id"]
    triggered_value = ctx.triggered[0].get("value", 0) or 0
    hidden = {"display": "none", "position": "fixed", "top": "0", "left": "0",
              "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)",
              "zIndex": "9999", "justifyContent": "center", "alignItems": "center"}
    shown  = {**hidden, "display": "flex"}
    if "open-complete-report" in triggered_id and triggered_value > 0:
        return shown
    if "close-complete-report-modal" in triggered_id or "download-complete-report-btn" in triggered_id:
        return hidden
    return no_update


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


@app.callback(
    Output("complete-report-download", "data"),
    Input("download-complete-report-btn", "n_clicks"),
    State("report-modal-years", "value"),
    State("report-modal-periods", "value"),
    State("report-modal-date-mode", "value"),
    State("country-filter", "value"),
    State("kpi-common-bike-or-car-filter", "value"),
    State("status-group-filter", "value"),
    prevent_initial_call=True,
)
def download_complete_report(n_clicks, *args):
    if not n_clicks:
        return no_update
    return fmd.generate_interactive_report_callback(n_clicks, *args)


# =============================================================================
# 11.  VIEWS 3-6 CALLBACKS  (delegate entirely to fmd – they use raw data)
# =============================================================================

_NA_FIG = go.Figure().update_layout(title="Not available in fast mode")
_NA_DIV = html.Div("Not available in fast mode — raw data not loaded.", className="small-note")


@app.callback(
    Output("v3-kpi91-graph", "figure"),
    Output("v3-kpi91-table-wrap", "children"),
    Output("v3-kpi92-graph", "figure"),
    Output("v3-kpi92-table-wrap", "children"),
    Input("view3-refresh-button", "n_clicks"),
    State("v3-country-filter", "value"),
    State("v3-year-filter", "value"),
    State("v3-vehicle-type-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view3(*_):
    return _NA_FIG, _NA_DIV, _NA_FIG, _NA_DIV


@app.callback(
    Output("v4-kpi10-graph", "figure"),
    Output("v4-kpi10-table-wrap", "children"),
    Input("view4-refresh-button", "n_clicks"),
    State("v4-country-filter", "value"),
    State("v4-status-filter", "value"),
    State("v4-bike-or-car-filter", "value"),
    State("v4-start-date-filter", "value"),
    State("v4-end-date-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view4(*_):
    return _NA_FIG, _NA_DIV


@app.callback(
    Output("v5-kpi11-graph", "figure"),
    Output("v5-kpi11-table-wrap", "children"),
    Input("view5-refresh-button", "n_clicks"),
    State("v5-status-filter", "value"),
    State("v5-source-filter", "value"),
    State("v5-variable-filter", "value"),
    State("v5-top-n-filter", "value"),
    State("v5-period-filter", "value"),
    State("v5-market-owner-filter", "value"),
    prevent_initial_call=True,
)
def fast_update_view5(*_):
    return _NA_FIG, _NA_DIV


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
def fast_update_view6(*_):
    return _NA_FIG, _NA_DIV


# =============================================================================
# 12.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\nStarting Fast Dashboard on http://127.0.0.1:8052/")
    app.run(debug=True, port=8052)
