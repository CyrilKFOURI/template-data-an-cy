from __future__ import annotations

import glob
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
        return pd.DataFrame()
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
    """Filter to IN FLEET CAR, enrich with market model/body type."""
    if df.empty:
        return df
    d = df[
        (df["NOVA_ASSET_STATUS"] == "IN FLEET") &
        (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
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

# Axes et métriques identiques au notebook use_case_1_heatmap.ipynb
Y_OPTIONS = [
    {"label": "Brand",                      "value": "BRAND_UPDATE"},
    {"label": "Power Category",             "value": "POWER_CATEGORY"},
    {"label": "CO2 Bucket",                 "value": "CO2_BUCKET"},
    {"label": "Industry (Arval CLS)",       "value": "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"},
    {"label": "Shared Client Flag",         "value": "SHARED_CLIENT_FLAG"},
]

X_OPTIONS = [
    {"label": "Group Rating",               "value": "GROUP_RATING"},
    {"label": "Counterparty Rating",        "value": "COUNTERPARTY_RATING"},
    {"label": "CLS Group Rating",           "value": "CLS_GROUP_RATING"},
    {"label": "Industry (Arval CLS)",       "value": "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"},
    {"label": "Shared Client Flag",         "value": "SHARED_CLIENT_FLAG"},
]

# Y axis options for simulation only (= notebook s_y — sans SHARED_CLIENT_FLAG)
SIM_Y_OPTIONS = [
    {"label": "Brand",                      "value": "BRAND_UPDATE"},
    {"label": "Power Category",             "value": "POWER_CATEGORY"},
    {"label": "CO2 Bucket",                 "value": "CO2_BUCKET"},
    {"label": "Industry (Arval CLS)",       "value": "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"},
]

METRIC_OPTIONS = [
    {"label": "Volume (nb contrats)",       "value": "volume"},
    {"label": "Concentration Financière",   "value": "concentration_financiere"},
    {"label": "Intensité Risk Asset",       "value": "intensite_risk_asset"},
]


# =============================================================================
# Core helpers: filter → COB filter → pivot → figure
# =============================================================================

def apply_filters(country, brands, models, bodytypes) -> pd.DataFrame:
    d = UC1_DF.copy()
    if country and country != "ALL" and "COUNTRY" in d.columns:
        d = d[d["COUNTRY"] == country]
    if brands:
        d = d[d["BRAND_UPDATE"].isin(brands)]
    if models and "MARKET_MODEL" in d.columns:
        d = d[d["MARKET_MODEL"].isin(models)]
    if bodytypes and "MARKET_BODY_GROUP" in d.columns:
        d = d[d["MARKET_BODY_GROUP"].isin(bodytypes)]
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


def build_pivot(d: pd.DataFrame, y_col: str, x_col: str, metric: str) -> pd.DataFrame:
    metric_cols = {"EXPOSURE_AMOUNT_TOT", "VEHICLE_PRICE_EUR"}
    extra = {"COB_DATE"} | metric_cols
    keep = list({y_col, x_col} | {c for c in UNIQUE_KEY_COLS if c in d.columns} | {c for c in extra if c in d.columns})
    d = d[[c for c in keep if c in d.columns]].dropna(subset=[y_col, x_col])
    if d.empty:
        return pd.DataFrame()

    # Dedup: latest COB snapshot per unique contract key
    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    if keys:
        if "COB_DATE" in d.columns:
            d = d.sort_values("COB_DATE")
        d = d.drop_duplicates(subset=keys, keep="last")

    if metric == "concentration_financiere" and "EXPOSURE_AMOUNT_TOT" in d.columns:
        piv = d.groupby([y_col, x_col])["EXPOSURE_AMOUNT_TOT"].sum().unstack(x_col, fill_value=0)
    elif metric == "intensite_risk_asset" and "VEHICLE_PRICE_EUR" in d.columns:
        piv = d.groupby([y_col, x_col])["VEHICLE_PRICE_EUR"].sum().unstack(x_col, fill_value=0)
    else:  # "volume" ou fallback
        piv = d.groupby([y_col, x_col]).size().unstack(x_col, fill_value=0)

    return piv


def make_heatmap_fig(
    piv: pd.DataFrame, title: str, colorscale: str = "Blues",
    page: int = 0, zmid=None, zmin=None, zmax=None,
) -> go.Figure:
    if piv.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font={"size": 14, "color": "#aaa"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220)
        return fig

    paged = piv.iloc[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]
    z = paged.values
    x_lbl = [str(c) for c in paged.columns]
    y_lbl = [str(r) for r in paged.index]

    has_floats = any(isinstance(v, float) and not np.isnan(v) for row in z for v in row)
    fmt = ".2f" if has_floats else ".0f"
    text = [
        [f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "" for v in row]
        for row in z
    ]

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


def build_sim_df(d_orig: pd.DataFrame, y_col: str, add_dict: dict, rem_dict: dict) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    d = d_orig.copy()

    for y_val, n in rem_dict.items():
        n = int(n or 0)
        if n <= 0:
            continue
        idx = d.index[d[y_col].astype(str) == str(y_val)].tolist()
        if idx:
            drop_idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
            d = d.drop(index=drop_idx)

    max_id = int(d_orig["ID_CONTRACT"].max()) if "ID_CONTRACT" in d_orig.columns and not d_orig["ID_CONTRACT"].isna().all() else 900000
    new_parts: list[pd.DataFrame] = []
    for y_val, n in add_dict.items():
        n = int(n or 0)
        if n <= 0:
            continue
        src = d_orig[d_orig[y_col].astype(str) == str(y_val)]
        if src.empty:
            continue
        sampled = src.sample(n=n, replace=True, random_state=42).copy().reset_index(drop=True)
        if "ID_CONTRACT" in sampled.columns:
            sampled["ID_CONTRACT"] = range(max_id + 1, max_id + 1 + len(sampled))
            max_id += len(sampled)
        new_parts.append(sampled)

    if new_parts:
        d = pd.concat([d] + new_parts, ignore_index=True)
    return d


# =============================================================================
# Dash app
# =============================================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "fleet_assets"),
)
app.title = "Use Case 1 — Portfolio Heatmap"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _lbl(options: list[dict], value: str) -> str:
    return next((o["label"] for o in options if o["value"] == value), value)


def _panel_title(text: str, **extra_style):
    style = {"fontWeight": "700", "fontSize": "13px", "color": "#1a1a2e",
              "marginBottom": "12px", "letterSpacing": "0.02em"}
    style.update(extra_style)
    return html.Div(text, style=style)


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Store(id="page-store",    data=0),
        dcc.Store(id="npages-store",  data=1),
        dcc.Store(id="refresh-ts",    data=0),
        dcc.Store(id="cob-store",     data={"granularity": "monthly", "period": _default_cob_period}),

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Use Case 1 — Portfolio Heatmap",
                                style={"margin": "0", "fontSize": "22px", "fontWeight": "700", "color": "#1a1a2e"}),
                        html.P("IN FLEET CAR portfolio · enriched with market model & body type · unique contract key dedup",
                               style={"margin": "4px 0 0", "fontSize": "13px", "color": "#718096"}),
                    ]
                ),
            ],
            style={"padding": "20px 28px 16px", "borderBottom": "1px solid #e2e8f0",
                   "background": "#ffffff"},
        ),

        # ── Global COB Date bar ──────────────────────────────────────────────
        html.Div(
            [
                html.Span("COB Date",
                          style={"fontWeight": "700", "fontSize": "11px", "color": "#718096",
                                 "letterSpacing": "0.06em", "textTransform": "uppercase",
                                 "marginRight": "14px", "whiteSpace": "nowrap"}),
                dcc.RadioItems(
                    id="cob-granularity",
                    options=[
                        {"label": " Monthly",   "value": "monthly"},
                        {"label": " Quarterly", "value": "quarterly"},
                        {"label": " Yearly",    "value": "yearly"},
                    ],
                    value="monthly",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "18px", "fontSize": "13px", "cursor": "pointer"},
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

                        html.Button("↺  Refresh", id="btn-refresh", n_clicks=0,
                                    style={"width": "100%", "padding": "8px", "fontWeight": "600",
                                           "fontSize": "13px", "background": "#3182ce",
                                           "color": "#fff", "border": "none", "borderRadius": "6px",
                                           "cursor": "pointer", "marginBottom": "8px"}),
                        html.Button("⚡  Simulation", id="btn-sim-toggle", n_clicks=0,
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
                                        _panel_title("Simulation Controls"),
                                        html.P(
                                            "Specify vehicles to add (+) or remove (−) per Y-axis category on the current page.",
                                            style={"fontSize": "12px", "color": "#718096",
                                                   "margin": "0 0 16px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Y Axis (Simulation)",
                                                           style={"fontSize": "11px", "fontWeight": "700",
                                                                  "color": "#4a5568", "marginBottom": "4px",
                                                                  "display": "block"}),
                                                dcc.Dropdown(
                                                    id="sim-y",
                                                    options=SIM_Y_OPTIONS,
                                                    value=SIM_Y_OPTIONS[0]["value"],
                                                    clearable=False,
                                                    style={"fontSize": "12px"},
                                                ),
                                            ],
                                            style={"marginBottom": "14px"},
                                        ),
                                        html.Div(id="sim-controls",
                                                 style={"maxHeight": "380px", "overflowY": "auto"}),
                                        html.Div(
                                            [
                                                html.Button("▶  Run Simulation", id="btn-sim-run", n_clicks=0,
                                                            style={"padding": "8px 20px", "fontWeight": "600",
                                                                   "fontSize": "13px", "background": "#276749",
                                                                   "color": "#fff", "border": "none",
                                                                   "borderRadius": "6px", "cursor": "pointer"}),
                                                html.Button("↺  Reset", id="btn-sim-reset", n_clicks=0,
                                                            style={"padding": "8px 20px", "fontWeight": "600",
                                                                   "fontSize": "13px", "background": "#f7fafc",
                                                                   "color": "#718096", "border": "1px solid #cbd5e0",
                                                                   "borderRadius": "6px", "cursor": "pointer"}),
                                            ],
                                            style={"display": "flex", "gap": "10px", "marginTop": "16px"},
                                        ),
                                    ],
                                    style={"background": "#ffffff", "borderRadius": "8px",
                                           "border": "1px solid #e2e8f0", "padding": "20px",
                                           "marginBottom": "16px"},
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
    State("cob-store",  "data"),
)
def _filter_opts(country, cob_store):
    cob = cob_store or {}
    d = apply_filters(country, None, None, None)
    d = apply_cob_filter(d, cob.get("granularity", "monthly"), cob.get("period") or "")
    brands  = [{"label": b,  "value": b}  for b  in sorted(d["BRAND_UPDATE"].dropna().unique())]
    models  = [{"label": m,  "value": m}  for m  in sorted(d["MARKET_MODEL"].dropna().unique())]  if "MARKET_MODEL"     in d.columns else []
    btypes  = [{"label": bt, "value": bt} for bt in sorted(d["MARKET_BODY_GROUP"].dropna().unique())] if "MARKET_BODY_GROUP" in d.columns else []
    return brands, models, btypes


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

@app.callback(
    Output("heatmap",       "figure"),
    Output("heatmap-title", "children"),
    Output("page-info",     "children"),
    Output("npages-store",  "data"),
    Input("page-store",  "data"),
    Input("refresh-ts",  "data"),
    State("f-country",   "value"),
    State("f-brand",     "value"),
    State("f-model",     "value"),
    State("f-bodytype",  "value"),
    State("f-y",         "value"),
    State("f-x",         "value"),
    State("f-metric",    "value"),
    State("cob-store",   "data"),
)
def _heatmap(page, _ts, country, brands, models, bodytypes, y_col, x_col, metric, cob_store):
    page   = page or 0
    y_col  = y_col  or "BRAND_UPDATE"
    x_col  = x_col  or "GROUP_RATING"
    metric = metric or "volume"
    cob    = cob_store or {}

    d = apply_filters(country, brands, models, bodytypes)
    d = apply_cob_filter(d, cob.get("granularity", "monthly"), cob.get("period") or "")

    piv     = build_pivot(d, y_col, x_col, metric)
    n_rows  = len(piv)
    n_pages = max(1, (n_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    page    = max(0, min(page, n_pages - 1))

    period_label = cob.get("period") or "All"
    title_text   = f"{_lbl(METRIC_OPTIONS, metric)}  ·  {_lbl(Y_OPTIONS, y_col)} × {_lbl(X_OPTIONS, x_col)}  [{period_label}]"
    fig          = make_heatmap_fig(piv, title_text, "Blues", page)
    info         = f"Page {page + 1} / {n_pages}  ({n_rows} categories)"
    return fig, title_text, info, n_pages


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
# Callbacks — Simulation controls (dynamic inputs per Y-value)
# =============================================================================

@app.callback(
    Output("sim-controls", "children"),
    Input("btn-sim-toggle", "n_clicks"),
    Input("page-store", "data"),
    Input("sim-y",      "value"),
    State("f-country",  "value"),
    State("f-brand",    "value"),
    State("f-model",    "value"),
    State("f-bodytype", "value"),
    State("cob-store",  "data"),
    prevent_initial_call=True,
)
def _sim_controls(_, page, y_col, country, brands, models, bodytypes, cob_store):
    y_col = y_col or SIM_Y_OPTIONS[0]["value"]
    cob   = cob_store or {}
    d = apply_filters(country, brands, models, bodytypes)
    d = apply_cob_filter(d, cob.get("granularity", "monthly"), cob.get("period") or "")

    if d.empty or y_col not in d.columns:
        return html.P("No data.", style={"color": "#999", "fontSize": "13px"})

    y_vals    = sorted(d[y_col].dropna().unique().tolist())
    page      = page or 0
    page_vals = y_vals[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]

    LBL = "190px"
    INP = "95px"
    header = html.Div(
        [
            html.Div("Category",  style={"minWidth": LBL, "fontSize": "11px", "color": "#718096", "fontWeight": "700"}),
            html.Div("Add +",     style={"minWidth": INP, "fontSize": "11px", "color": "#276749", "fontWeight": "700", "textAlign": "center"}),
            html.Div("Remove −",  style={"minWidth": INP, "fontSize": "11px", "color": "#9b2c2c", "fontWeight": "700", "textAlign": "center"}),
        ],
        style={"display": "flex", "gap": "10px", "padding": "4px 0 10px",
               "borderBottom": "1px solid #e2e8f0", "marginBottom": "8px"},
    )

    rows = [header]
    for y_val in page_vals:
        cnt = int((d[y_col].astype(str) == str(y_val)).sum())
        rows.append(html.Div(
            [
                html.Div(
                    [html.Span(str(y_val), style={"fontWeight": "500", "fontSize": "13px"}),
                     html.Span(f" ({cnt})", style={"fontSize": "11px", "color": "#aaa"})],
                    style={"minWidth": LBL, "display": "flex", "alignItems": "center"},
                ),
                dcc.Input(
                    id={"type": "sim-add", "index": str(y_val)},
                    type="number", min=0, step=10, value=0,
                    style={"width": INP, "textAlign": "center",
                           "border": "1px solid #c6f6d5", "borderRadius": "4px", "padding": "4px"},
                ),
                dcc.Input(
                    id={"type": "sim-rem", "index": str(y_val)},
                    type="number", min=0, step=10, value=0,
                    style={"width": INP, "textAlign": "center",
                           "border": "1px solid #fed7d7", "borderRadius": "4px", "padding": "4px"},
                ),
            ],
            style={"display": "flex", "gap": "10px", "alignItems": "center", "padding": "5px 0"},
        ))
    return rows


# =============================================================================
# Callbacks — Run / Reset simulation
# =============================================================================

@app.callback(
    Output("sim-result", "children"),
    Input("btn-sim-run",   "n_clicks"),
    Input("btn-sim-reset", "n_clicks"),
    State({"type": "sim-add", "index": ALL}, "value"),
    State({"type": "sim-add", "index": ALL}, "id"),
    State({"type": "sim-rem", "index": ALL}, "value"),
    State({"type": "sim-rem", "index": ALL}, "id"),
    State("f-country",   "value"),
    State("f-brand",     "value"),
    State("f-model",     "value"),
    State("f-bodytype",  "value"),
    State("sim-y",       "value"),
    State("f-x",         "value"),
    State("f-metric",    "value"),
    State("page-store",  "data"),
    State("cob-store",   "data"),
    prevent_initial_call=True,
)
def _sim_result(run, reset, add_vals, add_ids, rem_vals, rem_ids,
                country, brands, models, bodytypes, y_col, x_col, metric, page, cob_store):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    if "reset" in trig:
        return []

    add_dict = {id_obj["index"]: (v or 0) for id_obj, v in zip(add_ids, add_vals)}
    rem_dict = {id_obj["index"]: (v or 0) for id_obj, v in zip(rem_ids, rem_vals)}

    y_col  = y_col  or SIM_Y_OPTIONS[0]["value"]
    x_col  = x_col  or "GROUP_RATING"
    metric = metric or "volume"
    page   = page or 0
    cob    = cob_store or {}

    d_orig = apply_filters(country, brands, models, bodytypes)
    d_orig = apply_cob_filter(d_orig, cob.get("granularity", "monthly"), cob.get("period") or "")

    if d_orig.empty:
        return html.P("No data.", style={"color": "#999", "padding": "20px", "textAlign": "center"})

    d_sim = build_sim_df(d_orig, y_col, add_dict, rem_dict)

    piv_orig  = build_pivot(d_orig, y_col, x_col, metric)
    piv_sim   = build_pivot(d_sim,  y_col, x_col, metric)
    all_cols  = sorted(set(piv_orig.columns) | set(piv_sim.columns))
    all_idx   = sorted(set(piv_orig.index)   | set(piv_sim.index))
    piv_orig  = piv_orig.reindex(index=all_idx, columns=all_cols, fill_value=0)
    piv_sim   = piv_sim.reindex( index=all_idx, columns=all_cols, fill_value=0)
    piv_delta = piv_sim - piv_orig

    m_lbl     = _lbl(METRIC_OPTIONS, metric)
    fig_orig  = make_heatmap_fig(piv_orig,  f"Original — {m_lbl}",  "Blues", page)
    fig_sim   = make_heatmap_fig(piv_sim,   f"Simulated — {m_lbl}", "BuGn",  page)

    paged_d = piv_delta.iloc[page * PAGE_SIZE: (page + 1) * PAGE_SIZE]
    if not paged_d.empty:
        abs_max = max(abs(float(piv_delta.values.max())), abs(float(piv_delta.values.min())), 1.0)
        text_d  = [[f"{v:+.1f}" if not (isinstance(v, float) and np.isnan(v)) else "" for v in row]
                   for row in paged_d.values]
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

    return html.Div([
        html.Div("Simulation Results",
                 style={"fontWeight": "700", "fontSize": "14px", "color": "#1a1a2e",
                        "margin": "0 0 12px"}),
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


# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8051)
