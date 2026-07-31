
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update

# =============================================================================
# Config — adjust to your environment
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"

COUNTRIES_TO_READ: list[str] = ["SPAIN"]

START_YYYYMM = "202001"
END_YYYYMM   = "202512"

REAL_COLUMNS = [
    "COB_DATE", "ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION",
    "COUNTRY", "NOVA_ASSET_STATUS", "BIKE_OR_CAR",
    "BRAND_UPDATE", "VEHICLE_MODEL", "MODEL_CATALOG",
    "CONTRACT_START_DATE",
    "OBLIGOR_IDENTIFIER", "ID_CUSTOMER", "GROUP_RATING", "COUNTERPARTY_RATING", "CLS_GROUP_RATING",
    "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION",
    "VEHICLE_PRICE_EUR", "EXPOSURE_AMOUNT_LTR", "EXPOSURE_AMOUNT_MTR", "PENDING_ORDERS",
]


WANTED_ARRS_COLUMNS = [
    "ARRS_BTWN_0_30D", "ARRS_BTWN_31_60D", "ARRS_BTWN_61_90D",
    "ARRS_BTWN_91_180D", "ARRS_BTWN_181_270D",
    "ARRS_MORE_30D", "ARRS_MORE_60D", "ARRS_MORE_90D", "ARRS_MORE_180D", "ARRS_MORE_270D",
]

UNIQUE_KEY_COLS = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]


RATING_COLUMNS = [
    {"col": "GROUP_RATING", "label": "Group Rating", "numeric": False},
    {"col": "COUNTERPARTY_RATING", "label": "Counterparty Rating", "numeric": False},
    {"col": "CLS_GROUP_RATING", "label": "CLS Rating", "numeric": True},
]
RATING_COL_LABELS = {r["col"]: r["label"] for r in RATING_COLUMNS}
RATING_COL_NUMERIC = {r["col"]: r["numeric"] for r in RATING_COLUMNS}
NR_VALUE = "__NR__"
NR_LABEL = "NR (Not Rated)"


# =============================================================================
# Data loading — same filename convention as the other dashboards in this repo
# Filename pattern:  <prefix>-<COUNTRY>-<YYYYMM>.parquet
# =============================================================================

def _detect_available_columns(folder_path: Path, wanted_cols: list[str]) -> list[str]:
    """Only keep columns that actually exist in the parquet schema, so reading
    never fails locally and this file works unmodified once a fuller NOVA
    export (with the arrears columns) is available."""
    files = glob.glob(f"{folder_path}/*.parquet")
    if not files:
        return list(wanted_cols)
    try:
        import pyarrow.parquet as pq
        schema_cols = set(pq.read_schema(files[0]).names)
    except Exception:
        return list(wanted_cols)
    return [c for c in wanted_cols if c in schema_cols]


def load_country_monthly_data(
    folder_path: Path | str,
    countries: list[str],
    start_yyyymm: str,
    end_yyyymm: str,
    cols: list[str] | None = None,
) -> pd.DataFrame:
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
        return pd.DataFrame(columns=cols or REAL_COLUMNS)
    return pd.concat(dfs, ignore_index=True)


DATE_FIELDS = {"COB_DATE", "CONTRACT_START_DATE"}


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in DATE_FIELDS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for col in ["COUNTRY", "BRAND_UPDATE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# =============================================================================
# Startup: load once, kept fully in memory (same architecture as
# customer_vehicle_explorer.py) — no data is re-read from disk per click.
# =============================================================================

_wanted_columns = REAL_COLUMNS + WANTED_ARRS_COLUMNS
_columns_to_read = _detect_available_columns(DATA_FOLDER, _wanted_columns)
ARRS_COLUMNS_PRESENT = [c for c in WANTED_ARRS_COLUMNS if c in _columns_to_read]

GLOBAL_DF = load_country_monthly_data(
    DATA_FOLDER, COUNTRIES_TO_READ, START_YYYYMM, END_YYYYMM,
    cols=_columns_to_read,
)
GLOBAL_DF = prepare_dataset(GLOBAL_DF)


# =============================================================================
# Rating formatting / sorting — matches the credit-rating convention already
# established in use_case_1_heatmap_dashboard / customer_vehicle_explorer.py.
# =============================================================================

def _fmt_rating(col: str, v) -> str | None:
    """Normalized display string for a raw rating value, or None if missing."""
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


def _rating_sort_key(col: str, display_value: str):
    """NR first, then increasing grade — '02-' < '02' < '02+' for
    Group/Counterparty (string grades), plain numeric order for CLS."""
    if display_value == NR_LABEL:
        return (-1, 0)
    if RATING_COL_NUMERIC.get(col):
        try:
            return (0, float(display_value))
        except (TypeError, ValueError):
            return (0, 0.0)
    s = display_value
    suffix = 0
    if s.endswith("+"):
        suffix, s = 1, s[:-1]
    elif s.endswith("-"):
        suffix, s = -1, s[:-1]
    try:
        base = int(s)
    except ValueError:
        base = 999
    return (0, base, suffix)


# =============================================================================
# Customer current-state table — one row per customer with their latest known
# (forward-filled) rating on each of the 3 rating columns, latest COUNTRY and
# latest arrears figures. This powers the filter + results list; the raw,
# full-history GLOBAL_DF is queried separately (per customer, on "More
# details") for the rating-at-origination lookup.
# =============================================================================

def _build_current_state(df: pd.DataFrame) -> pd.DataFrame:
    rating_cols = [r["col"] for r in RATING_COLUMNS if r["col"] in df.columns]
    keep_cols = [c for c in (
        ["ID_CUSTOMER", "COUNTRY", "COB_DATE"] + rating_cols + ARRS_COLUMNS_PRESENT
    ) if c in df.columns]
    if df.empty or "ID_CUSTOMER" not in df.columns:
        return pd.DataFrame(columns=keep_cols)

    d = df[keep_cols].dropna(subset=["ID_CUSTOMER"]).copy()
    d["ID_CUSTOMER"] = d["ID_CUSTOMER"].astype(str).str.strip()
    d = d.sort_values("COB_DATE")

    fill_cols = [c for c in keep_cols if c not in ("ID_CUSTOMER", "COB_DATE")]
    if fill_cols:
        d[fill_cols] = d.groupby("ID_CUSTOMER")[fill_cols].ffill()
    d = d.drop_duplicates(subset=["ID_CUSTOMER"], keep="last")

    for col in rating_cols:
        d[f"{col}_DISP"] = d[col].apply(lambda v, c=col: _fmt_rating(c, v))

    veh_counts = (
        df.dropna(subset=["ID_CUSTOMER"])
          .assign(ID_CUSTOMER=lambda x: x["ID_CUSTOMER"].astype(str).str.strip())
          .drop_duplicates(subset=[k for k in UNIQUE_KEY_COLS if k in df.columns] + ["ID_CUSTOMER"])
          .groupby("ID_CUSTOMER").size().rename("N_VEHICLES")
    )
    d = d.merge(veh_counts, on="ID_CUSTOMER", how="left")
    d["N_VEHICLES"] = d["N_VEHICLES"].fillna(0).astype(int)
    return d


CURRENT_STATE_DF = _build_current_state(GLOBAL_DF)


def _rating_value_options(rating_col: str) -> list[dict]:
    disp_col = f"{rating_col}_DISP"
    if disp_col not in CURRENT_STATE_DF.columns:
        return [{"label": NR_LABEL, "value": NR_VALUE}]
    vals = sorted(
        CURRENT_STATE_DF[disp_col].dropna().unique(),
        key=lambda v: _rating_sort_key(rating_col, v),
    )
    opts = [{"label": NR_LABEL, "value": NR_VALUE}]
    opts += [{"label": v, "value": v} for v in vals]
    return opts


# =============================================================================
# Exposure — identical formula to customer_vehicle_explorer.py's (fixed)
# _customer_exposure: LTR + pending orders per row, dedup per contract (latest
# COB_DATE), then summed — never collapse a multi-contract customer to one row.
# =============================================================================

def _customer_exposure(customer_rows: pd.DataFrame) -> float:
    if customer_rows.empty:
        return 0.0
    d = customer_rows.copy()
    ltr = d["EXPOSURE_AMOUNT_LTR"] if "EXPOSURE_AMOUNT_LTR" in d.columns else 0.0
    pending = d["PENDING_ORDERS"] if "PENDING_ORDERS" in d.columns else 0.0
    d["EXPOSURE"] = (ltr.fillna(0) if hasattr(ltr, "fillna") else ltr) + \
                     (pending.fillna(0) if hasattr(pending, "fillna") else pending)
    if "COB_DATE" in d.columns:
        d = d.sort_values("COB_DATE")
    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    if keys:
        d = d.drop_duplicates(subset=keys, keep="last")
    return float(d["EXPOSURE"].sum())


def format_millions(value: float) -> str:
    s = f"{value / 1_000_000:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


# =============================================================================
# Rating-at-origination — the core ask: for each of a customer's contracts,
# check its CONTRACT_START_DATE, restrict to that same customer + country (the
# in-memory equivalent of "read the file that corresponds"), and find the
# latest known rating value at/before that date — i.e. what the client's
# rating was at the moment they leased that particular vehicle.
# =============================================================================

def _rating_at_date(customer_country_rows: pd.DataFrame, rating_col: str, asof_date) -> str | None:
    if rating_col not in customer_country_rows.columns or pd.isna(asof_date):
        return None
    d = customer_country_rows[customer_country_rows["COB_DATE"] <= asof_date]
    d = d.dropna(subset=[rating_col])
    if d.empty:
        return None
    d = d.sort_values("COB_DATE")
    return _fmt_rating(rating_col, d.iloc[-1][rating_col])


def _build_vehicle_origination_table(customer_id: str, country: str | None, rating_col: str) -> list[dict]:
    d = GLOBAL_DF[GLOBAL_DF["ID_CUSTOMER"].astype(str).str.strip() == customer_id]
    if country and "COUNTRY" in d.columns:
        d = d[d["COUNTRY"] == country]
    if d.empty:
        return []

    keys = [k for k in UNIQUE_KEY_COLS if k in d.columns]
    veh = d.sort_values("COB_DATE") if "COB_DATE" in d.columns else d
    if keys:
        veh = veh.drop_duplicates(subset=keys, keep="last")

    current_disp = _fmt_rating(rating_col, d.sort_values("COB_DATE")[rating_col].dropna().iloc[-1]) \
        if rating_col in d.columns and d[rating_col].notna().any() else None

    rows = []
    for v in veh.to_dict("records"):
        start = v.get("CONTRACT_START_DATE")
        origination_disp = _rating_at_date(d, rating_col, start)
        if origination_disp is None:
            status = "No data before contract start"
        elif current_disp is not None and origination_disp == current_disp:
            status = f"Already at {origination_disp} at origination"
        else:
            status = "Rating changed since leasing"
        rows.append({
            "VEHICLE_ID": v.get("VEHICLE_ID"),
            "BRAND_UPDATE": v.get("BRAND_UPDATE") or "—",
            "MODEL": v.get("VEHICLE_MODEL") or v.get("MODEL_CATALOG") or "—",
            "CONTRACT_START_DATE": start.strftime("%Y-%m-%d") if pd.notna(start) else "—",
            "RATING_AT_ORIGINATION": origination_disp or "—",
            "CURRENT_RATING": current_disp or "—",
            "STATUS": status,
        })
    return rows


# =============================================================================
# Dash app
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Client Rating at Origination"

PAGE_STYLE = {"fontFamily": "Inter, -apple-system, sans-serif", "minHeight": "100vh",
              "background": "#f7fafc"}
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
MORE_DETAILS_BTN_STYLE = {
    "padding": "6px 14px", "fontWeight": "700", "fontSize": "12px",
    "background": "#ffffff", "color": "#3182ce", "border": "1px solid #bee3f8",
    "borderRadius": "6px", "cursor": "pointer",
}
CARD_STYLE = {
    "background": "#ffffff", "borderRadius": "8px",
    "border": "1px solid #e2e8f0", "padding": "18px",
}
PANEL_TITLE_STYLE = {"fontWeight": "700", "fontSize": "13px", "color": "#1a1a2e",
                      "marginBottom": "12px", "letterSpacing": "0.02em"}
FIELD_TITLE_STYLE = {"fontSize": "11px", "color": "#718096", "fontWeight": "600",
                      "marginBottom": "4px"}
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
ERROR_STYLE = {"color": "#c53030", "fontSize": "13px", "fontWeight": "600",
               "padding": "10px 28px", "display": "none"}
TABLE_STYLE = {"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"}
TH_STYLE = {"textAlign": "left", "padding": "8px 10px", "borderBottom": "2px solid #e2e8f0",
            "color": "#718096", "fontSize": "11px", "fontWeight": "700", "textTransform": "uppercase"}
TD_STYLE = {"padding": "8px 10px", "borderBottom": "1px solid #f0f2f5", "color": "#1a1a2e"}
STATUS_BADGE_COLORS = {
    "Already at": "#c53030",
    "Rating changed": "#2f855a",
    "No data": "#a0aec0",
}


def _status_badge_color(status: str) -> str:
    for prefix, color in STATUS_BADGE_COLORS.items():
        if status.startswith(prefix):
            return color
    return "#a0aec0"


def _panel_title(text: str):
    return html.Div(text, style=PANEL_TITLE_STYLE)


def _table(headers: list[str], rows: list[list], row_ids: list | None = None):
    body_rows = []
    for i, r in enumerate(rows):
        extra = {"id": row_ids[i]} if row_ids else {}
        body_rows.append(html.Tr([html.Td(c, style=TD_STYLE) for c in r], **extra))
    return html.Table(
        [html.Thead(html.Tr([html.Th(h, style=TH_STYLE) for h in headers]))] +
        [html.Tbody(body_rows)],
        style=TABLE_STYLE,
    )


GRID_PAGE_SIZE = 20

app.layout = html.Div(
    [
        dcc.Store(id="clients-store", data=[]),
        dcc.Store(id="page-store", data=0),

        html.Div(
            [
                html.H1("Client Rating at Origination",
                        style={"margin": "0 0 6px", "fontSize": "22px", "fontWeight": "700",
                               "color": "#1a1a2e"}),
                html.Div("See a client's rating today vs. the rating they had when each of their "
                         "vehicles was leased (as of the contract start date).",
                         style={"fontSize": "12px", "color": "#718096", "marginBottom": "14px"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Rating Column", style=FIELD_TITLE_STYLE),
                                dcc.Dropdown(
                                    id="rating-column-input",
                                    options=[{"label": r["label"], "value": r["col"]} for r in RATING_COLUMNS],
                                    value=RATING_COLUMNS[0]["col"], clearable=False,
                                    style={"minWidth": "220px"},
                                ),
                            ],
                        ),
                        html.Div(
                            [
                                html.Div("Rating Value", style=FIELD_TITLE_STYLE),
                                dcc.Dropdown(
                                    id="rating-value-input",
                                    options=_rating_value_options(RATING_COLUMNS[0]["col"]),
                                    placeholder="Choose a rating…",
                                    style={"minWidth": "220px"},
                                ),
                            ],
                        ),
                        html.Div(
                            html.Button("Search", id="btn-search", n_clicks=0, style=PRIMARY_BTN_STYLE),
                            style={"alignSelf": "flex-end"},
                        ),
                    ],
                    style={"display": "flex", "gap": "14px", "alignItems": "flex-start"},
                ),
                html.Div(id="search-error", style=ERROR_STYLE),
            ],
            style={"padding": "20px 28px", "borderBottom": "1px solid #e2e8f0", "background": "#ffffff"},
        ),

        html.Div(
            "Pick a rating column and value, then press Search to list clients currently at that rating.",
            id="empty-state",
            style={"padding": "80px 20px", "textAlign": "center", "color": "#a0aec0", "fontSize": "14px"},
        ),

        html.Div(
            [
                html.Div(id="clients-table-wrap"),
                html.Div(
                    [
                        html.Button("←", id="btn-page-prev", n_clicks=0, className="page-nav-button"),
                        html.Span("Page 1 of 1", id="page-label",
                                  style={"fontSize": "12px", "color": "#718096", "fontWeight": "600"}),
                        html.Button("→", id="btn-page-next", n_clicks=0, className="page-nav-button"),
                    ],
                    style={"display": "flex", "justifyContent": "center", "alignItems": "center",
                           "gap": "14px", "marginTop": "18px"},
                ),
            ],
            id="results-section",
            style={"display": "none", "padding": "24px 28px"},
        ),

        html.Div(
            id="client-modal",
            style=MODAL_OVERLAY_STYLE,
            children=[
                html.Div(
                    style=MODAL_PANEL_STYLE,
                    children=[
                        html.Div(
                            html.Button("Close", id="client-modal-close", n_clicks=0,
                                        className="cro-secondary-btn", style=SECONDARY_BTN_STYLE),
                            style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"},
                        ),
                        html.Div(id="client-modal-body"),
                    ],
                ),
            ],
        ),
    ],
    style=PAGE_STYLE,
)


# =============================================================================
# Callbacks
# =============================================================================

@app.callback(
    Output("rating-value-input", "options"),
    Output("rating-value-input", "value"),
    Input("rating-column-input", "value"),
)
def _update_rating_value_options(rating_col):
    return _rating_value_options(rating_col), None


@app.callback(
    Output("clients-store", "data"),
    Output("search-error", "children"),
    Output("search-error", "style"),
    Output("results-section", "style"),
    Output("empty-state", "style"),
    Input("btn-search", "n_clicks"),
    State("rating-column-input", "value"),
    State("rating-value-input", "value"),
    prevent_initial_call=True,
)
def _search_clients(_n_clicks, rating_col, rating_val):
    if not rating_col or not rating_val:
        return (no_update, "Please choose a rating column and a rating value.",
                {**ERROR_STYLE, "display": "block"}, {"display": "none"}, {"display": "block"})

    disp_col = f"{rating_col}_DISP"
    if disp_col not in CURRENT_STATE_DF.columns:
        return (no_update, f"{RATING_COL_LABELS.get(rating_col, rating_col)} is not available in this dataset.",
                {**ERROR_STYLE, "display": "block"}, {"display": "none"}, {"display": "block"})

    if rating_val == NR_VALUE:
        matches = CURRENT_STATE_DF[CURRENT_STATE_DF[disp_col].isna()]
    else:
        matches = CURRENT_STATE_DF[CURRENT_STATE_DF[disp_col] == rating_val]

    if matches.empty:
        return ([], f"No client currently at {rating_val if rating_val != NR_VALUE else NR_LABEL} "
                     f"for {RATING_COL_LABELS.get(rating_col, rating_col)}.",
                {**ERROR_STYLE, "display": "block"}, {"display": "none"}, {"display": "block"})

    keep = ["ID_CUSTOMER", "COUNTRY", "N_VEHICLES"] + [f"{r['col']}_DISP" for r in RATING_COLUMNS
                                                        if f"{r['col']}_DISP" in matches.columns]
    out = matches[keep].sort_values("ID_CUSTOMER").to_dict("records")
    return (out, "", {**ERROR_STYLE, "display": "none"},
            {"display": "block"}, {"display": "none"})


@app.callback(
    Output("clients-table-wrap", "children"),
    Output("page-store", "data"),
    Output("page-label", "children"),
    Output("btn-page-prev", "disabled"),
    Output("btn-page-next", "disabled"),
    Input("clients-store", "data"),
    Input("btn-page-prev", "n_clicks"),
    Input("btn-page-next", "n_clicks"),
    State("page-store", "data"),
)
def _render_clients_table(clients, _prev, _next, page):
    clients = clients or []
    trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if not clients:
        return (html.P("No clients to show.", style={"color": "#a0aec0", "fontSize": "13px"}),
                0, "Page 0 of 0", True, True)

    n = len(clients)
    total_pages = max(1, -(-n // GRID_PAGE_SIZE))
    if trig == "btn-page-prev":
        page = max(0, (page or 0) - 1)
    elif trig == "btn-page-next":
        page = min(total_pages - 1, (page or 0) + 1)
    else:
        page = 0
    page = max(0, min(page, total_pages - 1))

    start = page * GRID_PAGE_SIZE
    page_rows = clients[start:start + GRID_PAGE_SIZE]

    headers = ["Customer ID", "Country", "Group Rating", "Counterparty Rating", "CLS Rating", "Vehicles", ""]
    rows, row_ids = [], []
    for c in page_rows:
        cid = c.get("ID_CUSTOMER")
        rows.append([
            cid, c.get("COUNTRY") or "—",
            c.get("GROUP_RATING_DISP") or "NR", c.get("COUNTERPARTY_RATING_DISP") or "NR",
            c.get("CLS_GROUP_RATING_DISP") or "NR", c.get("N_VEHICLES", 0),
            html.Button("More details", id={"type": "client-more-details", "index": str(cid)},
                        n_clicks=0, className="cro-more-details-btn", style=MORE_DETAILS_BTN_STYLE),
        ])
    table = _table(headers, rows)
    label = f"Page {page + 1} of {total_pages} ({n} client{'s' if n != 1 else ''})"
    return table, page, label, page <= 0, page >= total_pages - 1


@app.callback(
    Output("client-modal", "style"),
    Output("client-modal-body", "children"),
    Input({"type": "client-more-details", "index": ALL}, "n_clicks"),
    Input("client-modal-close", "n_clicks"),
    State("clients-store", "data"),
    State("rating-column-input", "value"),
    prevent_initial_call=True,
)
def _toggle_client_modal(card_clicks, _close_clicks, clients, rating_col):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if "client-modal-close" in trig:
        return {**MODAL_OVERLAY_STYLE, "display": "none"}, no_update

    if not any(card_clicks or []):
        return no_update, no_update

    prop_id = trig.rsplit(".", 1)[0]
    try:
        cust_id = json.loads(prop_id)["index"]
    except Exception:
        return no_update, no_update

    clients = clients or []
    match = next((c for c in clients if str(c.get("ID_CUSTOMER")) == str(cust_id)), None)
    if match is None:
        return no_update, no_update

    country = match.get("COUNTRY")
    customer_rows = GLOBAL_DF[GLOBAL_DF["ID_CUSTOMER"].astype(str).str.strip() == str(cust_id)]
    exposure = _customer_exposure(customer_rows)
    exposure_s = f"{format_millions(exposure)} million"

    arrears_row = customer_rows.sort_values("COB_DATE").iloc[-1] if not customer_rows.empty else None
    if ARRS_COLUMNS_PRESENT and arrears_row is not None:
        arrears_headers = [c.replace("ARRS_", "").replace("_", " ") for c in ARRS_COLUMNS_PRESENT]
        arrears_values = [arrears_row.get(c) if pd.notna(arrears_row.get(c)) else "—" for c in ARRS_COLUMNS_PRESENT]
        arrears_block = _table(arrears_headers, [arrears_values])
    else:
        arrears_block = html.P(
            "Arrears columns (ARRS_BTWN_0_30D, ARRS_MORE_30D, …) are not present in this local dataset.",
            style={"color": "#a0aec0", "fontSize": "12px"},
        )

    veh_rows = _build_vehicle_origination_table(str(cust_id), country, rating_col)
    veh_headers = ["Vehicle ID", "Brand", "Model", "Contract Start", "Rating at Origination",
                   "Current Rating", "Status"]
    veh_table_rows = []
    for r in veh_rows:
        veh_table_rows.append([
            r["VEHICLE_ID"], r["BRAND_UPDATE"], r["MODEL"], r["CONTRACT_START_DATE"],
            r["RATING_AT_ORIGINATION"], r["CURRENT_RATING"],
            html.Span(r["STATUS"], style={
                "display": "inline-block", "padding": "3px 10px", "borderRadius": "999px",
                "fontSize": "10px", "fontWeight": "700", "color": "#fff",
                "background": _status_badge_color(r["STATUS"]), "whiteSpace": "nowrap",
            }),
        ])
    veh_block = _table(veh_headers, veh_table_rows) if veh_table_rows else \
        html.P("No vehicles found for this client.", style={"color": "#a0aec0", "fontSize": "12px"})

    body = html.Div([
        html.Div(str(cust_id), style={"fontWeight": "700", "fontSize": "19px", "color": "#1a1a2e",
                                       "marginBottom": "4px"}),
        html.Div(f"{country or '—'} — {RATING_COL_LABELS.get(rating_col, rating_col)}",
                 style={"fontSize": "12px", "color": "#718096", "marginBottom": "18px"}),

        html.Div([
            _panel_title("Total Exposure"),
            html.Div(exposure_s, style={"fontSize": "22px", "fontWeight": "700", "color": "#3182ce"}),
        ], style={**CARD_STYLE, "marginBottom": "14px"}),

        html.Div([
            _panel_title("Arrears"),
            html.Div(arrears_block, style={"overflowX": "auto"}),
        ], style={**CARD_STYLE, "marginBottom": "14px"}),

        html.Div([
            _panel_title("Vehicles — Rating at Origination vs. Current Rating"),
            html.Div(veh_block, style={"overflowX": "auto"}),
        ], style=CARD_STYLE),
    ])

    return {**MODAL_OVERLAY_STYLE, "display": "flex"}, body


# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8052)
