from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update

# =============================================================================
# Config — adjust to your environment
#
# This dashboard never touches data/ (the raw NOVA parquets) or models.parquet
# — it reads only the already-prepared, already-enriched extract written
# offline by generate_customer_vehicle_explorer_precomputed.py. That script is
# the only thing that reads raw NOVA data and runs the models.parquet join;
# this file's entire data-loading is the single pd.read_parquet() call below.
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "a.jpg"

PRECOMPUTED_PATH = BASE_DIR / "precomputed_customer_vehicle_explorer" / "customer_vehicle_explorer_data.parquet"

UNIQUE_KEY_COLS = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]

DATE_FIELDS = {
    "COB_DATE", "CONTRACT_START_DATE", "CONTRACT_END_DATE", "CONTRACT_END_DATE_AMENDED",
    "CONTRACT_FINAL_END", "DATE_OF_ORDER", "DATE_OF_QUOTATION", "REGISTRATION_DATE",
    "DELIVERY_DATE", "EXTENSION_DATE",
}


# =============================================================================
# Startup: read the precomputed extract (no raw NOVA scan, no models.parquet
# join) — built by generate_customer_vehicle_explorer_precomputed.py, which
# owns the loading/enrichment logic this dashboard used to do at import time.
# =============================================================================

if PRECOMPUTED_PATH.exists():
    CV_DF = pd.read_parquet(PRECOMPUTED_PATH)
else:
    CV_DF = pd.DataFrame()
    print(f"WARNING: {PRECOMPUTED_PATH} not found. "
          f"Run generate_customer_vehicle_explorer_precomputed.py first.")


def _build_customer_index(df: pd.DataFrame) -> pd.DataFrame:
    """One row per known customer (id -> latest known country/name), used to
    power the searchable Customer ID list and the Country filter next to it."""
    if df.empty or "ID_CUSTOMER" not in df.columns:
        return pd.DataFrame(columns=["ID_CUSTOMER", "COUNTRY", "CUSTOMER_NAME"])
    cols = [c for c in ["ID_CUSTOMER", "COUNTRY", "CUSTOMER_NAME", "COB_DATE"] if c in df.columns]
    d = df[cols].dropna(subset=["ID_CUSTOMER"]).copy()
    d["ID_CUSTOMER"] = d["ID_CUSTOMER"].astype(str).str.strip()
    if "COB_DATE" in d.columns:
        d = d.sort_values("COB_DATE")
    fill_cols = [c for c in ["COUNTRY", "CUSTOMER_NAME"] if c in d.columns]
    if fill_cols:
        d[fill_cols] = d.groupby("ID_CUSTOMER")[fill_cols].ffill()
    d = d.drop_duplicates(subset=["ID_CUSTOMER"], keep="last")
    return d[[c for c in ["ID_CUSTOMER", "COUNTRY", "CUSTOMER_NAME"] if c in d.columns]].sort_values("ID_CUSTOMER")


CUSTOMER_INDEX_DF = _build_customer_index(CV_DF)
COUNTRY_FILTER_OPTIONS = (
    [{"label": c, "value": c} for c in sorted(CUSTOMER_INDEX_DF["COUNTRY"].dropna().unique())]
    if "COUNTRY" in CUSTOMER_INDEX_DF.columns else []
)

# Distinct reporting snapshot dates available in the data — lets the user view
# a customer's fleet "as of" a given COB_DATE instead of always the latest
# known snapshot (a vehicle's IN FLEET / SOLD / ... status is only ever true
# as of a specific COB_DATE).
COB_DATE_OPTIONS = (
    sorted(CV_DF["COB_DATE"].dropna().unique()) if "COB_DATE" in CV_DF.columns else []
)
ASOF_DATE_OPTIONS = [
    {"label": pd.Timestamp(d).strftime("%Y-%m-%d"), "value": pd.Timestamp(d).isoformat()}
    for d in COB_DATE_OPTIONS
]


# =============================================================================
# Domain helpers
# =============================================================================

def format_millions(value: float) -> str:
    s = f"{value / 1_000_000:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _json_safe_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _json_safe_dict(d: dict) -> dict:
    return {k: _json_safe_value(v) for k, v in d.items()}


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    return [_json_safe_dict(row) for row in df.to_dict("records")]


def _customer_profile(customer_rows: pd.DataFrame) -> dict:
    """Latest *known* (non-null) value per profile field for this customer —
    forward-filled across their snapshots so a blank latest row doesn't hide a
    real, earlier value (same fix already validated in the Use Case 1 dashboard)."""
    if customer_rows.empty:
        return {}
    d = customer_rows.sort_values("COB_DATE") if "COB_DATE" in customer_rows.columns else customer_rows
    fill_cols = [c for c in [
        "COUNTRY", "CUSTOMER_NAME", "GROUP_RATING", "COUNTERPARTY_RATING", "CLS_GROUP_RATING",
        "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION", "SHARED_CLIENT_FLAG",
    ] if c in d.columns]
    d = d.copy()
    if fill_cols:
        d[fill_cols] = d[fill_cols].ffill()
    last = d.iloc[-1]
    return {c: last.get(c) for c in fill_cols}


def _customer_exposure(customer_rows: pd.DataFrame) -> float:
    """EXPOSURE_AMOUNT_LTR / PENDING_ORDERS are reported per contract, not
    duplicated identically across every row of the customer (a sold contract
    reports 0 while another of the same customer's contracts still in fleet
    keeps its real exposure). So each distinct contract/vehicle is taken at
    its own latest known snapshot (same key as the vehicle list itself), then
    summed — this is what "same formula as Use Case 1" (LTR + pending orders,
    dedup before summing) means once a customer can have more than one
    contract: dedup per contract, not collapse the whole customer to one row."""
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


def _pair_units(columns) -> tuple[dict, set]:
    """For every "<X>_UNIT" column, pair it with the field(s) it describes:
    an exact-name match (ENGINE_SIZE_UNIT -> ENGINE_SIZE) pairs with that one
    field; otherwise every other column sharing the same prefix
    (FUEL_CONSUMPTION_UNIT -> FUEL_CONSUMPTION_THEORETICAL/HIGHWAY/URBAN) shares
    that one unit. Returns (value_field -> unit_field, set of unit fields that
    are consumed and must never render as their own row)."""
    cols = set(columns)
    unit_cols = [c for c in cols if c.endswith("_UNIT")]
    pairing: dict[str, str] = {}
    consumed: set[str] = set()
    for uc in unit_cols:
        prefix = uc[: -len("_UNIT")]
        if not prefix:
            continue
        if prefix in cols:
            pairing[prefix] = uc
            consumed.add(uc)
        else:
            group_prefix = prefix + "_"
            members = [c for c in cols if c.startswith(group_prefix) and c != uc and not c.endswith("_UNIT")]
            if members:
                for m in members:
                    pairing[m] = uc
                consumed.add(uc)
    return pairing, consumed


_LABEL_ACRONYMS = {"ID", "EUR", "HP", "CO2"}


def _pretty_label(field: str) -> str:
    words = field.split("_")
    out = []
    for w in words:
        if w.upper() in _LABEL_ACRONYMS:
            out.append(w.upper())
        elif w.upper() == "OF":
            out.append("of")
        else:
            out.append(w.capitalize())
    return " ".join(out)


FIELD_LABEL_OVERRIDES = {
    "VEHICLE_ID": "Vehicle ID",
    "BRAND_UPDATE": "Brand",
    "MARKET_MODEL": "Model",
    "VEHICLE_MODEL": "Model",
    "MODEL_CATALOG": "Model (Catalog)",
    "VEHICLE_CLASS": "Vehicle Class",
    "MARKET_BODY_GROUP": "Body Type",
    "VEHICLE_BODY_TYPE": "Body Type",
    "VEHICLE_SEGMENT_PROXY": "Segment",
    "NORMALISED_VEHICLE_TYPE": "Normalised Vehicle Type",
    "CLS_VEHICLE_TYPE": "CLS Vehicle Type",
    "BODY_COLOR": "Body Color",
    "VEHICLE_PRICE_EUR": "Vehicle Price",
    "CATALOG_PRICE": "Catalog Price",
    "NUMBER_OF_DOORS": "Doors",
    "NUMBER_OF_CYLENDER": "Cylinders",
    "NUMBER_OF_SEATS": "Seats",
    "FISCAL_POWER": "Fiscal Power",
    "NUMBER_OF_SPEED": "Gears / Speeds",
    "ENGINE_SIZE": "Engine Size",
    "ENGINE_POWER_HP": "Engine Power",
    "GEARBOX": "Gearbox",
    "AUTONOMY": "Autonomy (Electric Range)",
    "FUEL_TYPE": "Fuel Type",
    "FUEL_TYPE2": "Fuel Type (Secondary)",
    "FUEL_CONSUMPTION_THEORETICAL": "Fuel Consumption (Combined)",
    "FUEL_CONSUMPTION_HIGHWAY": "Fuel Consumption (Highway)",
    "FUEL_CONSUMPTION_URBAN": "Fuel Consumption (Urban)",
    "VA_CO2_EMSS_REAL": "CO2 Emissions",
    "NOVA_ASSET_STATUS": "Asset Status",
    "DELIVERY_DATE": "Delivery Date",
    "REGISTRATION_DATE": "Registration Date",
    "CONTRACT_START_DATE": "Contract Start",
    "CONTRACT_END_DATE": "Contract End",
    "CONTRACT_END_DATE_AMENDED": "Contract End (Amended)",
    "CONTRACT_FINAL_END": "Contract Final End",
    "DATE_OF_ORDER": "Order Date",
    "DATE_OF_QUOTATION": "Quotation Date",
    "EXTENSION_DATE": "Extension Date",
}


def _field_label(field: str) -> str:
    return FIELD_LABEL_OVERRIDES.get(field, _pretty_label(field))


def _fmt_date(value) -> str | None:
    if value in (None, "", "NaT"):
        return None
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _fmt_field_value(field: str, value, unit_value=None) -> str | None:
    if value in (None, ""):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if field in DATE_FIELDS:
        return _fmt_date(value)

    if isinstance(value, bool):
        s = "Yes" if value else "No"
    elif isinstance(value, (int, float, np.integer, np.floating)):
        num = float(value)
        s = f"{int(num):,}" if num.is_integer() else f"{num:,.2f}"
        if field.endswith("_EUR") or field == "CATALOG_PRICE":
            s = f"€{s}"
    else:
        s = str(value)

    if unit_value not in (None, ""):
        try:
            if not pd.isna(unit_value):
                s = f"{s} {unit_value}"
        except (TypeError, ValueError):
            s = f"{s} {unit_value}"
    return s


VEHICLE_DETAIL_SECTIONS: list[tuple[str, list]] = [
    ("Identification", [
        "VEHICLE_ID", "BRAND_UPDATE",
        "MARKET_MODEL",
        "VEHICLE_CLASS",
        ("MARKET_BODY_GROUP", "VEHICLE_BODY_TYPE"),
        "VEHICLE_SEGMENT_PROXY", "NORMALISED_VEHICLE_TYPE", "CLS_VEHICLE_TYPE", "BODY_COLOR",
    ]),
    ("Pricing", [
        "VEHICLE_PRICE_EUR", "CATALOG_PRICE",
    ]),
    ("Engine & Performance", [
        "ENGINE_SIZE", "ENGINE_SIZE_UNIT", "ENGINE_POWER_HP", "FISCAL_POWER",
        "NUMBER_OF_CYLENDER", "NUMBER_OF_SPEED", "GEARBOX", "AUTONOMY",
    ]),
    ("Fuel & Emissions", [
        "FUEL_TYPE", "FUEL_TYPE2",
        "FUEL_CONSUMPTION_THEORETICAL", "FUEL_CONSUMPTION_HIGHWAY", "FUEL_CONSUMPTION_URBAN",
        "FUEL_CONSUMPTION_UNIT", "VA_CO2_EMSS_REAL",
    ]),
    ("Body & Comfort", [
        "NUMBER_OF_DOORS", "NUMBER_OF_SEATS",
    ]),
    ("Status & Key Dates", [
        "NOVA_ASSET_STATUS", "DELIVERY_DATE", "REGISTRATION_DATE",
        "CONTRACT_START_DATE", "CONTRACT_END_DATE", "CONTRACT_END_DATE_AMENDED",
        "CONTRACT_FINAL_END", "DATE_OF_ORDER", "DATE_OF_QUOTATION", "EXTENSION_DATE",
    ]),
]

STATUS_COLORS = {
    "IN FLEET": "#2f855a",
    "SOLD": "#718096",
    "ORDER ACTIVE": "#3182ce",
    "ORDER ACTIVE - DELIVERED": "#2b6cb0",
    "ORDER ACTIVE - REGISTERED": "#2b6cb0",
    "TBI": "#d98943",
    "TBI ORD.": "#d98943",
    "TBI OTHER ORD. STATUS BLANK": "#d98943",
    "DEHIRE": "#9b2c2c",
    "ORD. REGISTERED": "#805ad5",
}
DEFAULT_STATUS_COLOR = "#a0aec0"
RATING_NR_LABEL = "NR"


def _rating_display(v) -> str:
    if v in (None, ""):
        return RATING_NR_LABEL
    try:
        if pd.isna(v):
            return RATING_NR_LABEL
    except (TypeError, ValueError):
        pass
    return str(v)


# =============================================================================
# Dash app
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Customer & Vehicle Explorer"


# ── Style tokens (matches the visual language of use_case_1_heatmap_dashboard_4) ──

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
    "width": "100%", "marginTop": "12px", "padding": "8px 14px",
    "fontWeight": "700", "fontSize": "12px",
    "background": "#ffffff", "color": "#3182ce", "border": "1px solid #bee3f8",
    "borderRadius": "6px", "cursor": "pointer",
}
CARD_STYLE = {
    "background": "#ffffff", "borderRadius": "8px",
    "border": "1px solid #e2e8f0", "padding": "18px",
}
CUSTOMER_CARD_STYLE = {
    **CARD_STYLE,
    "background": "#eef1f5", "border": "1px solid #dde3ea",
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
VEHICLE_MODAL_PANEL_STYLE = {
    "background": "#ffffff", "borderRadius": "12px", "width": "820px",
    "maxWidth": "94vw", "maxHeight": "90vh", "overflowY": "auto",
    "padding": "30px", "boxShadow": "0 20px 60px rgba(0,0,0,0.25)",
}
ERROR_STYLE = {"color": "#c53030", "fontSize": "13px", "fontWeight": "600",
               "padding": "10px 28px", "display": "none"}


def _panel_title(text: str):
    return html.Div(text, style=PANEL_TITLE_STYLE)


# ── Vehicle card ─────────────────────────────────────────────────────────────

def _status_badge(status):
    color = STATUS_COLORS.get(str(status).upper(), DEFAULT_STATUS_COLOR)
    return html.Span(status, style={
        "display": "inline-block", "padding": "3px 10px", "borderRadius": "999px",
        "fontSize": "10px", "fontWeight": "700", "color": "#fff", "background": color,
        "whiteSpace": "nowrap",
    })


def _vehicle_card(v: dict):
    vid = v.get("VEHICLE_ID")
    brand = v.get("BRAND_UPDATE") or "—"
    # Only the matched Market Model (models.parquet join) is shown next to the
    # brand — never a raw fallback field (VEHICLE_MODEL / MODEL_CATALOG), so
    # what's displayed is always the properly-matched model, or "—".
    model = v.get("MARKET_MODEL") or "—"
    bodytype = v.get("MARKET_BODY_GROUP") or v.get("VEHICLE_BODY_TYPE") or "—"
    status = v.get("NOVA_ASSET_STATUS") or "—"
    delivery_s = _fmt_date(v.get("DELIVERY_DATE")) or "—"
    price = v.get("VEHICLE_PRICE_EUR")
    price_s = _fmt_field_value("VEHICLE_PRICE_EUR", price) or "—"

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{brand} {model}", style={"fontWeight": "700", "fontSize": "14px",
                                                           "color": "#1a1a2e"}),
                    _status_badge(status),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                       "marginBottom": "8px", "gap": "8px"},
            ),
            html.Div(f"Vehicle ID {vid}", style={"fontSize": "11px", "color": "#a0aec0", "marginBottom": "10px"}),
            html.Div(bodytype, style={"fontSize": "12px", "color": "#4a5568", "marginBottom": "10px"}),
            html.Div(
                [
                    html.Span("Delivered", style={"fontSize": "11px", "color": "#a0aec0"}),
                    html.Span(delivery_s, style={"fontSize": "12px", "color": "#4a5568", "fontWeight": "600"}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"},
            ),
            html.Div(price_s, style={"fontSize": "15px", "fontWeight": "700", "color": "#3182ce"}),
            html.Button(
                "More details",
                id={"type": "vehicle-more-details", "index": str(vid)}, n_clicks=0,
                className="cve-more-details-btn", style=MORE_DETAILS_BTN_STYLE,
            ),
        ],
        style=CARD_STYLE,
    )


# ── Vehicle detail modal body ────────────────────────────────────────────────

def _render_vehicle_detail(v: dict):
    columns = set(v.keys())
    pairing, consumed_units = _pair_units(columns)

    sections_html = []
    for title, fields in VEHICLE_DETAIL_SECTIONS:
        rows = []
        for f in fields:
            if isinstance(f, tuple):
                # Alternative fields describing the same thing (e.g. MARKET_MODEL vs
                # VEHICLE_MODEL) — show only the first one that actually has data,
                # never several possibly-contradictory values for the same label.
                chosen = next((alt for alt in f if alt in v and _fmt_field_value(alt, v.get(alt)) is not None), None)
                if chosen is None:
                    continue
                label = _field_label(f[0])
                display = _fmt_field_value(chosen, v.get(chosen))
            else:
                if f in consumed_units or f not in v:
                    continue
                unit_field = pairing.get(f)
                unit_val = v.get(unit_field) if unit_field else None
                display = _fmt_field_value(f, v.get(f), unit_val)
                if display is None:
                    continue
                label = _field_label(f)
            rows.append(html.Div(
                [
                    html.Div(label, style=FIELD_TITLE_STYLE),
                    html.Div(display, style={"fontSize": "13px", "color": "#1a1a2e", "fontWeight": "600"}),
                ],
                style={"minWidth": "190px", "flex": "1 1 190px"},
            ))
        if not rows:
            continue
        sections_html.append(html.Div(
            [
                _panel_title(title),
                html.Div(rows, style={"display": "flex", "flexWrap": "wrap", "gap": "16px 20px"}),
            ],
            style={**CARD_STYLE, "marginBottom": "14px"},
        ))

    brand = v.get("BRAND_UPDATE") or ""
    model = v.get("MARKET_MODEL") or ""
    headline = f"{brand} {model}".strip() or f"Vehicle {v.get('VEHICLE_ID')}"

    return html.Div([
        html.Div(
            [
                html.Div(headline, style={"fontWeight": "700", "fontSize": "19px", "color": "#1a1a2e"}),
                _status_badge(v.get("NOVA_ASSET_STATUS") or "—"),
            ],
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                   "marginBottom": "4px"},
        ),
        html.Div(f"Vehicle ID {v.get('VEHICLE_ID')}",
                 style={"fontSize": "12px", "color": "#718096", "marginBottom": "18px"}),
        html.Div(sections_html),
    ])


# ── Customer info card ───────────────────────────────────────────────────────

def _info_row(label, value):
    return html.Div(
        [
            html.Span(label, style={"fontSize": "11px", "color": "#718096", "fontWeight": "600"}),
            html.Span(value, style={"fontSize": "13px", "color": "#1a1a2e", "fontWeight": "700"}),
        ],
        style={"display": "flex", "justifyContent": "space-between", "padding": "6px 0",
               "borderBottom": "1px solid #f0f2f5"},
    )


def _render_customer_info(info: dict):
    if not info:
        return None
    exposure_s = f"{format_millions(info.get('exposure') or 0)} million"
    asof = info.get("asof_date")
    name = info.get("customer_name")
    cust_id = info.get("customer_id")
    return html.Div(
        [
            _panel_title("Customer"),
            # Name in large text when known; falls back to the ID alone (as
            # before) if CUSTOMER_NAME isn't in this NOVA export yet.
            html.Div(name or cust_id, style={"fontWeight": "700", "fontSize": "17px",
                                              "color": "#1a1a2e", "marginBottom": "2px"}),
            *([html.Div(cust_id, style={"fontSize": "12px", "color": "#718096",
                                         "fontWeight": "600", "marginBottom": "2px"})]
              if name else []),
            html.Div(info.get("industry") or "—", style={"fontSize": "12px", "color": "#718096",
                                                           "marginBottom": "4px" if asof else "14px"}),
            *([html.Div(f"Date: {asof}", style={"fontSize": "11px", "color": "#a0aec0",
                                                  "fontWeight": "600", "marginBottom": "14px"})]
              if asof else []),
            _info_row("Country", info.get("country") or "—"),
            _info_row("Shared Client Flag", info.get("shared_flag") or "—"),
            _info_row("Number of Vehicles", info.get("n_vehicles", 0)),

            html.Div("Credit", style={**PANEL_TITLE_STYLE, "marginTop": "18px"}),
            _info_row("Group Rating", _rating_display(info.get("group_rating"))),
            _info_row("Counterparty Rating", _rating_display(info.get("counterparty_rating"))),
            _info_row("CLS Rating", _rating_display(info.get("cls_rating"))),
            _info_row("Total Exposure", exposure_s),
        ],
        style=CUSTOMER_CARD_STYLE,
    )


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Store(id="customer-info-store", data=None),
        dcc.Store(id="customer-vehicles-store", data=[]),
        dcc.Store(id="grid-page-store", data=0),

        html.Div(
            [
                html.H1("Customer & Vehicle Explorer",
                        style={"margin": "0 0 14px", "fontSize": "22px", "fontWeight": "700",
                               "color": "#1a1a2e"}),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="filter-country", options=COUNTRY_FILTER_OPTIONS,
                            placeholder="All countries", clearable=True,
                            style={"minWidth": "180px"},
                        ),
                        dcc.Dropdown(
                            id="customer-id-input", options=[], placeholder="Search a Customer ID…",
                            searchable=True, clearable=True,
                            style={"flex": "1", "maxWidth": "360px", "fontSize": "13px"},
                        ),
                        dcc.Dropdown(
                            id="asof-date-input", options=ASOF_DATE_OPTIONS,
                            placeholder="Date", clearable=True,
                            style={"minWidth": "170px"},
                        ),
                        html.Button("Search", id="btn-search", n_clicks=0, style=PRIMARY_BTN_STYLE),
                    ],
                    style={"display": "flex", "gap": "10px", "alignItems": "center"},
                ),
                html.Div(id="search-error", style=ERROR_STYLE),
            ],
            style={"padding": "20px 28px", "borderBottom": "1px solid #e2e8f0", "background": "#ffffff"},
        ),

        html.Div(
            "Enter a Customer ID above and press Search to see their fleet.",
            id="empty-state",
            style={"padding": "80px 20px", "textAlign": "center", "color": "#a0aec0",
                   "fontSize": "14px"},
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                _panel_title("Filters"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Brand", style=FIELD_TITLE_STYLE),
                                                dcc.Dropdown(id="filter-brand", options=[], multi=True,
                                                             placeholder="All brands",
                                                             style={"minWidth": "180px"}),
                                            ],
                                            style={"flex": "1"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Body Type", style=FIELD_TITLE_STYLE),
                                                dcc.Dropdown(id="filter-bodytype", options=[], multi=True,
                                                             placeholder="All body types",
                                                             style={"minWidth": "180px"}),
                                            ],
                                            style={"flex": "1"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Asset Status", style=FIELD_TITLE_STYLE),
                                                dcc.Dropdown(id="filter-status", options=[], multi=True,
                                                             placeholder="All statuses",
                                                             style={"minWidth": "180px"}),
                                            ],
                                            style={"flex": "1"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Sort by Delivery Date", style=FIELD_TITLE_STYLE),
                                                dcc.Dropdown(
                                                    id="sort-direction",
                                                    options=[
                                                        {"label": "Newest first", "value": "desc"},
                                                        {"label": "Oldest first", "value": "asc"},
                                                    ],
                                                    value="desc", clearable=False,
                                                    style={"minWidth": "160px"},
                                                ),
                                            ],
                                            style={"flex": "1"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                                ),
                                html.Div(
                                    html.Button("Refresh", id="btn-refresh-filters", n_clicks=0,
                                                className="cve-secondary-btn", style=SECONDARY_BTN_STYLE),
                                    style={"display": "flex", "justifyContent": "flex-end", "marginTop": "12px"},
                                ),
                            ],
                            style={**CARD_STYLE, "marginBottom": "16px"},
                        ),
                        html.Div(
                            id="vehicle-grid",
                            style={"display": "grid",
                                   "gridTemplateColumns": "repeat(auto-fill, minmax(260px, 1fr))",
                                   "gap": "14px"},
                        ),
                        html.Div(
                            [
                                html.Button("←", id="btn-page-prev", n_clicks=0,
                                            className="page-nav-button"),
                                html.Span("Page 1 of 1", id="grid-page-label",
                                          style={"fontSize": "12px", "color": "#718096", "fontWeight": "600"}),
                                html.Button("→", id="btn-page-next", n_clicks=0,
                                            className="page-nav-button"),
                            ],
                            style={"display": "flex", "justifyContent": "center", "alignItems": "center",
                                   "gap": "14px", "marginTop": "18px"},
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                ),
                html.Div(id="customer-info-card", style={"width": "300px", "minWidth": "300px"}),
            ],
            id="results-section",
            style={"display": "none", "gap": "20px", "alignItems": "flex-start",
                   "padding": "24px 28px"},
        ),

        html.Div(
            id="vehicle-modal",
            style=MODAL_OVERLAY_STYLE,
            children=[
                html.Div(
                    style=VEHICLE_MODAL_PANEL_STYLE,
                    children=[
                        html.Div(
                            html.Button("Close", id="vehicle-modal-close", n_clicks=0,
                                        className="cve-secondary-btn", style=SECONDARY_BTN_STYLE),
                            style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"},
                        ),
                        html.Div(id="vehicle-modal-body"),
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

_CUSTOMER_OPTIONS_CAP = 200  # cap the list shown before the user types anything, so the
                             # dropdown doesn't try to render every customer at once


@app.callback(
    Output("customer-id-input", "options"),
    Input("customer-id-input", "search_value"),
    Input("filter-country", "value"),
    State("customer-id-input", "value"),
)
def _update_customer_options(search_value, country, current_value):
    d = CUSTOMER_INDEX_DF
    if country and "COUNTRY" in d.columns:
        d = d[d["COUNTRY"] == country]
    if search_value:
        d = d[d["ID_CUSTOMER"].str.upper().str.contains(search_value.strip().upper(), na=False, regex=False)]
    else:
        d = d.head(_CUSTOMER_OPTIONS_CAP)

    has_name = "CUSTOMER_NAME" in d.columns
    rows = list(d[["ID_CUSTOMER", "CUSTOMER_NAME"]].itertuples(index=False)) if has_name else \
        [(cid, None) for cid in d["ID_CUSTOMER"]]
    ids = [r[0] for r in rows]
    if current_value and current_value not in ids:
        extra_name = None
        if has_name:
            match = CUSTOMER_INDEX_DF[CUSTOMER_INDEX_DF["ID_CUSTOMER"] == current_value]
            extra_name = match["CUSTOMER_NAME"].iloc[0] if not match.empty else None
        rows = [(current_value, extra_name)] + rows

    return [{"label": f"{cid} — {name}" if pd.notna(name) and name else cid, "value": cid} for cid, name in rows]


@app.callback(
    Output("customer-info-store", "data"),
    Output("customer-vehicles-store", "data"),
    Output("search-error", "children"),
    Output("search-error", "style"),
    Output("results-section", "style"),
    Output("empty-state", "style"),
    Input("btn-search", "n_clicks"),
    State("customer-id-input", "value"),
    State("asof-date-input", "value"),
    prevent_initial_call=True,
)
def _search_customer(_n_clicks, cust_id, asof_date):
    cust_id = (cust_id or "").strip()
    if not cust_id:
        return (no_update, no_update, "Please enter a Customer ID.",
                {**ERROR_STYLE, "display": "block"}, {"display": "none"}, {"display": "block"})

    mask = pd.Series(False, index=CV_DF.index)
    if "ID_CUSTOMER" in CV_DF.columns:
        mask = mask | (CV_DF["ID_CUSTOMER"].astype(str).str.strip().str.upper() == cust_id.upper())
    if "OBLIGOR_IDENTIFIER" in CV_DF.columns:
        mask = mask | (CV_DF["OBLIGOR_IDENTIFIER"].astype(str).str.strip().str.upper() == cust_id.upper())
    customer_rows = CV_DF[mask]

    # "As of" a COB Date: a vehicle is only ever IN FLEET / SOLD / ... as of a
    # specific snapshot, so restrict to snapshots at or before the chosen date
    # before anything downstream (profile/exposure/vehicle list) picks the
    # "latest" row per contract — that then naturally becomes "latest as of
    # that date", and contracts that only appear later correctly drop out.
    asof_ts = None
    if asof_date and "COB_DATE" in customer_rows.columns:
        asof_ts = pd.to_datetime(asof_date)
        customer_rows = customer_rows[customer_rows["COB_DATE"] <= asof_ts]

    if customer_rows.empty:
        suffix = f" as of {asof_ts.date()}" if asof_ts is not None else ""
        return (None, [], f'No customer found for ID "{cust_id}"{suffix}.',
                {**ERROR_STYLE, "display": "block"}, {"display": "none"}, {"display": "block"})

    profile = _customer_profile(customer_rows)
    exposure = _customer_exposure(customer_rows)

    keys = [k for k in UNIQUE_KEY_COLS if k in customer_rows.columns]
    veh = customer_rows.sort_values("COB_DATE") if "COB_DATE" in customer_rows.columns else customer_rows
    if keys:
        veh = veh.drop_duplicates(subset=keys, keep="last")

    info = _json_safe_dict({
        "customer_id": cust_id,
        "customer_name": profile.get("CUSTOMER_NAME"),
        "country": profile.get("COUNTRY"),
        "industry": profile.get("ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION"),
        "shared_flag": profile.get("SHARED_CLIENT_FLAG"),
        "group_rating": profile.get("GROUP_RATING"),
        "counterparty_rating": profile.get("COUNTERPARTY_RATING"),
        "cls_rating": profile.get("CLS_GROUP_RATING"),
        "exposure": exposure,
        "n_vehicles": int(len(veh)),
        "asof_date": asof_ts.strftime("%Y-%m-%d") if asof_ts is not None else None,
    })
    vehicles = _json_safe_records(veh)

    return (info, vehicles, "", {**ERROR_STYLE, "display": "none"},
            {"display": "flex", "gap": "20px", "alignItems": "flex-start", "padding": "24px 28px"},
            {"display": "none"})


@app.callback(
    Output("customer-info-card", "children"),
    Input("customer-info-store", "data"),
)
def _update_customer_card(info):
    return _render_customer_info(info)


GRID_PAGE_SIZE = 20


@app.callback(
    Output("vehicle-grid", "children"),
    Output("filter-brand", "options"),
    Output("filter-bodytype", "options"),
    Output("filter-status", "options"),
    Output("filter-brand", "value"),
    Output("filter-bodytype", "value"),
    Output("filter-status", "value"),
    Output("grid-page-store", "data"),
    Output("grid-page-label", "children"),
    Output("btn-page-prev", "disabled"),
    Output("btn-page-next", "disabled"),
    Input("customer-vehicles-store", "data"),
    Input("btn-refresh-filters", "n_clicks"),
    Input("btn-page-prev", "n_clicks"),
    Input("btn-page-next", "n_clicks"),
    State("filter-brand", "value"),
    State("filter-bodytype", "value"),
    State("filter-status", "value"),
    State("sort-direction", "value"),
    State("grid-page-store", "data"),
)
def _render_grid(vehicles, _refresh_clicks, _prev_clicks, _next_clicks,
                  brands_f, bodytypes_f, statuses_f, sort_dir, page):
    # Filters/sort only ever apply on an explicit click (Refresh, or Prev/Next
    # while paging) — never live while a dropdown value is merely being
    # changed, per the user's request. A fresh customer search always resets
    # the filter selections and goes back to page 1.
    trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
    is_new_list = trig in (None, "customer-vehicles-store")
    if is_new_list:
        brands_f, bodytypes_f, statuses_f = None, None, None
    brand_val = None if is_new_list else brands_f
    bt_val = None if is_new_list else bodytypes_f
    status_val = None if is_new_list else statuses_f

    vehicles = vehicles or []
    if not vehicles:
        return (html.P("No vehicles for this customer.", style={"color": "#a0aec0", "fontSize": "13px"}),
                [], [], [], brand_val, bt_val, status_val, 0, "Page 0 of 0", True, True)

    df = pd.DataFrame(vehicles)

    brand_opts = ([{"label": b, "value": b} for b in sorted(df["BRAND_UPDATE"].dropna().unique())]
                  if "BRAND_UPDATE" in df.columns else [])
    bt_col = "MARKET_BODY_GROUP" if "MARKET_BODY_GROUP" in df.columns else (
        "VEHICLE_BODY_TYPE" if "VEHICLE_BODY_TYPE" in df.columns else None)
    bt_opts = ([{"label": b, "value": b} for b in sorted(df[bt_col].dropna().unique())]
               if bt_col else [])
    status_opts = ([{"label": s, "value": s} for s in sorted(df["NOVA_ASSET_STATUS"].dropna().unique())]
                   if "NOVA_ASSET_STATUS" in df.columns else [])

    d = df
    if brands_f and "BRAND_UPDATE" in d.columns:
        d = d[d["BRAND_UPDATE"].isin(brands_f)]
    if bodytypes_f and bt_col:
        d = d[d[bt_col].isin(bodytypes_f)]
    if statuses_f and "NOVA_ASSET_STATUS" in d.columns:
        d = d[d["NOVA_ASSET_STATUS"].isin(statuses_f)]

    if "DELIVERY_DATE" in d.columns:
        d = d.copy()
        d["_delivery_sort"] = pd.to_datetime(d["DELIVERY_DATE"], errors="coerce")
        d = d.sort_values("_delivery_sort", ascending=(sort_dir == "asc"), na_position="last")

    n = len(d)
    total_pages = max(1, -(-n // GRID_PAGE_SIZE))  # ceil division, no extra import
    if trig == "btn-page-prev":
        page = max(0, (page or 0) - 1)
    elif trig == "btn-page-next":
        page = min(total_pages - 1, (page or 0) + 1)
    else:
        page = 0
    page = max(0, min(page, total_pages - 1))

    if d.empty:
        return (html.P("No vehicles match these filters.", style={"color": "#a0aec0", "fontSize": "13px"}),
                brand_opts, bt_opts, status_opts, brand_val, bt_val, status_val, 0, "Page 0 of 0", True, True)

    start = page * GRID_PAGE_SIZE
    page_d = d.iloc[start:start + GRID_PAGE_SIZE]
    # Re-sanitize: rebuilding a DataFrame from the store's records and calling
    # .to_dict() again turns None back into NaN for numeric-ish columns, and
    # NaN is truthy in Python — an unguarded `v.get(...) or "—"` fallback
    # would then render the literal string "nan" instead of falling back.
    cards = [_vehicle_card(_json_safe_dict(row)) for row in page_d.to_dict("records")]
    label = f"Page {page + 1} of {total_pages} ({n} vehicle{'s' if n != 1 else ''})"
    return (cards, brand_opts, bt_opts, status_opts, brand_val, bt_val, status_val,
            page, label, page <= 0, page >= total_pages - 1)


@app.callback(
    Output("vehicle-modal", "style"),
    Output("vehicle-modal-body", "children"),
    Input({"type": "vehicle-more-details", "index": ALL}, "n_clicks"),
    Input("vehicle-modal-close", "n_clicks"),
    State("customer-vehicles-store", "data"),
    prevent_initial_call=True,
)
def _toggle_vehicle_modal(card_clicks, _close_clicks, vehicles):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if "vehicle-modal-close" in trig:
        return {**MODAL_OVERLAY_STYLE, "display": "none"}, no_update

    if not any(card_clicks or []):
        return no_update, no_update

    prop_id = trig.rsplit(".", 1)[0]
    try:
        vid = json.loads(prop_id)["index"]
    except Exception:
        return no_update, no_update

    vehicles = vehicles or []
    match = next((v for v in vehicles if str(v.get("VEHICLE_ID")) == str(vid)), None)
    if match is None:
        return no_update, no_update

    return {**MODAL_OVERLAY_STYLE, "display": "flex"}, _render_vehicle_detail(match)


# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8057)
