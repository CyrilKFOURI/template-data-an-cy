
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, dash_table
from dash.exceptions import PreventUpdate


BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
APP_TITLE = "Data Explorer"
MAX_PLOT_ROWS = 50_000
MAX_TOP_CATEGORIES = 20
MAX_CATEGORICAL_VALUES_FOR_SELECTOR = 100
MAX_HEATMAP_LEVELS = 25
MAX_HIGH_CARDINALITY = 100_000
MAX_KPI_CHART_POINTS = 36
MAX_CATEGORY_TREND = 10
MAX_INTERACTION_CORR_COLS = 15
DEFAULT_LOAD_SAMPLE_PCT = 100.0
DEFAULT_SELECTED_LOAD_COLUMNS = [
    "EXTENSION_DATE",
    "CLASS_CATALOG",
    "CONTRACT_START_DATE",
    "CONTRACT_END_DATE",
    "COB_DATE",
    "DATE_OF_ORDER",
    "CONTRACT_END_DATE_AMENDED",
    "ID_CONTRACT",
    "VEHICLE_ID",
    "ID_QUOTATION",
    "FINAL_CONTRACT_DURATION",
    "POWER_CATEGORY",
    "BIKE_OR_CAR",
    "CLS_VEHICLE_TYPE",
    "COUNTRY",
    "BRAND_UPDATE",
    "VEHICLE_CLASS",
    "VEHICLE_MODEL",
    "MODEL_CATALOG",
    "OEM_UPDATE",
    "NOVA_ASSET_STATUS",
    "FUEL_TYPE2",
    "FUEL_TYPE",
]
DEFAULT_COUNTRIES = ["ITALY", "SPAIN", "BELGIUM", "FRANCE", "GERMANY", "LUXEMBOURG", "NETHERLANDS", "UNITED KINGDOM"]
TIME_GRAINS = ["monthly", "quarterly", "yearly"]
AGG_FUNCS = ["count", "sum", "mean", "median", "min", "max"]
PLOT_TYPES = ["auto", "histogram", "box", "violin", "bar", "scatter", "heatmap", "line"]
COB_TEMPORAL_COLUMNS = {"COB_YEAR", "COB_MONTH", "COB_QUARTER"}
ADV_FILTER_ROWS = 3
MAX_ADV_FILTER_ROWS = 12
ADV_FILTER_LOGIC_OPTIONS = ["AND", "OR"]
TEXT_FILTER_THRESHOLD = 100
MONTH_OPTIONS = [{"label": f"{m:02d}", "value": m} for m in range(1, 13)]

DATA_CACHE: dict[str, pd.DataFrame] = {}
CATALOG_CACHE: pd.DataFrame | None = None


@dataclass(frozen=True)
class CatalogItem:
    file_path: Path
    country: str
    yyyymm: str
    period: pd.Timestamp


app = Dash(
    __name__,
    title=APP_TITLE,
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "assets"),
)
server = app.server


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text


def truncate_label(value: Any, max_length: int = 28) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def metric_kv_card(label: str, value: Any) -> html.Div:
    return html.Div(
        [
            html.Div(str(label), className="metric-label"),
            html.Div(str(value), className="metric-value-small"),
        ],
        className="summary-kv-card",
    )


def month_period_start(yyyymm: str) -> pd.Timestamp:
    return pd.Timestamp(year=int(yyyymm[:4]), month=int(yyyymm[4:6]), day=1)


def month_year_to_start(year_value: int | None, month_value: int | None) -> pd.Timestamp | None:
    if year_value is None or month_value is None:
        return None
    return pd.Timestamp(year=int(year_value), month=int(month_value), day=1)


def month_year_to_end(year_value: int | None, month_value: int | None) -> pd.Timestamp | None:
    if year_value is None or month_value is None:
        return None
    return pd.Timestamp(year=int(year_value), month=int(month_value), day=1) + pd.offsets.MonthEnd(1)


def build_month_year_range_block(
    start_year_id: Any,
    start_month_id: Any,
    end_year_id: Any,
    end_month_id: Any,
    year_options: list[dict[str, Any]],
    start_year_value: int | None,
    start_month_value: int | None,
    end_year_value: int | None,
    end_month_value: int | None,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Debut", className="hint-text"),
                            html.Div(
                                [
                                    dcc.Dropdown(id=start_year_id, options=year_options, value=start_year_value, placeholder="Année", clearable=False),
                                    dcc.Dropdown(id=start_month_id, options=MONTH_OPTIONS, value=start_month_value, placeholder="Mois", clearable=False),
                                ],
                                className="month-pair-row",
                            ),
                        ],
                        className="month-side",
                    ),
                    html.Div(className="month-range-divider"),
                    html.Div(
                        [
                            html.Div("Fin", className="hint-text"),
                            html.Div(
                                [
                                    dcc.Dropdown(id=end_year_id, options=year_options, value=end_year_value, placeholder="Année", clearable=False),
                                    dcc.Dropdown(id=end_month_id, options=MONTH_OPTIONS, value=end_month_value, placeholder="Mois", clearable=False),
                                ],
                                className="month-pair-row",
                            ),
                        ],
                        className="month-side",
                    ),
                ],
                className="month-range-columns",
            ),
        ]
        ,
        className="month-range-block",
    )


def format_year_month(year: int | None, month: int | None) -> str:
    if year is None or month is None:
        return ""
    return f"{int(year):04d}-{int(month):02d}"


def scan_catalog() -> pd.DataFrame:
    global CATALOG_CACHE
    if CATALOG_CACHE is not None:
        return CATALOG_CACHE.copy()

    rows: list[dict[str, Any]] = []
    # Use Path() to ensure we have a Path object, then glob directly
    # glob() will return an empty list if the path is invalid or empty
    data_path = Path(DATA_FOLDER)
    
    # Use a more flexible regex to handle potential extra spaces
    pattern = re.compile(r"^NOVA\s*-\s*(?P<country>.+?)\s*-\s*(?P<yyyymm>\d{6})$", re.IGNORECASE)
    
    for file_path in sorted(data_path.glob("*.parquet")):
        # Ensure we are working with the filename without extension
        match = pattern.match(file_path.stem)
        if not match:
            continue
            
        country = match.group("country").strip().upper()
        yyyymm = match.group("yyyymm")
        
        try:
            period_val = month_period_start(yyyymm)
        except Exception:
            continue

        rows.append(
            {
                "file_path": file_path,
                "country": country,
                "yyyymm": yyyymm,
                "period": period_val,
            }
        )

    catalog = pd.DataFrame(rows, columns=["file_path", "country", "yyyymm", "period"])
    if not catalog.empty:
        catalog = catalog.sort_values(["country", "yyyymm"]).reset_index(drop=True)
    CATALOG_CACHE = catalog
    return catalog.copy()


def available_country_options(catalog: pd.DataFrame) -> list[dict[str, str]]:
    countries = sorted(catalog["country"].dropna().astype(str).unique().tolist()) if not catalog.empty else []
    return [{"label": country.title(), "value": country} for country in countries]


def available_date_bounds(catalog: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if catalog.empty:
        return None, None
    return catalog["period"].min(), catalog["period"].max()


def discover_available_load_columns(catalog: pd.DataFrame) -> list[str]:
    if catalog.empty or "file_path" not in catalog.columns:
        return []

    sample_row = catalog.sample(n=1, random_state=42).iloc[0]
    sample_path = Path(sample_row["file_path"])

    try:
        sample_df = pd.read_parquet(sample_path).head(1)
        return [str(column) for column in sample_df.columns.tolist()]
    except Exception:
        return []


def load_selected_data(
    countries: list[str],
    start_date: str | None,
    end_date: str | None,
    sample_pct: float = DEFAULT_LOAD_SAMPLE_PCT,
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    catalog = scan_catalog()
    if catalog.empty:
        raise FileNotFoundError(f"Aucun parquet trouvé dans {DATA_FOLDER}")

    selected = catalog.copy()
    countries_norm = {normalize_text(country) for country in countries if country}
    if countries_norm:
        selected = selected[selected["country"].isin(countries_norm)]

    if start_date:
        selected = selected[selected["period"] >= pd.to_datetime(start_date)]
    if end_date:
        selected = selected[selected["period"] <= pd.to_datetime(end_date)]

    if selected.empty:
        raise ValueError("Aucun fichier ne correspond aux filtres pays / dates.")

    safe_sample_pct = float(sample_pct if sample_pct is not None else DEFAULT_LOAD_SAMPLE_PCT)
    safe_sample_pct = min(100.0, max(0.0, safe_sample_pct))
    sample_frac = safe_sample_pct / 100.0
    selected_columns_clean = [str(column) for column in (selected_columns or []) if str(column).strip()]

    frames: list[pd.DataFrame] = []
    for idx, file_path in enumerate(selected["file_path"].tolist()):
        if selected_columns_clean:
            try:
                frame = pd.read_parquet(file_path, columns=selected_columns_clean)
            except Exception:
                frame = pd.read_parquet(file_path)
                available_cols = [column for column in selected_columns_clean if column in frame.columns]
                frame = frame[available_cols] if available_cols else frame.iloc[:, 0:0]
        else:
            frame = pd.read_parquet(file_path)

        if sample_frac < 1.0 and not frame.empty:
            keep_rows = max(1, int(np.floor(len(frame) * sample_frac)))
            frame = frame.sample(n=min(keep_rows, len(frame)), random_state=42 + idx)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return prepare_dataframe(df)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    date_candidates = [
        "COB_DATE",
        "CONTRACT_START_DATE",
        "CONTRACT_END_DATE",
        "CONTRACT_END_DATE_AMENDED",
        "EXTENSION_DATE",
        "DATE_OF_ORDER",
        "CONTRACT_FINAL_END",
    ]
    for column in date_candidates:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")

    if "COB_DATE" in out.columns:
        out["COB_YEAR"] = out["COB_DATE"].dt.year
        out["COB_MONTH"] = out["COB_DATE"].dt.month
        out["COB_QUARTER"] = "Q" + out["COB_DATE"].dt.quarter.astype("Int64").astype(str)

    if "CONTRACT_START_DATE" in out.columns:
        out["CONTRACT_START_YEAR"] = out["CONTRACT_START_DATE"].dt.year
        out["CONTRACT_START_MONTH"] = out["CONTRACT_START_DATE"].dt.month

    for column in out.columns:
        if pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].astype(str)

    return out


def infer_column_kind(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "unknown"
    series = df[column]
    upper = column.upper()
    if upper in COB_TEMPORAL_COLUMNS:
        return "date"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Avoid false positives like BRAND_UPDATE / OEM_UPDATE (which end with "DATE" in UPDATE).
    if upper.endswith("_DATE") or upper.endswith("_DT"):
        return "date"
    return "categorical"


def coerce_temporal_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="datetime64[ns]")

    base = df[column]
    if pd.api.types.is_datetime64_any_dtype(base):
        return pd.to_datetime(base, errors="coerce")

    upper = column.upper()
    if upper == "COB_DATE":
        return pd.to_datetime(df[column], errors="coerce")

    if upper == "COB_YEAR":
        year = pd.to_numeric(df[column], errors="coerce")
        return pd.to_datetime({"year": year, "month": 1, "day": 1}, errors="coerce")

    if upper == "COB_MONTH":
        month = pd.to_numeric(df[column], errors="coerce")
        if "COB_YEAR" in df.columns:
            year = pd.to_numeric(df["COB_YEAR"], errors="coerce")
        elif "COB_DATE" in df.columns:
            year = pd.to_datetime(df["COB_DATE"], errors="coerce").dt.year
        else:
            year = pd.Series(2000, index=df.index)
        return pd.to_datetime({"year": year, "month": month, "day": 1}, errors="coerce")

    if upper == "COB_QUARTER":
        quarter = (
            df[column]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
        month = (quarter - 1) * 3 + 1
        if "COB_YEAR" in df.columns:
            year = pd.to_numeric(df["COB_YEAR"], errors="coerce")
        elif "COB_DATE" in df.columns:
            year = pd.to_datetime(df["COB_DATE"], errors="coerce").dt.year
        else:
            year = pd.Series(2000, index=df.index)
        return pd.to_datetime({"year": year, "month": month, "day": 1}, errors="coerce")

    return pd.to_datetime(df[column], errors="coerce")


def temporal_period_labels(df: pd.DataFrame, date_col: str) -> pd.Series:
    upper = date_col.upper()
    if upper == "COB_YEAR":
        dt = coerce_temporal_series(df, date_col)
        return dt.dt.to_period("Y").astype(str)
    if upper == "COB_QUARTER":
        return df[date_col].astype(str).astype(object)
    if upper == "COB_MONTH":
        month_num = df[date_col].astype(str).str.zfill(2)
        return ("M" + month_num).astype(object)
    dt = coerce_temporal_series(df, date_col)
    return dt.dt.to_period("M").astype(str)


def get_column_choices(df: pd.DataFrame) -> list[dict[str, str]]:
    return [{"label": column, "value": column} for column in df.columns]


def top_categories(series: pd.Series, limit: int = MAX_TOP_CATEGORIES) -> pd.Series:
    counts = series.fillna("<NA>").astype(str).value_counts(dropna=False)
    if len(counts) <= limit:
        return series.fillna("<NA>").astype(str)
    top_values = set(counts.head(limit).index.tolist())
    return series.fillna("<NA>").astype(str).where(series.fillna("<NA>").astype(str).isin(top_values), other="OTHER")


def sample_dataframe(df: pd.DataFrame, max_rows: int = MAX_PLOT_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def month_grain_series(series: pd.Series, grain: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if grain == "monthly":
        return values.dt.to_period("M").astype(str)
    if grain == "quarterly":
        return values.dt.to_period("Q").astype(str)
    return values.dt.year.astype("Int64").astype(str)


def build_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 16})
    fig.update_layout(template="plotly_white", height=420, margin={"l": 30, "r": 30, "t": 40, "b": 30})
    return fig


def build_preview_table(df: pd.DataFrame, max_rows: int = 100) -> dash_table.DataTable:
    preview = df.head(max_rows).copy()
    if preview.empty:
        return dash_table.DataTable(data=[], columns=[])

    columns = []
    for column in preview.columns:
        columns.append(
            {
                "name": column,
                "id": column,
                "type": "numeric" if pd.api.types.is_numeric_dtype(preview[column]) else "text",
            }
        )

    return dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=columns,
        page_size=min(max_rows, 20),
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "minWidth": "120px",
            "maxWidth": "220px",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "fontFamily": "Inter, Segoe UI, sans-serif",
            "fontSize": "13px",
        },
        style_header={"fontWeight": "700", "backgroundColor": "#F6F7FB"},
    )


def advanced_operator_options(kind: str) -> list[dict[str, str]]:
    if kind == "numeric":
        return [
            {"label": "Equals", "value": "equals"},
            {"label": "Greater than", "value": ">"},
            {"label": "Greater than or equal", "value": ">="},
            {"label": "Less than", "value": "<"},
            {"label": "Less than or equal", "value": "<="},
        ]
    if kind == "date":
        return [
            {"label": "After", "value": "after"},
            {"label": "Before", "value": "before"},
            {"label": "Between", "value": "between"},
            {"label": "Equals", "value": "equals"},
        ]
    return [
        {"label": "Equals", "value": "equals"},
        {"label": "Contains", "value": "contains"},
        {"label": "Starts with", "value": "starts with"},
        {"label": "Ends with", "value": "ends with"},
    ]


def build_advanced_value_widget(row_index: int, df: pd.DataFrame, column: str | None, operator: str | None) -> html.Div:
    if not column or column not in df.columns:
        return html.Div("Choisis une colonne pour activer le filtre.", className="filter-helper")

    kind = infer_column_kind(df, column)
    operator = (operator or "").lower()

    if kind == "numeric":
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            return html.Div("Aucune valeur numérique exploitable.", className="filter-helper")
        min_v = float(numeric.min())
        max_v = float(numeric.max())
        return html.Div(
            [
                dcc.Input(
                    id={"type": "adv-num-value", "row": row_index},
                    type="number",
                    placeholder="Valeur",
                    debounce=True,
                    className="advanced-input",
                    persistence=True,
                    persistence_type="memory",
                ),
                html.Div(f"Plage disponible: {min_v:.2f} → {max_v:.2f}", className="filter-helper"),
            ]
        )

    if kind == "date":
        date_series = pd.to_datetime(df[column], errors="coerce").dropna()
        if date_series.empty:
            return html.Div("Aucune date exploitable.", className="filter-helper")
        year_values = sorted(int(year) for year in date_series.dt.year.dropna().unique())
        year_options = [{"label": str(year), "value": year} for year in year_values]
        if operator == "between":
            return build_month_year_range_block(
                start_year_id={"type": "adv-date-start-year", "row": row_index},
                start_month_id={"type": "adv-date-start-month", "row": row_index},
                end_year_id={"type": "adv-date-end-year", "row": row_index},
                end_month_id={"type": "adv-date-end-month", "row": row_index},
                year_options=year_options,
                start_year_value=None,
                start_month_value=None,
                end_year_value=None,
                end_month_value=None,
            )
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id={"type": "adv-date-year", "row": row_index},
                            options=year_options,
                            value=None,
                            placeholder="Année",
                            clearable=True,
                            persistence=True,
                            persistence_type="memory",
                        ),
                        dcc.Dropdown(
                            id={"type": "adv-date-month", "row": row_index},
                            options=MONTH_OPTIONS,
                            value=None,
                            placeholder="Mois",
                            clearable=True,
                            persistence=True,
                            persistence_type="memory",
                        ),
                    ],
                    className="month-pair-row",
                ),
                html.Div("Sélectionne un mois et une année.", className="filter-helper"),
            ]
        )

    series = df[column].fillna("<NA>").astype(str)
    counts = series.value_counts(dropna=False)
    if operator == "equals":
        options = [{"label": f"{truncate_label(index, 42)} ({count})", "value": index} for index, count in counts.items()]
        return html.Div(
            [
                dcc.Checklist(
                    id={"type": "adv-cat-all", "row": row_index},
                    options=[{"label": "All", "value": "all"}],
                    value=[],
                    inline=True,
                    persistence=True,
                    persistence_type="memory",
                ),
                dcc.Dropdown(
                    id={"type": "adv-cat-values", "row": row_index},
                    options=options,
                    value=[],
                    multi=True,
                    placeholder="Sélectionne une ou plusieurs valeurs",
                    persistence=True,
                    persistence_type="memory",
                ),
                html.Div("Laisse vide pour ne rien filtrer sur cette règle.", className="filter-helper"),
            ]
        )

    placeholder = "Texte à rechercher"
    if operator in {"starts with", "ends with", "contains", "equals"}:
        placeholder = f"{operator.title()}..."
    return html.Div(
        [
            dcc.Input(
                id={"type": "adv-text-value", "row": row_index},
                type="text",
                placeholder=placeholder,
                debounce=True,
                className="advanced-input",
                persistence=True,
                persistence_type="memory",
            )
        ]
    )


def extract_advanced_rule_mask(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    column = rule.get("column")
    if not column or column not in df.columns:
        return pd.Series(True, index=df.index)

    kind = rule.get("kind") or infer_column_kind(df, column)
    operator = str(rule.get("operator") or "").lower()
    value = rule.get("value")

    if kind == "numeric":
        numeric = pd.to_numeric(df[column], errors="coerce")
        scalar_value = None
        if isinstance(value, dict):
            scalar_value = value.get("value")
        else:
            scalar_value = value
        if scalar_value in (None, ""):
            return pd.Series(True, index=df.index)
        scalar = float(scalar_value)
        if operator == "equals":
            return numeric == scalar
        if operator == ">":
            return numeric > scalar
        if operator == ">=":
            return numeric >= scalar
        if operator == "<":
            return numeric < scalar
        if operator == "<=":
            return numeric <= scalar
        return pd.Series(True, index=df.index)

    if kind == "date":
        dates = pd.to_datetime(df[column], errors="coerce")
        periods = dates.dt.to_period("M")
        if operator == "between":
            start_year = value.get("start_year") if isinstance(value, dict) else None
            start_month = value.get("start_month") if isinstance(value, dict) else None
            end_year = value.get("end_year") if isinstance(value, dict) else None
            end_month = value.get("end_month") if isinstance(value, dict) else None
            if start_year in (None, "") or start_month in (None, "") or end_year in (None, "") or end_month in (None, ""):
                return pd.Series(True, index=df.index)
            start_period = pd.Period(f"{int(start_year):04d}-{int(start_month):02d}", freq="M")
            end_period = pd.Period(f"{int(end_year):04d}-{int(end_month):02d}", freq="M")
            if start_period > end_period:
                return pd.Series(True, index=df.index)
            return (periods >= start_period) & (periods <= end_period)
        date_year = value.get("year") if isinstance(value, dict) else None
        date_month = value.get("month") if isinstance(value, dict) else None
        if date_year in (None, "") or date_month in (None, ""):
            return pd.Series(True, index=df.index)
        selected_period = pd.Period(f"{int(date_year):04d}-{int(date_month):02d}", freq="M")
        if operator == "equals":
            return periods == selected_period
        if operator == "after":
            return periods >= selected_period
        if operator == "before":
            return periods <= selected_period
        return pd.Series(True, index=df.index)

    text = df[column].fillna("<NA>").astype(str)
    if isinstance(value, list):
        cleaned_values = [str(item) for item in value if item not in (None, "")]
    else:
        cleaned_values = [str(value)] if value not in (None, "") else []

    if not cleaned_values:
        return pd.Series(True, index=df.index)

    needle = cleaned_values[0]
    if operator == "contains":
        return text.str.contains(re.escape(needle), case=False, na=False, regex=True)
    if operator == "equals":
        return text.str.upper().isin({item.upper() for item in cleaned_values})
    if operator == "starts with":
        return text.str.startswith(needle, na=False)
    if operator == "ends with":
        return text.str.endswith(needle, na=False)
    return pd.Series(True, index=df.index)


def apply_advanced_filters(df: pd.DataFrame, filter_bundle: dict[str, Any] | None) -> pd.DataFrame:
    if not filter_bundle:
        return df

    rules = filter_bundle.get("rules", [])
    logic = str(filter_bundle.get("logic", "AND")).upper()
    masks: list[pd.Series] = []
    for rule in rules:
        if not rule or not rule.get("column"):
            continue
        masks.append(extract_advanced_rule_mask(df, rule))

    if not masks:
        return df

    final_mask = masks[0].copy()
    for mask in masks[1:]:
        final_mask = final_mask | mask if logic == "OR" else final_mask & mask

    return df[final_mask]


def advanced_filter_summary(filter_bundle: dict[str, Any] | None) -> str:
    if not filter_bundle:
        return "Aucun filtre avancé appliqué."
    rules = filter_bundle.get("rules", [])
    active_rules = [rule for rule in rules if rule.get("column")]
    if not active_rules:
        return "Aucun filtre avancé appliqué."
    logic = str(filter_bundle.get("logic", "AND")).upper()
    parts: list[str] = []
    for idx, rule in enumerate(active_rules, start=1):
        operator = str(rule.get("operator", ""))
        value = rule.get("value")
        if isinstance(value, dict):
            if operator == "between":
                start = format_year_month(value.get("start_year"), value.get("start_month"))
                end = format_year_month(value.get("end_year"), value.get("end_month"))
                value_repr = f"{start} → {end}"
            elif value.get("year") is not None or value.get("month") is not None:
                value_repr = format_year_month(value.get("year"), value.get("month"))
            else:
                value_repr = ", ".join([str(v) for v in value.values() if v not in (None, "")])
        elif isinstance(value, list):
            value_repr = ", ".join([str(item) for item in value[:5]])
        else:
            value_repr = str(value)
        parts.append(f"Filtre {idx}: {rule.get('column')} {operator} {truncate_label(value_repr, 40)}")
    return "\n".join([f"Logique {logic}"] + parts)


def cramers_v(table: pd.DataFrame) -> float:
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    n = total
    r, c = observed.shape
    denom = n * (min(r - 1, c - 1) or 1)
    return float(np.sqrt(max(chi2 / denom, 0.0)))


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    cat = categories.fillna("<NA>").astype(str)
    meas = pd.to_numeric(measurements, errors="coerce")
    valid = meas.notna()
    cat = cat[valid]
    meas = meas[valid]
    if cat.empty or meas.empty:
        return 0.0
    grand_mean = meas.mean()
    ss_between = 0.0
    ss_total = float(((meas - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    for _, group in meas.groupby(cat):
        if group.empty:
            continue
        ss_between += len(group) * float((group.mean() - grand_mean) ** 2)
    return float(np.sqrt(min(ss_between / ss_total, 1.0)))


def interaction_score(df: pd.DataFrame, x_var: str, y_var: str) -> tuple[float, str]:
    x_kind = infer_column_kind(df, x_var)
    y_kind = infer_column_kind(df, y_var)
    work = df[[x_var, y_var]].dropna().copy()
    if work.empty:
        return 0.0, "insufficient-data"

    if x_kind == "numeric" and y_kind == "numeric":
        corr = pd.to_numeric(work[x_var], errors="coerce").corr(pd.to_numeric(work[y_var], errors="coerce"))
        return float(abs(corr)) if pd.notna(corr) else 0.0, "numeric-numeric"
    if x_kind == "categorical" and y_kind == "categorical":
        ctab = pd.crosstab(work[x_var].astype(str), work[y_var].astype(str))
        return cramers_v(ctab), "categorical-categorical"
    if x_kind == "numeric" and y_kind == "categorical":
        return correlation_ratio(work[y_var], work[x_var]), "categorical-numeric"
    if x_kind == "categorical" and y_kind == "numeric":
        return correlation_ratio(work[x_var], work[y_var]), "categorical-numeric"
    if x_kind == "date" and y_kind == "numeric":
        temp = pd.to_datetime(work[x_var], errors="coerce")
        if temp.notna().any():
            temp_num = pd.Series(np.where(temp.notna(), temp.astype("int64"), np.nan), index=temp.index, dtype="float64")
            return float(abs(temp_num.corr(pd.to_numeric(work[y_var], errors="coerce")))), "date-numeric"
    if x_kind == "numeric" and y_kind == "date":
        temp = pd.to_datetime(work[y_var], errors="coerce")
        if temp.notna().any():
            temp_num = pd.Series(np.where(temp.notna(), temp.astype("int64"), np.nan), index=temp.index, dtype="float64")
            return float(abs(pd.to_numeric(work[x_var], errors="coerce").corr(temp_num))), "date-numeric"
    return 0.0, "other"


def build_interaction_ranking(df: pd.DataFrame, focal_column: str | None, limit: int = 10) -> pd.DataFrame:
    if not focal_column or focal_column not in df.columns:
        return pd.DataFrame(columns=["PAIR", "SCORE", "KIND"])

    rows: list[dict[str, Any]] = []
    for column in df.columns:
        if column == focal_column:
            continue
        score, kind = interaction_score(df, focal_column, column)
        if score > 0:
            rows.append(
                {
                    "PAIR": f"{focal_column} × {column}",
                    "LEFT": focal_column,
                    "RIGHT": column,
                    "SCORE": round(score, 4),
                    "KIND": kind,
                    "SUGGESTED_PLOT": "scatter" if kind == "numeric-numeric" else ("heatmap" if kind == "categorical-categorical" else "box"),
                }
            )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values("SCORE", ascending=False).head(limit).reset_index(drop=True)


def build_numeric_correlation_figure(df: pd.DataFrame, columns: list[str] | None = None) -> go.Figure:
    numeric_cols = columns or [column for column in df.columns if infer_column_kind(df, column) == "numeric"]
    numeric_cols = numeric_cols[:15]
    if len(numeric_cols) < 2:
        return build_empty_figure("Pas assez de variables numériques.")
    corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", aspect="auto", title="Matrice de corrélation numérique")
    fig.update_layout(template="plotly_white")
    return fig


def build_encoded_corr_matrix(df: pd.DataFrame, selected_columns: list[str] | None = None, max_cols: int = MAX_INTERACTION_CORR_COLS) -> pd.DataFrame:
    usable = [
        column
        for column in df.columns
        if infer_column_kind(df, column) in {"numeric", "categorical", "date"}
    ]
    if len(usable) < 2:
        return pd.DataFrame()

    if not selected_columns:
        return pd.DataFrame()

    if "__ALL__" in selected_columns and len(selected_columns) == 1:
        ordered = usable[:max_cols]
    else:
        ordered = [column for column in selected_columns if column != "__ALL__" and column in usable][:max_cols]

    if len(ordered) < 2:
        return pd.DataFrame()

    encoded: dict[str, pd.Series] = {}
    for column in ordered:
        kind = infer_column_kind(df, column)
        if kind == "numeric":
            encoded[column] = pd.to_numeric(df[column], errors="coerce")
        elif kind == "date":
            dt = pd.to_datetime(df[column], errors="coerce")
            encoded[column] = pd.Series(np.where(dt.notna(), dt.astype("int64"), np.nan), index=df.index, dtype="float64")
        else:
            text = df[column].fillna("<NA>").astype(str)
            codes, _ = pd.factorize(text, sort=True)
            encoded[column] = pd.Series(np.where(codes >= 0, codes, np.nan), index=df.index, dtype="float64")

    corr = pd.DataFrame(encoded).corr().fillna(0)
    return corr


def build_encoded_correlation_figure(df: pd.DataFrame, selected_columns: list[str] | None = None, max_cols: int = MAX_INTERACTION_CORR_COLS) -> go.Figure:
    corr = build_encoded_corr_matrix(df, selected_columns=selected_columns, max_cols=max_cols)
    if corr.empty or corr.shape[0] < 2:
        return build_empty_figure("Sélectionne au moins 2 variables (ou ALL) pour calculer la corrélation.")
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"Matrice de corrélation encodée (max {max_cols} variables)",
    )
    fig.update_layout(template="plotly_white")
    return fig


def compute_focal_correlation_scores(df: pd.DataFrame, focal_column: str | None, max_cols: int = MAX_INTERACTION_CORR_COLS) -> pd.DataFrame:
    if not focal_column or focal_column not in df.columns:
        return pd.DataFrame(columns=["VARIABLE", "CORR", "ABS_CORR"])

    usable = [
        column
        for column in df.columns
        if column != focal_column and infer_column_kind(df, column) in {"numeric", "categorical", "date"}
    ]
    ordered = [focal_column] + usable[: max(0, max_cols - 1)]
    corr = build_encoded_corr_matrix(df, selected_columns=ordered, max_cols=max_cols)
    if corr.empty or focal_column not in corr.columns:
        return pd.DataFrame(columns=["VARIABLE", "CORR", "ABS_CORR"])

    scores = corr[focal_column].drop(labels=[focal_column], errors="ignore").sort_values(key=np.abs, ascending=False)
    if scores.empty:
        return pd.DataFrame(columns=["VARIABLE", "CORR", "ABS_CORR"])

    plot_df = scores.reset_index()
    plot_df.columns = ["VARIABLE", "CORR"]
    plot_df["ABS_CORR"] = plot_df["CORR"].abs().round(4)
    plot_df["CORR"] = plot_df["CORR"].round(4)
    return plot_df


def build_focal_correlation_figure(df: pd.DataFrame, focal_column: str | None) -> go.Figure:
    plot_df = compute_focal_correlation_scores(df, focal_column, max_cols=MAX_INTERACTION_CORR_COLS)
    if plot_df.empty:
        return build_empty_figure("Corrélation indisponible pour cette variable.")
    fig = px.bar(plot_df, x="VARIABLE", y="CORR", color="CORR", color_continuous_scale="RdBu", range_color=[-1, 1], title=f"Corrélation de {focal_column} avec les autres (encodée)")
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(tickangle=35)
    return fig


def build_conditional_insight_figure(df: pd.DataFrame, focal_column: str | None, target_column: str | None) -> go.Figure:
    if not focal_column or not target_column or focal_column not in df.columns or target_column not in df.columns:
        return build_empty_figure("Choisis une variable focale et une variable cible.")
    if focal_column == target_column:
        return build_empty_figure("La focale et la cible doivent être différentes.")

    work = df[[focal_column, target_column]].copy()
    target = top_categories(work[target_column], limit=MAX_CATEGORY_TREND)
    focal_text = work[focal_column].fillna("").astype(str).str.strip()
    focal_is_null = work[focal_column].isna() | focal_text.eq("") | focal_text.eq("<NA>")
    agg = (
        pd.DataFrame({"TARGET": target, "FOCAL_IS_NULL": focal_is_null})
        .groupby("TARGET")
        .agg(VOLUME=("FOCAL_IS_NULL", "size"), NULL_RATE=("FOCAL_IS_NULL", "mean"))
        .reset_index()
    )
    if agg.empty:
        return build_empty_figure("Pas assez de données pour l'insight conditionnel.")
    agg["NULL_RATE"] = agg["NULL_RATE"] * 100.0

    fig = go.Figure()
    fig.add_bar(x=agg["TARGET"], y=agg["VOLUME"], name="Volume", yaxis="y")
    fig.add_scatter(x=agg["TARGET"], y=agg["NULL_RATE"], mode="lines+markers", name=f"{focal_column} null (%)", yaxis="y2")
    fig.update_layout(
        template="plotly_white",
        title=f"Insight conditionnel: {focal_column} selon {target_column}",
        xaxis_title=target_column,
        yaxis={"title": "Volume"},
        yaxis2={"title": "Taux de null (%)", "overlaying": "y", "side": "right"},
        barmode="group",
    )
    fig.update_xaxes(tickangle=30)
    return fig


def build_encoded_corr_matrix(df: pd.DataFrame, selected_columns: list[str] | None = None, max_cols: int = MAX_INTERACTION_CORR_COLS) -> pd.DataFrame:
    usable = [
        column
        for column in df.columns
        if infer_column_kind(df, column) in {"numeric", "categorical", "date"}
    ]
    if len(usable) < 2:
        return pd.DataFrame()

    if not selected_columns:
        return pd.DataFrame()

    if "__ALL__" in selected_columns and len(selected_columns) == 1:
        ordered = usable[:max_cols]
    else:
        ordered = [column for column in selected_columns if column != "__ALL__" and column in usable][:max_cols]

    if len(ordered) < 2:
        return pd.DataFrame()

    encoded: dict[str, pd.Series] = {}
    for column in ordered:
        kind = infer_column_kind(df, column)
        if kind == "numeric":
            encoded[column] = pd.to_numeric(df[column], errors="coerce")
        elif kind == "date":
            dt = pd.to_datetime(df[column], errors="coerce")
            encoded[column] = pd.Series(np.where(dt.notna(), dt.astype("int64"), np.nan), index=df.index, dtype="float64")
        else:
            text = df[column].fillna("<NA>").astype(str)
            codes, _ = pd.factorize(text, sort=True)
            encoded[column] = pd.Series(np.where(codes >= 0, codes, np.nan), index=df.index, dtype="float64")

    corr = pd.DataFrame(encoded).corr().fillna(0)
    return corr


def build_encoded_correlation_figure(df: pd.DataFrame, selected_columns: list[str] | None = None, max_cols: int = MAX_INTERACTION_CORR_COLS) -> go.Figure:
    corr = build_encoded_corr_matrix(df, selected_columns=selected_columns, max_cols=max_cols)
    if corr.empty or corr.shape[0] < 2:
        return build_empty_figure("Sélectionne au moins 2 variables (ou ALL) pour calculer la corrélation.")
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"Matrice de corrélation encodée (max {max_cols} variables)",
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_focal_correlation_figure(df: pd.DataFrame, focal_column: str | None) -> go.Figure:
    plot_df = compute_focal_correlation_scores(df, focal_column, max_cols=MAX_INTERACTION_CORR_COLS)
    if plot_df.empty:
        return build_empty_figure("Corrélation indisponible pour cette variable.")
    fig = px.bar(plot_df, x="VARIABLE", y="CORR", color="CORR", color_continuous_scale="RdBu", range_color=[-1, 1], title=f"Corrélation de {focal_column} avec les autres (encodée)")
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(tickangle=35)
    return fig


def build_conditional_insight_figure(df: pd.DataFrame, focal_column: str | None, target_column: str | None) -> go.Figure:
    if not focal_column or not target_column or focal_column not in df.columns or target_column not in df.columns:
        return build_empty_figure("Choisis une variable focale et une variable cible.")
    if focal_column == target_column:
        return build_empty_figure("La focale et la cible doivent être différentes.")

    work = df[[focal_column, target_column]].copy()
    target = top_categories(work[target_column], limit=MAX_CATEGORY_TREND)
    focal_text = work[focal_column].fillna("").astype(str).str.strip()
    focal_is_null = work[focal_column].isna() | focal_text.eq("") | focal_text.eq("<NA>")
    agg = (
        pd.DataFrame({"TARGET": target, "FOCAL_IS_NULL": focal_is_null})
        .groupby("TARGET")
        .agg(VOLUME=("FOCAL_IS_NULL", "size"), NULL_RATE=("FOCAL_IS_NULL", "mean"))
        .reset_index()
    )
    if agg.empty:
        return build_empty_figure("Pas assez de données pour l'insight conditionnel.")
    agg["NULL_RATE"] = agg["NULL_RATE"] * 100.0

    fig = go.Figure()
    fig.add_bar(x=agg["TARGET"], y=agg["VOLUME"], name="Volume", yaxis="y")
    fig.add_scatter(x=agg["TARGET"], y=agg["NULL_RATE"], mode="lines+markers", name=f"{focal_column} null (%)", yaxis="y2")
    fig.update_layout(
        template="plotly_white",
        title=f"Insight conditionnel: {focal_column} selon {target_column}",
        xaxis_title=target_column,
        yaxis={"title": "Volume"},
        yaxis2={"title": "Taux de null (%)", "overlaying": "y", "side": "right"},
        barmode="group",
    )
    fig.update_xaxes(tickangle=30)
    return fig


def first_or_none(values: list[Any] | None) -> Any:
    if not values:
        return None
    return values[0]


def safe_top_value_counts(series: pd.Series, top_n: int = MAX_CATEGORICAL_VALUES_FOR_SELECTOR) -> pd.Series:
    as_text = series.fillna("<NA>").astype(str)
    if len(as_text) > 800_000:
        sample = as_text.sample(250_000, random_state=42)
        return sample.value_counts().head(top_n)
    return as_text.value_counts().head(top_n)


def compare_plot_options(x_kind: str, y_kind: str) -> list[dict[str, str]]:
    if x_kind == "numeric" and y_kind == "numeric":
        return [
            {"label": "Auto", "value": "auto"},
            {"label": "Scatter", "value": "scatter"},
            {"label": "Densité 2D", "value": "density"},
        ]
    if x_kind == "categorical" and y_kind == "categorical":
        return [
            {"label": "Auto", "value": "auto"},
            {"label": "Heatmap", "value": "heatmap"},
            {"label": "Barres empilées", "value": "stacked"},
        ]
    if (x_kind == "date" and y_kind == "categorical") or (x_kind == "categorical" and y_kind == "date"):
        return [
            {"label": "Auto", "value": "auto"},
            {"label": "Barres groupées + ligne (%)", "value": "grouped-line"},
            {"label": "Barres groupées", "value": "grouped"},
            {"label": "Lignes par catégorie (Top 10)", "value": "line-category"},
            {"label": "Aire empilée (Top 10)", "value": "stacked-area"},
            {"label": "Barres empilées (Top 10)", "value": "stacked"},
        ]
    if (x_kind == "date" and y_kind == "numeric") or (x_kind == "numeric" and y_kind == "date"):
        return [
            {"label": "Auto", "value": "auto"},
            {"label": "Line", "value": "line"},
            {"label": "Scatter", "value": "scatter"},
        ]
    if (x_kind == "numeric" and y_kind == "categorical") or (x_kind == "categorical" and y_kind == "numeric"):
        return [
            {"label": "Auto", "value": "auto"},
            {"label": "Boxplot", "value": "box"},
            {"label": "Violin", "value": "violin"},
            {"label": "Barres (moyenne)", "value": "bar"},
        ]
    return [{"label": "Auto", "value": "auto"}]


def eda_plot_options(kind: str) -> list[dict[str, str]]:
    if kind == "date":
        allowed = ["auto", "line", "bar"]
    elif kind == "numeric":
        allowed = ["auto", "histogram", "bar", "box", "violin"]
    else:
        allowed = ["auto", "bar"]
    return [{"label": value.title(), "value": value} for value in allowed]


def format_metric_card(label: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ],
        className="metric-card",
    )


def build_dataset_summary(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("Aucune donnée chargée.", className="empty-state")
    n_obs = int(len(df))
    n_vars = int(df.shape[1])
    total_cells = max(n_obs * n_vars, 1)
    missing_cells = int(df.isna().sum().sum())
    missing_pct = missing_cells / total_cells * 100.0
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / n_obs * 100.0) if n_obs else 0.0
    mem_total = int(df.memory_usage(deep=True).sum())
    avg_record = int(mem_total / n_obs) if n_obs else 0

    kinds = [infer_column_kind(df, column) for column in df.columns]
    kind_counts = pd.Series(kinds).value_counts().to_dict()
    kind_repr = " | ".join([f"{key}: {value}" for key, value in sorted(kind_counts.items())])

    return html.Div(
        [
            html.H4("Dataset statistics"),
            html.Div(
                [
                    metric_kv_card("Number of variables", n_vars),
                    metric_kv_card("Number of observations", f"{n_obs:,}".replace(",", " ")),
                    metric_kv_card("Missing cells", f"{missing_cells:,}".replace(",", " ")),
                    metric_kv_card("Missing cells (%)", format_pct(missing_pct)),
                    metric_kv_card("Duplicate rows", f"{dup_rows:,}".replace(",", " ")),
                    metric_kv_card("Duplicate rows (%)", format_pct(dup_pct)),
                    metric_kv_card("Total size in memory", format_bytes(mem_total)),
                    metric_kv_card("Average record size in memory", format_bytes(avg_record)),
                    metric_kv_card("Variable types", kind_repr),
                ],
                className="summary-kv-grid",
            ),
        ],
        className="panel",
    )


def resolve_filter_mask(df: pd.DataFrame, filter_column: str | None, filter_values: list[str] | None, min_value: float | None, max_value: float | None, date_start: str | None, date_end: str | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if not filter_column or filter_column not in df.columns:
        return mask

    kind = infer_column_kind(df, filter_column)
    series = df[filter_column]

    if kind == "categorical":
        if filter_values:
            mask &= series.fillna("<NA>").astype(str).isin(filter_values)
    elif kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        if min_value is not None:
            mask &= numeric >= float(min_value)
        if max_value is not None:
            mask &= numeric <= float(max_value)
    elif kind == "date":
        dt = pd.to_datetime(series, errors="coerce")
        if date_start:
            mask &= dt >= pd.to_datetime(date_start)
        if date_end:
            mask &= dt <= pd.to_datetime(date_end)

    return mask


def apply_basic_filters(df: pd.DataFrame, country_selection: list[str], date_start: str | None, date_end: str | None) -> pd.DataFrame:
    out = df.copy()
    if "COUNTRY" in out.columns and country_selection:
        countries = {normalize_text(country) for country in country_selection}
        out = out[out["COUNTRY"].astype(str).str.upper().isin(countries)]
    if "COB_DATE" in out.columns:
        cob = pd.to_datetime(out["COB_DATE"], errors="coerce")
        if date_start:
            out = out[cob >= pd.to_datetime(date_start)]
        if date_end:
            out = out[cob <= pd.to_datetime(date_end)]
    return out


def build_eda_figure(df: pd.DataFrame, variable: str, plot_type: str) -> go.Figure:
    if not variable or variable not in df.columns:
        return build_empty_figure("Choisis une variable pour explorer.")

    work = df[[variable]].copy()
    kind = infer_column_kind(df, variable)

    if kind == "date":
        series = pd.to_datetime(work[variable], errors="coerce").dropna()
        if series.empty:
            return build_empty_figure("Aucune date exploitable.")
        grouped = series.dt.to_period("M").astype(str).value_counts().sort_index().reset_index()
        grouped.columns = ["period", "count"]
        if len(grouped) > 120:
            period_q = pd.to_datetime(series, errors="coerce").dt.to_period("Q").astype(str)
            grouped = period_q.value_counts().sort_index().reset_index()
            grouped.columns = ["period", "count"]
        if len(grouped) > 80:
            period_y = pd.to_datetime(series, errors="coerce").dt.to_period("Y").astype(str)
            grouped = period_y.value_counts().sort_index().reset_index()
            grouped.columns = ["period", "count"]
        fig = px.line(grouped, x="period", y="count", markers=True, title=f"Répartition temporelle de {variable}")
        fig.update_layout(template="plotly_white", xaxis_title=variable, yaxis_title="Volume")
        fig.update_xaxes(tickangle=45)
        return fig

    if kind == "numeric":
        numeric = pd.to_numeric(work[variable], errors="coerce")
        work = work.assign(_value=numeric).dropna(subset=["_value"])
        if work.empty:
            return build_empty_figure("Aucune valeur numérique exploitable.")
        if plot_type in ("auto", "histogram"):
            fig = px.histogram(work, x="_value", nbins=40, title=f"Distribution de {variable}")
            fig.update_layout(template="plotly_white", xaxis_title=variable, yaxis_title="Count")
            return fig
        if plot_type == "bar":
            bins = min(24, max(8, int(np.sqrt(len(work)))))
            bucket = pd.cut(work["_value"], bins=bins, include_lowest=True, duplicates="drop")
            b = bucket.value_counts(dropna=False).sort_index().reset_index()
            b.columns = ["bucket", "count"]
            b["bucket"] = b["bucket"].astype(str)
            fig = px.bar(b, x="bucket", y="count", title=f"Distribution (bins) de {variable}")
            fig.update_layout(template="plotly_white", xaxis_title=variable, yaxis_title="Count")
            fig.update_xaxes(tickangle=35)
            return fig
        if plot_type == "box":
            fig = px.box(work, y="_value", title=f"Boxplot de {variable}")
            fig.update_layout(template="plotly_white", yaxis_title=variable)
            return fig
        if plot_type == "violin":
            fig = px.violin(work, y="_value", box=True, points="outliers", title=f"Violin de {variable}")
            fig.update_layout(template="plotly_white", yaxis_title=variable)
            return fig
        fig = px.histogram(work, x="_value", nbins=40, title=f"Distribution de {variable}")
        fig.update_layout(template="plotly_white")
        return fig

    # categorical
    cat_series = top_categories(work[variable], limit=MAX_TOP_CATEGORIES)
    work = work.assign(_value=cat_series)
    counts = work["_value"].fillna("<NA>").astype(str).value_counts().reset_index()
    counts.columns = [variable, "count"]
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100.0).round(1)
    counts["pct_label"] = counts["pct"].astype(str) + "%"
    counts[variable] = counts[variable].map(lambda value: truncate_label(value, 15))

    if plot_type in ("auto", "bar"):
        fig = px.bar(counts, x=variable, y="count", text="pct_label", title=f"Répartition de {variable}")
        fig.update_layout(template="plotly_white", xaxis_title=variable, yaxis_title="Count")
        fig.update_xaxes(tickangle=35)
        return fig

    fig = px.bar(counts, x=variable, y="count", text="pct_label", title=f"Répartition de {variable}")
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(tickangle=35)
    return fig


def build_eda_output_table(df: pd.DataFrame, variable: str, page: int = 1, page_size: int = 12) -> tuple[html.Div, int]:
    kind = infer_column_kind(df, variable)
    if kind == "categorical":
        series = df[variable].fillna("<NA>").astype(str)
        distinct = int(series.nunique(dropna=False))
        missing = int(df[variable].isna().sum())
        total = len(df)
        counts = series.value_counts(dropna=False).reset_index()
        counts.columns = [variable, "COUNT"]
        counts["SHARE_PCT"] = (counts["COUNT"] / max(total, 1) * 100.0).round(2)
        counts[variable] = counts[variable].map(lambda value: truncate_label(value, 15))
        total_rows = len(counts)
        pages = max(1, int(np.ceil(total_rows / max(1, page_size))))
        page = max(1, min(int(page), pages))
        start = (page - 1) * page_size
        end = min(start + page_size, total_rows)
        counts_page = counts.iloc[start:end].copy()
        cards = [
            metric_kv_card("Distinct", distinct),
            metric_kv_card("Distinct (%)", format_pct(distinct / max(total, 1) * 100.0)),
            metric_kv_card("Missing", missing),
            metric_kv_card("Missing (%)", format_pct(missing / max(total, 1) * 100.0)),
        ]
        rows = [
            html.Div(
                [
                    html.Div(str(row[variable]), className="list-cell main"),
                    html.Div(f"{int(row['COUNT'])}", className="list-cell"),
                    html.Div(f"{row['SHARE_PCT']:.2f}%", className="list-cell"),
                ],
                className="summary-list-row",
            )
            for _, row in counts_page.iterrows()
        ]
        return html.Div([
            html.Div(cards, className="summary-kv-grid"),
            html.Div([html.Div(variable, className="list-head main"), html.Div("Count", className="list-head"), html.Div("Share", className="list-head")], className="summary-list-row summary-list-head"),
            html.Div(rows, className="summary-list-block"),
        ]), pages

    if kind == "numeric":
        numeric_raw = pd.to_numeric(df[variable], errors="coerce")
        numeric = numeric_raw.dropna()
        if numeric.empty:
            return html.Div("Aucune valeur numérique exploitable.", className="empty-state"), 1
        total = len(df)
        missing = int(numeric_raw.isna().sum())
        infinite = int(np.isinf(numeric_raw.fillna(0)).sum())
        zeros = int((numeric == 0).sum())
        negative = int((numeric < 0).sum())
        distinct = int(numeric.nunique(dropna=False))
        skew_value = pd.to_numeric(pd.Series([numeric.skew()]), errors="coerce").iloc[0]
        kurt_value = pd.to_numeric(pd.Series([numeric.kurt()]), errors="coerce").iloc[0]
        stats = pd.DataFrame(
            [
                {"Metric": "Distinct", "Value": distinct},
                {"Metric": "Distinct (%)", "Value": format_pct(distinct / max(total, 1) * 100.0)},
                {"Metric": "Missing", "Value": missing},
                {"Metric": "Missing (%)", "Value": format_pct(missing / max(total, 1) * 100.0)},
                {"Metric": "Infinite", "Value": infinite},
                {"Metric": "Infinite (%)", "Value": format_pct(infinite / max(total, 1) * 100.0)},
                {"Metric": "Mean", "Value": float(numeric.mean())},
                {"Metric": "Std", "Value": float(numeric.std())},
                {"Metric": "Minimum", "Value": float(numeric.min())},
                {"Metric": "Maximum", "Value": float(numeric.max())},
                {"Metric": "Median", "Value": float(numeric.median())},
                {"Metric": "Q10", "Value": float(numeric.quantile(0.10))},
                {"Metric": "Q1", "Value": float(numeric.quantile(0.25))},
                {"Metric": "Q3", "Value": float(numeric.quantile(0.75))},
                {"Metric": "Q90", "Value": float(numeric.quantile(0.90))},
                {"Metric": "Skewness", "Value": float(skew_value) if pd.notna(skew_value) else np.nan},
                {"Metric": "Kurtosis", "Value": float(kurt_value) if pd.notna(kurt_value) else np.nan},
                {"Metric": "Zeros", "Value": zeros},
                {"Metric": "Zeros (%)", "Value": format_pct(zeros / max(total, 1) * 100.0)},
                {"Metric": "Negative", "Value": negative},
                {"Metric": "Negative (%)", "Value": format_pct(negative / max(total, 1) * 100.0)},
            ]
        )
        cards = [metric_kv_card(row["Metric"], row["Value"]) for _, row in stats.iterrows()]
        return html.Div(cards, className="summary-kv-grid"), 1

    dates = pd.to_datetime(df[variable], errors="coerce").dropna()
    if dates.empty:
        return html.Div("Aucune date exploitable.", className="empty-state"), 1
    month_counts = dates.dt.to_period("M").astype(str).value_counts().sort_index().reset_index()
    month_counts.columns = ["PERIOD", "COUNT"]
    total_rows = len(month_counts)
    pages = max(1, int(np.ceil(total_rows / max(1, page_size))))
    page = max(1, min(int(page), pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total_rows)
    month_page = month_counts.iloc[start:end].copy()
    rows = [
        html.Div(
            [
                html.Div(str(row["PERIOD"]), className="list-cell main"),
                html.Div(f"{int(row['COUNT'])}", className="list-cell"),
            ],
            className="summary-list-row",
        )
        for _, row in month_page.iterrows()
    ]
    return html.Div([
        html.Div([html.Div("Period", className="list-head main"), html.Div("Count", className="list-head")], className="summary-list-row summary-list-head"),
        html.Div(rows, className="summary-list-block"),
    ]), pages


def eda_output_total_pages(df: pd.DataFrame, variable: str | None, page_size: int = 12) -> int:
    if not variable or variable not in df.columns:
        return 1

    kind = infer_column_kind(df, variable)
    if kind == "categorical":
        series = df[variable].fillna("<NA>").astype(str)
        return max(1, int(np.ceil(series.value_counts(dropna=False).shape[0] / max(1, page_size))))

    if kind == "date":
        dates = pd.to_datetime(df[variable], errors="coerce").dropna()
        if dates.empty:
            return 1
        month_counts = dates.dt.to_period("M").astype(str).value_counts().sort_index()
        return max(1, int(np.ceil(month_counts.shape[0] / max(1, page_size))))

    return 1


def build_all_variables_grid(df: pd.DataFrame, page: int = 1, page_size: int = 12) -> html.Div:
    columns = list(df.columns)
    total_cols = len(columns)
    if total_cols == 0:
        return html.Div("Aucune variable disponible.", className="empty-state")

    page = max(1, int(page))
    page_size = max(1, int(page_size))
    start = (page - 1) * page_size
    end = min(start + page_size, total_cols)
    columns_page = columns[start:end]

    cards: list[html.Div] = []
    for column in columns_page:
        kind = infer_column_kind(df, column)
        fig: go.Figure
        if kind == "numeric":
            vals = pd.to_numeric(df[column], errors="coerce").dropna()
            if vals.empty:
                continue
            fig = px.histogram(vals.to_frame(name="value"), x="value", nbins=30, title=truncate_label(column, 40))
        elif kind == "date":
            dt = pd.to_datetime(df[column], errors="coerce").dropna()
            if dt.empty:
                continue
            s = dt.dt.to_period("M").astype(str).value_counts().sort_index().reset_index()
            s.columns = ["period", "count"]
            if len(s) > 120:
                sq = dt.dt.to_period("Q").astype(str).value_counts().sort_index().reset_index()
                sq.columns = ["period", "count"]
                s = sq
            if len(s) > 80:
                sy = dt.dt.to_period("Y").astype(str).value_counts().sort_index().reset_index()
                sy.columns = ["period", "count"]
                s = sy
            fig = px.line(s, x="period", y="count", markers=True, title=truncate_label(column, 40))
        else:
            series = df[column].fillna("<NA>").astype(str)
            counts = series.value_counts().head(20).reset_index()
            counts.columns = ["category", "count"]
            counts["category"] = counts["category"].map(lambda value: truncate_label(value, 15))
            counts["pct"] = (counts["count"] / counts["count"].sum() * 100.0).round(1).astype(str) + "%"
            fig = px.bar(counts, x="category", y="count", text="pct", title=truncate_label(column, 40))
        fig.update_layout(template="plotly_white", height=280, margin={"l": 20, "r": 20, "t": 40, "b": 20}, showlegend=False)
        cards.append(html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], className="control-card"))
    header = html.Div(f"Variables affichées: {start + 1} à {end} sur {total_cols}", className="hint-text")
    return html.Div([header, html.Div(cards, className="advanced-filter-grid")])


def build_advanced_rule_card(row_index: int, column_options: list[dict[str, str]]) -> html.Div:
    return html.Div(
        [
            html.Label(f"Règle {row_index}"),
            dcc.Dropdown(id={"type": "adv-column", "row": row_index}, options=column_options, value=None, placeholder="Colonne", persistence=True, persistence_type="memory"),
            dcc.Dropdown(id={"type": "adv-operator", "row": row_index}, options=[], value=None, placeholder="Opérateur", persistence=True, persistence_type="memory"),
            html.Div(id={"type": "adv-value-container", "row": row_index}),
        ],
        className="control-card advanced-rule-card",
    )


def build_compare_figure(df: pd.DataFrame, x_var: str, y_var: str, hue: str | None, plot_type: str) -> go.Figure:
    if not x_var or not y_var or x_var not in df.columns or y_var not in df.columns:
        return build_empty_figure("Choisis au moins deux variables.")

    work = df[[x_var, y_var] + ([hue] if hue and hue in df.columns and hue not in {x_var, y_var} else [])].copy()
    x_kind = infer_column_kind(df, x_var)
    y_kind = infer_column_kind(df, y_var)

    if x_kind == "date" and x_var.upper() not in COB_TEMPORAL_COLUMNS:
        work[x_var] = coerce_temporal_series(work, x_var)
    if y_kind == "date" and y_var.upper() not in COB_TEMPORAL_COLUMNS:
        work[y_var] = coerce_temporal_series(work, y_var)

    if x_kind == "numeric" and y_kind == "numeric":
        work = work.dropna(subset=[x_var, y_var])
        work = sample_dataframe(work)
        if plot_type in ("auto", "scatter"):
            fig = px.scatter(work, x=x_var, y=y_var, color=hue if hue in work.columns else None, opacity=0.55, title=f"{x_var} vs {y_var}")
            fig.update_layout(template="plotly_white")
            return fig
        fig = px.density_heatmap(work, x=x_var, y=y_var, nbinsx=30, nbinsy=30, title=f"Densité {x_var} x {y_var}")
        fig.update_layout(template="plotly_white")
        return fig

    if x_kind == "categorical" and y_kind == "categorical":
        work[x_var] = top_categories(work[x_var])
        work[y_var] = top_categories(work[y_var])
        ctab = pd.crosstab(work[x_var].fillna("<NA>").astype(str), work[y_var].fillna("<NA>").astype(str))
        if ctab.shape[0] > MAX_HEATMAP_LEVELS:
            ctab = ctab.iloc[:MAX_HEATMAP_LEVELS, :]
        if ctab.shape[1] > MAX_HEATMAP_LEVELS:
            ctab = ctab.iloc[:, :MAX_HEATMAP_LEVELS]
        if plot_type == "stacked":
            id_col = ctab.index.name or x_var
            ctab_long = ctab.reset_index().melt(id_vars=[id_col], var_name=y_var, value_name="count")
            fig = px.bar(ctab_long, x=id_col, y="count", color=y_var, barmode="stack", title=f"{x_var} x {y_var}")
            fig.update_layout(template="plotly_white")
            fig.update_xaxes(tickangle=35)
            return fig
        fig = px.imshow(ctab, text_auto=True, aspect="auto", title=f"{x_var} x {y_var}")
        fig.update_layout(template="plotly_white")
        return fig

    if x_kind == "date" or y_kind == "date":
        date_col = x_var if x_kind == "date" else y_var
        value_col = y_var if date_col == x_var else x_var
        value_kind = infer_column_kind(df, value_col)
        work = work.dropna(subset=[date_col])
        work["period"] = temporal_period_labels(work, date_col).astype(str)

        if value_kind == "categorical":
            cats = work[value_col].fillna("<NA>").astype(str)
            cat_counts = cats.value_counts()
            if len(cat_counts) > MAX_CATEGORY_TREND:
                top_cats = set(cat_counts.head(MAX_CATEGORY_TREND).index.tolist())
                cats = cats.where(cats.isin(top_cats), "OTHER")
                work[value_col] = cats
            trend = (
                work.groupby(["period", value_col])
                .size()
                .reset_index(name="count")
                .sort_values("period")
            )
            if plot_type in ("auto", "grouped-line"):
                total_per_period = trend.groupby("period")["count"].sum().rename("period_total")
                trend = trend.merge(total_per_period, on="period", how="left")
                trend["share_pct"] = np.where(
                    trend["period_total"] > 0,
                    trend["count"] / trend["period_total"] * 100.0,
                    0.0,
                )

                fig = go.Figure()
                categories = trend[value_col].dropna().astype(str).unique().tolist()
                for category in categories:
                    sub = trend[trend[value_col].astype(str) == str(category)].sort_values("period")
                    fig.add_bar(
                        x=sub["period"],
                        y=sub["count"],
                        name=f"{category} volume",
                        yaxis="y",
                    )
                    fig.add_scatter(
                        x=sub["period"],
                        y=sub["share_pct"],
                        mode="lines+markers",
                        name=f"{category} %",
                        yaxis="y2",
                    )

                fig.update_layout(
                    template="plotly_white",
                    title=f"Évolution de {value_col} (volume + part)",
                    xaxis_title="Période",
                    yaxis=dict(title="Volume"),
                    yaxis2=dict(title="Part (%)", overlaying="y", side="right"),
                    barmode="group",
                )
                fig.update_xaxes(tickangle=45)
                return fig
            if plot_type in ("line", "line-category"):
                fig = px.line(trend, x="period", y="count", color=value_col, markers=True, title=f"Évolution par catégorie de {value_col}")
                fig.update_layout(template="plotly_white")
                fig.update_xaxes(tickangle=45)
                return fig
            if plot_type == "stacked-area":
                fig = px.area(trend, x="period", y="count", color=value_col, title=f"Évolution empilée de {value_col}")
                fig.update_layout(template="plotly_white")
                fig.update_xaxes(tickangle=45)
                return fig
            if plot_type == "grouped":
                fig = px.bar(trend, x="period", y="count", color=value_col, barmode="group", title=f"Volumes par catégorie de {value_col}")
                fig.update_layout(template="plotly_white")
                fig.update_xaxes(tickangle=45)
                return fig
            fig = px.bar(trend, x="period", y="count", color=value_col, barmode="stack", title=f"Volumes par catégorie de {value_col}")
            fig.update_layout(template="plotly_white")
            fig.update_xaxes(tickangle=45)
            return fig

        if value_kind == "numeric":
            agg = work.groupby("period")[value_col].mean().reset_index()
            if plot_type in ("scatter",):
                fig = px.scatter(agg, x="period", y=value_col, title=f"Évolution de {value_col}")
            else:
                fig = px.line(agg, x="period", y=value_col, markers=True, title=f"Évolution de {value_col}")
        else:
            agg = work.groupby("period").size().reset_index(name="count")
            fig = px.line(agg, x="period", y="count", markers=True, title=f"Volumes dans le temps")
        fig.update_layout(template="plotly_white")
        fig.update_xaxes(tickangle=45)
        return fig

    if x_kind == "numeric" and y_kind == "categorical":
        work = work.dropna(subset=[x_var, y_var])
        work[y_var] = top_categories(work[y_var])
        if plot_type == "violin":
            fig = px.violin(work, x=y_var, y=x_var, color=hue if hue in work.columns else None, box=True, title=f"{x_var} par {y_var}")
        elif plot_type == "bar":
            avg_df = work.groupby(y_var)[x_var].mean().reset_index(name="value")
            fig = px.bar(avg_df, x=y_var, y="value", title=f"Moyenne de {x_var} par {y_var}")
        else:
            fig = px.box(work, x=y_var, y=x_var, color=hue if hue in work.columns else None, title=f"{x_var} par {y_var}")
        fig.update_layout(template="plotly_white")
        fig.update_xaxes(tickangle=30)
        return fig

    if x_kind == "categorical" and y_kind == "numeric":
        work = work.dropna(subset=[x_var, y_var])
        work[x_var] = top_categories(work[x_var])
        if plot_type == "violin":
            fig = px.violin(work, x=x_var, y=y_var, color=hue if hue in work.columns else None, box=True, title=f"{y_var} par {x_var}")
        elif plot_type == "bar":
            avg_df = work.groupby(x_var)[y_var].mean().reset_index(name="value")
            fig = px.bar(avg_df, x=x_var, y="value", title=f"Moyenne de {y_var} par {x_var}")
        else:
            fig = px.box(work, x=x_var, y=y_var, color=hue if hue in work.columns else None, title=f"{y_var} par {x_var}")
        fig.update_layout(template="plotly_white")
        fig.update_xaxes(tickangle=30)
        return fig

    return build_empty_figure("Combinaison de variables non gérée.")


def build_kpi_table(
    df: pd.DataFrame,
    grain: str,
    date_col: str,
    group_cols: list[str],
    value_col: str | None,
    agg_func: str,
    pivot_rows: str | None,
    pivot_cols: str | None,
    chart_style: str,
) -> tuple[pd.DataFrame, go.Figure]:
    work = df.copy()
    if date_col not in work.columns:
        raise ValueError("Colonne date invalide.")

    work = work.dropna(subset=[date_col])
    work["PERIOD"] = month_grain_series(work[date_col], grain)

    resolved_groups = [column for column in group_cols if column in work.columns and column not in {date_col, "PERIOD"}]
    display_work = work.copy()
    display_group_col: str | None = None
    if resolved_groups:
        normalized_group_cols: list[str] = []
        for column in resolved_groups:
            kind = infer_column_kind(display_work, column)
            if kind == "date":
                display_work[column] = month_grain_series(display_work[column], grain)
            else:
                display_work[column] = display_work[column].fillna("<NA>").astype(str)
            if kind == "categorical":
                display_work[column] = top_categories(display_work[column], limit=MAX_CATEGORY_TREND)
            normalized_group_cols.append(column)

        if len(normalized_group_cols) == 1:
            display_group_col = normalized_group_cols[0]
        else:
            display_work["GROUP_LABEL"] = display_work[normalized_group_cols].astype(str).agg(" | ".join, axis=1)
            display_work["GROUP_LABEL"] = top_categories(display_work["GROUP_LABEL"], limit=MAX_CATEGORY_TREND)
            display_group_col = "GROUP_LABEL"

    group_keys = ["PERIOD"] + ([display_group_col] if display_group_col else [])

    if agg_func == "count" or not value_col or value_col not in display_work.columns:
        grouped = display_work.groupby(group_keys).size().reset_index(name="VALUE")
    else:
        numeric_metric = pd.to_numeric(display_work[value_col], errors="coerce")
        display_work = display_work.assign(_metric=numeric_metric)
        agg_map = {"sum": "sum", "mean": "mean", "median": "median", "min": "min", "max": "max"}
        grouped = display_work.groupby(group_keys)["_metric"].agg(agg_map.get(agg_func, "sum")).reset_index(name="VALUE")

    pivot_df = grouped.copy()
    if pivot_rows and pivot_rows in pivot_df.columns and pivot_cols and pivot_cols in pivot_df.columns:
        pivot_df = pivot_df.pivot_table(index=pivot_rows, columns=pivot_cols, values="VALUE", aggfunc="sum", fill_value=0).reset_index()

    chart_df = grouped.copy()
    if grain == "monthly" and chart_df["PERIOD"].nunique() > MAX_KPI_CHART_POINTS:
        last_buckets = sorted(chart_df["PERIOD"].dropna().unique().tolist())[-MAX_KPI_CHART_POINTS:]
        chart_df = chart_df[chart_df["PERIOD"].isin(last_buckets)]

    chart = go.Figure()
    if not display_group_col:
        if chart_style == "bar":
            chart = px.bar(chart_df, x="PERIOD", y="VALUE", title="KPI time series")
        elif chart_style == "area":
            chart = px.area(chart_df, x="PERIOD", y="VALUE", title="KPI time series")
        elif chart_style == "spline":
            chart = px.line(chart_df, x="PERIOD", y="VALUE", markers=False, title="KPI time series")
            chart.update_traces(line_shape="spline")
        elif chart_style == "line-markers":
            chart = px.line(chart_df, x="PERIOD", y="VALUE", markers=True, title="KPI time series")
        else:
            chart = px.line(chart_df, x="PERIOD", y="VALUE", markers=True, title="KPI time series")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    trend_df = chart_df.copy()
    trend_df[display_group_col] = trend_df[display_group_col].fillna("<NA>").astype(str)

    if chart_style == "line":
        chart = px.line(trend_df, x="PERIOD", y="VALUE", color=display_group_col, markers=True, title="KPI grouped")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    if chart_style == "spline":
        chart = px.line(trend_df, x="PERIOD", y="VALUE", color=display_group_col, markers=False, title="KPI grouped")
        chart.update_traces(line_shape="spline")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    if chart_style == "line-markers":
        chart = px.line(trend_df, x="PERIOD", y="VALUE", color=display_group_col, markers=True, title="KPI grouped")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    if chart_style == "area":
        chart = px.area(trend_df, x="PERIOD", y="VALUE", color=display_group_col, title="KPI grouped")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    if chart_style == "stacked-area":
        chart = px.area(trend_df, x="PERIOD", y="VALUE", color=display_group_col, title="KPI grouped")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    if chart_style == "bar":
        chart = px.bar(trend_df, x="PERIOD", y="VALUE", color=display_group_col, barmode="group", title="KPI grouped")
        chart.update_layout(template="plotly_white")
        chart.update_xaxes(tickangle=45)
        return pivot_df, chart

    trend_df = trend_df.groupby(["PERIOD", display_group_col], as_index=False)["VALUE"].sum()
    totals = trend_df.groupby("PERIOD")["VALUE"].transform("sum")
    trend_df["SHARE"] = np.where(totals > 0, trend_df["VALUE"] / totals * 100, 0)

    chart = go.Figure()
    for category in trend_df[display_group_col].dropna().astype(str).unique().tolist():
        subset = trend_df[trend_df[display_group_col].astype(str) == str(category)].sort_values("PERIOD")
        chart.add_bar(x=subset["PERIOD"], y=subset["VALUE"], name=f"{category} volume")
        chart.add_scatter(x=subset["PERIOD"], y=subset["SHARE"], mode="lines+markers", name=f"{category} %", yaxis="y2")
    chart.update_layout(
        template="plotly_white",
        barmode="group",
        title="KPI grouped",
        yaxis2={"title": "Share %", "overlaying": "y", "side": "right"},
    )
    chart.update_xaxes(tickangle=45)
    return pivot_df, chart


catalog = scan_catalog()
country_options = available_country_options(catalog)
available_load_columns = discover_available_load_columns(catalog)
load_column_options = [{"label": "ALL", "value": "__ALL__"}] + [{"label": column, "value": column} for column in available_load_columns]
default_load_columns = [column for column in DEFAULT_SELECTED_LOAD_COLUMNS if column in available_load_columns]
if not default_load_columns:
    default_load_columns = available_load_columns.copy()
min_date, max_date = available_date_bounds(catalog)
year_options = []
start_year_default = None
end_year_default = None
start_month_default = 1
end_month_default = 12
if min_date is not None and max_date is not None:
    year_values = list(range(int(min_date.year), int(max_date.year) + 1))
    year_options = [{"label": str(year), "value": year} for year in year_values]
    start_year_default = int(min_date.year)
    end_year_default = int(max_date.year)
    start_month_default = int(min_date.month)
    end_month_default = int(max_date.month)

app.layout = html.Div([
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Pays"),
                                dcc.Dropdown(
                                    id="country-selector",
                                    options=country_options,
                                    value=DEFAULT_COUNTRIES,
                                    multi=True,
                                    placeholder="Sélectionne un ou plusieurs pays",
                                ),
                                html.Div(
                                    [
                                        html.Label("Colonnes à lire"),
                                        dcc.Dropdown(
                                            id="load-columns-selector",
                                            options=load_column_options,
                                            value=default_load_columns,
                                            multi=True,
                                            placeholder="Choisis les colonnes à charger",
                                        ),
                                    ],
                                    style={"marginTop": "8px"},
                                ),
                            ],
                            className="control-card",
                        ),
                        html.Div(
                            [
                                html.Label("Echantillon par dataset (%)"),
                                dcc.Input(
                                    id="load-sample-pct",
                                    type="number",
                                    min=0,
                                    max=100,
                                    step="any",
                                    value=DEFAULT_LOAD_SAMPLE_PCT,
                                    className="advanced-input sample-pct-input",
                                ),
                                html.Div("100 = lecture complète. Tu peux mettre une très petite valeur.", className="filter-helper"),
                                html.Button("Charger les données", id="load-data-button", n_clicks=0, className="primary-button"),
                                html.Div(id="load-status", className="load-status"),
                            ],
                            className="control-card load-card",
                        ),
                    ],
                    className="top-load-grid",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Période de lecture (mois/année)"),
                                build_month_year_range_block(
                                    start_year_id="load-start-year",
                                    start_month_id="load-start-month",
                                    end_year_id="load-end-year",
                                    end_month_id="load-end-month",
                                    year_options=year_options,
                                    start_year_value=start_year_default,
                                    start_month_value=start_month_default,
                                    end_year_value=end_year_default,
                                    end_month_value=end_month_default,
                                ),
                            ],
                            className="control-card period-card",
                        ),
                    ],
                    className="period-row",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Panneau de filtres avancés", className="panel-title"),
                                html.Div(
                                    "Construis des règles combinables en AND / OR. Les filtres s'appliquent après le chargement pays / dates.",
                                    className="small-note",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Logique"),
                                        dcc.Dropdown(
                                            id="adv-filter-logic",
                                            options=[{"label": option, "value": option} for option in ADV_FILTER_LOGIC_OPTIONS],
                                            value="AND",
                                            clearable=False,
                                        ),
                                    ],
                                    className="control-card",
                                ),
                                html.Div(id="adv-filter-summary", className="control-card filter-summary-box"),
                            ],
                            className="controls-grid",
                        ),
                        html.Div(
                            [
                                html.Button("Ajouter une règle", id="adv-add-rule-button", n_clicks=0, className="secondary-button"),
                            ],
                            className="action-row add-rule-row",
                        ),
                        html.Div(id="adv-rules-container", className="advanced-filter-grid"),
                    ],
                    className="panel advanced-panel",
                ),
                dcc.Store(id="adv-rule-count", data=ADV_FILTER_ROWS),
                dcc.Store(id="advanced-filter-store"),
                dcc.Store(id="dataset-key"),
                dcc.Store(id="dataset-meta"),
                html.Div(id="summary-cards", className="summary-container"),
                dcc.Tabs(
                    id="main-tabs",
                    value="tab-eda",
                    children=[
                        dcc.Tab(label="Exploration", value="tab-eda", className="app-tab", selected_className="app-tab--selected"),
                        dcc.Tab(label="Comparer", value="tab-compare", className="app-tab", selected_className="app-tab--selected"),
                        dcc.Tab(label="Builder KPI", value="tab-kpi", className="app-tab", selected_className="app-tab--selected"),
                        dcc.Tab(label="Insights", value="tab-interactions", className="app-tab", selected_className="app-tab--selected"),
                        dcc.Tab(label="Prévisualisation", value="tab-preview", className="app-tab", selected_className="app-tab--selected"),
                    ],
                ),
                html.Div(id="tab-content", className="tab-content"),
            ],
            className="app-shell",
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("dataset-key", "data"),
    Output("dataset-meta", "data"),
    Output("load-status", "children"),
    Input("load-data-button", "n_clicks"),
    State("country-selector", "value"),
    State("load-start-year", "value"),
    State("load-start-month", "value"),
    State("load-end-year", "value"),
    State("load-end-month", "value"),
    State("load-columns-selector", "value"),
    State("load-sample-pct", "value"),
    prevent_initial_call=True,
)
def load_dataset_callback(
    n_clicks: int,
    countries: list[str],
    start_year: int | None,
    start_month: int | None,
    end_year: int | None,
    end_month: int | None,
    selected_columns: list[str] | None,
    sample_pct: float | None,
) -> tuple[str, dict[str, Any], str]:
    if not countries:
        raise PreventUpdate

    start_ts = month_year_to_start(start_year, start_month)
    end_ts = month_year_to_end(end_year, end_month)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        return "", {}, "Période invalide: début > fin"

    start_date = start_ts.date().isoformat() if start_ts is not None else None
    end_date = end_ts.date().isoformat() if end_ts is not None else None

    selected_columns_clean = [str(column) for column in (selected_columns or []) if str(column).strip()]
    if "__ALL__" in selected_columns_clean:
        selected_columns_clean = available_load_columns.copy()
    if not selected_columns_clean:
        return "", {}, "Sélectionne au moins une colonne à charger"

    resolved_sample_pct = float(sample_pct if sample_pct is not None else DEFAULT_LOAD_SAMPLE_PCT)
    resolved_sample_pct = min(100.0, max(0.0, resolved_sample_pct))

    df = load_selected_data(
        countries,
        start_date,
        end_date,
        sample_pct=resolved_sample_pct,
        selected_columns=selected_columns_clean,
    )
    cache_key = str(uuid.uuid4())
    DATA_CACHE[cache_key] = df

    meta = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "countries": sorted(df["COUNTRY"].astype(str).str.upper().unique().tolist()) if "COUNTRY" in df.columns else [],
        "start_date": str(pd.to_datetime(df["COB_DATE"], errors="coerce").min()) if "COB_DATE" in df.columns else None,
        "end_date": str(pd.to_datetime(df["COB_DATE"], errors="coerce").max()) if "COB_DATE" in df.columns else None,
    }

    status = f"Données chargées: {meta['rows']:,} lignes | colonnes: {len(meta['columns'])} | échantillon: {resolved_sample_pct:.6f}%"
    status = status.replace(",", " ").replace(".000%", "%")
    return cache_key, meta, status


@app.callback(
    Output("summary-cards", "children"),
    Input("dataset-key", "data"),
    Input("advanced-filter-store", "data"),
)
def update_summary_cards(dataset_key: str | None, advanced_filter_store: dict[str, Any] | None) -> html.Div:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return html.Div("Charge les données pour voir le résumé.", className="empty-state")
    filtered = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    return build_dataset_summary(filtered)


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("dataset-key", "data"),
)
def render_tab(tab_value: str, dataset_key: str | None) -> html.Div:
    loaded = bool(dataset_key) and dataset_key in DATA_CACHE
    if loaded and dataset_key is not None:
        df = DATA_CACHE[dataset_key]
    else:
        df = pd.DataFrame()
    column_options = get_column_choices(df) if loaded else []
    categorical_columns = [column for column in df.columns if infer_column_kind(df, column) == "categorical"] if loaded else []
    numeric_columns = [column for column in df.columns if infer_column_kind(df, column) == "numeric"] if loaded else []
    date_columns = [column for column in df.columns if infer_column_kind(df, column) == "date"] if loaded else []

    if not loaded:
        return html.Div("Charge d’abord un sous-ensemble de données pour activer les analyses.", className="empty-state")

    x_default = None
    y_default = None
    x_kind = "unknown"
    y_kind = "unknown"
    compare_options = compare_plot_options(x_kind, y_kind)
    date_default = date_columns[0] if date_columns else None
    numeric_value_options = [{"label": "Aucune (count)", "value": ""}] + [{"label": col, "value": col} for col in numeric_columns]

    if tab_value == "tab-eda":
        eda_options = [{"label": "ALL (toutes les variables)", "value": "__ALL__"}] + column_options
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Variable à explorer"),
                                dcc.Dropdown(id="eda-variable", options=eda_options, value=None, placeholder="Choisis une variable"),
                            ],
                            className="control-card",
                        ),
                        html.Div(
                            [
                                html.Label("Type de graphique"),
                                dcc.Dropdown(id="eda-plot-type", options=[{"label": "Auto", "value": "auto"}], value="auto"),
                            ],
                            className="control-card",
                        ),
                    ],
                    className="controls-grid",
                ),
                html.Div(id="eda-filter-panel"),
                html.Div(dcc.Graph(id="eda-graph", figure=build_empty_figure("Sélectionne une variable.")), className="graph-wrap"),
                html.Div(id="eda-all-grid"),
                html.Div(
                    [
                        html.Button("←", id="eda-all-prev", n_clicks=0, className="page-nav-button"),
                        html.Div(id="eda-all-page-label", className="hint-text"),
                        html.Button("→", id="eda-all-next", n_clicks=0, className="page-nav-button"),
                    ],
                    id="eda-all-pager",
                    className="action-row",
                    style={"justifyContent": "center", "gap": "12px", "marginTop": "8px", "marginBottom": "12px"},
                ),
                html.Div(id="eda-unique-values", style={"display": "none"}),
                html.Div(
                    [
                        html.Button("←", id="eda-output-prev", n_clicks=0, className="page-nav-button"),
                        html.Div(id="eda-output-page-label", className="hint-text"),
                        html.Button("→", id="eda-output-next", n_clicks=0, className="page-nav-button"),
                    ],
                    id="eda-output-pager",
                    className="action-row",
                ),
                html.Div(id="eda-output-table"),
                dcc.Store(id="eda-all-page-store", data=1),
                dcc.Store(id="eda-all-total-pages", data=1),
                dcc.Store(id="eda-output-page-store", data=1),
                dcc.Store(id="eda-output-total-pages", data=1),
            ]
        )

    if tab_value == "tab-compare":
        return html.Div(
            [
                html.Div(
                    [
                        html.Div([html.Label("X"), dcc.Dropdown(id="compare-x", options=column_options, value=x_default)], className="control-card"),
                        html.Div([html.Label("Y"), dcc.Dropdown(id="compare-y", options=column_options, value=y_default)], className="control-card"),
                        html.Div([html.Label("Hue"), dcc.Dropdown(id="compare-hue", options=[{"label": "Aucun", "value": ""}] + column_options, value="", clearable=False)], className="control-card"),
                        html.Div([html.Label("Type de plot"), dcc.Dropdown(id="compare-plot-type", options=compare_options, value=compare_options[0]["value"] if compare_options else "auto")], className="control-card"),
                    ],
                    className="controls-grid",
                ),
                html.Div(id="compare-filter-panel"),
                html.Div(dcc.Graph(id="compare-graph", figure=build_empty_figure("Choisis X et Y.")), className="graph-wrap"),
            ]
        )

    if tab_value == "tab-kpi":
        return html.Div(
            [
                html.Div(
                    [
                        html.Div([html.Label("Temporalité"), dcc.Dropdown(id="kpi-grain", options=[{"label": grain.title(), "value": grain} for grain in TIME_GRAINS], value="monthly")], className="control-card"),
                        html.Div([html.Label("Date de référence (mois/année)"), dcc.Dropdown(id="kpi-date-col", options=[{"label": col, "value": col} for col in date_columns], value=date_default)], className="control-card"),
                        html.Div([html.Label("Group by"), dcc.Dropdown(id="kpi-group-cols", options=column_options, multi=True, value=["COUNTRY"] if "COUNTRY" in df.columns else [])], className="control-card"),
                        html.Div([html.Label("Mesure numérique (optionnel)"), dcc.Dropdown(id="kpi-value-col", options=numeric_value_options, value="")], className="control-card"),
                        html.Div([html.Label("Agrégation"), dcc.Dropdown(id="kpi-agg-func", options=[{"label": value.title(), "value": value} for value in AGG_FUNCS], value="count")], className="control-card"),
                        html.Div([html.Label("Style graphe"), dcc.Dropdown(id="kpi-chart-style", options=[{"label": "Auto", "value": "auto"}, {"label": "Barres groupées", "value": "bar"}, {"label": "Barres + lignes (%)", "value": "grouped-line"}, {"label": "Ligne seule", "value": "line"}, {"label": "Ligne douce", "value": "spline"}, {"label": "Ligne avec points", "value": "line-markers"}, {"label": "Aire", "value": "area"}, {"label": "Aire empilée", "value": "stacked-area"}], value="auto")], className="control-card"),
                    ],
                    className="controls-grid",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Période KPI (mois/année)"),
                                build_month_year_range_block(
                                    start_year_id="kpi-start-year",
                                    start_month_id="kpi-start-month",
                                    end_year_id="kpi-end-year",
                                    end_month_id="kpi-end-month",
                                    year_options=[],
                                    start_year_value=None,
                                    start_month_value=1,
                                    end_year_value=None,
                                    end_month_value=12,
                                ),
                            ],
                            className="control-card",
                        ),
                    ],
                    className="controls-grid",
                ),
                html.Div(
                    [
                        html.Div([html.Label("Pivot rows"), dcc.Dropdown(id="kpi-pivot-rows", options=[{"label": "Aucun", "value": ""}], value="", clearable=False)], className="control-card"),
                        html.Div([html.Label("Pivot columns"), dcc.Dropdown(id="kpi-pivot-cols", options=[{"label": "Aucun", "value": ""}], value="", clearable=False)], className="control-card"),
                    ],
                    className="controls-grid",
                ),
                html.Div(id="kpi-filter-panel"),
                html.Div(
                    [
                        html.Button("Construire KPI", id="run-kpi-button", n_clicks=0, className="primary-button"),
                        html.Div(id="kpi-status", className="load-status kpi-status"),
                    ],
                    className="action-row kpi-action-row",
                ),
                html.Div(dcc.Graph(id="kpi-graph", figure=build_empty_figure("Construis ton KPI.")), className="graph-wrap"),
                html.Div(id="kpi-table-container"),
            ]
        )

    if tab_value == "tab-interactions":
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Variable cible"),
                                dcc.Dropdown(
                                    id="interaction-target",
                                    options=column_options,
                                    value=None,
                                    placeholder="Choisis une variable cible",
                                ),
                            ],
                            className="control-card compact-card",
                        ),
                        html.Div(
                            [
                                html.Label("Variables pour corrélation"),
                                dcc.Dropdown(
                                    id="interaction-corr-vars",
                                    options=[{"label": "ALL", "value": "__ALL__"}] + column_options,
                                    value=None,
                                    multi=True,
                                    placeholder="ALL par défaut si vide",
                                ),
                                html.Div("ALL: matrice globale. 1 variable: focus cible vs autres. Plusieurs: mini-matrice sur la sélection.", className="filter-helper"),
                            ],
                            className="control-card compact-card",
                        ),
                    ],
                    className="controls-grid insights-controls-grid",
                ),
                html.Div(
                    [
                        html.Div(dcc.Graph(id="interaction-main-graph", figure=build_empty_figure("Choisis une variable cible.")), className="graph-wrap"),
                        html.Div(dcc.Graph(id="interaction-heatmap", figure=build_empty_figure("Matrice de corrélation encodée.")), className="graph-wrap"),
                    ],
                    className="interaction-grid",
                ),
                html.Div(
                    [
                        html.Div(id="interaction-ranking-table"),
                    ],
                    className="panel",
                ),
            ]
        )

    return html.Div(
        [
            html.H3("Prévisualisation"),
            html.P("Vue brute des premières lignes après filtrage."),
            html.Div(id="preview-table-container"),
        ]
    )


@app.callback(
    Output("eda-filter-panel", "children"),
    Input("eda-variable", "value"),
    Input("advanced-filter-store", "data"),
    Input("dataset-key", "data"),
)
def render_eda_filter_panel(variable: str | None, advanced_filter_store: dict[str, Any] | None, dataset_key: str | None) -> html.Div:
    if not dataset_key or dataset_key not in DATA_CACHE or not variable:
        return html.Div()

    if variable == "__ALL__":
        return html.Div("Mode ALL: pas de filtre local unique. Les règles avancées du haut sont appliquées.", className="hint-text")

    df = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    if variable not in df.columns:
        return html.Div()

    kind = infer_column_kind(df, variable)
    if kind == "categorical":
        series = df[variable].fillna("<NA>").astype(str)
        counts = safe_top_value_counts(series, MAX_CATEGORICAL_VALUES_FOR_SELECTOR)
        unique_estimate = int(series.nunique(dropna=False))
        options = [{"label": f"{truncate_label(idx, 15)} ({count})", "value": idx} for idx, count in counts.items()]
        return html.Div(
            [
                html.Label("Filtre sur les valeurs de la variable"),
                dcc.Checklist(
                    id={"type": "eda-filter-all", "slot": "main"},
                    options=[{"label": "All", "value": "all"}],
                    value=[],
                    inline=True,
                ),
                dcc.Dropdown(id={"type": "eda-filter-values", "slot": "main"}, options=options, value=[], multi=True, placeholder="Sélectionne une ou plusieurs valeurs"),
                html.Div(
                    f"Cardinalité: {unique_estimate:,} valeurs distinctes. Affichage limité aux modalités les plus fréquentes.".replace(",", " "),
                    className="hint-text",
                ),
            ],
            className="control-card",
        )

    if kind == "numeric":
        numeric = pd.to_numeric(df[variable], errors="coerce").dropna()
        if numeric.empty:
            return html.Div()
        min_v = float(numeric.min())
        max_v = float(numeric.max())
        return html.Div(
            [
                html.Label("Filtre numérique"),
                dcc.RangeSlider(id={"type": "eda-filter-range", "slot": "main"}, min=min_v, max=max_v, value=[min_v, max_v], tooltip={"placement": "bottom", "always_visible": False}),
            ],
            className="control-card",
        )

    if kind == "date":
        date_series = pd.to_datetime(df[variable], errors="coerce").dropna()
        if date_series.empty:
            return html.Div()
        year_values = list(range(int(date_series.dt.year.min()), int(date_series.dt.year.max()) + 1))
        year_opts = [{"label": str(year), "value": year} for year in year_values]
        return html.Div(
            [
                html.Label("Filtre date"),
                build_month_year_range_block(
                    start_year_id={"type": "eda-filter-date-start-year", "slot": "main"},
                    start_month_id={"type": "eda-filter-date-start-month", "slot": "main"},
                    end_year_id={"type": "eda-filter-date-end-year", "slot": "main"},
                    end_month_id={"type": "eda-filter-date-end-month", "slot": "main"},
                    year_options=year_opts,
                    start_year_value=year_values[0],
                    start_month_value=int(date_series.dt.month.min()),
                    end_year_value=year_values[-1],
                    end_month_value=int(date_series.dt.month.max()),
                ),
            ],
            className="control-card",
        )

    return html.Div()


@app.callback(
    Output("adv-rule-count", "data"),
    Input("adv-add-rule-button", "n_clicks"),
    State("adv-rule-count", "data"),
    prevent_initial_call=True,
)
def add_advanced_rule(n_clicks: int, current_count: int | None) -> int:
    current = current_count or ADV_FILTER_ROWS
    if n_clicks <= 0:
        return current
    return min(MAX_ADV_FILTER_ROWS, current + 1)


@app.callback(
    Output("adv-rules-container", "children"),
    Input("adv-rule-count", "data"),
    Input("dataset-key", "data"),
)
def render_advanced_rules(rule_count: int | None, dataset_key: str | None) -> list[html.Div]:
    count = max(ADV_FILTER_ROWS, int(rule_count or ADV_FILTER_ROWS))
    options: list[dict[str, str]] = []
    if dataset_key and dataset_key in DATA_CACHE:
        options = get_column_choices(DATA_CACHE[dataset_key])
    return [build_advanced_rule_card(i, options) for i in range(1, count + 1)]


@app.callback(
    Output({"type": "adv-operator", "row": MATCH}, "options"),
    Output({"type": "adv-operator", "row": MATCH}, "value"),
    Input({"type": "adv-column", "row": MATCH}, "value"),
    Input("dataset-key", "data"),
)
def update_advanced_operator_for_row(column: str | None, dataset_key: str | None) -> tuple[list[dict[str, str]], str | None]:
    if not dataset_key or dataset_key not in DATA_CACHE or not column:
        return [], None
    df = DATA_CACHE[dataset_key]
    kind = infer_column_kind(df, column)
    options = advanced_operator_options(kind)
    return options, None


@app.callback(
    Output({"type": "adv-value-container", "row": MATCH}, "children"),
    Input({"type": "adv-column", "row": MATCH}, "value"),
    Input({"type": "adv-operator", "row": MATCH}, "value"),
    Input({"type": "adv-column", "row": MATCH}, "id"),
    Input("dataset-key", "data"),
)
def update_advanced_value_for_row(column: str | None, operator: str | None, row_id: dict[str, Any], dataset_key: str | None) -> html.Div:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return html.Div()
    row_index = int(row_id.get("row", 0))
    return build_advanced_value_widget(row_index, DATA_CACHE[dataset_key], column, operator)


@app.callback(
    Output({"type": "adv-cat-values", "row": MATCH}, "value"),
    Input({"type": "adv-cat-all", "row": MATCH}, "value"),
    State({"type": "adv-cat-values", "row": MATCH}, "options"),
    prevent_initial_call=True,
)
def select_all_advanced_category_values(all_value: list[str] | None, options: list[dict[str, Any]] | None) -> list[str]:
    if not options or not all_value or "all" not in all_value:
        return []
    return [str(option.get("value")) for option in options if option.get("value") not in (None, "")]


@app.callback(
    Output({"type": "eda-filter-values", "slot": MATCH}, "value"),
    Input({"type": "eda-filter-all", "slot": MATCH}, "value"),
    State({"type": "eda-filter-values", "slot": MATCH}, "options"),
    prevent_initial_call=True,
)
def select_all_eda_filter_values(all_value: list[str] | None, options: list[dict[str, Any]] | None) -> list[str]:
    if not options or not all_value or "all" not in all_value:
        return []
    return [str(option.get("value")) for option in options if option.get("value") not in (None, "")]


@app.callback(
    Output({"type": "compare-filter-values", "slot": MATCH}, "value"),
    Input({"type": "compare-filter-all", "slot": MATCH}, "value"),
    State({"type": "compare-filter-values", "slot": MATCH}, "options"),
    prevent_initial_call=True,
)
def select_all_compare_filter_values(all_value: list[str] | None, options: list[dict[str, Any]] | None) -> list[str]:
    if not options or not all_value or "all" not in all_value:
        return []
    return [str(option.get("value")) for option in options if option.get("value") not in (None, "")]


@app.callback(
    Output({"type": "kpi-filter-values", "slot": MATCH}, "value"),
    Input({"type": "kpi-filter-all", "slot": MATCH}, "value"),
    State({"type": "kpi-filter-values", "slot": MATCH}, "options"),
    prevent_initial_call=True,
)
def select_all_kpi_filter_values(all_value: list[str] | None, options: list[dict[str, Any]] | None) -> list[str]:
    if not options or not all_value or "all" not in all_value:
        return []
    return [str(option.get("value")) for option in options if option.get("value") not in (None, "")]


@app.callback(
    Output("advanced-filter-store", "data"),
    Output("adv-filter-summary", "children"),
    Input("adv-filter-logic", "value"),
    Input({"type": "adv-column", "row": ALL}, "id"),
    Input({"type": "adv-column", "row": ALL}, "value"),
    Input({"type": "adv-operator", "row": ALL}, "id"),
    Input({"type": "adv-operator", "row": ALL}, "value"),
    Input({"type": "adv-cat-values", "row": ALL}, "id"),
    Input({"type": "adv-cat-values", "row": ALL}, "value"),
    Input({"type": "adv-num-value", "row": ALL}, "id"),
    Input({"type": "adv-num-value", "row": ALL}, "value"),
    Input({"type": "adv-date-year", "row": ALL}, "id"),
    Input({"type": "adv-date-year", "row": ALL}, "value"),
    Input({"type": "adv-date-month", "row": ALL}, "id"),
    Input({"type": "adv-date-month", "row": ALL}, "value"),
    Input({"type": "adv-date-start-year", "row": ALL}, "id"),
    Input({"type": "adv-date-start-year", "row": ALL}, "value"),
    Input({"type": "adv-date-start-month", "row": ALL}, "id"),
    Input({"type": "adv-date-start-month", "row": ALL}, "value"),
    Input({"type": "adv-date-end-year", "row": ALL}, "id"),
    Input({"type": "adv-date-end-year", "row": ALL}, "value"),
    Input({"type": "adv-date-end-month", "row": ALL}, "id"),
    Input({"type": "adv-date-end-month", "row": ALL}, "value"),
    Input({"type": "adv-text-value", "row": ALL}, "id"),
    Input({"type": "adv-text-value", "row": ALL}, "value"),
    Input("dataset-key", "data"),
)
def sync_advanced_filter_store(
    logic: str,
    column_ids: list[dict[str, Any]],
    column_values: list[str | None],
    operator_ids: list[dict[str, Any]],
    operator_values: list[str | None],
    cat_ids: list[dict[str, Any]],
    cat_values: list[list[str] | None],
    num_value_ids: list[dict[str, Any]],
    num_value_values: list[float | None],
    date_year_ids: list[dict[str, Any]],
    date_year_values: list[int | None],
    date_month_ids: list[dict[str, Any]],
    date_month_values: list[int | None],
    date_start_year_ids: list[dict[str, Any]],
    date_start_year_values: list[int | None],
    date_start_month_ids: list[dict[str, Any]],
    date_start_month_values: list[int | None],
    date_end_year_ids: list[dict[str, Any]],
    date_end_year_values: list[int | None],
    date_end_month_ids: list[dict[str, Any]],
    date_end_month_values: list[int | None],
    text_ids: list[dict[str, Any]],
    text_values: list[str | None],
    dataset_key: str | None,
) -> tuple[dict[str, Any], html.Div]:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return {"logic": logic, "rules": []}, html.Div("Charge les données pour activer les filtres avancés.", className="filter-helper")

    df = DATA_CACHE[dataset_key]
    column_map = {int(item["row"]): value for item, value in zip(column_ids, column_values)}
    operator_map = {int(item["row"]): value for item, value in zip(operator_ids, operator_values)}
    cat_map = {int(item["row"]): value for item, value in zip(cat_ids, cat_values)}
    num_value_map = {int(item["row"]): value for item, value in zip(num_value_ids, num_value_values)}
    date_year_map = {int(item["row"]): value for item, value in zip(date_year_ids, date_year_values)}
    date_month_map = {int(item["row"]): value for item, value in zip(date_month_ids, date_month_values)}
    date_start_year_map = {int(item["row"]): value for item, value in zip(date_start_year_ids, date_start_year_values)}
    date_start_month_map = {int(item["row"]): value for item, value in zip(date_start_month_ids, date_start_month_values)}
    date_end_year_map = {int(item["row"]): value for item, value in zip(date_end_year_ids, date_end_year_values)}
    date_end_month_map = {int(item["row"]): value for item, value in zip(date_end_month_ids, date_end_month_values)}
    text_map = {int(item["row"]): value for item, value in zip(text_ids, text_values)}

    rules: list[dict[str, Any]] = []
    for row_index in sorted(column_map.keys()):
        column = column_map.get(row_index)
        operator = operator_map.get(row_index)
        if not column or column not in df.columns or not operator:
            continue
        kind = infer_column_kind(df, column)
        if kind == "numeric":
            value = num_value_map.get(row_index)
        elif kind == "date":
            if operator == "between":
                value = {
                    "start_year": date_start_year_map.get(row_index),
                    "start_month": date_start_month_map.get(row_index),
                    "end_year": date_end_year_map.get(row_index),
                    "end_month": date_end_month_map.get(row_index),
                }
            else:
                value = {
                    "year": date_year_map.get(row_index),
                    "month": date_month_map.get(row_index),
                }
        elif kind == "categorical" and operator == "equals":
            value = cat_map.get(row_index) or []
        else:
            text_value = text_map.get(row_index)
            value = text_value or ""
        rules.append({"row": row_index, "column": column, "kind": kind, "operator": operator, "value": value})

    bundle = {"logic": logic or "AND", "rules": rules}
    return bundle, html.Div(advanced_filter_summary(bundle), className="filter-helper")


@app.callback(
    Output("compare-filter-panel", "children"),
    Input("compare-x", "value"),
    Input("compare-y", "value"),
    Input("dataset-key", "data"),
)
def render_compare_filter_panel(x_var: str | None, y_var: str | None, dataset_key: str | None) -> html.Div:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return html.Div()

    df = DATA_CACHE[dataset_key]
    candidates = [column for column in [x_var, y_var] if column and column in df.columns]
    if not candidates:
        return html.Div()

    filter_column = candidates[0]
    kind = infer_column_kind(df, filter_column)
    if kind == "categorical":
        series = df[filter_column].fillna("<NA>").astype(str)
        counts = safe_top_value_counts(series, MAX_CATEGORICAL_VALUES_FOR_SELECTOR)
        unique_estimate = int(series.nunique(dropna=False))
        options = [{"label": f"{truncate_label(idx, 42)} ({count})", "value": idx} for idx, count in counts.items()]
        return html.Div(
            [
                html.Label(f"Filtre sur {filter_column}"),
                dcc.Checklist(
                    id={"type": "compare-filter-all", "slot": "main"},
                    options=[{"label": "All", "value": "all"}],
                    value=[],
                    inline=True,
                ),
                dcc.Dropdown(id={"type": "compare-filter-values", "slot": "main"}, options=options, value=[], multi=True),
                html.Div(
                    f"Cardinalité: {unique_estimate:,}. Affichage top fréquences uniquement pour éviter une UI lourde.".replace(",", " "),
                    className="hint-text",
                ),
            ],
            className="control-card",
        )
    if kind == "numeric":
        numeric = pd.to_numeric(df[filter_column], errors="coerce").dropna()
        if numeric.empty:
            return html.Div()
        return html.Div(
            [
                html.Label(f"Filtre numérique sur {filter_column}"),
                dcc.RangeSlider(id={"type": "compare-filter-range", "slot": "main"}, min=float(numeric.min()), max=float(numeric.max()), value=[float(numeric.min()), float(numeric.max())]),
            ],
            className="control-card",
        )
    if kind == "date":
        return html.Div(
            [
                html.Label(f"Filtre date sur {filter_column}"),
                html.Div("Pas de filtre start/end en mode Comparer pour les axes temporels.", className="hint-text"),
            ],
            className="control-card",
        )
    return html.Div()


@app.callback(
    Output("kpi-filter-panel", "children"),
    Input("kpi-group-cols", "value"),
    Input("dataset-key", "data"),
)
def render_kpi_filter_panel(group_cols: list[str] | None, dataset_key: str | None) -> html.Div:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return html.Div()

    df = DATA_CACHE[dataset_key]
    if not group_cols:
        return html.Div()

    candidate = group_cols[0]
    if candidate not in df.columns:
        return html.Div()

    kind = infer_column_kind(df, candidate)
    if kind == "categorical":
        series = df[candidate].fillna("<NA>").astype(str)
        counts = safe_top_value_counts(series, MAX_CATEGORICAL_VALUES_FOR_SELECTOR)
        unique_estimate = int(series.nunique(dropna=False))
        options = [{"label": f"{truncate_label(idx, 42)} ({count})", "value": idx} for idx, count in counts.items()]
        return html.Div(
            [
                html.Label(f"Filtre métier sur {candidate}"),
                dcc.Checklist(
                    id={"type": "kpi-filter-all", "slot": "main"},
                    options=[{"label": "All", "value": "all"}],
                    value=[],
                    inline=True,
                ),
                dcc.Dropdown(id={"type": "kpi-filter-values", "slot": "main"}, options=options, value=[], multi=True),
                html.Div(
                    f"Cardinalité: {unique_estimate:,}. Affichage limité aux valeurs les plus fréquentes.".replace(",", " "),
                    className="hint-text",
                ),
            ],
            className="control-card",
        )
    if kind == "numeric":
        numeric = pd.to_numeric(df[candidate], errors="coerce").dropna()
        if numeric.empty:
            return html.Div()
        return html.Div(
            [
                html.Label(f"Filtre numérique sur {candidate}"),
                dcc.RangeSlider(id={"type": "kpi-filter-range", "slot": "main"}, min=float(numeric.min()), max=float(numeric.max()), value=[float(numeric.min()), float(numeric.max())]),
            ],
            className="control-card",
        )
    if kind == "date":
        date_series = pd.to_datetime(df[candidate], errors="coerce").dropna()
        if date_series.empty:
            return html.Div()
        year_values = list(range(int(date_series.dt.year.min()), int(date_series.dt.year.max()) + 1))
        year_opts = [{"label": str(year), "value": year} for year in year_values]
        return html.Div(
            [
                html.Label(f"Filtre date sur {candidate}"),
                build_month_year_range_block(
                    start_year_id={"type": "kpi-filter-date-start-year", "slot": "main"},
                    start_month_id={"type": "kpi-filter-date-start-month", "slot": "main"},
                    end_year_id={"type": "kpi-filter-date-end-year", "slot": "main"},
                    end_month_id={"type": "kpi-filter-date-end-month", "slot": "main"},
                    year_options=year_opts,
                    start_year_value=year_values[0],
                    start_month_value=int(date_series.dt.month.min()),
                    end_year_value=year_values[-1],
                    end_month_value=int(date_series.dt.month.max()),
                ),
            ],
            className="control-card",
        )
    return html.Div()


@app.callback(
    Output("compare-plot-type", "options"),
    Output("compare-plot-type", "value"),
    Input("compare-x", "value"),
    Input("compare-y", "value"),
    State("dataset-key", "data"),
)
def update_compare_plot_options(x_var: str | None, y_var: str | None, dataset_key: str | None) -> tuple[list[dict[str, str]], str]:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return [{"label": "Auto", "value": "auto"}], "auto"
    df = DATA_CACHE[dataset_key]
    if not x_var or not y_var or x_var not in df.columns or y_var not in df.columns:
        return [{"label": "Auto", "value": "auto"}], "auto"
    x_kind = infer_column_kind(df, x_var)
    y_kind = infer_column_kind(df, y_var)
    options = compare_plot_options(x_kind, y_kind)
    return options, options[0]["value"] if options else "auto"


@app.callback(
    Output("kpi-start-year", "options"),
    Output("kpi-start-year", "value"),
    Output("kpi-end-year", "options"),
    Output("kpi-end-year", "value"),
    Input("kpi-date-col", "value"),
    Input("dataset-key", "data"),
)
def update_kpi_year_bounds(date_col: str | None, dataset_key: str | None) -> tuple[list[dict[str, int]], int | None, list[dict[str, int]], int | None]:
    if not dataset_key or dataset_key not in DATA_CACHE or not date_col:
        return [], None, [], None

    df = DATA_CACHE[dataset_key]
    if date_col not in df.columns:
        return [], None, [], None

    years = pd.to_datetime(df[date_col], errors="coerce").dt.year.dropna().astype(int)
    if years.empty:
        return [], None, [], None
    year_values = sorted(years.unique().tolist())
    options = [{"label": str(year), "value": year} for year in year_values]
    return options, year_values[0], options, year_values[-1]


@app.callback(
    Output("eda-all-pager", "style"),
    Output("eda-output-pager", "style"),
    Input("eda-variable", "value"),
    Input("eda-all-total-pages", "data"),
    Input("eda-output-total-pages", "data"),
)
def toggle_eda_all_controls(variable: str | None, all_total_pages: int | None, output_total_pages: int | None) -> tuple[dict[str, str], dict[str, str]]:
    show_all = variable == "__ALL__" and int(all_total_pages or 1) > 1
    show_output = variable not in (None, "__ALL__") and int(output_total_pages or 1) > 1
    all_style = {"display": "flex", "justifyContent": "center", "gap": "12px", "marginTop": "8px", "marginBottom": "12px"} if show_all else {"display": "none"}
    out_style = {"display": "flex", "justifyContent": "center", "gap": "12px", "marginTop": "8px", "marginBottom": "12px"} if show_output else {"display": "none"}
    return all_style, out_style


@app.callback(
    Output("eda-all-page-store", "data"),
    Output("eda-all-total-pages", "data"),
    Output("eda-all-page-label", "children"),
    Input("eda-all-prev", "n_clicks"),
    Input("eda-all-next", "n_clicks"),
    Input("eda-variable", "value"),
    Input("dataset-key", "data"),
    State("eda-all-page-store", "data"),
)
def update_eda_all_page_state(
    prev_clicks: int,
    next_clicks: int,
    variable: str | None,
    dataset_key: str | None,
    current_page: int | None,
) -> tuple[int, int, str]:
    if variable != "__ALL__" or not dataset_key or dataset_key not in DATA_CACHE:
        return 1, 1, ""

    size = 12
    n_cols = max(1, len(DATA_CACHE[dataset_key].columns))
    total_pages = max(1, int(np.ceil(n_cols / size)))
    page = max(1, min(int(current_page or 1), total_pages))

    trigger = ctx.triggered_id
    if trigger == "eda-all-prev":
        page = max(1, page - 1)
    elif trigger == "eda-all-next":
        page = min(total_pages, page + 1)
    elif trigger in {"dataset-key", "eda-variable"}:
        page = 1

    return page, total_pages, f"Page {page}/{total_pages}"


@app.callback(
    Output("eda-output-page-store", "data"),
    Output("eda-output-page-label", "children"),
    Input("eda-output-prev", "n_clicks"),
    Input("eda-output-next", "n_clicks"),
    Input("eda-variable", "value"),
    Input("dataset-key", "data"),
    Input("advanced-filter-store", "data"),
    State("eda-output-page-store", "data"),
)
def update_eda_output_page_state(
    prev_clicks: int,
    next_clicks: int,
    variable: str | None,
    dataset_key: str | None,
    advanced_filter_store: dict[str, Any] | None,
    current_page: int | None,
) -> tuple[int, str]:
    if not dataset_key or dataset_key not in DATA_CACHE or variable in (None, "__ALL__"):
        return 1, ""

    df = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    pages = eda_output_total_pages(df, variable, page_size=12)
    pages = max(1, int(pages))

    page = max(1, min(int(current_page or 1), pages))
    trigger = ctx.triggered_id
    if trigger == "eda-output-prev":
        page = max(1, page - 1)
    elif trigger == "eda-output-next":
        page = min(pages, page + 1)
    elif trigger in {"eda-variable", "dataset-key"}:
        page = 1

    return page, f"Page {page}/{pages}"


@app.callback(
    Output("eda-graph", "figure"),
    Output("eda-graph", "style"),
    Output("eda-all-grid", "children"),
    Output("eda-unique-values", "children"),
    Output("eda-output-table", "children"),
    Output("eda-output-total-pages", "data"),
    Input("eda-variable", "value"),
    Input("eda-plot-type", "value"),
    Input("eda-all-page-store", "data"),
    Input("eda-output-page-store", "data"),
    Input({"type": "eda-filter-values", "slot": ALL}, "value"),
    Input({"type": "eda-filter-range", "slot": ALL}, "value"),
    Input({"type": "eda-filter-date-start-year", "slot": ALL}, "value"),
    Input({"type": "eda-filter-date-start-month", "slot": ALL}, "value"),
    Input({"type": "eda-filter-date-end-year", "slot": ALL}, "value"),
    Input({"type": "eda-filter-date-end-month", "slot": ALL}, "value"),
    Input("advanced-filter-store", "data"),
    Input("dataset-key", "data"),
)
def update_eda_graph(
    variable: str | None,
    plot_type: str,
    all_page: int | None,
    output_page: int | None,
    filter_values_all: list[list[str] | None],
    filter_range_all: list[list[float] | None],
    filter_start_year_all: list[int | None],
    filter_start_month_all: list[int | None],
    filter_end_year_all: list[int | None],
    filter_end_month_all: list[int | None],
    advanced_filter_store: dict[str, Any] | None,
    dataset_key: str | None,
) -> tuple[go.Figure, dict[str, str], html.Div, html.Div, html.Div, int]:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return build_empty_figure("Charge les données."), {"display": "none"}, html.Div(), html.Div(), html.Div(), 1

    df = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    if not variable:
        return build_empty_figure("Sélectionne une variable."), {"display": "none"}, html.Div(), html.Div(), html.Div(), 1

    if variable == "__ALL__":
        total_pages = max(1, int(np.ceil(max(1, len(df.columns)) / 12)))
        return (
            go.Figure(),
            {"display": "none"},
            build_all_variables_grid(df, page=int(all_page or 1), page_size=12),
            html.Div(),
            html.Div(),
            total_pages,
        )

    if variable not in df.columns:
        return build_empty_figure("Sélectionne une variable."), {"display": "block"}, html.Div(), html.Div(), html.Div(), 1

    filter_values = first_or_none(filter_values_all)
    filter_range = first_or_none(filter_range_all)
    filter_date_start = month_year_to_start(first_or_none(filter_start_year_all), first_or_none(filter_start_month_all))
    filter_date_end = month_year_to_end(first_or_none(filter_end_year_all), first_or_none(filter_end_month_all))

    mask = pd.Series(True, index=df.index)
    kind = infer_column_kind(df, variable)
    if kind == "categorical" and filter_values:
        mask &= df[variable].fillna("<NA>").astype(str).isin(filter_values)
    elif kind == "numeric" and filter_range:
        numeric = pd.to_numeric(df[variable], errors="coerce")
        mask &= numeric >= float(filter_range[0])
        mask &= numeric <= float(filter_range[1])
    elif kind == "date" and (filter_date_start or filter_date_end):
        dt = pd.to_datetime(df[variable], errors="coerce")
        if filter_date_start is not None:
            mask &= dt >= filter_date_start
        if filter_date_end is not None:
            mask &= dt <= filter_date_end

    filtered = df.loc[mask].copy()
    if filtered.empty:
        return build_empty_figure("Aucune ligne après filtre."), {"display": "block"}, html.Div(), html.Div(), html.Div("Aucune ligne à afficher.", className="empty-state"), 1

    fig = build_eda_figure(filtered, variable, plot_type)
    out_table, out_pages = build_eda_output_table(filtered, variable, page=int(output_page or 1), page_size=12)
    return fig, {"display": "block"}, html.Div(), html.Div(), out_table, out_pages


@app.callback(
    Output("eda-plot-type", "options"),
    Output("eda-plot-type", "value"),
    Input("eda-variable", "value"),
    Input("dataset-key", "data"),
)
def update_eda_plot_options(variable: str | None, dataset_key: str | None) -> tuple[list[dict[str, str]], str]:
    if not dataset_key or dataset_key not in DATA_CACHE or not variable:
        return [{"label": "Auto", "value": "auto"}], "auto"

    if variable == "__ALL__":
        return [{"label": "Auto", "value": "auto"}], "auto"

    df = DATA_CACHE[dataset_key]
    if variable not in df.columns:
        return [{"label": "Auto", "value": "auto"}], "auto"

    options = eda_plot_options(infer_column_kind(df, variable))
    return options, options[0]["value"] if options else "auto"


@app.callback(
    Output("compare-graph", "figure"),
    Input("compare-x", "value"),
    Input("compare-y", "value"),
    Input("compare-hue", "value"),
    Input("compare-plot-type", "value"),
    Input({"type": "compare-filter-values", "slot": ALL}, "value"),
    Input({"type": "compare-filter-range", "slot": ALL}, "value"),
    Input("advanced-filter-store", "data"),
    Input("dataset-key", "data"),
)
def update_compare_graph(
    x_var: str | None,
    y_var: str | None,
    hue: str | None,
    plot_type: str,
    filter_values_all: list[list[str] | None],
    filter_range_all: list[list[float] | None],
    advanced_filter_store: dict[str, Any] | None,
    dataset_key: str | None,
) -> go.Figure:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return build_empty_figure("Charge les données.")

    df = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    if not x_var or not y_var or x_var not in df.columns or y_var not in df.columns:
        return build_empty_figure("Choisis X et Y.")

    filter_values = first_or_none(filter_values_all)
    filter_range = first_or_none(filter_range_all)
    mask = pd.Series(True, index=df.index)
    filter_candidates = [column for column in [x_var, y_var] if column in df.columns]
    if filter_candidates:
        filter_column = filter_candidates[0]
        kind = infer_column_kind(df, filter_column)
        if kind == "categorical" and filter_values:
            mask &= df[filter_column].fillna("<NA>").astype(str).isin(filter_values)
        elif kind == "numeric" and filter_range:
            numeric = pd.to_numeric(df[filter_column], errors="coerce")
            mask &= numeric >= float(filter_range[0])
            mask &= numeric <= float(filter_range[1])

    filtered = df.loc[mask].copy()
    if filtered.empty:
        return build_empty_figure("Aucune ligne après filtre.")

    return build_compare_figure(filtered, x_var, y_var, hue if hue else None, plot_type)


@app.callback(
    Output("kpi-graph", "figure"),
    Output("kpi-table-container", "children"),
    Output("kpi-status", "children"),
    Input("run-kpi-button", "n_clicks"),
    State("kpi-grain", "value"),
    State("kpi-date-col", "value"),
    State("kpi-group-cols", "value"),
    State("kpi-value-col", "value"),
    State("kpi-agg-func", "value"),
    State("kpi-chart-style", "value"),
    State("kpi-start-year", "value"),
    State("kpi-start-month", "value"),
    State("kpi-end-year", "value"),
    State("kpi-end-month", "value"),
    State("kpi-pivot-rows", "value"),
    State("kpi-pivot-cols", "value"),
    State({"type": "kpi-filter-values", "slot": ALL}, "value"),
    State({"type": "kpi-filter-range", "slot": ALL}, "value"),
    State({"type": "kpi-filter-date-start-year", "slot": ALL}, "value"),
    State({"type": "kpi-filter-date-start-month", "slot": ALL}, "value"),
    State({"type": "kpi-filter-date-end-year", "slot": ALL}, "value"),
    State({"type": "kpi-filter-date-end-month", "slot": ALL}, "value"),
    State("advanced-filter-store", "data"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def update_kpi(
    n_clicks: int,
    grain: str,
    date_col: str | None,
    group_cols: list[str] | None,
    value_col: str | None,
    agg_func: str,
    chart_style: str,
    start_year: int | None,
    start_month: int | None,
    end_year: int | None,
    end_month: int | None,
    pivot_rows: str | None,
    pivot_cols: str | None,
    filter_values_all: list[list[str] | None],
    filter_range_all: list[list[float] | None],
    filter_start_year_all: list[int | None],
    filter_start_month_all: list[int | None],
    filter_end_year_all: list[int | None],
    filter_end_month_all: list[int | None],
    advanced_filter_store: dict[str, Any] | None,
    dataset_key: str | None,
) -> tuple[go.Figure, html.Div, str]:
    if not dataset_key or dataset_key not in DATA_CACHE:
        raise PreventUpdate

    df = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    if df.empty:
        return build_empty_figure("Aucune donnée."), html.Div(), "Aucune donnée chargée"
    if not date_col or date_col not in df.columns:
        return build_empty_figure("Choisis une colonne date."), html.Div(), "Choisis une colonne date"

    filter_values = first_or_none(filter_values_all)
    filter_range = first_or_none(filter_range_all)
    filter_date_start = month_year_to_start(first_or_none(filter_start_year_all), first_or_none(filter_start_month_all))
    filter_date_end = month_year_to_end(first_or_none(filter_end_year_all), first_or_none(filter_end_month_all))

    working = df.copy()
    date_series = pd.to_datetime(working[date_col], errors="coerce")
    if start_year is not None and start_month is not None:
        start_ts = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
        working = working[date_series >= start_ts]
    if end_year is not None and end_month is not None:
        end_ts = pd.Timestamp(year=int(end_year), month=int(end_month), day=1) + pd.offsets.MonthEnd(1)
        working = working[date_series <= end_ts]
        if start_year is not None and start_month is not None and start_ts > end_ts:
            return build_empty_figure("Période invalide: début > fin."), html.Div(), "Période invalide"

    if group_cols:
        filter_candidate = group_cols[0]
        kind = infer_column_kind(working, filter_candidate)
        if kind == "categorical" and filter_values:
            working = working[working[filter_candidate].fillna("<NA>").astype(str).isin(filter_values)]
        elif kind == "numeric" and filter_range:
            numeric = pd.to_numeric(working[filter_candidate], errors="coerce")
            working = working[(numeric >= float(filter_range[0])) & (numeric <= float(filter_range[1]))]
        elif kind == "date" and (filter_date_start or filter_date_end):
            dt = pd.to_datetime(working[filter_candidate], errors="coerce")
            if filter_date_start is not None:
                working = working[dt >= filter_date_start]
            if filter_date_end is not None:
                working = working[dt <= filter_date_end]

    pivot_candidates = [column for column in [date_col] + list(group_cols or []) if column and column in working.columns]
    pivot_rows_resolved = pivot_rows if pivot_rows in pivot_candidates else (pivot_candidates[0] if pivot_candidates else None)
    pivot_cols_resolved = pivot_cols if pivot_cols in pivot_candidates and pivot_cols != pivot_rows_resolved else None
    if pivot_cols_resolved is None:
        for candidate in pivot_candidates:
            if candidate != pivot_rows_resolved:
                pivot_cols_resolved = candidate
                break

    try:
        pivot_df, chart = build_kpi_table(
            working,
            grain=grain,
            date_col=date_col,
            group_cols=group_cols or [],
            value_col=value_col if value_col else None,
            agg_func=agg_func,
            pivot_rows=pivot_rows_resolved,
            pivot_cols=pivot_cols_resolved,
            chart_style=chart_style or "line",
        )
    except Exception as exc:
        return build_empty_figure(f"Erreur KPI: {exc}"), html.Div(), f"Erreur: {exc}"

    table = build_preview_table(pivot_df, max_rows=100)
    status = f"KPI construit: {len(pivot_df):,} lignes".replace(",", " ")
    if grain == "monthly" and "TIME_BUCKET" in pivot_df.columns and pivot_df["TIME_BUCKET"].nunique() > MAX_KPI_CHART_POINTS:
        status += f" | graphe limité aux {MAX_KPI_CHART_POINTS} derniers mois"
    return chart, table, status


@app.callback(
    Output("kpi-pivot-rows", "options"),
    Output("kpi-pivot-cols", "options"),
    Output("kpi-pivot-rows", "value"),
    Output("kpi-pivot-cols", "value"),
    Input("kpi-date-col", "value"),
    Input("kpi-group-cols", "value"),
    Input("dataset-key", "data"),
    State("kpi-pivot-rows", "value"),
    State("kpi-pivot-cols", "value"),
)
def update_kpi_pivot_options(
    date_col: str | None,
    group_cols: list[str] | None,
    dataset_key: str | None,
    current_rows: str | None,
    current_cols: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str]:
    options = [{"label": "Aucun", "value": ""}]
    if not dataset_key or dataset_key not in DATA_CACHE:
        return options, options, "", ""

    df = DATA_CACHE[dataset_key]
    selected_columns: list[str] = []
    if date_col and date_col in df.columns:
        selected_columns.append(date_col)
    for column in group_cols or []:
        if column in df.columns and column not in selected_columns:
            selected_columns.append(column)

    options = [{"label": column, "value": column} for column in selected_columns]
    options = ([{"label": "Aucun", "value": ""}] + options) if options else [{"label": "Aucun", "value": ""}]

    valid_values = {option["value"] for option in options}
    rows_value = current_rows if current_rows in valid_values else ""
    cols_value = current_cols if current_cols in valid_values and current_cols != rows_value else ""
    if not cols_value:
        for column in selected_columns:
            if column != rows_value:
                cols_value = column
                break

    return options, options, rows_value, cols_value


@app.callback(
    Output("preview-table-container", "children"),
    Input("dataset-key", "data"),
    Input("advanced-filter-store", "data"),
)
def update_preview(dataset_key: str | None, advanced_filter_store: dict[str, Any] | None) -> html.Div:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return html.Div("Charge les données pour voir l'aperçu.", className="empty-state")
    return build_preview_table(apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store), max_rows=80)


@app.callback(
    Output("interaction-main-graph", "figure"),
    Output("interaction-heatmap", "figure"),
    Output("interaction-ranking-table", "children"),
    Input("interaction-target", "value"),
    Input("interaction-corr-vars", "value"),
    Input("advanced-filter-store", "data"),
    Input("dataset-key", "data"),
)
def update_interactions(
    target_column: str | None,
    corr_columns: list[str] | None,
    advanced_filter_store: dict[str, Any] | None,
    dataset_key: str | None,
) -> tuple[go.Figure, go.Figure, html.Div]:
    if not dataset_key or dataset_key not in DATA_CACHE:
        return build_empty_figure("Charge les données."), build_empty_figure("Matrice de corrélation encodée."), html.Div()

    df = apply_advanced_filters(DATA_CACHE[dataset_key], advanced_filter_store)
    if df.empty:
        return build_empty_figure("Aucune donnée après filtres."), build_empty_figure("Aucune donnée après filtres."), html.Div("Aucune donnée après filtres.", className="empty-state")

    selected_corr_columns = [column for column in (corr_columns or []) if column]
    if not selected_corr_columns:
        heatmap = build_empty_figure("Choisis des variables de corrélation (ou ALL).")
    else:
        if "__ALL__" in selected_corr_columns and len(selected_corr_columns) > 1:
            selected_corr_columns = [column for column in selected_corr_columns if column != "__ALL__"]
        heatmap = build_encoded_correlation_figure(df, selected_columns=selected_corr_columns, max_cols=MAX_INTERACTION_CORR_COLS)

    ranking_table: html.Div | Any = html.Div("Choisis une variable cible pour voir le ranking corrélé.", className="hint-text")
    if not target_column or target_column not in df.columns:
        return build_empty_figure("Choisis une variable cible."), heatmap, ranking_table

    top_df = compute_focal_correlation_scores(df, target_column, max_cols=MAX_INTERACTION_CORR_COLS)
    if top_df.empty:
        return build_empty_figure("Corrélation indisponible pour cette cible."), heatmap, html.Div("Aucune corrélation exploitable.", className="empty-state")

    main_fig = px.bar(
        top_df,
        x="VARIABLE",
        y="CORR",
        color="CORR",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        title=f"Corrélation encodée de {target_column} avec les autres",
    )
    main_fig.update_layout(template="plotly_white")
    main_fig.update_xaxes(tickangle=30)

    ranking_table = build_preview_table(top_df[["VARIABLE", "CORR", "ABS_CORR"]], max_rows=12)
    return main_fig, heatmap, ranking_table


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
