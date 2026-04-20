from __future__ import annotations

import calendar
import glob
import os
import re
from pathlib import Path
from html import escape as html_escape
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update



BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FOLDER = BASE_DIR / "data"
DEFAULT_MARKET_DATA_PATH = BASE_DIR / "market_dataset_europe.parquet"

HARDCODED_DATA_PATH = ""
COUNTRIES_TO_READ = ["SPAIN"]
START_YYYYMM = "202501"
END_YYYYMM = "202512"



COLUMNS_TO_READ = [
    'EXTENSION_DATE',
    'CLASS_CATALOG',
    'CONTRACT_START_DATE',
    'CONTRACT_END_DATE',
    'COB_DATE',
    'DATE_OF_ORDER',
    'CONTRACT_END_DATE_AMENDED',
    'ID_CONTRACT',
    'VEHICLE_ID',
    'ID_QUOTATION',
    'FINAL_CONTRACT_DURATION',
    'POWER_CATEGORY',
    'BIKE_OR_CAR',
    'CLS_VEHICLE_TYPE',
    'COUNTRY',
    'BRAND_UPDATE',
    'VEHICLE_CLASS',
    'VEHICLE_MODEL',
    'MODEL_CATALOG',
    'OEM_UPDATE',
    'NOVA_ASSET_STATUS',
    'FUEL_TYPE2',
    'FUEL_TYPE',
]

def normalize_copied_path(path_value: str) -> str:
    return path_value.strip().strip('"').strip("'")


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
    end_int = int(end_yyyymm)
    countries_upper = {c.strip().upper() for c in countries}

    for file_path in files:
        filename = os.path.basename(file_path).replace(".parquet", "")
        parts = [p.strip() for p in filename.split("-")]
        if len(parts) < 3:
            continue

        file_country = parts[1].upper()
        file_yyyymm = int(parts[2])

        if file_country in countries_upper and start_int <= file_yyyymm <= end_int:
            df_part = pd.read_parquet(file_path, columns=cols)
            dfs.append(df_part)

    return pd.concat(dfs, ignore_index=True)


def load_dataset() -> pd.DataFrame:
    raw_path = normalize_copied_path(HARDCODED_DATA_PATH)
    data_folder = Path(raw_path) if raw_path else DEFAULT_DATA_FOLDER

    return load_country_monthly_data(
        folder_path=data_folder,
        countries=COUNTRIES_TO_READ,
        start_yyyymm=START_YYYYMM,
        end_yyyymm=END_YYYYMM,
        cols=COLUMNS_TO_READ,
    )


def prepare_data_set(df: pd.DataFrame) -> pd.DataFrame:
    df["COB_DATE"] = pd.to_datetime(df["COB_DATE"], errors="coerce")
    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month
    if "CONTRACT_START_DATE" in df.columns:
        df["CONTRACT_S" \
        "TART_DATE"] = pd.to_datetime(df["CONTRACT_START_DATE"], errors="coerce")

    fuel_mapping = {
        "ES": {"GASOLINA": "PETROL", "DIESEL": "DIESEL"},
        "SPAIN": {"GASOLINA": "PETROL", "DIESEL": "DIESEL"},
        "IT": {"BENZINA": "PETROL", "DIESEL": "DIESEL"},
        "ITALY": {"BENZINA": "PETROL", "DIESEL": "DIESEL"},
    }
    fuel_suffix_map = {
        "GASOLINA": "G",
        "BENZINA": "G",
        "PETROL": "G",
        "DIESEL": "D",
    }

    country = df["COUNTRY"].astype(str).str.strip().str.upper()
    power_category = df["POWER_CATEGORY"].astype(str).str.strip().str.upper()
    fuel_type2 = df.get("FUEL_TYPE2", pd.Series(index=df.index, dtype="object")).astype(str).str.strip().str.upper()
    fuel_type = df.get("FUEL_TYPE", pd.Series(index=df.index, dtype="object")).astype(str).str.strip().str.upper()

    mapping = country.map(fuel_mapping).apply(lambda value: value if isinstance(value, dict) else {})
    mapped_fuel2 = [row_mapping.get(value) if isinstance(row_mapping, dict) else None for row_mapping, value in zip(mapping, fuel_type2)]
    mapped_fuel1 = [row_mapping.get(value) if isinstance(row_mapping, dict) else None for row_mapping, value in zip(mapping, fuel_type)]

    suffix2 = fuel_type2.map(fuel_suffix_map)
    suffix1 = fuel_type.map(fuel_suffix_map)

    df["POWER_CATEGORY_2"] = np.where(
        power_category.eq("MHEV"),
        pd.Series(mapped_fuel2, index=df.index).fillna(pd.Series(mapped_fuel1, index=df.index)).fillna(df["POWER_CATEGORY"]),
        df["POWER_CATEGORY"],
    )

    df["POWER_CATEGORY_3"] = np.select(
        [
            power_category.eq("MHEV") & suffix2.notna(),
            power_category.eq("MHEV") & suffix2.isna() & suffix1.notna(),
            power_category.eq("PLUG-IN HYBRID") & suffix2.notna(),
            power_category.eq("PLUG-IN HYBRID") & suffix2.isna() & suffix1.notna(),
        ],
        [
            "MHEV-" + suffix2,
            "MHEV-" + suffix1,
            "PHEV-" + suffix2,
            "PHEV-" + suffix1,
        ],
        default=df["POWER_CATEGORY"],
    )

    df["POWER_CATEGORY_3"] = df["POWER_CATEGORY_3"].fillna(df["POWER_CATEGORY"])

    text_columns = [
        "COUNTRY",
        "NOVA_ASSET_STATUS",
        "BIKE_OR_CAR",
        "BRAND_UPDATE",
        "OEM_UPDATE",
        "POWER_CATEGORY",
        "POWER_CATEGORY_2",
        "POWER_CATEGORY_3",
        "CLS_VEHICLE_TYPE",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str)

    if "CLASS_CATALOG" in df.columns:
        vehicle_models = ["QASHQAI", "TUCSON", "X1", "XC40", "SPORTAGE", "T-ROC", "IBIZA", "ARONA", "XC60", "EXPRESS"]
        pattern = "|".join(vehicle_models)
        df["VEHICLE_MODEL_MAPED"] = df["CLASS_CATALOG"].astype(str).str.extract(
            f"({pattern})", flags=re.IGNORECASE, expand=False
        )
    else:
        df["VEHICLE_MODEL_MAPED"] = pd.NA

    if {"CONTRACT_END_DATE_AMENDED", "EXTENSION_DATE", "CONTRACT_END_DATE"}.issubset(df.columns):
        final_end_candidates = df[["CONTRACT_END_DATE_AMENDED", "EXTENSION_DATE", "CONTRACT_END_DATE"]].apply(
            pd.to_datetime,
            errors="coerce",
        )
        df["CONTRACT_FINAL_END"] = final_end_candidates.max(axis=1)
    else:
        df["CONTRACT_FINAL_END"] = pd.NaT

    return df


def load_market_dataset() -> pd.DataFrame:

    market_df = pd.read_parquet(DEFAULT_MARKET_DATA_PATH).copy()

    if "date" in market_df.columns:
        market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")

    text_columns = ["Country/Territory-Number", "Make", "Make Group", "Fuel Type"]
    for column in text_columns:
        if column in market_df.columns:
            market_df[column] = market_df[column].astype(str)

    return market_df


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def percent_or_na(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def apply_status_filter(df: pd.DataFrame, status: str | None) -> pd.DataFrame:
    if "NOVA_ASSET_STATUS" not in df.columns or status in (None, "ALL"):
        return df

    status_norm = str(status).strip().upper()
    status_series = df["NOVA_ASSET_STATUS"].astype(str)

    if status_norm == "IN FLEET":
        return df[status_series.str.contains("IN FLEET", case=False, na=False)]
    if status_norm == "ORDER":
        return df[status_series.str.contains("ORDER", case=False, na=False)]

    return df[status_series.str.contains(status_norm, case=False, na=False)]


def filter_base(df: pd.DataFrame, country: str, year: int | str, status: str = "IN FLEET", bike_or_car: str = "CAR") -> pd.DataFrame:
    out = df.copy()

    if country != "ALL":
        out = out[out["COUNTRY"] == country]

    if year != "ALL":
        out = out[out["YEAR"] == int(year)]

    out = apply_status_filter(out, status)

    if bike_or_car != "ALL" and "BIKE_OR_CAR" in out.columns:
        out = out[out["BIKE_OR_CAR"] == bike_or_car]

    return out


def filter_status_group(df: pd.DataFrame, status_group: str) -> pd.DataFrame:
    out = df.copy()
    if status_group == "Fleet":
        return apply_status_filter(out, "IN FLEET")
    if status_group == "Order":
        return apply_status_filter(out, "ORDER")
    if status_group in ("IN FLEET", "ORDER"):
        return apply_status_filter(out, status_group)
    return out


def available_months(df: pd.DataFrame, country: str, year: int | str) -> list[int]:
    subset = filter_base(df, country, year, status="IN FLEET", bike_or_car="CAR")
    return sorted(subset["MONTH"].dropna().astype(int).unique().tolist())


def latest_month(df: pd.DataFrame, country: str, year: int | str) -> int | None:
    months = available_months(df, country, year)
    if not months:
        return None
    return months[-1]


def kpi_lease_under_25(df: pd.DataFrame, country: str, year: int | str, month_value: int | str | None = "ALL") -> float | None:
    subset = df[
        (df["COUNTRY"] == country)
        & (df["YEAR"] == int(year))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()

    if month_value not in (None, "ALL"):
        subset = subset[subset["MONTH"] == int(month_value)]

    subset = subset.drop_duplicates(subset=["VEHICLE_ID"])
    if subset.empty:
        return None

    return float((subset["FINAL_CONTRACT_DURATION"] < 25).mean() * 100)


def kpi_lease_25_30(df: pd.DataFrame, country: str, year: int | str, month_value: int | str | None = "ALL") -> float | None:
    subset = df[
        (df["COUNTRY"] == country)
        & (df["YEAR"] == int(year))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()

    if month_value not in (None, "ALL"):
        subset = subset[subset["MONTH"] == int(month_value)]

    subset = subset.drop_duplicates(subset=["VEHICLE_ID"])
    if subset.empty:
        return None

    return float(subset["FINAL_CONTRACT_DURATION"].between(25, 30, inclusive="both").mean() * 100)


def kpi_diesel_non_diesel(df: pd.DataFrame, country: str, year: int | str, month: int) -> tuple[float | None, float | None]:
    subset = df[
        (df["COUNTRY"] == country)
        & (df["YEAR"] == int(year))
        & (df["MONTH"] == int(month))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
    if subset.empty:
        return None, None
    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY_2", "POWER_CATEGORY"])
    if power_col is None:
        return None, None
    values = subset[power_col].astype(str).str.upper()
    diesel_share = float((values == "DIESEL").mean() * 100)
    return diesel_share, float(100 - diesel_share)


def kpi_hybrid_share(df: pd.DataFrame, country: str, year: int | str, month: int) -> float | None:
    subset = df[
        (df["COUNTRY"] == country)
        & (df["YEAR"] == int(year))
        & (df["MONTH"] == int(month))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
    if subset.empty:
        return None
    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY"])
    if power_col is None:
        return None
    values = subset[power_col].astype(str).str.upper()
    return float(values.isin(["FULL HYBRID", "PLUG-IN HYBRID"]).mean() * 100)


def kpi_ev_share(df: pd.DataFrame, country: str, year: int | str, month: int) -> float | None:
    subset = df[
        (df["COUNTRY"] == country)
        & (df["YEAR"] == int(year))
        & (df["MONTH"] == int(month))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
    if subset.empty:
        return None
    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY"])
    if power_col is None:
        return None
    values = subset[power_col].astype(str).str.upper()
    return float((values == "ELECTRIC").mean() * 100)


def kpi_pv_lcv(df: pd.DataFrame, country: str, year: int | str, month: int) -> tuple[float | None, float | None]:
    subset = df[
        (df["COUNTRY"] == country)
        & (df["YEAR"] == int(year))
        & (df["MONTH"] == int(month))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
    if subset.empty:
        return None, None
    values = subset["CLS_VEHICLE_TYPE"].astype(str).str.upper()
    pv_share = float((values == "PV").mean() * 100)
    lcv_share = float(values.isin(["LCV", "LV"]).mean() * 100)
    return pv_share, lcv_share


def detect_fuel_column(df: pd.DataFrame) -> str | None:
    return "POWER_CATEGORY_2"


def kpi7_fuel_by_period(
    df: pd.DataFrame,
    country: str,
    status_group: str,
    metric_mode: str,
    period_mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    subset = df.copy()

    if country != "ALL":
        subset = subset[subset["COUNTRY"] == country]

    if start_date and end_date:
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(start) and pd.notna(end):
            if start > end:
                start, end = end, start
            subset = subset[(subset["COB_DATE"] >= start) & (subset["COB_DATE"] <= end)]

    subset = filter_status_group(subset, status_group)
    fuel_col = "POWER_CATEGORY_2"

    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.dropna(subset=[fuel_col, "MONTH", "YEAR"])
    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.copy()
    subset[fuel_col] = subset[fuel_col].astype(str).str.upper()
    subset["COB_DATE"] = pd.to_datetime(subset["COB_DATE"], errors="coerce")
    subset = subset.dropna(subset=["COB_DATE"])

    if subset.empty:
        return pd.DataFrame(), "", "", ""

    if period_mode == "monthly":
        periods = subset["COB_DATE"].dt.to_period("M")
        subset["PERIOD_SORT"] = periods.astype(str)
        subset["PERIOD_LABEL"] = subset["COB_DATE"].dt.strftime("%Y-%m")
        x_title = "Month"
        period_label = "Monthly"
    elif period_mode == "quarterly":
        periods = subset["COB_DATE"].dt.to_period("Q")
        subset["PERIOD_SORT"] = periods.astype(str)
        subset["PERIOD_LABEL"] = periods.astype(str)
        x_title = "Quarter"
        period_label = "Quarterly"
    else:
        subset["PERIOD_SORT"] = subset["COB_DATE"].dt.year
        subset["PERIOD_LABEL"] = subset["COB_DATE"].dt.year.astype(str)
        x_title = "Year"
        period_label = "Yearly"

    grouped = subset.groupby([fuel_col, "PERIOD_SORT", "PERIOD_LABEL"]).size().reset_index(name="VOLUME")
    if grouped.empty:
        return pd.DataFrame(), "", "", ""

    if metric_mode.lower() == "share":
        grouped["METRIC"] = grouped.groupby(["PERIOD_SORT", "PERIOD_LABEL"])["VOLUME"].transform(lambda x: x / x.sum() * 100)
        y_title = "Share (%)"
    else:
        grouped["METRIC"] = grouped["VOLUME"]
        y_title = "Volume"

    pivot_multi = grouped.pivot(index=fuel_col, columns=["PERIOD_SORT", "PERIOD_LABEL"], values="METRIC").fillna(0)
    ordered_columns = sorted(pivot_multi.columns.tolist(), key=lambda c: c[0])
    pivot = pivot_multi[ordered_columns]
    pivot.columns = [label for _, label in ordered_columns]

    fuel_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[fuel_order]

    return pivot.round(2), y_title, x_title, period_label


def summary_month_label(year: int | str, month: int | None) -> str:
    if month is None:
        return str(year)
    return f"{calendar.month_abbr[int(month)]}-{int(year)}"


def resolve_month_value(country: str, year: int | str, month_value: int | str) -> int:
    if month_value in (None, "ALL"):
        resolved_month = latest_month(df, country, year)
        if resolved_month is not None:
            return int(resolved_month)
        return 1
    return int(month_value)


def kpi_limit_status(value: float | None, limit: float | None) -> str:
    if value is None or pd.isna(value) or limit is None or pd.isna(limit):
        return "neutral"
    if value >= limit:
        return "red"
    if value >= limit * 0.8:
        return "orange"
    return "green"


def kpi_limit_color(status: str) -> str:
    if status == "red":
        return "#d64545"
    if status == "orange":
        return "#f0a202"
    if status == "green":
        return "#2f855a"
    return "#9aa5b1"


def build_kpi_result_cell(value: float | None, limit: float | None) -> html.Div:
    status = kpi_limit_status(value, limit)
    dot = html.Span("●", style={"color": kpi_limit_color(status), "fontSize": "18px", "marginRight": "8px"})
    text = percent_or_na(value)
    return html.Div([dot, html.Span(text)])


def build_kpi_text_cell(text: str) -> html.Div:
    dot = html.Span("●", style={"color": "#9aa5b1", "fontSize": "18px", "marginRight": "8px"})
    return html.Div([dot, html.Span(text)])


def render_kpi_summary_table(rows: list[dict[str, object]]) -> dash_table.DataTable:
    table_rows = []
    row_styles: list[dict[str, object]] = [{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}]
    for row in rows:
        result_text = row["result_text"]
        signal = str(row["signal"])
        if signal == "green":
            row_styles.append({"if": {"row_index": len(table_rows), "column_id": "Result"}, "color": "#2f855a"})
        elif signal == "orange":
            row_styles.append({"if": {"row_index": len(table_rows), "column_id": "Result"}, "color": "#f0a202"})
        elif signal == "red":
            row_styles.append({"if": {"row_index": len(table_rows), "column_id": "Result"}, "color": "#d64545"})

        table_rows.append(
            {
                "Asset Risk, Financed Fleet": str(row["label"]),
                "Period": str(row["period"]),
                "Result": str(result_text),
                "Volume": int(cast(Any, row["volume"])),
                "Unit": str(row["unit"]),
                "Comment": str(row["comment"]),
            }
        )

    columns = [
        {"name": "Asset Risk, Financed Fleet", "id": "Asset Risk, Financed Fleet"},
        {"name": "Period", "id": "Period"},
        {"name": "Result", "id": "Result"},
        {"name": "Volume", "id": "Volume"},
        {"name": "Unit", "id": "Unit"},
        {"name": "Comment", "id": "Comment"},
    ]

    return dash_table.DataTable(
        data=cast(Any, table_rows),
        columns=cast(Any, columns),
        sort_action="native",
        filter_action="native",
        page_size=6,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#102a43", "color": "white", "fontWeight": "700", "border": "none"},
        style_cell={
            "padding": "8px 10px",
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": "13px",
            "border": "1px solid #e6eaf0",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "110px",
        },
        style_data_conditional=cast(Any, row_styles),
    )


def kpi7_fuel_share_quarter(df: pd.DataFrame, year: int | str, status: str = "IN FLEET") -> pd.DataFrame:
    """Retourne la part de carburant par trimestre."""
    subset = df[
        (df["NOVA_ASSET_STATUS"].str.contains(status, case=False, na=False))
        & (df["YEAR"] == int(year))
    ].copy()
    
    subset = subset.dropna(subset=["POWER_CATEGORY_2"])
    
    if subset.empty:
        return pd.DataFrame()
    
    subset["Quarter"] = "Q" + (((subset["MONTH"] - 1) // 3) + 1).astype(str)
    
    grouped = (
        subset.groupby(["POWER_CATEGORY_2", "Quarter"])
        ["VEHICLE_ID"]
        .count()
        .reset_index(name="VOLUME")
    )
    
    pivot = grouped.pivot(index="POWER_CATEGORY_2", columns="Quarter", values="VOLUME").fillna(0)
    share = pivot.div(pivot.sum(axis=0), axis=1) * 100
    share = share.loc[share.sum(axis=1).sort_values(ascending=False).index]
    
    return share.round(2)


def kpi8_volume_by_power(df: pd.DataFrame, year: int | str, status: str = "IN FLEET", country: str = "ALL") -> pd.DataFrame:
    """Retourne le volume de production par carburant (YEAR, MONTH, POWER_CATEGORY)."""
    subset = df[
        (df["NOVA_ASSET_STATUS"].str.contains(status, case=False, na=False))
        & (df["YEAR"] == int(year))
    ].copy()
    
    if country != "ALL":
        subset = subset[subset["COUNTRY"] == country]
    
    grouped = (
        subset.groupby(["YEAR", "MONTH", "POWER_CATEGORY"])
        .size()
        .reset_index(name="VOLUME")
    )
    
    table = (
        grouped.pivot(index=["YEAR", "MONTH"], columns="POWER_CATEGORY", values="VOLUME")
        .fillna(0)
        .reset_index()
        .sort_values(["YEAR", "MONTH"])
    )   
    
    return table


def build_card(title: str, main_value: str, subtitle: str, accent: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="kpi-card-title"),
            html.Div(main_value, className="kpi-card-value"),
            html.Div(subtitle, className="kpi-card-subtitle"),
        ],
        style={"borderTop": f"4px solid {accent}"},
        className="kpi-card",
    )


def build_card_body(main_value: str, subtitle: str, accent: str) -> html.Div:
    return html.Div(
        [
            html.Div(main_value, className="kpi-card-value"),
            html.Div(subtitle, className="kpi-card-subtitle"),
        ],
        style={"borderTop": f"4px solid {accent}"},
        className="kpi-card",
    )


def figure_from_pivot(pivot: pd.DataFrame, y_title: str, x_title: str, title: str) -> go.Figure:
    fig = go.Figure()
    if pivot.empty:
        fig.update_layout(title=title)
        return fig

    x_values = pivot.columns.tolist()
    for fuel in pivot.index:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=pivot.loc[fuel].tolist(),
                name=str(fuel),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=pivot.loc[fuel].tolist(),
                mode="lines+markers",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="Fuel type",
        height=520,
        template="plotly_white",
    )
    return fig


def detect_vehicle_type_column(df: pd.DataFrame) -> str | None:
    return pick_first_existing_column(df, ["VEHICLE_TYPE_TEMP", "CLS_VEHICLE_TYPE"])


def build_table(df_table: pd.DataFrame, page_size: int = 15) -> dash_table.DataTable:
    columns = cast(Any, [{"name": str(c), "id": str(c)} for c in df_table.columns])
    data = cast(Any, df_table.to_dict("records"))
    row_style = cast(Any, [{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}])
    return dash_table.DataTable(
        data=data,
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#102a43", "color": "white", "fontWeight": "700", "border": "none"},
        style_cell={
            "padding": "8px 10px",
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": "13px",
            "border": "1px solid #e6eaf0",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "110px",
        },
        style_data_conditional=row_style,
    )


def reorder_concentration_columns(df_table: pd.DataFrame, metric_mode: str = "share") -> pd.DataFrame:
    if df_table.empty:
        return df_table

    metric_suffix = "SHARE" if metric_mode == "share" else "VOLUME"
    metric_cols = sorted([c for c in df_table.columns if c.endswith(f"_{metric_suffix}")])
    var_cols = sorted([c for c in df_table.columns if c.endswith("_VAR")])
    base_cols = [c for c in ["COUNTRY"] if c in df_table.columns]
    ordered_cols = base_cols + metric_cols + var_cols
    return df_table[[c for c in ordered_cols if c in df_table.columns]].copy()


def should_show_total_column(table_df: pd.DataFrame, metric_cols: list[str], share_mode: bool) -> bool:
    if not metric_cols:
        return False
    if not share_mode:
        return True

    row_totals = table_df[metric_cols].sum(axis=1)
    if row_totals.empty:
        return False

    return not row_totals.between(99.9, 100.1).all()


def html_report_document(title: str, sections: list[str]) -> str:
        body = "\n".join(sections)
        return f"""<!doctype html>
<html lang=\"fr\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #102a43; background: #f7f9fc; }}
        .page {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 28px; border: 1px solid #e6eaf0; border-radius: 12px; }}
        h1, h2, h3 {{ color: #102a43; }}
        .meta {{ background: #f1f5f9; padding: 16px; border-radius: 10px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
        th {{ background: #102a43; color: #fff; }}
        .section {{ margin-bottom: 28px; }}
        .muted {{ color: #627d98; }}
    </style>
</head>
<body>
    <div class=\"page\">
        <h1>{title}</h1>
        {body}
    </div>
</body>
</html>"""


def build_kpi_params_list(params: list[tuple[str, object]]) -> str:
        items = "".join(f"<li><strong>{name}:</strong> {value}</li>" for name, value in params)
        return f"<div class=\"meta section\"><h2>Paramètres</h2><ul>{items}</ul></div>"


def build_html_table_from_df(df_table: pd.DataFrame, title: str) -> str:
        if df_table.empty:
                return f"<div class=\"section\"><h2>{title}</h2><p class=\"muted\">Aucune donnée disponible.</p></div>"
        return f"<div class=\"section\"><h2>{title}</h2>{df_table.to_html(index=False, escape=False)}</div>"


def figure_to_html_block(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})


def build_view1_download_report(
    country: str,
    year: int | str,
    month_value: int | str,
    kpi1_country: str,
    kpi1_year: int | str,
    kpi1_month: int | str,
    kpi2_country: str,
    kpi2_year: int | str,
    kpi2_month: int | str,
    kpi3_country: str,
    kpi3_year: int | str,
    kpi3_month: int | str,
    kpi4_country: str,
    kpi4_year: int | str,
    kpi4_month: int | str,
    kpi5_country: str,
    kpi5_year: int | str,
    kpi5_month: int | str,
    kpi6_country: str,
    kpi6_year: int | str,
    kpi6_month: int | str,
    kpi1_limit: float | None,
    kpi2_limit: float | None,
    kpi7_country: str,
    kpi7_start_date: str | None,
    kpi7_end_date: str | None,
    kpi7_status_group: str,
    kpi7_metric_mode: str,
    kpi7_period_mode: str,
) -> str:
    month3 = resolve_month_value(kpi3_country, kpi3_year, kpi3_month)
    month4 = resolve_month_value(kpi4_country, kpi4_year, kpi4_month)
    month5 = resolve_month_value(kpi5_country, kpi5_year, kpi5_month)
    month6 = resolve_month_value(kpi6_country, kpi6_year, kpi6_month)

    kpi1_val = kpi_lease_under_25(df, kpi1_country, kpi1_year, kpi1_month)
    kpi2_val = kpi_lease_25_30(df, kpi2_country, kpi2_year, kpi2_month)
    diesel_non = kpi_diesel_non_diesel(df, kpi3_country, kpi3_year, month3)
    hybrid_val = kpi_hybrid_share(df, kpi4_country, kpi4_year, month4)
    ev_val = kpi_ev_share(df, kpi5_country, kpi5_year, month5)
    pv_lcv_val = kpi_pv_lcv(df, kpi6_country, kpi6_year, month6)

    kpi7_pivot, y_title7, x_title7, period_label7 = kpi7_fuel_by_period(
        df,
        kpi7_country,
        kpi7_status_group,
        kpi7_metric_mode,
        kpi7_period_mode,
        kpi7_start_date,
        kpi7_end_date,
    )
    kpi7_fig = figure_from_pivot(
        kpi7_pivot,
        y_title7,
        x_title7,
        f"KPI 7 - {('Share' if kpi7_metric_mode == 'share' else 'Volume')} of fuel type ({period_label7}, {kpi7_country})",
    )

    kpi_rows = [
        {
            "label": "LTR < 25 (production)",
            "period": f"{kpi1_country} / {summary_month_label(kpi1_year, None if kpi1_month in (None, 'ALL') else int(kpi1_month))}",
            "result": percent_or_na(kpi1_val),
            "volume": len(df[(df["COUNTRY"] == kpi1_country) & (df["YEAR"] == int(kpi1_year)) & (df["NOVA_ASSET_STATUS"] == 'IN FLEET') & (df["BIKE_OR_CAR"] == 'CAR')]),
        },
        {
            "label": "25m <= LTR <= 30m (production)",
            "period": f"{kpi2_country} / {summary_month_label(kpi2_year, None if kpi2_month in (None, 'ALL') else int(kpi2_month))}",
            "result": percent_or_na(kpi2_val),
            "volume": len(df[(df["COUNTRY"] == kpi2_country) & (df["YEAR"] == int(kpi2_year)) & (df["NOVA_ASSET_STATUS"] == 'IN FLEET') & (df["BIKE_OR_CAR"] == 'CAR')]),
        },
        {
            "label": "DI vs Non-DI",
            "period": f"{kpi3_country} / {summary_month_label(kpi3_year, month3)}",
            "result": f"{percent_or_na(diesel_non[0])} DI & {percent_or_na(diesel_non[1])} Non DI" if diesel_non[0] is not None else "N/A",
            "volume": len(df[(df["COUNTRY"] == kpi3_country) & (df["YEAR"] == int(kpi3_year)) & (df["MONTH"] == int(month3)) & (df["NOVA_ASSET_STATUS"] == 'IN FLEET') & (df["BIKE_OR_CAR"] == 'CAR')]),
        },
        {
            "label": "Hybrid share",
            "period": f"{kpi4_country} / {summary_month_label(kpi4_year, month4)}",
            "result": percent_or_na(hybrid_val),
            "volume": len(df[(df["COUNTRY"] == kpi4_country) & (df["YEAR"] == int(kpi4_year)) & (df["MONTH"] == int(month4)) & (df["NOVA_ASSET_STATUS"] == 'IN FLEET') & (df["BIKE_OR_CAR"] == 'CAR')]),
        },
        {
            "label": "EV share",
            "period": f"{kpi5_country} / {summary_month_label(kpi5_year, month5)}",
            "result": percent_or_na(ev_val),
            "volume": len(df[(df["COUNTRY"] == kpi5_country) & (df["YEAR"] == int(kpi5_year)) & (df["MONTH"] == int(month5)) & (df["NOVA_ASSET_STATUS"] == 'IN FLEET') & (df["BIKE_OR_CAR"] == 'CAR')]),
        },
        {
            "label": "PC vs LCV",
            "period": f"{kpi6_country} / {summary_month_label(kpi6_year, month6)}",
            "result": f"{percent_or_na(pv_lcv_val[0])} PV & {percent_or_na(pv_lcv_val[1])} LCV" if pv_lcv_val[0] is not None else "N/A",
            "volume": len(df[(df["COUNTRY"] == kpi6_country) & (df["YEAR"] == int(kpi6_year)) & (df["MONTH"] == int(month6)) & (df["NOVA_ASSET_STATUS"] == 'IN FLEET') & (df["BIKE_OR_CAR"] == 'CAR')]),
        },
    ]
    summary_df = pd.DataFrame(kpi_rows)

    params = [
        ("Country", country),
        ("Year", year),
        ("Month", month_value),
        ("KPI 1 country", kpi1_country),
        ("KPI 1 year", kpi1_year),
        ("KPI 1 month", kpi1_month),
        ("KPI 2 country", kpi2_country),
        ("KPI 2 year", kpi2_year),
        ("KPI 2 month", kpi2_month),
        ("KPI 3 country", kpi3_country),
        ("KPI 3 year", kpi3_year),
        ("KPI 3 month", kpi3_month),
        ("KPI 4 country", kpi4_country),
        ("KPI 4 year", kpi4_year),
        ("KPI 4 month", kpi4_month),
        ("KPI 5 country", kpi5_country),
        ("KPI 5 year", kpi5_year),
        ("KPI 5 month", kpi5_month),
        ("KPI 6 country", kpi6_country),
        ("KPI 6 year", kpi6_year),
        ("KPI 6 month", kpi6_month),
        ("KPI 1 limit", kpi1_limit),
        ("KPI 2 limit", kpi2_limit),
        ("KPI 7 country", kpi7_country),
        ("KPI 7 start date", kpi7_start_date),
        ("KPI 7 end date", kpi7_end_date),
        ("KPI 7 status", kpi7_status_group),
        ("KPI 7 metric", kpi7_metric_mode),
        ("KPI 7 period", kpi7_period_mode),
    ]

    return html_report_document(
        "Fleet Monitoring - Vue 1",
        [
            build_kpi_params_list(params),
            build_html_table_from_df(summary_df, "KPI summary"),
            f"<div class=\"section\"><h2>KPI 7</h2>{figure_to_html_block(kpi7_fig)}</div>",
            build_html_table_from_df(kpi7_pivot.reset_index(), "KPI 7 pivot"),
        ],
    )


def build_view2_download_report(
    country: str,
    year: int | str,
    asset_status: str,
    metric_mode: str,
) -> str:
    table_kpi8, y_title, x_title, period_label = kpi8_production_ytd(df, country, year, asset_status, metric_mode)

    fig = go.Figure()
    if not table_kpi8.empty:
        x_values = table_kpi8["MONTH"].tolist()
        metric_cols = [c for c in table_kpi8.columns if c not in ["YEAR", "MONTH"]]
        for col in metric_cols:
            fig.add_trace(go.Scatter(x=x_values, y=table_kpi8[col].tolist(), mode="lines+markers", name=str(col)))
    fig.update_layout(title=f"KPI 8 - {('Share' if metric_mode == 'share' else 'Volume')} per vehicule category ({period_label})", xaxis_title=x_title, yaxis_title=y_title, template="plotly_white", height=520)

    params = [
        ("Country", country),
        ("Year", year),
        ("Status", asset_status),
        ("Metric", metric_mode),
    ]

    return html_report_document(
        "Fleet Monitoring - Vue 2",
        [
            build_kpi_params_list(params),
            f"<div class=\"section\"><h2>KPI 8 graph</h2>{figure_to_html_block(fig)}</div>",
            build_html_table_from_df(table_kpi8, "KPI 8 table"),
        ],
    )


def kpi8_production_ytd(df: pd.DataFrame, country: str, year: int | str, asset_status: str, metric_mode: str) -> tuple[pd.DataFrame, str, str, str]:
    subset = filter_base(df, country, year, status="ALL", bike_or_car="CAR")
    subset = apply_status_filter(subset, asset_status)

    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY_3", "POWER_CATEGORY_2", "POWER_CATEGORY"])
    if power_col is None:
        return pd.DataFrame(), "", "", ""

    subset = subset.copy()
    start_dates = pd.to_datetime(subset["CONTRACT_START_DATE"], errors="coerce")
    start_subset = subset[start_dates.dt.year == int(year)].copy()
    if not start_subset.empty:
        start_subset["MONTH"] = start_dates.loc[start_subset.index].dt.month
        subset = start_subset

    subset = subset.dropna(subset=[power_col, "MONTH"])
    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset[power_col] = subset[power_col].astype(str).str.upper()
    subset["MONTH"] = subset["MONTH"].astype(int)
    subset["YEAR"] = int(year)

    grouped = subset.groupby(["YEAR", "MONTH", power_col]).size().reset_index(name="VOLUME")
    if grouped.empty:
        return pd.DataFrame(), "", "", ""

    table = (
        grouped.pivot(index=["YEAR", "MONTH"], columns=power_col, values="VOLUME")
        .fillna(0)
        .reset_index()
        .sort_values(["YEAR", "MONTH"])
    )

    metric_cols = [c for c in table.columns if c not in ["YEAR", "MONTH"]]
    if not metric_cols:
        return pd.DataFrame(), "", "", ""

    if metric_mode.lower() == "share":
        totals = table[metric_cols].sum(axis=1)
        table[metric_cols] = table[metric_cols].div(totals.replace(0, 1), axis=0) * 100
        y_title = "Share (%)"
    else:
        y_title = "Volume"

    return table.round(2), y_title, "Month", f"XTD {year}"


def kpi9_1_vehicle_share_quarter(df: pd.DataFrame, country: str, year: int | str, status_value: str, vehicle_type: str) -> pd.DataFrame:
    if "VEHICLE_TYPE_TEMP" not in df.columns:
        return pd.DataFrame()

    subset = filter_base(df, country, year, status="ALL", bike_or_car="CAR")
    subset = subset[
        subset["NOVA_ASSET_STATUS"].astype(str).str.contains("IN FLEET|ORDER", case=False, na=False)
    ]
    subset = apply_status_filter(subset, status_value)
    subset = subset.dropna(subset=["MONTH", "VEHICLE_TYPE_TEMP"])
    if subset.empty:
        return pd.DataFrame()

    subset = subset.copy()
    subset["MONTH"] = subset["MONTH"].astype(int)
    subset["Quarter"] = "Q" + (((subset["MONTH"] - 1) // 3) + 1).astype(str)
    subset["VEHICLE_TYPE_TEMP"] = subset["VEHICLE_TYPE_TEMP"].astype(str).str.upper()

    if vehicle_type != "ALL":
        subset = subset[subset["VEHICLE_TYPE_TEMP"] == vehicle_type.upper()]
    if subset.empty:
        return pd.DataFrame()

    grouped = subset.groupby(["NOVA_ASSET_STATUS", "Quarter"])["VEHICLE_ID"].count().reset_index()
    pivot = grouped.pivot(index="NOVA_ASSET_STATUS", columns="Quarter", values="VEHICLE_ID").fillna(0)
    share = pivot.div(pivot.sum(axis=0), axis=1) * 100
    return share.round(2)


def kpi9_2_vehicle_energy_share_quarter(df: pd.DataFrame, country: str, year: int | str, status_value: str, vehicle_type: str) -> tuple[pd.DataFrame, str, str, str]:
    if "VEHICLE_TYPE_TEMP" not in df.columns:
        return pd.DataFrame(), "", "", ""

    subset = filter_base(df, country, year, status="ALL", bike_or_car="CAR")
    subset = apply_status_filter(subset, status_value)
    subset = subset.dropna(subset=["MONTH", "VEHICLE_TYPE_TEMP", "POWER_CATEGORY"])
    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.copy()
    subset["VEHICLE_TYPE_TEMP"] = subset["VEHICLE_TYPE_TEMP"].astype(str).str.upper()
    if vehicle_type != "ALL":
        subset = subset[subset["VEHICLE_TYPE_TEMP"] == vehicle_type.upper()]
    if subset.empty:
        return pd.DataFrame(), "", "", ""

    valid_power = ["DIESEL", "PETROL", "FULL HYBRID", "PLUG-IN HYBRID", "ELECTRIC"]
    subset["POWER_CATEGORY"] = subset["POWER_CATEGORY"].where(subset["POWER_CATEGORY"].isin(valid_power), "Others")
    subset["MONTH"] = subset["MONTH"].astype(int)
    subset["Quarter"] = "Q" + (((subset["MONTH"] - 1) // 3) + 1).astype(str)

    grouped = subset.groupby(["POWER_CATEGORY", "Quarter"])["VEHICLE_ID"].count().reset_index()
    pivot = grouped.pivot(index="Quarter", columns="POWER_CATEGORY", values="VEHICLE_ID").fillna(0)
    return pivot, "Volume", "Quarter", f"{vehicle_type} by energy"


def kpi_count_share_quarterly(df: pd.DataFrame, asset_status: str, var_col: str, bike_or_car: str = "CAR") -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ]
    out = out.dropna(subset=["MONTH", "YEAR", var_col, "VEHICLE_ID"])
    out = out.drop_duplicates(subset=["VEHICLE_ID"])
    if out.empty:
        return pd.DataFrame()

    out = out.copy()
    out["Quarter"] = "Q" + (((out["MONTH"].astype(int) - 1) // 3) + 1).astype(str)

    grouped = out.groupby(["COUNTRY", var_col, "YEAR", "Quarter"])["VEHICLE_ID"].count().reset_index(name="VOLUME")
    grouped["TOTAL"] = grouped.groupby(["COUNTRY", "YEAR", "Quarter"])["VOLUME"].transform("sum")
    grouped["SHARE"] = grouped["VOLUME"].div(grouped["TOTAL"].where(grouped["TOTAL"] != 0, 1)).mul(100)
    return grouped


def kpi_count_share_monthly(df: pd.DataFrame, asset_status: str, var_col: str, bike_or_car: str = "CAR") -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ]
    out = out.dropna(subset=["MONTH", "YEAR", var_col, "VEHICLE_ID"])
    out = out.drop_duplicates(subset=["VEHICLE_ID"])
    if out.empty:
        return pd.DataFrame()

    grouped = out.groupby([var_col, "YEAR", "MONTH"])["VEHICLE_ID"].count().reset_index(name="VOLUME")
    grouped["TOTAL"] = grouped.groupby(["YEAR", "MONTH"])["VOLUME"].transform("sum")
    grouped["SHARE"] = grouped["VOLUME"].div(grouped["TOTAL"].where(grouped["TOTAL"] != 0, 1)).mul(100)
    return grouped


def kpi_count_share_ytd_by_quarter(df: pd.DataFrame, asset_status: str, var_col: str, bike_or_car: str = "CAR") -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ]
    out = out.dropna(subset=["MONTH", "YEAR", var_col, "VEHICLE_ID"])

    if out.empty:
        return pd.DataFrame()

    out = out.copy()
    out["MONTH"] = out["MONTH"].astype(int)
    out["Quarter"] = "Q" + (((out["MONTH"] - 1) // 3) + 1).astype(str)
    out["VOLUME_MONTH"] = 1
    out = out.sort_values(["COUNTRY", var_col, "YEAR", "MONTH"])
    out["VOLUME_YTD"] = out.groupby(["COUNTRY", var_col, "YEAR"])["VOLUME_MONTH"].cumsum()

    grouped = (
        out.groupby(["COUNTRY", var_col, "YEAR", "Quarter"], as_index=False)
        .agg(VOLUME_YTD=("VOLUME_YTD", "max"))
    )
    grouped["TOTAL_YTD"] = grouped.groupby(["COUNTRY", "YEAR", "Quarter"])["VOLUME_YTD"].transform("sum")
    grouped["SHARE_YTD"] = grouped["VOLUME_YTD"].div(grouped["TOTAL_YTD"].where(grouped["TOTAL_YTD"] != 0, 1)).mul(100)
    return grouped


def plot_kpi_share(df_plot: pd.DataFrame, col: str) -> go.Figure:
    out = df_plot.copy()
    if out.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Market Share by {col} (Quarterly, all countries)")
        return fig

    out["TIME"] = out["YEAR"].astype(str) + "-" + out["Quarter"].astype(str)

    agg = out.groupby(["TIME", col], as_index=False)["VOLUME"].sum()
    total = agg.groupby("TIME", as_index=False)["VOLUME"].sum().rename(columns={"VOLUME": "TOTAL_VOLUME"})
    agg = agg.merge(total, on="TIME", how="left")
    agg["SHARE"] = agg["VOLUME"] / agg["TOTAL_VOLUME"] * 100

    rank = agg.groupby(col)["VOLUME"].sum().sort_values(ascending=False)
    cats = rank.index.tolist()
    if len(cats) > 5:
        cats = cats[:5]
        barmode = "group"
    else:
        barmode = "stack"

    agg = agg[agg[col].isin(cats)].copy()
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    for c in cats:
        d = agg[agg[col] == c]
        if d.empty:
            continue
        fig.add_trace(
            go.Bar(x=d["TIME"], y=d["SHARE"], name=str(c), legendgroup=str(c)),
            row=1,
            col=1,
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=total["TIME"],
            y=total["TOTAL_VOLUME"],
            mode="lines+markers",
            name="TOTAL",
            legendgroup="TOTAL",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        barmode=barmode,
        title=f"Market Share by {col} (Quarterly, all countries)",
        height=550,
        xaxis_title="Time",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Share (%)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    return fig


def plot_top_var_kpi(df_kpi: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Plot top variable per quarter with volume (bars) and share (lines overlay)."""
    fig = go.Figure()
    if df_kpi.empty:
        fig.update_layout(title=f"Top Variable Volume & Share per Quarter {title_suffix}")
        return fig

    volume_cols = sorted([c for c in df_kpi.columns if c.endswith("_VOLUME")])
    for _, row in df_kpi.iterrows():
        country = row.get("COUNTRY", "N/A")
        x_labels: list[str] = []
        volumes: list[float] = []
        shares: list[float] = []

        for col in volume_cols:
            var_col = col.replace("_VOLUME", "_VAR")
            share_col = col.replace("_VOLUME", "_SHARE")
            var_value = row.get(var_col, None)
            volume = row.get(col, 0)
            share = row.get(share_col, 0)
            if pd.isna(var_value) or pd.isna(volume) or float(volume) == 0:
                continue

            quarter_label = col.replace("_VOLUME", "")
            x_labels.append(f"{quarter_label}: {var_value}")
            volumes.append(float(volume))
            shares.append(float(share) if not pd.isna(share) else 0.0)

        if not x_labels:
            continue

        fig.add_trace(go.Bar(x=x_labels, y=volumes, name=f"{country} Volume", legendgroup=str(country)))
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=shares,
                mode="lines+markers",
                name=f"{country} Share",
                legendgroup=str(country),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=f"Top Variable Volume & Share per Quarter {title_suffix}",
        barmode="group",
        yaxis={"title": "Volume"},
        yaxis2={"title": "Share (%)", "overlaying": "y", "side": "right"},
        height=560,
        template="plotly_white",
    )
    return fig


def kpi_top_brand_vs_market(df_portfolio: pd.DataFrame, df_market: pd.DataFrame, var_col_portfolio: str, var_col_market: str, asset_status: str) -> pd.DataFrame:
    # Étape 1: Trouver les quarters disponibles en market
    df_market_temp = df_market.copy()
    df_market_temp["date"] = pd.to_datetime(df_market_temp["date"], errors="coerce")
    df_market_temp["QUARTER"] = df_market_temp["date"].dt.to_period('Q').astype(str)
    available_quarters = set(df_market_temp["QUARTER"].unique())
    
    # Étape 2: Filtrer portfolio sur ces quarters uniquement
    df_portfolio = df_portfolio.copy()
    df_portfolio["COB_DATE"] = pd.to_datetime(df_portfolio["COB_DATE"], errors="coerce")
    df_portfolio["QUARTER"] = df_portfolio["COB_DATE"].dt.to_period('Q').astype(str)
    df_portfolio = df_portfolio[df_portfolio["QUARTER"].isin(available_quarters)]
    df_portfolio = apply_status_filter(df_portfolio, asset_status)

    port = (
        df_portfolio.groupby(["COUNTRY", "QUARTER", var_col_portfolio])["VEHICLE_ID"]
        .count()
        .reset_index(name="volume_portfolio")
    )
    port["total_portfolio"] = port.groupby(["COUNTRY", "QUARTER"])["volume_portfolio"].transform("sum")
    port["share_portfolio"] = port["volume_portfolio"] / port["total_portfolio"]
    port = port.rename(columns={var_col_portfolio: "BRAND"})
    port["QUARTER"] = port["QUARTER"].astype(str)

    port_top = (
        port.sort_values("volume_portfolio", ascending=False)
        .groupby(["COUNTRY", "QUARTER"])
        .head(1)
    )

    df_market["QUARTER"] = df_market["date"].dt.to_period('Q').astype(str)
    df_market[var_col_market] = df_market[var_col_market].str.upper()

    market = (
        df_market.groupby(["Country/Territory-Number", "QUARTER", var_col_market])["volume"]
        .sum()
        .reset_index(name="volume_market")
    )
    market["total_market"] = market.groupby(["Country/Territory-Number", "QUARTER"])["volume_market"].transform("sum")
    market["share_market"] = market["volume_market"] / market["total_market"]
    
    market = market.rename(columns={
        "Country/Territory-Number": "COUNTRY",
        var_col_market: "BRAND"
    })

    df_final = port_top.merge(
        market, on=["COUNTRY", "QUARTER", "BRAND"], how="left"
    )

    df_final["ratio"] = df_final["total_portfolio"] / df_final["total_market"]
    df_final = df_final.sort_values(['COUNTRY', 'QUARTER'])

    return df_final


def figure_top_brand_vs_market(df_kpi13: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if df_kpi13.empty:
        fig.update_layout(title=title, template="plotly_white", height=500)
        return fig

    agg = (
        df_kpi13.groupby("QUARTER", as_index=False)
        .agg(
            portfolio_share=("share_portfolio", "mean"),
            market_share=("share_market", "mean"),
            portfolio_volume=("volume_portfolio", "sum"),
        )
        .sort_values("QUARTER")
    )

    fig.add_trace(go.Bar(x=agg["QUARTER"], y=agg["portfolio_share"], name="Portfolio share %"))
    fig.add_trace(go.Bar(x=agg["QUARTER"], y=agg["market_share"], name="Market share %"))
    fig.add_trace(
        go.Scatter(
            x=agg["QUARTER"],
            y=agg["portfolio_volume"],
            mode="lines+markers",
            name="Portfolio volume",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis={"title": "Share (%)"},
        yaxis2={"title": "Volume", "overlaying": "y", "side": "right"},
        xaxis_title="Quarter",
        template="plotly_white",
        height=520,
    )
    return fig


def kpi_top_per_quarter_with_share(
    df: pd.DataFrame,
    asset_status: str,
    var_col: str,
    bike_or_car: str = "CAR",
    metric_mode: str = "share",
) -> pd.DataFrame:
    """Compute top variable per quarter with volume and share. Returns rows per country."""
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ].dropna(subset=["COB_DATE", var_col, "COUNTRY", "VEHICLE_ID"])
    out = out.drop_duplicates(subset=["VEHICLE_ID"])
    if out.empty:
        return pd.DataFrame()

    out["YEAR"] = out["COB_DATE"].dt.year.astype(int)
    out["Q"] = "Q" + (((out["COB_DATE"].dt.month - 1) // 3) + 1).astype(str)
    out["PERIOD"] = out["YEAR"].astype(str) + "_" + out["Q"]

    rows: list[dict] = []
    for country in sorted(out["COUNTRY"].dropna().unique().tolist()):
        country_df = out[out["COUNTRY"] == country]
        row_data: dict[str, str | int | float | None] = {"COUNTRY": country}
        for period in sorted(country_df["PERIOD"].dropna().unique().tolist()):
            period_df = country_df[country_df["PERIOD"] == period]
            counts = period_df.groupby(var_col)["VEHICLE_ID"].count()
            total_volume = int(counts.sum())

            if counts.empty or total_volume == 0:
                top_var = None
                top_volume = 0
                top_share = 0.0
            else:
                top_var = str(counts.idxmax())
                top_volume = int(counts.max())
                top_share = round((top_volume / total_volume) * 100, 2)

            row_data[f"{period}_VAR"] = top_var
            row_data[f"{period}_VOLUME"] = top_volume
            row_data[f"{period}_SHARE"] = top_share
        rows.append(row_data)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    volume_cols = [c for c in out_df.columns if c.endswith("_VOLUME")]
    share_cols = [c for c in out_df.columns if c.endswith("_SHARE")]
    out_df[volume_cols] = out_df[volume_cols].fillna(0)
    out_df[share_cols] = out_df[share_cols].fillna(0.0)
    return reorder_concentration_columns(out_df, metric_mode=metric_mode)


def kpi_top_per_quarter_with_share_market(df: pd.DataFrame, var_col: str, metric_mode: str = "share") -> pd.DataFrame:
    out = df.copy()
    if out.empty or var_col not in out.columns:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "Country/Territory-Number", var_col, "volume"])
    if out.empty:
        return pd.DataFrame()

    out["YEAR"] = out["date"].dt.year
    out["QUARTER"] = out["date"].dt.to_period("Q").astype(str)

    results: list[dict] = []
    countries_local = out["Country/Territory-Number"].dropna().unique().tolist()
    for country in countries_local:
        country_df = out[out["Country/Territory-Number"] == country]
        row_data: dict[str, str | float | int | None] = {"COUNTRY": country}

        for year in sorted(country_df["YEAR"].dropna().unique().tolist()):
            year_df = country_df[country_df["YEAR"] == year]
            for q, df_q in year_df.groupby("QUARTER"):
                counts = df_q.groupby(var_col, as_index=False)["volume"].sum()
                total_volume = counts["volume"].sum()

                if counts.empty or total_volume == 0:
                    top_var = None
                    top_volume = 0.0
                    top_share = 0.0
                else:
                    top_row = counts.sort_values("volume", ascending=False).iloc[0]
                    top_var = top_row[var_col]
                    top_volume = float(top_row["volume"])
                    top_share = round((top_volume / total_volume) * 100, 2)

                row_data[f"{q}_VAR"] = top_var
                row_data[f"{q}_VOLUME"] = top_volume
                row_data[f"{q}_SHARE"] = top_share

        results.append(row_data)

    out_df = pd.DataFrame(results)
    if out_df.empty:
        return out_df

    return reorder_concentration_columns(out_df, metric_mode=metric_mode)


def co2_bucket_from_value(val: object) -> str:
    try:
        v = float(cast(Any, val))
        b = int(v) // 10 * 10
        return f"[{b}-{b + 9}]"
    except Exception:
        return "UNK"


def prepare_portfolio_concentration_source(df: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()

    if variable == "CO2_BUCKET":
        base_col = pick_first_existing_column(out, ["VA_CO2_EMSS_REAL"])
        out["CO2_BUCKET"] = out[base_col].apply(co2_bucket_from_value) if base_col else pd.NA
        return out, "CO2_BUCKET"

    if variable == "HIGHEST_BEV":
        if "POWER_CATEGORY" not in out.columns or "BRAND_UPDATE" not in out.columns:
            return out.iloc[0:0].copy(), "HIGHEST_BEV"
        out = out[out["POWER_CATEGORY"].astype(str).str.contains("ELECTRIC", case=False, na=False)].copy()
        out["HIGHEST_BEV"] = out["BRAND_UPDATE"]
        return out, "HIGHEST_BEV"

    return out, variable


def prepare_market_concentration_source(df: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if out.empty:
        return out, variable

    if variable == "BRAND_UPDATE":
        return out, "Make"

    if variable == "OEM_UPDATE":
        oem_col = pick_first_existing_column(out, ["Make Group", "Make"])
        return out, (oem_col if oem_col is not None else "Make")

    if variable == "CO2_BUCKET":
        base_col = pick_first_existing_column(out, ["VA_CO2_EMSS_REAL", "Engine (ccm)", "Engine (kw)"])
        out["CO2_BUCKET"] = out[base_col].apply(co2_bucket_from_value) if base_col else pd.NA
        return out, "CO2_BUCKET"

    if variable == "HIGHEST_BEV":
        if "Fuel Type" not in out.columns or "Make" not in out.columns:
            return out.iloc[0:0].copy(), "HIGHEST_BEV"
        out = out[out["Fuel Type"].astype(str).str.contains("ELECTRIC", case=False, na=False)].copy()
        out["HIGHEST_BEV"] = out["Make"]
        return out, "HIGHEST_BEV"

    return out, variable


CONCENTRATION_STATUS_OPTIONS = ["IN FLEET", "ORDER"]
CONCENTRATION_VARIABLE_OPTIONS = ["BRAND_UPDATE", "OEM_UPDATE", "POWER_CATEGORY"]
CONCENTRATION_SOURCE_OPTIONS = ["portfolio", "market"]
MARKET_STATUS_OPTIONS = ["IN FLEET", "ORDER"]
MARKET_VARIABLE_OPTIONS = ["BRAND", "OEM", "BEV"]





















df = load_dataset()
df = prepare_data_set(df)
market_df = load_market_dataset()

countries = sorted(df["COUNTRY"].dropna().unique().tolist())
years = sorted(df["YEAR"].dropna().astype(int).unique().tolist())
default_country = countries[0] if countries else None
default_year = years[-1] if years else None
default_month = latest_month(df, default_country, default_year) if default_country and default_year else None
default_month_value = f"{default_month:02d}" if default_month is not None else "ALL"
min_cob_date = df["COB_DATE"].min()
max_cob_date = df["COB_DATE"].max()
default_start_date = min_cob_date.date().isoformat() if pd.notna(min_cob_date) else None
default_end_date = max_cob_date.date().isoformat() if pd.notna(max_cob_date) else None
month_options = ["ALL"] + [f"{month:02d}" for month in range(1, 13)]
country_options = countries + ["ALL"]
year_options = years

statuses = sorted(df["NOVA_ASSET_STATUS"].dropna().astype(str).unique().tolist())
default_status_v2 = "IN FLEET" if "IN FLEET" in statuses else (statuses[0] if statuses else "ALL")

vehicle_col_global = detect_vehicle_type_column(df)
vehicle_types = sorted(df[vehicle_col_global].dropna().astype(str).str.upper().unique().tolist()) if vehicle_col_global else []
default_vehicle_type = "SUV" if "SUV" in vehicle_types else (vehicle_types[0] if vehicle_types else "ALL")
has_vehicle_type_temp = "VEHICLE_TYPE_TEMP" in df.columns



















app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(BASE_DIR / "fleet_assets"),
)
app.title = "Fleet Monitoring Dashboard"

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(
            [
                dcc.Link("Vue 1 - Main KPIs + KPI 7", href="/", className="nav-link"),
                dcc.Link("Vue 2 - KPI 8", href="/vue-2", className="nav-link"),
                dcc.Link("Vue 3 - SUV", href="/vue-3", className="nav-link") if has_vehicle_type_temp else html.Span(),
                dcc.Link("Vue 4 - Concentration Risk", href="/vue-4", className="nav-link"),
                dcc.Link("Vue 5 - Concentration", href="/vue-5", className="nav-link"),
                dcc.Link("Vue 6 - Concentration by Country", href="/vue-6", className="nav-link"),
                dcc.Link("Vue 7 - Portfolio vs Market", href="/vue-7", className="nav-link"),
            ],
            className="top-nav",
        ),
        html.Div(id="page-content"),
    ],
    className="page-wrap",
)


def view1_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Main kpis et KPI7."),
                    html.P("KPI 1-6 are in fleet, KPI 7 can switch fleet order and volume share."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Year", className="filter-label"),
                            dcc.Dropdown(id="year-filter", options=year_options, value=None, placeholder="Select year", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Month (KPI 3 to 6)", className="filter-label"),
                            dcc.Dropdown(id="month-filter", options=month_options, value=None, placeholder="Select month", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view1-refresh-button", n_clicks=0, className="primary-button"),
                            html.Button("Télécharger HTML", id="view1-download-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("KPI 1 Lease time < 25 months", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi1-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi1-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label"), dcc.Dropdown(id="kpi1-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-1-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("KPI 2 Lease time 25-30 months", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi2-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi2-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label"), dcc.Dropdown(id="kpi2-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-2-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("KPI 3 Diesel vs Non Diesel", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi3-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi3-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label"), dcc.Dropdown(id="kpi3-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-3-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("KPI 4 Hybrid Share", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi4-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi4-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label"), dcc.Dropdown(id="kpi4-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-4-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("KPI 5 EV Share", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi5-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi5-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label"), dcc.Dropdown(id="kpi5-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-5-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("KPI 6 PV vs LCV", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi6-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi6-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label"), dcc.Dropdown(id="kpi6-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-6-card"),
                        ],
                        className="kpi-card",
                    ),
                ],
                className="cards-grid",
            ),
            html.Div(
                "KPI 1 and KPI 2 use the selected country/year. KPI 3-6 use the selected month as well.",
                className="small-note",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("KPI summary table", className="panel-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("KPI 1 limit (%)", className="filter-label"),
                                            dcc.Input(id="summary-kpi1-limit-filter", type="number", value=5, min=0, step=0.1, className="numeric-input"),
                                        ],
                                        className="filter-box",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("KPI 2 limit (%)", className="filter-label"),
                                            dcc.Input(id="summary-kpi2-limit-filter", type="number", value=10, min=0, step=0.1, className="numeric-input"),
                                        ],
                                        className="filter-box",
                                    ),
                                ],
                                className="summary-controls",
                            ),
                            html.Div(id="kpi-summary-wrap"),
                        ]
                    )
                ],
                className="panel",
            ),
            html.Div(id="kpi-period-note", className="small-note"),
            html.Div(
                [
                    html.Div("KPI 7 - Fuel type", className="panel-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Country", className="filter-label"),
                                    dcc.Dropdown(id="kpi7-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Date range", className="filter-label"),
                                    dcc.DatePickerRange(
                                        id="kpi7-date-range",
                                        min_date_allowed=default_start_date,
                                        max_date_allowed=default_end_date,
                                        start_date=None,
                                        end_date=None,
                                        display_format="YYYY-MM-DD",
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Status", className="filter-label"),
                                    dcc.RadioItems(
                                        id="status-group-filter",
                                        options=[
                                                    {"label": "IN FLEET", "value": "IN FLEET"},
                                                    {"label": "ORDER", "value": "ORDER"},
                                        ],
                                        value=None,
                                        inline=True,
                                        labelStyle={"marginRight": "18px"},
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("View", className="filter-label"),
                                    dcc.RadioItems(
                                        id="metric-mode-filter",
                                        options=["share", "volume"],
                                        value=None,
                                        inline=True,
                                        labelStyle={"marginRight": "18px"},
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Period", className="filter-label"),
                                    dcc.RadioItems(
                                        id="period-mode-filter",
                                        options=[
                                            {"label": "Quarterly", "value": "quarterly"},
                                            {"label": "Monthly", "value": "monthly"},
                                            {"label": "Yearly", "value": "yearly"},
                                        ],
                                        value=None,
                                        inline=True,
                                        labelStyle={"marginRight": "18px"},
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Action", className="filter-label"),
                                    html.Button("Actualiser", id="view1-kpi7-refresh-button", n_clicks=0, className="primary-button"),
                                ],
                                className="filter-box",
                            ),
                        ],
                        className="controls-inline",
                    ),
                    dcc.Graph(id="kpi-7-graph", config={"displayModeBar": False}),
                    html.Div(id="kpi-7-table-wrap"),
                ],
                className="panel",
            ),
            dcc.Download(id="view1-html-download"),
        ]
    )


def view2_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Vue 2 - PA Order"),
                    html.P("KPI 8 Production volume and % per vehicule category (YTD)."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="v2-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Year", className="filter-label"),
                            dcc.Dropdown(id="v2-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Status", className="filter-label"),
                            dcc.Dropdown(
                                id="v2-status-filter",
                                options=["IN FLEET", "ORDER"],
                                value=None,
                                clearable=True,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("View", className="filter-label"),
                            dcc.RadioItems(
                                id="v2-metric-mode-filter",
                                options=["volume", "share"],
                                value=None,
                                inline=True,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view2-refresh-button", n_clicks=0, className="primary-button"),
                            html.Button("Télécharger HTML", id="view2-download-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 8 - Production volume and % per vehicule category (YTD)", className="panel-title"),
                    dcc.Graph(id="v2-kpi8-graph", config={"displayModeBar": False}),
                    html.Div(id="v2-kpi8-table-wrap"),
                ],
                className="panel",
            ),
            dcc.Download(id="view2-html-download"),
        ]
    )


def view3_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Vue 3 - SUV"),
                    html.P("KPI 9_1 SUV Share per quarter and KPI 9_2 SUV Share by energy per quarter."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="v3-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Year", className="filter-label"),
                            dcc.Dropdown(id="v3-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Status", className="filter-label"),
                            dcc.RadioItems(
                                id="v3-status-group-filter",
                                options=["IN FLEET", "ORDER"],
                                value=None,
                                inline=True,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Vehicle type", className="filter-label"),
                            dcc.Dropdown(
                                id="v3-vehicle-type-filter",
                                options=["ALL"] + vehicle_types,
                                value=None,
                                clearable=True,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view3-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 9_1 - SUV Share per quarter", className="panel-title"),
                    dcc.Graph(id="v3-kpi91-graph", config={"displayModeBar": False}),
                    html.Div(id="v3-kpi91-table-wrap"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("KPI 9_2 - SUV Share by energy per quarter", className="panel-title"),
                    dcc.Graph(id="v3-kpi92-graph", config={"displayModeBar": False}),
                    html.Div(id="v3-kpi92-table-wrap"),
                ],
                className="panel",
            ),
        ]
    )


def view4_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Vue 4 - Concentration Risk at EOC"),
                    html.P("KPI 10 - Top vehicle models at End of Contract by quarter."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="v4-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Status", className="filter-label"),
                            dcc.RadioItems(
                                id="v4-status-filter",
                                options=["IN FLEET", "ORDER"],
                                value=None,
                                inline=True,
                                labelStyle={"marginRight": "18px"},
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("EOC window", className="filter-label"),
                            dcc.Dropdown(
                                id="v4-eoc-window-filter",
                                options=["1", "3", "6", "12", "24", "ALL"],
                                value=None,
                                clearable=True,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view4-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 10 - EOC Model Concentration", className="panel-title"),
                    dcc.Graph(id="v4-kpi10-graph", config={"displayModeBar": False}),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("KPI 10 - Vehicle Models at EOC", className="panel-title"),
                    html.Div(id="v4-kpi10-table-wrap"),
                ],
                className="panel",
            ),
        ]
    )


def view5_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Vue 5 - Concentration"),
                    html.P("KPI 11 - Concentration by variable (global, in fleet)."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Status", className="filter-label"),
                            dcc.Dropdown(id="v5-status-filter", options=CONCENTRATION_STATUS_OPTIONS, value=None, placeholder="Select status", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Source", className="filter-label"),
                            dcc.Dropdown(id="v5-source-filter", options=CONCENTRATION_SOURCE_OPTIONS, value=None, placeholder="Select source", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Variable", className="filter-label"),
                            dcc.Dropdown(id="v5-variable-filter", options=CONCENTRATION_VARIABLE_OPTIONS, value=None, placeholder="Select variable", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view5-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 11 - Concentration (IN FLEET, global)", className="panel-title"),
                    dcc.Graph(id="v5-kpi11-graph", config={"displayModeBar": False}),
                    html.Div(id="v5-kpi11-table-wrap"),
                ],
                className="panel",
            ),
        ]
    )


def view6_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Vue 6 - Concentration by Country"),
                    html.P("Top variable per country and quarter with volume and share."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Status", className="filter-label"),
                            dcc.Dropdown(id="v6-status-filter", options=CONCENTRATION_STATUS_OPTIONS, value=None, placeholder="Select status", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Source", className="filter-label"),
                            dcc.Dropdown(id="v6-source-filter", options=CONCENTRATION_SOURCE_OPTIONS, value=None, placeholder="Select source", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Variable", className="filter-label"),
                            dcc.Dropdown(id="v6-variable-filter", options=CONCENTRATION_VARIABLE_OPTIONS, value=None, placeholder="Select variable", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Metric", className="filter-label"),
                            dcc.RadioItems(
                                id="v6-metric-mode-filter",
                                options=[{"label": "Share", "value": "share"}, {"label": "Volume", "value": "volume"}],
                                value=None,
                                inline=True,
                                labelStyle={"marginRight": "18px"},
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view6-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 6 - Top variable per country", className="panel-title"),
                    html.Div(id="v6-kpi6-table-wrap"),
                ],
                className="panel",
            ),
        ]
    )


def view7_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Vue 7 - Concentration Portfolio vs Market"),
                    html.P("KPI 13 Top BRAND/OEM/BEV portfolio share vs market share per quarter and country."),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Status", className="filter-label"),
                            dcc.Dropdown(id="v7-status-filter", options=MARKET_STATUS_OPTIONS, value=None, placeholder="Select status", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Variable", className="filter-label"),
                            dcc.Dropdown(id="v7-variable-filter", options=MARKET_VARIABLE_OPTIONS, value=None, placeholder="Select variable", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Actualiser", id="view7-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 13 - Portfolio share vs market share", className="panel-title"),
                    dcc.Graph(id="v7-kpi13-graph", config={"displayModeBar": False}),
                    html.Div(id="v7-kpi13-table-wrap"),
                ],
                className="panel",
            ),
        ]
    )


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname: str):
    if pathname == "/vue-2":
        return view2_layout()
    if pathname == "/vue-3":
        if not has_vehicle_type_temp:
            return view1_layout()
        return view3_layout()
    if pathname == "/vue-4":
        return view4_layout()
    if pathname == "/vue-5":
        return view5_layout()
    if pathname == "/vue-6":
        return view6_layout()
    if pathname == "/vue-7":
        return view7_layout()
    return view1_layout()


def has_missing_filters(*values: object) -> bool:
    return any(value is None for value in values)


@app.callback(
    Output("month-filter", "options"),
    Output("month-filter", "value"),
    Input("country-filter", "value"),
    Input("year-filter", "value"),
)
def update_month_filter(country: str, year: int | str):
    if has_missing_filters(country, year):
        return month_options, None
    months = available_months(df, country, year)
    options = ["ALL"] + [f"{month:02d}" for month in months]
    default_value = f"{months[-1]:02d}" if months else None
    return options, default_value


@app.callback(
    Output("kpi1-country-filter", "value"),
    Output("kpi1-year-filter", "value"),
    Output("kpi1-month-filter", "value"),
    Output("kpi2-country-filter", "value"),
    Output("kpi2-year-filter", "value"),
    Output("kpi2-month-filter", "value"),
    Output("kpi3-country-filter", "value"),
    Output("kpi3-year-filter", "value"),
    Output("kpi3-month-filter", "value"),
    Output("kpi4-country-filter", "value"),
    Output("kpi4-year-filter", "value"),
    Output("kpi4-month-filter", "value"),
    Output("kpi5-country-filter", "value"),
    Output("kpi5-year-filter", "value"),
    Output("kpi5-month-filter", "value"),
    Output("kpi6-country-filter", "value"),
    Output("kpi6-year-filter", "value"),
    Output("kpi6-month-filter", "value"),
    Input("country-filter", "value"),
    Input("year-filter", "value"),
    Input("month-filter", "value"),
    prevent_initial_call=True,
)
def sync_top_filters_to_cards(country: str, year: int | str, month_value: int | str):
    month = month_value
    return (
        country,
        year,
        month,
        country,
        year,
        month,
        country,
        year,
        month,
        country,
        year,
        month,
        country,
        year,
        month,
        country,
        year,
        month,
    )


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
    prevent_initial_call=True,
)
def update_kpi_cards(
    _refresh_clicks: int,
    kpi1_country: str,
    kpi1_year: int | str,
    kpi1_month: int | str,
    kpi2_country: str,
    kpi2_year: int | str,
    kpi2_month: int | str,
    kpi3_country: str,
    kpi3_year: int | str,
    kpi3_month: int | str,
    kpi4_country: str,
    kpi4_year: int | str,
    kpi4_month: int | str,
    kpi5_country: str,
    kpi5_year: int | str,
    kpi5_month: int | str,
    kpi6_country: str,
    kpi6_year: int | str,
    kpi6_month: int | str,
):
    required_filters = [
        kpi1_country, kpi1_year,
        kpi2_country, kpi2_year,
        kpi3_country, kpi3_year, kpi3_month,
        kpi4_country, kpi4_year, kpi4_month,
        kpi5_country, kpi5_year, kpi5_month,
        kpi6_country, kpi6_year, kpi6_month,
    ]
    if has_missing_filters(*required_filters):
        waiting_card = build_card_body("--", "Sélectionnez les filtres puis cliquez sur Actualiser.", "#9aa5b1")
        return waiting_card, waiting_card, waiting_card, waiting_card, waiting_card, waiting_card

    kpi1 = kpi_lease_under_25(df, kpi1_country, kpi1_year, kpi1_month)
    kpi2 = kpi_lease_25_30(df, kpi2_country, kpi2_year, kpi2_month)

    month3 = resolve_month_value(kpi3_country, kpi3_year, kpi3_month)
    month4 = resolve_month_value(kpi4_country, kpi4_year, kpi4_month)
    month5 = resolve_month_value(kpi5_country, kpi5_year, kpi5_month)
    month6 = resolve_month_value(kpi6_country, kpi6_year, kpi6_month)

    diesel_share, non_diesel_share = kpi_diesel_non_diesel(df, kpi3_country, kpi3_year, month3)
    hybrid_share = kpi_hybrid_share(df, kpi4_country, kpi4_year, month4)
    ev_share = kpi_ev_share(df, kpi5_country, kpi5_year, month5)
    pv_share, lcv_share = kpi_pv_lcv(df, kpi6_country, kpi6_year, month6)

    card1 = build_card_body(percent_or_na(kpi1), "Goal: no more than 5% of the current fleet.", "#1d5f99")
    card2 = build_card_body(percent_or_na(kpi2), "Goal: no more than 10% of the current fleet.", "#2f855a")
    card3 = build_card_body(f"{percent_or_na(diesel_share)} / {percent_or_na(non_diesel_share)}", "Selected month on the IN FLEET population.", "#b7791f")
    card4 = build_card_body(percent_or_na(hybrid_share), "HEV / PHEV share on the selected month.", "#7b4fe2")
    card5 = build_card_body(percent_or_na(ev_share), "Electric share on the selected month.", "#00a3a3")
    card6 = build_card_body(f"{percent_or_na(pv_share)} / {percent_or_na(lcv_share)}", "PV / LCV share on the selected month.", "#d64545")

    return card1, card2, card3, card4, card5, card6


@app.callback(
    Output("kpi-7-graph", "figure"),
    Output("kpi-7-table-wrap", "children"),
    Input("view1-refresh-button", "n_clicks"),
    Input("view1-kpi7-refresh-button", "n_clicks"),
    State("kpi7-country-filter", "value"),
    State("kpi7-date-range", "start_date"),
    State("kpi7-date-range", "end_date"),
    State("status-group-filter", "value"),
    State("metric-mode-filter", "value"),
    State("period-mode-filter", "value"),
    prevent_initial_call=True,
)
def update_kpi7(_refresh_clicks: int, _kpi7_refresh_clicks: int, country: str, start_date: str | None, end_date: str | None, status_group: str, metric_mode: str, period_mode: str):
    if has_missing_filters(country, status_group, metric_mode, period_mode):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 7 - Sélectionnez les filtres puis cliquez sur Actualiser")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    pivot, y_title, x_title, period_label = kpi7_fuel_by_period(df, country, status_group, metric_mode, period_mode, start_date, end_date)
    title_mode = "Share" if metric_mode == "share" else "Volume"
    country_label = country if country != "ALL" else "All countries"
    title = f"KPI 7 - {title_mode} of fuel type ({period_label}, {country_label}, {status_group})"

    fig = figure_from_pivot(pivot, y_title, x_title, title)

    if pivot.empty:
        return fig, html.Div("No data available for the selected filters.", className="small-note")

    table_df = pivot.copy()
    metric_cols = table_df.columns.tolist()
    if metric_mode == "share":
        volume_pivot, _, _, _ = kpi7_fuel_by_period(df, country, status_group, "volume", period_mode, start_date, end_date)
        if not volume_pivot.empty:
            volume_totals = volume_pivot.sum(axis=0).reindex(metric_cols, fill_value=0)
            total_row = pd.DataFrame([volume_totals], index=["TOTAL_VOLUME"])
            table_df = pd.concat([table_df, total_row])
    elif should_show_total_column(table_df, metric_cols, share_mode=False):
        table_df["TOTAL"] = table_df[metric_cols].sum(axis=1)
        total_row = pd.DataFrame([table_df.sum(axis=0)], index=["TOTAL"])
        table_df = pd.concat([table_df, total_row])

    table_df.index.name = "POWER_CATEGORY"
    table_df = table_df.reset_index()
    table = build_table(table_df, page_size=15)

    return fig, html.Div([html.H4(f"{period_label} fuel table (pivot)", className="panel-title"), table])


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
    State("summary-kpi1-limit-filter", "value"),
    State("summary-kpi2-limit-filter", "value"),
    prevent_initial_call=True,
)
def update_kpi_summary(
    _refresh_clicks: int,
    kpi1_country: str,
    kpi1_year: int | str,
    kpi1_month: int | str,
    kpi2_country: str,
    kpi2_year: int | str,
    kpi2_month: int | str,
    kpi3_country: str,
    kpi3_year: int | str,
    kpi3_month: int | str,
    kpi4_country: str,
    kpi4_year: int | str,
    kpi4_month: int | str,
    kpi5_country: str,
    kpi5_year: int | str,
    kpi5_month: int | str,
    kpi6_country: str,
    kpi6_year: int | str,
    kpi6_month: int | str,
    kpi1_limit: float | None,
    kpi2_limit: float | None,
):
    required_filters = [
        kpi1_country, kpi1_year,
        kpi2_country, kpi2_year,
        kpi3_country, kpi3_year, kpi3_month,
        kpi4_country, kpi4_year, kpi4_month,
        kpi5_country, kpi5_year, kpi5_month,
        kpi6_country, kpi6_year, kpi6_month,
    ]
    if has_missing_filters(*required_filters):
        return html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    month3 = resolve_month_value(kpi3_country, kpi3_year, kpi3_month)
    month4 = resolve_month_value(kpi4_country, kpi4_year, kpi4_month)
    month5 = resolve_month_value(kpi5_country, kpi5_year, kpi5_month)
    month6 = resolve_month_value(kpi6_country, kpi6_year, kpi6_month)
    month1 = None if kpi1_month in (None, "ALL") else int(kpi1_month)
    month2 = None if kpi2_month in (None, "ALL") else int(kpi2_month)

    kpi1_base = df[
        (df["COUNTRY"] == kpi1_country)
        & (df["YEAR"] == int(kpi1_year))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
    if month1 is not None:
        kpi1_base = kpi1_base[kpi1_base["MONTH"] == month1].copy()

    kpi2_base = df[
        (df["COUNTRY"] == kpi2_country)
        & (df["YEAR"] == int(kpi2_year))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()
    if month2 is not None:
        kpi2_base = kpi2_base[kpi2_base["MONTH"] == month2].copy()

    month3_base = df[
        (df["COUNTRY"] == kpi3_country)
        & (df["YEAR"] == int(kpi3_year))
        & (df["MONTH"] == int(month3))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()

    month4_base = df[
        (df["COUNTRY"] == kpi4_country)
        & (df["YEAR"] == int(kpi4_year))
        & (df["MONTH"] == int(month4))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()

    month5_base = df[
        (df["COUNTRY"] == kpi5_country)
        & (df["YEAR"] == int(kpi5_year))
        & (df["MONTH"] == int(month5))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()

    month6_base = df[
        (df["COUNTRY"] == kpi6_country)
        & (df["YEAR"] == int(kpi6_year))
        & (df["MONTH"] == int(month6))
        & (df["NOVA_ASSET_STATUS"] == "IN FLEET")
        & (df["BIKE_OR_CAR"] == "CAR")
    ].copy()

    kpi1_val = kpi_lease_under_25(df, kpi1_country, kpi1_year, kpi1_month)
    kpi2_val = kpi_lease_25_30(df, kpi2_country, kpi2_year, kpi2_month)
    diesel_non = kpi_diesel_non_diesel(df, kpi3_country, kpi3_year, month3)
    hybrid_val = kpi_hybrid_share(df, kpi4_country, kpi4_year, month4)
    ev_val = kpi_ev_share(df, kpi5_country, kpi5_year, month5)
    pv_lcv_val = kpi_pv_lcv(df, kpi6_country, kpi6_year, month6)

    period3 = summary_month_label(kpi3_year, month3)
    period4 = summary_month_label(kpi4_year, month4)
    period5 = summary_month_label(kpi5_year, month5)
    period6 = summary_month_label(kpi6_year, month6)
    period1 = summary_month_label(kpi1_year, month1)
    period2 = summary_month_label(kpi2_year, month2)
    kpi1_limit_text = "N/A" if kpi1_limit is None else f"{float(kpi1_limit):g}%"
    kpi2_limit_text = "N/A" if kpi2_limit is None else f"{float(kpi2_limit):g}%"
    rows = [
        {
            "label": "LTR < 25 (production)",
            "period": f"{kpi1_country} / {period1}",
            "result_text": percent_or_na(kpi1_val),
            "signal": kpi_limit_status(kpi1_val, kpi1_limit),
            "volume": len(kpi1_base),
            "unit": "%",
            "comment": f"Limit is {kpi1_limit_text}",
        },
        {
            "label": "25m <= LTR <= 30m (production)",
            "period": f"{kpi2_country} / {period2}",
            "result_text": percent_or_na(kpi2_val),
            "signal": kpi_limit_status(kpi2_val, kpi2_limit),
            "volume": len(kpi2_base),
            "unit": "%",
            "comment": f"Limit is {kpi2_limit_text}",
        },
        {
            "label": "DI vs Non-DI",
            "period": f"{kpi3_country} / {period3}",
            "result_text": f"{percent_or_na(diesel_non[0])} DI & {percent_or_na(diesel_non[1])} Non DI" if diesel_non[0] is not None else "N/A",
            "signal": "neutral",
            "volume": len(month3_base),
            "unit": "%",
            "comment": "Total current fleet",
        },
        {
            "label": "Hybrid share",
            "period": f"{kpi4_country} / {period4}",
            "result_text": percent_or_na(hybrid_val),
            "signal": "neutral",
            "volume": len(month4_base),
            "unit": "%",
            "comment": "Hyb is within Non-DI share. Total current fleet",
        },
        {
            "label": "EV share",
            "period": f"{kpi5_country} / {period5}",
            "result_text": percent_or_na(ev_val),
            "signal": "neutral",
            "volume": len(month5_base),
            "unit": "%",
            "comment": "EV is within Non-DI share. Total current fleet",
        },
        {
            "label": "PC vs LCV",
            "period": f"{kpi6_country} / {period6}",
            "result_text": f"{percent_or_na(pv_lcv_val[0])} PV & {percent_or_na(pv_lcv_val[1])} LCV" if pv_lcv_val[0] is not None else "N/A",
            "signal": "neutral",
            "volume": len(month6_base),
            "unit": "%",
            "comment": "Total current fleet",
        },
    ]

    return render_kpi_summary_table(rows)


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
    State("summary-kpi1-limit-filter", "value"),
    State("summary-kpi2-limit-filter", "value"),
    State("kpi7-country-filter", "value"),
    State("kpi7-date-range", "start_date"),
    State("kpi7-date-range", "end_date"),
    State("status-group-filter", "value"),
    State("metric-mode-filter", "value"),
    State("period-mode-filter", "value"),
    prevent_initial_call=True,
)
def download_view1_html(
    _n_clicks: int,
    country: str,
    year: int | str,
    month_value: int | str,
    kpi1_country: str,
    kpi1_year: int | str,
    kpi1_month: int | str,
    kpi2_country: str,
    kpi2_year: int | str,
    kpi2_month: int | str,
    kpi3_country: str,
    kpi3_year: int | str,
    kpi3_month: int | str,
    kpi4_country: str,
    kpi4_year: int | str,
    kpi4_month: int | str,
    kpi5_country: str,
    kpi5_year: int | str,
    kpi5_month: int | str,
    kpi6_country: str,
    kpi6_year: int | str,
    kpi6_month: int | str,
    kpi1_limit: float | None,
    kpi2_limit: float | None,
    kpi7_country: str,
    kpi7_start_date: str | None,
    kpi7_end_date: str | None,
    kpi7_status_group: str,
    kpi7_metric_mode: str,
    kpi7_period_mode: str,
):
    if has_missing_filters(
        country,
        year,
        month_value,
        kpi1_country,
        kpi1_year,
        kpi1_month,
        kpi2_country,
        kpi2_year,
        kpi2_month,
        kpi3_country,
        kpi3_year,
        kpi3_month,
        kpi4_country,
        kpi4_year,
        kpi4_month,
        kpi5_country,
        kpi5_year,
        kpi5_month,
        kpi6_country,
        kpi6_year,
        kpi6_month,
        kpi7_country,
        kpi7_status_group,
        kpi7_metric_mode,
        kpi7_period_mode,
    ):
        return no_update

    html_content = build_view1_download_report(
        country,
        year,
        month_value,
        kpi1_country,
        kpi1_year,
        kpi1_month,
        kpi2_country,
        kpi2_year,
        kpi2_month,
        kpi3_country,
        kpi3_year,
        kpi3_month,
        kpi4_country,
        kpi4_year,
        kpi4_month,
        kpi5_country,
        kpi5_year,
        kpi5_month,
        kpi6_country,
        kpi6_year,
        kpi6_month,
        kpi1_limit,
        kpi2_limit,
        kpi7_country,
        kpi7_start_date,
        kpi7_end_date,
        kpi7_status_group,
        kpi7_metric_mode,
        kpi7_period_mode,
    )
    safe_country = str(country).replace(" ", "_")
    safe_year = str(year)
    filename = f"fleet_monitoring_vue1_{safe_country}_{safe_year}.html"
    return dcc.send_string(html_content, filename=filename)


@app.callback(
    Output("kpi-period-note", "children"),
    Input("view1-refresh-button", "n_clicks"),
    State("year-filter", "value"),
    State("month-filter", "value"),
    prevent_initial_call=True,
)
def update_kpi_period_note(_refresh_clicks: int, year: int | str, month_value: int | str):
    if has_missing_filters(year):
        return ""
    if month_value in (None, "ALL"):
        return f"Période appliquée: {year} (mois automatique sur le dernier mois disponible)"
    return f"Période appliquée: {int(month_value):02d}-{year}"


@app.callback(
    Output("view2-html-download", "data"),
    Input("view2-download-button", "n_clicks"),
    State("v2-country-filter", "value"),
    State("v2-year-filter", "value"),
    State("v2-status-filter", "value"),
    State("v2-metric-mode-filter", "value"),
    prevent_initial_call=True,
)
def download_view2_html(_n_clicks: int, country: str, year: int | str, asset_status: str, metric_mode: str):
    if has_missing_filters(country, year, asset_status, metric_mode):
        return no_update

    html_content = build_view2_download_report(country, year, asset_status, metric_mode)
    safe_country = str(country).replace(" ", "_")
    safe_year = str(year)
    safe_metric = str(metric_mode)
    filename = f"fleet_monitoring_vue2_{safe_country}_{safe_year}_{safe_metric}.html"
    return dcc.send_string(html_content, filename=filename)


@app.callback(
    Output("v2-kpi8-graph", "figure"),
    Output("v2-kpi8-table-wrap", "children"),
    Input("view2-refresh-button", "n_clicks"),
    State("v2-country-filter", "value"),
    State("v2-year-filter", "value"),
    State("v2-status-filter", "value"),
    State("v2-metric-mode-filter", "value"),
    prevent_initial_call=True,
)
def update_view2_kpi8(_refresh_clicks: int, country: str, year: int | str, asset_status: str, metric_mode: str):
    if has_missing_filters(country, year, asset_status, metric_mode):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 8 - Sélectionnez les filtres puis cliquez sur Actualiser")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    table_kpi8, y_title, x_title, period_label = kpi8_production_ytd(df, country, year, asset_status, metric_mode)
    title_mode = "Share" if metric_mode == "share" else "Volume"
    title = f"KPI 8 - {title_mode} per vehicule category ({period_label})"

    fig = go.Figure()
    if not table_kpi8.empty:
        x_values = table_kpi8["MONTH"].tolist()
        metric_cols = [c for c in table_kpi8.columns if c not in ["YEAR", "MONTH"]]
        for col in metric_cols:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=table_kpi8[col].tolist(),
                    mode="lines+markers",
                    name=str(col),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=520,
    )

    if table_kpi8.empty:
        return fig, html.Div("No data available for the selected filters.", className="small-note")

    table_kpi8 = table_kpi8.copy()
    metric_cols = [c for c in table_kpi8.columns if c not in ["YEAR", "MONTH"]]
    if should_show_total_column(table_kpi8, metric_cols, share_mode=(metric_mode == "share")):
        table_kpi8["TOTAL"] = table_kpi8[metric_cols].sum(axis=1)
        total_row = {"YEAR": "TOTAL", "MONTH": "ALL"}
        for c in metric_cols + ["TOTAL"]:
            total_row[c] = table_kpi8[c].sum()
        table_kpi8 = pd.concat([table_kpi8, pd.DataFrame([total_row])], ignore_index=True)

    table = build_table(table_kpi8, page_size=15)
    return fig, html.Div([html.H4("KPI 8 table (pivot)", className="panel-title"), table])


@app.callback(
    Output("v3-kpi91-graph", "figure"),
    Output("v3-kpi91-table-wrap", "children"),
    Output("v3-kpi92-graph", "figure"),
    Output("v3-kpi92-table-wrap", "children"),
    Input("view3-refresh-button", "n_clicks"),
    State("v3-country-filter", "value"),
    State("v3-year-filter", "value"),
    State("v3-status-group-filter", "value"),
    State("v3-vehicle-type-filter", "value"),
    prevent_initial_call=True,
)
def update_view3_kpis(_refresh_clicks: int, country: str, year: int | str, status_group: str, vehicle_type: str):
    if has_missing_filters(country, year, status_group, vehicle_type):
        empty_fig_1 = go.Figure()
        empty_fig_1.update_layout(title="KPI 9_1 - Sélectionnez les filtres puis cliquez sur Actualiser")
        empty_fig_2 = go.Figure()
        empty_fig_2.update_layout(title="KPI 9_2 - Sélectionnez les filtres puis cliquez sur Actualiser")
        note = html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")
        return empty_fig_1, note, empty_fig_2, note

    status_value = status_group
    kpi91 = kpi9_1_vehicle_share_quarter(df, country, year, status_value, vehicle_type)

    fig91 = go.Figure()
    if not kpi91.empty:
        for idx in kpi91.index:
            fig91.add_trace(go.Bar(x=kpi91.columns.tolist(), y=kpi91.loc[idx].tolist(), name=str(idx)))
            fig91.add_trace(go.Scatter(x=kpi91.columns.tolist(), y=kpi91.loc[idx].tolist(), mode="lines+markers", showlegend=False))
    fig91.update_layout(
        title=f"KPI 9_1 - {vehicle_type} share per quarter ({year}, {status_group})",
        xaxis_title="Quarter",
        yaxis_title="Share (%)",
        template="plotly_white",
        height=420,
    )

    if kpi91.empty:
        table91_wrap = html.Div("No data available for KPI 9_1.", className="small-note")
    else:
        table91 = kpi91.copy()
        table91["TOTAL"] = table91.sum(axis=1)
        table91_total = pd.DataFrame([table91.sum(axis=0)], index=["TOTAL"])
        table91 = pd.concat([table91, table91_total]).reset_index().rename(columns={"index": "NOVA_ASSET_STATUS"})
        table91_wrap = html.Div([html.H4("KPI 9_1 table", className="panel-title"), build_table(table91, page_size=15)])

    pivot92, y_title92, x_title92, _ = kpi9_2_vehicle_energy_share_quarter(df, country, year, status_value, vehicle_type)
    fig92 = go.Figure()
    if not pivot92.empty:
        for c in pivot92.columns:
            fig92.add_trace(go.Scatter(x=pivot92.index.tolist(), y=pivot92[c].tolist(), mode="lines+markers", name=str(c)))
    fig92.update_layout(
        title=f"KPI 9_2 - {vehicle_type} share by energy per quarter ({year}, {status_group})",
        xaxis_title=x_title92 if x_title92 else "Quarter",
        yaxis_title=y_title92 if y_title92 else "Volume",
        template="plotly_white",
        height=420,
    )

    if pivot92.empty:
        table92_wrap = html.Div("No data available for KPI 9_2.", className="small-note")
    else:
        table92 = pivot92.copy()
        metric_cols_92 = table92.columns.tolist()
        if should_show_total_column(table92, metric_cols_92, share_mode=True):
            table92["TOTAL"] = table92[metric_cols_92].sum(axis=1)
            table92_total = pd.DataFrame([table92.sum(axis=0)], index=["TOTAL"])
            table92 = pd.concat([table92, table92_total])
        table92 = table92.reset_index().rename(columns={"index": "Quarter"})
        table92_wrap = html.Div([html.H4("KPI 9_2 table (pivot)", className="panel-title"), build_table(table92, page_size=15)])

    return fig91, table91_wrap, fig92, table92_wrap


@app.callback(
    Output("v4-kpi10-graph", "figure"),
    Output("v4-kpi10-table-wrap", "children"),
    Input("view4-refresh-button", "n_clicks"),
    State("v4-country-filter", "value"),
    State("v4-status-filter", "value"),
    State("v4-eoc-window-filter", "value"),
    prevent_initial_call=True,
)
def update_view4_kpi10(_refresh_clicks: int, country: str, status: str, eoc_window: int | str):
    if has_missing_filters(country, status, eoc_window):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 10 - Sélectionnez les filtres puis cliquez sur Actualiser")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    out = df.copy()
    out = out[
        (out["COUNTRY"] == country)
        & (out["BIKE_OR_CAR"] == "CAR")
    ].copy()
    out = apply_status_filter(out, status)

    out = out.dropna(subset=["CONTRACT_FINAL_END", "VEHICLE_MODEL_MAPED"])
    out["CONTRACT_FINAL_END"] = pd.to_datetime(out["CONTRACT_FINAL_END"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    if eoc_window != "ALL":
        window_end = today + pd.DateOffset(months=int(eoc_window))
        out = out[(out["CONTRACT_FINAL_END"] >= today) & (out["CONTRACT_FINAL_END"] <= window_end)]
    
    if out.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 10 - No data")
        return empty_fig, html.Div("No data available for the selected filters.", className="small-note")

    out["Quarter"] = out["CONTRACT_FINAL_END"].dt.to_period("Q").astype(str)

    pivot = out.pivot_table(
        index="Quarter",
        columns="VEHICLE_MODEL_MAPED",
        values="VEHICLE_ID",
        aggfunc="count",
        fill_value=0,
    )

    total_row = pd.DataFrame(pivot.sum()).T
    total_row.index = ["Total"]
    pivot = pd.concat([pivot, total_row])

    total_series = pd.Series(pivot.loc["Total"].to_numpy(), index=pivot.columns)
    col_order = total_series.sort_values(ascending=False).index
    pivot = pivot[col_order]
    kpi10_table = pivot.reset_index().rename(columns={"Quarter": "QUARTER"})

    quarters = [q for q in pivot.index.tolist() if q != "Total"]
    if not quarters:
        fig = go.Figure()
        fig.update_layout(title="KPI 10 - No data")
    else:
        top_models = col_order[:5].tolist()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for model in top_models:
            fig.add_trace(
                go.Bar(
                    x=quarters,
                    y=pivot.loc[quarters, model].astype(float),
                    name=str(model),
                ),
                secondary_y=False,
            )

        quarter_totals = pivot.loc[quarters, :].sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=quarters,
                y=quarter_totals,
                mode="lines+markers",
                name="TOTAL",
            ),
            secondary_y=True,
        )

        window_label = "all horizon" if eoc_window == "ALL" else f"next {eoc_window} month(s)"
        fig.update_layout(
            title=f"KPI 10 - Top models at EOC ({country}, {status}, {window_label})",
            barmode="group",
            template="plotly_white",
            xaxis_title="Quarter",
            height=460,
        )
        fig.update_yaxes(title_text="Volume by model", secondary_y=False)
        fig.update_yaxes(title_text="Total volume", secondary_y=True)

    table = build_table(kpi10_table, page_size=15)
    return fig, html.Div([html.H4("KPI 10 table (models at EOC)", className="panel-title"), table])


@app.callback(
    Output("v5-kpi11-graph", "figure"),
    Output("v5-kpi11-table-wrap", "children"),
    Input("view5-refresh-button", "n_clicks"),
    State("v5-status-filter", "value"),
    State("v5-source-filter", "value"),
    State("v5-variable-filter", "value"),
    prevent_initial_call=True,
)
def update_view5_kpi11(_refresh_clicks: int, status_value: str, source_value: str, variable: str):
    if has_missing_filters(status_value, source_value, variable):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 11 - Sélectionnez les filtres puis cliquez sur Actualiser")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    if source_value == "market":
        market_ready, market_var = prepare_market_concentration_source(market_df, variable)
        kpi11_table = kpi_top_per_quarter_with_share_market(market_ready, market_var)
        fig = plot_top_var_kpi(kpi11_table, title_suffix=f"(Market, {variable})")
    else:
        portfolio_ready, portfolio_var = prepare_portfolio_concentration_source(df, variable)
        if "YTD" in status_value.upper():
            kpi11_table = kpi_count_share_ytd_by_quarter(portfolio_ready, asset_status=status_value.replace(" YTD", ""), var_col=portfolio_var, bike_or_car="CAR")
            plot_df = kpi11_table.rename(columns={"VOLUME_YTD": "VOLUME", "SHARE_YTD": "SHARE"}).copy()
        else:
            kpi11_table = kpi_count_share_quarterly(portfolio_ready, asset_status=status_value, var_col=portfolio_var, bike_or_car="CAR")
            plot_df = kpi11_table.copy()

        if plot_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="KPI 11 - No data")
            return empty_fig, html.Div("No data available for the selected filters.", className="small-note")

        fig = plot_kpi_share(plot_df, portfolio_var)

    if kpi11_table.empty:
        return fig, html.Div("No data available for the selected filters.", className="small-note")

    table = build_table(kpi11_table, page_size=15)
    return fig, html.Div([html.H4("KPI 11 table", className="panel-title"), table])


@app.callback(
    Output("v6-kpi6-table-wrap", "children"),
    Input("view6-refresh-button", "n_clicks"),
    State("v6-status-filter", "value"),
    State("v6-source-filter", "value"),
    State("v6-variable-filter", "value"),
    State("v6-metric-mode-filter", "value"),
    prevent_initial_call=True,
)
def update_view6_kpi6(_refresh_clicks: int, status_value: str, source_value: str, variable: str, metric_mode: str):
    if has_missing_filters(status_value, source_value, variable, metric_mode):
        return html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    if source_value == "market":
        market_ready, market_var = prepare_market_concentration_source(market_df, variable)
        kpi6_table = kpi_top_per_quarter_with_share_market(market_ready, market_var, metric_mode=metric_mode)
    else:
        portfolio_ready, portfolio_var = prepare_portfolio_concentration_source(df, variable)
        kpi6_table = kpi_top_per_quarter_with_share(portfolio_ready, asset_status=status_value, var_col=portfolio_var, bike_or_car="CAR", metric_mode=metric_mode)

    if kpi6_table.empty:
        return html.Div("No data available for the selected filters.", className="small-note")

    table = build_table(kpi6_table, page_size=15)
    return html.Div([html.H4(f"Top {variable} per country and quarter", className="panel-title"), table])


@app.callback(
    Output("v7-kpi13-graph", "figure"),
    Output("v7-kpi13-table-wrap", "children"),
    Input("view7-refresh-button", "n_clicks"),
    State("v7-status-filter", "value"),
    State("v7-variable-filter", "value"),
    prevent_initial_call=True,
)
def update_view7_kpi13(_refresh_clicks: int, status_value: str, variable: str):
    if has_missing_filters(status_value, variable):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 13 - Sélectionnez les filtres puis cliquez sur Actualiser")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Actualiser.", className="small-note")

    if market_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 13 - Market dataset not found")
        return empty_fig, html.Div("Market dataset is not available.", className="small-note")

    if variable == "OEM":
        var_portfolio = "OEM_UPDATE"
        var_market = "Make Group"
        bev_only = False
    elif variable == "BEV":
        var_portfolio = "BRAND_UPDATE"
        var_market = "Make"
        bev_only = True
    else:
        var_portfolio = "BRAND_UPDATE"
        var_market = "Make"
        bev_only = False

    kpi13 = kpi_top_brand_vs_market(
        df_portfolio=df,
        df_market=market_df,
        var_col_portfolio=var_portfolio,
        var_col_market=var_market,
        asset_status=status_value,
    )

    title = f"KPI 13 - {variable} Portfolio vs Market ({status_value})"
    fig = figure_top_brand_vs_market(kpi13, title)

    if kpi13.empty:
        return fig, html.Div("No data available for selected filters.", className="small-note")

    columns = [
        "COUNTRY",
        "QUARTER",
        "BRAND",
        "volume_portfolio",
        "share_portfolio",
        "volume_market",
        "share_market",
        "ratio",
    ]
    table_df = kpi13[[c for c in columns if c in kpi13.columns]].copy()
    table = build_table(table_df, page_size=15)
    return fig, html.Div([html.H4("KPI 13 table", className="panel-title"), table])


if __name__ == "__main__":
    app.run(debug=True)
