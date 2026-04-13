from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

try:
    from dash import Dash, Input, Output, dash_table, dcc, html
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Dash is required to run this app. Install it with: pip install dash plotly pandas"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "synthetic_vehicle_dataset.parquet"

# Paste here the exact output of Windows "Copy as path" if needed.
# Example: "C:\\Users\\cyril\\Documents\\data\\my_file.csv"
HARDCODED_DATA_PATH = ""


def normalize_copied_path(path_value: str) -> str:
    return path_value.strip().strip('"').strip("'")


def load_dataset() -> pd.DataFrame:
    raw_path = normalize_copied_path(HARDCODED_DATA_PATH)
    dataset_path = Path(raw_path) if raw_path else DEFAULT_DATA_PATH

    if dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path).copy()
    else:
        df = pd.read_parquet(dataset_path).copy()
    df["COB_DATE"] = pd.to_datetime(df["COB_DATE"], errors="coerce")
    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month

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

    return df


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def percent_or_na(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def filter_base(
    df: pd.DataFrame,
    country: str,
    year: int | str,
    status: str = "IN FLEET",
    bike_or_car: str = "CAR",
) -> pd.DataFrame:
    out = df.copy()

    if country != "ALL":
        out = out[out["COUNTRY"] == country]

    if year != "ALL":
        out = out[out["YEAR"] == int(year)]

    if status != "ALL" and "NOVA_ASSET_STATUS" in out.columns:
        out = out[out["NOVA_ASSET_STATUS"] == status]

    if bike_or_car != "ALL" and "BIKE_OR_CAR" in out.columns:
        out = out[out["BIKE_OR_CAR"] == bike_or_car]

    return out


def filter_status_group(df: pd.DataFrame, status_group: str) -> pd.DataFrame:
    out = df.copy()
    if status_group == "Fleet":
        return out[out["NOVA_ASSET_STATUS"].str.contains("FLEET", case=False, na=False)]
    if status_group == "Order":
        return out[out["NOVA_ASSET_STATUS"].str.contains("ORDER", case=False, na=False)]
    return out


def available_months(df: pd.DataFrame, country: str, year: int | str) -> list[int]:
    subset = filter_base(df, country, year, status="IN FLEET", bike_or_car="CAR")
    return sorted(subset["MONTH"].dropna().astype(int).unique().tolist())


def latest_month(df: pd.DataFrame, country: str, year: int | str) -> int | None:
    months = available_months(df, country, year)
    if not months:
        return None
    return months[-1]


def month_subset(df: pd.DataFrame, country: str, year: int | str, month: int | None) -> pd.DataFrame:
    subset = filter_base(df, country, year, status="IN FLEET", bike_or_car="CAR")
    if month is not None:
        subset = subset[subset["MONTH"] == month]
    return subset


def kpi_lease_under_25(df: pd.DataFrame, country: str, year: int | str) -> float | None:
    subset = filter_base(df, country, year, status="IN FLEET", bike_or_car="CAR")
    subset = subset.drop_duplicates(subset=["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"])
    if subset.empty:
        return None
    durations = pd.to_numeric(subset["FINAL_CONTRACT_DURATION"], errors="coerce").dropna()
    if durations.empty:
        return None
    return float((durations < 25).mean() * 100)


def kpi_lease_25_30(df: pd.DataFrame, country: str, year: int | str) -> float | None:
    subset = filter_base(df, country, year, status="IN FLEET", bike_or_car="CAR")
    subset = subset.drop_duplicates(subset=["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"])
    if subset.empty:
        return None
    durations = pd.to_numeric(subset["FINAL_CONTRACT_DURATION"], errors="coerce").dropna()
    if durations.empty:
        return None
    return float(durations.between(25, 30, inclusive="both").mean() * 100)


def kpi_diesel_non_diesel(df: pd.DataFrame, country: str, year: int | str, month: int | None) -> tuple[float | None, float | None]:
    subset = month_subset(df, country, year, month)
    power_col = pick_first_existing_column(df, ["POWER_CATEGORY_2", "POWER_CATEGORY"])
    if power_col is None or subset.empty:
        return None, None
    values = subset[power_col].astype(str).str.upper()
    diesel_share = float((values == "DIESEL").mean() * 100)
    return diesel_share, float(100 - diesel_share)


def kpi_hybrid_share(df: pd.DataFrame, country: str, year: int | str, month: int | None) -> float | None:
    subset = month_subset(df, country, year, month)
    power_col = pick_first_existing_column(df, ["POWER_CATEGORY"])
    if power_col is None or subset.empty:
        return None
    values = subset[power_col].astype(str).str.upper()
    return float(values.isin(["FULL HYBRID", "PLUG-IN HYBRID"]).mean() * 100)


def kpi_ev_share(df: pd.DataFrame, country: str, year: int | str, month: int | None) -> float | None:
    subset = month_subset(df, country, year, month)
    power_col = pick_first_existing_column(df, ["POWER_CATEGORY"])
    if power_col is None or subset.empty:
        return None
    values = subset[power_col].astype(str).str.upper()
    return float((values == "ELECTRIC").mean() * 100)


def kpi_pv_lcv(df: pd.DataFrame, country: str, year: int | str, month: int | None) -> tuple[float | None, float | None]:
    subset = month_subset(df, country, year, month)
    if subset.empty:
        return None, None
    values = subset["CLS_VEHICLE_TYPE"].astype(str).str.upper()
    pv_share = float((values == "PV").mean() * 100)
    lcv_share = float(values.isin(["LCV", "LV"]).mean() * 100)
    return pv_share, lcv_share


def detect_fuel_column(df: pd.DataFrame) -> str | None:
    return pick_first_existing_column(df, ["POWER_CATEGORY_3", "POWER_CATEGORY_2", "POWER_CATEGORY"])


def kpi7_fuel_by_period(
    df: pd.DataFrame,
    country: str,
    year: int | str,
    status_group: str,
    metric_mode: str,
    period_mode: str,
) -> tuple[pd.DataFrame, str, str, str]:
    if period_mode in ["monthly", "quarterly"]:
        subset = filter_base(df, country, year, status="ALL", bike_or_car="CAR")
    else:
        subset = filter_base(df, country, "ALL", status="ALL", bike_or_car="CAR")

    subset = filter_status_group(subset, status_group)
    subset = subset.drop_duplicates(subset=["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"])

    fuel_col = detect_fuel_column(subset)
    if fuel_col is None or subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.dropna(subset=[fuel_col, "MONTH", "YEAR"])
    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.copy()
    subset[fuel_col] = subset[fuel_col].astype(str).str.upper()
    subset["YEAR"] = subset["YEAR"].astype(int)
    subset["MONTH"] = subset["MONTH"].astype(int)

    if period_mode == "monthly":
        subset["PERIOD_SORT"] = subset["MONTH"]
        subset["PERIOD_LABEL"] = subset["MONTH"].map(lambda m: f"{int(m):02d}")
        x_title = "Month"
        period_label = f"Monthly ({year})"
    elif period_mode == "quarterly":
        subset["PERIOD_SORT"] = ((subset["MONTH"] - 1) // 3) + 1
        subset["PERIOD_LABEL"] = subset["PERIOD_SORT"].map(lambda q: f"Q{int(q)}")
        x_title = "Quarter"
        period_label = f"Quarterly ({year})"
    else:
        subset["PERIOD_SORT"] = subset["YEAR"]
        subset["PERIOD_LABEL"] = subset["YEAR"].astype(str)
        x_title = "Year"
        period_label = "Yearly"

    grouped = subset.groupby([fuel_col, "PERIOD_SORT", "PERIOD_LABEL"]).size().reset_index(name="VOLUME")
    if grouped.empty:
        return pd.DataFrame(), "", "", ""

    if metric_mode == "share":
        grouped["METRIC"] = grouped.groupby(["PERIOD_SORT", "PERIOD_LABEL"])["VOLUME"].transform(
            lambda x: x / x.sum() * 100
        )
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


def build_kpi_summary_dataframe(df: pd.DataFrame, country: str, year: int | str, month: int | None) -> pd.DataFrame:
    kpi1 = kpi_lease_under_25(df, country, year)
    kpi2 = kpi_lease_25_30(df, country, year)
    diesel_share, non_diesel_share = kpi_diesel_non_diesel(df, country, year, month)
    hybrid_share = kpi_hybrid_share(df, country, year, month)
    ev_share = kpi_ev_share(df, country, year, month)
    pv_share, lcv_share = kpi_pv_lcv(df, country, year, month)

    monthly_period = summary_month_label(year, month)

    return pd.DataFrame(
        {
            "Asset Risk, Financed Fleet": [
                "LTR < 25 (production)",
                "25m <= LTR <= 30m (production)",
                "DI vs Non-DI",
                "Hybrid share",
                "EV share",
                "PC vs LCV",
            ],
            "Period": [str(year), str(year), monthly_period, monthly_period, monthly_period, monthly_period],
            "Result": [
                round(kpi1, 2) if kpi1 is not None else "N/A",
                (
                    round(kpi2, 2) if kpi2 is not None else "N/A"
                ),
                (
                    f"{round(diesel_share, 2)} DI & {round(non_diesel_share, 2)} Non DI"
                    if diesel_share is not None and non_diesel_share is not None
                    else "N/A"
                ),
                round(hybrid_share, 2) if hybrid_share is not None else "N/A",
                round(ev_share, 2) if ev_share is not None else "N/A",
                (
                    f"{round(pv_share, 2)} PV & {round(lcv_share, 2)} LCV"
                    if pv_share is not None and lcv_share is not None
                    else "N/A"
                ),
            ],
            "Unit": ["%", "%", "%", "%", "%", "%"],
            "Comment": [
                "Limit is 5%",
                "Limit is 10%",
                "Total current fleet",
                "Hyb is within Non-DI share. Total current fleet",
                "EV is within Non-DI share. Total current fleet",
                "Total current fleet",
            ],
        }
    )


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


df = load_dataset()

countries = sorted(df["COUNTRY"].dropna().unique().tolist())
years = sorted(df["YEAR"].dropna().astype(int).unique().tolist())
default_country = countries[0] if countries else None
default_year = years[-1] if years else None
default_month = latest_month(df, default_country, default_year) if default_country and default_year else None

app = Dash(__name__)
app.title = "Fleet Monitoring Dashboard"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Fleet Monitoring Dashboard"),
                html.P(
                    "Vue 1: Main KPIs et KPI 7, construits a partir de synthetic_vehicle_dataset.parquet et des fonctions du notebook fleet_monitoring.ipynb."
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Country", className="filter-label"),
                        dcc.Dropdown(
                            id="country-filter",
                            options=[{"label": c, "value": c} for c in countries],
                            value=default_country,
                            clearable=False,
                        ),
                    ],
                    className="filter-box",
                ),
                html.Div(
                    [
                        html.Div("Year", className="filter-label"),
                        dcc.Dropdown(
                            id="year-filter",
                            options=[{"label": str(y), "value": y} for y in years],
                            value=default_year,
                            clearable=False,
                        ),
                    ],
                    className="filter-box",
                ),
                html.Div(
                    [
                        html.Div("Month (KPI 3 to 6)", className="filter-label"),
                        dcc.Dropdown(
                            id="month-filter",
                            options=[{"label": "All", "value": "ALL"}],
                            value=default_month if default_month is not None else "ALL",
                            clearable=False,
                        ),
                    ],
                    className="filter-box",
                ),
                html.Div(
                    [
                        html.Div("Note", className="filter-label"),
                        html.Div(
                            "KPI 1-6 are forced on IN FLEET; KPI 7 can switch Fleet/Order and Volume/Share.",
                            className="small-note",
                        ),
                    ],
                    className="filter-box",
                ),
            ],
            className="filter-bar",
        ),
        html.Div(
            [
                html.Div(id="kpi-1-card"),
                html.Div(id="kpi-2-card"),
                html.Div(id="kpi-3-card"),
                html.Div(id="kpi-4-card"),
                html.Div(id="kpi-5-card"),
                html.Div(id="kpi-6-card"),
            ],
            className="cards-grid",
        ),
        html.Div(id="kpi-summary-wrap", className="panel"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("KPI 7 - Share of fuel type", className="panel-title"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Status", className="filter-label"),
                                        dcc.RadioItems(
                                            id="status-group-filter",
                                            options=[
                                                {"label": "Fleet", "value": "Fleet"},
                                                {"label": "Order", "value": "Order"},
                                                {"label": "All", "value": "All"},
                                            ],
                                            value="Fleet",
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
                                            options=[
                                                {"label": "Share", "value": "share"},
                                                {"label": "Volume", "value": "volume"},
                                            ],
                                            value="share",
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
                                            value="quarterly",
                                            inline=True,
                                            labelStyle={"marginRight": "18px"},
                                        ),
                                    ],
                                    className="filter-box",
                                ),
                            ],
                            className="controls-inline",
                        ),
                        dcc.Graph(id="kpi-7-graph", config={"displayModeBar": False}),
                        html.Div(id="kpi-7-table-wrap"),
                    ]
                )
            ],
            className="panel",
        ),
    ],
    className="page-wrap",
)


@app.callback(
    Output("month-filter", "options"),
    Output("month-filter", "value"),
    Input("country-filter", "value"),
    Input("year-filter", "value"),
)
def update_month_filter(country: str, year: int | str):
    months = available_months(df, country, year)
    options = [{"label": "All", "value": "ALL"}] + [{"label": f"{month:02d}", "value": month} for month in months]
    default_value = months[-1] if months else "ALL"
    return options, default_value


@app.callback(
    Output("kpi-1-card", "children"),
    Output("kpi-2-card", "children"),
    Output("kpi-3-card", "children"),
    Output("kpi-4-card", "children"),
    Output("kpi-5-card", "children"),
    Output("kpi-6-card", "children"),
    Input("country-filter", "value"),
    Input("year-filter", "value"),
    Input("month-filter", "value"),
)
def update_kpi_cards(country: str, year: int | str, month_value: int | str):
    month = None if month_value in (None, "ALL") else int(month_value)

    kpi1 = kpi_lease_under_25(df, country, year)
    kpi2 = kpi_lease_25_30(df, country, year)
    diesel_share, non_diesel_share = kpi_diesel_non_diesel(df, country, year, month)
    hybrid_share = kpi_hybrid_share(df, country, year, month)
    ev_share = kpi_ev_share(df, country, year, month)
    pv_share, lcv_share = kpi_pv_lcv(df, country, year, month)

    card1 = build_card("KPI 1 Lease time < 25 months", percent_or_na(kpi1), "Goal: no more than 5% of the current fleet.", "#1d5f99")
    card2 = build_card("KPI 2 Lease time 25-30 months", percent_or_na(kpi2), "Goal: no more than 10% of the current fleet.", "#2f855a")
    card3 = build_card("KPI 3 Diesel vs Non Diesel", f"{percent_or_na(diesel_share)} / {percent_or_na(non_diesel_share)}", "Selected month on the IN FLEET population.", "#b7791f")
    card4 = build_card("KPI 4 Hybrid Share", percent_or_na(hybrid_share), "HEV / PHEV share on the selected month.", "#7b4fe2")
    card5 = build_card("KPI 5 EV Share", percent_or_na(ev_share), "Electric share on the selected month.", "#00a3a3")
    card6 = build_card("KPI 6 PV vs LCV", f"{percent_or_na(pv_share)} / {percent_or_na(lcv_share)}", "PV / LCV share on the selected month.", "#d64545")

    return card1, card2, card3, card4, card5, card6


@app.callback(
    Output("kpi-7-graph", "figure"),
    Output("kpi-7-table-wrap", "children"),
    Input("country-filter", "value"),
    Input("year-filter", "value"),
    Input("status-group-filter", "value"),
    Input("metric-mode-filter", "value"),
    Input("period-mode-filter", "value"),
)
def update_kpi7(country: str, year: int | str, status_group: str, metric_mode: str, period_mode: str):
    pivot, y_title, x_title, period_label = kpi7_fuel_by_period(df, country, year, status_group, metric_mode, period_mode)
    title_mode = "Share" if metric_mode == "share" else "Volume"
    title = f"KPI 7 - {title_mode} of fuel type ({period_label}, {status_group})"

    fig = figure_from_pivot(pivot, y_title, x_title, title)

    if pivot.empty:
        return fig, html.Div("No data available for the selected filters.", className="small-note")

    table_df = pivot.copy()
    table_df.index.name = "POWER_CATEGORY"
    table_df = table_df.reset_index()
    columns = [{"name": c, "id": c} for c in table_df.columns]

    table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=12,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#102a43", "color": "white", "fontWeight": "700", "border": "none"},
        style_cell={
            "padding": "8px 10px",
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": "13px",
            "border": "1px solid #e6eaf0",
            "minWidth": "110px",
            "maxWidth": "180px",
            "whiteSpace": "normal",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}],
    )

    return fig, html.Div([html.H4(f"{period_label} fuel table (pivot)", className="panel-title"), table])


@app.callback(
    Output("kpi-summary-wrap", "children"),
    Input("country-filter", "value"),
    Input("year-filter", "value"),
    Input("month-filter", "value"),
)
def update_kpi_summary(country: str, year: int | str, month_value: int | str):
    month = None if month_value in (None, "ALL") else int(month_value)
    if month is None:
        month = latest_month(df, country, year)

    summary_df = build_kpi_summary_dataframe(df, country, year, month)
    table = dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary_df.columns],
        sort_action="native",
        page_size=8,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#102a43", "color": "white", "fontWeight": "700", "border": "none"},
        style_cell={
            "padding": "8px 10px",
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": "13px",
            "border": "1px solid #e6eaf0",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "140px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}],
    )

    return html.Div([html.H4("KPI summary table", className="panel-title"), table])


if __name__ == "__main__":
    app.run(debug=True)
