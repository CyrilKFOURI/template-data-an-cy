from __future__ import annotations

import calendar
import glob
import os
import re
from functools import lru_cache
from pathlib import Path
from html import escape as html_escape
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update, ALL, callback_context
import dash

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FOLDER = BASE_DIR / "data"
DEFAULT_MARKET_DATA_PATH = BASE_DIR / "market_dataset_europe.parquet"
LOGO_PATH = BASE_DIR / "a.jpg"

# --- Data Loading Scope ---
HARDCODED_DATA_PATH = ""
COUNTRIES_TO_READ = ["SPAIN"]
START_YYYYMM = "202301"
END_YYYYMM = "202512"


COLUMNS_TO_READ = ['EXTENSION_DATE','CLASS_CATALOG','CONTRACT_START_DATE','CONTRACT_END_DATE','COB_DATE',
    'DATE_OF_ORDER','CONTRACT_END_DATE_AMENDED','ID_CONTRACT','VEHICLE_ID','ID_QUOTATION',
    'FINAL_CONTRACT_DURATION','POWER_CATEGORY','BIKE_OR_CAR','CLS_VEHICLE_TYPE','COUNTRY',
    'BRAND_UPDATE', 'VEHICLE_CLASS','VEHICLE_MODEL','MODEL_CATALOG','OEM_UPDATE',
    'NOVA_ASSET_STATUS','FUEL_TYPE2','FUEL_TYPE','VA_CO2_EMSS_REAL','DELIVERY_DATE']

USE_UNIQUE_KEY_LOGIC = True
UNIQUE_KEY_COLS = ['ID_CONTRACT', 'VEHICLE_ID', 'ID_QUOTATION']




# --- KPI REGISTRY: Single Source of Truth for Labels, Limits, and Logic ---
KPI_REGISTRY = {
    "kpi1": {
        "label": "KPI 1 - LTR < 25",
        "description": "LTR < 25 share",
        "limit": 25.0,
        "color": "#1d5f99",
        "func_name": "kpi_lease_under_25"
    },
    "kpi2": {
        "label": "KPI 2 - LTR [25-30]",
        "description": "LTR [25-30] share",
        "limit": 30.0,
        "color": "#2f855a",
        "func_name": "kpi_lease_25_30"
    },
    "kpi3": {
        "label": "KPI 3 - DI vs Non-DI",
        "description": "Diesel vs Non-Diesel",
        "limit": None,
        "color": "#d98943",
        "func_name": "kpi_diesel_non_diesel"
    },
    "kpi4": {
        "label": "KPI 4 - Hybrid Share",
        "description": "Hybrid Share (HEV + PHEV)",
        "limit": None,
        "color": "#b08b2f",
        "func_name": "kpi_hybrid_share"
    },
    "kpi5": {
        "label": "KPI 5 - EV Share",
        "description": "EV (Electric) Share",
        "limit": None,
        "color": "#2f7f5f",
        "func_name": "kpi_ev_share"
    },
    "kpi6": {
        "label": "KPI 6 - PC vs LCV",
        "description": "PC vs LCV share",
        "limit": None,
        "color": "#d64545",
        "func_name": "kpi_pv_lcv"
    }
}

# =============================================================================
# =============================================================================

def get_kpi_data(df: pd.DataFrame, col: str | None = None, extra_keys: list[str] | None = None) -> pd.Series | pd.DataFrame:
    """
    Centralized helper to handle unique key logic for all KPIs.
    If USE_UNIQUE_KEY_LOGIC is True, it aggregates data by the unique key.
    If col is provided, it returns a Series of the first value found for that column.
    extra_keys: list of additional columns to include in the unique key (e.g. ['PERIOD_LABEL']).
    """
    if df.empty:
        return pd.Series(dtype=float) if col else df
    if not USE_UNIQUE_KEY_LOGIC:
        return df[col] if col else df
    
    keys = UNIQUE_KEY_COLS.copy()
    if extra_keys:
        for k in extra_keys:
            if k in df.columns and k not in keys:
                keys.append(k)
    
    missing = [c for c in UNIQUE_KEY_COLS if c not in df.columns]
    if missing:
        return df[col] if col else df
        
    if col:
        return df.groupby(keys)[col].first()
    return df.drop_duplicates(subset=keys)



FUEL_COLOR_SEQUENCE = [
    "#2c3e50", # Diesel
    "#e67e22", # Petrol
    "#f1c40f", # HEV
    "#2ecc71", # PHEV
    "#16a085", # BEV
    "#3498db", 
    "#8e44ad", 
    "#d35400", 
    "#7f8c8d",
]

def fuel_label_priority(label: str) -> int:
    """Orders fuel types for consistent visualization colors."""
    value = str(label).strip().upper()
    if "DIESEL" in value or value in {"DI", "D"}: return 0
    if any(token in value for token in ["PETROL", "GASOLINA", "BENZINA", "ESSENCE", "GAZ", "GAS", "NON-DI"]): return 1
    if "HEV" in value or "HYBRID" in value or "MHEV" in value: return 2
    if "PHEV" in value or "PLUG-IN" in value or "PLUGIN" in value: return 3
    if "EV" in value or "ELECTRIC" in value or "BEV" in value: return 4
    return 10

def fuel_color_map(labels: list[object]) -> dict[str, str]:
    """Generates a mapping of fuel types to fixed colors."""
    names = [str(label) for label in labels]
    ordered = sorted(enumerate(names), key=lambda item: (fuel_label_priority(item[1]), item[0]))
    color_by_name: dict[str, str] = {}
    for position, (_, name) in enumerate(ordered):
        color_by_name[name] = FUEL_COLOR_SEQUENCE[position % len(FUEL_COLOR_SEQUENCE)]
    return color_by_name

# --- Data Loading Logic ---
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
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        print("Warning: Dataset is empty or None. Returning empty DataFrame.")
        return pd.DataFrame()

    if "COB_DATE" in df.columns:
        df["COB_DATE"] = pd.to_datetime(df["COB_DATE"], errors="coerce")
    
    if "CONTRACT_START_DATE" in df.columns:
        df["CONTRACT_START_DATE"] = pd.to_datetime(df["CONTRACT_START_DATE"], errors="coerce")
    
    if "DELIVERY_DATE" in df.columns:
        df["DELIVERY_DATE"] = pd.to_datetime(df["DELIVERY_DATE"], errors="coerce")
    
    df["YEAR"] = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month

    fuel_mapping = {
        "ES": {"GASOLINA": "PETROL", "DIESEL": "DIESEL"},
        "SPAIN": {"GASOLINA": "PETROL", "DIESEL": "DIESEL"},
        "IT": {"BENZINA": "PETROL", "DIESEL": "DIESEL"},
        "ITALY": {"BENZINA": "PETROL", "DIESEL": "DIESEL"},
        "LUXEMBOURG": {"UNLEADED": "PETROL", "DIESEL": "DIESEL"},
        "GERMANY": {"UNLEADED": "PETROL", "DIESEL": "DIESEL"},
        "POLAND": {"UNLEADED": "PETROL", "DIESEL": "DIESEL"},
        "BELGIUM": {"UNLEADED": "PETROL", "DIESEL": "DIESEL"},
        "NETHERLANDS": {"UNLEADED": "PETROL", "DIESEL": "DIESEL"},
    }
    fuel_suffix_map = {
        "GASOLINA": "G", "BENZINA": "G", "PETROL": "G", "UNLEADED": "G", "DIESEL": "D",
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
        pd.Series(mapped_fuel2, index=df.index).fillna(pd.Series(mapped_fuel1, index=df.index)).fillna("MHEV NOT IDENTIFIED"),
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

    for col in ["POWER_CATEGORY", "POWER_CATEGORY_2", "POWER_CATEGORY_3"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("MHEV", "MHEV NOT IDENTIFIED")

    text_columns = ["COUNTRY", "NOVA_ASSET_STATUS", "BIKE_OR_CAR", "BRAND_UPDATE", "OEM_UPDATE", "CLS_VEHICLE_TYPE"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    # if "CLASS_CATALOG" in df.columns:
    #     vehicle_models = ["QASHQAI", "TUCSON", "X1", "XC40", "SPORTAGE", "T-ROC", "IBIZA", "ARONA", "XC60", "EXPRESS"]
    #     pattern = "|".join(vehicle_models)
    #     df["VEHICLE_MODEL_MAPED"] = df["CLASS_CATALOG"].astype(str).str.extract(f"({pattern})", flags=re.IGNORECASE, expand=False)
    # else:
    #     df["VEHICLE_MODEL_MAPED"] = pd.NA

    # if {"CONTRACT_END_DATE_AMENDED", "EXTENSION_DATE", "CONTRACT_END_DATE"}.issubset(df.columns):
    #     final_end_candidates = df[["CONTRACT_END_DATE_AMENDED", "EXTENSION_DATE", "CONTRACT_END_DATE"]].apply(pd.to_datetime, errors="coerce")
    #     df["CONTRACT_FINAL_END"] = final_end_candidates.max(axis=1)
    # else:
    #     df["CONTRACT_FINAL_END"] = pd.NaT

    return df


def load_market_dataset() -> pd.DataFrame:

    market_df = pd.read_parquet(DEFAULT_MARKET_DATA_PATH).copy()

    if "date" in market_df.columns:
        market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")

    # Detect Registration Type column (look for similar names)
    reg_col = pick_first_existing_column(market_df, ["Registration Type", "RegistrationType", "Vehicle Type", "Category"])
    
    if reg_col:
        # Standardize existing values to user's preferred labels
        mapping = {
            "Passenger Cars": "Passenger Cars",
            "Light Commercial Vehicles": "Light Commercial Vehicle",
            "Heavy Commercial Vehicles": "Heavy Commercial Vehicle"
        }
        market_df["Registration Type"] = market_df[reg_col].map(mapping).fillna(market_df[reg_col])
    
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


def percent_or_na_precision(value: float | None, decimals: int) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_volume(value: float | int | str | None) -> str:
    if value is None or pd.isna(value) or value == "N/A":
        return "N/A"
    try:
        return f"{int(float(value)):,}".replace(",", " ")
    except Exception:
        return str(value)


import base64

import os

def html_logo_block(class_name: str = "report-logo") -> str:
    if os.path.exists(str(LOGO_PATH)):
        try:
            with open(str(LOGO_PATH), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                mime_type = "image/jpeg" if str(LOGO_PATH).lower().endswith((".jpg", ".jpeg")) else "image/png"
                return f'<img class="{class_name}" src="data:{mime_type};base64,{encoded_string}" alt="Fleet Monitoring logo" />'
        except Exception:
            pass
    return f'<img class="{class_name}" src="{LOGO_PATH}" alt="Fleet Monitoring logo" />'

def get_logo_dash_src() -> str:
    if os.path.exists(str(LOGO_PATH)):
        try:
            with open(str(LOGO_PATH), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                mime_type = "image/jpeg" if str(LOGO_PATH).lower().endswith((".jpg", ".jpeg")) else "image/png"
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception:
            pass
    return ""


def html_status_dot(status: str) -> str:
    color = kpi_limit_color(status)
    return f'<span class="status-dot" style="background-color: {color};"></span>'

# =============================================================================
# 3. VIEW 1 - FLEET MONITORING (MAIN KPI CALCULATION FUNCTIONS)
# =============================================================================


def kpi_selected_volume(
    df: pd.DataFrame,
    country: str,
    year: int | str,
    month_value: int | str | None,
    status: str = "IN FLEET",
    bike_or_car: str = "CAR",
    date_mode: str = "NONE",
) -> int:
    subset = filter_base(df, country, year, month=month_value, status=status, bike_or_car=bike_or_car, date_mode=date_mode)
    unique_data = get_kpi_data(subset)
    return int(len(unique_data))



def apply_status_filter(df: pd.DataFrame, status: str | None) -> pd.DataFrame:
    if "NOVA_ASSET_STATUS" not in df.columns or status in (None, "ALL"):
        return df

    status_norm = str(status).strip().upper()
    status_series = df["NOVA_ASSET_STATUS"].astype(str).str.strip().str.upper()

    if status_norm == "IN FLEET":
       
        return df[status_series == "IN FLEET"]
    
    if status_norm == "ORDER":
        return df[status_series.str.contains("ORDER", na=False)]

    if status_norm == "SOLD":
        return df[status_series.str.contains("SOLD", na=False)]

    return df[status_series.str.contains(status_norm, na=False)]


def filter_status_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Helper for KPI 7 to handle status group filtering."""
    if df is None or df.empty:
        return df
    if group in (None, "ALL"):
        return df
    return apply_status_filter(df, group)
def filter_base(df: pd.DataFrame, country: str, year: int | str, month: str | int | None = "ALL", status: str = "IN FLEET", bike_or_car: str = "CAR", date_mode: str = "NONE") -> pd.DataFrame:
    subset = df.copy()


    if status and status != "ALL":
        subset = apply_status_filter(subset, status)

    if year and year != "ALL":
        subset = subset[subset["COB_DATE"].dt.year == int(year)]

    if month and month != "ALL":
        if isinstance(month, str) and month.startswith("Q"):
            try:
                q_num = int(month[1])
                months = list(range((q_num - 1) * 3 + 1, q_num * 3 + 1))
                subset = subset[subset["COB_DATE"].dt.month.isin(months)]
            except:
                pass
        else:
            try:
                subset = subset[subset["COB_DATE"].dt.month == int(month)]
            except:
                pass

    if bike_or_car and bike_or_car != "ALL":
        subset = subset[subset["BIKE_OR_CAR"] == bike_or_car]

    if country and country != "ALL":
        subset = subset[subset["COUNTRY"] == country]

    if date_mode in ("CONTRACT_START_DATE", "DELIVERY_DATE"):
        subset = subset[
            (subset[date_mode].dt.month == subset["COB_DATE"].dt.month) &
            (subset[date_mode].dt.year == subset["COB_DATE"].dt.year)
        ]

    return subset




def available_months(df: pd.DataFrame, country: str, year: int | str) -> list[int]:
    subset = filter_base(df, country, year, status="IN FLEET", bike_or_car="CAR")
    return sorted(subset["MONTH"].dropna().astype(int).unique().tolist())


def latest_month(df: pd.DataFrame, country: str, year: int | str) -> int | None:
    months = available_months(df, country, year)
    if not months:
        return None
    return months[-1]


def kpi_lease_under_25(df, country, year, month, bike_or_car="CAR", date_mode="NONE", status="IN FLEET"):
    subset = filter_base(df, country, year, month=month, status=status, bike_or_car=bike_or_car, date_mode=date_mode)
    data = get_kpi_data(subset, "FINAL_CONTRACT_DURATION")
    if data.empty: return None
    res = (data < 25).mean() * 100
    return float(res) if pd.notna(res) else None


def kpi_lease_25_30(df, country, year, month, bike_or_car="CAR", date_mode="NONE", status="IN FLEET"):
    subset = filter_base(df, country, year, month=month, status=status, bike_or_car=bike_or_car, date_mode=date_mode)
    data = get_kpi_data(subset, "FINAL_CONTRACT_DURATION")
    if data.empty: return None
    res = ((data >= 25) & (data <= 30)).mean() * 100
    return float(res) if pd.notna(res) else None


def kpi_diesel_non_diesel(df, country, year, month, bike_or_car="CAR", date_mode="NONE", status="IN FLEET"):
    subset = filter_base(df, country, year, month=month, status=status, bike_or_car=bike_or_car, date_mode=date_mode)
    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY_2", "POWER_CATEGORY"])
    if not power_col: return None, None
    data = get_kpi_data(subset, power_col).astype(str).str.upper()
    if data.empty: return None, None
    diesel_share = (data == "DIESEL").mean() * 100
    return float(diesel_share), float(100 - diesel_share)


def kpi_hybrid_share(df, country, year, month, bike_or_car="CAR", date_mode="NONE", status="IN FLEET"):
    subset = filter_base(df, country, year, month=month, status=status, bike_or_car=bike_or_car, date_mode=date_mode)
    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY"])
    if not power_col: return None
    data = get_kpi_data(subset, power_col).astype(str).str.upper()
    if data.empty: return None
    mask = data.isin(["FULL HYBRID", "PLUG-IN HYBRID"])
    res = mask.mean() * 100
    return float(res) if pd.notna(res) else None


def kpi_ev_share(df, country, year, month, bike_or_car="CAR", date_mode="NONE", status="IN FLEET"):
    subset = filter_base(df, country, year, month=month, status=status, bike_or_car=bike_or_car, date_mode=date_mode)
    power_col = pick_first_existing_column(subset, ["POWER_CATEGORY"])
    if not power_col: return None
    data = get_kpi_data(subset, power_col).astype(str).str.upper()
    if data.empty: return None
    res = (data == "ELECTRIC").mean() * 100
    return float(res) if pd.notna(res) else None


def kpi_pv_lcv(
    df: pd.DataFrame,
    country: str,
    year: int | str,
    month: int,
    bike_or_car: str = "CAR",
    date_mode: str = "NONE",
    status: str = "IN FLEET",
) -> tuple[float | None, float | None]:
    current_status = status if date_mode == "NONE" else "ALL"
    subset = filter_base(df, country, year, status=current_status, bike_or_car=bike_or_car)

    if date_mode in ("CONTRACT_START_DATE", "DELIVERY_DATE"):
        if year != "ALL":
            subset = subset[subset[date_mode].dt.year == int(year)]
        if month and month != "ALL":
            subset = subset[subset[date_mode].dt.month == int(month)]

    if month not in (None, "ALL"):
        subset = subset[subset["MONTH"] == int(month)]
        
    data = get_kpi_data(subset, 'CLS_VEHICLE_TYPE').astype(str).str.upper()
    if data.empty:
        return None, None
    pv_share_raw = (data == "PV").mean() * 100
    if pd.isna(pv_share_raw):
        return None, None
    lcv_share_raw = data.isin(["LCV", "LV"]).mean() * 100
    if pd.isna(lcv_share_raw):
        return None, None
    return float(pv_share_raw), float(lcv_share_raw)


def detect_fuel_column(df: pd.DataFrame) -> str | None:
    return "POWER_CATEGORY_2"

# =============================================================================
# 5. VIEW 3 - EVOLUTION & FUEL ANALYSIS (KPI 7)
# =============================================================================

def kpi7_fuel_by_period(
    df: pd.DataFrame,
    country: str,
    status_group: str,
    metric_mode: str,
    period_mode: str,
    bike_or_car: str = "CAR",
    date_mode: str = "COB_DATE",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    subset = df.copy()

    if country != "ALL":
        subset = subset[subset["COUNTRY"] == country]

    if start_date and end_date:
        start_p = pd.Period(start_date, freq="M")
        end_p = pd.Period(end_date, freq="M")
        if start_p > end_p:
            start_p, end_p = end_p, start_p
        
        cob_p = subset["COB_DATE"].dt.to_period("M")
        subset = subset[(cob_p >= start_p) & (cob_p <= end_p)]

    subset = filter_status_group(subset, status_group)
    if bike_or_car != "ALL" and "BIKE_OR_CAR" in subset.columns:
        subset = subset[subset["BIKE_OR_CAR"].astype(str) == bike_or_car]
    fuel_col = "POWER_CATEGORY_2"
    date_col = date_mode if date_mode in subset.columns else "COB_DATE"

    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.dropna(subset=[fuel_col, "MONTH", "YEAR"])
    if subset.empty:
        return pd.DataFrame(), "", "", ""

    subset = subset.copy()
    subset[fuel_col] = subset[fuel_col].astype(str).str.upper()
    subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
    subset = subset.dropna(subset=[date_col])

    if subset.empty:
        return pd.DataFrame(), "", "", ""

    if period_mode == "monthly":
        periods = subset[date_col].dt.to_period("M")
        subset["PERIOD_SORT"] = periods.astype(str)
        subset["PERIOD_LABEL"] = subset[date_col].dt.strftime("%Y-%m")
        x_title = "Month"
        period_label = "Monthly"
    elif period_mode == "quarterly":
        periods = subset[date_col].dt.to_period("Q")
        subset["PERIOD_SORT"] = periods.astype(str)
        subset["PERIOD_LABEL"] = periods.astype(str)
        x_title = "Quarter"
        period_label = "Quarterly"
    else:
        subset["PERIOD_SORT"] = subset[date_col].dt.year
        subset["PERIOD_LABEL"] = subset[date_col].dt.year.astype(str)
        x_title = "Year"
        period_label = "Yearly"

    subset = get_kpi_data(subset, extra_keys=["PERIOD_LABEL"])
    
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


def summary_month_label(year: int | str, month: int | str | None) -> str:
    if month is None or month == "ALL":
        return str(year)
    if isinstance(month, str) and month.startswith("Q"):
        return f"{month}-{year}"
    try:
        return f"{calendar.month_abbr[int(month)]}-{int(year)}"
    except:
        return f"{month}-{year}"


def resolve_month_value(country: str, year: int | str, month_value: int | str) -> int:
    if month_value in (None, "ALL"):
        resolved_month = latest_month(df, country, year)
        if resolved_month is not None:
            return int(resolved_month)
        return 1
    if isinstance(month_value, str) and month_value.startswith("Q"):
        try:
            q_num = int(month_value[1])
            return q_num * 3
        except:
            return 1
    return int(month_value)


def kpi_summary_description_map() -> dict[str, str]:
    return {k: v["description"] for k, v in KPI_REGISTRY.items()}


@lru_cache(maxsize=512)
def get_view1_metrics_cached(
    kpi1_country: str, kpi1_year: int | str, kpi1_month: int | str,
    kpi2_country: str, kpi2_year: int | str, kpi2_month: int | str,
    kpi3_country: str, kpi3_year: int | str, kpi3_month: int | str,
    kpi4_country: str, kpi4_year: int | str, kpi4_month: int | str,
    kpi5_country: str, kpi5_year: int | str, kpi5_month: int | str,
    kpi6_country: str, kpi6_year: int | str, kpi6_month: int | str,
    kpi_common_bike_or_car: str,
    date_mode: str = "NONE",
    kpi1_date_mode: str = "NONE",
    kpi2_date_mode: str = "NONE",
    kpi3_date_mode: str = "NONE",
    kpi4_date_mode: str = "NONE",
    kpi5_date_mode: str = "NONE",
    kpi6_date_mode: str = "NONE",
    granularity: str = "monthly",
    kpi1_status: str = "IN FLEET",
    kpi2_status: str = "IN FLEET",
    kpi3_status: str = "IN FLEET",
    kpi4_status: str = "IN FLEET",
    kpi5_status: str = "IN FLEET",
    kpi6_status: str = "IN FLEET",
    kpi1_vehicle: str = None,
    kpi2_vehicle: str = None,
    kpi3_vehicle: str = None,
    kpi4_vehicle: str = None,
    kpi5_vehicle: str = None,
    kpi6_vehicle: str = None,
) -> dict[str, object]:
    if granularity == "yearly":
        m1, m2, m3, m4, m5, m6 = "ALL", "ALL", "ALL", "ALL", "ALL", "ALL"
    else:
        m1 = resolve_month_value(kpi1_country, kpi1_year, kpi1_month)
        m2 = resolve_month_value(kpi2_country, kpi2_year, kpi2_month)
        m3 = resolve_month_value(kpi3_country, kpi3_year, kpi3_month)
        m4 = resolve_month_value(kpi4_country, kpi4_year, kpi4_month)
        m5 = resolve_month_value(kpi5_country, kpi5_year, kpi5_month)
        m6 = resolve_month_value(kpi6_country, kpi6_year, kpi6_month)

    dm1 = kpi1_date_mode if kpi1_date_mode != "NONE" else date_mode
    dm2 = kpi2_date_mode if kpi2_date_mode != "NONE" else date_mode
    dm3 = kpi3_date_mode if kpi3_date_mode != "NONE" else date_mode
    dm4 = kpi4_date_mode if kpi4_date_mode != "NONE" else date_mode
    dm5 = kpi5_date_mode if kpi5_date_mode != "NONE" else date_mode
    dm6 = kpi6_date_mode if kpi6_date_mode != "NONE" else date_mode

    vc1 = kpi1_vehicle or kpi_common_bike_or_car
    vc2 = kpi2_vehicle or kpi_common_bike_or_car
    vc3 = kpi3_vehicle or kpi_common_bike_or_car
    vc4 = kpi4_vehicle or kpi_common_bike_or_car
    vc5 = kpi5_vehicle or kpi_common_bike_or_car
    vc6 = kpi6_vehicle or kpi_common_bike_or_car

    s1 = kpi1_status or "IN FLEET"
    s2 = kpi2_status or "IN FLEET"
    s3 = kpi3_status or "IN FLEET"
    s4 = kpi4_status or "IN FLEET"
    s5 = kpi5_status or "IN FLEET"
    s6 = kpi6_status or "IN FLEET"

    return {
        "kpi1": kpi_lease_under_25(df, kpi1_country, kpi1_year, m1, vc1, dm1, s1),
        "kpi2": kpi_lease_25_30(df, kpi2_country, kpi2_year, m2, vc2, dm2, s2),
        "diesel_non": kpi_diesel_non_diesel(df, kpi3_country, kpi3_year, m3, vc3, dm3, s3),
        "hybrid": kpi_hybrid_share(df, kpi4_country, kpi4_year, m4, vc4, dm4, s4),
        "ev": kpi_ev_share(df, kpi5_country, kpi5_year, m5, vc5, dm5, s5),
        "pv_lcv": kpi_pv_lcv(df, kpi6_country, kpi6_year, m6, vc6, dm6, s6),
        "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6,
        "kpi1_volume": kpi_selected_volume(df, kpi1_country, kpi1_year, m1, status=s1, bike_or_car=vc1, date_mode=dm1),
        "kpi2_volume": kpi_selected_volume(df, kpi2_country, kpi2_year, m2, status=s2, bike_or_car=vc2, date_mode=dm2),
        "kpi3_volume": kpi_selected_volume(df, kpi3_country, kpi3_year, m3, status=s3, bike_or_car=vc3, date_mode=dm3),
        "kpi4_volume": kpi_selected_volume(df, kpi4_country, kpi4_year, m4, status=s4, bike_or_car=vc4, date_mode=dm4),
        "kpi5_volume": kpi_selected_volume(df, kpi5_country, kpi5_year, m5, status=s5, bike_or_car=vc5, date_mode=dm5),
        "kpi6_volume": kpi_selected_volume(df, kpi6_country, kpi6_year, m6, status=s6, bike_or_car=vc6, date_mode=dm6),
    }


@lru_cache(maxsize=256)
def get_kpi7_cached(
    country: str,
    start_date: str | None,
    end_date: str | None,
    status_group: str,
    metric_mode: str,
    period_mode: str,
    bike_or_car: str,
    date_mode: str,
) -> tuple[pd.DataFrame, str, str, str]:
    return kpi7_fuel_by_period(df, country, status_group, metric_mode, period_mode, bike_or_car, date_mode, start_date, end_date)


@lru_cache(maxsize=512)
def get_kpi8_cached(country: str, year: int | str, asset_status: str, metric_mode: str, bike_or_car: str, date_mode: str, period_mode: str = "monthly") -> tuple[pd.DataFrame, str, str, str]:
    return kpi8_production_ytd(df, country, year, asset_status, metric_mode, bike_or_car, date_mode, period_mode)


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
    signal_dot = {
        "green": "🟢",
        "orange": "🟠",
        "red": "🔴",
    }
    for row in rows:
        result_text = row["result_text"]
        signal = str(row["signal"])
        row_index = len(table_rows)
        dot = signal_dot.get(signal, "") if row_index < 2 else ""
        result_value = f"{dot} {result_text}" if dot else str(result_text)

        table_rows.append(
            {
                "Asset Risk, Financed Fleet": str(row["label"]),
                "Country": str(row["country"]),
                "Period": str(row["period"]),
                "Result": result_value,
                "Volume": format_volume(row["volume"]),
                "Unit": str(row["unit"]),
                "Comment": str(row["comment"]),
            }
        )

    columns = [
        {"name": "Asset Risk, Financed Fleet", "id": "Asset Risk, Financed Fleet"},
        {"name": "Country", "id": "Country"},
        {"name": "Period", "id": "Period"},
        {"name": "Result", "id": "Result"},
        {"name": "Volume", "id": "Volume"},
        {"name": "Unit", "id": "Unit"},
        {"name": "Comment", "id": "Comment"},
    ]

    return dash_table.DataTable(
        id="v1-summary-table",
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
            "textAlign": "center",
        },
        style_data_conditional=cast(Any, row_styles + [
            {"if": {"column_id": "Asset Risk, Financed Fleet"}, "textAlign": "left"},
            {"if": {"column_id": "Country"}, "textAlign": "center"},
            {"if": {"column_id": "Period"}, "textAlign": "center"},
            {"if": {"column_id": "Result"}, "textAlign": "center"},
            {"if": {"column_id": "Volume"}, "textAlign": "center"},
            {"if": {"column_id": "Unit"}, "textAlign": "center"},
            {"if": {"column_id": "Comment"}, "textAlign": "center"},
        ]),
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
    fuels = [str(fuel) for fuel in pivot.index]
    color_by_fuel = fuel_color_map(fuels)
    ordered_fuels = sorted(fuels, key=lambda name: (fuel_label_priority(name), fuels.index(name)))
    for fuel in ordered_fuels:
        color = color_by_fuel[fuel]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=pivot.loc[fuel].tolist(),
                name=str(fuel),
                marker_color=color,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=pivot.loc[fuel].tolist(),
                name=str(fuel),
                mode="lines",
                showlegend=False,
                line={"color": color, "width": 2.2, "dash": "dot"},
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="Fuel Type",
        height=520,
        template="plotly_white",
    )
    return fig


def detect_vehicle_type_column(df: pd.DataFrame) -> str | None:
    return pick_first_existing_column(df, ["VEHICLE_TYPE_TEMP", "CLS_VEHICLE_TYPE"])


def build_table(df_table: pd.DataFrame, page_size: int = 15, heatmap: bool = False, table_id: str = None) -> dash_table.DataTable:
    columns = cast(Any, [{"name": str(c), "id": str(c)} for c in df_table.columns])
    data = cast(Any, df_table.to_dict("records"))
    
    # Base row styling
    style_data_conditional = cast(list[Any], [{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}])
    
    # Heatmap intensity logic
    if heatmap:
        # User colors: Red (FF0000), Orange (FFC000), Yellow (FFEB9C), Green (C6EFCE)
        # Assuming values are percentages (0-100) or relative intensities
        numeric_cols = [c for c in df_table.columns if c not in ["YEAR", "PERIOD", "MONTH", "Date", "COUNTRY"]]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df_table[col]):
                continue
            
            style_data_conditional.extend([
                {"if": {"column_id": str(col), "filter_query": f"{{{col}}} >= 35"}, "backgroundColor": "#FF0000", "color": "white"},
                {"if": {"column_id": str(col), "filter_query": f"{{{col}}} >= 20 && {{{col}}} < 35"}, "backgroundColor": "#FFC000", "color": "black"},
                {"if": {"column_id": str(col), "filter_query": f"{{{col}}} >= 10 && {{{col}}} < 20"}, "backgroundColor": "#FFEB9C", "color": "black"},
                {"if": {"column_id": str(col), "filter_query": f"{{{col}}} > 0 && {{{col}}} < 10"}, "backgroundColor": "#C6EFCE", "color": "black"},
            ])

    total_row_index = None
    first_column = str(df_table.columns[0]) if len(df_table.columns) > 0 else None
    for idx, row in enumerate(data):
        if first_column is not None and str(row.get(first_column, "")).startswith("TOTAL"):
            total_row_index = idx
            break
        if row.get("YEAR") == "TOTAL" or row.get("MONTH") == "TOTAL" or row.get("Date") == "TOTAL":
            total_row_index = idx
            break
    
    if total_row_index is not None:
        style_data_conditional.append({"if": {"row_index": total_row_index}, "fontWeight": "bold", "backgroundColor": "#F2F2F2"})
    
    column_styles = [{"if": {"column_id": str(column)}, "textAlign": "center"} for column in df_table.columns]
    if len(df_table.columns) > 0:
        column_styles[0] = {"if": {"column_id": str(df_table.columns[0])}, "textAlign": "left"}
    num_cols = len(df_table.columns)
    if num_cols > 15:
        tb_font_size = "10px"
        tb_padding = "4px 4px"
        tb_min_width = "40px"
    elif num_cols > 8:
        tb_font_size = "11px"
        tb_padding = "6px 8px"
        tb_min_width = "70px"
    else:
        tb_font_size = "13px"
        tb_padding = "8px 10px"
        tb_min_width = "110px"

    return dash_table.DataTable(
        id=table_id,
        data=data,
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#102a43", "color": "white", "fontWeight": "700", "border": "none"},
        style_cell={
            "padding": tb_padding,
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": tb_font_size,
            "border": "1px solid #e6eaf0",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": tb_min_width,
            "textAlign": "center",
        },
        style_data_conditional=cast(Any, style_data_conditional + column_styles),
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
        :root {{
            --ink: #102a43;
            --muted: #627d98;
            --line: #d9e2ec;
            --line-soft: #e6eaf0;
            --bg: #f7f9fc;
            --card: #ffffff;
            --accent: #1f6feb;
            --accent-soft: #eaf2ff;
        }}
        body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; color: var(--ink); background: linear-gradient(180deg, #f3f7fb 0%, var(--bg) 220px); }}
        .page {{ max-width: 1240px; margin: 0 auto; background: #fff; padding: 28px; border: 1px solid var(--line-soft); border-radius: 16px; box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08); }}
        h1, h2, h3 {{ color: #102a43; }}
        .report-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid var(--line-soft); }}
        .report-logo {{ width: 72px; height: 72px; object-fit: contain; border-radius: 0; background: #fff; border: 1px solid var(--line-soft); box-shadow: 0 6px 18px rgba(16, 42, 67, 0.08); }}
        .report-heading {{ display: flex; flex-direction: column; gap: 4px; }}
        .report-title {{ margin: 0; font-size: 30px; line-height: 1.1; }}
        .report-subtitle {{ margin: 0; color: var(--muted); font-size: 13px; }}
        .kpi-cards-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); grid-auto-rows: 1fr; gap: 12px; margin-top: 14px; align-items: stretch; }}
        .kpi-result-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px 8px; box-shadow: 0 6px 18px rgba(16, 42, 67, 0.04); display: flex; flex-direction: column; height: 100%; box-sizing: border-box; }}
        .kpi-card-top {{ display: flex; flex-direction: column; align-items: center; justify-content: flex-start; gap: 0; margin-bottom: 12px; text-align: center; width: 100%; }}
        .kpi-title {{ font-size: 14px; font-weight: 800; color: var(--ink); letter-spacing: 0.05em; text-transform: uppercase; line-height: 1.1; width: 100%; padding-bottom: 4px; border-bottom: 1px solid var(--line-soft); }}
        .kpi-result-value {{ font-size: 28px; font-weight: 800; color: #163e63; line-height: 1.05; display: flex; align-items: center; justify-content: center; gap: 8px; margin: 12px 0 4px; padding: 4px 0; width: 100%; }}
        .kpi-result-value .kpi-value-text {{ color: #163e63; }}
        .status-dot {{ width: 10px; height: 10px; border-radius: 999px; flex: 0 0 auto; display: inline-block; }}
        .kpi-meta-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 14px; margin-top: 4px; padding-bottom: 6px; }}
        .kpi-meta-column {{ display: flex; flex-direction: column; gap: 6px; }}
        .kpi-meta-line {{ display: flex; justify-content: space-between; gap: 10px; font-size: 11px; color: var(--muted); line-height: 1.25; }}
        .kpi-meta-key {{ font-weight: 700; color: var(--ink); white-space: nowrap; }}
        .kpi-meta-value {{ text-align: right; word-break: break-word; overflow-wrap: anywhere; }}
        .kpi-legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; margin-top: 4px; padding-top: 6px; border-top: 1px solid var(--line-soft); font-size: 10px; color: var(--muted); }}
        .kpi-legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
        .kpi-legend-chip {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; }}
        .table-wrap {{ width: 100%; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; table-layout: fixed; }}
        th, td {{ border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; text-align: center; word-break: break-word; overflow-wrap: anywhere; font-size: 12px; }}
        th:first-child, td:first-child {{ text-align: left; }}
        th {{ background: #102a43; color: #fff; }}
        .table-dense th, .table-dense td {{ padding: 4px 6px; font-size: 10px; }}
        .table-ultra-dense th, .table-ultra-dense td {{ padding: 2px 3px; font-size: 8px; }}
        .section {{ margin-bottom: 28px; }}
        .muted {{ color: var(--muted); }}
        @media (max-width: 1000px) {{
            .kpi-cards-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        @media (max-width: 768px) {{
            body {{ margin: 14px; }}
            .page {{ padding: 16px; }}
            .report-header {{ align-items: flex-start; }}
            .report-logo {{ width: 56px; height: 56px; }}
            .report-title {{ font-size: 24px; }}
            .kpi-cards-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class=\"page\">
        <div class=\"report-header\">
            {html_logo_block()}
            <div class=\"report-heading\">
                <h1 class=\"report-title\">{title}</h1>
            </div>
        </div>
        {body}
    </div>
</body>
</html>"""


def build_kpi7_metadata_section(
    country: str,
    status_group: str,
    metric_mode: str,
    period_mode: str,
    bike_or_car: str,
    start_date: str | None,
    end_date: str | None,
    figure: go.Figure,
) -> str:
    """Build a simple KPI 7 metadata section with refined title structure."""
    if start_date and end_date:
        date_text = f"{start_date} to {end_date}"
    else:
        min_cob = df["COB_DATE"].min()
        max_cob = df["COB_DATE"].max()
        min_str = min_cob.strftime("%Y-%m-%d") if pd.notna(min_cob) else "N/A"
        max_str = max_cob.strftime("%Y-%m-%d") if pd.notna(max_cob) else "N/A"
        date_text = f"{min_str} to {max_str}"

    title_mode = "Share" if metric_mode == "share" else "Volume"
    main_title = f"Fuel Type {title_mode.lower()} - {status_group} - {title_mode} - {period_mode.capitalize()}"
    subtitle = f"{country} | {date_text} | {bike_or_car}"
    
    figure_copy = go.Figure(figure)
    figure_copy.update_layout(title=None, margin=dict(t=30))
    
    return f"<div class=\"section\"><h2>{html_escape(main_title)}</h2><p class=\"muted\">{html_escape(subtitle)}</p>{figure_to_html_block(figure_copy)}</div>"


def build_kpi_cards_section(title: str, cards: list[dict[str, object]]) -> str:
    if not cards:
        return f"<div class=\"section\"><h2>{html_escape(title)}</h2><p class=\"muted\">Aucune donnée disponible.</p></div>"

    html_cards: list[str] = []
    grid_style = "" 
    if len(cards) == 1:
        grid_style = "grid-template-columns: minmax(320px, 560px); justify-content: center; justify-items: center; margin: 14px auto 0;"
    elif len(cards) == 2 or len(cards) == 4:
        grid_style = "grid-template-columns: repeat(2, 1fr);"

    for card in cards:
        label = html_escape(str(card.get("label", "KPI")))
        result = html_escape(str(card.get("result", "N/A")))
        status = str(card.get("status", "neutral"))
        params = card.get("params", [])

        params_list = cast(list[tuple[str, object]], params)
        midpoint = (len(params_list) + 1) // 2
        left_params = params_list[:midpoint]
        right_params = params_list[midpoint:]
        left_params_html = "<div class=\"kpi-meta-column\">" + "".join(
            f"<div class=\"kpi-meta-line\"><span class=\"kpi-meta-key\">{html_escape(str(name))}</span><span class=\"kpi-meta-value\">{html_escape(str(value))}</span></div>"
            for name, value in left_params
        ) + "</div>"
        right_params_html = "<div class=\"kpi-meta-column\">" + "".join(
            f"<div class=\"kpi-meta-line\"><span class=\"kpi-meta-key\">{html_escape(str(name))}</span><span class=\"kpi-meta-value\">{html_escape(str(value))}</span></div>"
            for name, value in right_params
        ) + "</div>"

        legend_html = ""
        if bool(card.get("show_legend", False)):
            legend_html = (
                "<div class=\"kpi-legend\">"
                "<span class=\"kpi-legend-item\"><span class=\"kpi-legend-chip\" style=\"background:#2f855a\"></span>Green: below limit</span>"
                "<span class=\"kpi-legend-item\"><span class=\"kpi-legend-chip\" style=\"background:#f0a202\"></span>Orange: near limit</span>"
                "<span class=\"kpi-legend-item\"><span class=\"kpi-legend-chip\" style=\"background:#d64545\"></span>Red: above limit</span>"
                "</div>"
            )

        html_cards.append(
            "<div class=\"kpi-result-card\">"
            "<div class=\"kpi-card-top\">"
            f"<div class=\"kpi-title\">{label}</div>"
            f"<div class=\"kpi-result-value\">{html_status_dot(status) if status != 'neutral' else ''}{f'<span class=\"kpi-value-text\">{result}</span>' if result and result != 'N/A' else ''}</div>"
            "</div>"
            f"<div class=\"kpi-meta-grid\">{left_params_html}{right_params_html}</div>"
            f"{legend_html}"
            "</div>"
        )

    return (
        "<div class=\"section\">"
        f"<h2>{html_escape(title)}</h2>"
        f"<div class=\"kpi-cards-grid\" style=\"{grid_style}\">{''.join(html_cards)}</div>"
        "</div>"
    )


def build_html_table_from_df(df_table: pd.DataFrame, title: str) -> str:
    if df_table.empty:
        return f"<div class=\"section\"><h2>{title}</h2><p class=\"muted\">Aucune donnée disponible.</p></div>"
    table_df = df_table.copy()
    if "signal" in table_df.columns and "result" in table_df.columns:
        signal_style = {
            "green": "color: #2f855a; margin-right: 4px;",
            "orange": "color: #f0a202; margin-right: 4px;",
            "red": "color: #d64545; margin-right: 4px;"
        }
        
        def format_result_with_signal(row):
            sig = str(row.get("signal", "neutral"))
            val = str(row.get("result", ""))
            if sig in signal_style:
                return f"<span style='{signal_style[sig]}'>●</span> {val}"
            return val

        table_df["result"] = table_df.apply(format_result_with_signal, axis=1)
        table_df = table_df.drop(columns=["signal"])

    if "description" in table_df.columns:
        ordered = [column for column in table_df.columns if column != "description"] + ["description"]
        table_df = table_df[ordered]

    if not table_df.empty and len(table_df.columns) > 0:
        first_column = str(table_df.columns[0])
        total_mask = table_df[first_column].astype(str) == "TOTAL"
        if total_mask.any():
            total_index = table_df.index[total_mask][0]
            for column in table_df.columns:
                value = table_df.at[total_index, column]
                if not str(value).startswith("<strong>"):
                    table_df.at[total_index, column] = f"<strong>{value}</strong>"

    num_cols = len(table_df.columns)
    table_class = "table-wrap"
    if num_cols > 15:
        table_class = "table-wrap table-ultra-dense"
    elif num_cols > 8:
        table_class = "table-wrap table-dense"

    return f"<div class=\"section\"><h2>{title}</h2><div class=\"{table_class}\">{table_df.to_html(index=False, escape=False)}</div></div>"


def format_view2_table_for_display(table_df: pd.DataFrame) -> pd.DataFrame:
    display_df = table_df.copy()
    if "YEAR" in display_df.columns:
        display_df = display_df.drop(columns=["YEAR"])
    if "MONTH" in display_df.columns:
        display_df = display_df.rename(columns={"MONTH": "Date"})
    ordered_columns = [column for column in ["Date"] if column in display_df.columns]
    ordered_columns.extend([column for column in display_df.columns if column != "Date"])
    return display_df[ordered_columns].copy()

# Consolidated with the one below to avoid confusion

def build_view3_download_report(country: str, start_date: str, end_date: str, bike_or_car: str) -> str:
    pivot_v3, y_title, x_title, period_label = get_view3_ageing_cached(country, start_date, end_date, bike_or_car)
    fig_v3 = figure_from_pivot(pivot_v3, y_title, x_title, f"Ageing report - {country} - {bike_or_car}")
    summary_text = f"<p class=\"muted\">{country} | {start_date} to {end_date} | {bike_or_car}</p>"
    return html_report_document(f"Fleet Monitoring - View 3 - Ageing report", [
        f"<div class=\"section\"><h2>Ageing report</h2>{summary_text}</div>",
        f"<div class=\"section\">{figure_to_html_block(fig_v3)}</div>",
        build_html_table_from_df(pivot_v3, "Ageing Table")
    ])


def build_kpi7_table_with_total(
    kpi7_pivot: pd.DataFrame,
    metric_mode: str,
    country: str,
    status_group: str,
    period_mode: str,
    bike_or_car: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    table_df = kpi7_pivot.copy()
    metric_cols = table_df.columns.tolist()

    if metric_mode == "share":
        volume_pivot, _, _, _ = kpi7_fuel_by_period(df, country, status_group, "volume", period_mode, bike_or_car, "COB_DATE", start_date, end_date)
        if not volume_pivot.empty:
            volume_totals = volume_pivot.sum(axis=0).reindex(metric_cols, fill_value=0)
            total_row = pd.DataFrame([volume_totals], index=["TOTAL"])
            table_df = pd.concat([table_df, total_row])
    else:
        total_row = pd.DataFrame([table_df[metric_cols].sum(axis=0)], index=["TOTAL"])
        table_df = pd.concat([table_df, total_row])

    table_df.index.name = "Fuel type"
    table_df = table_df.reset_index()

    total_mask = table_df["Fuel type"].astype(str) == "TOTAL"
    if metric_mode == "share":
        for column in [c for c in table_df.columns if c != "Fuel type"]:
            display_col = table_df[column].astype("object")
            display_col.loc[~total_mask] = table_df.loc[~total_mask, column].map(
                lambda value: percent_or_na_precision(cast(float | None, value), 1) if pd.notna(value) else "N/A"
            )
            display_col.loc[total_mask] = table_df.loc[total_mask, column].map(
                lambda value: format_volume(value)
            )
            table_df[column] = display_col
    else:
        for column in [c for c in table_df.columns if c != "Fuel type"]:
            table_df[column] = table_df[column].map(lambda value: format_volume(value))

    return table_df


def should_show_total_column(df: pd.DataFrame, metric_cols: list[str], share_mode: bool = False) -> bool:
    if df.empty or not metric_cols:
        return False
    if share_mode:
        return len(metric_cols) > 1
    return True

def build_view2_table_with_total(table_df: pd.DataFrame, metric_mode: str) -> pd.DataFrame:
    display_df = table_df.copy()
    # Exclude non-metric columns from summation to avoid concatenation bugs (e.g. Q1Q2Q3Q4) and crashes in share mode
    exclude_cols = ["YEAR", "MONTH", "PERIOD", "Date"]
    metric_cols = [column for column in display_df.columns if column not in exclude_cols]

    if not display_df.empty:
        total_row = {"YEAR": "TOTAL", "MONTH": "TOTAL", "PERIOD": "TOTAL"}
        if "Date" in display_df.columns:
            total_row["Date"] = "TOTAL"
            
        for column in metric_cols:
            total_row[column] = display_df[column].sum()
        display_df = pd.concat([display_df, pd.DataFrame([total_row])], ignore_index=True)

    if metric_mode == "share":
        # Note: Summing percentages might be > 100 due to rounding, 
        # but we use format_volume for the total row if it's very large or just stick to percent_or_na
        total_idx = display_df.index[-1]
        for column in metric_cols:
            # Format rows as percentages
            display_df.loc[display_df.index[:-1], column] = display_df.loc[display_df.index[:-1], column].map(
                lambda v: percent_or_na_precision(cast(float | None, v), 1) if pd.notna(v) else "N/A"
            )
            # For the total row, if it's share mode, it should ideally show 100% or volume
            # To match kpi7 requested style (bold total), we'll format it as percentage for now 
            # unless it's way above 100 (which shouldn't happen here as it's a simple sum of shares)
            display_df.loc[total_idx, column] = percent_or_na_precision(cast(float | None, display_df.loc[total_idx, column]), 1)
    else:
        for column in metric_cols:
            display_df[column] = display_df[column].map(lambda value: format_volume(value))

    return display_df


def figure_to_html_block(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})


def build_view1_download_report(
    kpi1_country, kpi1_year, kpi1_month, kpi1_val,
    kpi2_country, kpi2_year, kpi2_month, kpi2_val,
    kpi3_country, kpi3_year, kpi3_month, kpi3_val,
    kpi4_country, kpi4_year, kpi4_month, kpi4_val,
    kpi5_country, kpi5_year, kpi5_month, kpi5_val,
    kpi6_country, kpi6_year, kpi6_month, kpi6_val,
    kpi_common_bike_or_car,
    kpi7_country, kpi7_start_date, kpi7_end_date,
    kpi7_status_group, kpi7_metric_mode, kpi7_period_mode,
    kpi7_bike_or_car,
    summary_data: list[dict],
    kpi7_fig: dict,
    kpi7_table_data: list[dict],
    kpi7_table_cols: list[dict],
) -> str:
    kpi_cards = []
    kpi_params = {
        "kpi1": (kpi1_country, kpi1_year, kpi1_month, kpi1_val),
        "kpi2": (kpi2_country, kpi2_year, kpi2_month, kpi2_val),
        "kpi3": (kpi3_country, kpi3_year, kpi3_month, kpi3_val),
        "kpi4": (kpi4_country, kpi4_year, kpi4_month, kpi4_val),
        "kpi5": (kpi5_country, kpi5_year, kpi5_month, kpi5_val),
        "kpi6": (kpi6_country, kpi6_year, kpi6_month, kpi6_val),
    }

    for kid, config in KPI_REGISTRY.items():
        c_p, y_p, m_p, val_p = kpi_params.get(kid, (None,None,None,"N/A"))
        period_lbl = summary_month_label(y_p, None if m_p in (None, 'ALL') else m_p)
        
        # Try to determine status from val_p if possible (heuristic for coloring)
        status = "neutral"
        if isinstance(val_p, str) and "%" in val_p:
            try:
                num_val = float(val_p.split("%")[0].replace(",",""))
                status = kpi_limit_status(num_val, config["limit"])
            except: pass

        kpi_cards.append({
            "label": config["label"],
            "result": val_p,
            "status": status,
            "show_legend": True,
            "params": [
                ("Country", c_p),
                ("Period", period_lbl),
                ("Vehicle type", kpi_common_bike_or_car),
                ("Asset status", "IN FLEET"),
            ],
        })

    summary_df = pd.DataFrame(summary_data)
    kpi7_table_df = pd.DataFrame(kpi7_table_data, columns=[c["id"] for c in kpi7_table_cols])

    return html_report_document(
        "Fleet Monitoring - View 1 - Main KPIs",
        [
            build_kpi_cards_section("Main KPIs", kpi_cards),
            build_html_table_from_df(summary_df, "KPI summary table"),
            build_kpi7_metadata_section(kpi7_country, kpi7_status_group, kpi7_metric_mode, kpi7_period_mode, kpi7_bike_or_car, kpi7_start_date, kpi7_end_date, kpi7_fig),
            build_html_table_from_df(
                kpi7_table_df,
                f"Fuel Type {'Share' if kpi7_metric_mode == 'share' else 'Volume'} Table - {kpi7_status_group} - {'Share' if kpi7_metric_mode == 'share' else 'Volume'} - {kpi7_period_mode.capitalize()}",
            ),
        ],
    )

def build_view2_download_report(
    country,
    year,
    asset_status,
    metric_mode,
    bike_or_car,
    v2_fig: dict,
    v2_table_data: list[dict],
    v2_table_cols: list[dict],
) -> str:
    params = [
        ("Country", country),
        ("Year", year),
        ("Status", asset_status),
        ("Metric", metric_mode),
        ("Vehicle type", bike_or_car),
    ]

    v2_table_df = pd.DataFrame(v2_table_data, columns=[c["id"] for c in v2_table_cols])
    
    # Extract total from table if available (BEFORE removing the Period column)
    kpi8_result = "N/A"
    if not v2_table_df.empty:
        first_col = v2_table_df.columns[0]
        total_row = v2_table_df[v2_table_df[first_col].astype(str).str.contains("TOTAL", na=False)]
        if not total_row.empty:
            kpi8_result = total_row.iloc[0, 1] if len(total_row.columns) > 1 else "TOTAL"

    # Remove the first column (Period) as requested
    if not v2_table_df.empty:
        v2_table_df = v2_table_df.iloc[:, 1:]

    title_mode_cap = "Share" if metric_mode == "share" else "Volume"
    main_title = f"Production by Fuel Type - {asset_status} - {title_mode_cap}"
    subtitle = f"{country} | {year} | {bike_or_car}"
    summary_text = f"<p class=\"muted\">{html_escape(subtitle)}</p>"

    # Ensure figure title is clean for report
    fig_obj = go.Figure(v2_fig)
    fig_obj.update_layout(title=None, margin=dict(t=30))

    return html_report_document(
        f"Fleet Monitoring - View 2 - {main_title}",
        [
            f"<div class=\"section\"><h2>{html_escape(main_title)}</h2>{summary_text}</div>",
            f"<div class=\"section\">{figure_to_html_block(fig_obj)}</div>",
            build_html_table_from_df(
                v2_table_df,
                f"Production by Fuel Type Table - {asset_status} - {title_mode_cap}",
            ),
        ],
    )

# =============================================================================
# 4. VIEW 2 - PRODUCTION TRENDS (KPI 8)
# =============================================================================

def kpi8_production_ytd(
    df: pd.DataFrame,
    country: str,
    year: int | str,
    asset_status: str = "ALL",
    metric_mode: str = "volume",
    bike_or_car: str = "CAR",
    date_mode: str = "CONTRACT_START_DATE",
    period_mode: str = "monthly",
) -> tuple[pd.DataFrame, str, str, str]:
    subset = filter_base(
        df,
        country,
        year,
        status="ALL",
        bike_or_car="CAR",
    ).copy()

    if asset_status != "ALL":
        subset = subset[subset["NOVA_ASSET_STATUS"] == asset_status]
    if bike_or_car != "ALL" and "BIKE_OR_CAR" in subset.columns:
        subset = subset[subset["BIKE_OR_CAR"].astype(str) == bike_or_car]

    date_col = "DELIVERY_DATE" if date_mode == "DELIVERY_DATE" else "CONTRACT_START_DATE"
    subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
    subset = subset[
        (subset[date_col].dt.year == subset["YEAR"])
        & (subset[date_col].dt.month == subset["MONTH"])
    ].copy()

    subset = get_kpi_data(subset)

    if period_mode == "monthly":
        subset["PERIOD"] = subset["MONTH"].astype(str).str.zfill(2)
        period_label = "Month"
    elif period_mode == "quarterly":
        subset["PERIOD"] = "Q" + ((subset["MONTH"].astype(int) - 1) // 3 + 1).astype(str)
        period_label = "Quarter"
    else:
        subset["PERIOD"] = subset["YEAR"].astype(str)
        period_label = "Year"

    grouped = (
        subset.groupby(["PERIOD", "POWER_CATEGORY_3"])
        .size()
        .reset_index(name="VOLUME")
    )

    if grouped.empty:
        return pd.DataFrame(), "", "", ""

    if metric_mode == "share":
        grouped["TOTAL"] = grouped.groupby("PERIOD")["VOLUME"].transform("sum")
        grouped["SHARE"] = (grouped["VOLUME"] / grouped["TOTAL"] * 100).round(2)
        pivot = grouped.pivot(index="PERIOD", columns="POWER_CATEGORY_3", values="SHARE").fillna(0)
        y_title = "Share (%)"
    else:
        pivot = grouped.pivot(index="PERIOD", columns="POWER_CATEGORY_3", values="VOLUME").fillna(0)
        y_title = "Volume"

    pivot = pivot.reset_index().sort_values("PERIOD")
    return pivot, y_title, period_label, f"{year}"


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

    subset = get_kpi_data(subset, extra_keys=["Quarter"])
    
    grouped = subset.groupby(["NOVA_ASSET_STATUS", "Quarter"]).size().reset_index(name="VEHICLE_ID")
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

    subset = get_kpi_data(subset, extra_keys=["Quarter"])
    
    grouped = subset.groupby(["POWER_CATEGORY", "Quarter"]).size().reset_index(name="VEHICLE_ID")
    pivot = grouped.pivot(index="Quarter", columns="POWER_CATEGORY", values="VEHICLE_ID").fillna(0)
    return pivot, "Volume", "Quarter", f"{vehicle_type} by energy"


def apply_top_n_filter(grouped: pd.DataFrame, top_n: str, group_cols: list[str]) -> pd.DataFrame:
    if str(top_n).upper() != "ALL":
        try:
            n = int(top_n)
            return grouped.groupby(group_cols).head(n).reset_index(drop=True)
        except ValueError:
            pass
    return grouped

def kpi_count_share_quarterly(df: pd.DataFrame, asset_status: str, var_col: str, bike_or_car: str = "CAR", top_n: str = "1", period_mode: str = "quarterly") -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ]
    out = out.dropna(subset=["MONTH", "YEAR", var_col])
    if out.empty:
        return pd.DataFrame()

    out = out.copy()
    out["MONTH"] = out["MONTH"].astype(int)
    if period_mode == "monthly":
        out["PERIOD"] = out["MONTH"].astype(str).str.zfill(2)
    elif period_mode == "yearly":
        out["PERIOD"] = "ALL"
    else:
        out["PERIOD"] = "Q" + (((out["MONTH"] - 1) // 3) + 1).astype(str)

    out = get_kpi_data(out, extra_keys=["PERIOD"])

    grouped = out.groupby(["COUNTRY", var_col, "YEAR", "PERIOD"]).size().reset_index(name="VOLUME")
    grouped = grouped.sort_values(["COUNTRY", "YEAR", "PERIOD", "VOLUME"], ascending=[True, True, True, False])
    
    grouped["TOTAL"] = grouped.groupby(["COUNTRY", "YEAR", "PERIOD"])["VOLUME"].transform("sum")
    grouped["SHARE"] = grouped["VOLUME"].div(grouped["TOTAL"].where(grouped["TOTAL"] != 0, 1)).mul(100)

    return apply_top_n_filter(grouped, top_n, ["COUNTRY", "YEAR", "PERIOD"])


def kpi_count_share_quarterly_market(df: pd.DataFrame, var_col: str, top_n: str = "1", period_mode: str = "quarterly") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", var_col, "volume"])
    if out.empty:
        return pd.DataFrame()

    out["YEAR"] = out["date"].dt.year
    if period_mode == "monthly":
        out["PERIOD"] = out["date"].dt.month.astype(str).str.zfill(2)
    elif period_mode == "yearly":
        out["PERIOD"] = "ALL"
    else:
        out["PERIOD"] = out["date"].dt.to_period("Q").astype(str).str[-2:]

    grouped = out.groupby([var_col, "YEAR", "PERIOD"])["volume"].sum().reset_index(name="VOLUME")
    grouped = grouped.sort_values(["YEAR", "PERIOD", "VOLUME"], ascending=[True, True, False])
    
    grouped["TOTAL"] = grouped.groupby(["YEAR", "PERIOD"])["VOLUME"].transform("sum")
    grouped["SHARE"] = grouped["VOLUME"].div(grouped["TOTAL"].where(grouped["TOTAL"] != 0, 1)).mul(100)

    return apply_top_n_filter(grouped, top_n, ["YEAR", "PERIOD"])




def kpi_count_share_ytd_by_quarter(df: pd.DataFrame, asset_status: str, var_col: str, bike_or_car: str = "CAR", top_n: str = "1", period_mode: str = "quarterly") -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ]
    out = out.dropna(subset=["MONTH", "YEAR", var_col])

    if out.empty:
        return pd.DataFrame()

    out = out.copy()
    out["MONTH"] = out["MONTH"].astype(int)
    if period_mode == "monthly":
        out["PERIOD"] = out["MONTH"].astype(str).str.zfill(2)
    elif period_mode == "yearly":
        out["PERIOD"] = "ALL"
    else:
        out["PERIOD"] = "Q" + (((out["MONTH"] - 1) // 3) + 1).astype(str)
    
    out = get_kpi_data(out, extra_keys=["PERIOD"])

    out["VOLUME_INC"] = 1
    out = out.sort_values(["COUNTRY", var_col, "YEAR", "MONTH"])
    out["VOLUME_YTD"] = out.groupby(["COUNTRY", var_col, "YEAR"])["VOLUME_INC"].cumsum()

    grouped = (
        out.groupby(["COUNTRY", var_col, "YEAR", "PERIOD"], as_index=False)
        .agg(VOLUME_YTD=("VOLUME_YTD", "max"))
    )
    grouped = grouped.sort_values(["COUNTRY", "YEAR", "PERIOD", "VOLUME_YTD"], ascending=[True, True, True, False])
    
    grouped["TOTAL_YTD"] = grouped.groupby(["COUNTRY", "YEAR", "PERIOD"])["VOLUME_YTD"].transform("sum")
    grouped["SHARE_YTD"] = grouped["VOLUME_YTD"].div(grouped["TOTAL_YTD"].where(grouped["TOTAL_YTD"] != 0, 1)).mul(100)

    return apply_top_n_filter(grouped, top_n, ["COUNTRY", "YEAR", "PERIOD"])


def plot_kpi_share(df_plot: pd.DataFrame, col: str, period_mode: str = "quarterly") -> go.Figure:
    out = df_plot.copy()
    if out.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Concentration by {col}")
        return fig

    if period_mode == "yearly":
        out["TIME"] = out["YEAR"].astype(str)
    else:
        out["TIME"] = out["YEAR"].astype(str) + "-" + out["PERIOD"].astype(str)

    if "SHARE" in out.columns:
        agg = out.groupby(["TIME", col], as_index=False).agg({"VOLUME": "sum", "SHARE": "sum"})
    elif "SHARE_YTD" in out.columns:
        agg = out.groupby(["TIME", col], as_index=False).agg({"VOLUME_YTD": "sum", "SHARE_YTD": "sum"}).rename(columns={"VOLUME_YTD": "VOLUME", "SHARE_YTD": "SHARE"})
    else:
        agg = out.groupby(["TIME", col], as_index=False)["VOLUME"].sum()
        total_per_time = agg.groupby("TIME", as_index=False)["VOLUME"].sum().rename(columns={"VOLUME": "TOTAL_VOLUME"})
        agg = agg.merge(total_per_time, on="TIME", how="left")
        agg["SHARE"] = (agg["VOLUME"] / agg["TOTAL_VOLUME"] * 100).round(2)

    times = sorted(agg["TIME"].unique())
    latest_time = times[-1]
    latest_df = agg[agg["TIME"] == latest_time].sort_values("VOLUME", ascending=False)
    
    overall_rank = agg.groupby(col)["VOLUME"].sum().sort_values(ascending=False)
    all_cats = overall_rank.index.tolist()
    top_cats = all_cats[:10]
    is_limited = len(all_cats) > 10
    evolution_title = f"Evolution of {'Top 10 ' if is_limited else ''}Segments"
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        specs=[[{"type": "treemap"}, {"secondary_y": False}]],
        subplot_titles=(f"Current State ({latest_time})", evolution_title)
    )

    fig.add_trace(
        go.Treemap(
            labels=latest_df[col],
            parents=[""] * len(latest_df),
            values=latest_df["VOLUME"],
            textinfo="label+percent entry+value",
            hovertemplate="<b>%{label}</b><br>Volume: %{value}<br>Share: %{percentEntry:.1%}<extra></extra>",
            marker=dict(colorscale='Viridis')
        ),
        row=1, col=1
    )

    plot_mode = "line" if len(top_cats) == 1 else "bar"
    for c in top_cats:
        d = agg[agg[col] == c]
        if plot_mode == "line":
            fig.add_trace(
                go.Scatter(
                    x=d["TIME"], y=d["SHARE"],
                    name=f"{c} (%)",
                    mode="lines+markers",
                    line=dict(width=4),
                    hovertemplate="%{y:.1f}%"
                ),
                row=1, col=2
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=d["TIME"], y=d["SHARE"],
                    name=f"{c} (%)",
                    hovertemplate="%{y:.1f}%"
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=d["TIME"], y=d["SHARE"],
                    name=str(c),
                    mode="lines",
                    showlegend=False,
                    line=dict(width=2, dash="dot"),
                    hovertemplate="<b>%{fullData.name}</b>: %{y:.1f}%<extra></extra>"
                ),
                row=1, col=2
            )



    fig.update_layout(
        height=600,
        barmode='group',
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=20, l=20, r=20)
    )
    fig.update_yaxes(autorange=True, row=1, col=2)
    
    fig.update_yaxes(title_text="Share (%)", row=1, col=2)

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


def kpi_top_brand_vs_market(df_portfolio: pd.DataFrame, df_market: pd.DataFrame, var_col_portfolio: str, var_col_market: str, asset_status: str, reg_type_portfolio: list | str = "ALL", reg_type_market: list | str = "ALL", bev_only: bool = False, top_n: str = "ALL", period_mode: str = "quarterly") -> pd.DataFrame:
    df_port = df_portfolio.copy()
    
    # Handle Period
    if period_mode == "monthly":
        df_port["PERIOD"] = pd.to_datetime(df_port["COB_DATE"]).dt.strftime("%Y-%m")
    elif period_mode == "yearly":
        df_port["PERIOD"] = pd.to_datetime(df_port["COB_DATE"]).dt.year.astype(str)
    else:
        df_port["PERIOD"] = pd.to_datetime(df_port["COB_DATE"]).dt.to_period('Q').astype(str)

    # Handle ALL for status
    if asset_status != "ALL":
        df_port = apply_status_filter(df_port, asset_status)

    if bev_only:
        df_port = df_port[df_port["POWER_CATEGORY"].astype(str).str.contains("ELECTRIC", case=False, na=False)]
    
    # Handle Registration Type for Portfolio
    if reg_type_portfolio and reg_type_portfolio != "ALL" and reg_type_portfolio != ["ALL"]:
        if isinstance(reg_type_portfolio, str):
            reg_type_portfolio = [reg_type_portfolio]
        if "CLS_VEHICLE_TYPE" in df_port.columns:
            df_port = df_port[df_port["CLS_VEHICLE_TYPE"].isin(reg_type_portfolio)]

    df_port = get_kpi_data(df_port, extra_keys=["PERIOD"])

    df_mkt = df_market.copy()
    # Handle Registration Type for Market
    if reg_type_market and reg_type_market != "ALL" and reg_type_market != ["ALL"]:
        if isinstance(reg_type_market, str):
            reg_type_market = [reg_type_market]
        if "Registration Type" in df_mkt.columns:
            df_mkt = df_mkt[df_mkt["Registration Type"].isin(reg_type_market)]
    
    if period_mode == "monthly":
        df_mkt["PERIOD"] = pd.to_datetime(df_mkt["date"]).dt.strftime("%Y-%m")
    elif period_mode == "yearly":
        df_mkt["PERIOD"] = pd.to_datetime(df_mkt["date"]).dt.year.astype(str)
    else:
        df_mkt["PERIOD"] = pd.to_datetime(df_mkt["date"]).dt.to_period('Q').astype(str)
    
    if bev_only:
        df_mkt = df_mkt[df_mkt["Fuel Type"].astype(str).str.contains("ELECTRIC", case=False, na=False)]

    available_periods = set(df_mkt["PERIOD"].unique())
    df_port = df_port[df_port["PERIOD"].isin(available_periods)]

    port = (
        df_port.groupby(["COUNTRY", "PERIOD", var_col_portfolio])
        .size()
        .reset_index(name="volume_portfolio")
    )
    
    # Map Country Codes to Names for merging with Market
    country_map = {
        "FR": "France", "BE": "Belgium", "UK": "United Kingdom", 
        "NL": "Netherlands", "LU": "Luxembourg", "IT": "Italy", 
        "ES": "Spain", "DE": "Germany"
    }
    port["COUNTRY"] = port["COUNTRY"].map(country_map).fillna(port["COUNTRY"])

    port_total = port.groupby(["COUNTRY", "PERIOD"])["volume_portfolio"].transform("sum")
    port["share_portfolio"] = (port["volume_portfolio"] / port_total * 100).round(2)

    mkt = (
        df_mkt.groupby(["Country/Territory-Name", "PERIOD", var_col_market])["volume"]
        .sum()
        .reset_index(name="volume_market")
    )
    mkt_total = mkt.groupby(["Country/Territory-Name", "PERIOD"])["volume_market"].transform("sum")
    mkt["share_market"] = (mkt["volume_market"] / mkt_total * 100).round(2)

    # Rename for merging
    port = port.rename(columns={var_col_portfolio: "BRAND"})
    mkt = mkt.rename(columns={"Country/Territory-Name": "COUNTRY", var_col_market: "BRAND"})

    # Ensure BRAND is upper case for matching
    port["BRAND"] = port["BRAND"].astype(str).str.upper()
    mkt["BRAND"] = mkt["BRAND"].astype(str).str.upper()

    merged = pd.merge(port, mkt, on=["COUNTRY", "PERIOD", "BRAND"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    
    merged["ratio"] = (merged["share_portfolio"] / merged["share_market"]).round(2)
    
    # Top N logic
    if top_n != "ALL":
        try:
            n = int(top_n)
            merged = merged.sort_values(["COUNTRY", "PERIOD", "share_portfolio"], ascending=[True, True, False])
            merged = merged.groupby(["COUNTRY", "PERIOD"]).head(n)
        except:
            pass

    return merged.sort_values(["COUNTRY", "PERIOD", "share_portfolio"], ascending=[True, True, False])



def figure_top_brand_vs_market(df_kpi13: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if df_kpi13.empty:
        fig.update_layout(title=title, template="plotly_white", height=500)
        return fig

    agg = (
        df_kpi13.groupby("PERIOD", as_index=False)
        .agg(
            portfolio_share=("share_portfolio", "mean"),
            market_share=("share_market", "mean"),
            portfolio_volume=("volume_portfolio", "sum"),
        )
        .sort_values("PERIOD")
    )

    fig.add_trace(go.Bar(x=agg["PERIOD"], y=agg["portfolio_share"], name="Portfolio share %"))
    fig.add_trace(go.Bar(x=agg["PERIOD"], y=agg["market_share"], name="Market share %"))
    fig.add_trace(
        go.Scatter(
            x=agg["PERIOD"],
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
    top_n: str = "1",
    period_mode: str = "quarterly",
) -> pd.DataFrame:
    """Compute top variable per period with volume and share. Returns rows per country."""
    out = df.copy()
    out = out[
        (out["NOVA_ASSET_STATUS"].astype(str).str.contains(asset_status, na=False))
        & (out["BIKE_OR_CAR"].astype(str) == bike_or_car)
    ].dropna(subset=["COB_DATE", var_col, "COUNTRY"])
    if out.empty:
        return pd.DataFrame()

    out["YEAR"] = out["COB_DATE"].dt.year.astype(int)
    if period_mode == "monthly":
        out["PERIOD"] = out["YEAR"].astype(str) + "_" + out["COB_DATE"].dt.month.astype(str).str.zfill(2)
    elif period_mode == "yearly":
        out["PERIOD"] = out["YEAR"].astype(str)
    else:
        out["Q"] = "Q" + (((out["COB_DATE"].dt.month - 1) // 3) + 1).astype(str)
        out["PERIOD"] = out["YEAR"].astype(str) + "_" + out["Q"]

    out = get_kpi_data(out, extra_keys=["PERIOD"])

    rows: list[dict] = []
    
    n_limit = 1
    if str(top_n).upper() != "ALL":
        try:
            n_limit = int(top_n)
        except:
            n_limit = 1
    else:
        n_limit = 99999

    for country in sorted(out["COUNTRY"].dropna().unique().tolist()):
        country_df = out[out["COUNTRY"] == country]
        row_data: dict[str, str | int | float | None] = {"COUNTRY": country}
        for period in sorted(country_df["PERIOD"].dropna().unique().tolist()):
            period_df = country_df[country_df["PERIOD"] == period]
            counts = period_df.groupby(var_col).size().sort_values(ascending=False)
            total_volume = int(counts.sum())

            if counts.empty or total_volume == 0:
                pass
            else:
                top_items = counts.head(n_limit)
                for i, (var, vol) in enumerate(top_items.items()):
                    suffix = f"_{i+1}" if n_limit > 1 else ""
                    row_data[f"{period}{suffix}_VAR"] = str(var)
                    row_data[f"{period}{suffix}_VOLUME"] = int(vol)
                    row_data[f"{period}{suffix}_SHARE"] = round((vol / total_volume) * 100, 2)
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


CONCENTRATION_STATUS_OPTIONS = ["IN FLEET", "ORDER", "ORDER YTD", "DEHIRE"]
CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO = [
    {"label": "Brand", "value": "BRAND_UPDATE"},
    {"label": "OEM", "value": "OEM_UPDATE"},
    {"label": "CO2 Bucket", "value": "CO2_BUCKET"},
    {"label": "BEV", "value": "HIGHEST_BEV"},
]
CONCENTRATION_VARIABLE_OPTIONS_MARKET = [
    {"label": "Market Make", "value": "BRAND_UPDATE"},
    {"label": "Market Make Group", "value": "OEM_UPDATE"},
    {"label": "Market CO2 Bucket", "value": "CO2_BUCKET"},
    {"label": "BEV", "value": "HIGHEST_BEV"},
]
CONCENTRATION_SOURCE_OPTIONS = ["portfolio", "market"]

OWNER_TYPE_OPTIONS = [
    {"label": "All", "value": "ALL"},
    {"label": "Car manufacturer / dealer", "value": "Car manufacturer / dealer"},
    {"label": "Company Cars", "value": "Company Cars"},
    {"label": "Administration Govt.", "value": "Administration Govt."},
    {"label": "Private", "value": "Private"},
    {"label": "Rental", "value": "Rental"},
    {"label": "Unspecified", "value": "Unspecified"}
]

MARKET_STATUS_OPTIONS = ["IN FLEET", "ORDER"]
MARKET_VARIABLE_OPTIONS = [
    {"label": "Brand", "value": "BRAND_UPDATE"},
    {"label": "OEM", "value": "OEM_UPDATE"},
    {"label": "BEV", "value": "HIGHEST_BEV"},
    {"label": "CO2 Bucket", "value": "CO2_BUCKET"},
]






















df = load_dataset()
if df is None:
    df = pd.DataFrame()
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
cob_date_options = sorted(df["COB_DATE"].dropna().dt.to_period("M").astype(str).unique().tolist())
month_options = ["ALL"] + [f"{month:02d}" for month in range(1, 13)]
country_options = countries + ["ALL"]
year_options = [{"label": "ALL", "value": "ALL"}] + [{"label": str(y), "value": y} for y in years]

statuses = sorted(df["NOVA_ASSET_STATUS"].dropna().astype(str).unique().tolist())
default_status_v2 = "IN FLEET" if "IN FLEET" in statuses else (statuses[0] if statuses else "ALL")

vehicle_col_global = detect_vehicle_type_column(df)
vehicle_types = sorted(df[vehicle_col_global].dropna().astype(str).str.upper().unique().tolist()) if vehicle_col_global else []
default_vehicle_type = "SUV" if "SUV" in vehicle_types else (vehicle_types[0] if vehicle_types else "ALL")
has_vehicle_type_temp = "VEHICLE_TYPE_TEMP" in df.columns
bike_or_car_options = ["ALL"] + sorted(df["BIKE_OR_CAR"].dropna().astype(str).str.upper().unique().tolist()) if "BIKE_OR_CAR" in df.columns else ["ALL"]
asset_status_options = [
    {"label": "ALL", "value": "ALL"},
    {"label": "IN FLEET", "value": "IN FLEET"},
    {"label": "Orders", "value": "ORDER"},
    {"label": "SOLD", "value": "SOLD"},
]
date_mode_options = [
    {"label": "None", "value": "NONE"},
    {"label": "Contract start date", "value": "CONTRACT_START_DATE"},
    {"label": "Delivery date", "value": "DELIVERY_DATE"},
]




















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
                dcc.Link("Main KPIs + Fuel", href="/", className="nav-link"),
                dcc.Link("Production by Energy", href="/vue-2", className="nav-link"),
                dcc.Link("Vue 3 - SUV", href="/vue-3", className="nav-link") if has_vehicle_type_temp else html.Span(),
                dcc.Link("Vue 4 - Concentration Risk", href="/vue-4", className="nav-link"),
                dcc.Link("Vue 5 - Concentration", href="/vue-5", className="nav-link"),
                dcc.Link("Vue 6 - Portfolio vs Market", href="/vue-6", className="nav-link"),
            ],
            className="top-nav",
        ),
        html.Div(id="page-content"),
        html.Div(
            id="complete-report-modal",
            style={"display": "none", "position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)", "zIndex": "9999", "justifyContent": "center", "alignItems": "center"},
            children=[
                html.Div(
                    className="panel",
                    style={"width": "450px", "maxWidth": "90%", "padding": "24px"},
                    children=[
                        html.H2("Generate Complete Report", style={"marginTop": "0"}),
                        html.P("Select the years and granularities for your interactive report.", className="small-note", style={"marginBottom": "16px"}),
                        html.Div("Years to include", className="filter-label", style={"marginTop": "10px", "fontWeight": "bold"}),
                        dcc.Checklist(
                            id="report-modal-years",
                            options=[{"label": str(y), "value": str(y)} for y in sorted([y for y in df["YEAR"].unique() if pd.notna(y)], reverse=True)] if "YEAR" in df.columns else [],
                            value=[],
                            labelStyle={"display": "inline-block", "marginRight": "12px", "padding": "4px 0"}
                        ),
                        html.Div("Periods to include", className="filter-label", style={"marginTop": "16px", "fontWeight": "bold"}),
                        dcc.Checklist(
                            id="report-modal-periods",
                            options=[
                                {"label": "Monthly", "value": "monthly"},
                                {"label": "Quarterly", "value": "quarterly"},
                                {"label": "Annually", "value": "annually"}
                            ],
                            value=["annually"],
                            labelStyle={"display": "inline-block", "marginRight": "12px", "padding": "4px 0"}
                        ),
                        html.Div("Date Rule", className="filter-label", style={"marginTop": "16px", "fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="report-modal-date-mode",
                            options=[
                                {"label": "None (Current Fleet)", "value": "NONE"},
                                {"label": "Contract Start Date", "value": "CONTRACT_START_DATE"},
                                {"label": "Delivery Date", "value": "DELIVERY_DATE"}
                            ],
                            value="NONE",
                            clearable=False,
                            style={"width": "100%"}
                        ),
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "marginTop": "24px"},
                            children=[
                                html.Button("Cancel", id="close-complete-report-modal", n_clicks=0, className="secondary-button"),
                                html.Button("Download Report", id="download-complete-report-btn", n_clicks=0, className="primary-button")
                            ]
                        ),
                        dcc.Download(id="complete-report-download")
                    ]
                )
            ]
        ),
    ],
    className="page-wrap",
)

# =============================================================================
# =============================================================================

def view1_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Img(src=get_logo_dash_src(), style={"width": "72px", "height": "72px", "objectFit": "contain", "borderRadius": "0px", "background": "#ffffff", "boxShadow": "0 8px 18px rgba(0,0,0,0.12)", "marginRight": "16px"}),
                    html.Div(
                        [
                            html.H1("Fleet Monitoring - View 1 - Main KPIs"),
                            html.P("Main KPIs and Fuel Type"),
                        ]
                    ),
                ],
                className="hero",
                style={"display": "flex", "alignItems": "center"},
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
                            dcc.Dropdown(id="year-filter", options=year_options, value=default_year, placeholder="Select year", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Period", className="filter-label"),
                            dcc.Dropdown(
                                id="granularity-filter",
                                options=[
                                    {"label": "Monthly", "value": "monthly"},
                                    {"label": "Quarterly", "value": "quarterly"},
                                    {"label": "Yearly", "value": "yearly"},
                                ],
                                value="monthly",
                                clearable=False,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Month", className="filter-label", id="month-filter-label"),
                            dcc.Dropdown(id="month-filter", options=month_options, value=None, placeholder="Select month", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Date Rule", className="filter-label"),
                            dcc.Dropdown(
                                id="kpi-date-mode-filter",
                                options=[
                                    {"label": "None (Current Fleet)", "value": "NONE"},
                                    {"label": "Contract Start Date", "value": "CONTRACT_START_DATE"},
                                    {"label": "Delivery Date", "value": "DELIVERY_DATE"},
                                ],
                                value="NONE",
                                clearable=False,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Asset Status", className="filter-label"),
                            dcc.Dropdown(
                                id="status-filter",
                                options=[
                                    {"label": "IN FLEET", "value": "IN FLEET"},
                                    {"label": "Orders", "value": "ORDER"},
                                    {"label": "SOLD", "value": "SOLD"},
                                    {"label": "DEHIRE", "value": "DEHIRE"},
                                    {"label": "ALL", "value": "ALL"},
                                ],
                                value="IN FLEET",
                                clearable=False,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Vehicle type", className="filter-label"),
                            dcc.Dropdown(id="kpi-common-bike-or-car-filter", options=bike_or_car_options, value="CAR", clearable=False),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Refresh", id="view1-refresh-button", n_clicks=0, className="primary-button"),
                            html.Button("Download report", id="view1-download-button", n_clicks=0, className="primary-button"),
                            html.Button("Complete Report", id={"type": "open-complete-report", "index": 1}, n_clicks=0, className="secondary-button"),
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
                            html.Div("LTR < 25", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi1-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi1-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label", id="kpi1-month-label"), dcc.Dropdown(id="kpi1-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Date Rule", className="filter-label"), dcc.Dropdown(id="kpi1-date-mode-filter", options=date_mode_options, value="NONE", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Asset Status", className="filter-label"), dcc.Dropdown(id="kpi1-status-filter", options=asset_status_options, value="IN FLEET", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Vehicle Type", className="filter-label"), dcc.Dropdown(id="kpi1-vehicle-filter", options=bike_or_car_options, value="CAR", clearable=False)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-1-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("LTR [25-30]", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi2-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi2-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label", id="kpi2-month-label"), dcc.Dropdown(id="kpi2-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Date Rule", className="filter-label"), dcc.Dropdown(id="kpi2-date-mode-filter", options=date_mode_options, value="NONE", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Asset Status", className="filter-label"), dcc.Dropdown(id="kpi2-status-filter", options=asset_status_options, value="IN FLEET", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Vehicle Type", className="filter-label"), dcc.Dropdown(id="kpi2-vehicle-filter", options=bike_or_car_options, value="CAR", clearable=False)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-2-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("DI vs Non-DI", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi3-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi3-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label", id="kpi3-month-label"), dcc.Dropdown(id="kpi3-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Date Rule", className="filter-label"), dcc.Dropdown(id="kpi3-date-mode-filter", options=date_mode_options, value="NONE", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Asset Status", className="filter-label"), dcc.Dropdown(id="kpi3-status-filter", options=asset_status_options, value="IN FLEET", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Vehicle Type", className="filter-label"), dcc.Dropdown(id="kpi3-vehicle-filter", options=bike_or_car_options, value="CAR", clearable=False)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-3-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("Hybrid share", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi4-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi4-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label", id="kpi4-month-label"), dcc.Dropdown(id="kpi4-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Date Rule", className="filter-label"), dcc.Dropdown(id="kpi4-date-mode-filter", options=date_mode_options, value="NONE", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Asset Status", className="filter-label"), dcc.Dropdown(id="kpi4-status-filter", options=asset_status_options, value="IN FLEET", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Vehicle Type", className="filter-label"), dcc.Dropdown(id="kpi4-vehicle-filter", options=bike_or_car_options, value="CAR", clearable=False)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-4-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("EV share", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi5-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi5-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label", id="kpi5-month-label"), dcc.Dropdown(id="kpi5-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Date Rule", className="filter-label"), dcc.Dropdown(id="kpi5-date-mode-filter", options=date_mode_options, value="NONE", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Asset Status", className="filter-label"), dcc.Dropdown(id="kpi5-status-filter", options=asset_status_options, value="IN FLEET", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Vehicle Type", className="filter-label"), dcc.Dropdown(id="kpi5-vehicle-filter", options=bike_or_car_options, value="CAR", clearable=False)], className="card-filter-box"),
                                ],
                                className="card-filter-bar",
                            ),
                            html.Div(id="kpi-5-card"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("Passenger car vs LCV", className="kpi-card-title"),
                            html.Div(
                                [
                                    html.Div([html.Div("Country", className="filter-label"), dcc.Dropdown(id="kpi6-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Year", className="filter-label"), dcc.Dropdown(id="kpi6-year-filter", options=year_options, value=None, placeholder="Select year", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Month", className="filter-label", id="kpi6-month-label"), dcc.Dropdown(id="kpi6-month-filter", options=month_options, value=None, placeholder="Select month", clearable=True)], className="card-filter-box"),
                                    html.Div([html.Div("Date Rule", className="filter-label"), dcc.Dropdown(id="kpi6-date-mode-filter", options=date_mode_options, value="NONE", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Asset Status", className="filter-label"), dcc.Dropdown(id="kpi6-status-filter", options=asset_status_options, value="IN FLEET", clearable=False)], className="card-filter-box"),
                                    html.Div([html.Div("Vehicle Type", className="filter-label"), dcc.Dropdown(id="kpi6-vehicle-filter", options=bike_or_car_options, value="CAR", clearable=False)], className="card-filter-box"),
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
                [
                    html.Div(
                        [
                            html.Div("Summary table", className="panel-title"),
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
            html.Div(
                [
                    html.Div("Fuel Type Share", className="panel-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Country", className="filter-label"),
                                    dcc.Dropdown(id="kpi7-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                                ],
                                className="kpi7-filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Date range (M/Y)", className="filter-label"),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="kpi7-start-date-filter",
                                                options=cob_date_options,
                                                placeholder="Start",
                                                clearable=True,
                                                style={"minWidth": "100px"},
                                            ),
                                            html.Div("-", style={"alignSelf": "center", "color": "#627d98"}),
                                            dcc.Dropdown(
                                                id="kpi7-end-date-filter",
                                                options=cob_date_options,
                                                placeholder="End",
                                                clearable=True,
                                                style={"minWidth": "100px"},
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "8px", "flexWrap": "nowrap"},
                                    ),
                                ],
                                className="kpi7-filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Asset Status", className="filter-label"),
                                    dcc.Dropdown(
                                        id="status-group-filter",
                                        options=[
                                            {"label": "IN FLEET", "value": "IN FLEET"},
                                            {"label": "Orders", "value": "ORDER"},
                                            {"label": "SOLD", "value": "SOLD"},
                                            {"label": "DEHIRE", "value": "DEHIRE"},
                                            {"label": "ALL", "value": "ALL"},
                                        ],
                                        value=None,
                                        placeholder="Select status",
                                        clearable=True,
                                    ),
                                ],
                                className="kpi7-filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Period", className="filter-label"),
                                    dcc.Dropdown(
                                        id="period-mode-filter",
                                        options=[
                                            {"label": "Monthly", "value": "monthly"},
                                            {"label": "Quarterly", "value": "quarterly"},
                                            {"label": "Yearly", "value": "yearly"},
                                        ],
                                        value=None,
                                        placeholder="Select period",
                                        clearable=True,
                                    ),
                                ],
                                className="kpi7-filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Vehicle Type", className="filter-label"),
                                    dcc.Dropdown(id="kpi7-bike-or-car-filter", options=bike_or_car_options, value=None, placeholder="Select vehicle type", clearable=True),
                                ],
                                className="kpi7-filter-box",
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
                                        value=None,
                                        inline=True,
                                        labelStyle={"marginRight": "14px"},
                                    ),
                                ],
                                className="kpi7-filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Action", className="filter-label"),
                                    html.Button("Refresh", id="view1-kpi7-refresh-button", n_clicks=0, className="primary-button"),
                                ],
                                className="kpi7-filter-box",
                            ),
                        ],
                        className="kpi7-filter-bar",
                    ),
                    dcc.Graph(id="kpi-7-graph", config={"displayModeBar": False}),
                    html.Div(id="kpi-7-table-wrap"),
                ],
                className="panel",
            ),
            dcc.Download(id="view1-html-download"),
            dcc.Store(id="v1-results-store"),
        ]
    )


def view2_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Img(src=get_logo_dash_src(), style={"width": "72px", "height": "72px", "objectFit": "contain", "borderRadius": "0px", "background": "#ffffff", "boxShadow": "0 8px 18px rgba(0,0,0,0.12)", "marginRight": "16px"}),
                    html.Div(
                        [
                            html.H1("Fleet Monitoring - View 2 - Production by fuel type"),
                            html.P("Production by fuel type (volume or share)"),
                        ]
                    ),
                ],
                className="hero",
                style={"display": "flex", "alignItems": "center"},
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
                            dcc.Dropdown(id="v2-year-filter", options=year_options, value=default_year, placeholder="Select year", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Granularity", className="filter-label"),
                            dcc.Dropdown(
                                id="v2-period-filter",
                                options=[
                                    {"label": "Monthly", "value": "monthly"},
                                    {"label": "Quarterly", "value": "quarterly"},
                                    {"label": "Yearly", "value": "yearly"},
                                ],
                                value="monthly",
                                clearable=False,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Vehicle type", className="filter-label"),
                            dcc.Dropdown(id="v2-bike-or-car-filter", options=bike_or_car_options, value=None, placeholder="Select vehicle type", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("View", className="filter-label"),
                            dcc.RadioItems(
                                id="v2-metric-mode-filter",
                                options=[
                                    {"label": "Volume", "value": "volume"},
                                    {"label": "Share", "value": "share"},
                                ],
                                value=None,
                                inline=True,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Date source", className="filter-label"),
                            dcc.RadioItems(
                                id="v2-date-mode-filter",
                                options=[
                                    {"label": "Contract start date", "value": "CONTRACT_START_DATE"},
                                    {"label": "Delivery date", "value": "DELIVERY_DATE"},
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
                            html.Button("Refresh", id="view2-refresh-button", n_clicks=0, className="primary-button"),
                            html.Button("Download HTML", id="view2-download-button", n_clicks=0, className="primary-button"),
                            html.Button("Complete Report", id={"type": "open-complete-report", "index": 2}, n_clicks=0, className="secondary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("Production by Fuel Type", className="panel-title"),
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
                    html.Img(src=get_logo_dash_src(), style={"width": "72px", "height": "72px", "objectFit": "contain", "borderRadius": "0px", "background": "#ffffff", "boxShadow": "0 8px 18px rgba(0,0,0,0.12)", "marginRight": "16px"}),
                    html.Div(
                        [
                            html.H1("Vue 3 - SUV"),
                            html.P("KPI 9_1 SUV Share per quarter and KPI 9_2 SUV Share by energy per quarter."),
                        ]
                    ),
                ],
                className="hero",
                style={"display": "flex", "alignItems": "center"},
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
                            html.Button("Reload", id="view3-refresh-button", n_clicks=0, className="primary-button"),
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
                    html.Img(src=get_logo_dash_src(), style={"width": "72px", "height": "72px", "objectFit": "contain", "borderRadius": "0px", "background": "#ffffff", "boxShadow": "0 8px 18px rgba(0,0,0,0.12)", "marginRight": "16px"}),
                    html.Div(
                        [
                            html.H1("Vue 4 - Concentration Risk at EOC"),
                            html.P("KPI 10 - Top vehicle models at End of Contract by quarter."),
                        ]
                    ),
                ],
                className="hero",
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="v4-country-filter", options=country_options, value=None, placeholder="Select country", clearable=True),
                        ],
                        className="filter-box",
                        style={"flex": "0.6 1 0", "minWidth": "100px"},
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
                        style={"flex": "0.7 1 0", "minWidth": "120px"},
                    ),
                    html.Div(
                        [
                            html.Div("Vehicle type", className="filter-label"),
                            dcc.Dropdown(id="v4-bike-or-car-filter", options=bike_or_car_options, value=None, placeholder="Select vehicle type", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Date range (M/Y)", className="filter-label"),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="v4-start-date-filter",
                                        options=[f"{y}-{str(m).zfill(2)}" for y in range(2020, 2036) for m in range(1, 13)],
                                        placeholder="Start",
                                        clearable=True,
                                        style={"minWidth": "100px"},
                                    ),
                                    html.Div("-", style={"alignSelf": "center", "color": "#627d98"}),
                                    dcc.Dropdown(
                                        id="v4-end-date-filter",
                                        options=[f"{y}-{str(m).zfill(2)}" for y in range(2020, 2036) for m in range(1, 13)],
                                        placeholder="End",
                                        clearable=True,
                                        style={"minWidth": "100px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "8px", "flexWrap": "nowrap"},
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Reload", id="view4-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="controls-inline",
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
                    html.Img(src=get_logo_dash_src(), style={"width": "72px", "height": "72px", "objectFit": "contain", "borderRadius": "0px", "background": "#ffffff", "boxShadow": "0 8px 18px rgba(0,0,0,0.12)", "marginRight": "16px"}),
                    html.Div(
                        [
                            html.H1("Vue 5 - Concentration"),
                            html.P("Global concentration risk analysis followed by country-level breakdown."),
                        ]
                    ),
                ],
                className="hero",
                style={"display": "flex", "alignItems": "center"},
            ),
            # --- GLOBAL CONCENTRATION (Ex-View 5) ---
            html.Div(
                [
                    html.H3("Global Concentration Analysis", className="panel-title", style={"padding": "10px 0"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Source", className="filter-label"),
                                    dcc.Dropdown(id="v5-source-filter", options=CONCENTRATION_SOURCE_OPTIONS, value="portfolio", placeholder="Select source", clearable=False),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Status", className="filter-label", id="v5-status-filter-label"),
                                    dcc.Dropdown(id="v5-status-filter", options=CONCENTRATION_STATUS_OPTIONS, value="IN FLEET", placeholder="Select status", clearable=True),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Variable", className="filter-label", id="v5-variable-filter-label"),
                                    dcc.Dropdown(id="v5-variable-filter", options=CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO, value=None, placeholder="Select variable", clearable=True),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Market owner", className="filter-label"),
                                    dcc.Dropdown(id="v5-market-owner-filter", options=OWNER_TYPE_OPTIONS, value="ALL", placeholder="Select owner", clearable=False),
                                ],
                                className="filter-box",
                                id="v5-market-owner-container",
                                style={"display": "none"} # Hidden by default since source is portfolio
                            ),
                            html.Div(
                                [
                                    html.Div("Period", className="filter-label"),
                                    dcc.Dropdown(
                                        id="v5-period-filter",
                                        options=[{"label": "Monthly", "value": "monthly"}, {"label": "Quarterly", "value": "quarterly"}, {"label": "Yearly", "value": "yearly"}],
                                        value="quarterly",
                                        clearable=False,
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Top N", className="filter-label"),
                                    dcc.Input(
                                        id="v5-top-n-filter",
                                        type="text",
                                        value="1",
                                        placeholder="All, 1, 5...",
                                        list="top-n-options-v5",
                                        className="dash-input",
                                    ),
                                    html.Datalist(
                                        id="top-n-options-v5",
                                        children=[
                                            html.Option(value="1"),
                                            html.Option(value="5"),
                                            html.Option(value="10"),
                                            html.Option(value="20"),
                                            html.Option(value="All"),
                                        ]
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Action", className="filter-label"),
                                    html.Button("Reload", id="view5-refresh-button", n_clicks=0, className="primary-button"),
                                ],
                                className="filter-box",
                            ),
                        ],
                        className="filter-bar",
                    ),
                    html.Div(
                        [
                            html.Div("KPI 11 - Concentration (Global)", className="panel-title"),
                            dcc.Graph(id="v5-kpi11-graph", config={"displayModeBar": False}),
                            html.Div(id="v5-kpi11-table-wrap"),
                        ],
                        className="panel",
                    ),
                ],
                style={"marginBottom": "40px"}
            ),
            
            html.Hr(style={"borderTop": "2px solid #dbe6f2", "margin": "40px 0"}),
            
            # --- COUNTRY BREAKDOWN (Ex-View 6) ---
            html.Div(
                [
                    html.H3("Country Breakdown Analysis", className="panel-title", style={"padding": "10px 0"}),
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
                                    dcc.Dropdown(id="v6-variable-filter", options=CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO, value=None, placeholder="Select variable", clearable=True),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Metric", className="filter-label"),
                                    dcc.RadioItems(
                                        id="v6-metric-mode-filter",
                                        options=[{"label": "Share", "value": "share"}, {"label": "Volume", "value": "volume"}],
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
                                    dcc.Dropdown(
                                        id="v6-period-filter",
                                        options=[{"label": "Monthly", "value": "monthly"}, {"label": "Quarterly", "value": "quarterly"}, {"label": "Yearly", "value": "yearly"}],
                                        value="quarterly",
                                        clearable=False,
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Top N", className="filter-label"),
                                    dcc.Input(
                                        id="v6-top-n-filter",
                                        type="text",
                                        value="1",
                                        placeholder="All, 1, 5...",
                                        list="top-n-options-v6",
                                        className="dash-input",
                                    ),
                                    html.Datalist(
                                        id="top-n-options-v6",
                                        children=[
                                            html.Option(value="1"),
                                            html.Option(value="5"),
                                            html.Option(value="10"),
                                            html.Option(value="20"),
                                            html.Option(value="All"),
                                        ]
                                    ),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Action", className="filter-label"),
                                    html.Button("Reload", id="view6-refresh-button", n_clicks=0, className="primary-button"),
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
            ),
        ]
    )


def view6_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Img(src=get_logo_dash_src(), style={"width": "72px", "height": "72px", "objectFit": "contain", "borderRadius": "0px", "background": "#ffffff", "boxShadow": "0 8px 18px rgba(0,0,0,0.12)", "marginRight": "16px"}),
                    html.Div(
                        [
                            html.H1("Vue 6 - Concentration Portfolio vs Market"),
                            html.P("KPI 13 Top BRAND/OEM/BEV portfolio share vs market share per quarter and country."),
                        ]
                    ),
                ],
                className="hero",
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Portfolio Status type", className="filter-label"),
                            dcc.Dropdown(id="v6-kpi13-status-filter", options=[{"label": "ALL", "value": "ALL"}] + [{"label": s, "value": s} for s in CONCENTRATION_STATUS_OPTIONS], value="IN FLEET", placeholder="Select status", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Portfolio Registration type", className="filter-label"),
                            dcc.Dropdown(id="v6-kpi13-port-regtype-filter", options=[
                                {"label": "PV", "value": "PV"},
                                {"label": "LCV", "value": "LCV"}
                            ], value=["PV", "LCV"], multi=True, placeholder="Select type", clearable=True),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Market Registration type", className="filter-label"),
                            dcc.Dropdown(id="v6-kpi13-mkt-regtype-filter", options=[
                                {"label": "Passenger Cars", "value": "Passenger Cars"},
                                {"label": "Light Commercial Vehicle", "value": "Light Commercial Vehicle"},
                                {"label": "Heavy Commercial Vehicle", "value": "Heavy Commercial Vehicle"}
                            ], value=["Passenger Cars"], multi=True, placeholder="Select type", clearable=True),
                        ],
                        className="filter-box",
                    ),
                            html.Div(
                                [
                                    html.Div("Portfolio vs Market Variable", className="filter-label"),
                                    dcc.Dropdown(id="v6-kpi13-variable-filter", options=MARKET_VARIABLE_OPTIONS, value="BRAND", placeholder="Select variable", clearable=True),
                                ],
                                className="filter-box",
                            ),
                            html.Div(
                                [
                                    html.Div("Market owner type", className="filter-label"),
                                    dcc.Dropdown(id="v6-kpi13-owner-filter", options=OWNER_TYPE_OPTIONS, value="ALL", placeholder="Select owner", clearable=False),
                                ],
                                className="filter-box",
                            ),
                    html.Div(
                        [
                            html.Div("Period", className="filter-label"),
                            dcc.Dropdown(
                                id="v6-kpi13-period-filter",
                                options=[{"label": "Monthly", "value": "monthly"}, {"label": "Quarterly", "value": "quarterly"}, {"label": "Yearly", "value": "yearly"}],
                                value="quarterly",
                                clearable=False,
                            ),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Top N", className="filter-label"),
                            dcc.Dropdown(id="v6-kpi13-top-n-filter", options=["3", "5", "10", "ALL"], value="5", clearable=False),
                        ],
                        className="filter-box",
                    ),
                    html.Div(
                        [
                            html.Div("Action", className="filter-label"),
                            html.Button("Reload", id="view6-kpi13-refresh-button", n_clicks=0, className="primary-button"),
                        ],
                        className="filter-box",
                    ),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Div("KPI 13 - Portfolio share vs market share", className="panel-title"),
                    dcc.Graph(id="v6-kpi13-graph", config={"displayModeBar": False}),
                    html.Div(id="v6-kpi13-table-wrap"),
                ],
                className="panel",
            ),
        ]
    )

# =============================================================================
# =============================================================================

@app.callback(
    [
        Output("month-filter", "options"),
        Output("month-filter", "value"),
        Output("month-filter-label", "children"),
        Output("period-mode-filter", "value"),
    ] + [
        Output(f"kpi{i}-month-filter", "options") for i in range(1, 7)
    ] + [
        Output(f"kpi{i}-month-filter", "value") for i in range(1, 7)
    ] + [
        Output(f"kpi{i}-country-filter", "value") for i in range(1, 7)
    ] + [
        Output(f"kpi{i}-year-filter", "value") for i in range(1, 7)
    ] + [
        Output(f"kpi{i}-date-mode-filter", "value") for i in range(1, 7)
    ] + [
        Output(f"kpi{i}-month-label", "children") for i in range(1, 7)
    ],
    [
        Input("granularity-filter", "value"),
        Input("country-filter", "value"),
        Input("year-filter", "value"),
        Input("month-filter", "value"),
        Input("kpi-date-mode-filter", "value"),
    ],
    prevent_initial_call=True
)
def unified_time_and_card_sync(gran, country, year, month, date_mode):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    month_opts = [{"label": "ALL", "value": "ALL"}] + [{"label": calendar.month_name[i], "value": str(i).zfill(2)} for i in range(1, 13)]
    quarter_opts = [{"label": "ALL", "value": "ALL"}] + [{"label": f"Q{i}", "value": f"Q{i}"} for i in range(1, 5)]
    yearly_opts = [{"label": "ALL", "value": "ALL"}]

    if gran == "yearly":
        opts = yearly_opts
        label = "Month"
        if triggered_id == "granularity-filter":
            month = "ALL"
    elif gran == "quarterly":
        opts = quarter_opts
        label = "Quarter"
        if triggered_id == "granularity-filter":
            month = month if any(o["value"] == month for o in quarter_opts) else "ALL"
    else:
        label = "Month"
        if country and year and country != "ALL" and year != "ALL":
             try:
                 avail = available_months(df, country, year)
                 opts = [{"label": "ALL", "value": "ALL"}] + [{"label": calendar.month_name[m], "value": str(m).zfill(2)} for m in avail]
             except:
                 opts = month_opts
        else:
             opts = month_opts
        
        if triggered_id == "granularity-filter":
            month = month if any(o["value"] == month for o in opts) else "ALL"

    return [opts, month, label, gran] + [opts]*6 + [month]*6 + [country]*6 + [year]*6 + [date_mode]*6 + [label]*6

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
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
    return view1_layout()


def has_missing_filters(*values: object) -> bool:
    return any(value is None for value in values)


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
    kpi1_dm: str,
    kpi2_dm: str,
    kpi3_dm: str,
    kpi4_dm: str,
    kpi5_dm: str,
    kpi6_dm: str,
    kpi_common_bike_or_car: str,
    date_mode: str,
    kpi1_status: str,
    kpi2_status: str,
    kpi3_status: str,
    kpi4_status: str,
    kpi5_status: str,
    kpi6_status: str,
    kpi1_vehicle: str,
    kpi2_vehicle: str,
    kpi3_vehicle: str,
    kpi4_vehicle: str,
    kpi5_vehicle: str,
    kpi6_vehicle: str,
):
    required_filters = [
        kpi1_country, kpi1_year,
        kpi2_country, kpi2_year,
        kpi3_country, kpi3_year, kpi3_month,
        kpi4_country, kpi4_year, kpi4_month,
        kpi5_country, kpi5_year, kpi5_month,
        kpi6_country, kpi6_year, kpi6_month,
        kpi_common_bike_or_car,
    ]
    if has_missing_filters(*required_filters):
        waiting_card = build_card_body("--", "Select the filters then click Refresh.", "#9aa5b1")
        return waiting_card, waiting_card, waiting_card, waiting_card, waiting_card, waiting_card

    # Use per-card vehicle if set, else fall back to global kpi_common_bike_or_car
    vc1 = kpi1_vehicle or kpi_common_bike_or_car
    vc2 = kpi2_vehicle or kpi_common_bike_or_car
    vc3 = kpi3_vehicle or kpi_common_bike_or_car
    vc4 = kpi4_vehicle or kpi_common_bike_or_car
    vc5 = kpi5_vehicle or kpi_common_bike_or_car
    vc6 = kpi6_vehicle or kpi_common_bike_or_car

    metrics = get_view1_metrics_cached(
        kpi1_country, kpi1_year, kpi1_month,
        kpi2_country, kpi2_year, kpi2_month,
        kpi3_country, kpi3_year, kpi3_month,
        kpi4_country, kpi4_year, kpi4_month,
        kpi5_country, kpi5_year, kpi5_month,
        kpi6_country, kpi6_year, kpi6_month,
        kpi_common_bike_or_car,
        date_mode,
        kpi1_dm, kpi2_dm, kpi3_dm, kpi4_dm, kpi5_dm, kpi6_dm,
        "monthly", # Default for cards
        kpi1_status, kpi2_status, kpi3_status, kpi4_status, kpi5_status, kpi6_status,
        kpi1_vehicle, kpi2_vehicle, kpi3_vehicle, kpi4_vehicle, kpi5_vehicle, kpi6_vehicle
    )

    results = {
        "kpi1": metrics["kpi1"],
        "kpi2": metrics["kpi2"],
        "kpi3": metrics["diesel_non"],
        "kpi4": metrics["hybrid"],
        "kpi5": metrics["ev"],
        "kpi6": metrics["pv_lcv"]
    }
    kpi_statuses = {
        "kpi1": kpi1_status, "kpi2": kpi2_status, "kpi3": kpi3_status,
        "kpi4": kpi4_status, "kpi5": kpi5_status, "kpi6": kpi6_status,
    }
    cards = []
    for kid, config in KPI_REGISTRY.items():
        val = results.get(kid)
        status_lbl = kpi_statuses.get(kid, "IN FLEET") or "IN FLEET"
        display_val = "N/A"
        if val is not None:
            if isinstance(val, tuple):
                display_val = f"{percent_or_na_precision(val[0], 2)} / {percent_or_na_precision(val[1], 2)}"
            else:
                display_val = percent_or_na_precision(val, 2)
        cards.append(build_card_body(display_val, f"{config['label']} | {status_lbl}", config['color']))

    return tuple(cards)


@app.callback(
    Output("kpi-7-graph", "figure"),
    Output("kpi-7-table-wrap", "children"),
    Input("view1-refresh-button", "n_clicks"),
    Input("view1-kpi7-refresh-button", "n_clicks"),
    State("kpi7-country-filter", "value"),
    State("kpi7-start-date-filter", "value"),
    State("kpi7-end-date-filter", "value"),
    State("status-group-filter", "value"),
    State("metric-mode-filter", "value"),
    State("period-mode-filter", "value"),
    State("kpi7-bike-or-car-filter", "value"),
    prevent_initial_call=True,
)
def update_kpi7(_refresh_clicks: int, _kpi7_refresh_clicks: int, country: str, start_date: str | None, end_date: str | None, status_group: str, metric_mode: str, period_mode: str, bike_or_car: str):
    if has_missing_filters(country, status_group, metric_mode, period_mode, bike_or_car):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select the filters then click Refresh")
        return empty_fig, html.Div("Select the filters then click Refresh.", className="small-note")

    pivot, y_title, x_title, period_label = get_kpi7_cached(country, start_date, end_date, status_group, metric_mode, period_mode, bike_or_car, "COB_DATE")
    pivot = pivot.copy()
    title_mode = "share" if metric_mode == "share" else "volume"
    country_label = country if country != "ALL" else "All countries"
    if period_mode == "quarterly":
        period_title = "by quarter"
    elif period_mode == "monthly":
        period_title = "by month"
    else:
        period_title = "yearly"
    if start_date and end_date:
        date_info = f" from {start_date} to {end_date}"
    else:
        date_info = ""
    title = f"Fuel Type {title_mode} {period_title}{date_info}<br><sup>{country_label} | {status_group} | {bike_or_car}</sup>"

    fig = figure_from_pivot(pivot, y_title, x_title, title)

    if pivot.empty:
        return fig, html.Div("No data available for the selected filters.", className="small-note")

    table_df = build_kpi7_table_with_total(pivot, metric_mode, country, status_group, period_mode, bike_or_car, start_date, end_date)
    table = build_table(table_df, page_size=15, table_id="v1-kpi7-table")

    title_mode_cap = "Share" if metric_mode == "share" else "Volume"
    table_title = f"Fuel Type {title_mode_cap} Table - {status_group} - {title_mode_cap} - {period_mode.capitalize()}"
    return fig, html.Div([html.H4(table_title, className="panel-title"), table])


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
    kpi1_dm: str,
    kpi2_dm: str,
    kpi3_dm: str,
    kpi4_dm: str,
    kpi5_dm: str,
    kpi6_dm: str,
    kpi_common_bike_or_car: str,
    date_mode: str,
    kpi1_limit: float | None,
    kpi2_limit: float | None,
    granularity: str,
    kpi1_status: str,
    kpi2_status: str,
    kpi3_status: str,
    kpi4_status: str,
    kpi5_status: str,
    kpi6_status: str,
    kpi1_vehicle: str,
    kpi2_vehicle: str,
    kpi3_vehicle: str,
    kpi4_vehicle: str,
    kpi5_vehicle: str,
    kpi6_vehicle: str,
):
    required_filters = [
        kpi1_country, kpi1_year,
        kpi2_country, kpi2_year,
        kpi3_country, kpi3_year, kpi3_month,
        kpi4_country, kpi4_year, kpi4_month,
        kpi5_country, kpi5_year, kpi5_month,
        kpi6_country, kpi6_year, kpi6_month,
        kpi_common_bike_or_car,
    ]
    if has_missing_filters(*required_filters):
        return html.Div("Select the filters then click Refresh.", className="small-note")

    metrics = get_view1_metrics_cached(
        kpi1_country, kpi1_year, kpi1_month,
        kpi2_country, kpi2_year, kpi2_month,
        kpi3_country, kpi3_year, kpi3_month,
        kpi4_country, kpi4_year, kpi4_month,
        kpi5_country, kpi5_year, kpi5_month,
        kpi6_country, kpi6_year, kpi6_month,
        kpi_common_bike_or_car,
        date_mode,
        kpi1_dm, kpi2_dm, kpi3_dm, kpi4_dm, kpi5_dm, kpi6_dm,
        granularity,
        kpi1_status, kpi2_status, kpi3_status, kpi4_status, kpi5_status, kpi6_status,
        kpi1_vehicle, kpi2_vehicle, kpi3_vehicle, kpi4_vehicle, kpi5_vehicle, kpi6_vehicle
    )

    month3 = cast(int, metrics["month3"])
    month4 = cast(int, metrics["month4"])
    month5 = cast(int, metrics["month5"])
    month6 = cast(int, metrics["month6"])
    month1 = cast(int | None, metrics["month1"])
    month2 = cast(int | None, metrics["month2"])

    kpi1_val = cast(float | None, metrics["kpi1"])
    kpi2_val = cast(float | None, metrics["kpi2"])
    diesel_non = cast(tuple[float | None, float | None], metrics["diesel_non"])
    hybrid_val = cast(float | None, metrics["hybrid"])
    ev_val = cast(float | None, metrics["ev"])
    pv_lcv_val = cast(tuple[float | None, float | None], metrics["pv_lcv"])

    period3 = summary_month_label(kpi3_year, metrics.get("month3_orig"))
    period4 = summary_month_label(kpi4_year, metrics.get("month4_orig"))
    period5 = summary_month_label(kpi5_year, metrics.get("month5_orig"))
    period6 = summary_month_label(kpi6_year, metrics.get("month6_orig"))
    period1 = summary_month_label(kpi1_year, metrics.get("month1_orig"))
    period2 = summary_month_label(kpi2_year, metrics.get("month2_orig"))
    desc = kpi_summary_description_map()
    rows = [
        {
            "label": "LTR < 25",
            "country": kpi1_country,
            "period": period1,
            "result_text": percent_or_na_precision(kpi1_val, 2),
            "signal": kpi_limit_status(kpi1_val, kpi1_limit),
            "volume": cast(int, metrics["kpi1_volume"]),
            "unit": "%",
            "comment": f"LTR <25 share, Limit {('N/A' if kpi1_limit is None else f'{float(kpi1_limit):g}%')}",
        },
        {
            "label": "LTR [25-30]",
            "country": kpi2_country,
            "period": period2,
            "result_text": percent_or_na_precision(kpi2_val, 2),
            "signal": kpi_limit_status(kpi2_val, kpi2_limit),
            "volume": cast(int, metrics["kpi2_volume"]),
            "unit": "%",
            "comment": f"LTR [25-30] share, Limit {('N/A' if kpi2_limit is None else f'{float(kpi2_limit):g}%')}",
        },
        {
            "label": "Diesel vs non-diesel",
            "country": kpi3_country,
            "period": period3,
            "result_text": f"{percent_or_na_precision(diesel_non[0], 2)} DI & {percent_or_na_precision(diesel_non[1], 2)} Non DI" if diesel_non[0] is not None else "N/A",
            "signal": "neutral",
            "volume": cast(int, metrics["kpi3_volume"]),
            "unit": "%",
            "comment": "Diesel Share, In fleet",
        },
        {
            "label": "Hybrid share",
            "country": kpi4_country,
            "period": period4,
            "result_text": percent_or_na_precision(hybrid_val, 2),
            "signal": "neutral",
            "volume": cast(int, metrics["kpi4_volume"]),
            "unit": "%",
            "comment": "Hyb (HEV + PHEV) In fleet",
        },
        {
            "label": "EV share",
            "country": kpi5_country,
            "period": period5,
            "result_text": percent_or_na_precision(ev_val, 2),
            "signal": "neutral",
            "volume": cast(int, metrics["kpi5_volume"]),
            "unit": "%",
            "comment": "EV share. In fleet",
        },
        {
            "label": "Passenger car vs LCV",
            "country": kpi6_country,
            "period": period6,
            "result_text": f"{percent_or_na_precision(pv_lcv_val[0], 2)} PV & {percent_or_na_precision(pv_lcv_val[1], 2)} LCV" if pv_lcv_val[0] is not None else "N/A",
            "signal": "neutral",
            "volume": cast(int, metrics["kpi6_volume"]),
            "unit": "%",
            "comment": "PC vs LCV share, In fleet",
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
    State("kpi7-start-date-filter", "value"),
    State("kpi7-end-date-filter", "value"),
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
    _n_clicks, country, year, month_value,
    kpi1_country, kpi1_year, kpi1_month,
    kpi2_country, kpi2_year, kpi2_month,
    kpi3_country, kpi3_year, kpi3_month,
    kpi4_country, kpi4_year, kpi4_month,
    kpi5_country, kpi5_year, kpi5_month,
    kpi6_country, kpi6_year, kpi6_month,
    kpi1_dm, kpi2_dm, kpi3_dm, kpi4_dm, kpi5_dm, kpi6_dm,
    kpi_common_bike_or_car, date_mode,
    kpi1_limit, kpi2_limit,
    kpi7_country, kpi7_start_date, kpi7_end_date,
    kpi7_status_group, kpi7_metric_mode, kpi7_period_mode,
    kpi7_bike_or_car, granularity,
    kpi1_status, kpi2_status, kpi3_status, kpi4_status, kpi5_status, kpi6_status,
    kpi1_vehicle, kpi2_vehicle, kpi3_vehicle, kpi4_vehicle, kpi5_vehicle, kpi6_vehicle,
    kpi1_children, kpi2_children, kpi3_children, kpi4_children, kpi5_children, kpi6_children,
    summary_data, kpi7_fig, kpi7_table_data, kpi7_table_cols
):
    def get_val(children):
        try: return children[0]['props']['children']
        except: return "N/A"

    try:
        html_content = build_view1_download_report(
            kpi1_country, kpi1_year, kpi1_month, get_val(kpi1_children),
            kpi2_country, kpi2_year, kpi2_month, get_val(kpi2_children),
            kpi3_country, kpi3_year, kpi3_month, get_val(kpi3_children),
            kpi4_country, kpi4_year, kpi4_month, get_val(kpi4_children),
            kpi5_country, kpi5_year, kpi5_month, get_val(kpi5_children),
            kpi6_country, kpi6_year, kpi6_month, get_val(kpi6_children),
            kpi_common_bike_or_car,
            kpi7_country, kpi7_start_date, kpi7_end_date,
            kpi7_status_group, kpi7_metric_mode, kpi7_period_mode,
            kpi7_bike_or_car,
            summary_data or [],
            kpi7_fig or {},
            kpi7_table_data or [],
            kpi7_table_cols or []
        )
    except Exception as e:
        print(f"Error building View 1 report: {e}")
        return no_update

    safe_country = str(country or "ALL").replace(" ", "_")
    safe_year = str(year or "ALL")
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
    return ""


@app.callback(
    Output("view2-html-download", "data"),
    Input("view2-download-button", "n_clicks"),
    State("v2-country-filter", "value"),
    State("v2-year-filter", "value"),
    State("v2-metric-mode-filter", "value"),
    State("v2-bike-or-car-filter", "value"),
    State("v2-date-mode-filter", "value"),
    State("v2-period-filter", "value"),
    State("kpi-8-graph", "figure"),
    State("v2-kpi8-table", "data"),
    State("v2-kpi8-table", "columns"),
    prevent_initial_call=True,
)
def download_view2_html(_n_clicks, country, year, metric_mode, bike_or_car, v2_date_mode_filter, period_mode, v2_fig, v2_table_data, v2_table_cols):
    asset_status = "IN FLEET"
    try:
        html_content = build_view2_download_report(
            country, year, asset_status, metric_mode, bike_or_car,
            v2_fig or {},
            v2_table_data or [],
            v2_table_cols or []
        )
    except Exception as e:
        print(f"Error building View 2 report: {e}")
        return no_update

    safe_country = str(country or "ALL").replace(" ", "_")
    safe_year = str(year or "ALL")
    safe_metric = str(metric_mode or "volume")
    filename = f"fleet_monitoring_vue2_{safe_country}_{safe_year}_{safe_metric}.html"
    return dcc.send_string(html_content, filename=filename)


@app.callback(
    Output("v2-kpi8-graph", "figure"),
    Output("v2-kpi8-table-wrap", "children"),
    Input("view2-refresh-button", "n_clicks"),
    State("v2-country-filter", "value"),
    State("v2-year-filter", "value"),
    State("v2-metric-mode-filter", "value"),
    State("v2-bike-or-car-filter", "value"),
    State("v2-date-mode-filter", "value"),
    State("v2-period-filter", "value"),
    prevent_initial_call=True,
)
def update_view2_kpi8(_refresh_clicks: int, country: str, year: int | str, metric_mode: str, bike_or_car: str, v2_date_mode_filter: str, period_mode: str):
    asset_status = "IN FLEET"
    if has_missing_filters(country, year, asset_status, metric_mode, bike_or_car, v2_date_mode_filter, period_mode):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select the filters then click Refresh")
        return empty_fig, html.Div("Select the filters then click Refresh.", className="small-note")

    table_kpi8, y_title, x_title, period_label = get_kpi8_cached(country, year, asset_status, metric_mode, bike_or_car, v2_date_mode_filter, period_mode)
    table_kpi8 = table_kpi8.copy()
    
    if table_kpi8.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"KPI 8 - No data available ({country}, {year})")
        return empty_fig, html.Div("No data available for the selected filters.", className="small-note")

    title_mode = "share" if metric_mode == "share" else "volume"
    title = f"Fuel Type {title_mode} by {x_title.lower()}<br><sup>{period_label} | {country} | {bike_or_car}</sup>"

    fig = go.Figure()
    x_values = table_kpi8["PERIOD"].tolist()
    metric_cols = [c for c in table_kpi8.columns if c not in ["YEAR", "MONTH", "PERIOD"]]
    color_by_metric = fuel_color_map(metric_cols)
    ordered_metric_cols = sorted(metric_cols, key=lambda name: (fuel_label_priority(name), metric_cols.index(name)))
    for col in ordered_metric_cols:
        color = color_by_metric[col]
        fig.add_trace(go.Scatter(
            name=col,
            x=x_values,
            y=table_kpi8[col],
            mode="lines+markers",
            line={"color": color, "width": 2.2},
            marker={"color": color, "size": 7},
            hovertemplate=f"%{{y:.1f}}",
        ))

    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=500,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    table_kpi8 = build_view2_table_with_total(table_kpi8, metric_mode)
    table = build_table(format_view2_table_for_display(table_kpi8), page_size=15, table_id="v2-kpi8-table")
    title_mode_cap = "Share" if metric_mode == "share" else "Volume"
    table_title = f"Production by Fuel Type Table - {asset_status} - {title_mode_cap} - {x_title}"
    return fig, html.Div([html.H4(table_title, className="panel-title"), table])


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
        empty_fig_1.update_layout(title="KPI 9_1 - Sélectionnez les filtres puis cliquez sur Reload")
        empty_fig_2 = go.Figure()
        empty_fig_2.update_layout(title="KPI 9_2 - Sélectionnez les filtres puis cliquez sur Reload")
        note = html.Div("Sélectionnez les filtres puis cliquez sur Reload.", className="small-note")
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
    State("v4-bike-or-car-filter", "value"),
    State("v4-start-date-filter", "value"),
    State("v4-end-date-filter", "value"),
    prevent_initial_call=True,
)
def update_view4_kpi10(_refresh_clicks: int, country: str, status: str, bike_or_car: str, start_date: str | None, end_date: str | None):
    if has_missing_filters(country, status, bike_or_car, start_date, end_date):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 10 - Sélectionnez les filtres puis cliquez sur Reload")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Reload.", className="small-note")

    out = df.copy()
    out = out[out["COUNTRY"] == country].copy()
    if bike_or_car != "ALL":
        if "BIKE_OR_CAR" in out.columns:
            out = out[out["BIKE_OR_CAR"] == bike_or_car]
    out = apply_status_filter(out, status)

    out = out.dropna(subset=["CONTRACT_FINAL_END", "VEHICLE_MODEL_MAPED"])
    out["CONTRACT_FINAL_END"] = pd.to_datetime(out["CONTRACT_FINAL_END"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    if start_date and end_date:
        start_p = pd.Period(start_date, freq="M")
        end_p = pd.Period(end_date, freq="M")
        eoc_p = out["CONTRACT_FINAL_END"].dt.to_period("M")
        out = out[(eoc_p >= start_p) & (eoc_p <= end_p)]
    
    if out.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 10 - No data")
        return empty_fig, html.Div("No data available for the selected filters.", className="small-note")

    out["Quarter"] = out["CONTRACT_FINAL_END"].dt.to_period("Q").astype(str)

    out = get_kpi_data(out, extra_keys=["QUARTER"])
    
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

        if start_date and end_date:
            window_label = f"from {start_date} to {end_date}"
        else:
            window_label = "all horizon"
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
    return fig, table


@app.callback(
    [
        Output("v5-status-filter-label", "children"),
        Output("v5-status-filter", "options"),
        Output("v5-status-filter", "value"),
        Output("v5-variable-filter-label", "children"),
        Output("v5-variable-filter", "options"),
        Output("v5-variable-filter", "value"),
        Output("v5-market-owner-container", "style"),
    ],
    Input("v5-source-filter", "value"),
)
def sync_v5_source_filters(source):
    if source == "market":
        status_label = "Market Registration type"
        status_options = [
            {"label": "Passenger Cars", "value": "Passenger Cars"},
            {"label": "Light Commercial Vehicle", "value": "Light Commercial Vehicle"},
            {"label": "Heavy Commercial Vehicle", "value": "Heavy Commercial Vehicle"}
        ]
        status_value = "Passenger Cars"
        var_label = "Market Variable"
        var_options = CONCENTRATION_VARIABLE_OPTIONS_MARKET
        var_value = var_options[0]["value"] if var_options else None
        owner_style = {"display": "block"}
    else:
        status_label = "Portfolio Status type"
        status_options = [{"label": s, "value": s} for s in CONCENTRATION_STATUS_OPTIONS]
        status_value = "IN FLEET"
        var_label = "Portfolio Variable"
        var_options = CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO
        var_value = var_options[0]["value"] if var_options else None
        owner_style = {"display": "none"}
    
    return status_label, status_options, status_value, var_label, var_options, var_value, owner_style


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
def update_view5_kpi11(_refresh_clicks: int, status_value: str, source_value: str, variable: str, top_n_val: str, period_mode: str, market_owner: str):
    if has_missing_filters(status_value, source_value, variable):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 11 - Sélectionnez les filtres puis cliquez sur Reload")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Reload.", className="small-note")

    top_n_val = top_n_val or "ALL"

    if source_value == "market":
        # Filter market data by Registration Type (selected via the repurposed status filter)
        m_filtered = market_df[market_df["Registration Type"] == status_value].copy()
        if market_owner and market_owner != "ALL":
            m_filtered = m_filtered[m_filtered["Ownertype"] == market_owner]
        
        market_ready, market_var = prepare_market_concentration_source(m_filtered, variable)
        kpi11_table_long = kpi_count_share_quarterly_market(market_ready, market_var, top_n=top_n_val, period_mode=period_mode)
        fig = plot_kpi_share(kpi11_table_long, market_var, period_mode)
        kpi11_table = kpi11_table_long
    else:
        portfolio_ready, portfolio_var = prepare_portfolio_concentration_source(df, variable)
        if "YTD" in status_value.upper():
            kpi11_table = kpi_count_share_ytd_by_quarter(portfolio_ready, asset_status=status_value.replace(" YTD", ""), var_col=portfolio_var, bike_or_car="CAR", top_n=top_n_val, period_mode=period_mode)
            plot_df = kpi11_table.rename(columns={"VOLUME_YTD": "VOLUME", "SHARE_YTD": "SHARE"}).copy()
        else:
            kpi11_table = kpi_count_share_quarterly(portfolio_ready, asset_status=status_value, var_col=portfolio_var, bike_or_car="CAR", top_n=top_n_val, period_mode=period_mode)
            plot_df = kpi11_table.copy()

        if plot_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="KPI 11 - No data")
            return empty_fig, html.Div("No data available for the selected filters.", className="small-note")

        fig = plot_kpi_share(plot_df, portfolio_var, period_mode)

    if kpi11_table.empty:
        return fig, html.Div("No data available for the selected filters.", className="small-note")

    table = build_table(kpi11_table, page_size=15, heatmap=True)
    return fig, html.Div([html.H4("KPI 11 table", className="panel-title"), table])


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
def update_view6_kpi6(_refresh_clicks: int, status_value: str, source_value: str, variable: str, metric_mode: str, top_n_val: str, period_mode: str):
    if has_missing_filters(status_value, source_value, variable, metric_mode):
        return html.Div("Sélectionnez les filtres puis cliquez sur Reload.", className="small-note")

    top_n_val = top_n_val or "ALL"

    if source_value == "market":
        market_ready, market_var = prepare_market_concentration_source(market_df, variable)
        kpi6_table = kpi_top_per_quarter_with_share_market(market_ready, market_var, metric_mode=metric_mode, top_n=top_n_val, period_mode=period_mode)
    else:
        portfolio_ready, portfolio_var = prepare_portfolio_concentration_source(df, variable)
        kpi6_table = kpi_top_per_quarter_with_share(portfolio_ready, asset_status=status_value, var_col=portfolio_var, bike_or_car="CAR", metric_mode=metric_mode, top_n=top_n_val, period_mode=period_mode)

    if kpi6_table.empty:
        return html.Div("No data available for the selected filters.", className="small-note")

    table = build_table(kpi6_table, page_size=15, heatmap=True)
    return html.Div([html.H4(f"Top {variable} per country and quarter", className="panel-title"), table])




@app.callback(
    Output("v6-variable-filter", "options"),
    Input("v6-source-filter", "value"),
)
def update_v6_variable_options(source):
    if source == "market":
        return CONCENTRATION_VARIABLE_OPTIONS_MARKET
    return CONCENTRATION_VARIABLE_OPTIONS_PORTFOLIO


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
def update_view6_kpi13(_refresh_clicks: int, status_value: str, port_reg: list, mkt_reg: list, variable: str, period_mode: str, top_n: str, owner_type: str):
    if has_missing_filters(status_value, port_reg, mkt_reg, variable):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 13 - Sélectionnez les filtres puis cliquez sur Reload")
        return empty_fig, html.Div("Sélectionnez les filtres puis cliquez sur Reload.", className="small-note")

    if market_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="KPI 13 - Market dataset not found")
        return empty_fig, html.Div("Market dataset is not available.", className="small-note")

    m_subset = market_df.copy()
    if mkt_reg and mkt_reg != "ALL" and mkt_reg != ["ALL"]:
        m_subset = m_subset[m_subset["Registration Type"].isin(mkt_reg)]
    
    if owner_type and owner_type != "ALL":
        m_subset = m_subset[m_subset["Ownertype"] == owner_type]

    # Use standardized prep functions from View 5 to ensure consistency (e.g. CO2 Bucket)
    portfolio_ready, portfolio_var = prepare_portfolio_concentration_source(df, variable)
    market_ready, market_var = prepare_market_concentration_source(m_subset, variable)
    
    bev_only = (variable == "HIGHEST_BEV")

    kpi13 = kpi_top_brand_vs_market(
        df_portfolio=portfolio_ready,
        df_market=market_ready,
        var_col_portfolio=portfolio_var,
        var_col_market=market_var,
        asset_status=status_value,
        reg_type_portfolio=port_reg,
        reg_type_market=mkt_reg,
        bev_only=bev_only,
        top_n=top_n,
        period_mode=period_mode
    )

    title = f"KPI 13 - {variable} Portfolio vs Market ({status_value} / {reg_type})"
    fig = figure_top_brand_vs_market(kpi13, title)

    if kpi13.empty:
        return fig, html.Div("No data available for selected filters.", className="small-note")

    columns = [
        "COUNTRY",
        "PERIOD",
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




@app.callback(
    Output('complete-report-modal', 'style'),
    [
        Input({"type": "open-complete-report", "index": ALL}, 'n_clicks'),
        Input('close-complete-report-modal', 'n_clicks')
    ],
    State('complete-report-modal', 'style'),
    prevent_initial_call=True
)
def toggle_complete_report_modal(open_clicks_list, n_close, current_style):
    import dash
    ctx = dash.callback_context
    new_style = current_style.copy() if current_style else {
        "display": "none", "position": "fixed", "top": "0", "left": "0", "width": "100%", 
        "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)", "zIndex": "9999", 
        "justifyContent": "center", "alignItems": "center"
    }

    if not ctx.triggered:
        return new_style
        
    prop_id = ctx.triggered[0]['prop_id']
    if 'close-complete-report-modal' in prop_id:
        new_style['display'] = 'none'
    elif 'open-complete-report' in prop_id:
        if any(c is not None and c > 0 for c in open_clicks_list):
            new_style['display'] = 'flex'
    return new_style



def build_interactive_complete_report(
    country: str,
    bike_or_car: str,
    status_group: str,
    selected_years: list[str],
    selected_periods: list[str],
    date_mode: str = "NONE",
    kpi1_limit: float | None = 25.0,
    kpi2_limit: float | None = 30.0,
) -> str:
    sections = []
    
    controls_html = f"""
    <div style="padding: 16px 28px; background: #fff; border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 1000; display: flex; gap: 20px; align-items: center; box-shadow: 0 4px 12px rgba(16,42,67,0.05);">
        <strong style="color: var(--ink);">Interactive Report Settings:</strong>
        <label style="color: var(--ink); margin-left: 10px;">Year:</label>
        <select id="report-year" onchange="updateReportView()" style="padding: 6px 12px; border-radius: 8px; border: 1px solid var(--line);">
            {''.join(f'<option value="{y}">{y}</option>' for y in selected_years)}
        </select>
        <label style="color: var(--ink); margin-left: 10px;">Granularity:</label>
        <select id="report-granularity" onchange="updateReportView()" style="padding: 6px 12px; border-radius: 8px; border: 1px solid var(--line);">
            {''.join(f'<option value="{p}">{p.capitalize()}</option>' for p in selected_periods)}
        </select>
        <label style="color: var(--ink); margin-left: 10px;">Period:</label>
        <select id="report-subperiod" onchange="updateReportView()" style="padding: 6px 12px; border-radius: 8px; border: 1px solid var(--line);">
            <!-- Populated by JS -->
        </select>
    </div>
    """
    
    js_logic = """
    <script>
    function updateReportView() {
        var year = document.getElementById('report-year').value;
        var gran = document.getElementById('report-granularity').value;
        var subSelect = document.getElementById('report-subperiod');
        
        // Save current sub before changing options to try and restore it if possible
        var currentSub = subSelect.value;
        
        // Update subperiod options based on granularity
        if (gran === 'monthly') {
            if (subSelect.options.length !== 12) {
                subSelect.innerHTML = '';
                for(var i=1; i<=12; i++) {
                    var opt = document.createElement('option');
                    opt.value = i; opt.innerHTML = i;
                    subSelect.appendChild(opt);
                }
            }
        } else if (gran === 'quarterly') {
            if (subSelect.options.length !== 4) {
                subSelect.innerHTML = '';
                for(var i=1; i<=4; i++) {
                    var opt = document.createElement('option');
                    opt.value = 'Q' + i; opt.innerHTML = 'Q' + i;
                    subSelect.appendChild(opt);
                }
            }
        } else {
            subSelect.innerHTML = '<option value="ALL">Annual</option>';
        }
        
        // Try to restore previous selection if it exists in new options
        var found = false;
        for(var i=0; i<subSelect.options.length; i++){
            if(subSelect.options[i].value === currentSub) { subSelect.value = currentSub; found = true; break; }
        }
        if(!found && subSelect.options.length > 0) subSelect.selectedIndex = 0;
        
        var sub = subSelect.value;
        
        // Hide all blocks, show matching
        var blocks = document.getElementsByClassName('report-block');
        for(var i=0; i < blocks.length; i++) {
            var b = blocks[i];
            if (b.getAttribute('data-year') === year && b.getAttribute('data-gran') === gran && b.getAttribute('data-sub') === sub) {
                b.style.display = 'block';
            } else {
                b.style.display = 'none';
            }
        }
        window.dispatchEvent(new Event('resize'));
    }
    window.onload = updateReportView;
    </script>
    """
    
    body_html = [controls_html]
    desc = kpi_summary_description_map()
    
    for year in selected_years:
        for gran in selected_periods:
            subs = []
            if gran == "monthly": subs = [str(i) for i in range(1, 13)]
            elif gran == "quarterly": subs = ["Q1", "Q2", "Q3", "Q4"]
            else: subs = ["ALL"]
            
            for sub in subs:
                month_val = sub
                if gran == "quarterly":
                    mapping = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}
                    month_val = mapping[sub]
                
                resolved_month = resolve_month_value(country, year, month_val)
                
                # --- CALCULATE DATA DYNAMICALLY ---
                results = {}
                for kid, config in KPI_REGISTRY.items():
                    func = globals().get(config["func_name"])
                    if func:
                        results[kid] = func(df, country, year, resolved_month, bike_or_car, date_mode)
                    else:
                        results[kid] = None
                
                kpi_vol = kpi_selected_volume(df, country, year, resolved_month, status="IN FLEET", bike_or_car=bike_or_car, date_mode=date_mode)
                
                if gran == "quarterly":
                    period_display = f"{sub} {year}"
                elif gran == "monthly":
                    period_display = summary_month_label(year, int(month_val))
                else:
                    period_display = str(year)
                
                # --- BUILD KPI CARDS DYNAMICALLY ---
                kpi_cards = []
                for kid, config in KPI_REGISTRY.items():
                    val = results[kid]
                    limit = config["limit"]
                    
                    display_val = "N/A"
                    status = "neutral"
                    if val is not None:
                        if isinstance(val, tuple):
                            display_val = f"{percent_or_na_precision(val[0], 2)} / {percent_or_na_precision(val[1], 2)}"
                        else:
                            display_val = percent_or_na_precision(val, 2)
                            status = kpi_limit_status(val, limit)
                    
                    kpi_cards.append({
                        "label": config["label"],
                        "result": display_val,
                        "status": status,
                        "show_legend": True,
                        "params": [
                            ("Country", country),
                            ("Period", period_display),
                            ("Vehicle type", bike_or_car),
                            ("Asset status", status_group),
                            ("Volume", format_volume(kpi_vol)),
                        ],
                    })
                
                cards_html = build_kpi_cards_section("Main KPIs", kpi_cards)
                
                # --- BUILD SUMMARY TABLE DYNAMICALLY ---
                kpi_rows = []
                for kid, config in KPI_REGISTRY.items():
                    val = results[kid]
                    limit = config["limit"]
                    
                    display_val = "N/A"
                    status = "neutral"
                    if val is not None:
                        if isinstance(val, tuple):
                            display_val = f"{percent_or_na_precision(val[0], 2)} / {percent_or_na_precision(val[1], 2)}"
                        else:
                            display_val = percent_or_na_precision(val, 2)
                            status = kpi_limit_status(val, limit)
                    
                    limit_desc = f", Limit {limit}%" if limit else ""
                    
                    kpi_rows.append({
                        "label": config["label"].split(" - ")[-1],
                        "description": config["description"] + limit_desc,
                        "country": country,
                        "period": period_display,
                        "result": display_val,
                        "volume": format_volume(kpi_vol),
                        "signal": status,
                    })
                
                summary_df = pd.DataFrame(kpi_rows)[["label", "country", "period", "result", "volume", "description", "signal"]]
                summary_table_html = build_html_table_from_df(summary_df, "KPI summary table")

                block += cards_html
                block += summary_table_html
                
                body_html.append(block)
                body_html.append("</div>")

                kpi7_pivot, y_title7, x_title7, period_label7 = kpi7_fuel_by_period(
                    df, country, status_group, "share", gran, bike_or_car, "COB_DATE", f"{year}-01-01", f"{year}-12-31"
                )
                kpi7_fig = figure_from_pivot(
                    kpi7_pivot, y_title7, x_title7,
                    f"Fuel Type share by {gran}<br><sup>{country} | {status_group} | {bike_or_car}</sup>",
                )
                kpi7_html = build_kpi7_metadata_section(country, status_group, "share", gran, bike_or_car, None, None, kpi7_fig)
                kpi7_table_html = build_html_table_from_df(
                    build_kpi7_table_with_total(kpi7_pivot, "share", country, status_group, gran, bike_or_car, None, None),
                    f"Fuel Type Share Table - {status_group} - Share - {gran.capitalize()}"
                )
                
                block += kpi7_html
                block += kpi7_table_html
                
                block += "</div>"
                body_html.append(block)
                
    body_html.append(js_logic)
    return html_report_document("Interactive Complete Report", body_html)


@app.callback(
    Output("complete-report-download", "data"),
    Input("download-complete-report-btn", "n_clicks"),
    State("report-modal-years", "value"),
    State("report-modal-periods", "value"),
    State("report-modal-date-mode", "value"),
    State("country-filter", "value"),
    State("kpi-common-bike-or-car-filter", "value"),
    State("status-group-filter", "value"),
    prevent_initial_call=True
)
def generate_interactive_report_callback(n_clicks, selected_years, selected_periods, date_mode, country, bike_or_car, status_group):
    if not selected_years or not selected_periods:
        return no_update
        
    country = country or "ALL"
    bike_or_car = bike_or_car or "CAR"
    status_group = status_group or "IN FLEET"
    date_mode = date_mode or "NONE"
    
    html_content = build_interactive_complete_report(
        country, bike_or_car, status_group, selected_years, selected_periods, date_mode
    )
    
    filename = f"Complete_Interactive_Report_{country}.html"
    return dcc.send_string(html_content, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
  




