
import os
import sys
import gc
import itertools
import time
import argparse
import re
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
PARENT_DIR       = Path(__file__).resolve().parent.parent          # …/Arval/New/
DATA_FOLDER      = PARENT_DIR / "data"                             # raw NOVA parquets
BASE_OUT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # precomputed output
MARKET_DATA_PATH = PARENT_DIR / "market_dataset_europe.parquet"   # used by View 5 & 6

ALL_COUNTRIES = [
    "BELGIUM", "FRANCE", "GERMANY", "ITALY",
    "LUXEMBOURG", "NETHERLANDS", "SPAIN", "UNITED KINGDOM",
]

# Default country list — override with --countries CLI arg
COUNTRIES_TO_RUN = ["SPAIN", "ITALY"]

START_YYYYMM = "202301"
END_YYYYMM   = "202602"

CONFIG = {
    "GENERATE_VIEW1":       False,
    "GENERATE_VIEW2":       False,
    "GENERATE_VIEW3":       False,
    "GENERATE_VIEW4":       False,
    "GENERATE_VIEW5":       True,
    "GENERATE_VIEW6":       True,
    "USE_UNIQUE_KEY_LOGIC": True,
    "UNIQUE_KEY_COLS":      ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"],
}

# ── Import fmd (no data loaded yet) ───────────────────────────────────────────
os.environ["FAST_READER_MODE"] = "1"
sys.path.insert(0, str(PARENT_DIR))
import fleet_monitoring_dashboard_key_refactor as fmd

fmd.USE_UNIQUE_KEY_LOGIC    = CONFIG["USE_UNIQUE_KEY_LOGIC"]
fmd.UNIQUE_KEY_COLS         = CONFIG["UNIQUE_KEY_COLS"]
fmd.DEFAULT_MARKET_DATA_PATH = MARKET_DATA_PATH          # View 5 & 6
MARKET_DF = fmd.load_market_dataset()


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.columns = df.columns.astype(str)
    df.to_parquet(path, index=False)


def slugify_value(value: object) -> str:
    text = "ALL" if value is None else str(value)
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "ALL"


def serialize_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        parts = [serialize_value(item) for item in value]
        return "|".join(part for part in parts if part)
    if value is None:
        return "ALL"
    if isinstance(value, float) and pd.isna(value):
        return "ALL"
    return str(value)


def annotate_frame(df: pd.DataFrame, **meta: object) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for key, value in meta.items():
        out[key] = serialize_value(value)
    return out


def non_empty_subsets(values: list[object]) -> list[list[object]]:
    cleaned = []
    seen = set()
    for value in values:
        marker = serialize_value(value)
        if marker not in seen:
            cleaned.append(value)
            seen.add(marker)

    subsets: list[list[object]] = [["ALL"]]
    for size in range(1, len(cleaned) + 1):
        for combo in itertools.combinations(cleaned, size):
            subsets.append(list(combo))
    return subsets


def view_output_path(*parts: object) -> Path:
    return Path(BASE_OUT).joinpath(*[slugify_value(part) for part in parts])


def get_years(df: pd.DataFrame) -> list[int]:
    return sorted(int(x) for x in df["YEAR"].dropna().unique())


def get_period_modes() -> list[str]:
    return ["monthly", "quarterly", "yearly"]


def get_country_code_column(df: pd.DataFrame) -> str | None:
    codes = [str(x) for x in df["COUNTRY"].dropna().unique()]
    return codes[0] if codes else None


def get_vehicle_type_values(df: pd.DataFrame) -> list[str]:
    vehicle_col = fmd.pick_first_existing_column(df, ["CLS_VEHICLE_TYPE", "VEHICLE_CLASS"])
    if not vehicle_col:
        return ["ALL"]
    values = sorted(df[vehicle_col].dropna().astype(str).str.upper().unique().tolist())
    return ["ALL"] + values


def get_market_body_values(df: pd.DataFrame) -> list[str]:
    if "MARKET_BODY_GROUP" not in df.columns:
        return ["ALL"]
    values = sorted(df["MARKET_BODY_GROUP"].dropna().astype(str).str.upper().unique().tolist())
    return ["ALL"] + values



# ── Per-country generators ────────────────────────────────────────────────────

def _get_country_code(df: pd.DataFrame) -> str | None:
    """Return the actual country code stored in the COUNTRY column (e.g. 'ES' for Spain)."""
    codes = [str(x) for x in df["COUNTRY"].dropna().unique()]
    return codes[0] if codes else None


def generate_kpis_for_country(df: pd.DataFrame) -> None:
    country_code = _get_country_code(df)
    if country_code is None:
        return

    years        = get_years(df)
    months       = ["ALL", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    actual_bocs  = sorted(df["BIKE_OR_CAR"].dropna().astype(str).unique().tolist())
    bocs         = ["ALL"] + actual_bocs
    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    dmodes       = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]

    for year in years:
        print(f"  KPI1-6 | {country_code} {year} computing...")
        df_year = df[df["YEAR"] == year].copy()
        records = []
        combos  = list(itertools.product(months, bocs, statuses, dmodes))

        for month, boc, status, dmode in combos:
            resolved   = fmd.resolve_month_value(country_code, year, month)
            kpi1       = fmd.kpi_lease_under_25(df_year, country_code, year, resolved, boc, dmode, status)
            kpi2       = fmd.kpi_lease_25_30(df_year, country_code, year, resolved, boc, dmode, status)
            diesel_non = fmd.kpi_diesel_non_diesel(df_year, country_code, year, resolved, boc, dmode, status)
            hybrid     = fmd.kpi_hybrid_share(df_year, country_code, year, resolved, boc, dmode, status)
            ev         = fmd.kpi_ev_share(df_year, country_code, year, resolved, boc, dmode, status)
            pv_lcv     = fmd.kpi_pv_lcv(df_year, country_code, year, resolved, boc, dmode, status)  # type: ignore[arg-type]
            volume     = fmd.kpi_selected_volume(df_year, country_code, year, resolved, status, boc, dmode)

            records.append({
                "COUNTRY":      country_code,
                "YEAR":         int(year),
                "MONTH":        str(month),
                "BIKE_OR_CAR":  boc,
                "ASSET_STATUS": status,
                "DATE_MODE":    dmode,
                "KPI1":         kpi1,
                "KPI2":         kpi2,
                "DIESEL":       diesel_non[0] if diesel_non else None,
                "NON_DIESEL":   diesel_non[1] if diesel_non else None,
                "HYBRID":       hybrid,
                "EV":           ev,
                "PV":           pv_lcv[0] if pv_lcv else None,
                "LCV":          pv_lcv[1] if pv_lcv else None,
                "VOLUME":       volume,
            })

        result_df    = pd.DataFrame(records)
        safe_country = country_code.replace(" ", "_")
        path = os.path.join(
            BASE_OUT, "view1", "kpis",
            f"year={year}", f"country={safe_country}", "kpis.parquet"
        )
        save_parquet(result_df, path)
        print(f"  KPI1-6 | {country_code} {year}: {len(result_df):,} records")


def generate_kpi7_for_country(df: pd.DataFrame) -> None:
    country_code = _get_country_code(df)
    if country_code is None:
        return

    years        = get_years(df)
    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    metric_modes = ["share", "volume"]
    period_modes = get_period_modes()
    actual_bocs  = sorted(df["BIKE_OR_CAR"].dropna().astype(str).unique().tolist())
    bocs         = ["ALL"] + actual_bocs
    combos       = list(itertools.product(statuses, metric_modes, period_modes, bocs))

    for year in years:
        print(f"  KPI7   | {country_code} {year} computing...")
        df_year    = df[df["YEAR"] == year].copy()
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        all_results = []

        for status, mm, pm, boc in combos:
            res_df, _, _, _ = fmd.kpi7_fuel_by_period(
                df_year, country_code, status, mm, pm,
                bike_or_car=boc, date_mode="COB_DATE",
                start_date=start_date, end_date=end_date,
            )
            if res_df is None or res_df.empty:
                continue

            res_df = res_df.copy()
            res_df.index.name = "FUEL_TYPE"
            if "FUEL_TYPE" not in res_df.columns:
                res_df = res_df.reset_index()

            res_df["COUNTRY"]      = country_code
            res_df["YEAR"]         = int(year)
            res_df["ASSET_STATUS"] = status
            res_df["METRIC_MODE"]  = mm
            res_df["PERIOD_MODE"]  = pm
            res_df["BIKE_OR_CAR"]  = boc
            all_results.append(res_df)

        if all_results:
            combined     = pd.concat(all_results, ignore_index=True)
            safe_country = country_code.replace(" ", "_")
            path = os.path.join(
                BASE_OUT, "view1", "kpi7",
                f"year={year}", f"country={safe_country}", "kpi7.parquet"
            )
            save_parquet(combined, path)
            print(f"  KPI7   | {country_code} {year}: {len(combined):,} rows")


def generate_kpi8_for_country(df: pd.DataFrame) -> None:
    country_code = _get_country_code(df)
    if country_code is None:
        return

    years        = get_years(df)
    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    metric_modes = ["share", "volume"]
    actual_bocs  = sorted(df["BIKE_OR_CAR"].dropna().astype(str).unique().tolist())
    bocs         = ["ALL"] + actual_bocs
    dmodes       = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]
    period_modes = get_period_modes()
    combos       = list(itertools.product(statuses, metric_modes, bocs, dmodes, period_modes))

    for year in years:
        print(f"  KPI8   | {country_code} {year} computing...")
        df_year     = df[df["YEAR"] == year].copy()
        all_results = []

        for status, mm, boc, dmode, pm in combos:
            res_df, _, _, _ = fmd.kpi8_production_ytd(
                df_year, country_code, year,
                asset_status=status, metric_mode=mm,
                bike_or_car=boc, date_mode=dmode, period_mode=pm,
            )
            if res_df is None or res_df.empty:
                continue

            res_df = res_df.copy()
            res_df["COUNTRY"]      = country_code
            res_df["YEAR"]         = int(year)
            res_df["ASSET_STATUS"] = status
            res_df["METRIC_MODE"]  = mm
            res_df["BIKE_OR_CAR"]  = boc
            res_df["DATE_MODE"]    = dmode
            res_df["PERIOD_MODE"]  = pm
            all_results.append(res_df)

        if all_results:
            combined     = pd.concat(all_results, ignore_index=True)
            safe_country = country_code.replace(" ", "_")
            path = os.path.join(
                BASE_OUT, "view2", "kpi8",
                f"year={year}", f"country={safe_country}", "kpi8.parquet"
            )
            save_parquet(combined, path)
            print(f"  KPI8   | {country_code} {year}: {len(combined):,} rows")


def generate_view3_for_country(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame(), pd.DataFrame()

    years = get_years(df)
    vehicle_types = get_vehicle_type_values(df)
    market_bodies = get_market_body_values(df)
    bike_or_cars = ["ALL"] + sorted(df["BIKE_OR_CAR"].dropna().astype(str).str.upper().unique().tolist()) if "BIKE_OR_CAR" in df.columns else ["ALL"]
    period_modes = get_period_modes()
    status_values = [item["value"] for item in fmd.asset_status_options if item["value"] != "ALL"]
    status_subsets = non_empty_subsets(status_values)

    rows_91: list[pd.DataFrame] = []
    rows_92: list[pd.DataFrame] = []

    for year in years:
        df_year = df[df["YEAR"] == year].copy()
        for period_mode in period_modes:
            for bike_or_car in bike_or_cars:
                for vehicle_class in vehicle_types:
                    local_df = df_year
                    if bike_or_car != "ALL" and "BIKE_OR_CAR" in local_df.columns:
                        local_df = local_df[local_df["BIKE_OR_CAR"].astype(str).str.upper() == bike_or_car]
                    if vehicle_class != "ALL":
                        vcol = fmd.vehicle_col_global if hasattr(fmd, "vehicle_col_global") else None
                        if vcol and vcol in local_df.columns:
                            local_df = local_df[local_df[vcol].astype(str).str.upper() == vehicle_class]

                    for vehicle_body in market_bodies:
                        for selected_statuses in status_subsets:
                            if not selected_statuses:
                                continue
                            if selected_statuses == ["ALL"]:
                                status_list = ["ALL"]
                            else:
                                status_list = selected_statuses

                            for status_value in status_list:
                                kpi91 = fmd.kpi9_1_type_share_period(local_df, year, vehicle_body, country_code, status_value, bike_or_car, period_mode)
                                if kpi91.empty:
                                    continue
                                current = kpi91.copy()
                                current["ASSET_STATUS"] = status_value
                                current["YEAR_FILTER"] = year
                                current["COUNTRY_FILTER"] = country_code
                                current["VEHICLE_CLASS_FILTER"] = vehicle_class
                                current["VEHICLE_BODY_FILTER"] = vehicle_body
                                current["BIKE_OR_CAR_FILTER"] = bike_or_car
                                current["PERIOD_MODE_FILTER"] = period_mode
                                current["STATUS_SELECTION"] = serialize_value(status_list)
                                rows_91.append(current)

                    for status_value in status_values:
                        kpi92 = fmd.kpi9_2_power_category_per_type_period(local_df, year, vehicle_body, country_code, status_value, bike_or_car, period_mode)
                        if kpi92.empty:
                            continue
                        kpi92 = kpi92.reset_index().rename(columns={"PERIOD": "Period"})
                        kpi92["YEAR_FILTER"] = year
                        kpi92["COUNTRY_FILTER"] = country_code
                        kpi92["VEHICLE_CLASS_FILTER"] = vehicle_class
                        kpi92["VEHICLE_BODY_FILTER"] = vehicle_body
                        kpi92["BIKE_OR_CAR_FILTER"] = bike_or_car
                        kpi92["PERIOD_MODE_FILTER"] = period_mode
                        kpi92["STATUS_FILTER"] = status_value
                        rows_92.append(kpi92)

    return (
        pd.concat(rows_91, ignore_index=True) if rows_91 else pd.DataFrame(),
        pd.concat(rows_92, ignore_index=True) if rows_92 else pd.DataFrame(),
    )


def generate_view4_for_country(df: pd.DataFrame) -> pd.DataFrame:
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame()

    years = get_years(df)
    bike_or_cars = ["ALL"] + sorted(df["BIKE_OR_CAR"].dropna().astype(str).str.upper().unique().tolist()) if "BIKE_OR_CAR" in df.columns else ["ALL"]
    statuses = [item["value"] for item in fmd.asset_status_options]
    period_modes = get_period_modes()

    rows: list[pd.DataFrame] = []
    for year in years:
        for status in statuses:
            for period_mode in period_modes:
                for bike_or_car in bike_or_cars:
                    out = df.copy()
                    out = out[out["COUNTRY"] == country_code].copy()
                    if year != "ALL":
                        out = out[pd.to_datetime(out["CONTRACT_FINAL_END"], errors="coerce").dt.year == int(year)]
                    if bike_or_car != "ALL" and "BIKE_OR_CAR" in out.columns:
                        out = out[out["BIKE_OR_CAR"].astype(str).str.upper() == bike_or_car]
                    out = fmd.apply_status_filter(out, status)
                    if "MARKET_MODEL" not in out.columns:
                        continue
                    out = out.dropna(subset=["CONTRACT_FINAL_END", "MARKET_MODEL"])
                    out["CONTRACT_FINAL_END"] = pd.to_datetime(out["CONTRACT_FINAL_END"], errors="coerce")
                    out = out.dropna(subset=["CONTRACT_FINAL_END"])
                    if out.empty:
                        continue

                    period_subset, _, _ = fmd.build_period_column(out, "CONTRACT_FINAL_END", period_mode)
                    period_subset["MARKET_MODEL"] = period_subset["MARKET_MODEL"].astype(str).str.strip()
                    key_cols = [column for column in ["ID_QUOTATION", "ID_CONTRACT", "VEHICLE_ID"] if column in period_subset.columns]
                    if key_cols:
                        period_subset = period_subset.drop_duplicates(subset=key_cols)

                    period_subset["_tmp_key"] = fmd.get_unique_key_series(period_subset)
                    grouped = period_subset.groupby(["PERIOD", "MARKET_MODEL"])["_tmp_key"].nunique().reset_index(name="COUNT")
                    pivot = grouped.pivot(index="PERIOD", columns="MARKET_MODEL", values="COUNT").fillna(0)
                    if pivot.empty:
                        continue

                    total_row = pd.DataFrame(pivot.sum()).T
                    total_row.index = ["Total"]
                    pivot = pd.concat([pivot, total_row])
                    total_series = pd.Series(pivot.loc["Total"].to_numpy(), index=pivot.columns)
                    top_models_order = total_series.sort_values(ascending=False).index[:10]
                    pivot = pivot[top_models_order]
                    table = pivot.reset_index().rename(columns={"PERIOD": "Period"})
                    table["COUNTRY_FILTER"] = country_code
                    table["YEAR_FILTER"] = year
                    table["ASSET_STATUS_FILTER"] = status
                    table["PERIOD_MODE_FILTER"] = period_mode
                    table["BIKE_OR_CAR_FILTER"] = bike_or_car
                    rows.append(table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def generate_view5_portfolio_for_country(df: pd.DataFrame) -> pd.DataFrame:
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame()

    statuses = ["IN FLEET", "ORDER", "ORDER YTD", "DEHIRE"]
    period_modes = get_period_modes()
    top_n_values = ["1", "5", "10", "20", "ALL"]
    rows: list[pd.DataFrame] = []

    for status_value in statuses:
        for variable in ["BRAND_UPDATE", "OEM_UPDATE", "CO2_BUCKET", "HIGHEST_BEV"]:
            portfolio_ready, portfolio_var = fmd.prepare_portfolio_concentration_source(df, variable)
            for period_mode in period_modes:
                for top_n in top_n_values:
                    if "YTD" in status_value.upper():
                        kpi11_table = fmd.kpi_count_share_ytd_by_quarter(portfolio_ready, asset_status=status_value.replace(" YTD", ""), var_col=portfolio_var, bike_or_car="CAR", top_n=top_n, period_mode=period_mode)
                    else:
                        kpi11_table = fmd.kpi_count_share_quarterly(portfolio_ready, asset_status=status_value, var_col=portfolio_var, bike_or_car="CAR", top_n=top_n, period_mode=period_mode)
                    if kpi11_table.empty:
                        continue
                    kpi11_table = kpi11_table.copy()
                    kpi11_table["SOURCE_FILTER"] = "portfolio"
                    kpi11_table["COUNTRY_FILTER"] = country_code
                    kpi11_table["STATUS_FILTER"] = status_value
                    kpi11_table["VARIABLE_FILTER"] = variable
                    kpi11_table["TOP_N_FILTER"] = top_n
                    kpi11_table["PERIOD_MODE_FILTER"] = period_mode
                    rows.append(kpi11_table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def generate_view5_market(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    reg_types = [
        ["Passenger Cars"],
        ["Light Commercial Vehicle"],
        ["Heavy Commercial Vehicle"],
        ["Passenger Cars", "Light Commercial Vehicle"],
        ["Passenger Cars", "Heavy Commercial Vehicle"],
        ["Light Commercial Vehicle", "Heavy Commercial Vehicle"],
        ["Passenger Cars", "Light Commercial Vehicle", "Heavy Commercial Vehicle"],
    ]
    owners = [
        "ALL",
        "Car manufacturer / dealer",
        "Company Cars",
        "Administration Govt.",
        "Private",
        "Rental",
        "Unspecified",
    ]
    period_modes = get_period_modes()
    top_n_values = ["1", "5", "10", "20", "ALL"]

    for variable in ["BRAND_UPDATE", "OEM_UPDATE", "CO2_BUCKET", "HIGHEST_BEV"]:
        market_ready, market_var = fmd.prepare_market_concentration_source(market_df, variable)
        for reg_value in reg_types:
            for owner in owners:
                m_filtered = market_ready.copy()
                if reg_value and reg_value != ["ALL"] and "Registration Type" in m_filtered.columns:
                    m_filtered = m_filtered[m_filtered["Registration Type"].isin(reg_value)]
                if owner and owner != "ALL" and "Ownertype" in m_filtered.columns:
                    m_filtered = m_filtered[m_filtered["Ownertype"] == owner]
                for period_mode in period_modes:
                    for top_n in top_n_values:
                        kpi11_table = fmd.kpi_count_share_quarterly_market(m_filtered, market_var, top_n=top_n, period_mode=period_mode)
                        if kpi11_table.empty:
                            continue
                        kpi11_table = kpi11_table.copy()
                        kpi11_table["SOURCE_FILTER"] = "market"
                        kpi11_table["REG_TYPE_FILTER"] = serialize_value(reg_value)
                        kpi11_table["OWNER_FILTER"] = owner
                        kpi11_table["VARIABLE_FILTER"] = variable
                        kpi11_table["TOP_N_FILTER"] = top_n
                        kpi11_table["PERIOD_MODE_FILTER"] = period_mode
                        rows.append(kpi11_table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def generate_view6_for_country(df: pd.DataFrame) -> pd.DataFrame:
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame()
    if MARKET_DF.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    status_values = ["IN FLEET", "ORDER", "ORDER YTD", "DEHIRE"]
    port_reg_values = [
        ["PV"],
        ["LCV"],
        ["PV", "LCV"],
    ]
    mkt_reg_values = [
        ["Passenger Cars"],
        ["Light Commercial Vehicle"],
        ["Heavy Commercial Vehicle"],
        ["Passenger Cars", "Light Commercial Vehicle"],
        ["Passenger Cars", "Heavy Commercial Vehicle"],
        ["Light Commercial Vehicle", "Heavy Commercial Vehicle"],
        ["Passenger Cars", "Light Commercial Vehicle", "Heavy Commercial Vehicle"],
    ]
    period_modes = get_period_modes()
    top_n_values = ["3", "5", "10", "ALL"]

    for status_value in status_values:
        for variable in ["BRAND_UPDATE", "OEM_UPDATE", "HIGHEST_BEV", "CO2_BUCKET"]:
            portfolio_ready, portfolio_var = fmd.prepare_portfolio_concentration_source(df, variable)
            bev_only = variable == "HIGHEST_BEV"
            for port_reg in port_reg_values:
                for mkt_reg in mkt_reg_values:
                    m_subset = MARKET_DF.copy()
                    if mkt_reg and mkt_reg != ["ALL"] and "Registration Type" in m_subset.columns:
                        m_subset = m_subset[m_subset["Registration Type"].isin(mkt_reg)]
                    market_ready, market_var = fmd.prepare_market_concentration_source(m_subset, variable)
                    for period_mode in period_modes:
                        for top_n in top_n_values:
                            kpi13 = fmd.kpi_top_brand_vs_market(
                                df_portfolio=portfolio_ready,
                                df_market=market_ready,
                                var_col_portfolio=portfolio_var,
                                var_col_market=market_var,
                                asset_status=status_value,
                                reg_type_portfolio=port_reg,
                                reg_type_market=mkt_reg,
                                bev_only=bev_only,
                                top_n=top_n,
                                period_mode=period_mode,
                            )
                            if kpi13.empty:
                                continue
                            kpi13 = kpi13.copy()
                            kpi13["COUNTRY_FILTER"] = country_code
                            kpi13["STATUS_FILTER"] = status_value
                            kpi13["PORT_REG_FILTER"] = serialize_value(port_reg)
                            kpi13["MKT_REG_FILTER"] = serialize_value(mkt_reg)
                            kpi13["VARIABLE_FILTER"] = variable
                            kpi13["PERIOD_MODE_FILTER"] = period_mode
                            kpi13["TOP_N_FILTER"] = top_n
                            kpi13["OWNER_FILTER"] = "ALL"
                            rows.append(kpi13)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Main loop: one country at a time ──────────────────────────────────────────

def run(countries: list[str]) -> None:
    total_start = time.time()
    view3_kpi91_frames: list[pd.DataFrame] = []
    view3_kpi92_frames: list[pd.DataFrame] = []
    view4_frames: list[pd.DataFrame] = []
    view5_portfolio_frames: list[pd.DataFrame] = []
    view6_frames: list[pd.DataFrame] = []

    for country in countries:
        print(f"\n=== {country} ===")
        t0   = time.time()
        _raw = fmd.load_country_monthly_data(
            folder_path=DATA_FOLDER,
            countries=[country],
            start_yyyymm=START_YYYYMM,
            end_yyyymm=END_YYYYMM,
            cols=fmd.COLUMNS_TO_READ,
        )
        df = fmd.prepare_data_set(_raw)
        del _raw
        print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

        if CONFIG["GENERATE_VIEW1"]:
            print("  View 1 generating...")
            generate_kpis_for_country(df)
            print("  View 1 KPI 7 generating...")
            generate_kpi7_for_country(df)
        if CONFIG["GENERATE_VIEW2"]:
            print("  View 2 generating...")
            generate_kpi8_for_country(df)
        if CONFIG["GENERATE_VIEW3"]:
            print("  View 3 generating...")
            v3_91, v3_92 = generate_view3_for_country(df)
            if not v3_91.empty:
                view3_kpi91_frames.append(v3_91)
            if not v3_92.empty:
                view3_kpi92_frames.append(v3_92)
        if CONFIG["GENERATE_VIEW4"]:
            print("  View 4 generating...")
            v4 = generate_view4_for_country(df)
            if not v4.empty:
                view4_frames.append(v4)
        if CONFIG["GENERATE_VIEW5"]:
            print("  View 5 generating...")
            v5_portfolio = generate_view5_portfolio_for_country(df)
            if not v5_portfolio.empty:
                view5_portfolio_frames.append(v5_portfolio)
        if CONFIG["GENERATE_VIEW6"]:
            print("  View 6 generating...")
            v6 = generate_view6_for_country(df)
            if not v6.empty:
                view6_frames.append(v6)

        del df
        gc.collect()
        print(f"  Done in {time.time()-t0:.1f}s")

    if CONFIG["GENERATE_VIEW5"]:
        market_rows = generate_view5_market(MARKET_DF)
        if not market_rows.empty:
            view5_portfolio_frames.append(market_rows)

    if view3_kpi91_frames:
        save_parquet(pd.concat(view3_kpi91_frames, ignore_index=True), os.path.join(BASE_OUT, "view3", "kpi9_1.parquet"))
    if view3_kpi92_frames:
        save_parquet(pd.concat(view3_kpi92_frames, ignore_index=True), os.path.join(BASE_OUT, "view3", "kpi9_2.parquet"))
    if view4_frames:
        save_parquet(pd.concat(view4_frames, ignore_index=True), os.path.join(BASE_OUT, "view4", "kpi10.parquet"))
    if view5_portfolio_frames:
        save_parquet(pd.concat(view5_portfolio_frames, ignore_index=True), os.path.join(BASE_OUT, "view5", "kpi11.parquet"))
    if view6_frames:
        save_parquet(pd.concat(view6_frames, ignore_index=True), os.path.join(BASE_OUT, "view6", "kpi13.parquet"))

    print(f"\nAll done in {time.time()-total_start:.1f}s")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arval Fleet - Offline Pre-computation Engine")
    parser.add_argument(
        "--countries", nargs="+", metavar="COUNTRY",
        help="Countries to generate. Use ALL for every country.",
    )
    args = parser.parse_args()

    if args.countries:
        COUNTRIES_TO_RUN = ALL_COUNTRIES if args.countries == ["ALL"] else args.countries
    else:
        COUNTRIES_TO_RUN = COUNTRIES_TO_RUN   # default from top of file

    run(COUNTRIES_TO_RUN)
