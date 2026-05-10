"""
Offline Pre-computation Engine
================================
Run this script once (or per country) to generate all Parquet result files.
Then run fast_dashboard_reader.py to display results instantly.

Usage:
  python generate_all_precomputed.py                        # default: SPAIN only
  python generate_all_precomputed.py --countries FRANCE GERMANY
  python generate_all_precomputed.py --countries ALL        # all countries

Memory strategy: one country loaded at a time, freed before the next.

Output structure:
  precomputed_fast/data/
    view1/kpis/year=YYYY/country=XXX/kpis.parquet
    view1/kpi7/year=YYYY/country=XXX/kpi7.parquet
    view2/kpi8/year=YYYY/country=XXX/kpi8.parquet
"""

import os
import sys
import gc
import itertools
import time
import argparse
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
PARENT_DIR = Path(__file__).resolve().parent.parent   # …/Arval/New/
DATA_FOLDER = PARENT_DIR / "data"
BASE_OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ALL_COUNTRIES = [
    "BELGIUM", "FRANCE", "GERMANY", "ITALY",
    "LUXEMBOURG", "NETHERLANDS", "SPAIN", "UNITED KINGDOM",
]

# Default country list — override with --countries CLI arg
COUNTRIES_TO_RUN = ["SPAIN", "ITALY"]

START_YYYYMM = "202301"
END_YYYYMM   = "202602"

CONFIG = {
    "generate_kpis_1_to_6": True,
    "generate_kpi7":        True,
    "generate_kpi8":        True,
    "USE_UNIQUE_KEY_LOGIC": True,
    "UNIQUE_KEY_COLS":      ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"],
    "TEST_MODE":            False,
    "TEST_YEARS":           [],
    "TEST_PERIOD_MODES":    [],
}

# ── Import fmd (no data loaded yet) ───────────────────────────────────────────
os.environ["FAST_READER_MODE"] = "1"
sys.path.insert(0, str(PARENT_DIR))
import fleet_monitoring_dashboard_key_refactor as fmd

fmd.USE_UNIQUE_KEY_LOGIC = CONFIG["USE_UNIQUE_KEY_LOGIC"]
fmd.UNIQUE_KEY_COLS      = CONFIG["UNIQUE_KEY_COLS"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.columns = df.columns.astype(str)
    df.to_parquet(path, index=False)


def get_years(df: pd.DataFrame) -> list[int]:
    if CONFIG["TEST_MODE"] and CONFIG["TEST_YEARS"]:
        return CONFIG["TEST_YEARS"]
    return sorted(int(x) for x in df["YEAR"].dropna().unique())


def get_period_modes() -> list[str]:
    if CONFIG["TEST_MODE"] and CONFIG["TEST_PERIOD_MODES"]:
        return CONFIG["TEST_PERIOD_MODES"]
    return ["monthly", "quarterly", "yearly"]



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


# ── Main loop: one country at a time ──────────────────────────────────────────

def run(countries: list[str]) -> None:
    total_start = time.time()
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

        if CONFIG["generate_kpis_1_to_6"]:
            generate_kpis_for_country(df)
        if CONFIG["generate_kpi7"]:
            generate_kpi7_for_country(df)
        if CONFIG["generate_kpi8"]:
            generate_kpi8_for_country(df)

        del df
        gc.collect()
        print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nAll done in {time.time()-total_start:.1f}s -> run merge_parquets.py")


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
