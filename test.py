
"""
Offline Pre-computation Engine
================================
Run this script once to generate all Parquet result files.
Then run fast_dashboard_reader.py to display results instantly.

Output structure:
  precomputed_fast/
    data/
      view1/
        kpis/
          year=YYYY/
            country=XXX/kpis.parquet   <- KPI 1-6 results
        kpi7/
          year=YYYY/kpi7.parquet       <- fuel type evolution
      view2/
        kpi8/
          year=YYYY/kpi8.parquet       <- production by fuel
"""

import os
import sys
import itertools
import time
import pandas as pd
from pathlib import Path

# ── Data loading configuration ────────────────────────────────────────────────
# Edit these to control which raw parquet files are read.

PARENT_DIR = Path(__file__).resolve().parent.parent   # …/Arval/New/
DATA_FOLDER = PARENT_DIR / "data"                     # folder with monthly parquets

COUNTRIES_TO_READ = ["SPAIN"]   # full country name as it appears in filenames
START_YYYYMM      = "202301"
END_YYYYMM        = "202512"

# ── Generation configuration ──────────────────────────────────────────────────
CONFIG = {
    "generate_kpis_1_to_6": True,
    "generate_kpi7": True,
    "generate_kpi8": True,
    "USE_UNIQUE_KEY_LOGIC": True,
    "UNIQUE_KEY_COLS": ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"],
    "TEST_MODE": False,
    "TEST_YEARS": [],
    "TEST_PERIOD_MODES": [],
    # Derived from COUNTRIES_TO_READ — restricts which country codes appear in outputs
    # (leave [] to include all countries present after loading)
    "COUNTRIES_FILTER": [],
}

# ── Import fmd without triggering its heavy data load ─────────────────────────
os.environ["FAST_READER_MODE"] = "1"   # fmd will skip load_dataset() on import
sys.path.insert(0, str(PARENT_DIR))
import fleet_monitoring_dashboard_key_refactor as fmd

# ── Load and inject dataset using our own config ──────────────────────────────
fmd.COUNTRIES_TO_READ = COUNTRIES_TO_READ
fmd.START_YYYYMM      = START_YYYYMM
fmd.END_YYYYMM        = END_YYYYMM

print("Loading dataset...")
_raw = fmd.load_country_monthly_data(
    folder_path=DATA_FOLDER,
    countries=COUNTRIES_TO_READ,
    start_yyyymm=START_YYYYMM,
    end_yyyymm=END_YYYYMM,
    cols=fmd.COLUMNS_TO_READ,
)
fmd.df = fmd.prepare_data_set(_raw)
print(f"  {len(fmd.df):,} rows loaded from {DATA_FOLDER}")

fmd.USE_UNIQUE_KEY_LOGIC = CONFIG["USE_UNIQUE_KEY_LOGIC"]
fmd.UNIQUE_KEY_COLS      = CONFIG["UNIQUE_KEY_COLS"]

BASE_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.columns = df.columns.astype(str)
    df.to_parquet(path, index=False)


def get_countries(df: pd.DataFrame) -> list:
    return ["ALL"] + sorted(str(x) for x in df["COUNTRY"].dropna().unique())


def progress(i: int, total: int, start: float, label: str = "") -> None:
    if i > 0 and i % 200 == 0:
        elapsed = time.time() - start
        rate = i / elapsed
        rem = (total - i) / rate
        print(f"  [{label}] {i}/{total} — est. {rem:.0f}s remaining")


# ── View 1 – KPI 1-6 ─────────────────────────────────────────────────────────
def generate_view1_kpis() -> None:
    df = fmd.df
    print(f"\n=== View 1 – KPIs 1–6  ({len(df)} source rows) ===")

    countries = get_countries(df)
    years     = CONFIG["TEST_YEARS"] if CONFIG["TEST_MODE"] else sorted(int(x) for x in df["YEAR"].dropna().unique())
    months    = ["ALL", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bocs      = ["CAR", "BIKE", "ALL"]
    statuses  = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    dmodes    = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]

    print(f"  Countries: {len(countries)} | Years: {len(years)} | Months: {len(months)}"
          f" | Bike/Car: {len(bocs)} | Status: {len(statuses)} | DateMode: {len(dmodes)}")

    total_per_year = len(countries) * len(months) * len(bocs) * len(statuses) * len(dmodes)
    print(f"  -> {total_per_year} combinations x {len(years)} years "
          f"= {total_per_year * len(years):,} total records")

    for year in years:
        print(f"\n--- Year {year} ---")
        df_year = df[df["YEAR"] == year].copy()
        combos  = list(itertools.product(countries, months, bocs, statuses, dmodes))
        records = []
        t0 = time.time()

        for i, (country, month, boc, status, dmode) in enumerate(combos):
            progress(i, total_per_year, t0, str(year))

            resolved = fmd.resolve_month_value(country, year, month)

            kpi1        = fmd.kpi_lease_under_25(df_year, country, year, resolved, boc, dmode, status)
            kpi2        = fmd.kpi_lease_25_30(df_year, country, year, resolved, boc, dmode, status)
            diesel_non  = fmd.kpi_diesel_non_diesel(df_year, country, year, resolved, boc, dmode, status)
            hybrid      = fmd.kpi_hybrid_share(df_year, country, year, resolved, boc, dmode, status)
            ev          = fmd.kpi_ev_share(df_year, country, year, resolved, boc, dmode, status)
            pv_lcv      = fmd.kpi_pv_lcv(df_year, country, year, resolved, boc, dmode, status)
            volume      = fmd.kpi_selected_volume(df_year, country, year, resolved, status, boc, dmode)

            records.append({
                "country":      country,
                "year":         str(year),
                "month":        str(month),
                "bike_or_car":  boc,
                "status":       status,
                "date_mode":    dmode,
                "kpi1":         kpi1,
                "kpi2":         kpi2,
                "diesel":       diesel_non[0] if diesel_non else None,
                "non_diesel":   diesel_non[1] if diesel_non else None,
                "hybrid":       hybrid,
                "ev":           ev,
                "pv":           pv_lcv[0] if pv_lcv else None,
                "lcv":          pv_lcv[1] if pv_lcv else None,
                "volume":       volume,
            })

        year_df = pd.DataFrame(records)

        # Save one file per (year, country) for fast indexed lookups
        for country in year_df["country"].unique():
            subset = year_df[year_df["country"] == country].copy()
            safe_country = str(country).replace(" ", "_")
            path = os.path.join(
                BASE_OUT, "view1", "kpis",
                f"year={year}", f"country={safe_country}", "kpis.parquet"
            )
            save_parquet(subset, path)

        # Also save a full-year file for convenience
        path_full = os.path.join(BASE_OUT, "view1", "kpis", f"year={year}", "kpis_all.parquet")
        save_parquet(year_df, path_full)
        print(f"  Saved {len(year_df):,} records for year {year}")


# ── View 1 – KPI 7 ───────────────────────────────────────────────────────────
def generate_view1_kpi7() -> None:
    df = fmd.df
    print("\n=== View 1 – KPI 7 (Fuel Evolution) ===")

    countries    = ["ALL"] + sorted(str(x) for x in df["COUNTRY"].dropna().unique())
    years        = CONFIG["TEST_YEARS"] if CONFIG["TEST_MODE"] else sorted(int(x) for x in df["YEAR"].dropna().unique())
    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    metric_modes = ["share", "volume"]
    period_modes = CONFIG["TEST_PERIOD_MODES"] if CONFIG["TEST_MODE"] else ["yearly", "quarterly", "monthly"]
    bocs         = ["CAR", "BIKE", "ALL"]

    combos = list(itertools.product(countries, statuses, metric_modes, period_modes, bocs))
    print(f"  {len(combos)} combinations × {len(years)} years")

    for year in years:
        print(f"  Processing year {year}...")
        start_date = f"{year}-01-01"
        end_date   = f"{year}-12-31"
        df_year    = df[df["YEAR"] == year].copy()
        all_results = []

        for country, status, mm, pm, boc in combos:
            res_df, _, _, _ = fmd.kpi7_fuel_by_period(
                df_year, country, status, mm, pm,
                bike_or_car=boc, date_mode="COB_DATE",
                start_date=start_date, end_date=end_date,
            )
            if res_df is not None and not res_df.empty:
                res_df = res_df.copy()
                res_df.index.name = "FUEL_TYPE"
                if "FUEL_TYPE" not in res_df.columns:
                    res_df = res_df.reset_index()
                res_df["_country"]     = country
                res_df["_status"]      = status
                res_df["_metric_mode"] = mm
                res_df["_period_mode"] = pm
                res_df["_bike_or_car"] = boc
                all_results.append(res_df)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            path = os.path.join(BASE_OUT, "view1", "kpi7", f"year={year}", "kpi7.parquet")
            save_parquet(combined, path)
            print(f"    Saved {len(combined):,} rows")


# ── View 2 – KPI 8 ───────────────────────────────────────────────────────────
def generate_view2_kpi8() -> None:
    df = fmd.df
    print("\n=== View 2 – KPI 8 (Production by Fuel) ===")

    countries    = ["ALL"] + sorted(str(x) for x in df["COUNTRY"].dropna().unique())
    years        = CONFIG["TEST_YEARS"] if CONFIG["TEST_MODE"] else sorted(int(x) for x in df["YEAR"].dropna().unique())
    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    metric_modes = ["share", "volume"]
    bocs         = ["CAR", "BIKE", "ALL"]
    dmodes       = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]
    period_modes = CONFIG["TEST_PERIOD_MODES"] if CONFIG["TEST_MODE"] else ["monthly", "quarterly", "yearly"]

    combos = list(itertools.product(countries, statuses, metric_modes, bocs, dmodes, period_modes))
    print(f"  {len(combos)} combinations × {len(years)} years")

    for year in years:
        print(f"  Processing year {year}...")
        df_year = df[df["YEAR"] == year].copy()
        all_results = []

        for country, status, mm, boc, dmode, pm in combos:
            res_df, _, _, _ = fmd.kpi8_production_ytd(
                df_year, country, year,
                asset_status=status, metric_mode=mm,
                bike_or_car=boc, date_mode=dmode, period_mode=pm,
            )
            if res_df is not None and not res_df.empty:
                res_df = res_df.copy()
                # kpi8 returns a pivot with PERIOD column and fuel columns
                res_df["_country"]     = country
                res_df["_status"]      = status
                res_df["_metric_mode"] = mm
                res_df["_bike_or_car"] = boc
                res_df["_date_mode"]   = dmode
                res_df["_period_mode"] = pm
                all_results.append(res_df)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            path = os.path.join(BASE_OUT, "view2", "kpi8", f"year={year}", "kpi8.parquet")
            save_parquet(combined, path)
            print(f"    Saved {len(combined):,} rows")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_start = time.time()
    print("=" * 60)
    print("  ARVAL Fleet – Offline Pre-computation Engine")
    print("=" * 60)
    print(f"  Output folder: {BASE_OUT}")

    if CONFIG["generate_kpis_1_to_6"]:
        generate_view1_kpis()

    if CONFIG["generate_kpi7"]:
        generate_view1_kpi7()

    if CONFIG["generate_kpi8"]:
        generate_view2_kpi8()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Pre-computation complete in {elapsed:.1f}s")
    print(f"  Run fast_dashboard_reader.py to explore results instantly.")
    print("=" * 60)
