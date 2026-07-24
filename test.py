
"""
Fixes KPI 6 (PV / LCV split) rows that came out as 0/0 in the precomputed View 1
output. Root cause: fleet_monitoring_dashboard_key_refactor.kpi_pv_lcv() compares
CLS_VEHICLE_TYPE against the literal strings "PV" / "LCV" / "LV" with no
normalisation — countries whose raw CLS_VEHICLE_TYPE holds fleet-classification
codes instead of text labels silently match neither, giving PV=0, LCV=0 instead
of the real share (a genuinely empty subset already returns None, None, so this
is a distinct failure mode).

This script does NOT recompute the other 5 KPIs (that's what generate_all_precomputed.py
is for) — it reads the already-generated View 1 KPI table, finds only the broken
PV/LCV rows, reloads raw NOVA data for just the (country, year) pairs those rows
belong to, recomputes PV/LCV for those exact rows with CLS_VEHICLE_TYPE normalised,
and writes one new consolidated Parquet file with everything else untouched.

Usage:  python regenerate_view1_pv_lcv.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
FAST_DIR    = Path(__file__).resolve().parent
PARENT_DIR  = FAST_DIR.parent
DATA_FOLDER = PARENT_DIR / "data"
BASE_DATA   = FAST_DIR / "data"

SOURCE_MERGED_PATH = BASE_DATA / "merged_v1.parquet"
SOURCE_TREE_ROOT    = BASE_DATA / "view1" / "kpis"
OUTPUT_PATH          = BASE_DATA / "merged_v1_pv_lcv_fixed.parquet"

START_YYYYMM = "202301"
END_YYYYMM   = "202602"

# ── Import fmd without triggering its own eager data load ─────────────────────
os.environ["FAST_READER_MODE"] = "1"
sys.path.insert(0, str(PARENT_DIR))
import fleet_monitoring_dashboard_key_refactor as fmd

fmd.USE_UNIQUE_KEY_LOGIC = True
fmd.UNIQUE_KEY_COLS      = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]

# Short country code (as stored in the KPI table / raw COUNTRY column) -> the
# full name used in NOVA parquet filenames — same mapping generate_all_precomputed.py
# already uses for View 6.
COUNTRY_CODE_TO_NAME = {
    "ES": "Spain", "IT": "Italy", "UK": "United Kingdom",
    "DE": "Germany", "FR": "France", "BE": "Belgium",
    "NL": "Netherlands", "LU": "Luxembourg",
    "SPAIN": "Spain", "ITALY": "Italy",
    "GERMANY": "Germany", "FRANCE": "France",
    "BELGIUM": "Belgium", "NETHERLANDS": "Netherlands",
    "LUXEMBOURG": "Luxembourg", "UNITED KINGDOM": "United Kingdom",
    "GB": "United Kingdom", "GREAT BRITAIN": "United Kingdom",
}

# ── CLS_VEHICLE_TYPE normalisation ─────────────────────────────────────────────
# kpi_pv_lcv() only recognises the literal strings "PV" and "LCV"/"LV". Only code
# 1 and code 4 / "VU" map onto those; every other code (station wagon, vans,
# medium/heavy CV, minibus, blank) correctly counts as neither, exactly like today.
CLS_VEHICLE_TYPE_TEXT_MAP = {
    "PV": "PV", "LCV": "LCV", "LV": "LV", "VU": "LCV",
}
CLS_VEHICLE_TYPE_CODE_MAP = {
    "1": "PV",
    "2": "STATION WAGON",
    "3": "VANS",
    "4": "LCV",
    "5": "MEDIUM-DUTY CV",
    "6": "HCV",
    "7": "MINIBUS / PEOPLE MOVER",
    "": "CV WITH TRUCK REGISTRATION",
}


def normalize_cls_vehicle_type(value) -> str:
    text = str(value).strip()
    upper = text.upper()
    if upper in CLS_VEHICLE_TYPE_TEXT_MAP:
        return CLS_VEHICLE_TYPE_TEXT_MAP[upper]
    try:
        code = str(int(float(text)))
    except (TypeError, ValueError):
        code = upper
    return CLS_VEHICLE_TYPE_CODE_MAP.get(code, "NOT IDENTIFIED")


# ── Load the existing View 1 KPI output ────────────────────────────────────────

def load_existing_view1() -> pd.DataFrame:
    if SOURCE_MERGED_PATH.exists():
        return pd.read_parquet(SOURCE_MERGED_PATH)
    dfs = []
    if SOURCE_TREE_ROOT.exists():
        for p in SOURCE_TREE_ROOT.rglob("*.parquet"):
            dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    t0 = time.time()
    df = load_existing_view1()
    if df.empty:
        print("No existing View 1 KPI data found (checked "
              f"{SOURCE_MERGED_PATH} and {SOURCE_TREE_ROOT}) — nothing to fix.")
        return

    print(f"Loaded {len(df):,} existing View 1 KPI rows, "
          f"countries={sorted(df['COUNTRY'].dropna().unique())}")

    broken_mask = (df["PV"] == 0) & (df["LCV"] == 0)
    if not broken_mask.any():
        print("No 0/0 PV/LCV rows found — writing an unmodified copy so the "
              "pipeline still produces a fresh output file.")
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Wrote {OUTPUT_PATH} in {time.time() - t0:.1f}s")
        return

    affected = df.loc[broken_mask, ["COUNTRY", "YEAR"]].drop_duplicates()
    print(f"Found {int(broken_mask.sum()):,} broken PV/LCV rows across "
          f"{len(affected)} (country, year) combination(s):")
    print(affected.to_string(index=False))

    df = df.copy()
    for country_code, year in affected.itertuples(index=False):
        country_name = COUNTRY_CODE_TO_NAME.get(str(country_code).upper(), country_code)
        raw = fmd.load_country_monthly_data(
            folder_path=str(DATA_FOLDER),
            countries=[country_name],
            start_yyyymm=START_YYYYMM,
            end_yyyymm=END_YYYYMM,
            cols=fmd.COLUMNS_TO_READ,
        )
        if raw.empty:
            print(f"  [{country_code} {year}] no raw NOVA data found for "
                  f"'{country_name}' — leaving these rows as-is")
            continue

        prepped = fmd.prepare_data_set(raw)
        prepped = prepped[prepped["YEAR"] == int(year)].copy()
        if prepped.empty:
            print(f"  [{country_code} {year}] no rows for that year — leaving as-is")
            continue
        if "CLS_VEHICLE_TYPE" in prepped.columns:
            prepped["CLS_VEHICLE_TYPE"] = prepped["CLS_VEHICLE_TYPE"].map(normalize_cls_vehicle_type)

        rows_to_fix = df[broken_mask & (df["COUNTRY"] == country_code) & (df["YEAR"] == year)]
        for idx, row in rows_to_fix.iterrows():
            resolved_month = fmd.resolve_month_value(country_code, year, row["MONTH"])
            pv, lcv = fmd.kpi_pv_lcv(
                prepped, country_code, year, resolved_month,
                bike_or_car=row["BIKE_OR_CAR"],
                date_mode=row["DATE_MODE"],
                status=row["ASSET_STATUS"],
            )
            df.loc[idx, "PV"]  = pv
            df.loc[idx, "LCV"] = lcv

        still_zero = ((df.loc[rows_to_fix.index, "PV"] == 0) & (df.loc[rows_to_fix.index, "LCV"] == 0)).sum()
        print(f"  [{country_code} {year}] recomputed {len(rows_to_fix):,} rows "
              f"({still_zero} still 0/0 — genuinely empty subsets)")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH} in {time.time() - t0:.1f}s")
    print("Update fast_dashboard_reader.py's CACHE_V1 source path to this file to use it.")


if __name__ == "__main__":
    main()
