"""
fix_bike_or_car.py
==================
Repair precomputed parquets dynamically:

  1. Remove rows for BIKE_OR_CAR values absent from the raw data
     (e.g. the old hardcoded "BIKE" value).
  2. Add rows for BIKE_OR_CAR values that DO exist in the raw data but
     are missing from the parquets (e.g. "2 WHEELS", "3 WHEELS") —
     all filter combinations (month × status × dmode …) are computed.
  3. CAR and ALL rows are never touched.
  4. Rebuilds merged parquets at the end.

Usage:
    python precomputed_fast/fix_bike_or_car.py
"""

import gc
import itertools
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PARENT_DIR   = SCRIPT_DIR.parent
DATA_FOLDER  = PARENT_DIR / "data"
BASE_OUT     = SCRIPT_DIR / "data"

START_YYYYMM = "202301"
END_YYYYMM   = "202602"

ALL_COUNTRY_NAMES = [
    "BELGIUM", "FRANCE", "GERMANY", "ITALY",
    "LUXEMBOURG", "NETHERLANDS", "SPAIN", "UNITED KINGDOM",
]

os.environ["FAST_READER_MODE"] = "1"
sys.path.insert(0, str(PARENT_DIR))   # fmd
sys.path.insert(0, str(SCRIPT_DIR))   # merge_parquets

import fleet_monitoring_dashboard_key_refactor as fmd

fmd.USE_UNIQUE_KEY_LOGIC = True
fmd.UNIQUE_KEY_COLS      = ["ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.columns = df.columns.astype(str)
    df.to_parquet(path, index=False)


def _code_to_name_map() -> dict[str, str]:
    """Scan one raw file per country to build {country_code: country_name}."""
    mapping: dict[str, str] = {}
    for name in ALL_COUNTRY_NAMES:
        files = sorted(DATA_FOLDER.glob(f"*{name}*.parquet"))
        if not files:
            continue
        try:
            code = str(pd.read_parquet(files[0], columns=["COUNTRY"])["COUNTRY"].iloc[0])
            mapping[code] = name
        except Exception:
            pass
    return mapping


def _codes_in_parquets() -> set[str]:
    """Return country codes found in partition folder names (country=XX)."""
    codes: set[str] = set()
    for d in BASE_OUT.rglob("country=*"):
        if d.is_dir():
            raw_code = d.name.removeprefix("country=").replace("_", " ")
            codes.add(raw_code)
    return codes


def _year_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("year="):
            try:
                return int(part.removeprefix("year="))
            except ValueError:
                pass
    return None


# ── Per-dataset fixers ────────────────────────────────────────────────────────

def _fix_kpis(path: Path, df_raw: pd.DataFrame, actual_bocs: list[str]) -> None:
    """Remove wrong BOC rows, add missing BOC rows — KPI 1-6 partition."""
    df = pd.read_parquet(path)
    if "BIKE_OR_CAR" not in df.columns or df.empty:
        return

    # Step 1 – remove wrong values
    valid   = set(actual_bocs) | {"ALL"}
    bad     = ~df["BIKE_OR_CAR"].astype(str).isin(valid)
    n_del   = int(bad.sum())
    df      = df[~bad].reset_index(drop=True)

    # Step 2 – detect which non-CAR bocs are missing
    present = set(df["BIKE_OR_CAR"].astype(str).unique())
    to_add  = [b for b in actual_bocs if b != "CAR" and b not in present]

    if not to_add and n_del == 0:
        print(f"    [skip]  {path.parent.name}/{path.name}")
        return

    year = _year_from_path(path)
    if year is None:
        _save_parquet(df, path)
        return

    country_code = str(df["COUNTRY"].iloc[0])
    df_year      = df_raw[df_raw["YEAR"] == year].copy()

    months   = ["ALL", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    statuses = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    dmodes   = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]
    records: list[dict] = []

    for boc in to_add:
        for month, status, dmode in itertools.product(months, statuses, dmodes):
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

    if records:
        df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)

    _save_parquet(df, path)
    print(f"    [fixed] {path.parent.name}/{path.name}  -={n_del} +={len(records)} rows  bocs={to_add}")


def _fix_kpi7(path: Path, df_raw: pd.DataFrame, actual_bocs: list[str]) -> None:
    """Remove wrong BOC rows, add missing BOC rows — KPI 7 partition."""
    df = pd.read_parquet(path)
    if "BIKE_OR_CAR" not in df.columns or df.empty:
        return

    valid   = set(actual_bocs) | {"ALL"}
    bad     = ~df["BIKE_OR_CAR"].astype(str).isin(valid)
    n_del   = int(bad.sum())
    df      = df[~bad].reset_index(drop=True)

    present = set(df["BIKE_OR_CAR"].astype(str).unique())
    to_add  = [b for b in actual_bocs if b != "CAR" and b not in present]

    if not to_add and n_del == 0:
        print(f"    [skip]  {path.parent.name}/{path.name}")
        return

    year = _year_from_path(path)
    if year is None:
        _save_parquet(df, path)
        return

    country_code = str(df["COUNTRY"].iloc[0])
    df_year      = df_raw[df_raw["YEAR"] == year].copy()
    start_date   = f"{year}-01-01"
    end_date     = f"{year}-12-31"

    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    metric_modes = ["share", "volume"]
    period_modes = ["monthly", "quarterly", "yearly"]
    new_frames: list[pd.DataFrame] = []

    for boc in to_add:
        for status, mm, pm in itertools.product(statuses, metric_modes, period_modes):
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
            new_frames.append(res_df)

    if new_frames:
        df = pd.concat([df] + new_frames, ignore_index=True)

    _save_parquet(df, path)
    added = sum(len(f) for f in new_frames)
    print(f"    [fixed] {path.parent.name}/{path.name}  -={n_del} +={added} rows  bocs={to_add}")


def _fix_kpi8(path: Path, df_raw: pd.DataFrame, actual_bocs: list[str]) -> None:
    """Remove wrong BOC rows, add missing BOC rows — KPI 8 partition."""
    df = pd.read_parquet(path)
    if "BIKE_OR_CAR" not in df.columns or df.empty:
        return

    valid   = set(actual_bocs) | {"ALL"}
    bad     = ~df["BIKE_OR_CAR"].astype(str).isin(valid)
    n_del   = int(bad.sum())
    df      = df[~bad].reset_index(drop=True)

    present = set(df["BIKE_OR_CAR"].astype(str).unique())
    to_add  = [b for b in actual_bocs if b != "CAR" and b not in present]

    if not to_add and n_del == 0:
        print(f"    [skip]  {path.parent.name}/{path.name}")
        return

    year = _year_from_path(path)
    if year is None:
        _save_parquet(df, path)
        return

    country_code = str(df["COUNTRY"].iloc[0])
    df_year      = df_raw[df_raw["YEAR"] == year].copy()

    statuses     = ["IN FLEET", "ORDER", "DEHIRE", "ALL"]
    metric_modes = ["share", "volume"]
    dmodes       = ["NONE", "CONTRACT_START_DATE", "DELIVERY_DATE"]
    period_modes = ["monthly", "quarterly", "yearly"]
    new_frames: list[pd.DataFrame] = []

    for boc in to_add:
        for status, mm, dmode, pm in itertools.product(statuses, metric_modes, dmodes, period_modes):
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
            new_frames.append(res_df)

    if new_frames:
        df = pd.concat([df] + new_frames, ignore_index=True)

    _save_parquet(df, path)
    added = sum(len(f) for f in new_frames)
    print(f"    [fixed] {path.parent.name}/{path.name}  -={n_del} +={added} rows  bocs={to_add}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    code_to_name    = _code_to_name_map()
    codes_present   = _codes_in_parquets()

    if not codes_present:
        print("No partition parquets found under", BASE_OUT)
        return

    for code in sorted(codes_present):
        country_name = code_to_name.get(code)
        if not country_name:
            print(f"\n[warn] No raw files found for code '{code}' — skipping.")
            continue

        print(f"\n=== {country_name} ({code}) ===")

        _raw   = fmd.load_country_monthly_data(
            folder_path=DATA_FOLDER,
            countries=[country_name],
            start_yyyymm=START_YYYYMM,
            end_yyyymm=END_YYYYMM,
            cols=fmd.COLUMNS_TO_READ,
        )
        df_raw = fmd.prepare_data_set(_raw)
        del _raw

        actual_bocs = sorted(df_raw["BIKE_OR_CAR"].dropna().astype(str).unique().tolist())
        to_fix      = [b for b in actual_bocs if b != "CAR"]
        print(f"  Raw BIKE_OR_CAR values : {actual_bocs}")
        print(f"  Will add/verify        : {to_fix}")

        safe = code.replace(" ", "_")

        print("  KPI 1-6:")
        for p in sorted((BASE_OUT / "view1" / "kpis").rglob(f"country={safe}/kpis.parquet")):
            _fix_kpis(p, df_raw, actual_bocs)

        print("  KPI 7:")
        for p in sorted((BASE_OUT / "view1" / "kpi7").rglob(f"country={safe}/kpi7.parquet")):
            _fix_kpi7(p, df_raw, actual_bocs)

        print("  KPI 8:")
        for p in sorted((BASE_OUT / "view2" / "kpi8").rglob(f"country={safe}/kpi8.parquet")):
            _fix_kpi8(p, df_raw, actual_bocs)

        del df_raw
        gc.collect()

    # Rebuild merged parquets
    print("\nRebuilding merged parquets...")
    from merge_parquets import merge_all
    merge_all()


if __name__ == "__main__":
    run()
