
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_FOLDER  = BASE_DIR / "data"
MODELS_PATH  = BASE_DIR / "models.parquet"
OUT_DIR      = BASE_DIR / "precomputed_use_case_1"
OUT_FILE     = OUT_DIR / "use_case_1_data.parquet"

ALL_COUNTRIES = [
    "BELGIUM", "FRANCE", "GERMANY", "ITALY",
    "LUXEMBOURG", "NETHERLANDS", "SPAIN", "UNITED KINGDOM",
]

# Default country list — override with --countries CLI arg
COUNTRIES_TO_RUN = ["LUXEMBOURG"]

START_YYYYMM = "202512"
END_YYYYMM   = "202512"

# Same column list use_case_1_heatmap_dashboard_4 reads.
COLUMNS_TO_READ = [
    "COB_DATE", "ID_CONTRACT", "VEHICLE_ID", "ID_QUOTATION",
    "COUNTRY", "NOVA_ASSET_STATUS", "BIKE_OR_CAR",
    "CLASS_CATALOG", "BRAND_UPDATE", "VEHICLE_CLASS",
    "POWER_CATEGORY", "FUEL_TYPE2", "FUEL_TYPE",
    "FINAL_CONTRACT_DURATION", "VA_CO2_EMSS_REAL",
    "OEM_UPDATE",
    "GROUP_RATING", "COUNTERPARTY_RATING", "CLS_GROUP_RATING",
    "ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION", "SHARED_CLIENT_FLAG",
    "VEHICLE_PRICE_EUR", "EXPOSURE_AMOUNT_LTR", "EXPOSURE_AMOUNT_MTR",
    "OBLIGOR_IDENTIFIER",
    "PENDING_ORDERS", "ID_CUSTOMER",
]

_MODELS_ENRICH_COLS = [
    "BRAND_UPDATE", "MODEL",
    "MARKET_MODEL", "MARKET_BODY_GROUP",
    "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE",
]


# ── Data loading — same filename convention as the other dashboards ───────────
# Filename pattern:  <prefix>-<COUNTRY>-<YYYYMM>.parquet

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
    end_int   = int(end_yyyymm)
    countries_upper = {c.strip().upper() for c in countries}

    for fp in files:
        name  = os.path.basename(fp).replace(".parquet", "")
        parts = [p.strip() for p in name.split("-")]
        if len(parts) < 3:
            continue
        file_country = parts[1].upper()
        try:
            file_yyyymm = int(parts[2])
        except ValueError:
            continue
        if file_country in countries_upper and start_int <= file_yyyymm <= end_int:
            chunk = pd.read_parquet(fp, columns=cols)
            dfs.append(chunk)

    if not dfs:
        return pd.DataFrame(columns=cols or COLUMNS_TO_READ)
    return pd.concat(dfs, ignore_index=True)


# ── Models enrichment (Market Model / Market Body Type) — same join as UC1 ────

def _load_models_df() -> pd.DataFrame:
    path = str(MODELS_PATH)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(path).names
        cols_to_read = [c for c in _MODELS_ENRICH_COLS if c in schema_cols]
        mdf = pd.read_parquet(path, columns=cols_to_read)
        mdf["MODEL"]        = mdf["MODEL"].astype(str).str.strip()
        mdf["BRAND_UPDATE"] = mdf["BRAND_UPDATE"].astype(str).str.strip()
        return mdf.drop_duplicates(subset=["BRAND_UPDATE", "MODEL"])
    except Exception:
        return pd.DataFrame()


MODELS_DF = _load_models_df()


def enrich_with_models(df: pd.DataFrame) -> pd.DataFrame:
    enrich_cols = ["MARKET_MODEL", "MARKET_BODY_GROUP", "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE"]

    if "CLASS_CATALOG" not in df.columns or "BRAND_UPDATE" not in df.columns:
        return df

    out = df.copy()
    out["MODEL"]        = out["CLASS_CATALOG"].astype(str).str.split("/").str[0].str.strip()
    out["BRAND_UPDATE"] = out["BRAND_UPDATE"].astype(str).str.strip()

    if MODELS_DF.empty:
        for c in enrich_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    if all(c in out.columns and out[c].notna().any() for c in ["MARKET_MODEL", "MARKET_BODY_GROUP"]):
        return out

    out = out.drop(columns=[c for c in enrich_cols if c in out.columns], errors="ignore")
    merge_cols = [c for c in _MODELS_ENRICH_COLS if c in MODELS_DF.columns]
    out = out.merge(MODELS_DF[merge_cols], how="left", on=["BRAND_UPDATE", "MODEL"])
    return out


# ── Data preparation — identical to use_case_1_heatmap_dashboard_4 ────────────

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "COB_DATE" in df.columns:
        df["COB_DATE"] = pd.to_datetime(df["COB_DATE"], errors="coerce")
    df["YEAR"]  = df["COB_DATE"].dt.year
    df["MONTH"] = df["COB_DATE"].dt.month
    for col in ["COUNTRY", "NOVA_ASSET_STATUS", "BIKE_OR_CAR", "BRAND_UPDATE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "VA_CO2_EMSS_REAL" in df.columns:
        def _co2_bucket(v):
            try:
                n = int(float(v)); lo = (n // 10) * 10
                return f"[{lo}-{lo+9}]"
            except Exception:
                return "UNK"
        df["CO2_BUCKET"] = df["VA_CO2_EMSS_REAL"].apply(_co2_bucket)
    if "EXPOSURE_AMOUNT_LTR" in df.columns and "EXPOSURE_AMOUNT_MTR" in df.columns:
        df["EXPOSURE_AMOUNT_TOT"] = df["EXPOSURE_AMOUNT_LTR"].fillna(0) + df["EXPOSURE_AMOUNT_MTR"].fillna(0)
    return df


def prepare_uc1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[df["BIKE_OR_CAR"] == "CAR"].copy()
    d = enrich_with_models(d)
    return d


# ── Output ──────────────────────────────────────────────────────────────────

def save_parquet_merge(df: pd.DataFrame, path: Path) -> None:
    """Merge new rows with any existing precomputed file, replacing rows for the
    regenerated countries only, then save — same merge-by-country logic as
    precomputed_fast/generate_all_precomputed.py's save_parquet_merge."""
    os.makedirs(path.parent, exist_ok=True)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if "COUNTRY" in existing.columns and "COUNTRY" in df.columns:
                new_countries = set(df["COUNTRY"].dropna().unique())
                n_before = len(existing)
                existing = existing[~existing["COUNTRY"].isin(new_countries)]
                print(f"  merge: removed {n_before - len(existing):,} existing rows for "
                      f"{sorted(new_countries)}, appending {len(df):,} new rows")
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not read existing precomputed file at {path}: {e}. Overwriting.")
    df.columns = df.columns.astype(str)
    df.to_parquet(path, index=False)


def run(countries: list[str]) -> None:
    t0 = time.time()
    print(f"Generating Use Case 1 precomputed data for {countries}, "
          f"COB {START_YYYYMM}–{END_YYYYMM}...")

    raw = load_country_monthly_data(
        DATA_FOLDER, countries, START_YYYYMM, END_YYYYMM,
        cols=[c for c in COLUMNS_TO_READ if c],
    )
    print(f"  Loaded {len(raw):,} raw rows in {time.time() - t0:.1f}s")

    raw = prepare_dataset(raw)
    uc1_df = prepare_uc1(raw)
    print(f"  {len(uc1_df):,} CAR rows after prep + models enrichment")

    if uc1_df.empty:
        print("  Nothing to write — no matching rows found. Check DATA_FOLDER / country / date range.")
        return

    save_parquet_merge(uc1_df, OUT_FILE)
    print(f"Done in {time.time() - t0:.1f}s — wrote {OUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Case 1 - Offline Pre-computation")
    parser.add_argument(
        "--countries", nargs="+", metavar="COUNTRY",
        help="Countries to generate. Use ALL for every country.",
    )
    args = parser.parse_args()

    if args.countries:
        COUNTRIES_TO_RUN = ALL_COUNTRIES if args.countries == ["ALL"] else args.countries

    run(COUNTRIES_TO_RUN)
