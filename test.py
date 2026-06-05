
import os
import sys
import gc
import itertools
import time
import argparse
import re
import pandas as pd
from pathlib import Path
from collections.abc import Sequence

# ── Configuration ─────────────────────────────────────────────────────────────
PARENT_DIR       = Path(__file__).resolve().parent.parent          # …/Arval/New/
DATA_FOLDER      = PARENT_DIR / "data"                             # raw NOVA parquets
BASE_OUT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # precomputed output
MARKET_DATA_PATH = PARENT_DIR / "market_dataset_europe.parquet"   # used by View 5 & 6
MODELS_DATA_PATH = PARENT_DIR / "models.parquet"                   # model enrichment for View 3 & 4

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

_MODELS_COLS = ["BRAND_UPDATE", "MODEL", "MARKET_MODEL", "MARKET_BODY_GROUP", "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE"]
def _load_models() -> pd.DataFrame:
    path = str(MODELS_DATA_PATH)
    if not os.path.exists(path):
        print(f"Warning: models.parquet not found at {path}. Views 3 & 4 will have no model enrichment.")
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(path).names
        cols_to_read = [c for c in _MODELS_COLS if c in schema_cols]
        df = pd.read_parquet(path, columns=cols_to_read)
        df["MODEL"] = df["MODEL"].astype(str).str.strip()
        df["BRAND_UPDATE"] = df["BRAND_UPDATE"].astype(str).str.strip()
        return df.drop_duplicates(subset=["BRAND_UPDATE", "MODEL"])
    except Exception as e:
        print(f"Warning: could not load models.parquet: {e}")
        return pd.DataFrame()

MODELS_DF = _load_models()
print(f"Models loaded: {len(MODELS_DF):,} rows" if not MODELS_DF.empty else "Models: not available")


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.columns = df.columns.astype(str)
    df.to_parquet(path, index=False)


def save_parquet_merge(df: pd.DataFrame, path: str) -> None:
    """Merge new rows with existing parquet, replacing rows for regenerated countries, then save."""
    if os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            if "COUNTRY" in existing.columns and "COUNTRY" in df.columns:
                new_countries = set(df["COUNTRY"].dropna().unique())
                existing = existing[~existing["COUNTRY"].isin(new_countries)]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as e:
            print(f"Warning: could not read existing parquet at {path}: {e}. Overwriting.")
    save_parquet(df, path)


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


_ENRICH_TARGET = ["MARKET_MODEL", "MARKET_BODY_GROUP", "CDN_CLF_SEGMENT", "CDN_CLF_BODY_TYPE"]
# Minimal columns needed to build models.parquet from nova
_NOVA_MODELS_COLS = ["BRAND_UPDATE", "CLASS_CATALOG"] + _ENRICH_TARGET


def _ensure_models_parquet(countries: list[str]) -> None:
    """
    If models.parquet doesn't exist, do a lightweight pre-pass over nova to build it
    from enrichment columns already present in the data (MARKET_MODEL, MARKET_BODY_GROUP, …).
    Updates MODELS_DF globally when successful.
    """
    global MODELS_DF
    models_path = str(MODELS_DATA_PATH)
    if os.path.exists(models_path):
        return

    print("models.parquet not found — scanning nova data to build it…")
    frames: list[pd.DataFrame] = []
    found_enrich = False

    for country in countries:
        try:
            # Try reading only the needed columns; fall back to all cols if the
            # parquet schema doesn't include some enrichment columns.
            try:
                raw = fmd.load_country_monthly_data(
                    folder_path=DATA_FOLDER,
                    countries=[country],
                    start_yyyymm=START_YYYYMM,
                    end_yyyymm=END_YYYYMM,
                    cols=_NOVA_MODELS_COLS,
                )
            except Exception:
                raw = fmd.load_country_monthly_data(
                    folder_path=DATA_FOLDER,
                    countries=[country],
                    start_yyyymm=START_YYYYMM,
                    end_yyyymm=END_YYYYMM,
                    cols=fmd.COLUMNS_TO_READ,
                )
        except Exception as e:
            print(f"  [{country}] load error: {e}")
            continue

        if raw is None or raw.empty:
            continue
        if "CLASS_CATALOG" not in raw.columns or "BRAND_UPDATE" not in raw.columns:
            continue

        present = [c for c in _ENRICH_TARGET if c in raw.columns and raw[c].notna().any()]
        if not present:
            print(f"  [{country}] no enrichment columns in nova — cannot build models.parquet on this machine")
            return

        found_enrich = True
        raw = raw.copy()
        raw["MODEL"] = raw["CLASS_CATALOG"].astype(str).str.split("/").str[0].str.strip()
        raw["BRAND_UPDATE"] = raw["BRAND_UPDATE"].astype(str).str.strip()
        build_cols = ["BRAND_UPDATE", "MODEL"] + present
        frames.append(raw[build_cols].dropna(subset=["BRAND_UPDATE", "MODEL"]))
        del raw

    if not found_enrich or not frames:
        print("models.parquet: no enrichment data found in nova, skipping creation.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["MODEL"] = combined["MODEL"].astype(str).str.strip()
    combined["BRAND_UPDATE"] = combined["BRAND_UPDATE"].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["BRAND_UPDATE", "MODEL"])
    combined.to_parquet(models_path, index=False)
    print(f"Created models.parquet: {len(combined):,} unique (brand, model) rows")
    MODELS_DF = _load_models()


def enrich_with_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge NOVA data with models.parquet to obtain MARKET_MODEL, MARKET_BODY_GROUP, etc.
    MODEL is extracted from CLASS_CATALOG.  Only merges columns actually present in MODELS_DF.
    """
    if "CLASS_CATALOG" not in df.columns or "BRAND_UPDATE" not in df.columns:
        return df

    out = df.copy()
    out["MODEL"] = out["CLASS_CATALOG"].astype(str).str.split("/").str[0].str.strip()
    out["BRAND_UPDATE"] = out["BRAND_UPDATE"].astype(str).str.strip()

    if MODELS_DF.empty:
        return out

    if all(c in out.columns and out[c].notna().any() for c in ["MARKET_MODEL", "MARKET_BODY_GROUP"]):
        return out

    enrich_cols = [c for c in _ENRICH_TARGET if c in MODELS_DF.columns]
    out = out.drop(columns=[c for c in enrich_cols if c in out.columns], errors="ignore")
    merge_cols = ["BRAND_UPDATE", "MODEL"] + enrich_cols
    out = out.merge(MODELS_DF[merge_cols], how="left", on=["BRAND_UPDATE", "MODEL"])
    return out


def non_empty_subsets(values: Sequence[object]) -> list[list[object]]:
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


def _apply_status_mask_on_agg(agg: pd.DataFrame, status_value: str) -> pd.DataFrame:
    """Filter a pre-aggregated DataFrame (with NOVA_ASSET_STATUS column) by status.
    Mirrors apply_status_filter logic so results are identical to the live path."""
    if status_value == "ALL":
        return agg
    if "NOVA_ASSET_STATUS" not in agg.columns:
        return agg  # no status column → any status passes (mirrors apply_status_filter)
    s = agg["NOVA_ASSET_STATUS"]
    if (s == "_ALL_").all():  # sentinel used when NOVA_ASSET_STATUS absent in source
        return agg
    sv = str(status_value).strip().upper()
    if sv == "IN FLEET":
        return agg[s == "IN FLEET"]
    return agg[s.str.contains(sv, na=False, regex=False)]


def _kpi91_from_agg(agg: pd.DataFrame, body_val: str, status_value: str) -> pd.DataFrame:
    """Derive kpi9_1 (PCT by PERIOD) from pre-aggregated data without re-scanning rows."""
    filtered = _apply_status_mask_on_agg(agg, status_value)
    if filtered.empty:
        return pd.DataFrame(columns=["PERIOD", "PCT"])
    total = filtered.groupby("PERIOD")["COUNT"].sum()
    if body_val == "ALL":
        selected = total.copy()
    else:
        body_rows = filtered[filtered["BODY_GROUP"] == body_val]
        selected = body_rows.groupby("PERIOD")["COUNT"].sum()
    pct = (selected / total * 100).fillna(0).round(2)
    result = pct.reset_index(name="PCT")
    result["PERIOD"] = result["PERIOD"].astype(str)
    return result.sort_values("PERIOD")


def _kpi92_from_agg(agg: pd.DataFrame, body_val: str, status_value: str) -> pd.DataFrame:
    """Derive kpi9_2 pivot (PERIOD × POWER_CATEGORY) from pre-aggregated data."""
    filtered = _apply_status_mask_on_agg(agg, status_value)
    if body_val != "ALL":
        filtered = filtered[filtered["BODY_GROUP"] == body_val]
    if filtered.empty:
        return pd.DataFrame()
    grouped = filtered.groupby(["PERIOD", "POWER_CATEGORY"])["COUNT"].sum().reset_index()
    pivot = grouped.pivot(index="PERIOD", columns="POWER_CATEGORY", values="COUNT").fillna(0)
    return pivot.sort_index()


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
    """
    Optimised version: precompute one groupby per (year, period_mode, bike_or_car, vehicle_class)
    instead of calling kpi9_1 / kpi9_2 (~60 000 times) for every body × status combination.
    Matches view-6 pattern: aggregate once, derive all filter combos by slicing.
    """
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame(), pd.DataFrame()

    years = get_years(df)
    vehicle_types = get_vehicle_type_values(df)
    market_bodies = get_market_body_values(df)
    bike_or_cars = (
        ["ALL"] + sorted(df["BIKE_OR_CAR"].dropna().astype(str).str.upper().unique().tolist())
        if "BIKE_OR_CAR" in df.columns else ["ALL"]
    )
    period_modes = get_period_modes()
    status_values = [item["value"] for item in fmd.asset_status_options if item["value"] != "ALL"]
    status_subsets = non_empty_subsets(status_values)

    rows_91: list[pd.DataFrame] = []
    rows_92: list[pd.DataFrame] = []

    n_bodies   = len(market_bodies)
    n_vtypes   = len(vehicle_types)
    n_bocs     = len(bike_or_cars)
    n_modes    = len(period_modes)
    n_subsets  = len(status_subsets)
    n_statuses = len(status_values)
    print(f"    V3 {get_country_code_column(df)}: {len(years)} years × {n_modes} modes × {n_bocs} bocs × {n_vtypes} vtypes → {n_bodies} bodies × {n_subsets} subsets, {n_statuses} statuses")

    # Determine vehicle type column once — same for all data in this country
    vcol = fmd.vehicle_col_global if hasattr(fmd, "vehicle_col_global") else None

    for year in years:
        t_year = time.time()
        df_year = df[df["YEAR"] == year].copy()
        n91_before = len(rows_91)
        n92_before = len(rows_92)
        for period_mode in period_modes:
            for bike_or_car in bike_or_cars:
                # ── Filter by bike_or_car ONCE (not per vehicle_class) ────────
                local_df = df_year
                if bike_or_car != "ALL" and "BIKE_OR_CAR" in local_df.columns:
                    local_df = local_df[local_df["BIKE_OR_CAR"].astype(str).str.upper() == bike_or_car]

                if local_df.empty:
                    continue

                body_group_col = fmd.pick_first_existing_column(
                    local_df, ["MARKET_BODY_GROUP", "CLS_VEHICLE_TYPE", "VEHICLE_CLASS"]
                )
                if body_group_col is None:
                    continue

                # ── Build work DataFrame ONCE per (year, mode, bike) ─────────
                _needed = [c for c in ["COB_DATE", body_group_col, "VEHICLE_ID"] if c in local_df.columns]
                work = local_df.dropna(subset=_needed).copy()
                if work.empty:
                    continue

                # Inline period computation — bypasses build_period_column's internal copy
                _dates = pd.to_datetime(work["COB_DATE"], errors="coerce")
                if period_mode == "monthly":
                    work["PERIOD"] = _dates.dt.strftime("%Y-%m")
                elif period_mode == "yearly":
                    work["PERIOD"] = _dates.dt.year.astype("Int64").astype(str)
                else:
                    work["PERIOD"] = _dates.dt.to_period("Q").astype(str)

                work[body_group_col] = work[body_group_col].astype(str).str.strip().str.upper()
                if "NOVA_ASSET_STATUS" in work.columns:
                    work["NOVA_ASSET_STATUS"] = work["NOVA_ASSET_STATUS"].astype(str).str.strip().str.upper()
                else:
                    work["NOVA_ASSET_STATUS"] = "_ALL_"  # sentinel: no status column

                # Include vcol in groupby — one aggregation covers all vehicle_class slices
                has_vcol = vcol is not None and vcol in work.columns
                if has_vcol:
                    work["_vcol_upper"] = work[vcol].astype(str).str.upper()
                    gb91 = ["PERIOD", "_vcol_upper", body_group_col, "NOVA_ASSET_STATUS"]
                    gb92 = ["PERIOD", "_vcol_upper", body_group_col, "NOVA_ASSET_STATUS", "POWER_CATEGORY"]
                else:
                    gb91 = ["PERIOD", body_group_col, "NOVA_ASSET_STATUS"]
                    gb92 = ["PERIOD", body_group_col, "NOVA_ASSET_STATUS", "POWER_CATEGORY"]

                # ── Precompute kpi9_1 base aggregation ────────────────────────
                agg91 = (
                    work.groupby(gb91, observed=True)["VEHICLE_ID"]
                    .nunique()
                    .reset_index(name="COUNT")
                    .rename(columns={body_group_col: "BODY_GROUP"})
                )

                # ── Precompute kpi9_2 base aggregation ────────────────────────
                has_power = "POWER_CATEGORY" in work.columns
                if has_power:
                    work92 = work.dropna(subset=["POWER_CATEGORY"]).copy()
                    work92["POWER_CATEGORY"] = work92["POWER_CATEGORY"].astype(str).str.strip().str.upper()
                    agg92 = (
                        work92.groupby(gb92, observed=True)["VEHICLE_ID"]
                        .nunique()
                        .reset_index(name="COUNT")
                        .rename(columns={body_group_col: "BODY_GROUP"})
                    )
                else:
                    agg92 = pd.DataFrame()

                # ── Slice per vehicle_class, derive body × status combos ──────
                for vehicle_class in vehicle_types:
                    if has_vcol:
                        if vehicle_class == "ALL":
                            agg91_vc = agg91
                            agg92_vc = agg92
                        else:
                            agg91_vc = agg91[agg91["_vcol_upper"] == vehicle_class]
                            agg92_vc = agg92[agg92["_vcol_upper"] == vehicle_class] if not agg92.empty else agg92
                    else:
                        agg91_vc = agg91
                        agg92_vc = agg92

                    if agg91_vc.empty:
                        continue

                    for vehicle_body in market_bodies:
                        body_val = "ALL" if vehicle_body == "ALL" else str(vehicle_body).strip().upper()

                        for selected_statuses in status_subsets:
                            if not selected_statuses:
                                continue
                            status_list = ["ALL"] if selected_statuses == ["ALL"] else selected_statuses
                            for status_value in status_list:
                                sv = str(status_value)
                                result = _kpi91_from_agg(agg91_vc, body_val, sv)
                                if result.empty:
                                    continue
                                result["ASSET_STATUS"] = sv
                                result["YEAR_FILTER"] = year
                                result["COUNTRY_FILTER"] = country_code
                                result["VEHICLE_CLASS_FILTER"] = vehicle_class
                                result["VEHICLE_BODY_FILTER"] = vehicle_body
                                result["BIKE_OR_CAR_FILTER"] = bike_or_car
                                result["PERIOD_MODE_FILTER"] = period_mode
                                result["STATUS_SELECTION"] = serialize_value(status_list)
                                rows_91.append(result)

                        # ── Derive kpi9_2 for all status values via fast slicing ──
                        if not agg92_vc.empty:
                            for status_value in status_values:
                                result = _kpi92_from_agg(agg92_vc, body_val, status_value)
                                if result.empty:
                                    continue
                                result = result.reset_index().rename(columns={"PERIOD": "Period"})
                                result["YEAR_FILTER"] = year
                                result["COUNTRY_FILTER"] = country_code
                                result["VEHICLE_CLASS_FILTER"] = vehicle_class
                                result["VEHICLE_BODY_FILTER"] = vehicle_body
                                result["BIKE_OR_CAR_FILTER"] = bike_or_car
                                result["PERIOD_MODE_FILTER"] = period_mode
                                result["STATUS_FILTER"] = status_value
                                rows_92.append(result)

        n91_after = len(rows_91)
        n92_after = len(rows_92)
        print(f"    V3 {get_country_code_column(df)} year={year}: +{n91_after - n91_before} kpi91 frames, +{n92_after - n92_before} kpi92 frames in {time.time()-t_year:.1f}s")

    return (
        pd.concat(rows_91, ignore_index=True) if rows_91 else pd.DataFrame(),
        pd.concat(rows_92, ignore_index=True) if rows_92 else pd.DataFrame(),
    )


def generate_view4_for_country(df: pd.DataFrame) -> pd.DataFrame:
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame()

    # CONTRACT_FINAL_END = first non-null among (AMENDED → EXTENSION → END)
    _date_priority = ["CONTRACT_END_DATE_AMENDED", "EXTENSION_DATE", "CONTRACT_END_DATE"]
    _existing = [c for c in _date_priority if c in df.columns]
    if _existing:
        _candidates = df[_existing].apply(pd.to_datetime, errors="coerce")
        df = df.copy()
        df["CONTRACT_FINAL_END"] = _candidates.bfill(axis=1).iloc[:, 0]
    else:
        return pd.DataFrame()

    eoc_year_series = pd.to_datetime(df["CONTRACT_FINAL_END"], errors="coerce").dt.year.dropna().astype(int)
    _now_year = pd.Timestamp.now().year
    eoc_years_raw = sorted(eoc_year_series.unique().tolist())
    eoc_years = [y for y in eoc_years_raw if (_now_year - 8) <= y <= (_now_year + 8)]
    n_dropped = len(eoc_years_raw) - len(eoc_years)
    if not eoc_years:
        print(f"    V4 {country_code}: no valid EOC years — skipping")
        return pd.DataFrame()

    period_modes = get_period_modes()  # ["monthly", "quarterly", "yearly"]
    bike_or_cars = (
        ["ALL"] + sorted(df["BIKE_OR_CAR"].dropna().astype(str).str.upper().unique().tolist())
        if "BIKE_OR_CAR" in df.columns else ["ALL"]
    )
    statuses = [item["value"] for item in fmd.asset_status_options]

    # Pre-filter country + parse datetime ONCE
    out_base = df[df["COUNTRY"] == country_code].copy()
    out_base["CONTRACT_FINAL_END"] = pd.to_datetime(out_base["CONTRACT_FINAL_END"], errors="coerce")
    out_base["_eoc_year"] = out_base["CONTRACT_FINAL_END"].dt.year

    print(f"    V4 {country_code}: {len(eoc_years)} EOC years [{eoc_years[0]}–{eoc_years[-1]}] "
          f"× {len(period_modes)} modes × {len(statuses)} statuses × {len(bike_or_cars)} bocs"
          + (f" | {n_dropped} out-of-range years dropped" if n_dropped else ""))

    rows: list[pd.DataFrame] = []
    t_v4 = time.time()

    for year in eoc_years:
        out_year = out_base[out_base["_eoc_year"] == year]
        if out_year.empty or "MARKET_MODEL" not in out_year.columns:
            continue

        work = out_year.dropna(subset=["CONTRACT_FINAL_END", "MARKET_MODEL"])
        if work.empty:
            continue
        work = work.copy()
        work["MARKET_MODEL"] = work["MARKET_MODEL"].astype(str).str.strip()
        work["_tmp_key"] = fmd.get_unique_key_series(work)
        if "BIKE_OR_CAR" in work.columns:
            work["_boc_upper"] = work["BIKE_OR_CAR"].astype(str).str.upper()

        _ends = work["CONTRACT_FINAL_END"]

        for period_mode in period_modes:
            # Compute PERIOD from CONTRACT_FINAL_END once per (year, period_mode)
            if period_mode == "monthly":
                work["PERIOD"] = _ends.dt.strftime("%Y-%m")
            elif period_mode == "quarterly":
                work["PERIOD"] = _ends.dt.to_period("Q").astype(str)
            else:  # yearly
                work["PERIOD"] = _ends.dt.year.astype("Int64").astype(str)

            for bike_or_car in bike_or_cars:
                sub_boc = work
                if bike_or_car != "ALL" and "_boc_upper" in work.columns:
                    sub_boc = sub_boc[sub_boc["_boc_upper"] == bike_or_car]
                if sub_boc.empty:
                    continue

                for status in statuses:
                    sub = fmd.apply_status_filter(sub_boc, status)
                    if sub.empty:
                        continue

                    counts = (
                        sub.groupby(["PERIOD", "MARKET_MODEL"])["_tmp_key"]
                        .nunique()
                        .reset_index(name="COUNT")
                    )
                    if counts.empty:
                        continue
                    counts["EOC_YEAR"]            = year
                    counts["PERIOD_MODE_FILTER"]  = period_mode
                    counts["COUNTRY_FILTER"]      = country_code
                    counts["ASSET_STATUS_FILTER"] = status
                    counts["BIKE_OR_CAR_FILTER"]  = bike_or_car
                    rows.append(counts)

    print(f"    V4 {country_code} TOTAL: {time.time()-t_v4:.1f}s | {len(rows)} result frames")
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
                # Compute once with top_n="ALL", then slice per top_n (5× fewer heavy calls)
                if "YTD" in status_value.upper():
                    base = fmd.kpi_count_share_ytd_by_quarter(portfolio_ready, asset_status=status_value.replace(" YTD", ""), var_col=portfolio_var, bike_or_car="CAR", top_n="ALL", period_mode=period_mode)
                else:
                    base = fmd.kpi_count_share_quarterly(portfolio_ready, asset_status=status_value, var_col=portfolio_var, bike_or_car="CAR", top_n="ALL", period_mode=period_mode)
                if base.empty:
                    continue
                base = base.copy()
                if "HIGHEST_BEV" in base.columns:
                    base = base.rename(columns={"HIGHEST_BEV": "BEV"})
                grp_cols = [c for c in ["COUNTRY", "YEAR", "PERIOD"] if c in base.columns]
                for top_n in top_n_values:
                    if top_n == "ALL":
                        kpi11_table = base.copy()
                    else:
                        kpi11_table = base.groupby(grp_cols).head(int(top_n)).copy()
                    kpi11_table["SOURCE_FILTER"]     = "portfolio"
                    kpi11_table["COUNTRY_FILTER"]    = country_code
                    kpi11_table["STATUS_FILTER"]     = status_value
                    kpi11_table["VARIABLE_FILTER"]   = variable
                    kpi11_table["TOP_N_FILTER"]      = top_n
                    kpi11_table["PERIOD_MODE_FILTER"] = period_mode
                    rows.append(kpi11_table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _market_kpi_with_country(df: pd.DataFrame, var_col: str, top_n: str = "1", period_mode: str = "quarterly") -> pd.DataFrame:
    """Like kpi_count_share_quarterly_market but adds per-country grouping using 'Country/Territory-Number'."""
    COUNTRY_COL = "Country/Territory-Number"
    out = df.copy()
    if out.empty or COUNTRY_COL not in out.columns:
        return fmd.kpi_count_share_quarterly_market(df, var_col, top_n=top_n, period_mode=period_mode)
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

    grouped = out.groupby([COUNTRY_COL, var_col, "YEAR", "PERIOD"])["volume"].sum().reset_index()
    grouped = grouped.rename(columns={COUNTRY_COL: "COUNTRY", "volume": "VOLUME"})
    grouped = grouped.sort_values(["COUNTRY", "YEAR", "PERIOD", "VOLUME"], ascending=[True, True, True, False])

    grouped["TOTAL"] = grouped.groupby(["COUNTRY", "YEAR", "PERIOD"])["VOLUME"].transform("sum")
    grouped["SHARE"] = grouped["VOLUME"].div(grouped["TOTAL"].where(grouped["TOTAL"] != 0, 1)).mul(100)

    return fmd.apply_top_n_filter(grouped, top_n, ["COUNTRY", "YEAR", "PERIOD"])


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
    if not market_df.empty and "Ownertype" in market_df.columns:
        _owner_vals = sorted(market_df["Ownertype"].dropna().astype(str).unique().tolist())
        owners = ["ALL"] + [v for v in _owner_vals if v not in ("ALL", "nan", "")]
    else:
        owners = ["ALL"]
    period_modes = get_period_modes()
    top_n_values = ["1", "5", "10", "20", "ALL"]

    for variable in ["BRAND_UPDATE", "OEM_UPDATE", "CO2_BUCKET", "HIGHEST_BEV"]:
        market_ready, market_var = fmd.prepare_market_concentration_source(market_df, variable)
        for reg_value in reg_types:
            # Filter once per (reg_value) — no .copy() needed since we don't mutate
            m_reg = market_ready[market_ready["Registration Type"].isin(reg_value)] \
                if reg_value and "Registration Type" in market_ready.columns else market_ready
            for owner in owners:
                m_filtered = m_reg[m_reg["Ownertype"] == owner] \
                    if owner and owner != "ALL" and "Ownertype" in m_reg.columns else m_reg
                for period_mode in period_modes:
                    # Compute once with top_n="ALL", then slice per top_n (5× fewer heavy calls)
                    base = _market_kpi_with_country(m_filtered, market_var, top_n="ALL", period_mode=period_mode)
                    if base.empty:
                        continue
                    base = base.copy()
                    if "HIGHEST_BEV" in base.columns:
                        base = base.rename(columns={"HIGHEST_BEV": "BEV"})
                    grp_cols = [c for c in ["COUNTRY", "YEAR", "PERIOD"] if c in base.columns]
                    for top_n in top_n_values:
                        if top_n == "ALL":
                            kpi11_table = base.copy()
                        else:
                            kpi11_table = base.groupby(grp_cols).head(int(top_n)).copy()
                        kpi11_table["SOURCE_FILTER"]     = "market"
                        kpi11_table["REG_TYPE_FILTER"]   = serialize_value(reg_value)
                        kpi11_table["OWNER_FILTER"]      = owner
                        kpi11_table["VARIABLE_FILTER"]   = variable
                        kpi11_table["TOP_N_FILTER"]      = top_n
                        kpi11_table["PERIOD_MODE_FILTER"] = period_mode
                        rows.append(kpi11_table)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def generate_view6_for_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimised version: precompute market and portfolio aggregations once,
    then build rows via a simple merge + head() per combination.

    Old: 4,032 calls to kpi_top_brand_vs_market (each does 2 .copy(), groupby, merge)
         + 336 MARKET_DF.copy() calls
    New: 84 market aggregations + 144 portfolio aggregations + 1,008 simple merges
    """
    country_code = get_country_code_column(df)
    if country_code is None:
        return pd.DataFrame()
    if MARKET_DF.empty:
        return pd.DataFrame()

    variables     = ["BRAND_UPDATE", "OEM_UPDATE", "HIGHEST_BEV", "CO2_BUCKET"]
    status_values = ["IN FLEET", "ORDER", "ORDER YTD", "DEHIRE"]
    port_reg_values = [["PV"], ["LCV"], ["PV", "LCV"]]
    mkt_reg_values  = [
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

    country_map = {
        "FR": "France", "BE": "Belgium", "UK": "United Kingdom",
        "NL": "Netherlands", "LU": "Luxembourg", "IT": "Italy",
        "ES": "Spain", "DE": "Germany",
    }

    # ── 1. Pre-compute market aggregations: (variable, mkt_reg_key, period_mode) ──
    # Done 4 × 7 × 3 = 84 times instead of 4 × 4 × 3 × 7 × 3 × 4 = 336+ times
    market_cache: dict = {}
    for variable in variables:
        for mkt_reg in mkt_reg_values:
            mkt_key = tuple(mkt_reg)
            m_sub = MARKET_DF[MARKET_DF["Registration Type"].isin(mkt_reg)] \
                if "Registration Type" in MARKET_DF.columns else MARKET_DF
            market_ready, market_var = fmd.prepare_market_concentration_source(m_sub, variable)
            if market_ready.empty:
                continue
            mkt_dates = pd.to_datetime(market_ready["date"], errors="coerce")
            for period_mode in period_modes:
                if period_mode == "monthly":
                    period_col = mkt_dates.dt.strftime("%Y-%m")
                elif period_mode == "yearly":
                    period_col = mkt_dates.dt.year.astype(str)
                else:
                    period_col = mkt_dates.dt.to_period("Q").astype(str)
                m_pm = market_ready.copy()
                m_pm["PERIOD"] = period_col.values
                mkt_agg = (
                    m_pm.groupby(["Country/Territory-Name", "PERIOD", market_var])["volume"]
                    .sum().reset_index(name="volume_market")
                )
                mkt_total = mkt_agg.groupby(["Country/Territory-Name", "PERIOD"])["volume_market"].transform("sum")
                mkt_agg["share_market"] = (
                    mkt_agg["volume_market"] / mkt_total.where(mkt_total != 0, 1) * 100
                ).round(2)
                mkt_agg = mkt_agg.rename(columns={"Country/Territory-Name": "COUNTRY", market_var: "BRAND"})
                mkt_agg["BRAND"] = mkt_agg["BRAND"].astype(str).str.upper()
                market_cache[(variable, mkt_key, period_mode)] = mkt_agg

    # ── 2. Pre-compute portfolio aggregations: (variable, status, port_reg_key, period_mode) ──
    # Done 4 × 4 × 3 × 3 = 144 times instead of being redone inside each kpi_top_brand_vs_market call
    portfolio_cache: dict = {}
    for variable in variables:
        portfolio_ready, portfolio_var = fmd.prepare_portfolio_concentration_source(df, variable)
        if portfolio_ready.empty:
            continue
        for status_value in status_values:
            port_s = fmd.apply_status_filter(portfolio_ready, status_value)
            if port_s.empty:
                continue
            for port_reg in port_reg_values:
                port_key = tuple(port_reg)
                port_r = port_s[port_s["CLS_VEHICLE_TYPE"].isin(port_reg)] \
                    if "CLS_VEHICLE_TYPE" in port_s.columns else port_s
                if port_r.empty:
                    continue
                port_dates = pd.to_datetime(port_r["COB_DATE"], errors="coerce")
                for period_mode in period_modes:
                    if period_mode == "monthly":
                        period_col = port_dates.dt.strftime("%Y-%m")
                    elif period_mode == "yearly":
                        period_col = port_dates.dt.year.astype(str)
                    else:
                        period_col = port_dates.dt.to_period("Q").astype(str)
                    port_pm = port_r.copy()
                    port_pm["PERIOD"] = period_col.values
                    port_pm["_tmp_key"] = fmd.get_unique_key_series(port_pm)
                    port_agg = (
                        port_pm.groupby(["COUNTRY", "PERIOD", portfolio_var])["_tmp_key"]
                        .nunique().reset_index(name="volume_portfolio")
                    )
                    port_agg["COUNTRY"] = port_agg["COUNTRY"].map(country_map).fillna(port_agg["COUNTRY"])
                    port_total = port_agg.groupby(["COUNTRY", "PERIOD"])["volume_portfolio"].transform("sum")
                    port_agg["share_portfolio"] = (
                        port_agg["volume_portfolio"] / port_total.where(port_total != 0, 1) * 100
                    ).round(2)
                    port_agg = port_agg.rename(columns={portfolio_var: "BRAND"})
                    port_agg["BRAND"] = port_agg["BRAND"].astype(str).str.upper()
                    portfolio_cache[(variable, status_value, port_key, period_mode)] = port_agg

    # ── 3. Build rows: merge pre-aggregated data, slice by top_n ──────────────
    rows: list[pd.DataFrame] = []
    for variable in variables:
        for status_value in status_values:
            for port_reg in port_reg_values:
                port_key = tuple(port_reg)
                for mkt_reg in mkt_reg_values:
                    mkt_key = tuple(mkt_reg)
                    for period_mode in period_modes:
                        port_agg = portfolio_cache.get((variable, status_value, port_key, period_mode))
                        mkt_agg  = market_cache.get((variable, mkt_key, period_mode))
                        if port_agg is None or mkt_agg is None or port_agg.empty or mkt_agg.empty:
                            continue
                        avail = set(mkt_agg["PERIOD"].unique())
                        port_filt = port_agg[port_agg["PERIOD"].isin(avail)]
                        if port_filt.empty:
                            continue
                        merged = pd.merge(port_filt, mkt_agg, on=["COUNTRY", "PERIOD", "BRAND"], how="inner")
                        if merged.empty:
                            continue
                        merged["ratio"] = (
                            merged["share_portfolio"] / merged["share_market"].where(merged["share_market"] != 0, 1)
                        ).round(2)
                        merged = merged.sort_values(
                            ["COUNTRY", "PERIOD", "share_portfolio"], ascending=[True, True, False]
                        )
                        for top_n in top_n_values:
                            if top_n == "ALL":
                                kpi13 = merged.copy()
                            else:
                                kpi13 = merged.groupby(["COUNTRY", "PERIOD"]).head(int(top_n)).copy()
                            kpi13["COUNTRY_FILTER"]     = country_code
                            kpi13["STATUS_FILTER"]      = status_value
                            kpi13["PORT_REG_FILTER"]    = serialize_value(port_reg)
                            kpi13["MKT_REG_FILTER"]     = serialize_value(mkt_reg)
                            kpi13["VARIABLE_FILTER"]    = variable
                            kpi13["PERIOD_MODE_FILTER"] = period_mode
                            kpi13["TOP_N_FILTER"]       = top_n
                            kpi13["OWNER_FILTER"]       = "ALL"
                            rows.append(kpi13)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Main loop: one country at a time ──────────────────────────────────────────

def run(countries: list[str]) -> None:
    _ensure_models_parquet(countries)
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
        if CONFIG["GENERATE_VIEW3"] or CONFIG["GENERATE_VIEW4"]:
            df_enriched = enrich_with_models(df)
        else:
            df_enriched = df

        if CONFIG["GENERATE_VIEW3"]:
            print("  View 3 generating...")
            v3_91, v3_92 = generate_view3_for_country(df_enriched)
            if not v3_91.empty:
                view3_kpi91_frames.append(v3_91)
            if not v3_92.empty:
                view3_kpi92_frames.append(v3_92)
        if CONFIG["GENERATE_VIEW4"]:
            print("  View 4 generating...")
            v4 = generate_view4_for_country(df_enriched)
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
        save_parquet_merge(pd.concat(view5_portfolio_frames, ignore_index=True), os.path.join(BASE_OUT, "view5", "kpi11.parquet"))
    if view6_frames:
        save_parquet_merge(pd.concat(view6_frames, ignore_index=True), os.path.join(BASE_OUT, "view6", "kpi13.parquet"))

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
