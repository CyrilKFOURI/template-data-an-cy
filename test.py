import os
import pandas as pd

BASE_DATA = "precomputed_fast/data"   # ajuster si besoin


def _merge_parquet_tree(root: str, suffix: str = ".parquet") -> pd.DataFrame:
    """Parcourt récursivement root et concatène tous les .parquet trouvés."""
    frames = []
    if not os.path.exists(root):
        print(f"[warn] dossier introuvable : {root}")
        return pd.DataFrame()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(suffix):
                try:
                    frames.append(pd.read_parquet(os.path.join(dirpath, f)))
                except Exception as e:
                    print(f"[warn] lecture échouée {dirpath}/{f} : {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(frames)} fichiers mergés → {len(df):,} lignes  ({root})")
    return df


def load_kpi1_6(base_data: str = BASE_DATA) -> pd.DataFrame:
    """
    Merge tous les kpis.parquet de view1/kpis/year=*/country=*/.
    Colonnes : COUNTRY, YEAR, MONTH, BIKE_OR_CAR, ASSET_STATUS, DATE_MODE,
               KPI1, KPI2, DIESEL, NON_DIESEL, HYBRID, EV, PV, LCV, VOLUME
    """
    return _merge_parquet_tree(os.path.join(base_data, "view1", "kpis"))


def load_kpi7(base_data: str = BASE_DATA) -> pd.DataFrame:
    """
    Merge tous les kpi7.parquet de view1/kpi7/year=*/country=*/.
    Colonnes : COUNTRY, YEAR, FUEL_TYPE, ASSET_STATUS, METRIC_MODE,
               PERIOD_MODE, BIKE_OR_CAR, + colonnes de périodes
    """
    return _merge_parquet_tree(os.path.join(base_data, "view1", "kpi7"))


def load_kpi8(base_data: str = BASE_DATA) -> pd.DataFrame:
    """
    Merge tous les kpi8.parquet de view2/kpi8/year=*/country=*/.
    Colonnes : COUNTRY, YEAR, ASSET_STATUS, METRIC_MODE, BIKE_OR_CAR,
               DATE_MODE, PERIOD_MODE, + colonnes de périodes (PERIOD ou mois)
    """
    return _merge_parquet_tree(os.path.join(base_data, "view2", "kpi8"))


df_kpis = load_kpi1_6()   # tous pays, toutes années mergés
df_kpi7 = load_kpi7()
df_kpi8 = load_kpi8()

# Filtrer ensuite selon besoin
df_es_2025 = df_kpis[
    (df_kpis["COUNTRY"] == "ES") &
    (df_kpis["YEAR"]    == 2025)
]