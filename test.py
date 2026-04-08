import os
import pandas as pd

country_map = {
    "DE": "Germany",
    "BE": "Belgium",
    "NL": "Netherlands",
    "LU": "Luxembourg"
}

for country_code in df["COUNTRY"].unique():
    if country_code not in country_map:
        continue
    country_name = country_map[country_code]
    df_country = df[df["COUNTRY"] == country_code]
    for yyyymm in df_country["YYYYMM"].unique():
        df_slice = df_country[df_country["YYYYMM"] == yyyymm]
        output_file = f"/full/path/to/save/{country_name}_{yyyymm}.parquet"
        df_slice.to_parquet(output_file, index=False)
        print(f"Saved {output_file}")
