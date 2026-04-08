import pandas as pd

country_map = {
    "DE": "GERMANY",
    "BE": "BELGIUM",
    "NL": "NETHERLANDS",
    "LU": "LUXEMBOURG"
}

df["COB_DATE"] = pd.to_datetime(df["COB_DATE"])

df = df[
    (df["COB_DATE"] >= "2023-02-01") &
    (df["COB_DATE"] < "2026-03-01")
]

for country_code in df["COUNTRY"].unique():
    if country_code not in country_map:
        continue
    country_name = country_map[country_code]
    df_country = df[df["COUNTRY"] == country_code]
    for period in df_country["COB_DATE"].dt.to_period("M").unique():
        df_slice = df_country[df_country["COB_DATE"].dt.to_period("M") == period]
        yyyymm = period.strftime("%Y%m")
        output_file = f"/full/path/to/save/NOVA - {country_name} - {yyyymm}.parquet"
        df_slice.to_parquet(output_file, index=False)
        print(f"Saved {output_file}")
