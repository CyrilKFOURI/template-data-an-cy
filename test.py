import os
import pandas as pd

output_folder = "output_parquets"
os.makedirs(output_folder, exist_ok=True)

for country in df["COUNTRY"].unique():
    df_country = df[df["COUNTRY"] == country]
    for yyyymm in df_country["YYYYMM"].unique():
        df_slice = df_country[df_country["YYYYMM"] == yyyymm]
        output_file = os.path.join(output_folder, f"{country}_{yyyymm}.parquet")
        df_slice.to_parquet(output_file, index=False)
        print(f"Saved {output_file}")
