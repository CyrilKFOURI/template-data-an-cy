import os
import glob
import pandas as pd

def load_country_monthly_data(
    folder_path,
    countries,
    start_yyyymm,
    end_yyyymm,
    cols=None
):
    files = glob.glob(f"{folder_path}/*.parquet")

    start_int = int(start_yyyymm)
    end_int = int(end_yyyymm)

    countries = [c.upper() for c in countries]

    dfs_by_month = {}
    file_prefix = None

    for f in files:
        filename = os.path.basename(f).replace(".parquet", "")
        parts = [p.strip() for p in filename.split("-")]

        # attendu : NOVA - COUNTRY - YYYYMM
        if len(parts) < 3:
            continue

        current_prefix = parts[0]
        file_country = parts[1].upper()
        file_yyyymm = int(parts[2])

        if file_country in countries and start_int <= file_yyyymm <= end_int:
            if file_prefix is None:
                file_prefix = current_prefix

            df = pd.read_parquet(f, columns=cols)
            dfs_by_month.setdefault(file_yyyymm, []).append(df)

    output_files = []

    for month, dfs in sorted(dfs_by_month.items()):
        full_df = pd.concat(dfs, ignore_index=True)
        output_file = os.path.join(folder_path, f"{file_prefix} - G4 - {month}.parquet")
        full_df.to_parquet(output_file, index=False)
        output_files.append(output_file)

    return output_files
