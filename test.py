import glob
import os
import pandas as pd


def load_country_monthly_data(
    folder_path,
    countries,
    start_yyyymm,
    end_yyyymm,
    cols=None
):

    files = glob.glob(f"{folder_path}/*.parquet")

    dfs = []

    start_int = int(start_yyyymm)
    end_int = int(end_yyyymm)

    # Met tous les pays en uppercase
    countries = [c.upper() for c in countries]

    for f in files:

        filename = os.path.basename(f).replace(".parquet", "")

        parts = [p.strip() for p in filename.split("-")]

        file_country = parts[1].upper()
        file_yyyymm = int(parts[2])

        if (
            file_country in countries
            and start_int <= file_yyyymm <= end_int
        ):

            df = pd.read_parquet(f, columns=cols)

            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    return full_df


df = load_country_monthly_data(
    folder_path=data_path,
    countries=["FR", "DE", "IT"],
    start_yyyymm="202501",
    end_yyyymm="202512",
    cols=columns_to_read
)