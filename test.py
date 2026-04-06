import os
import pyarrow.csv as pv
import pyarrow.parquet as pq
from datetime import datetime

def convert_gz_range(input_dir, output_dir, country, start_yyyymm, end_yyyymm):
    os.makedirs(output_dir, exist_ok=True)

    start = datetime.strptime(start_yyyymm, "%Y%m")
    end = datetime.strptime(end_yyyymm, "%Y%m")

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith(".gz"):
            continue

        if country not in file_name:
            continue

        try:
            parts = file_name.replace(".gz", "").split(" - ")
            date_part = parts[-1]
            file_date = datetime.strptime(date_part, "%Y%m")
        except:
            continue

        if not (start <= file_date <= end):
            continue

        gz_path = os.path.join(input_dir, file_name)
        parquet_name = file_name.replace(".gz", ".parquet")
        parquet_path = os.path.join(output_dir, parquet_name)

        print(f"Processing {file_name}...")

        try:
            table = pv.read_csv(
                gz_path,
                parse_options=pv.ParseOptions(
                    delimiter=';',
                    newlines_in_values=True
                )
            )

            pq.write_table(table, parquet_path)
            print(f"→ Saved {parquet_name} ✅")

        except Exception as e:
            print(f"⚠️ Error on {file_name}: {e}")

    print("Done ✅")


convert_gz_range(
    input_dir=r"C:\Users\j21958\OneDrive BNP Paribas\Documents\Nova\Data\NEW\GZ",
    output_dir="NewDataParquet",
    country="ITALY",
    start_yyyymm="202602",
    end_yyyymm="202604"
)



