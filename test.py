import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq

def load_country_monthly_data_pyarrow(
    folder_path,
    countries,
    start_yyyymm,
    end_yyyymm
):
    files = glob.glob(f"{folder_path}/*.parquet")

    start_int = int(start_yyyymm)
    end_int = int(end_yyyymm)
    countries = [c.upper().strip() for c in countries]

    by_month = {}
    file_prefix = None

    for f in files:
        filename = os.path.basename(f).replace(".parquet", "")
        parts = [p.strip() for p in filename.split("-")]

        if len(parts) < 3:
            continue

        current_prefix = parts[0]
        file_country = parts[1].upper()
        file_yyyymm = int(parts[2])

        if file_country in countries and start_int <= file_yyyymm <= end_int:
            if file_prefix is None:
                file_prefix = current_prefix
            by_month.setdefault(file_yyyymm, []).append(f)

    output_files = []

    for month, month_files in sorted(by_month.items()):
        output_file = os.path.join(folder_path, f"{file_prefix} - G4 - {month}.parquet")

        writer = None
        for f in month_files:
            table = pq.read_table(f)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")
            writer.write_table(table)

        if writer is not None:
            writer.close()
            print(f"Parquet créé : {output_file}")
            output_files.append(output_file)

    return output_files
