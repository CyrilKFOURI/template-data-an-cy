import pyarrow as pa
import pyarrow.parquet as pq

pf = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402.parquet")
pf_2 = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202401.parquet")

table = pf.read()
new_names = pf_2.schema.names

table = table.rename_columns(new_names)

pq.write_table(table, r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402_fixed.parquet")

