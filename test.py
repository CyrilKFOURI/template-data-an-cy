import pyarrow as pa
import pyarrow.parquet as pq

pf = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402.parquet")
pf2 = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202401.parquet")

table = pf2.read()
table = pa.Table.from_arrays(table.columns, names=pf.schema.names)

pq.write_table(table, r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402_fixed.parquet")
