import pyarrow.parquet as pq

pf = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402.parquet")
pf2 = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202401.parquet")

table = pf2.read()
table = table.cast(pf.schema_arrow, safe=False)

pq.write_table(table, r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402_fixed.parquet")
