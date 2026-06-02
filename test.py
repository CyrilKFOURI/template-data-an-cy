import pyarrow.parquet as pq

pf = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402.parquet")
pf2 = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202401.parquet")

table2 = pf2.read()
table2 = table2.replace_schema_metadata(pf.schema_arrow.metadata)
table2 = table2.cast(pf.schema_arrow, safe=False)

pq.write_table(table2, r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402_fixed.parquet")
