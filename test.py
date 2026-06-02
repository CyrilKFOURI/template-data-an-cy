import pyarrow.parquet as pq

pf = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402.parquet")
pf_2 = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202401.parquet")

df = pf.read().to_pandas()
df.columns = pf_2.schema.names

df.head()
