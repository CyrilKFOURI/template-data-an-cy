import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

pf = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402.parquet")
pf2 = pq.ParquetFile(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202401.parquet")

t2 = pf2.read()
t1 = pf.read()

print(len(t1.schema.names), len(t2.schema.names))

# Reconstruit la table de pf2 avec exactement les noms de pf
df2 = t2.to_pandas()
df2.columns = t1.schema.names

# si les noms sont dupliqués, on les rend uniques uniquement pour pouvoir écrire
cols = pd.Index(df2.columns)
if cols.duplicated().any():
    counts = {}
    new_cols = []
    for c in df2.columns:
        counts[c] = counts.get(c, 0) + 1
        new_cols.append(c if counts[c] == 1 else f"{c}__{counts[c]}")
    df2.columns = new_cols

df2.to_parquet(r"C:\Users\j21958\OneDrive - BNP Paribas\Documents\Nov\Data\Parquets\NOVA - TR - 202402_fixed.parquet", index=False)
