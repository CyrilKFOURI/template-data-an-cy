import pandas as pd

error_cols = []

for col in df.columns:
    try:
        df[[col]].to_parquet("/tmp/test.parquet", index=False)
    except Exception as e:
        if "Expected bytes, got" in str(e):
            error_cols.append(col)

print(error_cols)