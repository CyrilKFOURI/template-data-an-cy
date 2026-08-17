
for col in df.columns:
    try:
        df[[col]].to_parquet("/tmp/test.parquet", index=False)
    except Exception as e:
        print(f"❌ {col} : {e}")