
df.columns = [str(col).encode("utf-8", "ignore").decode("utf-8") for col in df.columns]