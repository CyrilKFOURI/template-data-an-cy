result = df.groupby(
    ["POWER_CATEGORY", "CLS_VEHICLE_TYPE"]
).agg(
    avg_duration=("FINAL_CONTRACT_DURATION", lambda x: x[x > 0].mean())
).reset_index()