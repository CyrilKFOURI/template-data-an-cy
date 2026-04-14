def kp12_lease_25_30(self, NOVA_ASSET_STATUS, country, bike_or_car='CAR'):

    df = self.df[
        (self.df["NOVA_ASSET_STATUS"] == NOVA_ASSET_STATUS) &
        (self.df["COUNTRY"] == country) &
        (self.df["BIKE_OR_CAR"] == bike_or_car) &
        (self.df["YEAR"] == 2025)
    ]

    f = lambda x: (
        x.loc[
            (x["FINAL_CONTRACT_DURATION"] > 25) &
            (x["FINAL_CONTRACT_DURATION"] <= 30),
            "VEHICLE_ID"
        ].nunique()
        /
        x["VEHICLE_ID"].nunique()
    ) * 100

    kpis = df.groupby("MONTH").apply(f)

    kpis = kpis.reset_index().rename(columns={0: "25-30_months_%"})

    return kpis