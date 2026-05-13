nova = nova.merge(
    market_df[["Make", "Sub Model Short", "BODY_GROUP"]],
    how="left",
    left_on=["BRAND_UPDATE", "Market_Model"],
    right_on=["Make", "Sub Model Short"]
)
nova.drop(columns=["Make", "Sub Model Short"], inplace=True)