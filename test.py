nova = nova.merge(
    nova_models[["BRAND_UPDATE", "MODEL", "MODEL_2"]],
    how="left",
    on=["BRAND_UPDATE", "MODEL"]
)

nova["Market_Model"] = nova["MODEL_2"]
nova.drop(columns=["MODEL_2"], inplace=True)