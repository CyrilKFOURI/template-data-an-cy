import pandas as pd

df_clean = (
    df.melt(id_vars="categories", var_name="category", value_name="models")
      .dropna(subset=["models"])
)

df_clean["model"] = df_clean["models"].str.split("/")

df_clean = df_clean.explode("model")

df_clean["model"] = df_clean["model"].str.strip()

df_clean = df_clean.drop(columns=["models"]).rename(columns={"categories": "brand"})





mapping = {
    "mercedes": "mercedes benz",
    "mercedes-benz": "mercedes benz",
    "bmw ": "bmw",
    "vw": "volkswagen"
}

df["brand"] = df["brand"].str.lower().str.strip().replace(mapping)