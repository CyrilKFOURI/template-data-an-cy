import pandas as pd

for col in df.columns:
    # On essaie de convertir en numérique
    numeric = pd.to_numeric(df[col], errors="coerce")

    # Si toutes les valeurs non-nulles sont numériques
    if numeric.notna().sum() == df[col].notna().sum():

        # Si toutes les valeurs numériques sont entières
        if (numeric.dropna() % 1 == 0).all():
            df[col] = numeric.astype("Int64")
        else:
            df[col] = numeric.astype("float64")

    # Sinon, on garde en texte
    else:
        df[col] = df[col].astype("string")