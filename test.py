import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher

def match_models_between_dfs(
    df1,
    df2,
    brand_col_df1,
    model_col_df1,
    brand_col_df2,
    model_col_df2,
    output_col="MODEL_MATCH",
    threshold=80 # Threshold mis à 80 pour correspondre au score de 0.8
):
    def normalize_text(text):
        if pd.isna(text): return ""
        text = str(text).upper()
        # 1. Remplacement des séparateurs par des espaces
        text = text.replace("-", " ").replace("_", " ").replace("/", " ")
        # 2. Normalisation Unicode (accents, tildes)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        # 3. Garder seulement alphanumérique et espaces
        text = re.sub(r"[^A-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df1 = df1.copy()
    df2 = df2.copy()

    # Nettoyage systématique des deux datasets
    df1["_BRAND_CLEAN"] = df1[brand_col_df1].apply(normalize_text)
    df1["_MODEL_CLEAN"] = df1[model_col_df1].apply(normalize_text)
    df2["_BRAND_CLEAN"] = df2[brand_col_df2].apply(normalize_text)
    df2["_MODEL_CLEAN"] = df2[model_col_df2].apply(normalize_text)

    df1[output_col] = None
    df1[f"{output_col}_SCORE"] = 0

    for brand in df2["_BRAND_CLEAN"].unique():
        if not brand: continue
        
        df2_brand = df2[df2["_BRAND_CLEAN"] == brand]
        df1_brand_idx = df1[df1["_BRAND_CLEAN"] == brand].index

        for idx in df1_brand_idx:
            model_1 = df1.at[idx, "_MODEL_CLEAN"]
            best_score = 0
            best_match = None

            for _, row2 in df2_brand.iterrows():
                model_2 = row2["_MODEL_CLEAN"]
                if not model_2: continue
                
                # Similarité de chaîne (0 à 1) convertie en score 0-100
                score = int(SequenceMatcher(None, model_1, model_2).ratio() * 100)
                
                if score > best_score:
                    best_score = score
                    best_match = row2[model_col_df2]

            if best_score >= threshold:
                df1.at[idx, output_col] = best_match
                df1.at[idx, f"{output_col}_SCORE"] = best_score
                
    return df1
