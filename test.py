import pandas as pd
import re
import unicodedata

def match_models_between_dfs(df1, df2, brand_col_df1, model_col_df1, brand_col_df2, model_col_df2, output_col="MODEL_MATCH", threshold=10):
    
    # --- Fonction de nettoyage robuste ---
    def normalize_text(text):
        if pd.isna(text): return ""
        # 1. Séparateurs -> espaces
        text = str(text).replace("-", " ").replace("_", " ").replace("/", " ")
        text = text.upper()
        # 2. Suppression accents/tildes
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        # 3. Suppression tout ce qui n'est pas lettre/chiffre
        text = re.sub(r"[^A-Z0-9\s]", " ", text)
        # 4. Nettoyage espaces
        return re.sub(r"\s+", " ", text).strip()
    # -------------------------------------

    df1 = df1.copy()
    df2 = df2.copy()

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
            tokens_1 = set(model_1.split())
            best_score = 0
            best_match = None

            for _, row2 in df2_brand.iterrows():
                model_2 = row2["_MODEL_CLEAN"]
                if not model_2: continue
                tokens_2 = set(model_2.split())
                
                score = 0
                if model_1 == model_2: score += 100
                
                # Bonus intersection de mots
                common = tokens_1.intersection(tokens_2)
                score += len(common) * 20
                
                # Bonus si l'un est inclus dans l'autre
                if model_2 in model_1 or model_1 in model_2:
                    score += 30
                
                if score > best_score:
                    best_score = score
                    best_match = row2[model_col_df2]

            if best_score >= threshold:
                df1.at[idx, output_col] = best_match
                df1.at[idx, f"{output_col}_SCORE"] = best_score
                
    return df1

# --- Exécution ---
nova_models = match_models_between_dfs(
    df1=nova_models,
    df2=market_models,
    brand_col_df1="BRAND_UPDATE",
    model_col_df1="MODEL",
    brand_col_df2="Make",
    model_col_df2="Sub Model Short",
    output_col="MODEL_2"
)
