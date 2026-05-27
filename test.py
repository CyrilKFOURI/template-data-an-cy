import pandas as pd
import re
import unicodedata
import ipywidgets as widgets
from IPython.display import display, HTML

# Fonction de normalisation unifiée et ultra-robuste
def normalize_text(text):
    if pd.isna(text):
        return ""
    
    # 1. Conversion en texte
    text = str(text)
    
    # 2. Gestion explicite des séparateurs avant toute chose
    text = text.replace("-", " ").replace("_", " ").replace("/", " ")
    
    # 3. Conversion en majuscule
    text = text.upper()
    
    # 4. Normalisation Unicode (NFKD) pour décomposer les accents/tildes
    text = unicodedata.normalize("NFKD", text)
    
    # 5. Suppression des caractères "combining" (accents, tildes, etc.)
    text = "".join(c for c in text if not unicodedata.combining(c))
    
    # 6. Suppression de TOUT ce qui n'est pas lettre ou chiffre
    # Cela vire apostrophes, tildes isolés, etc.
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    
    # 7. Nettoyage des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def match_models_between_dfs(df1, df2, brand_col_df1, model_col_df1, brand_col_df2, model_col_df2, output_col="MODEL_MATCH", threshold=10):
    weak_tokens = {"CLASS", "SERIES", "SERIE", "MODEL", "NEW", "PHASE", "TYPE", "MY"}
    strong_numeric_patterns = [r"Q\d", r"X\d", r"\d{3,4}"]

    def is_strong_numeric(token):
        return any(re.fullmatch(pattern, token) for pattern in strong_numeric_patterns)

    df1 = df1.copy()
    df2 = df2.copy()

    # Application stricte de la normalisation sur les deux datasets
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
            tokens_1 = model_1.split()
            best_score = 0
            best_match = None

            for _, row2 in df2_brand.iterrows():
                model_2 = row2["_MODEL_CLEAN"]
                if not model_2: continue
                
                tokens_2 = model_2.split()
                score = 0
                
                if model_1 == model_2: score += 100
                
                common_tokens = set(tokens_1) & set(tokens_2)
                for token in common_tokens:
                    if token in weak_tokens: continue
                    elif token.isdigit(): score += 15 if is_strong_numeric(token) else 1
                    elif re.search(r"\d", token): score += 15
                    else: score += 10

                # Vos conditions regex originales
                if re.search(r"\s+", model_1): pass # Conserve logique initiale
                if model_2 in model_1 and len(model_2) > 2: score += 20
                if len(common_tokens) == 0: score -= 20
                
                if score > best_score:
                    best_score = score
                    best_match = row2[model_col_df2]

            if best_score >= threshold:
                df1.at[idx, output_col] = best_match
                df1.at[idx, f"{output_col}_SCORE"] = best_score
    return df1

# --- EXECUTION ---
nova_models = match_models_between_dfs(
    df1=nova_models,
    df2=market_models,
    brand_col_df1="BRAND_UPDATE",
    model_col_df1="MODEL",
    brand_col_df2="Make",
    model_col_df2="Sub Model Short",
    output_col="MODEL_2"
)

# compare_model_matching reste identique à votre version
