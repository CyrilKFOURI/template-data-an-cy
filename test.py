import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION SIMPLIFIÉE ---
# Remplacez uniquement par votre adresse IP et port (ex: 10.0.0.1:8080)
proxy_server = "votre-adresse-proxy:port" 

os.environ['HTTP_PROXY'] = f"http://{proxy_server}"
os.environ['HTTPS_PROXY'] = f"http://{proxy_server}"

# --- CHARGEMENT DU MODÈLE ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# --- DÉFINITION DE LA FONCTION ---
def match_models_between_dfs(df1, df2, brand_col_df1, model_col_df1, brand_col_df2, model_col_df2, output_col="MARKET_MODEL", threshold=0.75):
    df1_c = df1.copy()
    df2_c = df2.copy()

    df1_c["_TEXT"] = df1_c[brand_col_df1].astype(str) + " " + df1_c[model_col_df1].astype(str)
    df2_c["_TEXT"] = df2_c[brand_col_df2].astype(str) + " " + df2_c[model_col_df2].astype(str)
    
    emb_df2 = model.encode(df2_c["_TEXT"].tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    emb_df1 = model.encode(df1_c["_TEXT"].tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(emb_df2)
    
    dist, idx = nn.kneighbors(emb_df1)
    
    df1_c[output_col] = df2_c.iloc[idx.flatten()][model_col_df2].values
    df1_c[f"{output_col}_SCORE"] = 1 - dist.flatten()
    
    return df1_c

print("Configuration simplifiée appliquée.")
