import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


def match_models_between_dfs(
    df1,
    df2,
    brand_col_df1,
    model_col_df1,
    brand_col_df2,
    model_col_df2,
    output_col="MARKET_MODEL",
    threshold=0.75
):

    # =========================================================
    # 1. MODEL (LOCAL, NO API)
    # =========================================================
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df1 = df1.copy()
    df2 = df2.copy()

    # =========================================================
    # 2. BUILD TEXTS (SAME LOGIC AS BEFORE, BUT SIMPLE)
    # =========================================================
    df1["_TEXT"] = (
        df1[brand_col_df1].astype(str)
        + " " +
        df1[model_col_df1].astype(str)
    )

    df2["_TEXT"] = (
        df2[brand_col_df2].astype(str)
        + " " +
        df2[model_col_df2].astype(str)
    )

    # =========================================================
    # 3. EMBEDDINGS
    # =========================================================
    emb_df2 = model.encode(
        df2["_TEXT"].tolist(),
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    emb_df1 = model.encode(
        df1["_TEXT"].tolist(),
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # =========================================================
    # 4. NEAREST NEIGHBOR SEARCH
    # =========================================================
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(emb_df2)

    dist, idx = nn.kneighbors(emb_df1)

    # =========================================================
    # 5. OUTPUT (SAME BEHAVIOR AS YOUR OLD CODE)
    # =========================================================
    df1[output_col] = df2.iloc[idx.flatten()][model_col_df2].values
    df1[f"{output_col}_SCORE"] = 1 - dist.flatten()

    return df1



nova_models = match_models_between_dfs(
    df1=nova_models,
    df2=market_models,

    brand_col_df1="BRAND_UPDATE",
    model_col_df1="MODEL",

    brand_col_df2="Make",
    model_col_df2="Sub Model Short",

    output_col="MARKET_MODEL"
)