import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# 1. Pipeline de Matching Statistique (Rapide et précis sur l'orthographe)
# On utilise des n-grammes de caractères pour être robuste aux fautes
def get_statistical_matcher(df_market):
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    matrix = tfidf.fit_transform(df_market['_MODEL_CLEAN'])
    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(matrix)
    return tfidf, nn

# 2. Pipeline NLP "LSA" (Sémantique sans Transformers)
# On réduit les dimensions pour capturer des "concepts" au lieu de mots exacts
def get_lsa_matcher(df_market):
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=50) # Réduit à 50 dimensions latentes
    pipeline = make_pipeline(tfidf, svd)
    matrix = pipeline.fit_transform(df_market['_MODEL_CLEAN'])
    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(matrix)
    return pipeline, nn

# --- Workflow complet ---
# Étape A : Matching strict (Statistique)
tfidf_stat, nn_stat = get_statistical_matcher(market_models)
dist, idx = nn_stat.kneighbors(tfidf_stat.transform(nova_models['_MODEL_CLEAN']))

# Assigner les résultats si la distance est très faible (match quasi-parfait)
mask_match = dist.flatten() < 0.2
nova_models.loc[mask_match, 'MODEL_2'] = market_models.iloc[idx[mask_match]].values.flatten()

# Étape B : NLP sur les résidus (LSA)
unmatched_mask = nova_models['MODEL_2'].isna()
if unmatched_mask.any():
    lsa_pipe, nn_lsa = get_lsa_matcher(market_models)
    
    # On fait tourner l'algo sur les non-matchés
    unmatched_data = nova_models.loc[unmatched_mask, '_MODEL_CLEAN']
    dist_lsa, idx_lsa = nn_lsa.kneighbors(lsa_pipe.transform(unmatched_data))
    
    # On assigne avec un seuil de confiance plus souple
    nova_models.loc[unmatched_mask, 'MODEL_2'] = market_models.iloc[idx_lsa].values.flatten()

