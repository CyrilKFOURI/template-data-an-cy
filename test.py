import pandas as pd
import re
import unicodedata
import ipywidgets as widgets
from IPython.display import display, HTML
def match_models_between_dfs(
df1,
df2,
brand_col_df1,
model_col_df1,
brand_col_df2,
model_col_df2,
output_col="MODEL_MATCH",
threshold=10
):
weak_tokens = { "CLASS", "SERIES", "SERIE", "MODEL", "NEW", "PHASE", "TYPE", "MY" } strong_numeric_patterns = [ r"Q\d", r"X\d", r"\d{3,4}" ] def normalize_text(text): if pd.isna(text): return "" text = str(text).upper().strip() text = unicodedata.normalize("NFKD", text) text = "".join( c for c in text if not unicodedata.combining(c) ) text = ( text.replace("-", " ") .replace("_", " ") .replace("/", " ") ) text = re.sub(r"\s+", " ", text) text = re.sub(r"([A-Z]+)(\d)", r"\1 \2", text) text = re.sub(r"(\d)([A-Z]+)", r"\1 \2", text) return text.strip() def tokenize(text): return text.split() def is_strong_numeric(token): for pattern in strong_numeric_patterns: if re.fullmatch(pattern, token): return True return False df1 = df1.copy() df2 = df2.copy() df1["_BRAND_CLEAN"] = df1[brand_col_df1].apply(normalize_text) df1["_MODEL_CLEAN"] = df1[model_col_df1].apply(normalize_text) df2["_BRAND_CLEAN"] = df2[brand_col_df2].apply(normalize_text) df2["_MODEL_CLEAN"] = df2[model_col_df2].apply(normalize_text) df1[output_col] = None df1[f"{output_col}_SCORE"] = 0 for brand in df2["_BRAND_CLEAN"].unique(): if brand == "": continue df2_brand = df2[df2["_BRAND_CLEAN"] == brand] df1_brand_idx = df1[ df1["_BRAND_CLEAN"] == brand ].index for idx in df1_brand_idx: model_1 = df1.at[idx, "_MODEL_CLEAN"] tokens_1 = tokenize(model_1) best_score = 0 best_match = None for _, row2 in df2_brand.iterrows(): model_2 = row2["_MODEL_CLEAN"] if model_2 == "": continue tokens_2 = tokenize(model_2) score = 0 if model_1 == model_2: score += 100 common_tokens = set(tokens_1) & set(tokens_2) for token in common_tokens: if token in weak_tokens: score += 0 elif token.isdigit(): if is_strong_numeric(token): score += 15 else: score += 1 elif re.search(r"\d", token): score += 15 else: score += 10 if ( model_2 in model_1 and len(model_2) > 2 ): score += 20 if len(common_tokens) == 0: score -= 20 if score > best_score: best_score = score best_match = row2[model_col_df2] if best_score >= threshold: df1.at[idx, output_col] = best_match df1.at[idx, f"{output_col}_SCORE"] = best_score return df1 
def compare_model_matching(
df,
brand_col,
original_model_col,
matched_model_col,
score_col=None
):
def norm(x): if pd.isna(x): return "" x = str(x).upper().strip() x = unicodedata.normalize("NFKD", x) x = "".join( c for c in x if not unicodedata.combining(c) ) x = x.replace("-", " ") x = x.replace("_", " ") x = re.sub(r"\s+", " ", x) return x.strip() brands = sorted( df[brand_col] .dropna() .astype(str) .unique() ) dropdown = widgets.Dropdown( options=brands, description="Brand:", layout=widgets.Layout(width="400px") ) output = widgets.Output() def update(change): output.clear_output() brand = change["new"] sub = df[ df[brand_col] == brand ].copy() rows_html = "" matched_count = 0 unmatched_count = 0 for _, row in sub.iterrows(): model_1 = row[original_model_col] model_2 = row[matched_model_col] score = "" if score_col is not None: score = row[score_col] model_1_clean = norm(model_1) model_2_clean = norm(model_2) is_match = ( model_1_clean == model_2_clean or ( model_2_clean in model_1_clean and model_2_clean != "" ) ) if pd.isna(model_2): is_match = False color = "#d4edda" if is_match else "#f8d7da" if is_match: matched_count += 1 else: unmatched_count += 1 rows_html += f""" <tr style="background-color:{color}"> <td>{model_1}</td> <td>{model_2}</td> <td>{score}</td> </tr> """ html = f""" <h2>{brand}</h2> <p> ✅ Matched: {matched_count} <br> ❌ Unmatched: {unmatched_count} </p> <table border="1" style=" border-collapse:collapse; width:100%; " > <tr> <th>{original_model_col}</th> <th>{matched_model_col}</th> <th>Score</th> </tr> {rows_html} </table> """ with output: display(HTML(html)) dropdown.observe(update, names="value") update({"new": dropdown.value}) display(dropdown, output) 
nova_models = match_models_between_dfs(
df1=nova_models, df2=market_models, brand_col_df1="BRAND_UPDATE", model_col_df1="MODEL", brand_col_df2="Make", model_col_df2="Sub Model Short", output_col="MODEL_2" 
)
compare_model_matching(
df=nova_models, brand_col="BRAND_UPDATE", original_model_col="MODEL", matched_model_col="MODEL_2", score_col="MODEL_2_SCORE" 
)
