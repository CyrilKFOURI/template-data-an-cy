import pandas as pd
import re
import unicodedata

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

    weak_tokens = {
        "CLASS",
        "SERIES",
        "SERIE",
        "MODEL",
        "NEW",
        "PHASE",
        "TYPE",
        "MY"
    }

    strong_numeric_patterns = [
        r"Q\d",
        r"X\d",
        r"\d{3,4}"
    ]

    def normalize_text(text):

        if pd.isna(text):
            return ""

        text = str(text).upper().strip()

        text = unicodedata.normalize("NFKD", text)
        text = "".join(
            c for c in text
            if not unicodedata.combining(c)
        )

        text = (
            text.replace("-", " ")
                .replace("_", " ")
                .replace("/", " ")
        )

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"([A-Z]+)(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([A-Z]+)", r"\1 \2", text)

        return text.strip()

    def tokenize(text):
        return text.split()

    def is_strong_numeric(token):

        for pattern in strong_numeric_patterns:

            if re.fullmatch(pattern, token):
                return True

        return False

    df1 = df1.copy()
    df2 = df2.copy()

    df1[brand_col_df1] = df1[brand_col_df1].apply(normalize_text)
    df1[model_col_df1] = df1[model_col_df1].apply(normalize_text)

    df2[brand_col_df2] = df2[brand_col_df2].apply(normalize_text)
    df2[model_col_df2] = df2[model_col_df2].apply(normalize_text)

    df1[output_col] = None
    df1[f"{output_col}_SCORE"] = 0

    for brand in df2[brand_col_df2].unique():

        if brand == "":
            continue

        df2_brand = df2[df2[brand_col_df2] == brand]

        df1_brand_idx = df1[df1[brand_col_df1] == brand].index

        for idx in df1_brand_idx:

            model_1 = df1.at[idx, model_col_df1]

            tokens_1 = tokenize(model_1)

            best_score = 0
            best_match = None

            for model_2 in df2_brand[model_col_df2].unique():

                if model_2 == "":
                    continue

                tokens_2 = tokenize(model_2)

                score = 0

                if model_1 == model_2:
                    score += 100

                common_tokens = set(tokens_1) & set(tokens_2)

                for token in common_tokens:

                    if token in weak_tokens:
                        score += 0

                    elif token.isdigit():

                        if is_strong_numeric(token):
                            score += 15
                        else:
                            score += 1

                    elif re.search(r"\d", token):
                        score += 15

                    else:
                        score += 10

                if (
                    model_2 in model_1
                    and len(model_2) > 2
                ):
                    score += 20

                if len(common_tokens) == 0:
                    score -= 20

                if score > best_score:
                    best_score = score
                    best_match = model_2

            if best_score >= threshold:

                df1.at[idx, output_col] = best_match
                df1.at[idx, f"{output_col}_SCORE"] = best_score

    return df1