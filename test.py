import pandas as pd

def match_models_between_dfs(
    df1,
    df2,
    brand_col_df1,
    model_col_df1,
    brand_col_df2,
    model_col_df2,
    output_col="MODEL_MATCH"
):

    # =========================
    # CLEAN FUNCTION
    # =========================
    def norm(s):
        return (
            s.fillna("")
             .astype(str)
             .str.upper()
             .str.strip()
        )

    # =========================
    # COPY + CLEAN
    # =========================
    df1 = df1.copy()
    df2 = df2.copy()

    df1[brand_col_df1] = norm(df1[brand_col_df1])
    df1[model_col_df1] = norm(df1[model_col_df1])

    df2[brand_col_df2] = norm(df2[brand_col_df2])
    df2[model_col_df2] = norm(df2[model_col_df2])

    # output col
    df1[output_col] = None

    # =========================
    # MATCHING
    # =========================
    for brand in df2[brand_col_df2].unique():

        df2_brand = df2[df2[brand_col_df2] == brand]
        df1_brand_mask = df1[brand_col_df1] == brand

        for model in df2_brand[model_col_df2].unique():

            if model == "":
                continue

            mask = (
                df1_brand_mask &
                df1[model_col_df1].str.contains(model, na=False, regex=False)
            )

            df1.loc[mask, output_col] = model

    return df1