import pandas as pd
from dtreeviz.trees import dtreeviz

model = xgboost_simple_model.model
X_train = xgboost_simple_model.X_train
y_train = xgboost_simple_model.y_train

X_train_df = pd.DataFrame(
    X_train,
    columns=[f"f{i}" for i in range(X_train.shape[1])]
)

df = X_train_df.copy()
df["target"] = y_train

df = df.dropna().reset_index(drop=True)

X_train_clean = df.drop(columns=["target"])
y_train_clean = df["target"]

model.get_booster().feature_names = X_train_clean.columns.tolist()

viz = dtreeviz(
    model,
    X_train_clean,
    y_train_clean,
    target_name="target",
    tree_index=0
)

viz