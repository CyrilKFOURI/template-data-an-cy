import pandas as pd
from dtreeviz.trees import dtreeviz

model = xgboost_simple_model.model
X_train = xgboost_simple_model.X_train
y_train = xgboost_simple_model.y_train

X_train_df = pd.DataFrame(
    X_train,
    columns=[f"f{i}" for i in range(X_train.shape[1])]
)

X_train_df["target"] = y_train
X_train_df = X_train_df.dropna()

y_train_series = X_train_df["target"]
X_train_df = X_train_df.drop(columns=["target"])

model.get_booster().feature_names = X_train_df.columns.tolist()

viz = dtreeviz(
    model,
    X_train_df,
    y_train_series,
    target_name="target",
    tree_index=0
)

viz