import pandas as pd
from sklearn.impute import SimpleImputer
from dtreeviz.trees import dtreeviz

model = xgboost_simple_model.model
X_train = xgboost_simple_model.X_train
y_train = xgboost_simple_model.y_train

# sécuriser format 2D
X_train_df = pd.DataFrame(X_train)

imputer = SimpleImputer(strategy="median")
X_train_clean = imputer.fit_transform(X_train_df)

# vérifier shape AVANT conversion
if X_train_clean.shape[1] == 0:
    raise ValueError("0 colonnes après preprocessing → problème sur X_train")

X_train_clean = pd.DataFrame(X_train_clean)

y_train_clean = pd.Series(y_train)

model.get_booster().feature_names = [f"f{i}" for i in range(X_train_clean.shape[1])]

viz = dtreeviz(
    model,
    X_train_clean,
    y_train_clean,
    target_name="target",
    tree_index=0
)

viz