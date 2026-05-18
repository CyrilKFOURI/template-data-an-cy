import pandas as pd
from dtreeviz.trees import dtreeviz

# 1. récupérer les données depuis la classe
model = xgboost_simple_model.model
X_train = xgboost_simple_model.X_train
y_train = xgboost_simple_model.y_train

# 2. numpy -> DataFrame
X_train_df = pd.DataFrame(
    X_train,
    columns=[f"f{i}" for i in range(X_train.shape[1])]
)

# 3. y -> Series
y_train_series = pd.Series(y_train)

# 4. (optionnel mais recommandé)
model.get_booster().feature_names = X_train_df.columns.tolist()

# 5. dtreeviz
viz = dtreeviz(
    model,
    X_train_df,
    y_train_series,
    target_name="target",
    tree_index=0
)

viz