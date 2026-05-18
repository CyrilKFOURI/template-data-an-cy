import pandas as pd
from dtreeviz.trees import dtreeviz

model = xgboost_simple_model.model
X_train = pd.DataFrame(xgboost_simple_model.X_train)
y_train = pd.Series(xgboost_simple_model.y_train)

viz = dtreeviz(
    model,
    X_train,
    y_train,
    target_name="target",
    tree_index=0
)

viz