import pandas as pd

# récupérer modèle + encoder
model = xgboost_simple_model.model
le = xgboost_simple_model.label_encoder  # adapte si nom différent

# inverser les labels (juste pour comprendre les classes)
print("Classes :", le.classes_)

# afficher arbre brut du booster
booster = model.get_booster()

tree = booster.get_dump()[0]

print(tree)