
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Initialisation
label_encoders = {}
scaler = StandardScaler()

def prepare_data(df, numerical_features, categorical_features):
    data = df.copy()
    
    # 1. Scale numérique
    if len(numerical_features) > 0:
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    # 2. Encode catégoriel (un encodeur par colonne)
    for c in categorical_features:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c])
        label_encoders[c] = le
        
    return data

# Usage
# df_prepared = prepare_data(df, numerical_features, categorical_features)
# model.fit(df_prepared, target)



def print_readable_tree(model, numerical_features, label_encoders):
    booster = model.get_booster()
    trees = booster.get_dump()
    tree = trees[0] # Premier arbre
    
    # Remplacer les noms de colonnes encodées (f0, f1...) par les vrais noms
    # XGBoost utilise 'f' + index de colonne
    for i, col_name in enumerate(numerical_features + list(label_encoders.keys())):
        tree = tree.replace(f'f{i}', col_name)
    
    # Remplacer les valeurs encodées pour les catégories
    for col_name, le in label_encoders.items():
        for i, class_name in enumerate(le.classes_):
            # On remplace uniquement si c'est une comparaison précise
            tree = tree.replace(f'<{i}.', f'<{class_name} (')
            tree = tree.replace(f'={i}', f'={class_name}')
            
    print(tree)

# Appel
print_readable_tree(model, numerical_features, label_encoders)