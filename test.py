import pandas as pd
import json

# Récupérez la chaîne de caractères
json_str = wltp["data_json"].iloc[0]

# Convertissez la chaîne JSON en un objet Python (liste ou dictionnaire)
data = json.loads(json_str)

# Maintenant, créez le DataFrame
tmp = pd.DataFrame(data)

# Vous pouvez ensuite utiliser votre code Plotly
import plotly.express as px
fig = px.line(tmp, x="QTR", y="WLTP_Q_MEAN_VEH_NEW", markers=True)
fig.show()
