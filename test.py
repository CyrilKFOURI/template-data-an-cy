Voici le système de scoring, de façon précise et compréhensible.

- On commence le score à **0** pour chaque paire de modèles (un dans chaque table).

- On ajoute **+100** si les deux modèles nettoyés sont **exactement identiques** (`model_1 == model_2`).

- On regarde les **mots (tokens) en commun** entre les deux modèles :
  - Si le mot fait partie des mots “faibles” (`CLASS`, `SERIES`, `SERIE`, `MODEL`, `NEW`, `PHASE`, `TYPE`, `MY`), on **n’ajoute rien**.
  - Sinon, si le mot est **uniquement composé de chiffres** :
    - Si le nombre est **“fort”** (format type `Q5`, `X3`, ou un nombre de 3 ou 4 chiffres, par exemple `320`, `2008`), on ajoute **+15**.
    - Sinon (nombre plus simple, par exemple `50`), on ajoute **+1**.
  - Sinon, si le mot contient **à la fois des lettres et des chiffres** (ex. `A180`, `E200`), on ajoute **+15**.
  - Sinon (mot purement alphabétique normal, ex. `CLIO`, `GOLF`), on ajoute **+10**.

- Ensuite, on regarde si un modèle est **inclus** dans l’autre :
  - Si le texte du modèle 2 (nettoyé) est **contenu** dans le texte du modèle 1 (nettoyé) et que ce texte fait **plus de 2 caractères**, on ajoute **+20**.

- Si les deux modèles **n’ont aucun mot en commun** (intersection de tokens vide), on **pénalise** en enlevant **20 points** (score −= 20).

- Pour chaque modèle du premier tableau, on garde le modèle du second tableau qui a le **score le plus élevé**.
- Si ce meilleur score est **supérieur ou égal** au seuil (par défaut **10**), on le considère comme un match et on stocke :
  - le modèle trouvé,
  - le score.
- Si le meilleur score est **en dessous de 10**, on considère qu’il **n’y a pas de match** fiable pour ce modèle.

Tu veux que je t’en fasse une version en une seule phrase math (type formule) pour mettre dans un doc technique ?

Sources

