Here is the scoring system, clearly described in English.

- We start the **score at 0** for each pair of models (one from each table).

- We add **+100** if the two cleaned model strings are **exactly identical** (`model_1 == model_2`).

- Then we look at the **common tokens (words)** between the two cleaned models:
  - If a token is in the list of **weak tokens** (`CLASS`, `SERIES`, `SERIE`, `MODEL`, `NEW`, `PHASE`, `TYPE`, `MY`), we **do not add anything** for that token.
  - Otherwise, if the token is **only digits**:
    - If it is considered a **strong numeric token** (matches patterns like `Q5`, `X3`, or a 3–4 digit number such as `320`, `2008`), we add **+15**.
    - Otherwise (simpler numbers, e.g. `50`), we add **+1**.
  - Otherwise, if the token contains **both letters and digits** (e.g. `A180`, `E200`), we add **+15**.
  - Otherwise (a normal alphabetic word, e.g. `CLIO`, `GOLF`), we add **+10**.

- We also check for **substring inclusion**:
  - If the cleaned text of model 2 is **contained inside** the cleaned text of model 1 and its length is **greater than 2**, we add **+20**.

- If there are **no common tokens at all** between the two models, we **penalize** the pair by **subtracting 20 points** (score −= 20).

- For each model in the first table, we:
  - compute this score against all candidate models of the same brand in the second table,
  - keep the candidate with the **highest score** as the best match.

- If this best score is **greater than or equal to the threshold** (default **10**), we accept it as a **match** and store:
  - the matched model,
  - the associated score.
- If the best score is **below 10**, we consider that there is **no reliable match** for that model.

Sources
