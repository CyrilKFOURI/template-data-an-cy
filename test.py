CE QUE MONTRE LA VUE 
-----------------------------------------------------------------
- Recherche d'un client par ID_CUSTOMER (ou OBLIGOR_IDENTIFIER), avec
  filtre pays optionnel et filtre date "as of" (COB_DATE) optionnel.
- Carte client à droite : nom, pays, secteur, shared client flag,
  nombre de véhicules, ratings (Group / Counterparty / CLS) et
  exposition totale (LTR + pending orders, dédupliquée par contrat).
- Grille de véhicules à gauche (20 par page) : marque + modèle
  (Market Model matché via models.parquet), statut, date de livraison,
  prix. Filtrable par marque / type carrosserie / statut, triable par
  date de livraison (récent <-> ancien).
- Bouton "More details" sur chaque véhicule -> modal avec le détail
  complet (identification, prix, moteur, carburant/émissions,
  carrosserie, statut + dates contractuelles).
