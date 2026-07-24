Use Case 1 Précalculé : ce que fait l'interface et comment l'intégrer

Ce que ça fait

Un pipeline en deux temps.
• Un script offline (generate_use_case_1_precomputed.py, en Python) lit les données NOVA brutes pour un ou plusieurs pays et une période donnée, les enrichit (Market Model, Market Body Type), et écrit un seul fichier Parquet.
• Une interface lit ce fichier et affiche une heatmap interactive du portefeuille de véhicules, croisé sur deux dimensions, avec un mode simulation pour ajouter ou retirer des véhicules.

Le script de génération reste tel quel, il tourne offline, en Python, indépendamment du langage de l'interface. Ce document décrit le comportement de l'interface pour pouvoir la réécrire dans un autre langage ou framework.

Le contrat de données, le fichier Parquet

Chemin produit par le script : precomputed_use_case_1/use_case_1_data.parquet. C'est une table plate, une ligne par véhicule (contrat), lisible depuis n'importe quel langage ayant une librairie Parquet, Python avec pandas, Java, .NET, JavaScript.

Colonnes principales et leur usage.
• COB_DATE, date. Sert au filtre de période, mensuel, trimestriel ou annuel, dérivé de cette date.
• ID_CONTRACT, VEHICLE_ID, ID_QUOTATION, entiers. Forment ensemble la clé unique d'un véhicule. Il faut dédupliquer sur ces trois colonnes ensemble avant tout comptage.
• COUNTRY, NOVA_ASSET_STATUS, BIKE_OR_CAR, texte. Ce sont des filtres.
• BRAND_UPDATE, MARKET_MODEL, MARKET_BODY_GROUP, texte. Filtres, et aussi axes possibles de la heatmap. Market Model et Market Body Group viennent de l'enrichissement, pas de la donnée brute.
• POWER_CATEGORY, CO2_BUCKET, texte. Axes possibles.
• ARVAL_INDUSTRY_CODE_CLS_DESCRIPTION, texte. Axe possible, le secteur d'activité du client.
• GROUP_RATING, COUNTERPARTY_RATING, CLS_GROUP_RATING, texte ou nombre. Axes possibles, la notation crédit. Ils ont un tri spécial, voir plus bas.
• VEHICLE_PRICE_EUR, nombre. Sert à la métrique Vehicle Price.
• EXPOSURE_AMOUNT_LTR, PENDING_ORDERS, nombre. Composent la métrique Exposure.
• ID_CUSTOMER, texte. Clé client, utilisée pour dédupliquer l'Exposure.

Comportement fonctionnel

Filtres, tous optionnels et combinables entre eux.
• Country.
• Asset Status, soit ALL, soit une valeur précise.
• Brand, sélection multiple.
• Market Model, sélection multiple, dépend de la Brand choisie.
• Body Type, sélection multiple.
• Période, granularité mensuelle, trimestrielle ou annuelle, plus une valeur choisie parmi celles présentes dans les données.

Deux axes libres. L'utilisateur choisit un champ pour les lignes et un champ pour les colonnes, parmi la même liste, Brand, Power Category, CO2 Bucket, Industry, Group Rating, Counterparty Rating, CLS Rating. Un champ déjà choisi sur un axe disparaît des choix disponibles pour l'autre axe.

Une métrique à choisir parmi trois.
• Volume, le nombre de véhicules distincts, dédupliqués sur la clé unique, par cellule de la table croisée.
• Exposure, calculée en trois étapes dans cet ordre précis. D'abord EXPOSURE_AMOUNT_LTR plus PENDING_ORDERS pour chaque ligne. Ensuite dédupliquer par ID_CUSTOMER, en gardant la ligne la plus récente selon COB_DATE. Enfin sommer par cellule. L'ordre compte, il faut dédupliquer avant de sommer, jamais l'inverse.
• Vehicle Price, la somme de VEHICLE_PRICE_EUR par cellule, après déduplication sur la clé unique.

La heatmap affiche la table croisée résultante, paginée à quinze lignes par page, avec une ligne et une colonne Total.

Les champs de notation ont un tri spécial. Les valeurs manquantes deviennent une catégorie NR qui apparaît toujours en premier. Ensuite le tri se fait par grade croissant. Pour Group Rating et Counterparty Rating l'ordre est par exemple zéro un, puis zéro deux moins, puis zéro deux, puis zéro deux plus, et ainsi de suite. Pour CLS Rating, qui est numérique, l'ordre est simplement un, deux, trois, et ainsi de suite.

La simulation, ajout et retrait de véhicules

C'est le point le plus important à répliquer correctement.

Principe général. Il faut un journal d'actions ordonné, pas deux listes indépendantes pour les ajouts et les retraits. Chaque ajout ou retrait est une entrée ajoutée à la fin du journal, avec un type, add ou remove, et ses détails. Le portefeuille courant est toujours recalculé en rejouant tout le journal, dans l'ordre, par dessus le portefeuille original qui reste figé et n'est jamais modifié en place.

L'algorithme, en résumé.
• Partir d'une copie du portefeuille original.
• Pour chaque action du journal, dans l'ordre d'ajout au journal.
• Si l'action est un ajout, générer les nouvelles lignes synthétiques avec les champs choisis par l'utilisateur, et les ajouter au portefeuille courant.
• Si l'action est un retrait, parmi les lignes du portefeuille courant qui correspondent aux critères choisis, brand, model, body type, power category, CO2 bucket, en retirer le nombre demandé.

Parce que le retrait s'applique au portefeuille courant, déjà modifié par les actions précédentes, un retrait peut légitimement supprimer des véhicules qu'un ajout précédent vient tout juste de créer. L'ordre des actions a donc un effet réel sur le résultat final.

Pour l'ajout, l'utilisateur choisit explicitement la Brand, le Model, le Body Type, la Power Category, le CO2 Bucket, un client, soit existant et recherché par identifiant, soit nouveau avec industrie et notations saisies à la main, et un prix. Les autres champs du véhicule synthétique, carburant, classe, et le reste, sont remplis avec la valeur la plus fréquente observée pour cette Brand et ce Model dans les données réelles. Rien n'est tiré au hasard.

Pour le retrait, les mêmes champs sont à choisir, Brand, puis Model et Body Type, puis Power Category et CO2 Bucket. Chaque option proposée à l'utilisateur est filtrée pour ne jamais mener à zéro véhicule disponible, un champ n'apparaît comme choix possible que s'il reste au moins un véhicule qui a aussi tous les champs suivants renseignés. La quantité à retirer est plafonnée au nombre de véhicules réellement disponibles dans le portefeuille courant pour la combinaison choisie.

Le résultat affiché comprend trois heatmaps. La première montre le portefeuille Original, celui de départ, jamais modifié. La deuxième montre le Portefeuille Courant, après avoir rejoué tout le journal. La troisième montre le Delta, la différence entre les deux, colorée en rouge ou en vert selon le sens du changement.

Intégrer dans un autre langage

Trois étapes.
• Le script generate_use_case_1_precomputed.py continue de tourner en Python, offline, pour produire le fichier Parquet. Rien à changer de ce côté.
• L'interface, dans le langage cible, doit lire ce fichier Parquet, reproduire la logique de filtrage, de table croisée et de métrique décrite plus haut, et reproduire l'algorithme de replay ordonné pour la simulation. Le journal d'actions peut être gardé en mémoire côté serveur ou côté client, peu importe, ce qui compte c'est qu'il soit rejoué dans l'ordre à chaque affichage.
• Pour changer de périmètre, un autre pays ou une autre période, il suffit de relancer le script Python avec d'autres paramètres et de régénérer le fichier Parquet. Aucune logique côté interface n'a besoin d'être modifiée.
