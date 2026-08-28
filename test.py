Bonjour,

Vous trouverez ci-dessous un récapitulatif de l’organisation des fichiers et scripts développés dans le cadre des différents travaux NOVA.

1. NOVA – Scripts et exploration des données

Dans le dossier :

\\FRSHARES0371.france.intra.corp\RISK_Data_Analytics_And_Strategy\01 - NOVA\05 - Stream 4 - Data science\051 - Database\SCRIPTS

se trouvent notamment :

* les scripts de conversion des fichiers GZ vers Parquet ;
* le script de génération des fichiers G4 et G8 ;
* les fichiers/notebooks d’exploration des données NOVA.

L’ensemble des dashboards a été développé en Python avec la bibliothèque Dash.

2. Fleet Monitoring

Les scripts du Fleet Monitoring sont disponibles dans :

\\FRSHARES0371.france.intra.corp\RISK_Data_Analytics_And_Strategy\01 - NOVA\05 - Stream 4 - Data science\052 - Fleet Monitoring\0521 - Scripts

Ce dossier contient plusieurs sous-dossiers :

* Data Generation
* Notebooks
* Output Reader
* Computation

Le dossier Computation contient le script générique qui regroupe l’interface et l’ensemble des calculs nécessaires à la génération des vues.

À noter que ce dashboard peut être relativement lent au lancement, car les calculs sont effectués directement lors de l’exécution du script. Celui-ci récupère également les données directement depuis les fichiers de la base NOVA.

Dans le dossier Data Generation, un script Python permet de générer les fichiers Parquet contenant les résultats des KPI. Il est possible de sélectionner :

* les pays pour lesquels les résultats doivent être générés ;
* la période (dates de début et de fin).

Attention : le script de génération appelle le code présent dans le dossier Computation. Il faut donc bien vérifier et adapter le chemin utilisé pour appeler ce script lors de son exécution.

Le dossier Output Reader contient également un dashboard Dash. Celui-ci permet de reconstruire le dashboard à partir des fichiers Parquet de résultats, sans faire appel directement aux données de la base NOVA. Il utilise uniquement les résultats préalablement générés.

Les outputs du Fleet Monitoring sont disponibles dans :

\\FRSHARES0371.france.intra.corp\RISK_Data_Analytics_And_Strategy\01 - NOVA\05 - Stream 4 - Data science\052 - Fleet Monitoring\0522 - Outputs

La documentation correspondante se trouve dans :

\\FRSHARES0371.france.intra.corp\RISK_Data_Analytics_And_Strategy\01 - NOVA\05 - Stream 4 - Data science\052 - Fleet Monitoring\0520 - Documents

3. Car Models Identification Methodology

Le dossier :

0523 - Car Models Identification Methodology

contient les travaux relatifs à l’identification des modèles de véhicules à partir des données de remarketing/market data.

Le notebook 1_Model_Classification contient la logique d’identification des modèles. Il génère un fichier Parquet permettant d’associer, lorsqu’il est reconnu, chaque modèle NOVA à son modèle correspondant dans les market data.

Le deuxième fichier concerne le segment des véhicules à partir des données de remarketing.

Le troisième fichier permet de tester l’identification des modèles sur l’ensemble de la base.

Il y a également un fichier permettant d’explorer les données et d’effectuer le mapping de la classification PV / LCV sur l’ensemble des pays.

4. Use Case 1

Pour le Use Case 1, l’organisation et la logique sont similaires à celles du Fleet Monitoring :

* un script de génération des résultats ;
* des fichiers Parquet contenant les outputs ;
* un script de lecture des résultats ;
* un script générique contenant les calculs.

Cette organisation permet notamment de séparer la génération des résultats de leur visualisation.

5. Vehicle Customer View

Le Vehicle Customer View, actuellement présent dans le dossier correspondant au Use Case, suit également la même logique :

* un fichier/script de génération des résultats ;
* un fichier/script de lecture des résultats ;
* un script générique contenant les calculs nécessaires.

L’objectif est là aussi de pouvoir générer les résultats en amont puis reconstruire le dashboard à partir des outputs, sans avoir à recalculer l’ensemble des données à chaque consultation.

N’hésitez pas à me contacter si vous avez besoin de précisions supplémentaires sur l’organisation des dossiers ou le fonctionnement des différents scripts.

Bien cordialement,

Cyril
