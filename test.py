Le dataset utilisé pour ce KPI contient déjà plusieurs colonnes de filtres préparées en amont afin de faciliter la création dynamique du dashboard et des tables.

Colonnes principales disponibles dans l’output :

COUNTRY

BRAND_UPDATE

YEAR

PERIOD

VOLUME

TOTAL

SHARE

SOURCE_FILTER

COUNTRY_FILTER

STATUS_FILTER

VARIABLE_FILTER

TOP_N_FILTER

PERIOD_MODE_FILTER

OEM_UPDATE

CO2_BUCKET

BEV

VOLUME_YTD

TOTAL_YTD

SHARE_YTD

Make

REG_TYPE_FILTER

OWNER_FILTER

Make Group


Le premier filtre du dashboard sera le filtre SOURCE, avec deux choix possibles :

Portfolio

Market


En fonction de la valeur sélectionnée dans ce filtre, les autres filtres affichés dans l’interface changeront dynamiquement.

Si l’utilisateur sélectionne Portfolio, les filtres affichés seront :

PORTFOLIO STATUS

PORTFOLIO VARIABLE

PERIOD

TOP N


Si l’utilisateur sélectionne Market, les filtres affichés seront :

MARKET REGISTRATION

MARKET VARIABLE

MARKET OWNER

TOP N


Les colonnes utilisées pour les filtres Market sont notamment :

Make

REG_TYPE_FILTER

OWNER_FILTER


Le filtre MARKET REGISTRATION devra permettre une sélection multiple.
L’utilisateur pourra donc cocher plusieurs valeurs simultanément (par exemple deux types d’enregistrement différents) sans être limité à une seule sélection.