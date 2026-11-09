# Les données C-MAPSS

## Fichiers (dans `data/raw/`)

Pour chaque sous-ensemble FD001 à FD004 :

- `train_FD00x.txt` : trajectoires d'entraînement, jusqu'à la panne.
- `test_FD00x.txt` : trajectoires de test, arrêtées avant la panne.
- `RUL_FD00x.txt` : RUL réelle de chaque moteur de test (une valeur par moteur).

Plus `readme.txt` et `Damage Propagation Modeling.pdf` (article de référence).

## Format

Texte, valeurs séparées par des espaces. Une ligne = un cycle opérationnel d'un moteur.

D'après le `readme.txt`, 26 colonnes :

1. numéro d'unité (moteur)
2. temps, en cycles
3. à 5. réglages opérationnels 1 à 3
6. à 26. mesures de capteurs

Le `readme.txt` écrit « sensor measurement 1 ... 26 », mais 5 + 21 = 26 colonnes : il y a donc **21 capteurs** (cohérent avec l'article, Table 2). La mention « 26 mesures de capteurs » sur la page NASA est une coquille.

## Statistiques réelles (comptées sur les fichiers)

| Jeu | Train : lignes / moteurs | Cycles par moteur (train) | Test : lignes / moteurs | Cycles par moteur (test) |
|---|---|---|---|---|
| FD001 | 20 631 / 100 | min 128, moy 206, max 362 | 13 096 / 100 | min 31, moy 131, max 303 |
| FD002 | 53 759 / 260 | min 128, moy 207, max 378 | 33 991 / 259 | min 21, moy 131, max 367 |
| FD003 | 24 720 / 100 | min 145, moy 247, max 525 | 16 596 / 100 | min 38, moy 166, max 475 |
| FD004 | 61 249 / 249 | min 128, moy 246, max 543 | 41 214 / 248 | min 19, moy 166, max 486 |

Écart avec la page NASA pour FD004 : la page annonce 248 (train) / 249 (test), le comptage donne 249 / 248. Origine inconnue, à vérifier si cela compte.

## Y a-t-il assez de données ?

Oui pour un projet étudiant, avec une réserve.

- Ce qui compte est le nombre de **moteurs**, pas de lignes : les lignes d'un même moteur sont très corrélées. Il y a 100 à 260 exemples indépendants par jeu, ce qui est peu.
- Suffisant pour une baseline et un LSTM / CNN 1D : ce benchmark est publié depuis 2008 avec ces tailles.
- Le test ne donne **qu'une prédiction par moteur** (dernière fenêtre) : 100 points pour FD001. Le RMSE de test est bruité, à compléter par une validation croisée.

Précautions :

- Découper train/validation **par moteur** (GroupKFold), jamais par ligne, sinon fuite de données et score de validation trop optimiste.
- Modèles de taille modeste, avec régularisation (dropout, arrêt précoce).
- Entraîner sur plusieurs sous-ensembles à la fois est possible mais à justifier (conditions de vol et modes de panne différents).

## Limites à mentionner dans un rapport

- Données **synthétiques** (simulateur).
- Un seul composant dégradé (HPC) dans l'article, dégradation lisse et exponentielle : les bons scores ne se transfèrent pas tels quels à des données réelles.
