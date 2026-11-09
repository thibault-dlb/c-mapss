# Plan du projet

Objectif : entraîner un modèle d'IA de **prédiction de RUL** (maintenance prédictive) sur C-MAPSS, dans le cadre d'un projet étudiant.

## Environnement de la machine (vérifié)

- Python 3.12.10 installé.
- Aucune bibliothèque de data science détectée dans `pip list` (numpy, pandas, scikit-learn, torch, matplotlib, etc.).
- Pas de GPU NVIDIA (`nvidia-smi` introuvable) : entraînement sur CPU, suffisant pour ce dataset.

## Étapes proposées

1. **Environnement** : venv dans `C:\Users\thibd\C-MAPSS`, installation de pandas, numpy, scikit-learn, matplotlib et PyTorch (CPU).
2. **Exploration (EDA)** : charger les données, repérer les capteurs constants ou inutiles, tracer les dégradations.
3. **Préparation** :
   - normalisation des capteurs (par condition de vol pour FD002 et FD004) ;
   - calcul de la RUL par ligne d'entraînement, avec plafonnement (valeur usuelle d'environ 125 cycles) ;
   - fenêtres glissantes (environ 30 cycles).
4. **Modèles** : baseline simple (régression linéaire ou Random Forest), puis deep learning (LSTM ou CNN 1D) avec PyTorch.
5. **Évaluation** : RMSE sur le test (référence : `RUL_FD00x.txt`) et score asymétrique PHM08, courbes prédiction contre réalité, validation croisée par moteur.
6. **Livrable** : notebook ou scripts propres, figures pour le rapport.

Les valeurs « usuelles » (plafond de RUL, taille de fenêtre, nombre de capteurs inutiles sur FD001) viennent de la pratique courante sur ce benchmark, pas des fichiers du projet : à vérifier pendant l'exploration.

## Décisions en suspens

- Périmètre : commencer par **FD001** puis étendre à FD002 à FD004 ? (proposé, non confirmé)
- ~~Format~~ : **scripts Python, pas de notebook** (décidé le 2026-11-09).

## Architecture envisagée (recommandation, non encore validée)

- **Baseline** : Random Forest ou Gradient Boosting sur des statistiques de fenêtre + les 3 conditions de vol.
- **Modèle principal** : CNN 1D puis GRU (ou LSTM) sur une fenêtre d'environ 30 cycles ; entrées = capteurs utiles + 3 réglages opérationnels ; sortie = RUL plafonnée ; perte MSE ou Huber.
- Petit modèle volontairement (environ 549 moteurs au total) ; Transformer en option seulement.
- ~~Variable « sous-ensemble » en entrée~~ : **rejetée par l'utilisateur**. Au test, le modèle doit deviner lui-même le mode de panne. Piste : apprentissage multitâche (tête RUL + tête « mode de panne »), voir ci-dessous.
- Limite à garder en tête : les fichiers ne donnent pas le mode de panne **par moteur**, seulement par sous-ensemble (FD001 et FD002 : HPC seul ; FD003 et FD004 : HPC et fan). Une étiquette de mode ne peut donc être qu'un proxy grossier (« régime HPC seul » contre « régime à deux modes »).
- Les conditions de vol (1 ou 6) se déduisent des réglages d'entrée eux-mêmes : inutile de les prédire.

## Décisions du QCM (2026-11-09)

- Plafond de RUL : **tester plusieurs valeurs** (par exemple 100, 125, 150) et garder la meilleure sur la validation.
- Normalisation des capteurs : **par condition de vol**.
- Git : **commit et push automatiques à chaque étape terminée** (branche `main`).
- Sous-ensemble : pas d'identifiant en entrée (voir plus haut).

## Décisions prises

- **Entraînement sur les 4 sous-ensembles (FD001 à FD004) ensemble**, avec les **conditions de vol (3 réglages opérationnels) en entrée du modèle** (décision de l'utilisateur, 2026-11-09). Constat sur les fichiers : FD001 et FD003 ont une seule condition (réglages quasi constants autour de 0 / 0 / 100) ; FD002 et FD004 ont exactement 6 conditions discrètes (altitude, Mach, TRA) : (0, 0, 100), (10, 0,25, 100), (20, 0,70, 100), (25, 0,62, 60), (35, 0,84, 100), (42, 0,84, 100). L'ordre altitude / Mach / TRA des colonnes 3 à 5 est déduit des plages de l'article, pas écrit explicitement dans le `readme.txt`.

- Dépôt GitHub : <https://github.com/thibault-dlb/c-mapss>, branche `main`.
- Les données sont versionnées dans le dépôt, y compris `CMAPSSData.zip` (doublon de `raw/`, retirable si on veut alléger).
