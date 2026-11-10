# C-MAPSS : maintenance prédictive

Projet étudiant : entraîner un modèle d'IA pour la **maintenance prédictive** (prédiction de la durée de vie restante, RUL) sur le jeu de données NASA C-MAPSS.

## Objectif du projet

### Problème posé

Pour un moteur d'avion (turbofan) en service, on dispose de mesures de capteurs enregistrées à chaque cycle (un vol). L'objectif est de **prédire combien de cycles il reste avant la panne** : c'est la RUL (*Remaining Useful Life*, durée de vie utile restante). Une bonne prédiction permet de planifier la maintenance avant la panne, sans retirer un moteur trop tôt ni trop tard.

### Ce que le modèle doit faire

- **Entrées :** l'historique récent d'un moteur (une fenêtre d'environ 30 cycles), avec pour chaque cycle les **capteurs utiles** et les **3 réglages opérationnels (conditions de vol)**.
- **Sortie principale :** la RUL du moteur au dernier cycle observé.
- **Sortie secondaire :** le régime de panne probable, pour que le modèle **devine lui-même le mode de panne** au moment du test, sans qu'on le lui indique.

### Données utilisées

**Les quatre sous-ensembles FD001 à FD004 sont utilisés ensemble** pour l'entraînement, soit 160 359 lignes et 709 moteurs.

| Jeu | Conditions de vol | Modes de panne |
|---|---|---|
| FD001 | 1 | 1 (HPC) |
| FD002 | 6 | 1 (HPC) |
| FD003 | 1 | 2 (HPC et fan) |
| FD004 | 6 | 2 (HPC et fan) |

Le modèle ne reçoit **pas** l'identifiant du jeu en entrée : il doit s'adapter seul aux conditions de vol et au mode de panne.

### Choix de conception arrêtés

| Sujet | Choix |
|---|---|
| Normalisation des capteurs | Par condition de vol |
| Plafond de RUL à l'entraînement | À tester (par exemple 100, 125, 150), garder la meilleure valeur sur la validation |
| Fenêtres plus courtes que 30 cycles | Complétées en répétant le premier cycle |
| Mode de panne | Apprentissage multitâche : une tête RUL et une tête de classification avec étiquette proxy |
| Validation | Validation croisée à 5 plis, découpage par moteur (jamais par ligne) |
| Code | Scripts Python, pas de notebook |
| Matériel | Mise au point sur CPU, entraînement probablement sur un PC avec GPU CUDA |

### Architecture envisagée (recommandation, pas encore validée)

1. **Baseline :** Random Forest ou Gradient Boosting sur des statistiques calculées sur la fenêtre, plus les conditions de vol. Elle sert de point de comparaison.
2. **Modèle principal :** un petit réseau CNN 1D suivi d'un GRU (ou LSTM), avec une tête RUL et une tête de classification du régime de panne.

### Évaluation

- Chaque sous-ensemble est évalué **séparément** sur son propre jeu de test, avec les vraies RUL des fichiers `RUL_FD00x.txt`.
- Deux métriques : le **RMSE** et la **score function asymétrique de PHM08** (les prédictions tardives sont plus pénalisées que les précoces).
- Comme chaque test ne donne qu'une prédiction par moteur (100 pour FD001), le score est bruité : il sera complété par la validation croisée et par plusieurs entraînements avec des graines différentes.

### Livrables prévus

- Des scripts reproductibles (préparation des données, entraînement, évaluation).
- Les résultats chiffrés par jeu et des figures (prédiction contre RUL réelle).
- Cette documentation (`docs/`), qui sert de base au rapport.

### Limites à garder en tête

- Les données sont **synthétiques** (simulateur) : un bon score ici ne garantit pas les mêmes performances sur de vrais moteurs.
- Les fichiers ne donnent le mode de panne que **par jeu**, pas par moteur. La classification du régime est donc un proxy grossier (« HPC seul » contre « deux modes possibles »), pas un diagnostic exact.
- Il y a peu de moteurs indépendants (100 à 260 par jeu) : il faut un modèle de taille modeste et une validation par moteur pour éviter une évaluation trop optimiste.

## Contenu du dépôt

| Chemin | Description |
|---|---|
| `data/CMAPSSData.zip` | Archive d'origine téléchargée depuis la NASA |
| `data/raw/` | Fichiers extraits (`train_*`, `test_*`, `RUL_*`, `readme.txt`, article PDF) |
| `docs/datasets.md` | Les deux pages NASA analysées |
| `docs/donnees.md` | Format, statistiques et suffisance des données |
| `docs/article-saxena-2008.md` | Résumé de l'article de référence (PHM08) |
| `docs/exploration.md` | Résultats de l'exploration des données |
| `scripts/explore.py` | Script d'exploration (figures dans `reports/figures/`) |
| `docs/normalisation.md` | Normalisation par condition de vol, journal d'événements, page et vidéos |
| `cmapss/` | Code commun : chargement (`data.py`), normalisation (`normalisation.py`), plis (`folds.py`), journal (`journal.py`) |
| `scripts/normalise.py` | Plis, scalers par pli, contrôles ; `--journal` écrit le journal de la page |
| `scripts/serve_viz.py`, `scripts/record_viz.py` | Servir la page de visualisation ; produire la vidéo (MP4) |
| `viz/normalisation/` | Page D3 qui rejoue `journal.jsonl` (version simple et version `?mode=tech`) |
| `docs/fenetres.md` | Fenêtres glissantes : règles, effectifs réels, vidéo |
| `cmapss/windows.py`, `scripts/fenetres.py` | Fabrication des fenêtres de 30 cycles ; contrôles et journal de la vidéo |
| `viz/fenetres/`, `viz/lib/` | Page de la vidéo des fenêtres ; cœur commun des pages (`core.js`, style, D3) |
| `tests/` | Tests `pytest` (27) |
| `docs/plan.md` | Plan du projet et décisions en suspens |
| `docs/journal.md` | Journal chronologique des actions réalisées |

## État actuel

- Données téléchargées et extraites.
- Analyse documentaire terminée (pages NASA, article, statistiques).
- Exploration des données faite (`docs/exploration.md`).
- Normalisation par condition de vol faite et testée, avec sa visualisation filmable (`docs/normalisation.md`).
- Fenêtres glissantes faites et testées, avec leur vidéo (`docs/fenetres.md`).
- Pas encore de chargeur par pli, de métriques, de baseline ni de modèle.
- Exigence : normalisation et entraînement doivent être visuels et filmables (voir `docs/plan.md`). Fait pour la normalisation, à faire pour l'entraînement.
- Environnement Python configuré (`.venv`, voir `requirements.txt` et `docs/plan.md`).

## Commandes utiles

```powershell
.\.venv\Scripts\python.exe -m pytest                                  # tests
.\.venv\Scripts\python.exe scripts\normalise.py --journal            # plis, scalers, journal de la page
.\.venv\Scripts\python.exe scripts\fenetres.py --journal              # fenêtres sur les données réelles + journal de la vidéo
.\.venv\Scripts\python.exe scripts\serve_viz.py                      # pages de visualisation (localhost:8765/normalisation/ ou /fenetres/)
.\.venv\Scripts\python.exe -m playwright install chromium             # une fois, pour la vidéo automatique
.\.venv\Scripts\python.exe scripts\record_viz.py                     # vidéo de la normalisation (reports\videos\)
.\.venv\Scripts\python.exe scripts\record_viz.py --page fenetres/index.html   # vidéo des fenêtres
```

## Installation

Il y a deux fichiers selon la machine (PyTorch ne s'installe pas de la même façon avec ou sans GPU NVIDIA) :

| Machine | Fichier |
|---|---|
| Sans GPU NVIDIA (CPU) | `requirements-cpu.txt` |
| Avec GPU NVIDIA (CUDA) | `requirements-cuda.txt` |

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt    # ou requirements-cuda.txt
.\.venv\Scripts\python.exe scripts\check_gpu.py                      # vérifie torch et le GPU
```

`requirements-cuda.txt` vise CUDA 13.4 (`cu134`). Si `nvidia-smi` indique une version CUDA plus ancienne que 13.4 en haut à droite, remplacer `cu134` par `cu132` dans ce fichier. `requirements.txt` seul contient les bibliothèques communes, sans PyTorch.

Les scripts devront choisir le périphérique avec `"cuda" if torch.cuda.is_available() else "cpu"`.

## Source

- Page NASA : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Téléchargement direct : <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
- Licence : non spécifiée sur la page NASA.
