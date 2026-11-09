# C-MAPSS : maintenance prédictive

Projet étudiant : entraîner un modèle d'IA pour la **maintenance prédictive** (prédiction de la durée de vie restante, RUL) sur le jeu de données NASA C-MAPSS.

## Contenu du dépôt

| Chemin | Description |
|---|---|
| `data/CMAPSSData.zip` | Archive d'origine téléchargée depuis la NASA |
| `data/raw/` | Fichiers extraits (`train_*`, `test_*`, `RUL_*`, `readme.txt`, article PDF) |
| `docs/datasets.md` | Les deux pages NASA analysées |
| `docs/donnees.md` | Format, statistiques et suffisance des données |
| `docs/article-saxena-2008.md` | Résumé de l'article de référence (PHM08) |
| `docs/plan.md` | Plan du projet et décisions en suspens |
| `docs/journal.md` | Journal chronologique des actions réalisées |

## État actuel

- Données téléchargées et extraites.
- Analyse documentaire terminée (pages NASA, article, statistiques).
- Aucun code de modélisation écrit pour l'instant.
- Environnement Python configuré (`.venv`, voir `requirements.txt` et `docs/plan.md`).

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
