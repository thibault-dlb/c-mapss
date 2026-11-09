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

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Source

- Page NASA : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Téléchargement direct : <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
- Licence : non spécifiée sur la page NASA.
