# Normalisation par condition de vol

Étape terminée le 2026-11-10. Code : `cmapss/` ; scripts : `scripts/normalise.py`, `serve_viz.py`, `record_viz.py` ; page : `viz/normalisation/`.

## Méthode

- **Capteurs** : z-score `(x − μ) / σ`, avec μ et σ calculés **par condition de vol** (6 conditions, communes aux 4 jeux), tous jeux confondus. Le jeu n'étant pas connu au test, on ne peut pas normaliser par jeu.
- **17 capteurs** gardés (`SENSORS_KEPT` dans `cmapss/data.py`) ; `s1`, `s5`, `s18`, `s19` sont écartés (constants dans chaque condition).
- **Réglages** (altitude, Mach, TRA) : min-max sur [0, 1] avec les bornes de l'entraînement. Ils restent en entrée du modèle.
- **Capteur constant dans une condition** (cas de `s16`) : écart-type remplacé par 1, valeur normalisée 0. La liste est dans `ConditionScaler.flat`.
- **Statistiques par pli** : un `ConditionScaler` ajusté sur les moteurs d'entraînement de chaque pli, appliqué au pli de validation. Un scaler « full » (tout l'entraînement) sert au test final.
- **Plis** : 5 plis, découpage par moteur (`StratifiedGroupKFold`, graine 42, stratifié par jeu). Chaque pli contient 20 moteurs de FD001, 52 de FD002, 20 de FD003 et 49 ou 50 de FD004 en validation.
- **Calcul incrémental** : μ et σ sont calculés par blocs de 2 000 lignes, dans l'ordre des données, avec la fusion de statistiques partielles de Chan. C'est le code réellement utilisé par `fit`, et il égale le calcul direct de pandas (écart maximal 3,6e−12, vérifié à chaque génération du journal et par test).

## Fichiers produits

| Fichier | Contenu | Dans Git |
|---|---|---|
| `data/processed/folds.csv` | Pli de chaque moteur (`engine_id`, `subset`, `unit`, `fold`) | non (régénérable) |
| `data/processed/scalers.json` | Statistiques des 5 plis et du scaler « full » | non (régénérable) |
| `viz/normalisation/journal.jsonl` | Journal d'événements, source de la page et des vidéos | oui |
| `reports/videos/*.mp4` | Vidéos | non (lourdes, régénérables) |

Aucune copie normalisée des données n'est enregistrée : l'entraînement appliquera le scaler du pli à la volée.

## Contrôles (sortie de `normalise.py`)

- Entraînement : moyenne normalisée ≈ 0 (≤ 2e−11) et écart-type 1,000 dans chaque condition, pour chaque pli.
- Validation : la moyenne s'écarte de 0 jusqu'à 0,17 à 0,44 selon le pli, l'écart-type va de 0,72 à 2,00. Les plus gros écarts viennent de `s6` et `s16` (voir `exploration.md`).
- Test : 104 897 lignes, 707 moteurs, aucune condition inconnue, aucun NaN.
- `pytest` : 15 tests (conditions, plis, moyenne et écart-type, aller-retour JSON, condition inconnue, calcul incrémental, étapes de transformation).

## Le journal d'événements

La page ne calcule rien : elle rejoue `viz/normalisation/journal.jsonl`, écrit par `normalise.py --journal`. Une ligne par étape réellement exécutée, avec `seq` (contigu, la page refuse un journal tronqué) et `t` (secondes depuis le début du calcul).

| Événement | Contenu |
|---|---|
| `read` (×4) | Lignes et moteurs de chaque jeu |
| `conditions` | Les 6 conditions, leurs lignes, les jeux qui les utilisent |
| `settings_sample` | 3 000 cycles tirés au hasard (graine 0) : réglages, condition, jeu |
| `stats_begin`, `stats_update` (×81), `stats_final` | μ, σ et effectif par condition après chaque bloc ; valeurs finales des 17 capteurs |
| `check` | Écart avec le calcul direct de pandas |
| `engine_states` | Moteur FD002_1, 21 capteurs : brut, x − μ, (x − μ) / σ ; droite d'approximation (moindres carrés) ; moyenne glissante |
| `histogram` (×2) | Histogrammes de `s7` par condition, bruts puis normalisés |

Le journal ne couvre que le scaler « full ». Les scalers des plis utilisent le même code, sans journal.

## Page et vidéos

```powershell
.\.venv\Scripts\python.exe scripts\normalise.py --journal     # régénère le journal
.\.venv\Scripts\python.exe scripts\serve_viz.py               # page : http://localhost:8765/normalisation/index.html
.\.venv\Scripts\python.exe scripts\record_viz.py              # vidéo grand public -> reports\videos\normalisation_simple.mp4
.\.venv\Scripts\python.exe scripts\record_viz.py --mode tech  # version détaillée -> reports\videos\normalisation_technique.mp4
```

- **Mode simple** (défaut, 59 s, 4 scènes) : une phrase par scène, pas de formule. AVANT, puis le nom du traitement, puis APRÈS.
- **Mode technique** (`?mode=tech`, 86 s, 5 scènes) : calcul de μ et σ bloc par bloc, formule, traçabilité.
- **Cœur commun** : la lecture du journal, la chronologie, les contrôles et l'enregistrement sont dans `viz/lib/core.js` ; chaque page (`viz/normalisation/app.js`, `viz/fenetres/app.js`) ne contient que ses scènes. Le découpage a été vérifié sur 30 images (modes simple et technique) : identiques à l'octet près.
- **Rendu automatique** : `record_viz.py` rejoue le journal image par image (Playwright + ffmpeg). Deux rendus du même instant sont identiques à l'octet près (3 images comparées).
- **Bouton « Enregistrer »** de la page : capture l'onglet en temps réel (Chrome ou Edge). **Non testé**, il demande une action humaine.
- Il faut installer Chromium une fois : `.\.venv\Scripts\python.exe -m playwright install chromium`.

## Réel et illustratif

**Réel** : toutes les valeurs affichées, le calcul de μ et σ bloc par bloc, les histogrammes, les capteurs constants, la droite d'approximation.

**Illustratif** (animation, pas un calcul) :
- les glissements entre « avant » et « après » ;
- le rythme (la page choisit la durée de chaque étape) ;
- le réarrangement de la grille de 21 à 17 capteurs ;
- l'échantillon de 3 000 points de la scène 1 (pas les 160 359 cycles) ;
- chaque petite courbe de la scène 4 est recadrée sur son propre minimum et maximum, avant et après : les hauteurs ne se comparent pas d'une case à l'autre.

La version simple n'affiche qu'une seule mention en fin de dernière scène ; la version technique mentionne le journal et l'étiquette « transition visuelle » partout.

## Limites

- La droite d'approximation de la scène 2 a un **R² de 0,17** sur ce moteur et ce capteur bruité : elle montre le sens de la baisse, pas sa forme (la vraie dégradation accélère vers la fin).
- Avant normalisation, 99 à 100 % de la variation brute des capteurs (FD002 et FD004) vient du changement de condition de vol, sauf `s16` (90 %) et `s14` (97 %). C'est pourquoi tous les graphes changent à la normalisation.
- `s6` et `s16` rendent les statistiques instables d'un pli à l'autre : **ablation à faire** (retirer `s6`, `s10`, `s16`, comparer sur la validation).
- Les moteurs de test n'ont pas servi à ajuster le moindre paramètre : le scaler « full » n'est ajusté que sur l'entraînement.
