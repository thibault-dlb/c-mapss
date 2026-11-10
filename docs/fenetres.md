# Fenêtres glissantes

Étape terminée le 2026-11-10. Code : `cmapss/windows.py` ; script : `scripts/fenetres.py` ; page : `viz/fenetres/`.

## Principe

Une fenêtre est une tranche de l'historique récent d'un moteur : les **30 derniers cycles**, avec pour chaque cycle les 17 capteurs normalisés et les 3 réglages de vol, soit un tableau de 30 × 20 valeurs (`FEATURES` dans `cmapss/windows.py`). Son étiquette est la **RUL au dernier cycle** de la fenêtre.

## Règles

| Sujet | Règle |
|---|---|
| Entraînement | Une fenêtre se termine à chaque cycle à partir du 30ᵉ (fenêtres complètes seulement, décidé le 2026-11-10). Aucun remplissage. |
| Test | Une seule fenêtre par moteur : les 30 derniers cycles connus. |
| Moteur de test trop court | Complété au début en répétant le premier cycle (`pad_first`). 17 moteurs de test sont concernés, jusqu'à 11 cycles ajoutés (le plus court en a 19). |
| Étiquette | RUL **non plafonnée** dans les fenêtres. Le plafond (100, 125 ou 150, à choisir sur la validation) s'applique à l'entraînement avec `cap_rul`. |
| Régime de panne | Étiquette proxy par jeu : 0 = « HPC seul » (FD001, FD002), 1 = « deux modes possibles » (FD003, FD004). |
| Frontières | Les fenêtres sont fabriquées moteur par moteur : aucune ne chevauche deux moteurs. |
| Stockage | Rien n'est écrit sur disque : les fenêtres se fabriquent à la demande, avec le scaler du pli. Les 139 798 fenêtres d'entraînement pèsent 336 Mo en `float32`. |

## Effectifs réels (sortie de `scripts/fenetres.py`)

| Jeu | Moteurs d'entraînement | Fenêtres d'entraînement | Moteurs de test | Fenêtres de test |
|---|---|---|---|---|
| FD001 | 100 | 17 731 | 100 | 100 |
| FD002 | 260 | 46 219 | 259 | 259 |
| FD003 | 100 | 21 820 | 100 | 100 |
| FD004 | 249 | 54 028 | 248 | 248 |
| **Total** | **709** | **139 798** | **707** | **707** |

Formule : `n_windows(n_cycles) = n_cycles − 30 + 1`, soit 160 359 − 709 × 29 = 139 798. Les étiquettes d'entraînement vont de 0 à 513 (avant plafonnement), celles de test de 6 à 195 (les vraies RUL de `RUL_FD00x.txt`).

**Les fenêtres voisines ne sont pas indépendantes** : deux fenêtres consécutives partagent 29 cycles sur 30. D'où la validation croisée **par moteur** : dans le pli 0, 568 moteurs (112 708 fenêtres) servent à l'entraînement et 141 (27 090 fenêtres) à la validation, sans moteur en commun.

## Tests

`tests/test_windows.py` (12 tests) : formule du nombre de fenêtres, contenu et étiquette d'une fenêtre, 29 cycles partagés, aucune fenêtre à cheval sur deux moteurs, moteur trop court à l'entraînement, remplissage à l'entraînement et au test, rappel `on_window`, plafond de RUL, données réelles de FD001 (étiquettes de test égales à `RUL_FD001.txt`), cohérence du journal livré.

## Vidéo

Même principe que pour la normalisation (voir `docs/normalisation.md`) : `python scripts/fenetres.py --journal` écrit `viz/fenetres/journal.jsonl`, la page `viz/fenetres/` le rejoue, `python scripts/record_viz.py --page fenetres/index.html` produit `reports/videos/fenetres_simple.mp4` (60 s, 4 scènes, une phrase par scène).

Dans le journal, **chaque fenêtre du moteur de démonstration (FD002_1, 149 cycles, 120 fenêtres) est un événement créé par le vrai code `make_windows`**, pas une animation. Le script vérifie que chaque fenêtre est bien la tranche de la série et que deux fenêtres voisines partagent 29 cycles.

| Scène | Contenu réel |
|---|---|
| 1. Regarder le passé récent du moteur | Les 120 fenêtres du moteur, une par une |
| 2. Une étiquette par fenêtre | RUL des fenêtres 1, 60 et 120 (119, 60 et 0 cycles) |
| 3. Et si le moteur est trop court ? | Moteur de test FD004_141 : 19 cycles, 11 recopiés |
| 4. Entraînement, test et validation | 120 fenêtres pour un moteur d'entraînement contre une seule pour un moteur de test (FD002_1 de test, 258 cycles, RUL 18) ; 12 moteurs réels du pli 0 |

**Illustratif** : le rythme, l'escalier de fenêtres de la scène 4 (une barre par fenêtre réelle, mais décalée pour la lisibilité), le choix des 6 + 6 moteurs d'exemple (les premiers de chaque côté, répartis sur les 4 jeux). Le moteur de test de la scène 4 est un **autre moteur** que celui d'entraînement, malgré le même numéro.

Le capteur montré est `s11` (tendance de dégradation la plus nette), normalisé avec le scaler « full ».

## Limites

- Les fenêtres de ce script utilisent le scaler « full ». À l'entraînement, chaque pli devra utiliser son propre scaler (étape « chargeur par pli »).
- Plafond de RUL, longueur de fenêtre (30) et exclusion éventuelle de `s6`, `s10`, `s16` restent à valider sur la validation croisée.
