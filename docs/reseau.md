# Réseau CNN 1D + GRU, perte, protocole et interface d'entraînement

Étape de préparation terminée le 2026-11-10 : **tout est écrit et testé ; seuls des essais courts ont été faits, aucun entraînement complet n'a été lancé**. Code : `cmapss/model.py`, `cmapss/train.py`, `cmapss/search.py` ; script : `scripts/train.py` ; interface : `viz/entrainement/` ; tests : `tests/test_model.py`, `tests/test_train.py`, `tests/test_search.py`.

**Règle** : aucun entraînement (même court) n'est lancé sans l'accord explicite de l'utilisateur. Le périphérique par défaut est le **CPU de ce portable** (multi-threading) ; CUDA n'est utilisé que par `--device cuda`, sur décision de l'utilisateur seul.

## Réseau (`cmapss/model.py`)

Entrée : une fenêtre `(30 cycles, 20 valeurs)` (17 capteurs normalisés + 3 réglages ; 17 valeurs sans `s6`, `s10`, `s16`).

| Couche | Rôle | Paramètres |
|---|---|---|
| Convolution 1 (Conv1d, noyau 5, 32 canaux) + BatchNorm + ReLU | repère des motifs courts dans le temps | 3 296 |
| Convolution 2 (64 canaux) + BatchNorm + ReLU | combine les motifs | 10 432 |
| GRU (64 unités), dernier état | suit l'évolution sur la fenêtre | 24 960 |
| Tête RUL (Linear 64 → 1) | durée restante, divisée par le plafond | 65 |
| Tête régime (Linear 64 → 1) | logit : « HPC seul » contre « deux modes » | 65 |
| **Total** | | **38 818** |

Petit modèle volontairement (709 moteurs d'entraînement). La **structure** est fixe pendant l'apprentissage ; ce qui évolue, ce sont les poids et les activations, mesurés dans le journal.

## Perte

`perte = MSE(RUL / 150, cible / 150) + λ · BCE(régime)`.

- La RUL est plafonnée (150 par défaut, choix de la baseline ; le plafond fait partie de la recherche) puis divisée par ce plafond : une MSE de 0,03 correspond environ à un RMSE de 26 cycles. **λ doit donc rester petit** pour que la tête régime ne domine pas la perte ; λ fait partie de la recherche (`{0 ; 0,05 ; 0,2}`).
- Le RMSE et le score PHM08 restent mesurés en cycles, contre la **vraie RUL non plafonnée**, sur les fenêtres de validation de vraie RUL 6 à 195 (comme la baseline : comparable).

## Protocole (`scripts/train.py`)

L'essai de 12 époques a montré un surapprentissage dès l'époque 3 (voir plus bas). Le protocole commence donc par une **recherche d'hyperparamètres** large (régularisation, taille du réseau, taux d'apprentissage...), voulue longue : le portable peut tourner des heures ; CUDA n'est envisagé que si l'utilisateur le décide.

| Étape | Commande | Contenu | Entraînements (par défaut) |
|---|---|---|---|
| Essai | `--smoke [--epochs 3] [--preset NOM] [--set dropout=0.3,lr=0.003]` | pli 0, graine 0, journal pour l'interface | 1 |
| **R. Recherche** | `--search [--search-n 60] [--top 8]` | phase 1 « tri » : 6 réglages nommés + 60 tirés au hasard sur les plis 0 et 1 ; phase 2 « confirmation » : les 8 meilleurs (et « aucune ») sur les plis 2, 3 et 4 ; classement sur les 5 plis, **comparé pli par pli à « aucune »** | 132 + jusqu'à 27 |
| B. Validation croisée | `--cv [--preset rNNN]` | meilleur réglage de R, 5 plis × 5 graines (les 5 de la graine 0 sont repris de R) | 20 de plus |
| C. Final | `--final` | tout l'entraînement (scaler « full »), nombre d'époques = médiane des meilleures époques de B, 5 graines, **test évalué une seule fois par graine** ; moyenne des 5 prédictions aussi | 5 |

**Paramètres cherchés** (`cmapss/search.py`, `SPACE`) : dropout `{0 ; 0,1 ; 0,2 ; 0,3 ; 0,5}`, bruit gaussien sur les entrées `{0 ; 0,05 ; 0,1 ; 0,2}`, décroissance des poids (AdamW) `{0 ; 1e-3 ; 1e-2 ; 1e-1}`, taille du réseau `{petit 16/32 canaux et 32 unités ; moyen 32/64 et 64 ; grand 64/128 et 128}`, noyau des convolutions `{3 ; 5 ; 7}`, taux d'apprentissage `{3e-4 ; 1e-3 ; 3e-3}`, taille des lots `{128 ; 256 ; 512}`, poids λ de la tête régime `{0 ; 0,05 ; 0,2}`, plafond de RUL `{125 ; 150 ; 175}`.
**Réglages nommés** (toujours testés) : `aucune` (le réseau de l'essai, référence), `dropout` (0,3), `bruit` (0,1), `poids` (décroissance 1e-2), `petit`, `combinee` (dropout 0,3 + bruit 0,1 + décroissance 1e-2).

- Le tirage est **reproductible** (graine 0, `--search-seed`), sans doublon, et les N premiers réglages ne changent pas si on augmente N. Les réglages tirés sont enregistrés dans les résultats sous les identifiants `r001`, `r002`...
- Le choix se fait sur le RMSE de validation moyen. Si d'autres réglages sont à moins d'une erreur-type du meilleur, le script le dit (une seule graine : ils sont alors indiscernables).
- **`--dry-run`** n'entraîne rien : il affiche le plan (réglages tirés, nombre d'entraînements, durée estimée). À utiliser avant tout lancement.
- Adam (ou AdamW si décroissance des poids), écrêtage de gradient à 1, **arrêt précoce** sur le RMSE de validation (patience 8, 50 époques maximum), meilleurs poids restaurés. Un contrôle (`check`) vérifie que le modèle restauré redonne bien le RMSE de la meilleure époque.
- Graines : `torch`, `numpy` et ordre des lots dépendent de la graine ; `cudnn.deterministic` si CUDA. Deux exécutions sur CPU avec la même graine donnent les mêmes pertes (testé) ; **CPU et GPU ne donneraient pas des nombres identiques**.
- Résultats run par run dans `data/processed/train_results.json` (ignoré par Git) ; un run déjà fait est sauté (**reprise possible** après une interruption, il suffit de relancer la même commande).
- Options de parallélisme : `--workers K` entraînements en parallèle, `--threads N` threads par entraînement. Défaut : 4 × 2 threads.
- `--journal` (étapes B et C) écrit le journal du pli 0, graine 0 (ou de la graine 0 en final) dans `viz/entrainement/journal.jsonl` ; le tableau de bord affiche le réglage utilisé.


### Débit mesuré sur ce portable (8 cœurs logiques, 31,7 Go de RAM)

Mesures faites avec de vrais entraînements de 3 époques (pli 0, 112 708 fenêtres d'entraînement) :

| Configuration | Durée d'une époque par entraînement | Débit global (équivalent époques) |
|---|---|---|
| 1 entraînement × 8 threads | 15,7 s | 15,7 s par époque |
| 4 en parallèle × 2 threads | 27,5 s | **6,9 s par époque** |
| 8 en parallèle × 1 thread | 53 s | 6,6 s par époque |

Le parallélisme donne donc environ **2,3 fois** le débit d'un seul entraînement sur 8 threads ; 4 × 2 est retenu (presque aussi rapide que 8 × 1, moins de mémoire, machine restée utilisable).

**Estimation de la durée** (simulation `--dry-run`, d'après ce débit et **20 époques supposées** par entraînement ; le vrai nombre d'époques n'est connu qu'après les premiers entraînements, et les gros réseaux ou les petits lots sont plus lents) : recherche par défaut (66 réglages) ≈ 5 h pour la phase 1 (132 entraînements) plus ≈ 1 h pour la phase 2 (27), validation croisée ≈ 1 h (20 de plus), final < 1 h. Au total une demi-journée de calcul. Réduire avec `--search-n` (par exemple 20) ou `--top` ; augmenter avec `--search-n 150` si besoin. L'estimation sera corrigée par le script lui-même (reste estimé après chaque entraînement terminé).

### Essais courts déjà faits (avec l'accord de l'utilisateur pour le dernier)

Tous sur le pli 0, λ = 0,05, graine 0 (et 1 à 3 pour le premier), 17 capteurs, plafond 150, sans réglage particulier. La baseline donne **23,94** sur ce pli (24,10 en moyenne sur les 5 plis).

| Essai | RMSE de validation |
|---|---|
| 4 graines, 3 époques chacune | 24,3 à 24,9 (le réseau apprenait encore) |
| **1 graine, 12 époques** (arrêt précoce à l'époque 11, patience 8) | meilleur **24,46 à l'époque 3**, puis 24,9 · 26,2 · 26,3 · 26,8 · 27,7 · 28,2 · 28,1 · **29,1** |

**Constat : le réseau surapprend dès l'époque 3.** L'erreur d'entraînement continue de baisser (MSE normalisée 0,028 → 0,006) pendant que l'erreur de validation remonte (0,0191 → 0,0307). À son meilleur moment (époque 3), il **ne bat pas la baseline** (24,46 contre 23,94 sur ce pli). C'est une seule graine sur un seul pli : une indication forte, pas une preuve. Le journal complet de cet essai est `viz/entrainement/journal.jsonl` ; la vidéo d'essai `reports/videos/entrainement_essai_simple.mp4` le montre honnêtement (ses légendes sont choisies d'après les valeurs du journal : elles disent que l'erreur remonte et que le réseau ne bat pas encore la baseline).

**Conséquence : la recherche d'hyperparamètres (étape R) vient avant tout le reste.** Lancer la validation croisée avec le réseau de l'essai mesurerait un réseau qui, au mieux, égale la baseline. Les régularisations (dropout, bruit sur les entrées, décroissance des poids, réseau plus petit, leur combinaison) et d'autres paramètres (taille, noyau, taux d'apprentissage, lots, λ, plafond) sont donc comparés pli par pli à « aucune ». Le réglage « aucune » est exactement le réseau de cet essai : sa courbe sert de référence.

## Journal d'entraînement (`viz/entrainement/journal.jsonl`)

Écrit ligne par ligne par `cmapss/train.py` (format commun, `cmapss/journal.py`) :

| Événement | Contenu |
|---|---|
| `meta` | architecture et paramètres par couche, configuration, graine, périphérique, threads, **scores de la baseline** (lus dans `baseline_results.json`, jamais saisis à la main) |
| `probe` | moteur de démonstration de la validation (premier moteur FD001) et vraies RUL |
| `layers` | par couche : norme des poids, activations moyennes, part de canaux « éteints », **noyaux de la convolution 1** ; à l'époque 0 (départ) puis à chaque époque |
| `prediction` | prédictions du réseau sur le moteur de démonstration, à l'époque 0 puis à chaque époque |
| `batch` | perte moyenne tous les 50 lots |
| `epoch` | pertes d'entraînement et de validation, RMSE, score PHM08, biais, taux de retard, exactitude du régime, durée |
| `best`, `early_stop`, `final`, `check`, `end` | meilleure époque, arrêt, bilan, contrôle du modèle restauré, fin |

## Interface (`viz/entrainement/`)

- **`live.html`** : tableau de bord d'un seul écran, **mis à jour en direct** (relit le journal chaque seconde, ignore une dernière ligne en cours d'écriture) : état et durée, 4 chiffres clés (RMSE contre baseline, score, régime, durée d'une époque), schéma du réseau avec poids mesurés par couche et poids de la convolution 1, pertes, erreur de validation, prédictions sur un moteur de démonstration. Marche aussi sur un journal terminé. Lancer `python scripts/serve_viz.py --path entrainement/live.html`. Le bouton « Enregistrer » filme l'onglet (Chrome ou Edge ; Échap pour arrêter ; **non testé**, action humaine).
- **`index.html`** : vidéo grand public en 4 scènes (le réseau ; il apprend ; ses prédictions se précisent ; il s'arrête au bon moment et la comparaison à la baseline), une phrase par scène, rejouée depuis le journal avec `viz/lib/core.js`. `python scripts/record_viz.py --page entrainement/index.html` produit `reports/videos/entrainement_simple.mp4`.

**Vérifié** sur de vrais journaux : `live.html` pendant l'entraînement (captures à plusieurs instants, sans erreur console), `index.html` sur le journal complet de l'essai de 12 époques (captures regardées, légendes et chevauchements corrigés, vidéo d'essai rendue). Journal : `seq` contigu, événement `end`, contrôle du modèle restauré vrai.

## Limites

- **λ** est comparé avec une seule graine : si les écarts pli par pli sont sous le bruit, on le dira plutôt que de conclure.
- L'arrêt précoce utilise la validation : l'estimation de validation croisée est légèrement optimiste ; le test, lui, ne sert qu'à l'évaluation finale.
- Étiquette de régime : un proxy par jeu ; la baseline reconnaît mal FD003 et FD004 (49 % et 54 %), le réseau peut ne pas mieux faire.
- La structure du réseau ne change pas pendant l'entraînement : « l'évolution de l'architecture » montrée est celle des poids et des activations.
- CUDA non testé sur l'autre PC (`scripts/check_gpu.py`) : à l'utilisateur de décider.
