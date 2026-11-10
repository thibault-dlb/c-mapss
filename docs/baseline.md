# Chargeur par pli, métriques et baseline

Étape terminée le 2026-11-10. Code : `cmapss/loader.py`, `cmapss/metrics.py`, `cmapss/baseline.py` ; scripts : `scripts/baseline.py`, `scripts/baseline_journal.py` ; pages : `viz/metriques/`, `viz/baseline/` ; résultats chiffrés : `data/processed/baseline_results.json` (ignoré par Git, se régénère avec `scripts/baseline.py`).

But : avoir un **score de référence** avant d'entraîner le réseau, et trancher deux questions ouvertes (plafond de RUL, retrait de `s6`, `s10`, `s16`).

## 1. Chargeur par pli (`cmapss/loader.py`)

- `fold_data(k)` : scaler ajusté **sur les seuls moteurs d'entraînement du pli** (aucune fuite), fenêtres d'entraînement, fenêtres de validation et un masque « vraie RUL entre 6 et 195 » (`RUL_TEST_MIN`, `RUL_TEST_MAX`, relevées sur le test réel).
- `load_test_data()` : scaler « full » (tout l'entraînement) et une fenêtre par moteur de test, avec la vraie RUL de `RUL_FD00x.txt`.
- `features_without(...)` et `SENSORS_UNSTABLE` (`s6`, `s10`, `s16`) pour l'ablation de capteurs.
- `to_torch_dataset` / `make_loader` : lots PyTorch pour le futur réseau (testés sur de petites données).
- Mémoire : environ 340 Mo de fenêtres par pli, libérés entre deux plis.

## 2. Métriques (`cmapss/metrics.py`)

| Mesure | Définition |
|---|---|
| RMSE | Racine de l'erreur quadratique moyenne, en cycles |
| Score PHM08 | `d = RUL estimée − RUL réelle` ; `exp(−d/13) − 1` si `d < 0` (trop tôt), `exp(d/10) − 1` si `d ≥ 0` (trop tard) ; somme sur les moteurs |
| Biais | Erreur moyenne signée (négatif = trop tôt) |
| Taux de retard | Part des prédictions trop tardives |

**Correction d'une erreur de la documentation** : `docs/article-saxena-2008.md` avait les constantes inversées (10 pour le trop tôt). La branche tardive doit avoir la constante la plus petite (10), donc la pénalité la plus forte. **Réserve** : l'équation 11 du PDF n'est pas lisible à l'extraction du texte ; l'attribution repose sur le texte de l'article (« les prédictions tardives sont plus pénalisées ») et sur la littérature (article PHM Society n°2348, arXiv 2410.03134), pas sur une lecture directe de l'équation. Les constantes sont nommées (`A_EARLY = 13`, `A_LATE = 10`) et un test vérifie la propriété.

## 3. Baseline (`cmapss/baseline.py`)

- Variables : pour chacune des 20 colonnes de la fenêtre, 6 statistiques (moyenne, écart-type, minimum, maximum, dernière valeur, pente par moindres carrés), soit 120 variables.
- Modèle : `HistGradientBoostingRegressor` (200 itérations, taux 0,1, sans arrêt précoce interne : un découpage aléatoire des lignes mélangerait des fenêtres voisines d'un même moteur).
- Un classifieur du même type prédit le régime de panne (proxy), comme point de comparaison pour la sortie secondaire du réseau.

## 4. Protocole

- **Validation croisée** à 5 plis par moteur (`make_folds`), 2 jeux de capteurs (17, ou 14 sans `s6`, `s10`, `s16`) × 5 plafonds (100, 125, 150, 175, 200) = 50 ajustements.
- **Validation** : fenêtres des moteurs de validation dont la vraie RUL est entre 6 et 195 (plage du test), mesurées contre la **vraie RUL non plafonnée**, donc comparables d'un plafond à l'autre.
- **Choix** : RMSE moyen sur les 5 plis. Le score PHM08 est rapporté mais ne sert pas à choisir (quelques valeurs extrêmes le dominent : écart-type entre plis de l'ordre de sa moyenne).
- **Test** : évalué **une seule fois**, avec la configuration retenue, sur tout l'entraînement. Aucun réglage n'en dépend.
- La grille a d'abord été 100, 125, 150 ; l'optimum était au bord (150), donc 175 et 200 ont été ajoutés (accord de l'utilisateur). L'optimum est bien à l'intérieur : 150.

## 5. Résultats

### Validation croisée (moyenne ± écart-type entre les 5 plis)

| Capteurs | Plafond | RMSE | Score PHM08 | Biais |
|---|---|---|---|---|
| 17 | 100 | 34,76 ± 0,57 | 3 630 213 ± 2 677 353 | −17,4 |
| 17 | 125 | 26,94 ± 0,47 | 1 946 012 ± 1 778 594 | −8,1 |
| **17** | **150** | **24,10 ± 0,47** | 2 522 673 ± 2 857 769 | −1,9 |
| 17 | 175 | 24,20 ± 0,50 | 2 469 622 ± 2 349 727 | +2,1 |
| 17 | 200 | 25,58 ± 0,54 | 3 607 117 ± 3 319 134 | +4,5 |
| 14 | 100 | 34,80 ± 0,53 | 3 412 925 ± 2 097 717 | −17,4 |
| 14 | 125 | 27,01 ± 0,44 | 1 915 076 ± 1 705 469 | −8,2 |
| 14 | 150 | 24,14 ± 0,49 | 2 043 212 ± 1 991 924 | −1,9 |
| 14 | 175 | 24,27 ± 0,56 | 2 747 033 ± 2 707 273 | +2,1 |
| 14 | 200 | 25,53 ± 0,53 | 3 614 151 ± 3 279 083 | +4,5 |

### RMSE par tranche de vraie RUL (17 capteurs, moyenne sur les plis)

| Plafond | 6-49 | 50-99 | 100-149 | 150-195 |
|---|---|---|---|---|
| 100 | 9,2 | 14,2 | 32,1 | 73,3 |
| 125 | 10,0 | 19,6 | 20,7 | 54,5 |
| 150 | 10,4 | 24,1 | 19,8 | 41,5 |
| 175 | 10,8 | 26,6 | 23,5 | 34,7 |
| 200 | 11,0 | 28,3 | 27,4 | 33,3 |

### Décisions

- **Capteurs** : 17 contre 14, l'écart (0,04 à 0,07) est très inférieur à l'écart-type entre plis (environ 0,5). **Aucune différence mesurable** : le choix automatique garde les 17, mais retirer `s6`, `s10`, `s16` ne coûte rien non plus. Pour le réseau, on peut garder les 17 par défaut et refaire l'ablation avec lui.
- **Plafond** : 150 est le meilleur, mais 175 est à 0,1 près (bien en dessous de l'écart-type de 0,5) : **150 et 175 sont indiscernables**. Un plafond bas fait prédire trop tôt (biais −17 à 100) ; un plafond haut dégrade les RUL moyennes (50-99) pour mieux traiter les RUL élevées (150-195).
- **Réserve sur le critère** : la validation compte toutes les fenêtres de RUL 6 à 195, alors que le test n'a qu'une fenêtre par moteur, surtout à RUL faible ou moyenne. Les plafonds élevés, qui aident surtout sur les RUL rares dans le test, sont peut-être favorisés. Le test (ci-dessous) ne sert pas à trancher, mais il n'est pas contradictoire.

### Test (une fenêtre par moteur, configuration retenue : 17 capteurs, plafond 150)

| Jeu | n | RMSE | Score PHM08 | Biais | RMSE d'une prédiction constante | Régime exact |
|---|---|---|---|---|---|---|
| FD001 | 100 | 15,91 | 590 | +5,0 | 45,04 | 92,0 % |
| FD002 | 259 | 23,67 | 4 307 | −3,3 | 55,04 | 90,3 % |
| FD003 | 100 | 16,61 | 610 | +6,7 | 44,97 | 49,0 % |
| FD004 | 248 | 23,23 | 3 389 | −2,0 | 54,89 | 54,0 % |
| **Total** | **707** | **21,67** | **8 897** | **−0,3** | **52,34** | **72,0 %** |

La baseline bat nettement une prédiction constante (la RUL moyenne) sur les quatre jeux. **C'est le score à battre pour le réseau** : RMSE 21,67 au total, et par jeu 15,9 / 23,7 / 16,6 / 23,2.

**Régime de panne (point d'attention)** : l'exactitude globale de 72 % est trompeuse. Les jeux « HPC seul » (FD001, FD002) sont reconnus à plus de 90 %, mais FD003 et FD004 (deux modes) sont reconnus à peine plus d'une fois sur deux (49 % et 54 %). Comme le régime 0 représente 51 % des moteurs de test, 72 % n'est qu'un peu au-dessus d'un classifieur trop prudent. L'étiquette est un proxy par jeu (voir `fenetres.md`) : en fin de vie, un moteur FD003 ressemble beaucoup à un moteur FD001. À garder en tête pour la tête de classification du réseau.

## 6. Limites

- Le test n'a qu'une prédiction par moteur : le score PHM08 y est bruité (590 à 4 307 selon le jeu). Les écarts de quelques cycles de RMSE ne sont pas significatifs.
- Fenêtres voisines très corrélées : l'écart-type entre plis (non entre fenêtres) est la bonne mesure d'incertitude de la validation.
- La baseline n'utilise ni graines multiples ni arrêt précoce ; une seule graine (0).
- Pas de réglage fin des hyperparamètres du Gradient Boosting (200 itérations, taux 0,1) : le résultat est un point de repère, pas un optimum.

## 7. Vidéos

Même principe que les précédentes (journal d'événements rejoué par la page, rien d'inventé) : `python scripts/baseline.py --journal` écrit `viz/metriques/journal.jsonl` et `viz/baseline/journal.jsonl`, `python scripts/record_viz.py --page metriques/index.html` et `--page baseline/index.html` produisent `reports/videos/metriques_simple.mp4` (56 s) et `baseline_simple.mp4` (60 s), 4 scènes chacune.

| Vidéo | Scènes | Contenu réel |
|---|---|---|
| Métriques | 1. erreur par moteur ; 2. trop tôt n'est pas trop tard ; 3. le score additionne les pénalités ; 4. deux chiffres pour juger | 12 moteurs réels de validation (pli 0) ; courbe calculée par la vraie `phm_penalty` ; pénalités réelles ; RMSE et score du test |
| Baseline | 1. résumer une fenêtre ; 2. apprendre étape par étape ; 3. choisir les réglages ; 4. le score à battre | Les 6 statistiques de `s11` sur une vraie fenêtre ; prédictions du modèle réel après 0, 1, 2, … 200 itérations (`staged_predict`) ; les 10 configurations de la validation croisée ; test |

Contrôles dans le journal : l'ajustement de démonstration (pli 0, configuration retenue) redonne exactement le RMSE de la validation croisée (écart 0) et sa dernière étape égale le modèle final.

**Illustratif** : le rythme, le choix des 12 moteurs (3 premiers de chaque jeu, une fenêtre tirée avec la graine 0), les 13 étapes montrées parmi les 200.
