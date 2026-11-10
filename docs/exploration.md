# Exploration des données (2026-11-10)

Script : `scripts/explore.py` (affiche un résumé chiffré et écrit les figures dans `reports/figures/`). Tous les chiffres ci-dessous viennent des fichiers `train_FD00x.txt`.

## Constats

- **Aucune valeur manquante.**
- **Conditions de vol identiques dans les 4 jeux** : FD001 et FD003 n'ont qu'une condition `(0, 0, 100)` ; FD002 et FD004 ont les mêmes 6 conditions, dont `(0, 0, 100)`. On peut donc normaliser **par condition seule**, avec des statistiques calculées sur l'ensemble d'entraînement. C'est indispensable : le modèle ne reçoit pas l'identifiant du jeu, donc au test on ne sait pas dans quel jeu on est.
- **Piège rencontré** : arrondir les réglages donne `-0.0` et `0.0`, qui faisaient apparaître 4 fausses conditions dans FD001 et FD003. Corrigé en ajoutant `+ 0.0` après l'arrondi (à reprendre dans le script de préparation).
- **Capteurs constants à l'intérieur de chaque condition** : `s1`, `s5`, `s18`, `s19` (écart-type nul dans les 4 jeux). Ils ne varient dans FD002 et FD004 qu'à cause des conditions de vol : à écarter.
- **Capteurs retenus (17)** : `s2, s3, s4, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s20, s21`.
  - `s6`, `s10` et `s16` sont (quasi) constants dans FD001 et/ou FD003 et ne varient que dans FD002 et FD004. Ils sont gardés par le critère (|r| ≥ 0,1 dans au moins un jeu) mais leur utilité est faible : à tester par ablation.
  - **Correction (2026-11-10)** : `s16` est en réalité **constant dans 4 des 6 conditions de vol** (1, 4, 5 et 6) et ne prend que 2 valeurs dans les 2 autres. L'exploration initiale l'avait raté, car son critère de « capteur constant » regardait le maximum sur tous les jeux. La normalisation gère le cas (valeur normalisée 0 là où il est constant). Il est gardé pour l'instant, comme décidé, mais c'est le premier candidat à retirer.
  - **Instabilité mesurée en validation croisée** : les plus gros écarts de moyenne et d'écart-type entre entraînement et validation viennent de `s6` (condition 1 : moyenne 0,44 et écart-type 1,59 dans le pli 4) et de `s16` (condition 2 : écart-type 2,00 dans le pli 2). Ce sont deux capteurs quasi constants avec de rares sauts : leur z-score dépend de quelques événements. Cause probable, non démontrée. À trancher par ablation (voir `docs/normalisation.md`).
- **Durée de vie** : minimum 128 cycles dans tous les jeux, maximum 362 à 543. Moyenne autour de 206 (FD001, FD002) et 246 (FD003, FD004).

## Résultat à retenir : les jeux ne se dégradent pas de la même façon

Corrélation entre capteur normalisé et RUL plafonnée à 125 :

| Capteur | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| s7 | +0,73 | +0,46 | −0,36 | −0,34 |
| s12 | +0,75 | +0,49 | −0,38 | −0,36 |
| s20 | +0,71 | +0,47 | +0,05 | +0,03 |
| s21 | +0,71 | +0,46 | +0,04 | +0,03 |
| s15 | −0,72 | −0,72 | 0,00 | +0,02 |

Pour `s7` et `s12`, le **signe s'inverse** entre les jeux à un mode de panne (FD001, FD002) et ceux à deux modes (FD003, FD004). Pour `s15`, `s20` et `s21`, la corrélation tombe à zéro. Conséquences :

- Une corrélation calculée sur les 4 jeux mélangés (par exemple `s7` : −0,11) **masque** l'information : ne pas écarter un capteur sur cette base.
- Cela rend **plausible** que le modèle puisse deviner le régime de panne depuis les capteurs, ce qui est le principe de la tête de classification. Ce n'est qu'un indice : la cause (dégradation du fan) n'est pas vérifiée ici.

## Figures (`reports/figures/`)

| Fichier | Contenu |
|---|---|
| `degradation_fd001.png` | 6 capteurs en fonction de la RUL, 10 moteurs de FD001 |
| `normalisation_fd002.png` | `s7` d'un moteur de FD002, brut puis normalisé par condition |
| `correlation_rul.png` | Corrélation de chaque capteur avec la RUL, par jeu |
| `durees_de_vie.png` | Histogramme des durées de vie par jeu |

## Reste à vérifier

- Le plafond de RUL et le choix final des capteurs se décideront sur la validation, pas ici.
- Pas encore regardé : les jeux de test (distribution des longueurs, cohérence avec `RUL_FD00x.txt`).
