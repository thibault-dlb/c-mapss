# Résumé : *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*

Auteurs : A. Saxena, K. Goebel, D. Simon, N. Eklund. PHM08, Denver, octobre 2008. Fichier : `data/raw/Damage Propagation Modeling.pdf` (9 pages).

Texte extrait du PDF avec `pypdf` puis lu en entier.

## Motivation

Il manque des jeux de données publics « run-to-failure » (machines suivies jusqu'à la panne). La NASA a donc simulé ces données pour le premier concours de pronostic PHM (PHM08). Le pronostic y est défini comme l'estimation de la durée de vie utile restante (RUL), en unités de temps (ici des cycles).

## Simulateur

C-MAPSS (*Commercial Modular Aero-Propulsion System Simulation*), MATLAB/Simulink, turbofan de la classe 90 000 lb de poussée. Altitudes de 0 à 40 000 ft, Mach de 0 à 0,90, avec contrôleur en boucle fermée. 14 entrées (carburant + 13 modificateurs de santé). Sur 58 sorties, **21** sont utilisées comme capteurs.

Capteurs disponibles : T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32.

Non fournis aux participants (servent au calcul de la santé) : T48 (EGT), SmFan, SmLPC, SmHPC.

## Modèle de dégradation

- Forme exponentielle de l'usure, `w = A·exp(B(t))`.
- Indice de santé : `h(t) = 1 − d − exp{a·t^b}` (avec `d` la dégradation initiale).
- Efficacité `e(t)` et débit `f(t)` modélisés séparément, puis agrégés : `H(t) = g(e(t), f(t))`.
- `g` = minimum des marges opérationnelles (soufflante, HPC, LPC, EGT).
- **Critère de panne : H = 0.**
- Seul le **HPC** est dégradé dans les données du concours.
- Dégradation initiale tirée dans [0,99 ; 1] (moins de 1 %). Paramètres `a` dans [0,001 ; 0,003], `b` dans [1,4 ; 1,6].

## Réalisme et bruit

- Usure initiale et variations de fabrication aléatoires (non anormales).
- Bruit de processus : mélange de deux gaussiennes de variances légèrement différentes.
- Bruit de mesure ajouté sur tous les capteurs.
- Une mesure par vol ; la maintenance entre vols est incluse comme bruit, donc la dégradation n'est pas localement monotone.
- 6 conditions de vol : altitude 0 à 42 000 ft, Mach 0 à 0,84, TRA 20 à 100.

## Découpage des données

- Entraînement : trajectoires jusqu'à la panne.
- Test et validation : arrêtées avant la panne.
- RUL du jeu de test entre 10 et 150 cycles ; RUL du jeu de validation entre 6 et 190 cycles (validation publiée plus tard, sans retour de score pendant le concours).

## Métrique : score asymétrique

Avec `d = RUL estimée − RUL réelle` (équation 11 de l'article, constantes `a1 = 10` et `a2 = 13`) :

- si `d < 0` (prédiction **précoce**) : `s = Σ (exp(−d/13) − 1)`
- si `d ≥ 0` (prédiction **tardive**) : `s = Σ (exp(d/10) − 1)`

Une prédiction **tardive** est plus pénalisée qu'une précoce (logique pour la sécurité) : c'est la constante la plus petite, 10, qui s'applique à la branche tardive. La pénalité croît exponentiellement avec l'erreur.

**Correction (2026-11-10)** : une version précédente de cette note appliquait `a1 = 10` à la branche précoce, ce qui pénaliserait davantage les prédictions précoces et contredisait la phrase ci-dessus. Le texte de l'article dit que « les prédictions tardives sont plus pénalisées que les précoces » ; la littérature applique 13 au précoce et 10 au tardif (par exemple l'article n°2348 de la PHM Society et arXiv 2410.03134). L'extraction du texte du PDF brouille l'équation 11 : on n'y lit pas à quelle branche va chaque constante, donc cette attribution repose sur le texte de l'article et sur la littérature, pas sur une lecture directe de l'équation. Le code (`cmapss/metrics.py`) la fige dans des constantes nommées et un test vérifie la propriété « tardif plus pénalisé que précoce ».

L'article suggère aussi de pondérer plus fort les RUL courtes (plus importantes) et d'ajouter une **métrique de corrélation**, car des scores agrégés identiques peuvent cacher des comportements très différents selon les moteurs.

## Ce qu'il faut retenir pour le projet

- Les données sont synthétiques : à signaler dans le rapport.
- Évaluer avec le **RMSE** et la **score function** de PHM08.
- La dégradation est lisse et exponentielle sur un seul composant, ce qui explique en partie les bons résultats publiés.
