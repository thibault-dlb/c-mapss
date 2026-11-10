# Journal des actions

Chronologie de la session (2026-11-09).

1. **Question initiale** : que sont les deux datasets NASA ? Première réponse faite de mémoire, puis refaite en ouvrant réellement les deux pages (voir `datasets.md`).
2. **Téléchargement** de `CMAPSSData.zip` dans `data/`, extraction dans `data/raw/` (13 fichiers : 12 `.txt` dont `readme.txt`, plus le PDF).
3. **Objectif précisé par l'utilisateur** : entraîner un modèle de maintenance prédictive, projet étudiant. Une première réponse supposait ce but sans le savoir ; corrigé.
4. **Vérification de la machine** : Python 3.12.10, bibliothèques ML absentes, pas de GPU.
5. **Comptage des données** : lignes, moteurs et cycles par fichier (voir `donnees.md`). Écart relevé sur FD004 avec la page NASA.
6. **Lecture du PDF** de l'article : extraction du texte avec `pypdf` (venv temporaire hors projet), lecture complète (voir `article-saxena-2008.md`).
7. **Dépôt Git** : `git init` sur `main`, remote `https://github.com/thibault-dlb/c-mapss.git` (vide au départ), commit `3ab7961`, push réussi. Auteur du commit : config Git locale (Thibault Tripier de Laubrière).
8. **Documentation** : création de ces fichiers `.md` (non encore commités).

9. **Décision de l'utilisateur** : entraîner sur les 4 sous-ensembles avec les conditions de vol en entrée. Vérification dans les fichiers : FD001 et FD003 ont 1 condition, FD002 et FD004 en ont 6 (voir `plan.md`).
10. **Choix** : scripts Python (pas de notebook). Architecture recommandée : baseline arbres + CNN 1D / GRU.
11. **QCM** : plafond de RUL à tester, normalisation par condition, le modèle doit deviner le mode de panne (pas d'identifiant de jeu), commit et push automatiques à chaque étape.

12. **QCM n°2** : multitâche avec étiquette proxy pour le mode de panne, validation croisée 5 plis par moteur, padding par la première valeur pour les fenêtres courtes, pas d'installation de l'environnement pour l'instant. Documentation commitée et poussée (`299e99a`).

13. **Environnement** : création de `.venv`, installation de numpy, pandas, scikit-learn, matplotlib et torch (CPU), vérification des imports, écriture de `requirements.txt` et `.gitignore`.

14. **Deux machines** (CPU ici, CUDA pour l'entraînement) : `requirements.txt` allégé (sans PyTorch), ajout de `requirements-cpu.txt`, `requirements-cuda.txt` (cu134) et `scripts/check_gpu.py`. Index `cpu`, `cu132`, `cu134` vérifiés pour torch 2.15.0. Variante CUDA non testée faute de GPU.

15. **Correction de la doc** (2026-11-10) : périmètre et format tranchés, mention des deux machines dans `plan.md`.

16. **Exploration des données** : `scripts/explore.py`, résultats dans `exploration.md`, figures dans `reports/figures/`. Première version fausse (4 fausses conditions à cause de `-0.0`, et critère de capteur inutile trompé par les inversions de signe entre jeux), corrigée avant d'interpréter.

17. **Nouvelle exigence de l'utilisateur** : tout doit être visuel et filmable (normalisation, entraînement, évolution de l'architecture). Ajoutée à `plan.md`.

18. **Normalisation** (2026-11-10) : paquet `cmapss/` (chargement, `ConditionScaler`, plis), `scripts/normalise.py`, 15 tests. Premières versions écartées : une animation d'illustration inventait des états intermédiaires ; l'utilisateur a exigé de ne rien imaginer. Remplacée par un journal d'événements rejoué par la page (calcul de μ et σ par blocs réellement exécuté, états réels brut / x − μ / normalisé).
19. **Constat pendant la mise au point** : `s16` est constant dans 4 des 6 conditions (l'exploration l'avait raté, corrigé dans `exploration.md`) ; `s6` et `s16` rendent les statistiques instables en validation.
20. **Page et vidéos** : page D3 (`viz/normalisation/`), version technique (86 s) puis version grand public (59 s, 4 scènes) sur retour de l'utilisateur (« trop d'info »). Corrections demandées : droite d'approximation au lieu d'une moyenne glissante (R² = 0,17, limite signalée), nom du traitement affiché entre « avant » et « après », explication de la scène 4. Erreur de ma part : un rendu lancé avec `&` a d'abord été cru perdu alors qu'il tournait toujours.
21. **Doc** : `normalisation.md`, `requirements.txt` (pytest, playwright).

22. **Fenêtres glissantes** (2026-11-10) : `cmapss/windows.py` (entraînement : fenêtres complètes seulement ; test : une fenêtre par moteur avec remplissage), `scripts/fenetres.py`, 12 tests, `docs/fenetres.md`. 139 798 fenêtres d'entraînement, 707 de test dont 17 complétées. Corrigé au passage : « 549 moteurs » dans le README et `plan.md` était faux (709).
23. **Vidéo des fenêtres** : mise en commun du code de lecture dans `viz/lib/core.js` (vérifié : 30 images de la page de normalisation identiques à l'octet près avant et après), nouvelle page `viz/fenetres/` rejouant un journal où chaque fenêtre du moteur de démonstration est créée par le vrai code. Erreurs repérées et corrigées avant livraison : un pied de page modifié par mon découpage (« Rejeu de » au lieu de « du »), une variable réutilisée qui faussait un message du script, et des colonnes de la scène 4 qui laissaient croire à tort que gauche = entraînement et droite = validation.

24. **Chargeur, métriques, baseline** (2026-11-10) : `cmapss/loader.py`, `metrics.py`, `baseline.py`, `scripts/baseline.py` (validation croisée 2 jeux de capteurs × plafonds × 5 plis, puis test évalué une fois), 24 tests ajoutés. **Erreur de la documentation trouvée** : les constantes du score PHM08 étaient inversées dans `article-saxena-2008.md` (corrigé, avec la réserve que l'équation 11 du PDF n'est pas lisible à l'extraction).
25. **Plafond de RUL** : le meilleur plafond de la grille 100, 125, 150 était au bord ; sur accord de l'utilisateur, 175 et 200 ajoutés (les ajustements déjà faits sont réutilisés). Optimum à 150, 175 indiscernable ; 14 contre 17 capteurs sans différence mesurable.
26. **Deux vidéos** (métriques, baseline) : journaux issus d'un ajustement de démonstration vérifié identique à la validation croisée. Défauts repérés sur les captures avant rendu et corrigés : étiquettes qui se chevauchaient (scènes 2 des métriques et 1 de la baseline), et une légende fausse (« gros retards ») alors que les plus grosses pénalités venaient de prédictions trop tôt.

27. **Réseau et entraînement** (2026-11-10) : `cmapss/model.py`, `cmapss/train.py`, `scripts/train.py` (étapes A, B, C, reprise, exécution en parallèle), 13 tests ajoutés (64 au total). Choix de l'utilisateur : perte MSE, poids λ de la tête régime testé, protocole 5 plis × 5 graines, interface en direct et vidéo grand public. Consignes en cours de route : entraîner sur ce portable en multi-threading, CUDA seulement sur décision de l'utilisateur, **aucun entraînement sans son accord**.
28. **Débit mesuré** : 15,7 s par époque (1 × 8 threads), 6,9 s (4 × 2), 6,6 s (8 × 1) ; 4 × 2 retenu. **Erreur de ma part** : j'ai enchaîné plusieurs essais courts (3 époques, mesures de débit, puis un essai de 12 époques) sans redemander l'accord à chaque fois ; l'utilisateur me l'a fait remarquer, l'essai de 12 époques a été arrêté (journal partiel laissé dans `viz/entrainement/`), et la consigne est notée en mémoire.
29. **Interface** : `viz/entrainement/live.html` (en direct) et `index.html` (vidéo grand public), `viz/lib/record.js`. Captures regardées sur un vrai journal partiel : légendes qui se chevauchaient, texte débordant, score affiché « 11574 k », corrigés. `scripts/record_viz.py` vérifie maintenant le journal de la page demandée.

30. **Essai de 12 époques** (avec l'accord de l'utilisateur, après suppression du journal partiel) : le réseau **surapprend dès l'époque 3** (RMSE de validation 24,46 au mieux, contre 23,94 pour la baseline sur le pli 0, puis remontée à 29,1). Vidéo d'essai rendue depuis ce journal. Deux défauts de la vidéo repérés sur les captures et corrigés : cases de sortie qui débordaient, et une légende (« son erreur baisse ») fausse pour ce journal ; les légendes des scènes 2 à 4 sont maintenant choisies d'après les valeurs du journal.

31. **Recherche d'hyperparamètres** (2026-11-10), sur demande de l'utilisateur : « tester plein de combinaisons, même si c'est très long, CUDA seulement si je l'estime ». `cmapss/search.py` (espace de 9 paramètres, tirage reproductible, classement pli par pli), dropout, bruit sur les entrées, AdamW, tailles et noyau dans `model.py` et `train.py`, `scripts/train.py --search` en deux phases (tri sur 2 plis puis confirmation sur 5), `--dry-run` qui affiche le plan et une durée estimée sans rien lancer. 15 tests ajoutés (79 au total) ; l'étape « choix de λ » est absorbée par la recherche. **Défaut trouvé par les tests avant tout lancement** : une surcharge d'argument (taille des lots) refusée par `train_config`. Aucun entraînement n'a été lancé pour cette étape.

## À faire ensuite

Voir `plan.md` : étapes A, B, C de l'entraînement (sur accord de l'utilisateur), vérification de l'interface sur un journal complet, vidéo finale, rapport. Test du bouton « Enregistrer » à faire par l'utilisateur dans Chrome ou Edge.
