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

## À faire ensuite

Voir `plan.md` : choisir FD001 d'abord ou tous les jeux, notebook ou scripts, puis configurer l'environnement et explorer les données.
