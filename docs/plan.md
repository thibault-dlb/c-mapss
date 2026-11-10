# Plan du projet

Objectif : entraîner un modèle d'IA de **prédiction de RUL** (maintenance prédictive) sur C-MAPSS, dans le cadre d'un projet étudiant.

## Environnement de la machine (vérifié)

- Python 3.12.10 installé.
- Aucune bibliothèque de data science détectée dans `pip list` (numpy, pandas, scikit-learn, torch, matplotlib, etc.).
- Pas de GPU NVIDIA sur **cette** machine (`nvidia-smi` introuvable) : mise au point et exploration sur CPU. L'entraînement final se fera probablement sur un autre PC avec GPU CUDA (voir « Environnement Python »).

## Étapes proposées

1. ~~**Environnement**~~ : fait (voir « Environnement Python (installé) »).
2. **Exploration (EDA)** : charger les données, repérer les capteurs constants ou inutiles, tracer les dégradations.
3. **Préparation** :
   - normalisation des capteurs (par condition de vol pour FD002 et FD004) ;
   - calcul de la RUL par ligne d'entraînement, avec plafonnement (valeur usuelle d'environ 125 cycles) ;
   - fenêtres glissantes (environ 30 cycles).
4. **Modèles** : baseline simple (régression linéaire ou Random Forest), puis deep learning (LSTM ou CNN 1D) avec PyTorch.
5. **Évaluation** : RMSE sur le test (référence : `RUL_FD00x.txt`) et score asymétrique PHM08, courbes prédiction contre réalité, validation croisée par moteur.
6. **Livrable** : scripts Python propres (pas de notebook), figures pour le rapport.

Les valeurs « usuelles » (plafond de RUL, taille de fenêtre, nombre de capteurs inutiles sur FD001) viennent de la pratique courante sur ce benchmark, pas des fichiers du projet : à vérifier pendant l'exploration.

## Décisions en suspens

- Aucune pour l'instant.
- ~~Périmètre~~ : **les 4 sous-ensembles ensemble dès le départ** (décidé le 2026-11-09, voir « Décisions prises »).
- ~~Format~~ : **scripts Python, pas de notebook** (décidé le 2026-11-09).

## Architecture envisagée (recommandation, non encore validée)

- **Baseline** : Random Forest ou Gradient Boosting sur des statistiques de fenêtre + les 3 conditions de vol.
- **Modèle principal** : CNN 1D puis GRU (ou LSTM) sur une fenêtre d'environ 30 cycles ; entrées = capteurs utiles + 3 réglages opérationnels ; sortie = RUL plafonnée ; perte MSE ou Huber.
- Petit modèle volontairement (709 moteurs d'entraînement au total) ; Transformer en option seulement.
- ~~Variable « sous-ensemble » en entrée~~ : **rejetée par l'utilisateur**. Au test, le modèle doit deviner lui-même le mode de panne. Piste : apprentissage multitâche (tête RUL + tête « mode de panne »), voir ci-dessous.
- Limite à garder en tête : les fichiers ne donnent pas le mode de panne **par moteur**, seulement par sous-ensemble (FD001 et FD002 : HPC seul ; FD003 et FD004 : HPC et fan). Une étiquette de mode ne peut donc être qu'un proxy grossier (« régime HPC seul » contre « régime à deux modes »).
- Les conditions de vol (1 ou 6) se déduisent des réglages d'entrée eux-mêmes : inutile de les prédire.

## Décisions du QCM (2026-11-09)

- Plafond de RUL : **tester plusieurs valeurs** (par exemple 100, 125, 150) et garder la meilleure sur la validation.
- Normalisation des capteurs : **par condition de vol**.
- Git : **commit et push automatiques à chaque étape terminée** (branche `main`).
- Sous-ensemble : pas d'identifiant en entrée (voir plus haut).

## Décisions du QCM n°2 (2026-11-09)

- Mode de panne : **apprentissage multitâche avec étiquette proxy**. Une tête de classification (« HPC seul » contre « deux modes », étiquette tirée du jeu) en plus de la tête RUL ; au test, le modèle la prédit seul.
- Validation : **validation croisée 5 plis**, découpage par moteur (GroupKFold), identifiants de moteur uniques entre jeux.
- Fenêtres plus courtes que 30 cycles (certains moteurs de test n'en ont que 19 à 38) : **compléter avec la première valeur** (padding par répétition du premier cycle).
- Environnement : installé ensuite à la demande de l'utilisateur (voir plus bas).

## Avancement (2026-11-10)

- Terminé : environnement, exploration (`exploration.md`), **normalisation par condition de vol** (`normalisation.md`), **fenêtres glissantes** (`fenetres.md`), chacune avec sa vidéo.
- Prochaines étapes, dans l'ordre : chargeur de données par pli (scaler du pli, fenêtres, lots PyTorch) ; métriques (RMSE et score PHM08) ; baseline ; modèle CNN 1D + GRU et fonction de perte ; format du journal et tableau de bord d'entraînement ; protocole (graines, arrêt précoce, validation croisée).
- À trancher par ablation sur la validation : garder ou retirer `s6`, `s10`, `s16` (statistiques instables d'un pli à l'autre).

## Exigence : tout doit être visuel et filmable (2026-11-10)

Demande de l'utilisateur : pouvoir **enregistrer en vidéo** ce qui se passe, surtout pendant la **normalisation des données** et pendant **l'entraînement**, y compris **l'évolution de l'architecture du réseau de neurones**.

Conséquences pour les scripts :

- Normalisation : animation avant / après par condition de vol (pas seulement des figures statiques).
- Entraînement : tableau de bord **en direct** pendant l'apprentissage, avec au minimum le schéma du réseau (couches, poids et activations qui évoluent), la courbe de perte (entraînement et validation), et RUL prédite contre RUL réelle qui se précise au fil des époques.
- Chaque visualisation doit être lisible à l'écran (grande police, couleurs contrastées) et pouvoir aussi s'exporter en fichier vidéo ou GIF.
- Le tableau de bord doit fonctionner sur les deux machines (CPU ici, CUDA sur l'autre PC).
- **Décidé** : page web locale (D3.js), qui rejoue un **journal d'événements** écrit par les scripts (la page ne calcule rien). Vidéo par capture réelle de l'onglet (bouton, non testé) ou rendu automatique image par image depuis le journal. Fait pour la normalisation ; le même principe s'appliquera à l'entraînement (journal de lots, d'époques, de statistiques par couche).
- La vidéo grand public doit rester simple : une phrase par scène, pas de formule ni de chiffres techniques (retour de l'utilisateur).

## Environnement Python (installé)

- Venv : `.venv` à la racine du projet (ignoré par Git).
- Versions : numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, torch 2.15.0 (build CPU). Listées dans `requirements.txt`.
- Vérifié : les 5 bibliothèques s'importent, `torch.cuda.is_available()` vaut `False` (CPU seul, comme prévu), 4 threads CPU.
- Activer : `.\.venv\Scripts\Activate.ps1`.
- **Deux machines prévues** : celle-ci (CPU) et un autre PC avec CUDA, où l'entraînement final sera probablement fait. Fichiers : `requirements.txt` (commun, sans PyTorch), `requirements-cpu.txt`, `requirements-cuda.txt`. Voir le `README.md` pour les commandes.
- Index PyTorch vérifiés sur `download.pytorch.org` pour torch 2.15.0, Windows, Python 3.12 : `cpu`, `cu132` et `cu134` existent. `cu134` est choisi par défaut ; `cu132` est la solution de repli pour un pilote plus ancien. Quelle version CUDA accepte le pilote du PC cible : **inconnu**, à lire avec `nvidia-smi`.
- **Non testé** : l'installation CUDA n'a pas pu être essayée (pas de GPU ici, et le test aurait téléchargé plusieurs Go). Elle repose sur la règle de version de pip (`2.15.0+cu134` passe devant `2.15.0`), qui a fonctionné pour la variante CPU ici. À valider sur le PC CUDA avec `scripts/check_gpu.py`.
- Convention de code : choisir le périphérique avec `"cuda" if torch.cuda.is_available() else "cpu"`.

## Décisions prises

- **Entraînement sur les 4 sous-ensembles (FD001 à FD004) ensemble**, avec les **conditions de vol (3 réglages opérationnels) en entrée du modèle** (décision de l'utilisateur, 2026-11-09). Constat sur les fichiers : FD001 et FD003 ont une seule condition (réglages quasi constants autour de 0 / 0 / 100) ; FD002 et FD004 ont exactement 6 conditions discrètes (altitude, Mach, TRA) : (0, 0, 100), (10, 0,25, 100), (20, 0,70, 100), (25, 0,62, 60), (35, 0,84, 100), (42, 0,84, 100). L'ordre altitude / Mach / TRA des colonnes 3 à 5 est déduit des plages de l'article, pas écrit explicitement dans le `readme.txt`.

- Dépôt GitHub : <https://github.com/thibault-dlb/c-mapss>, branche `main`.
- Les données sont versionnées dans le dépôt, y compris `CMAPSSData.zip` (doublon de `raw/`, retirable si on veut alléger).
