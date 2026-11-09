# Les deux pages NASA

Contenu vérifié en ouvrant les pages (pas de mémoire).

## 1. CMAPSS Jet Engine Simulated Data

URL : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>

Séries temporelles multivariées simulant une flotte de moteurs d'avion du même type. Chaque moteur part avec une usure initiale et des variations de fabrication inconnues, considérées comme normales. Chaque série commence en fonctionnement normal puis développe un défaut. Objectif : prédire le nombre de cycles restants avant la panne (RUL) sur le jeu de test.

| Jeu | Trajectoires (train / test) selon la page | Conditions | Modes de défaut |
|---|---|---|---|
| FD001 | 100 / 100 | 1 (niveau de la mer) | 1 (HPC) |
| FD002 | 260 / 259 | 6 | 1 (HPC) |
| FD003 | 100 / 100 | 1 (niveau de la mer) | 2 (HPC, fan) |
| FD004 | 248 / 249 | 6 | 2 (HPC, fan) |

Remarque : le comptage réel de FD004 donne 249 en train et 248 en test, soit l'inverse de la page (voir `donnees.md`).

- Ressource unique : `CMAPSSData.zip` (<https://data.nasa.gov/docs/legacy/CMAPSSData.zip>).
- Format : texte, valeurs séparées par des espaces, 26 colonnes (voir `donnees.md`).
- Licence : non spécifiée.
- Dates : émis le 14/07/2022, modifié le 15/05/2024, mis à jour sur le portail le 29/05/2025.
- Référence : A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08, Denver, 2008.
- Contact : Chris Teubert (PCoE, NASA).
- Mots-clés : degradation, fan-degradation, ivhm, phm, prognostics.

## 2. C-MAPSS Aircraft Engine Simulator Data (DASHLINK_140)

URL : <https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data>

Jeu de données **différent** du premier, orienté détection et localisation de défaut (pas RUL).

- Simulation Matlab/Simulink de turboréacteurs commerciaux.
- Chaque vol comprend 7 conditions de vol, de la montée à 35 000 ft jusqu'à la descente.
- Enregistrements à 1 Hz, 30 paramètres (conditions de vol, indicateurs de santé, températures, pressions).
- Défauts : soufflante d'entrée, compresseurs basse et haute pression, turbines haute et basse pression. Un défaut est injecté à un instant donné d'un vol puis persiste.
- Objectif : identifier le vol et le moment d'apparition du défaut.
- **Aucun fichier disponible** : la page indique « This dataset has no data ». C-MAPSS et C-MAPSS40K sont indiqués comme indisponibles au téléchargement, accès « non-public ».
- Licence : non spécifiée.
- Mainteneur : Nikunj Oza. Contact : Jonathan Litt.
- Dates : publié le 22/09/2010, modifié le 29/01/2020.

## Conclusion

Seul le premier jeu est exploitable pour le projet. Le second n'est qu'une fiche descriptive sans données.
