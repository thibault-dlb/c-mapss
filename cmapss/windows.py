"""Fenêtres glissantes : tranches de l'historique récent d'un moteur, une par exemple d'apprentissage.

Une fenêtre = les WINDOW derniers cycles (WINDOW × 20 valeurs : 17 capteurs normalisés + 3 réglages).
Son étiquette est la RUL au dernier cycle de la fenêtre (non plafonnée : le plafond est appliqué à
l'entraînement avec `cap_rul`, pour pouvoir tester 100, 125 ou 150).

- Entraînement : une fenêtre se termine à chaque cycle à partir du WINDOW-ième (fenêtres complètes seulement).
- Test : une seule fenêtre par moteur, les WINDOW derniers cycles connus. Si le moteur en a moins, on complète
  au début en répétant le premier cycle.

Les fenêtres sont fabriquées moteur par moteur : aucune ne chevauche deux moteurs.
"""
from dataclasses import dataclass

import numpy as np

from .data import SENSORS_KEPT, SETTINGS

WINDOW = 30
FEATURES = SENSORS_KEPT + SETTINGS  # 17 capteurs + 3 réglages = 20 valeurs par cycle

# Étiquette proxy du régime de panne, donnée seulement par jeu (voir docs/plan.md) :
# 0 = « HPC seul » (FD001, FD002), 1 = « deux modes possibles » (FD003, FD004)
REGIME_OF_SUBSET = {"FD001": 0, "FD002": 0, "FD003": 1, "FD004": 1}


@dataclass
class Windows:
    X: np.ndarray          # (n, WINDOW, n_features) float32
    rul: np.ndarray        # (n,) RUL au dernier cycle, non plafonnée
    regime: np.ndarray     # (n,) 0 = HPC seul, 1 = deux modes possibles
    engine_id: np.ndarray  # (n,) identifiant du moteur
    end_cycle: np.ndarray  # (n,) numéro du dernier cycle de la fenêtre
    n_padded: np.ndarray   # (n,) cycles de remplissage au début (0 pour une fenêtre complète)

    def __len__(self):
        return len(self.X)


def n_windows(n_cycles, window=WINDOW):
    """Nombre de fenêtres complètes qu'un moteur de n_cycles cycles fournit à l'entraînement."""
    return max(n_cycles - window + 1, 0)


def pad_first(rows, window=WINDOW):
    """Complète au début, en répétant le premier cycle, jusqu'à `window` lignes. Retourne (fenêtre, nb de lignes ajoutées)."""
    missing = max(window - len(rows), 0)
    if missing == 0:
        return rows[-window:], 0
    return np.concatenate([np.repeat(rows[:1], missing, axis=0), rows]), missing


def cap_rul(rul, cap):
    """Plafonne la RUL (à appliquer à l'entraînement : un moteur neuf n'est pas « plus sain » à 300 cycles de la panne qu'à 150)."""
    return np.minimum(rul, cap)


def make_windows(df, kind, window=WINDOW, features=FEATURES, on_window=None):
    """Fabrique les fenêtres de df (déjà normalisé, colonnes engine_id, subset, cycle, rul + features).

    kind = "train" : toutes les fenêtres complètes ; kind = "test" : une fenêtre par moteur (remplissage si trop court).
    on_window(engine_id, start_cycle, end_cycle, rul, n_padded, fenêtre) est appelé pour chaque fenêtre créée :
    c'est ce qui permet de journaliser le vrai déroulement (très lent si appelé sur 140 000 fenêtres : à réserver à un moteur).
    """
    if kind not in ("train", "test"):
        raise ValueError("kind doit valoir 'train' ou 'test'")
    X, rul, regime, eng, end, npad = [], [], [], [], [], []
    for engine_id, g in df.groupby("engine_id", sort=False):
        g = g.sort_values("cycle")
        rows = g[features].to_numpy(dtype=np.float32)
        cycles, ruls = g["cycle"].to_numpy(), g["rul"].to_numpy()
        reg = REGIME_OF_SUBSET[g["subset"].iloc[0]]
        if kind == "train":
            if len(rows) < window:
                continue  # trop court pour une fenêtre complète
            ends = range(window - 1, len(rows))
            wins = [(rows[e - window + 1:e + 1], 0, e) for e in ends]
        else:
            w, missing = pad_first(rows, window)
            wins = [(w, missing, len(rows) - 1)]
        for w, missing, e in wins:
            X.append(w); rul.append(ruls[e]); regime.append(reg); eng.append(engine_id)
            end.append(cycles[e]); npad.append(missing)
            if on_window is not None:
                on_window(engine_id, int(cycles[max(e - (window - missing) + 1, 0)]), int(cycles[e]),
                          float(ruls[e]), missing, w)
    if not X:
        return Windows(np.empty((0, window, len(features)), np.float32), np.empty(0, np.float32),
                       np.empty(0, np.int8), np.empty(0, object), np.empty(0, int), np.empty(0, int))
    return Windows(np.stack(X).astype(np.float32), np.array(rul, dtype=np.float32), np.array(regime, dtype=np.int8),
                   np.array(eng, dtype=object), np.array(end), np.array(npad))
