"""Baseline : statistiques de fenêtre + Gradient Boosting (scikit-learn).

Chaque fenêtre (30 cycles × n colonnes) est résumée par 6 statistiques par colonne (moyenne, écart-type, minimum, maximum,
dernière valeur, pente par moindres carrés) : 20 colonnes -> 120 variables. Ordre « colonne par colonne » : les 6 statistiques
d'une même colonne sont contiguës, ce qui permet de retirer un capteur sans refaire les fenêtres (`columns_for`).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

STATS = ["mean", "std", "min", "max", "last", "slope"]


def window_features(X):
    """(n, cycles, colonnes) -> (n, 6 × colonnes), colonne par colonne, dans l'ordre de STATS."""
    X = np.asarray(X, dtype=np.float32)
    n, w, f = X.shape
    t = np.arange(w, dtype=np.float32)
    t_c = t - t.mean()                                   # Σ t_c = 0 : inutile de centrer x pour la pente
    slope = np.einsum("w,nwf->nf", t_c, X) / float((t_c ** 2).sum())
    stats = np.stack([X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1), X[:, -1, :], slope], axis=2)
    return stats.reshape(n, f * len(STATS))


def feature_names(features):
    return [f"{c}_{s}" for c in features for s in STATS]


def columns_for(features_all, features_keep):
    """Indices des colonnes de `window_features` qui correspondent aux colonnes gardées (dans l'ordre de features_all)."""
    keep = set(features_keep)
    return np.array([i * len(STATS) + j for i, c in enumerate(features_all) if c in keep for j in range(len(STATS))])


def make_regressor(seed=0, max_iter=200, learning_rate=0.1):
    """Pas d'arrêt précoce interne : il découperait au hasard des lignes de fenêtres voisines d'un même moteur (fuite)."""
    return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, early_stopping=False,
                                         random_state=seed)


def make_classifier(seed=0, max_iter=100, learning_rate=0.1):
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate, early_stopping=False,
                                          random_state=seed)
