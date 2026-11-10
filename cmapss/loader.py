"""Chargeur de données par pli : scaler du pli, fenêtres d'entraînement et de validation, jeu de test, lots PyTorch.

Règles :
- le scaler d'un pli est ajusté **sur les moteurs d'entraînement du pli seulement** (jamais sur la validation) ;
- la validation se mesure sur les fenêtres dont la **vraie RUL est dans la plage du test** (RUL_TEST_MIN à RUL_TEST_MAX),
  contre la vraie RUL non plafonnée : les moteurs d'entraînement vont jusqu'à la panne, ceux du test sont coupés avant ;
- le test n'est chargé que par `load_test_data`, avec le scaler « full » (tout l'entraînement) : à n'utiliser qu'à la fin.
"""
from dataclasses import dataclass

import numpy as np

from .data import load_all
from .folds import make_folds
from .normalisation import ConditionScaler
from .windows import FEATURES, Windows, cap_rul, make_windows

N_FOLDS = 5
RUL_TEST_MIN, RUL_TEST_MAX = 6, 195          # plage des vraies RUL du jeu de test, mesurée (RUL_FD001 à RUL_FD004)
SENSORS_UNSTABLE = ["s6", "s10", "s16"]       # statistiques instables d'un pli à l'autre (docs/normalisation.md)


def features_without(drop):
    """FEATURES privé de certains capteurs (par exemple SENSORS_UNSTABLE) : 14 capteurs + 3 réglages."""
    drop = set(drop)
    return [f for f in FEATURES if f not in drop]


@dataclass
class FoldData:
    fold: int
    scaler: ConditionScaler    # ajusté sur les moteurs d'entraînement du pli seulement
    train: Windows
    val: Windows
    val_in_range: np.ndarray   # masque sur val : vraie RUL dans [RUL_TEST_MIN, RUL_TEST_MAX]
    features: list


@dataclass
class TestSet:
    scaler: ConditionScaler    # scaler « full »
    windows: Windows           # une fenêtre par moteur de test
    subsets: np.ndarray        # jeu (FD001 à FD004) de chaque fenêtre
    features: list


def subset_of(engine_ids):
    return np.array([str(e).split("_")[0] for e in engine_ids])


def fold_data(fold, features=FEATURES, train=None, folds=None):
    """Fenêtres du pli `fold` (0 à 4). `train` et `folds` peuvent être passés pour éviter de les recharger."""
    train = load_all("train") if train is None else train
    folds = make_folds(train, n_splits=N_FOLDS) if folds is None else folds
    val_ids = set(folds.loc[folds["fold"] == fold, "engine_id"])
    if not val_ids:
        raise ValueError(f"pli inconnu : {fold}")
    is_val = train["engine_id"].isin(val_ids)
    scaler = ConditionScaler().fit(train[~is_val])
    w_tr = make_windows(scaler.transform(train[~is_val]), "train", features=features)
    w_va = make_windows(scaler.transform(train[is_val]), "train", features=features)
    in_range = (w_va.rul >= RUL_TEST_MIN) & (w_va.rul <= RUL_TEST_MAX)
    return FoldData(fold, scaler, w_tr, w_va, in_range, list(features))


def load_test_data(features=FEATURES, train=None):
    """Fenêtres de test (scaler « full »). À n'utiliser qu'une fois, pour l'évaluation finale."""
    train = load_all("train") if train is None else train
    scaler = ConditionScaler().fit(train)
    w = make_windows(scaler.transform(load_all("test")), "test", features=features)
    return TestSet(scaler, w, subset_of(w.engine_id), list(features))


def to_torch_dataset(w, cap=None, mask=None):
    """Fenêtres -> `TensorDataset` (X float32 [n, 30, n_features], RUL float32 éventuellement plafonnée, régime int64)."""
    import torch
    from torch.utils.data import TensorDataset
    sel = slice(None) if mask is None else mask
    rul = w.rul[sel] if cap is None else cap_rul(w.rul[sel], cap)
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(w.X[sel])),
                         torch.from_numpy(np.asarray(rul, dtype=np.float32)),
                         torch.from_numpy(w.regime[sel].astype(np.int64)))


def make_loader(dataset, batch_size=256, shuffle=True, seed=0):
    """DataLoader reproductible : l'ordre des lots ne dépend que de `seed`."""
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
