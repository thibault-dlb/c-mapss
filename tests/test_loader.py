import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmapss.data import CONDITIONS, SENSORS_KEPT, load_all
from cmapss.folds import make_folds
from cmapss.loader import (N_FOLDS, RUL_TEST_MAX, RUL_TEST_MIN, SENSORS_UNSTABLE, features_without, fold_data,
                           load_test_data, make_loader, to_torch_dataset)
from cmapss.windows import FEATURES, n_windows


@pytest.fixture(scope="module")
def train():
    return load_all("train")


@pytest.fixture(scope="module")
def folds(train):
    return make_folds(train, n_splits=N_FOLDS)


@pytest.fixture(scope="module")
def fd0(train, folds):
    return fold_data(0, train=train, folds=folds)


def test_moteurs_disjoints_entre_entrainement_et_validation(fd0):
    assert not (set(fd0.train.engine_id) & set(fd0.val.engine_id))


def test_effectifs(fd0, train, folds):
    lengths = train.groupby("engine_id").size()
    val_ids = set(folds.loc[folds["fold"] == 0, "engine_id"])
    assert len(fd0.val) == sum(n_windows(lengths[e]) for e in val_ids)
    assert len(fd0.train) + len(fd0.val) == 139798
    assert fd0.train.X.shape[1:] == (30, 20) and not np.isnan(fd0.train.X).any() and not np.isnan(fd0.val.X).any()


def test_les_5_plis_couvrent_tous_les_moteurs_une_fois(train, folds):
    lengths = train.groupby("engine_id").size()
    total = 0
    for k in range(N_FOLDS):
        ids = folds.loc[folds["fold"] == k, "engine_id"]
        total += sum(n_windows(lengths[e]) for e in ids)
    assert total == 139798 and folds["engine_id"].is_unique


def test_scaler_sans_fuite(fd0, train, folds):
    """Les statistiques du pli viennent des seuls moteurs d'entraînement du pli, pas de ceux de validation."""
    val_ids = set(folds.loc[folds["fold"] == 0, "engine_id"])
    rows = train[(~train["engine_id"].isin(val_ids)) & (train["cond"] == CONDITIONS[1])]
    assert np.allclose(fd0.scaler.mean[CONDITIONS[1]], rows[SENSORS_KEPT].mean().to_numpy())
    full = train[train["cond"] == CONDITIONS[1]][SENSORS_KEPT].mean().to_numpy()
    assert not np.allclose(fd0.scaler.mean[CONDITIONS[1]], full)


def test_masque_plage_du_test(fd0):
    r, m = fd0.val.rul, fd0.val_in_range
    assert m.dtype == bool and len(m) == len(fd0.val)
    assert ((r[m] >= RUL_TEST_MIN) & (r[m] <= RUL_TEST_MAX)).all()
    assert (r < RUL_TEST_MIN).any() and not m[r < RUL_TEST_MIN].any()     # les fins de vie (RUL 0 à 5) sont exclues
    assert (r > RUL_TEST_MAX).any() and not m[r > RUL_TEST_MAX].any()     # les débuts de vie très sains aussi


def test_sous_ensemble_de_capteurs(train, folds):
    feats = features_without(SENSORS_UNSTABLE)
    assert len(feats) == 17 and not set(SENSORS_UNSTABLE) & set(feats) and len(FEATURES) == 20
    fd = fold_data(1, features=feats, train=train, folds=folds)
    assert fd.train.X.shape[2] == 17 and fd.features == feats


def test_pli_inconnu(train, folds):
    with pytest.raises(ValueError):
        fold_data(9, train=train, folds=folds)


def test_donnees_de_test(train):
    td = load_test_data(train=train)
    assert len(td.windows) == 707 and td.windows.X.shape[1:] == (30, 20)
    assert int((td.windows.n_padded > 0).sum()) == 17
    assert {fd: int((td.subsets == fd).sum()) for fd in ("FD001", "FD002", "FD003", "FD004")} == \
        {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
    assert td.windows.rul.min() == 6 and td.windows.rul.max() == 195    # la plage réelle des fichiers RUL_FD00x.txt


def test_dataset_torch_et_plafond(fd0):
    ds = to_torch_dataset(fd0.val, cap=125, mask=fd0.val_in_range)
    X, rul, regime = ds.tensors
    assert X.dtype.is_floating_point and X.shape[1:] == (30, 20) and rul.max() <= 125 and len(X) == len(rul) == len(regime)
    assert regime.dtype.is_floating_point is False and set(regime.unique().tolist()) <= {0, 1}


def test_lots_reproductibles(fd0):
    ds = to_torch_dataset(fd0.val, cap=125, mask=fd0.val_in_range)
    a = next(iter(make_loader(ds, batch_size=64, seed=3)))[1]
    b = next(iter(make_loader(ds, batch_size=64, seed=3)))[1]
    c = next(iter(make_loader(ds, batch_size=64, seed=4)))[1]
    assert (a == b).all() and not (a == c).all()
