import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmapss.data import CONDITIONS, SENSORS_KEPT, SETTINGS, SUBSETS, add_condition, load_all
from cmapss.folds import make_folds
from cmapss.normalisation import ConditionScaler


@pytest.fixture(scope="module")
def train():
    return load_all("train")


@pytest.fixture(scope="module")
def folds(train):
    return make_folds(train)


def test_conditions_connues(train):
    assert set(train["cond"]) == set(CONDITIONS)
    test = load_all("test")
    assert set(test["cond"]) <= set(CONDITIONS)


def test_pas_de_zero_negatif():
    df = pd.DataFrame({"op1": [-0.0003], "op2": [-0.0001], "op3": [100.0]})
    assert add_condition(df).iloc[0] == "0.0|0.0|100.0"


def test_rul_test_coherente(train):
    test = load_all("test")
    # La RUL au dernier cycle de chaque moteur de test = valeur de RUL_FDxxx.txt, donc > 0
    last = test.sort_values("cycle").groupby("engine_id").tail(1)
    assert (last["rul"] > 0).all()
    assert len(last) == 100 + 259 + 100 + 248


def test_chaque_moteur_dans_un_seul_pli(train, folds):
    assert folds["engine_id"].is_unique
    assert len(folds) == train["engine_id"].nunique() == 709
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}


def test_chaque_pli_contient_les_4_jeux(folds):
    for k in range(5):
        assert set(folds.loc[folds["fold"] == k, "subset"]) == set(SUBSETS)


def test_moyenne_zero_ecart_type_un_par_condition(train):
    scaler = ConditionScaler()
    scaled = scaler.fit_transform(train)
    for c in CONDITIONS:
        part = scaled.loc[train["cond"] == c, SENSORS_KEPT]
        assert np.allclose(part.mean(), 0, atol=1e-9)
        flat = scaler.flat.get(c, [])
        for s in SENSORS_KEPT:
            # un capteur constant dans la condition reste à 0 (écart-type nul), les autres valent 1
            assert np.isclose(part[s].std(ddof=0), 0.0 if s in flat else 1.0, atol=1e-9), (c, s)
    assert not scaled[SENSORS_KEPT].isna().any().any()


def test_capteurs_constants_detectes(train):
    scaler = ConditionScaler().fit(train)
    # s16 est constant dans 4 des 6 conditions (voir docs/exploration.md)
    assert sum("s16" in v for v in scaler.flat.values()) == 4


def test_reglages_entre_0_et_1(train):
    scaled = ConditionScaler().fit_transform(train)
    assert scaled[SETTINGS].min().min() >= 0 and scaled[SETTINGS].max().max() <= 1


def test_aller_retour_dict(train):
    scaler = ConditionScaler().fit(train)
    clone = ConditionScaler.from_dict(scaler.to_dict())
    pd.testing.assert_frame_equal(scaler.transform(train.head(500)), clone.transform(train.head(500)))


def test_condition_inconnue(train):
    scaler = ConditionScaler().fit(train)
    bad = train.head(5).copy()
    bad["cond"] = "99.0|0.5|50.0"
    with pytest.raises(ValueError):
        scaler.transform(bad)


def test_non_ajuste_leve_une_erreur(train):
    with pytest.raises(RuntimeError):
        ConditionScaler().transform(train.head(5))


def test_scaler_n_utilise_que_les_moteurs_d_entrainement(train, folds):
    val_ids = set(folds.loc[folds["fold"] == 0, "engine_id"])
    tr = train[~train["engine_id"].isin(val_ids)]
    scaler = ConditionScaler().fit(tr)
    # Mêmes statistiques que sur le sous-ensemble d'entraînement seul, et pas celles du tout
    full = ConditionScaler().fit(train)
    c = CONDITIONS[0]
    expected = tr.loc[tr["cond"] == c, SENSORS_KEPT].mean().to_numpy()
    assert np.allclose(scaler.mean[c], expected)
    assert not np.allclose(scaler.mean[c], full.mean[c])
    assert scaler.n_rows[c] == (tr["cond"] == c).sum()


def test_calcul_incremental_identique_a_pandas(train):
    # μ et σ calculés par blocs doivent égaler le calcul direct, quelle que soit la taille de bloc
    for block in (777, 2000, 10**7):
        scaler = ConditionScaler().fit(train, block=block)
        for c in CONDITIONS:
            rows = train.loc[train["cond"] == c, SENSORS_KEPT]
            assert np.allclose(scaler.mean[c], rows.mean().to_numpy(), atol=1e-9)
            expected = rows.std(ddof=0).to_numpy()
            got = scaler.std[c]
            flat = expected < 1e-8
            assert np.allclose(got[~flat], expected[~flat], atol=1e-9)


def test_on_block_appele_a_chaque_bloc(train):
    calls = []
    ConditionScaler().fit(train.head(5000), block=2000, on_block=lambda k, nb, seen, blk, snap: calls.append((k, nb, seen)))
    assert calls == [(0, 3, 2000), (1, 3, 4000), (2, 3, 5000)]


def test_transform_steps_coherent(train):
    scaler = ConditionScaler().fit(train)
    sub = train.head(300)
    centered, scaled = scaler.transform_steps(sub)
    assert np.allclose(scaled, scaler.transform(sub)[SENSORS_KEPT].to_numpy())
    assert np.allclose(centered, sub[SENSORS_KEPT].to_numpy() - scaler.mean[sub["cond"].iloc[0]])
