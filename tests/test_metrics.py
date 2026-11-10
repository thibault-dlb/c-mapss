import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmapss.metrics import bias, evaluate, evaluate_by_subset, late_rate, phm_penalty, phm_score, rmse


def test_score_trop_tot():
    # d = −10 : exp(10/13) − 1
    assert phm_score([90], [100]) == pytest.approx(np.exp(10 / 13) - 1)
    assert phm_score([90], [100]) == pytest.approx(1.1582, abs=1e-4)


def test_score_trop_tard():
    # d = +10 : exp(10/10) − 1
    assert phm_score([110], [100]) == pytest.approx(np.e - 1)
    assert phm_score([110], [100]) == pytest.approx(1.7183, abs=1e-4)


def test_erreur_nulle():
    assert phm_score([50, 80], [50, 80]) == 0.0


def test_tardif_plus_penalise_que_precoce():
    for err in (5, 10, 25, 50):
        assert phm_score([100 + err], [100]) > phm_score([100 - err], [100])


def test_le_score_est_une_somme():
    assert phm_score([90, 110], [100, 100]) == pytest.approx(np.exp(10 / 13) - 1 + np.e - 1)


def test_penalite_croit_avec_l_erreur():
    d = np.array([-30, -10, 0, 10, 30])
    p = phm_penalty(d)
    assert p[2] == 0 and p[1] > 0 and p[0] > p[1] and p[3] > 0 and p[4] > p[3]


def test_rmse():
    assert rmse([3, 4], [0, 0]) == pytest.approx(np.sqrt(12.5))
    assert rmse([5], [5]) == 0.0


def test_biais_et_part_de_tardifs():
    assert bias([90, 110, 120], [100, 100, 100]) == pytest.approx(20 / 3)
    assert bias([90], [100]) < 0           # trop tôt
    assert late_rate([90, 110, 120, 100], [100, 100, 100, 100]) == 0.5


def test_evaluate_et_par_jeu():
    pred, true = np.array([90, 110, 50, 60.0]), np.array([100, 100, 40, 60.0])
    sub = np.array(["FD001", "FD001", "FD003", "FD003"])
    out = evaluate_by_subset(pred, true, sub)
    assert set(out) == {"FD001", "FD003", "total"}
    assert out["FD001"]["n"] == 2 and out["total"]["n"] == 4
    assert out["total"]["score"] == pytest.approx(evaluate(pred, true)["score"])
    assert out["FD001"]["rmse"] == pytest.approx(10.0)


def test_formes_differentes():
    with pytest.raises(ValueError):
        rmse([1, 2], [1])
