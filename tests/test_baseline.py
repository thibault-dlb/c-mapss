import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmapss.baseline import STATS, columns_for, feature_names, make_classifier, make_regressor, window_features


def ramp(a=2.0, b=5.0, n=4, w=30, f=3):
    t = np.arange(w, dtype=np.float32)
    return np.tile((a * t + b)[None, :, None], (n, 1, f)).astype(np.float32)


def test_statistiques_d_une_rampe_de_pente_connue():
    out = window_features(ramp(a=2.0, b=5.0)).reshape(4, 3, len(STATS))
    mean, std, mn, mx, last, slope = (out[0, 0, i] for i in range(6))
    assert slope == pytest.approx(2.0, abs=1e-4)
    assert mn == pytest.approx(5.0) and mx == pytest.approx(5.0 + 2.0 * 29) and last == mx
    assert mean == pytest.approx(5.0 + 2.0 * 14.5, abs=1e-4)
    assert std == pytest.approx(2.0 * np.sqrt((30 ** 2 - 1) / 12), abs=1e-3)


def test_forme_et_noms():
    X = ramp(f=20)
    assert window_features(X).shape == (4, 120)
    names = feature_names([f"c{i}" for i in range(20)])
    assert len(names) == 120 and names[:6] == [f"c0_{s}" for s in STATS]


def test_retirer_un_capteur_equivaut_a_refaire_les_variables():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 30, 5)).astype(np.float32)
    feats = ["a", "b", "c", "d", "e"]
    keep = ["a", "c", "e"]
    idx = [i for i, c in enumerate(feats) if c in keep]
    full = window_features(X)[:, columns_for(feats, keep)]
    assert full.shape == (7, 18)
    assert np.allclose(full, window_features(X[:, :, idx]))


def test_les_modeles_s_ajustent_sur_de_petites_donnees():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = X[:, 0] * 3 + rng.normal(scale=0.1, size=300)
    reg = make_regressor(max_iter=30).fit(X, y)
    assert np.isfinite(reg.predict(X)).all() and np.corrcoef(reg.predict(X), y)[0, 1] > 0.95
    clf = make_classifier(max_iter=30).fit(X, (X[:, 1] > 0).astype(int))
    assert (clf.predict(X) == (X[:, 1] > 0)).mean() > 0.9
