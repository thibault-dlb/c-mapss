import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmapss.data import RAW, load_test, load_train
from cmapss.normalisation import ConditionScaler
from cmapss.windows import FEATURES, WINDOW, cap_rul, make_windows, n_windows, pad_first


def synthetic(engines):
    """engines : {engine_id: (jeu, nb de cycles, valeur de base)}. Chaque feature vaut base + numéro de cycle."""
    parts = []
    for eid, (subset, n, base) in engines.items():
        cyc = np.arange(1, n + 1)
        d = pd.DataFrame({f: base + cyc for f in FEATURES})
        d["engine_id"], d["subset"], d["cycle"], d["rul"] = eid, subset, cyc, n - cyc
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_formule_du_nombre_de_fenetres():
    assert n_windows(200) == 171 and n_windows(30) == 1 and n_windows(29) == 0 and n_windows(5) == 0


def test_train_contenu_et_etiquette():
    w = make_windows(synthetic({"A": ("FD001", 40, 0)}), "train")
    assert len(w) == 11 and w.X.shape == (11, 30, 20) and w.X.dtype == np.float32
    # première fenêtre : cycles 1 à 30 ; dernière : cycles 11 à 40
    assert np.all(w.X[0][:, 0] == np.arange(1, 31)) and np.all(w.X[-1][:, 0] == np.arange(11, 41))
    assert list(w.end_cycle[[0, -1]]) == [30, 40]
    assert list(w.rul[[0, -1]]) == [10, 0]   # RUL au dernier cycle de la fenêtre
    assert not w.n_padded.any()


def test_fenetres_voisines_partagent_29_cycles():
    w = make_windows(synthetic({"A": ("FD001", 60, 0)}), "train")
    assert np.array_equal(w.X[1][:-1], w.X[0][1:])


def test_aucune_fenetre_ne_chevauche_deux_moteurs():
    w = make_windows(synthetic({"A": ("FD001", 35, 0), "B": ("FD003", 40, 1000)}), "train")
    assert len(w) == 6 + 11
    for X, e in zip(w.X, w.engine_id):
        assert (X[:, 0] < 500).all() if e == "A" else (X[:, 0] > 500).all()
    assert set(w.regime[w.engine_id == "A"]) == {0} and set(w.regime[w.engine_id == "B"]) == {1}


def test_moteur_trop_court_donne_zero_fenetre_a_l_entrainement():
    assert len(make_windows(synthetic({"A": ("FD001", 29, 0)}), "train")) == 0


def test_test_une_fenetre_par_moteur_avec_remplissage():
    df = synthetic({"court": ("FD004", 19, 0), "juste": ("FD001", 30, 0), "long": ("FD002", 100, 0)})
    w = make_windows(df, "test")
    assert len(w) == 3 and list(w.n_padded) == [11, 0, 0]
    court = w.X[0]
    assert np.all(court[:11, 0] == 1) and np.all(court[11:, 0] == np.arange(1, 20))   # premier cycle répété
    assert np.all(w.X[2][:, 0] == np.arange(71, 101))                                  # 30 derniers cycles
    assert list(w.end_cycle) == [19, 30, 100]


def test_pad_first():
    rows = np.arange(6, dtype=np.float32).reshape(3, 2)
    w, missing = pad_first(rows, 5)
    assert missing == 2 and w.shape == (5, 2) and np.array_equal(w[0], w[1]) and np.array_equal(w[1], rows[0])


def test_on_window_est_appele_pour_chaque_fenetre():
    calls = []
    w = make_windows(synthetic({"A": ("FD001", 33, 0)}), "train",
                     on_window=lambda eid, a, b, rul, pad, win: calls.append((eid, a, b, rul, pad, win.shape)))
    assert len(calls) == len(w) == 4
    assert calls[0] == ("A", 1, 30, 3.0, 0, (30, 20)) and calls[-1][1:5] == (4, 33, 0.0, 0)


def test_cap_rul():
    assert list(cap_rul(np.array([10, 125, 300]), 125)) == [10, 125, 125]


def test_mauvais_type():
    with pytest.raises(ValueError):
        make_windows(synthetic({"A": ("FD001", 40, 0)}), "autre")


@pytest.fixture(scope="module")
def fd001():
    train, test = load_train("FD001"), load_test("FD001")
    s = ConditionScaler().fit(train)
    return s.transform(train), s.transform(test)


def test_donnees_reelles_fd001(fd001):
    train, test = fd001
    w = make_windows(train, "train")
    assert len(w) == 20631 - 100 * (WINDOW - 1) == sum(n_windows(n) for n in train.groupby("engine_id").size())
    assert w.X.shape[1:] == (30, 20) and not np.isnan(w.X).any()
    wt = make_windows(test, "test")
    assert len(wt) == 100
    # l'étiquette d'un moteur de test est la RUL du fichier RUL_FD001.txt
    expected = pd.read_csv(RAW / "RUL_FD001.txt", header=None)[0].to_numpy()
    assert np.array_equal(wt.rul, expected)


def test_journal_des_fenetres_coherent():
    """Le journal livré avec la page doit être contigu, complet, et ses effectifs doivent suivre la formule."""
    import json
    path = Path(__file__).resolve().parent.parent / "viz" / "fenetres" / "journal.jsonl"
    ev = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in ev] == list(range(len(ev))) and ev[-1]["type"] == "end"
    by = lambda t: [e for e in ev if e["type"] == t]
    eng, wins, counts = by("engine")[0], by("window"), by("counts")[0]
    assert len(wins) == n_windows(eng["n_cycles"])
    assert wins[0]["start"] == 1 and wins[0]["end"] == WINDOW and wins[-1]["end"] == eng["n_cycles"]
    assert all(w["padded"] == 0 for w in wins) and wins[-1]["rul"] == 0
    assert [w["end"] - w["start"] + 1 for w in wins] == [WINDOW] * len(wins)
    short = by("short_engine")[0]
    assert short["n_padded"] == WINDOW - short["n_cycles"] > 0
    assert short["x_window"][: short["n_padded"]] == [short["x_window"][short["n_padded"]]] * short["n_padded"]
    assert counts["train_windows"] == 139798 and counts["test_windows"] == 707
    split = by("split")[0]
    assert split["train_engines"] + split["val_engines"] == 709
    assert all(c["ok"] for c in by("check"))
