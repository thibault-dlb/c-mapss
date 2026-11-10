"""Tests de la perte, de l'arrêt précoce et de la boucle d'entraînement (données synthétiques, quelques secondes)."""
import json

import numpy as np
import pytest
import torch

from cmapss.journal import Journal
from cmapss.model import CnnGru, ModelConfig
from cmapss.train import EarlyStopping, RunData, TrainConfig, multitask_loss, predict, train_run


def test_perte_multitache_calculee_a_la_main():
    r, y = torch.tensor([0.5, 1.0]), torch.tensor([0.0, 1.0])
    logit, reg = torch.tensor([0.0, 0.0]), torch.tensor([0, 1])
    total, mse, bce = multitask_loss(r, y, logit, reg, lam=0.1)
    assert float(mse) == pytest.approx(0.125)                 # (0,25 + 0) / 2
    assert float(bce) == pytest.approx(np.log(2))              # sigmoïde(0) = 0,5 pour les deux
    assert float(total) == pytest.approx(0.125 + 0.1 * np.log(2))


def test_lambda_zero_la_tete_regime_ne_recoit_aucun_gradient():
    m = CnnGru(ModelConfig(n_features=4))
    r, l = m(torch.randn(8, 30, 4))
    loss, _, _ = multitask_loss(r, torch.rand(8), l, torch.randint(0, 2, (8,)), lam=0.0)
    loss.backward()
    assert float(m.head_regime.weight.grad.abs().sum()) == 0.0
    assert float(m.head_rul.weight.grad.abs().sum()) > 0.0


def test_arret_precoce_patience_et_meilleure_epoque_restauree():
    m = CnnGru(ModelConfig(n_features=4))
    es = EarlyStopping(patience=2)
    seq = [5.0, 4.0, 4.5, 4.2]                                 # meilleure valeur à l'époque 2
    stops = []
    for e, v in enumerate(seq, 1):
        if e == 2:
            with torch.no_grad():
                m.head_rul.bias.fill_(7.0)                     # l'état de l'époque 2 est reconnaissable
        if e == 3:
            with torch.no_grad():
                m.head_rul.bias.fill_(-1.0)
        stops.append(es.step(v, e, m))
    assert stops == [False, False, False, True] and es.best_epoch == 2
    es.restore(m)
    assert float(m.head_rul.bias.detach()) == 7.0


def _donnees(n=600, nf=4, seed=0):
    """RUL = une fonction simple de la dernière valeur du canal 0 ; régime = signe du canal 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 30, nf)).astype(np.float32)
    rul = np.clip(75 + 40 * X[:, -1, 0], 0, 150).astype(np.float32)
    reg = (X[:, :, 1].mean(1) > 0).astype(np.int64)
    return X, rul, reg


def _run_data(with_probe=True):
    X, rul, reg = _donnees()
    Xv, rv, gv = _donnees(n=200, seed=1)
    probe = {"engine": "FD001_1", "X": Xv[:20], "rul": rv[:20], "end_cycle": np.arange(30, 50), "X_act": Xv[:32]} if with_probe else None
    return RunData(X, rul, reg, Xv, rv, gv, np.array(["FD001"] * len(Xv)), probe)


def test_mini_entrainement_fait_baisser_la_perte(tmp_path):
    cfg = TrainConfig(max_epochs=6, patience=6, batch_size=64, seed=0, lam=0.05, log_every=3)
    res, _ = train_run(_run_data(), cfg, journal=Journal(tmp_path / "j.jsonl"))
    h = res["history"]
    assert h[-1]["train_loss"] < h[0]["train_loss"]
    assert res["val"]["rmse"] < 45                             # un modèle constant donnerait environ 40 ; l'ordre de grandeur suffit
    assert 1 <= res["best_epoch"] <= 6


def test_meme_graine_memes_resultats():
    cfg = TrainConfig(max_epochs=3, patience=3, batch_size=64, seed=3)
    a, _ = train_run(_run_data(False), cfg)
    b, _ = train_run(_run_data(False), cfg)
    assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]


def test_journal_valide_et_ordre_des_evenements(tmp_path):
    p = tmp_path / "j.jsonl"
    cfg = TrainConfig(max_epochs=3, patience=3, batch_size=64, seed=0, log_every=2)
    train_run(_run_data(), cfg, journal=Journal(p))
    ev = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in ev] == list(range(len(ev)))      # contigu
    types = [e["type"] for e in ev]
    assert types[0] == "journal" and types[1] == "meta" and types[2] == "probe" and types[-1] == "end"
    assert types.count("epoch") == 3 and types.count("prediction") == 4 and types.count("layers") == 4   # époque 0 + 3
    assert all(e["ok"] for e in ev if e["type"] == "check")
    first_epoch = types.index("epoch")
    assert all(t in ("layers", "prediction", "probe", "meta", "journal") for t in types[:first_epoch] if t != "batch")
    meta = ev[1]
    assert sum(l["params"] for l in meta["layers"]) == meta["n_params"]
    ep = next(e for e in ev if e["type"] == "epoch")
    assert {"train_loss", "val_loss", "rmse", "score", "regime_acc", "seconds"} <= set(ep)


def test_epoques_imposees_sans_validation():
    X, rul, reg = _donnees()
    data = RunData(X, rul, reg)
    res, model = train_run(data, TrainConfig(batch_size=64), fixed_epochs=2)
    assert res["epochs_run"] == 2 and "val" not in res
    assert predict(model, X[:5])[0].shape == (5,)
