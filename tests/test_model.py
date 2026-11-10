"""Tests du réseau CNN 1D + GRU (formes, paramètres, déterminisme)."""
import numpy as np
import torch

from cmapss.model import CnnGru, ModelConfig, describe, n_params, weight_layers


def test_deux_sorties_de_la_bonne_forme():
    m = CnnGru(ModelConfig(n_features=20))
    rul, logit = m(torch.randn(7, 30, 20))
    assert rul.shape == (7,) and logit.shape == (7,)


def test_nombre_de_capteurs_variable():
    for nf in (17, 20):
        rul, logit = CnnGru(ModelConfig(n_features=nf))(torch.randn(3, 30, nf))
        assert rul.shape == (3,)


def test_describe_compte_les_vrais_parametres():
    m = CnnGru(ModelConfig(n_features=20))
    d = describe(m)
    assert sum(x["params"] for x in d) == n_params(m)
    assert [x["name"] for x in d] == ["input", "conv1", "conv2", "gru", "head_rul", "head_regime"]
    assert d[1]["shape"] == [32, 30] and d[3]["shape"] == [64]


def test_return_acts_donne_les_activations_des_trois_blocs():
    m = CnnGru(ModelConfig(n_features=20))
    _, _, acts = m(torch.randn(4, 30, 20), return_acts=True)
    assert acts["conv1"].shape == (4, 32, 30) and acts["conv2"].shape == (4, 64, 30) and acts["gru"].shape == (4, 64)
    assert (acts["conv1"] >= 0).all()          # sortie de ReLU


def test_meme_graine_memes_poids_graines_differentes_poids_differents():
    def w(seed):
        torch.manual_seed(seed)
        return CnnGru(ModelConfig(n_features=20)).head_rul.weight.detach().clone()
    assert torch.equal(w(1), w(1)) and not torch.equal(w(1), w(2))


def test_couches_a_poids():
    m = CnnGru(ModelConfig(n_features=20))
    assert [n for n, _ in weight_layers(m)] == ["conv1", "conv2", "gru", "head_rul", "head_regime"]
