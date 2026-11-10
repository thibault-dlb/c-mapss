"""Réseau CNN 1D + GRU à deux sorties : RUL (plafonnée, divisée par le plafond) et régime de panne (logit).

Entrée  : (lot, WINDOW, n_features), les fenêtres normalisées de `cmapss.windows`.
Sorties : `rul` (lot,) dans l'unité « RUL / plafond » (multiplier par le plafond pour revenir en cycles) et
          `logit` (lot,) du régime (0 = HPC seul, 1 = deux modes possibles).

La **structure** est fixe pendant l'entraînement ; ce qui évolue, ce sont les poids et les activations.
"""
from dataclasses import dataclass, asdict

import torch
from torch import nn

from .windows import WINDOW


@dataclass
class ModelConfig:
    n_features: int = 20
    channels: tuple = (32, 64)
    kernel: int = 5
    hidden: int = 64
    window: int = WINDOW
    dropout: float = 0.0      # probabilité d'éteindre un neurone pendant l'entraînement (0 = aucun)

    def to_dict(self):
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


class CnnGru(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        c1, c2 = cfg.channels
        # Le dropout, sans paramètre, suit chaque bloc de convolution et précède les têtes ; à 0 il ne fait rien.
        self.conv1 = nn.Sequential(nn.Conv1d(cfg.n_features, c1, cfg.kernel, padding="same"), nn.BatchNorm1d(c1), nn.ReLU(),
                                   nn.Dropout(cfg.dropout))
        self.conv2 = nn.Sequential(nn.Conv1d(c1, c2, cfg.kernel, padding="same"), nn.BatchNorm1d(c2), nn.ReLU(),
                                   nn.Dropout(cfg.dropout))
        self.drop_head = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(c2, cfg.hidden, batch_first=True)
        self.head_rul = nn.Linear(cfg.hidden, 1)
        self.head_regime = nn.Linear(cfg.hidden, 1)

    def forward(self, x, return_acts=False):
        a1 = self.conv1(x.transpose(1, 2))             # (lot, c1, T)
        a2 = self.conv2(a1)                             # (lot, c2, T)
        _, h = self.gru(a2.transpose(1, 2))             # h : (1, lot, hidden)
        h = self.drop_head(h[-1])
        rul, logit = self.head_rul(h).squeeze(-1), self.head_regime(h).squeeze(-1)
        if return_acts:
            return rul, logit, {"conv1": a1, "conv2": a2, "gru": h}
        return rul, logit


# Modules qui portent des poids, dans l'ordre du schéma (nom, module)
def weight_layers(model):
    return [("conv1", model.conv1[0]), ("conv2", model.conv2[0]), ("gru", model.gru),
            ("head_rul", model.head_rul), ("head_regime", model.head_regime)]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def describe(model):
    """Couches du schéma : nom, type, forme de sortie pour UNE fenêtre, nombre de paramètres (vérifié par les tests)."""
    c = model.cfg
    shape = {"conv1": [c.channels[0], c.window], "conv2": [c.channels[1], c.window], "gru": [c.hidden],
             "head_rul": [1], "head_regime": [1]}
    kind = {"conv1": "Conv1d + BatchNorm + ReLU", "conv2": "Conv1d + BatchNorm + ReLU", "gru": "GRU",
            "head_rul": "Linear", "head_regime": "Linear"}
    mods = {"conv1": model.conv1, "conv2": model.conv2, "gru": model.gru,
            "head_rul": model.head_rul, "head_regime": model.head_regime}
    out = [{"name": "input", "kind": "entrée", "shape": [c.window, c.n_features], "params": 0}]
    for name in ("conv1", "conv2", "gru", "head_rul", "head_regime"):
        out.append({"name": name, "kind": kind[name], "shape": shape[name],
                    "params": sum(p.numel() for p in mods[name].parameters())})
    return out
