"""Chargement des fichiers C-MAPSS et constantes communes."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + SETTINGS + SENSORS

# Capteurs constants à l'intérieur de chaque condition de vol (voir docs/exploration.md)
SENSORS_CONSTANT = ["s1", "s5", "s18", "s19"]
SENSORS_KEPT = [s for s in SENSORS if s not in SENSORS_CONSTANT]  # 17 capteurs

# Les 6 conditions de vol, communes aux 4 jeux (FD001 et FD003 n'ont que la première).
# Clé = "altitude|Mach|TRA" après arrondi.
CONDITIONS = [
    "0.0|0.0|100.0",
    "10.0|0.25|100.0",
    "20.0|0.7|100.0",
    "25.0|0.62|60.0",
    "35.0|0.84|100.0",
    "42.0|0.84|100.0",
]


def add_condition(df):
    """Clé de condition de vol : réglages arrondis (FD001 et FD003 : 1 seule, FD002 et FD004 : 6)."""
    # "+ 0.0" supprime les zéros négatifs (-0.0), qui créeraient de fausses conditions
    parts = [np.round(df["op1"], 0) + 0.0, np.round(df["op2"], 2) + 0.0, np.round(df["op3"], 0) + 0.0]
    return parts[0].astype(str) + "|" + parts[1].astype(str) + "|" + parts[2].astype(str)


def _read(path, subset):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["subset"] = subset
    df["engine_id"] = subset + "_" + df["unit"].astype(str)  # unique entre les jeux
    df["cond"] = add_condition(df)
    return df


def load_train(subset):
    """Trajectoires d'entraînement. `rul` = cycles restants avant la panne (non plafonnée)."""
    df = _read(RAW / f"train_{subset}.txt", subset)
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def load_test(subset):
    """Trajectoires de test (arrêtées avant la panne), avec la vraie RUL de `RUL_FDxxx.txt`.

    `rul` = RUL réelle à chaque cycle = RUL finale du fichier + cycles restants dans la trajectoire.
    """
    df = _read(RAW / f"test_{subset}.txt", subset)
    final = pd.read_csv(RAW / f"RUL_{subset}.txt", header=None)[0].to_numpy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last - df["cycle"] + df["unit"].map(lambda u: final[u - 1])
    return df


def load_all(kind="train"):
    """Les 4 jeux concaténés. kind = 'train' ou 'test'."""
    loader = load_train if kind == "train" else load_test
    return pd.concat([loader(fd) for fd in SUBSETS], ignore_index=True)
