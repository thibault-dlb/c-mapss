"""Exploration des données C-MAPSS : capteurs inutiles, conditions de vol, dégradation.

Usage : .\\.venv\\Scripts\\python.exe scripts\\explore.py
Écrit les figures dans reports/figures/ et affiche un résumé chiffré.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + SETTINGS + SENSORS
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def load_train(fd):
    df = pd.read_csv(RAW / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    df["subset"] = fd
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def add_condition(df):
    """Condition de vol = réglages arrondis (1 seule pour FD001/FD003, 6 pour FD002/FD004)."""
    # "+ 0.0" supprime les zéros négatifs (-0.0), qui créeraient de fausses conditions
    parts = [np.round(df["op1"], 0) + 0.0, np.round(df["op2"], 2) + 0.0, np.round(df["op3"], 0) + 0.0]
    return parts[0].astype(str) + "|" + parts[1].astype(str) + "|" + parts[2].astype(str)


def main():
    frames = [load_train(fd) for fd in SUBSETS]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["cond"] = add_condition(all_df)

    print("=== Valeurs manquantes ===")
    print("NaN au total :", int(all_df[COLS].isna().sum().sum()))

    print("\n=== Conditions de vol par sous-ensemble ===")
    for fd, g in all_df.groupby("subset"):
        print(fd, ":", g["cond"].nunique(), "conditions")

    # --- Capteurs constants / quasi constants, par sous-ensemble ---
    print("\n=== Écart-type de chaque capteur, par sous-ensemble (brut) ===")
    std = all_df.groupby("subset")[SENSORS].std().T
    print(std.round(4).to_string())

    # Normalisation par condition SEULE (le jeu n'est pas connu au test), statistiques pooled
    norm = all_df.copy()
    grp = norm.groupby("cond")[SENSORS]
    norm[SENSORS] = (norm[SENSORS] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    std_norm = norm.groupby("subset")[SENSORS].std().T
    print("\n=== Écart-type après normalisation par condition ===")
    print(std_norm.round(4).to_string())

    # --- Corrélation de chaque capteur (normalisé) avec la RUL plafonnée ---
    norm["rul_cap"] = norm["rul"].clip(upper=125)
    corr = pd.DataFrame(
        {fd: g[SENSORS].corrwith(g["rul_cap"]) for fd, g in norm.groupby("subset")}
    )
    corr["tous"] = norm[SENSORS].corrwith(norm["rul_cap"])
    print("\n=== Corrélation capteur normalisé / RUL plafonnée à 125 ===")
    print(corr.round(3).to_string())

    # Capteurs jugés utiles : écart-type normalisé non négligeable dans les 4 jeux
    # Constant = écart-type nul À L'INTÉRIEUR de chaque condition de vol, dans les 4 jeux
    within = all_df.groupby(["subset", "cond"])[SENSORS].std().max()
    constant = [s for s in SENSORS if within[s] < 1e-6]
    # Faible = |r| < 0,1 dans CHAQUE jeu (la corrélation poolée masque les changements de signe)
    weak = [s for s in SENSORS if s not in constant and corr.loc[s, SUBSETS].abs().max() < 0.1]
    useful = [s for s in SENSORS if s not in constant and s not in weak]
    print("\nCapteurs constants dans chaque condition :", constant)
    print("Capteurs faiblement corrélés à la RUL dans les 4 jeux (|r| < 0,1) :", weak)
    print("Capteurs retenus pour l'instant :", useful, f"({len(useful)})")

    # --- Longueur des trajectoires ---
    print("\n=== Durée de vie (cycles) par sous-ensemble ===")
    life = all_df.groupby(["subset", "unit"])["cycle"].max().groupby("subset").describe()
    print(life.round(1).to_string())

    # --- Figures ---
    # 1. Dégradation de quelques capteurs, FD001 (1 condition) : brut
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=False)
    f1 = all_df[all_df["subset"] == "FD001"]
    for ax, s in zip(axes.ravel(), ["s2", "s3", "s4", "s7", "s11", "s12"]):
        for u in f1["unit"].unique()[:10]:
            g = f1[f1["unit"] == u]
            ax.plot(g["rul"], g[s], lw=0.8, alpha=0.7)
        ax.set_title(f"FD001 – {s}")
        ax.invert_xaxis()
        ax.set_xlabel("RUL (cycles)")
    fig.suptitle("Dégradation de capteurs (10 premiers moteurs de FD001)")
    fig.tight_layout()
    fig.savefig(FIG / "degradation_fd001.png", dpi=120)
    plt.close(fig)

    # 2. FD002 : même capteur brut puis normalisé par condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    f2 = all_df[all_df["subset"] == "FD002"]
    n2 = norm[norm["subset"] == "FD002"]
    u = f2["unit"].unique()[0]
    axes[0].plot(f2[f2["unit"] == u]["cycle"], f2[f2["unit"] == u]["s7"], lw=0.7)
    axes[0].set_title("FD002 – s7 brut (moteur 1) : bruité par les conditions")
    axes[1].plot(n2[n2["unit"] == u]["cycle"], n2[n2["unit"] == u]["s7"], lw=0.7)
    axes[1].set_title("FD002 – s7 normalisé par condition")
    for ax in axes:
        ax.set_xlabel("cycle")
    fig.tight_layout()
    fig.savefig(FIG / "normalisation_fd002.png", dpi=120)
    plt.close(fig)

    # 3. Corrélation avec la RUL
    fig, ax = plt.subplots(figsize=(9, 4))
    corr.drop(columns="tous").plot.bar(ax=ax, width=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("corrélation avec RUL (plafond 125)")
    ax.set_title("Corrélation des capteurs normalisés avec la RUL")
    fig.tight_layout()
    fig.savefig(FIG / "correlation_rul.png", dpi=120)
    plt.close(fig)

    # 4. Distribution des durées de vie
    fig, ax = plt.subplots(figsize=(8, 4))
    for fd, g in all_df.groupby("subset"):
        ax.hist(g.groupby("unit")["cycle"].max(), bins=20, alpha=0.5, label=fd)
    ax.set_xlabel("durée de vie (cycles)")
    ax.set_ylabel("nombre de moteurs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "durees_de_vie.png", dpi=120)
    plt.close(fig)

    print(f"\nFigures écrites dans {FIG}")


if __name__ == "__main__":
    main()
