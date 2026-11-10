"""Recherche d'hyperparamètres : espace des réglages, tirage reproductible, classement pli par pli.

Un **réglage** est un petit dictionnaire de valeurs qui s'écartent des valeurs par défaut (DEFAULT). Les réglages nommés
(NAMED) sont toujours testés ; les autres sont tirés au hasard dans SPACE avec une graine fixe, donc deux exécutions
donnent la même liste (et les N premiers réglages ne changent pas quand on augmente N).

Ce module ne dépend pas de PyTorch : il est importé aussi par le script et par les tests.
"""
import numpy as np

# Valeurs essayées pour chaque paramètre (le tirage choisit une valeur par paramètre)
SPACE = {
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "input_noise": [0.0, 0.05, 0.1, 0.2],         # écart-type du bruit sur les entrées normalisées
    "weight_decay": [0.0, 1e-3, 1e-2, 1e-1],        # AdamW si > 0
    "size": ["petit", "moyen", "grand"],            # voir SIZES
    "kernel": [3, 5, 7],                            # taille du noyau des convolutions
    "lr": [3e-4, 1e-3, 3e-3],
    "batch_size": [128, 256, 512],
    "lam": [0.0, 0.05, 0.2],                        # poids de la tête régime dans la perte
    "cap": [125, 150, 175],                         # plafond de RUL (la baseline donnait 150 et 175 indiscernables)
}
# Valeurs par défaut = réseau de l'essai de 12 époques (sans régularisation)
DEFAULT = {"dropout": 0.0, "input_noise": 0.0, "weight_decay": 0.0, "size": "moyen", "kernel": 5, "lr": 1e-3,
           "batch_size": 256, "lam": 0.05, "cap": 150}
# size -> (canaux des deux convolutions, unités du GRU)
SIZES = {"petit": ((16, 32), 32), "moyen": ((32, 64), 64), "grand": ((64, 128), 128)}

# Réglages nommés, toujours testés (« aucune » est la référence)
NAMED = {
    "aucune": {},
    "dropout": {"dropout": 0.3},
    "bruit": {"input_noise": 0.1},
    "poids": {"weight_decay": 1e-2},
    "petit": {"size": "petit"},
    "combinee": {"dropout": 0.3, "input_noise": 0.1, "weight_decay": 1e-2},
}


def full(overrides=None):
    """Tous les paramètres d'un réglage : DEFAULT, modifié par `overrides` (vérifié contre DEFAULT)."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(DEFAULT)
    if unknown:
        raise ValueError(f"paramètre inconnu : {sorted(unknown)} (connus : {sorted(DEFAULT)})")
    return {**DEFAULT, **overrides}


def minimal(overrides):
    """Ne garde que les valeurs qui diffèrent du défaut (forme canonique, pour comparer deux réglages)."""
    f = full(overrides)
    return {k: v for k, v in f.items() if v != DEFAULT[k]}


def describe(overrides):
    """Texte court d'un réglage : « dropout 0,3 · bruit 0,1 », ou « défaut »."""
    m = minimal(overrides)
    if not m:
        return "défaut"
    names = {"input_noise": "bruit", "weight_decay": "décroissance", "batch_size": "lot", "size": "taille", "lam": "λ"}
    return " · ".join(f"{names.get(k, k)} {str(v).replace('.', ',')}" for k, v in m.items())


def sample_configs(n_random, seed=0):
    """Liste `[(identifiant, réglage), ...]` : les réglages nommés, puis `n_random` réglages tirés au hasard et distincts.

    Les identifiants aléatoires sont r001, r002... Les n premiers tirages ne dépendent pas de `n_random`.
    """
    out = [(name, dict(o)) for name, o in NAMED.items()]
    seen = {tuple(sorted(minimal(o).items())) for _, o in out}
    rng = np.random.RandomState(seed)
    i = 0
    attempts = 0
    while i < n_random and attempts < 200 * max(n_random, 1):
        attempts += 1
        cand = {k: v[rng.randint(len(v))] for k, v in SPACE.items()}
        key = tuple(sorted(minimal(cand).items()))
        if key in seen:
            continue
        seen.add(key)
        i += 1
        out.append((f"r{i:03d}", minimal(cand)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Clés des résultats et classement
def run_key(stage, sensors, cid, fold, seed):
    return f"{stage}|s{sensors}|c{cid}|fold{'all' if fold is None else fold}|seed{seed}"


def parse_key(key):
    """Décompose une clé de `run_key` ; renvoie None pour une clé d'un ancien format (ignorée par les classements)."""
    parts = key.split("|")
    if len(parts) != 5 or not parts[1].startswith("s") or not parts[2].startswith("c") \
            or not parts[3].startswith("fold") or not parts[4].startswith("seed"):
        return None
    stage, sensors, cid, fold, seed = parts
    return {"stage": stage, "sensors": sensors[1:], "cid": cid[1:],
            "fold": None if fold == "foldall" else int(fold[4:]), "seed": int(seed[4:])}


def by_fold(runs, stage, sensors, cid, seeds=None):
    """Résultats d'un réglage rangés par pli : {pli: [un résultat par graine]}."""
    out = {}
    for k, r in runs.items():
        p = parse_key(k)
        if p is not None and p["stage"] == stage and p["sensors"] == sensors and p["cid"] == cid and p["fold"] is not None \
                and (seeds is None or p["seed"] in seeds):
            out.setdefault(p["fold"], []).append(r)
    return out


def ranking(runs, stage, sensors, cids, folds, reference="aucune", seed=0):
    """Classe les réglages par RMSE de validation moyen sur `folds` (graine `seed`). Ne garde que les réglages complets.

    Chaque ligne : cid, n_folds, rmse_mean, rmse_std (entre plis), diff_mean et diff_std (pli par pli contre `reference`,
    sur les plis communs), score_mean, regime_acc, best_epoch. Trié du meilleur au moins bon.
    """
    folds = list(folds)
    table = {}
    for cid in cids:
        f = by_fold(runs, stage, sensors, cid, seeds=[seed])
        if all(k in f for k in folds):
            table[cid] = {k: f[k][0] for k in folds}
    ref = table.get(reference)
    rows = []
    for cid, per in table.items():
        rm = np.array([per[k]["val"]["rmse"] for k in folds])
        row = {"cid": cid, "n_folds": len(folds), "rmse_mean": float(rm.mean()), "rmse_std": float(rm.std()),
               "score_mean": float(np.mean([per[k]["val"]["score"] for k in folds])),
               "regime_acc": float(np.mean([per[k]["val"]["regime_acc"] for k in folds])),
               "best_epoch": float(np.mean([per[k]["best_epoch"] for k in folds]))}
        if ref is not None and cid != reference:
            d = rm - np.array([ref[k]["val"]["rmse"] for k in folds])
            row["diff_mean"], row["diff_std"] = float(d.mean()), float(d.std())
        rows.append(row)
    return sorted(rows, key=lambda r: r["rmse_mean"])
