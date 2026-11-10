"""Métriques d'évaluation de la RUL : RMSE, score asymétrique de PHM08, biais.

Score de PHM08 (Saxena et al. 2008, équation 11) avec d = RUL estimée − RUL réelle :
    d < 0 (trop tôt)  : exp(−d / 13) − 1
    d ≥ 0 (trop tard) : exp( d / 10) − 1
Les prédictions tardives sont plus pénalisées que les précoces. Le score est une somme (pas une moyenne) : il dépend du
nombre de moteurs et quelques erreurs tardives suffisent à le dominer. Voir docs/article-saxena-2008.md.
"""
import numpy as np

A_EARLY = 13.0  # dénominateur de la branche « trop tôt » (d < 0)
A_LATE = 10.0   # dénominateur de la branche « trop tard » (d ≥ 0), plus petit donc plus sévère

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def _arrays(pred, true):
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"formes différentes : {pred.shape} et {true.shape}")
    return pred, true


def phm_penalty(d):
    """Pénalité de chaque erreur d = estimée − réelle (tableau). C'est ce que `phm_score` additionne."""
    d = np.asarray(d, dtype=float)
    return np.where(d < 0, np.exp(-d / A_EARLY) - 1.0, np.exp(d / A_LATE) - 1.0)


def phm_score(pred, true):
    """Score de PHM08 : somme des pénalités. Plus petit = meilleur ; 0 = parfait."""
    pred, true = _arrays(pred, true)
    return float(phm_penalty(pred - true).sum())


def rmse(pred, true):
    pred, true = _arrays(pred, true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def bias(pred, true):
    """Erreur moyenne signée (estimée − réelle) : négatif = trop tôt en moyenne, positif = trop tard."""
    pred, true = _arrays(pred, true)
    return float(np.mean(pred - true))


def late_rate(pred, true):
    """Part des prédictions tardives (d > 0)."""
    pred, true = _arrays(pred, true)
    return float(np.mean(pred > true))


def evaluate(pred, true):
    """Toutes les métriques d'un coup."""
    pred, true = _arrays(pred, true)
    return {"n": int(pred.size), "rmse": rmse(pred, true), "score": phm_score(pred, true),
            "bias": bias(pred, true), "late_rate": late_rate(pred, true)}


def evaluate_by_subset(pred, true, subsets):
    """Un dict par jeu (FD001 à FD004) et un pour l'ensemble (clé « total »). `subsets` : un jeu par prédiction."""
    pred, true, subsets = _arrays(pred, true)[0], _arrays(pred, true)[1], np.asarray(subsets)
    out = {fd: evaluate(pred[subsets == fd], true[subsets == fd]) for fd in SUBSETS if (subsets == fd).any()}
    out["total"] = evaluate(pred, true)
    return out
