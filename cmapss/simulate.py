"""SIMULATION d'un entraînement : AUCUN réseau n'est entraîné, toutes les valeurs sont fictives.

Sert à valider l'interface et l'orchestration (`scripts/train.py --fake`) sans lancer d'entraînement. Les journaux ont le même
format que ceux de `cmapss/train.py`, mais portent `device = "simulation"` et la page affiche un bandeau « SIMULATION ».

Modèle fictif : la qualité d'un réglage est une somme d'effets plausibles (le dropout aide, un taux d'apprentissage trop fort
nuit...) plus un bruit propre à (réglage, pli, graine). Rien de tout cela n'est une mesure.
"""
import json
import time
import zlib

import numpy as np

from .metrics import evaluate_by_subset
from .search import full, minimal

_EFFECTS = {
    "dropout": {0.0: 0.0, 0.1: -0.7, 0.2: -1.2, 0.3: -1.5, 0.5: -0.9},
    "input_noise": {0.0: 0.0, 0.05: -0.5, 0.1: -0.9, 0.2: -0.7},
    "weight_decay": {0.0: 0.0, 1e-3: -0.1, 1e-2: -0.4, 1e-1: 0.4},
    "size": {"petit": 0.4, "moyen": 0.0, "grand": -0.3},
    "kernel": {3: 0.2, 5: 0.0, 7: -0.15},
    "lr": {3e-4: 0.3, 1e-3: 0.0, 3e-3: 0.5},
    "batch_size": {128: -0.1, 256: 0.0, 512: 0.25},
    "lam": {0.0: 0.2, 0.05: 0.0, 0.2: 0.1},
    "cap": {125: 0.6, 150: 0.0, 175: 0.15},
}
_SIZES = {"petit": ((16, 32), 32), "moyen": ((32, 64), 64), "grand": ((64, 128), 128)}
_FOLD_OFFSET = [-0.2, 0.15, 0.0, 0.25, -0.1]


def _rng(overrides, fold, seed):
    h = zlib.crc32(json.dumps([sorted(minimal(overrides).items()), fold, seed], default=str).encode())
    return np.random.RandomState(h)


def fake_layers(n_features, size, kernel):
    """Couches et paramètres du réseau (mêmes formules que cmapss/model.py)."""
    (c1, c2), h = _SIZES[size]
    conv1 = n_features * c1 * kernel + c1 + 2 * c1
    conv2 = c1 * c2 * kernel + c2 + 2 * c2
    gru = 3 * (c2 * h + h * h + 2 * h)
    return [{"name": "input", "kind": "entrée", "shape": [30, n_features], "params": 0},
            {"name": "conv1", "kind": "Conv1d + BatchNorm + ReLU", "shape": [c1, 30], "params": conv1},
            {"name": "conv2", "kind": "Conv1d + BatchNorm + ReLU", "shape": [c2, 30], "params": conv2},
            {"name": "gru", "kind": "GRU", "shape": [h], "params": gru},
            {"name": "head_rul", "kind": "Linear", "shape": [1], "params": h + 1},
            {"name": "head_regime", "kind": "Linear", "shape": [1], "params": h + 1}], (c1, c2), h


def expected_rmse(overrides, fold, seed):
    f = full(overrides)
    eff = sum(_EFFECTS[k][f[k]] for k in _EFFECTS)
    if f["size"] == "grand" and f["dropout"] >= 0.3 and f["kernel"] != 3:     # interaction fictive : la simulation a un meilleur réglage à trouver
        eff -= 1.6 + (0.4 if f["lr"] == 3e-4 else 0.0)
    rng = _rng(overrides, fold if fold is not None else 9, seed)
    return max(21.5, 27.0 + eff + _FOLD_OFFSET[fold % 5 if fold is not None else 0] + rng.normal(0, 0.35))


def simulate_run(overrides, fold, seed, journal=None, sec_per_epoch=0.0, max_epochs=50, patience=8, fixed_epochs=None,
                 meta_extra=None, label="", preset="", n_features=20):
    """Simule un entraînement : écrit un journal réaliste (avec des pauses `sec_per_epoch`) et renvoie un résultat de même forme."""
    f = full(overrides)
    rng = _rng(overrides, fold, seed)
    best_rmse = expected_rmse(overrides, fold, seed)
    reg = (f["dropout"] / 0.3 + f["input_noise"] / 0.1 + (f["weight_decay"] >= 1e-2)) / 3.0
    best_epoch = int(np.clip(round(3 + 7 * reg + rng.normal(0, 1)), 2, 25))
    epochs = fixed_epochs or min(max_epochs, best_epoch + patience)
    layers, channels, hidden = fake_layers(n_features, f["size"], f["kernel"])
    n_params = sum(l["params"] for l in layers)
    log = journal.log if journal is not None else (lambda *a, **k: None)

    config = {"cap": f["cap"], "lam": f["lam"], "lr": f["lr"], "batch_size": f["batch_size"], "max_epochs": max_epochs, "patience": patience,
              "clip": 1.0, "seed": seed, "log_every": 50, "preset": preset, "dropout": f["dropout"], "input_noise": f["input_noise"],
              "weight_decay": f["weight_decay"], "channels": list(channels), "hidden": hidden, "kernel": f["kernel"], "log_kernels": False}
    n_train = 112000
    log("meta", model={"n_features": n_features, "channels": list(channels), "kernel": f["kernel"], "hidden": hidden, "window": 30,
                       "dropout": f["dropout"]}, layers=layers, n_params=n_params, config=config, device="simulation", threads=0,
        n_train=n_train, n_val=23000, fixed_epochs=fixed_epochs, torch="aucun (simulation)", **(meta_extra or {}))
    end_cycle = np.arange(30, 289)
    true = 289 - end_cycle
    cap = f["cap"]
    capped = np.minimum(true, cap)
    log("probe", engine="FD001_2", end_cycle=end_cycle.tolist(), true=true.astype(float).tolist(), true_capped=capped.astype(float).tolist(), n_act=256)

    def predict_curve(epoch, rmse):
        learned = 1 - np.exp(-epoch / 2.2)
        noise = rng.normal(0, 0.6 + max(0.0, rmse - 22) * 1.2, size=len(capped))
        return np.clip(capped * learned + 85 * (1 - learned) + noise, 0, 200)

    def snapshot(epoch, rmse):
        grow = 1 + 0.12 * epoch * (1 - 0.5 * reg)
        lay = {n: {"weight_norm": round(base * grow, 4), "mean_abs_weight": round(0.05 * grow, 5)}
               for n, base in (("conv1", 3.25), ("conv2", 4.63), ("gru", 11.27), ("head_rul", 0.58), ("head_regime", 0.58))}
        for n in ("conv1", "conv2", "gru"):
            lay[n].update(act_mean=round(0.2 + 0.01 * epoch, 4), act_std=0.5, **({"dead_fraction": round(0.01 * epoch * (1 - reg), 4)} if n != "gru" else {}))
        log("layers", epoch=epoch, layers=lay)
        log("prediction", epoch=epoch, pred=np.round(predict_curve(epoch, rmse), 1).tolist())

    snapshot(0, best_rmse + 8)
    history, best_seen, best_ep, bad = [], 1e9, 0, 0
    t0 = time.perf_counter()
    for e in range(1, epochs + 1):
        if sec_per_epoch:
            time.sleep(sec_per_epoch * (0.8 + 0.4 * rng.rand()))
        if e <= best_epoch:
            rmse = best_rmse + 6.0 * np.exp(-e / max(best_epoch / 2.5, 0.7)) * (1 - e / (best_epoch + 1))
        else:
            rmse = best_rmse + 0.35 * (e - best_epoch) * (1 - 0.6 * reg)
        rmse += rng.normal(0, 0.12)
        tr_mse = (0.03 * np.exp(-e / 6) + 0.004) * (1 + 0.5 * reg)
        va_mse = (rmse / 150) ** 2 * 1.5
        row = {"epoch": e, "train_loss": tr_mse + 0.02, "train_mse": tr_mse, "train_bce": 0.6, "val_loss": va_mse + 0.03,
               "val_mse": va_mse, "val_bce": 0.62, "rmse": float(rmse), "score": float(2e6 * np.exp((rmse - 24) / 4)),
               "bias": float(rng.normal(0, 1)), "late_rate": 0.48, "regime_acc": float(0.6 + 0.08 * rng.rand()), "seconds": round(sec_per_epoch, 2)}
        history.append(row)
        for b in range(50, 451, 100):
            log("batch", epoch=e, batch=b, n_batches=440, mse=round(tr_mse * (1 + 0.05 * rng.randn()), 6), bce=0.6)
        log("epoch", **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, lr=f["lr"])
        snapshot(e, rmse)
        if fixed_epochs is None:
            if rmse < best_seen:
                best_seen, best_ep, bad = rmse, e, 0
                log("best", epoch=e, rmse=round(float(rmse), 4), score=round(row["score"], 2))
            else:
                bad += 1
                if bad >= patience:
                    log("early_stop", epoch=e, best_epoch=best_ep, patience=patience)
                    break
    seconds = round(time.perf_counter() - t0, 1)
    val = {"rmse": float(best_seen if fixed_epochs is None else history[-1]["rmse"]), "score": 4e6, "bias": -1.0, "late_rate": 0.48,
           "regime_acc": 0.66, "val_loss": 0.05}
    result = {"config": config, "n_params": n_params, "epochs_run": len(history), "best_epoch": best_ep or len(history),
              "stopped_early": fixed_epochs is None and len(history) < epochs + 1 and bad >= patience, "seconds": seconds,
              "device": "simulation", "threads": 0, "history": history, "simulated": True}
    if fixed_epochs is None:
        result["val"], result["val_by_subset"] = val, {}
        log("final", **val, best_epoch=result["best_epoch"], epochs_run=len(history), seconds=seconds)
        log("check", name="simulation", max_abs_diff_rmse=0.0, ok=True)
    else:                                                        # entraînement final : un faux jeu de test
        sizes = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
        subs = np.concatenate([[k] * v for k, v in sizes.items()])
        true_t = np.random.RandomState(123).randint(6, 196, size=len(subs)).astype(float)   # même vérité pour toutes les graines
        test_rmse = best_rmse - 2.0
        pred = np.clip(true_t + rng.normal(0, test_rmse, size=len(subs)), 0, 200)
        regime = np.concatenate([[0] * 359, [1] * 348])
        result["test"] = evaluate_by_subset(pred, true_t, subs)
        result["test_pred"], result["test_true"] = np.round(pred, 2).tolist(), true_t.tolist()
        result["test_subsets"], result["test_regime"] = subs.tolist(), regime.tolist()
        result["test_logit"] = np.round(rng.normal(0, 1, size=len(subs)), 3).tolist()
        log("final", epochs_run=len(history), seconds=seconds)
    if journal is not None:
        journal.close()
    return result
