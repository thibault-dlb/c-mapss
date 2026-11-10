"""Entraînement du réseau : perte multitâche, arrêt précoce, évaluation, journal d'événements.

Perte : MSE(RUL / plafond) + λ · BCE(régime). La RUL est divisée par le plafond : la MSE vaut alors environ 0,03 pour un
RMSE de 24 cycles, donc λ doit rester petit pour que la tête régime ne domine pas.

Validation : fenêtres des moteurs de validation dont la vraie RUL est dans la plage du test, mesurées contre la vraie RUL
**non plafonnée** (comme la baseline, pour pouvoir comparer). L'arrêt précoce se fait sur ce RMSE.

Journal (voir docs/reseau.md) : meta, probe, layers, prediction, batch, epoch, best, early_stop, final, check.
"""
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
from torch import nn

from .metrics import evaluate, evaluate_by_subset
from .model import CnnGru, ModelConfig, describe, n_params, weight_layers
from . import search
from .windows import cap_rul


@dataclass
class TrainConfig:
    cap: int = 150               # plafond de RUL (choisi par la baseline)
    lam: float = 0.05            # poids de la tête régime
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    clip: float = 1.0
    seed: int = 0
    log_every: int = 50          # un événement `batch` tous les 50 lots
    # Réglages recherchés (voir cmapss/search.py) ; `preset` est le nom du réglage, enregistré dans le journal
    preset: str = "aucune"
    dropout: float = 0.0
    input_noise: float = 0.0     # écart-type du bruit gaussien ajouté aux entrées (entraînement seulement), en unités normalisées
    weight_decay: float = 0.0    # AdamW si > 0, Adam sinon
    channels: tuple = (32, 64)
    hidden: int = 64
    kernel: int = 5
    log_kernels: bool = True     # écrire les noyaux de la convolution 1 à chaque époque (lourd : désactivé pour les journaux en direct)

    def to_dict(self):
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


def train_config(overrides=None, preset="", **kw):
    """TrainConfig d'un réglage de `cmapss.search` (dictionnaire d'écarts au défaut) ; `kw` : graine, époques, patience..."""
    f = search.full(overrides)
    channels, hidden = search.SIZES[f["size"]]
    fields = dict(preset=preset, cap=f["cap"], lam=f["lam"], lr=f["lr"], batch_size=f["batch_size"], dropout=f["dropout"],
                  input_noise=f["input_noise"], weight_decay=f["weight_decay"], channels=channels, hidden=hidden,
                  kernel=f["kernel"])
    fields.update(kw)                                        # les arguments explicites l'emportent sur le réglage
    return TrainConfig(**fields)


def preset_config(name, **kw):
    """TrainConfig d'un réglage nommé (`search.NAMED`)."""
    if name not in search.NAMED:
        raise ValueError(f"réglage inconnu : {name} (choix : {', '.join(search.NAMED)})")
    return train_config(search.NAMED[name], preset=name, **kw)


@dataclass
class RunData:
    """Tableaux d'un entraînement. `*_va` peut être None (entraînement final, sans validation)."""
    X_tr: np.ndarray
    rul_tr: np.ndarray           # RUL non plafonnée
    reg_tr: np.ndarray
    X_va: np.ndarray = None      # fenêtres de validation DANS la plage du test
    rul_va: np.ndarray = None
    reg_va: np.ndarray = None
    sub_va: np.ndarray = None    # jeu (FD001...) de chaque fenêtre de validation
    probe: dict = None           # moteur de démonstration : X, rul, end_cycle, engine, et X_act (lot pour les activations)


def set_seed(seed, device="cpu"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def multitask_loss(rul_pred, rul_true, logit, regime, lam):
    """(perte totale, MSE, BCE). rul_pred et rul_true sont déjà divisés par le plafond ; regime vaut 0 ou 1."""
    mse = torch.mean((rul_pred - rul_true) ** 2)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, regime.float())
    return mse + lam * bce, mse, bce


class EarlyStopping:
    """Arrêt sur une valeur à minimiser. `step(valeur, époque, modèle)` renvoie True quand il faut s'arrêter."""

    def __init__(self, patience):
        self.patience, self.best, self.best_epoch, self.state, self.bad = patience, float("inf"), 0, None, 0

    def step(self, value, epoch, model):
        if value < self.best:
            self.best, self.best_epoch, self.bad = value, epoch, 0
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)


def predict(model, X, device="cpu", cap=150, batch=2048):
    """(RUL en cycles, logits du régime) pour un tableau de fenêtres."""
    model.eval()
    rs, ls = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device)
            r, l = model(xb)
            rs.append(r.cpu().numpy()); ls.append(l.cpu().numpy())
    return np.concatenate(rs) * cap, np.concatenate(ls)


def layer_stats(model, x_act, device="cpu"):
    """Mesures réelles par couche sur un lot fixe : normes de poids, activations, noyaux de la 1ʳᵉ convolution."""
    model.eval()
    with torch.no_grad():
        _, _, acts = model(torch.from_numpy(np.ascontiguousarray(x_act)).to(device), return_acts=True)
        layers = {}
        for name, mod in weight_layers(model):
            w = [p for p in mod.parameters() if p.dim() > 1]
            sq = float(sum((p ** 2).sum() for p in w))
            n = int(sum(p.numel() for p in w))
            layers[name] = {"weight_norm": round(sq ** 0.5, 5), "mean_abs_weight": round(float(sum(p.abs().sum() for p in w)) / n, 6)}
        for name, a in acts.items():
            layers[name]["act_mean"] = round(float(a.mean()), 5)
            layers[name]["act_std"] = round(float(a.std()), 5)
            if a.dim() == 3:      # canaux ReLU : « éteint » = nul sur tout le lot
                layers[name]["dead_fraction"] = round(float((a.amax(dim=(0, 2)) == 0).float().mean()), 4)
        kernels = model.conv1[0].weight.detach().cpu().numpy()
    return layers, np.round(kernels, 3).tolist()


def val_metrics(model, data, cfg, device):
    pred, logit = predict(model, data.X_va, device, cap=cfg.cap)
    m = evaluate(pred, data.rul_va)
    y = torch.from_numpy(cap_rul(data.rul_va, cfg.cap) / cfg.cap).float()
    mse = float(torch.mean((torch.from_numpy(pred / cfg.cap).float() - y) ** 2))
    reg = torch.from_numpy(data.reg_va.astype(np.float32))
    bce = float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logit).float(), reg))
    m["val_loss"] = mse + cfg.lam * bce
    m["val_mse"], m["val_bce"] = mse, bce
    m["regime_acc"] = float(((logit > 0).astype(int) == data.reg_va).mean())
    return m, pred


def train_run(data, cfg, journal=None, device="cpu", fixed_epochs=None, model_cfg=None, meta_extra=None):
    """Entraîne un réseau. Retourne (résultat, modèle avec les meilleurs poids).

    - Avec validation : arrêt précoce sur le RMSE de validation, meilleurs poids restaurés.
    - `fixed_epochs` : nombre d'époques imposé, sans arrêt précoce (entraînement final).
    """
    set_seed(cfg.seed, device)
    n_feat = data.X_tr.shape[2]
    model = CnnGru(model_cfg or ModelConfig(n_features=n_feat, channels=tuple(cfg.channels), hidden=cfg.hidden,
                                            kernel=cfg.kernel, dropout=cfg.dropout)).to(device)
    if cfg.weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    X = torch.from_numpy(np.ascontiguousarray(data.X_tr)).to(device)
    y = torch.from_numpy((cap_rul(data.rul_tr, cfg.cap) / cfg.cap).astype(np.float32)).to(device)
    reg = torch.from_numpy(data.reg_tr.astype(np.int64)).to(device)
    n = len(X)
    gen = torch.Generator().manual_seed(cfg.seed)            # l'ordre des lots ne dépend que de la graine
    has_val = data.X_va is not None and fixed_epochs is None
    stopper = EarlyStopping(cfg.patience)
    log = journal.log if journal is not None else (lambda *a, **k: None)
    probe = data.probe

    log("meta", model=model.cfg.to_dict(), layers=describe(model), n_params=n_params(model), config=cfg.to_dict(),
        device=str(device), threads=torch.get_num_threads(), n_train=int(n),
        n_val=int(len(data.X_va)) if data.X_va is not None else 0, fixed_epochs=fixed_epochs,
        torch=torch.__version__, **(meta_extra or {}))
    if probe is not None:
        log("probe", engine=str(probe["engine"]), end_cycle=np.asarray(probe["end_cycle"]).tolist(),
            true=np.round(probe["rul"], 1).tolist(), true_capped=np.round(cap_rul(probe["rul"], cfg.cap), 1).tolist(),
            n_act=int(len(probe["X_act"])))

    def snapshot(epoch):
        if probe is None:
            return
        layers, kernels = layer_stats(model, probe["X_act"], device)
        log("layers", epoch=epoch, layers=layers, **({"kernels": kernels} if cfg.log_kernels else {}))
        p, _ = predict(model, probe["X"], device, cap=cfg.cap)
        log("prediction", epoch=epoch, pred=np.round(p, 1).tolist())

    snapshot(0)                                              # état initial, avant tout apprentissage
    history, t_start, epoch = [], time.perf_counter(), 0
    total_epochs = fixed_epochs or cfg.max_epochs
    stopped_early = False
    for epoch in range(1, total_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=gen)
        sums = np.zeros(3); count = 0
        run_s = np.zeros(2); run_n = 0
        for b, i in enumerate(range(0, n, cfg.batch_size)):
            idx = perm[i:i + cfg.batch_size].to(device)
            xb = X[idx]
            if cfg.input_noise > 0:                              # bruit sur les entrées : à l'entraînement seulement
                xb = xb + cfg.input_noise * torch.randn_like(xb)
            r, l = model(xb)
            loss, mse, bce = multitask_loss(r, y[idx], l, reg[idx], cfg.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            k = len(idx)
            lv, mv, bv = loss.item(), mse.item(), bce.item()
            sums += np.array([lv, mv, bv]) * k; count += k
            run_s += np.array([mv, bv]); run_n += 1
            if (b + 1) % cfg.log_every == 0:
                log("batch", epoch=epoch, batch=b + 1, n_batches=(n + cfg.batch_size - 1) // cfg.batch_size,
                    mse=round(run_s[0] / run_n, 6), bce=round(run_s[1] / run_n, 5))
                run_s[:] = 0; run_n = 0
        tr_loss, tr_mse, tr_bce = (sums / count).tolist()
        row = {"epoch": epoch, "train_loss": tr_loss, "train_mse": tr_mse, "train_bce": tr_bce}
        if has_val:
            m, _ = val_metrics(model, data, cfg, device)
            row.update({k: m[k] for k in ("val_loss", "val_mse", "val_bce", "rmse", "score", "bias", "late_rate", "regime_acc")})
        row["seconds"] = round(time.perf_counter() - t0, 2)
        history.append(row)
        log("epoch", **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, lr=cfg.lr)
        snapshot(epoch)
        if has_val:
            was_best = row["rmse"] < stopper.best
            stop = stopper.step(row["rmse"], epoch, model)
            if was_best:
                log("best", epoch=epoch, rmse=round(row["rmse"], 4), score=round(row["score"], 2))
            if stop:
                stopped_early = True
                log("early_stop", epoch=epoch, best_epoch=stopper.best_epoch, patience=cfg.patience)
                break
    if has_val:
        stopper.restore(model)
    result = {"config": cfg.to_dict(), "n_params": n_params(model), "epochs_run": epoch,
              "best_epoch": stopper.best_epoch if has_val else epoch, "stopped_early": stopped_early,
              "seconds": round(time.perf_counter() - t_start, 1), "device": str(device),
              "threads": torch.get_num_threads(), "history": history}
    if has_val:
        m, pred = val_metrics(model, data, cfg, device)
        result["val"] = {k: float(m[k]) for k in ("rmse", "score", "bias", "late_rate", "regime_acc", "val_loss")}
        result["val_by_subset"] = evaluate_by_subset(pred, data.rul_va, data.sub_va)
        diff = abs(m["rmse"] - stopper.best)
        log("final", **result["val"], best_epoch=stopper.best_epoch, epochs_run=epoch, seconds=result["seconds"])
        log("check", name="restored_model_equals_best_epoch", max_abs_diff_rmse=float(diff), ok=bool(diff < 1e-4))
    else:
        log("final", epochs_run=epoch, seconds=result["seconds"])
    if journal is not None:
        journal.close()
    return result, model


# ---------------------------------------------------------------------------------------------------------------------
def make_probe(fd, seed=0, n_act=256):
    """Moteur de démonstration (premier moteur FD001 de la validation, toutes ses fenêtres) et lot fixe pour les activations."""
    ids = np.asarray(fd.val.engine_id)
    demo = next(x for x in dict.fromkeys(ids) if str(x).startswith("FD001_"))
    idx = np.where(ids == demo)[0]
    idx = idx[np.argsort(fd.val.end_cycle[idx])]
    rng = np.random.default_rng(seed)
    act = rng.choice(len(fd.val.X), size=min(n_act, len(fd.val.X)), replace=False)
    return {"engine": demo, "X": fd.val.X[idx], "rul": fd.val.rul[idx], "end_cycle": fd.val.end_cycle[idx], "X_act": fd.val.X[act]}


def run_data_from_fold(fd, with_probe=True):
    from .loader import subset_of
    m = fd.val_in_range
    return RunData(X_tr=fd.train.X, rul_tr=fd.train.rul, reg_tr=fd.train.regime,
                   X_va=fd.val.X[m], rul_va=fd.val.rul[m], reg_va=fd.val.regime[m],
                   sub_va=subset_of(fd.val.engine_id[m]), probe=make_probe(fd) if with_probe else None)
