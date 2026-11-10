"""Entraînement du réseau CNN 1D + GRU : choix de λ, validation croisée, entraînement final, court essai.

Usage (CPU par défaut ; CUDA seulement avec --device cuda, à la seule décision de l'utilisateur) :
    .\\.venv\\Scripts\\python.exe scripts\\train.py --smoke [--epochs 3]     # court essai réel, pli 0, journal pour l'interface
    .\\.venv\\Scripts\\python.exe scripts\\train.py --lambda-grid             # étape A : 3 λ × 5 plis × 1 graine
    .\\.venv\\Scripts\\python.exe scripts\\train.py --cv [--lam 0.05]         # étape B : 5 plis × 5 graines
    .\\.venv\\Scripts\\python.exe scripts\\train.py --final                   # étape C : tout l'entraînement, 5 graines, test évalué une fois

Options communes : --threads N (threads de calcul par entraînement, défaut : tous les cœurs divisés par --workers),
--workers K (K entraînements en parallèle, défaut 4 : 4 × 2 threads donne environ 2,3 fois le débit d'un seul entraînement sur 8 threads, mesuré), --sensors 17|14, --journal (écrit viz/entrainement/journal.jsonl
pour le pli 0 et la graine 0, à suivre en direct dans viz/entrainement/live.html).

Les résultats vont dans data/processed/train_results.json, run par run : un run déjà fait est sauté (reprise possible).
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from cmapss.data import PROCESSED

RESULTS = PROCESSED / "train_results.json"
JOURNAL = ROOT / "viz" / "entrainement" / "journal.jsonl"
LAMBDAS = [0.0, 0.05, 0.2]       # grille de départ ; à ajuster d'après les ordres de grandeur du court essai (docs/reseau.md)
SEEDS = [0, 1, 2, 3, 4]
SENSOR_SETS = ("17", "14")
_CACHE = {}


def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"runs": {}}


def save_results(res):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    tmp.replace(RESULTS)


def baseline_reference(sensors, fold):
    """Scores de la baseline à comparer (mêmes plis, plafond 150) : lus dans data/processed/baseline_results.json, jamais saisis à la main."""
    f = PROCESSED / "baseline_results.json"
    if not f.exists():
        return None
    b = json.loads(f.read_text(encoding="utf-8"))
    ref = {}
    cv = b.get("cv", {})
    s = next((x for x in cv.get("summary", []) if x["sensors"] == sensors and x["cap"] == 150), None)
    if s:
        ref["baseline_val_rmse_mean"] = s["rmse_mean"]
    r = next((x for x in cv.get("rows", []) if x["sensors"] == sensors and x["cap"] == 150 and x["fold"] == fold), None)
    if r and fold is not None:
        ref["baseline_val_rmse_fold"] = r["metrics"]["rmse"]
    if "final" in b:
        ref["baseline_test_rmse"] = b["final"]["test"]["total"]["rmse"]
    return {"reference": ref} if ref else None


def key_of(stage, lam, fold, seed, sensors):
    return f"{stage}|s{sensors}|lam{lam}|fold{'all' if fold is None else fold}|seed{seed}"


# ---------------------------------------------------------------------------------------------------------------------
# Travail d'un entraînement (exécuté dans un processus à part quand --workers > 1)
def run_one(spec):
    import torch
    from cmapss.data import load_all
    from cmapss.folds import make_folds
    from cmapss.journal import Journal
    from cmapss.loader import features_without, fold_data, load_test_data, SENSORS_UNSTABLE, N_FOLDS
    from cmapss.metrics import evaluate_by_subset
    from cmapss.normalisation import ConditionScaler
    from cmapss.train import RunData, TrainConfig, predict, run_data_from_fold, train_run
    from cmapss.windows import FEATURES, make_windows

    torch.set_num_threads(spec["threads"])
    feats = FEATURES if spec["sensors"] == "17" else features_without(SENSORS_UNSTABLE)
    train = _CACHE.get("train")
    if train is None:
        train = _CACHE["train"] = load_all("train")
    cfg = TrainConfig(lam=spec["lam"], seed=spec["seed"], max_epochs=spec["max_epochs"], patience=spec["patience"])
    journal = Journal(spec["journal"]) if spec.get("journal") else None

    t0 = time.perf_counter()
    if spec["fold"] is not None:
        if _CACHE.get("fold") != (spec["fold"], spec["sensors"]):
            _CACHE.pop("data", None)
            fd = fold_data(spec["fold"], features=feats, train=train, folds=make_folds(train, n_splits=N_FOLDS))
            _CACHE["data"], _CACHE["fold"] = run_data_from_fold(fd), (spec["fold"], spec["sensors"])
            del fd
        data = _CACHE["data"]
        prep = time.perf_counter() - t0
        res, _ = train_run(data, cfg, journal=journal, device=spec["device"], meta_extra=baseline_reference(spec["sensors"], spec["fold"]))
    else:                                                    # entraînement final : tout l'entraînement, test évalué une fois
        scaler = ConditionScaler().fit(train)
        w = make_windows(scaler.transform(train), "train", features=feats)
        data = RunData(X_tr=w.X, rul_tr=w.rul, reg_tr=w.regime)
        prep = time.perf_counter() - t0
        res, model = train_run(data, cfg, journal=journal, device=spec["device"], fixed_epochs=spec["fixed_epochs"],
                               meta_extra=baseline_reference(spec["sensors"], None))
        td = load_test_data(features=feats, train=train)
        pred, logit = predict(model, td.windows.X, spec["device"], cap=cfg.cap)
        res["test"] = evaluate_by_subset(pred, td.windows.rul, td.subsets)
        res["test_pred"] = np.round(pred, 2).tolist()
        res["test_true"] = td.windows.rul.tolist()
        res["test_subsets"] = td.subsets.tolist()
        res["test_regime"] = td.windows.regime.tolist()
        res["test_logit"] = np.round(logit, 3).tolist()
    res["prep_seconds"] = round(prep, 1)
    return spec["key"], res


def run_specs(specs, workers, results):
    """Lance les entraînements (en parallèle si workers > 1), enregistre chaque résultat dès qu'il est fini."""
    todo = [s for s in specs if s["key"] not in results["runs"]]
    print(f"{len(specs)} entraînements demandés, {len(specs) - len(todo)} déjà faits, {len(todo)} à faire "
          f"({workers} en parallèle, {todo[0]['threads'] if todo else 0} threads chacun)", flush=True)
    if not todo:
        return
    t0, done = time.time(), 0

    def finish(key, res):
        nonlocal done
        done += 1
        results["runs"][key] = res
        save_results(results)
        v = res.get("val") or {}
        extra = f"RMSE val {v['rmse']:.2f} (meilleure époque {res['best_epoch']}/{res['epochs_run']})" if v else \
            f"test RMSE {res['test']['total']['rmse']:.2f}"
        el = time.time() - t0
        print(f"  [{done}/{len(todo)}] {key} : {extra}, {res['seconds']:.0f} s ; écoulé {el / 60:.1f} min, "
              f"reste environ {(el / done) * (len(todo) - done) / 60:.1f} min", flush=True)

    if workers == 1:
        for s in todo:
            finish(*run_one(s))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_one, s) for s in todo]
            for f in cf.as_completed(futs):
                finish(*f.result())


# ---------------------------------------------------------------------------------------------------------------------
def make_spec(stage, lam, fold, seed, args, journal=False, fixed_epochs=None):
    return {"key": key_of(stage, lam, fold, seed, args.sensors), "stage": stage, "lam": lam, "fold": fold, "seed": seed,
            "sensors": args.sensors, "device": args.device, "threads": args.threads, "max_epochs": args.epochs,
            "patience": args.patience, "fixed_epochs": fixed_epochs, "journal": str(JOURNAL) if journal else None}


def rows(results, stage, lam, sensors):
    return [r for k, r in results["runs"].items() if k.startswith(f"{stage}|s{sensors}|lam{lam}|")]


def by_fold(results, stage, lam, sensors, seeds=None):
    out = {}
    for k, r in results["runs"].items():
        p = k.split("|")
        if p[0] == stage and p[1] == f"s{sensors}" and p[2] == f"lam{lam}" and p[3].startswith("fold"):
            if seeds is None or int(p[4][4:]) in seeds:
                out.setdefault(int(p[3][4:]), []).append(r)
    return out


def stage_smoke(args, results):
    seeds = list(range(args.workers))
    specs = [make_spec("smoke", args.lam, 0, s, args, journal=(s == 0)) for s in seeds]
    for s in specs:
        results["runs"].pop(s["key"], None)                  # un essai est toujours refait
    t0 = time.time()
    run_specs(specs, args.workers, results)
    wall = time.time() - t0
    for s in specs:
        r = results["runs"][s["key"]]
        h = r["history"]
        print(f"  graine {s['seed']} : {len(h)} époques, {np.mean([x['seconds'] for x in h]):.1f} s par époque "
              f"(préparation des données {r['prep_seconds']} s)")
    print(f"Temps total {wall:.0f} s pour {len(specs)} entraînement(s) en parallèle ; journal : {JOURNAL}")


def stage_lambda_grid(args, results):
    specs = [make_spec("lam", lam, k, 0, args) for lam in LAMBDAS for k in range(5)]
    run_specs(specs, args.workers, results)
    print("\n=== Choix de λ (5 plis, graine 0) ===")
    base = by_fold(results, "lam", LAMBDAS[0], args.sensors)
    table = {}
    for lam in LAMBDAS:
        f = by_fold(results, "lam", lam, args.sensors)
        if len(f) < 5:
            continue
        rm = np.array([f[k][0]["val"]["rmse"] for k in range(5)])
        sc = np.array([f[k][0]["val"]["score"] for k in range(5)])
        ac = np.array([f[k][0]["val"]["regime_acc"] for k in range(5)])
        ep = np.array([f[k][0]["best_epoch"] for k in range(5)])
        table[lam] = rm
        line = f"λ = {lam:<5} RMSE {rm.mean():6.2f} ± {rm.std():.2f}   score {sc.mean():>10.0f}   régime {ac.mean():.1%}   meilleure époque {ep.mean():.0f}"
        if lam != LAMBDAS[0] and len(base) == 5:
            d = rm - np.array([base[k][0]["val"]["rmse"] for k in range(5)])
            line += f"   écart au λ = 0 pli par pli : {d.mean():+.2f} ± {d.std():.2f}"
        print(line)
    if table:
        best = min(table, key=lambda l: table[l].mean())
        print(f"Meilleur RMSE moyen : λ = {best}. Une seule graine : un écart inférieur à l'écart pli par pli n'est pas significatif.")
        results["lambda_grid"] = {"best": best, "rmse_mean": {str(l): float(v.mean()) for l, v in table.items()}}
        save_results(results)


def chosen_lambda(args, results):
    if args.lam is not None:
        return args.lam
    if "lambda_grid" in results:
        return results["lambda_grid"]["best"]
    sys.exit("Lancer d'abord --lambda-grid, ou donner --lam.")


def stage_cv(args, results):
    lam = chosen_lambda(args, results)
    specs = [make_spec("cv", lam, k, s, args, journal=(args.journal and k == 0 and s == 0)) for s in SEEDS for k in range(5)]
    # les 5 entraînements de la graine 0 existent déjà dans l'étape A : on les réutilise tels quels
    for k in range(5):
        a = results["runs"].get(key_of("lam", lam, k, 0, args.sensors))
        if a is not None and not (args.journal and k == 0):
            results["runs"].setdefault(key_of("cv", lam, k, 0, args.sensors), a)
    run_specs(specs, args.workers, results)
    f = by_fold(results, "cv", lam, args.sensors)
    if len(f) < 5 or any(len(v) < len(SEEDS) for v in f.values()):
        print("Validation croisée incomplète.")
        return
    rm = np.array([[r["val"]["rmse"] for r in f[k]] for k in range(5)])        # plis × graines
    sc = np.array([[r["val"]["score"] for r in f[k]] for k in range(5)])
    ac = np.array([[r["val"]["regime_acc"] for r in f[k]] for k in range(5)])
    ep = np.array([[r["best_epoch"] for r in f[k]] for k in range(5)])
    print(f"\n=== Validation croisée du réseau (λ = {lam}, 5 plis × {len(SEEDS)} graines) ===")
    print(f"RMSE   {rm.mean():.2f}   écart entre plis {rm.mean(1).std():.2f}   écart entre graines {rm.std(1).mean():.2f}")
    print(f"score  {sc.mean():.0f} ± {sc.mean(1).std():.0f}   exactitude du régime {ac.mean():.1%}   meilleure époque (médiane) {np.median(ep):.0f}")
    base = json.loads((PROCESSED / "baseline_results.json").read_text(encoding="utf-8"))["cv"]["summary"] if (PROCESSED / "baseline_results.json").exists() else []
    b = next((s for s in base if s["sensors"] == args.sensors and s["cap"] == 150), None)
    if b:
        print(f"baseline (mêmes plis, 17 capteurs, plafond 150) : RMSE {b['rmse_mean']:.2f} ± {b['rmse_std']:.2f}")
    results["cv_summary"] = {"lam": lam, "sensors": args.sensors, "rmse_mean": float(rm.mean()), "rmse_fold_std": float(rm.mean(1).std()),
                             "rmse_seed_std": float(rm.std(1).mean()), "median_best_epoch": float(np.median(ep))}
    save_results(results)


def stage_final(args, results):
    if "cv_summary" not in results:
        sys.exit("Lancer d'abord --cv.")
    lam, n_ep = results["cv_summary"]["lam"], int(round(results["cv_summary"]["median_best_epoch"]))
    print(f"Entraînement final : λ = {lam}, {n_ep} époques (médiane des meilleures époques de la validation croisée)")
    specs = [make_spec("final", lam, None, s, args, journal=(args.journal and s == 0), fixed_epochs=n_ep) for s in SEEDS]
    run_specs(specs, args.workers, results)
    rs = [results["runs"][s["key"]] for s in specs if s["key"] in results["runs"]]
    if len(rs) < len(SEEDS):
        return
    print("\n=== Test (une fenêtre par moteur, évalué une seule fois par graine) ===")
    from cmapss.metrics import evaluate_by_subset
    true, subs = np.array(rs[0]["test_true"]), np.array(rs[0]["test_subsets"])
    ens = evaluate_by_subset(np.mean([r["test_pred"] for r in rs], axis=0), true, subs)
    base = json.loads((PROCESSED / "baseline_results.json").read_text(encoding="utf-8"))["final"]["test"] if (PROCESSED / "baseline_results.json").exists() else {}
    print(f"{'jeu':<7}{'RMSE (5 graines)':>22}{'RMSE moyenne des 5':>20}{'score':>10}{'baseline':>10}{'régime exact':>14}")
    reg = np.array(rs[0]["test_regime"])
    for fd in ["FD001", "FD002", "FD003", "FD004", "total"]:
        v = np.array([r["test"][fd]["rmse"] for r in rs])
        acc = np.mean([((np.array(r["test_logit"]) > 0).astype(int) == reg)[(subs == fd) if fd != "total" else slice(None)].mean() for r in rs])
        print(f"{fd:<7}{v.mean():>14.2f} ± {v.std():<5.2f}{ens[fd]['rmse']:>20.2f}{np.mean([r['test'][fd]['score'] for r in rs]):>10.0f}"
              f"{base.get(fd, {}).get('rmse', float('nan')):>10.2f}{acc:>14.1%}")
    results["final_summary"] = {"lam": lam, "epochs": n_ep, "ensemble": ens}
    save_results(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("smoke", "lambda-grid", "cv", "final", "journal"):
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--lam", type=float, default=None, help="poids de la tête régime (défaut : celui retenu par --lambda-grid ; 0,05 pour --smoke)")
    ap.add_argument("--epochs", type=int, default=None, help="nombre maximal d'époques (défaut : 50 ; 3 pour --smoke)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sensors", choices=SENSOR_SETS, default="17")
    ap.add_argument("--device", default="cpu", help="cpu (défaut) ou cuda : à ne choisir que sur décision de l'utilisateur")
    ap.add_argument("--workers", type=int, default=None, help="entraînements en parallèle (défaut : 4, mesuré le plus efficace avec 8 cœurs ; 1 pour --smoke)")
    ap.add_argument("--threads", type=int, default=None, help="threads de calcul par entraînement (défaut : cœurs / workers)")
    args = ap.parse_args()
    if not (args.smoke or args.lambda_grid or args.cv or args.final):
        ap.print_help()
        return
    args.workers = args.workers or (1 if args.smoke else 4)
    args.threads = args.threads or max(1, (os.cpu_count() or 4) // args.workers)
    if args.smoke and args.lam is None:
        args.lam = 0.05
    args.epochs = args.epochs or (3 if args.smoke else 50)
    if args.smoke:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    results = load_results()
    if args.smoke:
        stage_smoke(args, results)
    if args.lambda_grid:
        stage_lambda_grid(args, results)
    if args.cv:
        stage_cv(args, results)
    if args.final:
        stage_final(args, results)


if __name__ == "__main__":
    main()
