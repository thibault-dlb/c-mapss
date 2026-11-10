"""Entraînement du réseau CNN 1D + GRU : recherche d'hyperparamètres, validation croisée, entraînement final, court essai.

Usage (CPU par défaut ; CUDA seulement avec --device cuda, à la seule décision de l'utilisateur ; rien ne se lance seul) :
    .\\.venv\\Scripts\\python.exe scripts\\train.py --smoke [--epochs 3] [--preset dropout] [--set dropout=0.3,lr=0.003]
                                                                    # court essai réel, pli 0, journal pour l'interface
    .\\.venv\\Scripts\\python.exe scripts\\train.py --search [--search-n 60] [--top 8]
                                                                    # étape R : recherche d'hyperparamètres (voir plus bas)
    .\\.venv\\Scripts\\python.exe scripts\\train.py --cv [--preset r012]   # étape B : 5 plis × 5 graines avec le meilleur réglage
    .\\.venv\\Scripts\\python.exe scripts\\train.py --final                # étape C : tout l'entraînement, 5 graines, test évalué une fois

Recherche (--search) en deux phases, pour que le coût reste raisonnable :
  1. tri : les 6 réglages nommés + N réglages tirés au hasard (graine fixe) sont entraînés sur les plis de tri (--screen-folds,
     défaut 0 et 1) ;
  2. confirmation : les --top meilleurs (et toujours « aucune », la référence) sont entraînés sur les plis restants ; le
     classement final se fait sur les 5 plis, comparé pli par pli à « aucune ».
Les paramètres cherchés (cmapss/search.py) : dropout, bruit sur les entrées, décroissance des poids, taille du réseau, taille du
noyau, taux d'apprentissage, taille des lots, poids λ de la tête régime, plafond de RUL.

Options communes : --threads N (threads de calcul par entraînement, défaut : tous les cœurs divisés par --workers),
--workers K (K entraînements en parallèle, défaut 4 : 4 × 2 threads donne environ 2,3 fois le débit d'un seul entraînement sur
8 threads, mesuré), --sensors 17|14, --journal (écrit viz/entrainement/journal.jsonl pour le pli 0 et la graine 0, à suivre en
direct dans viz/entrainement/live.html).

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

from cmapss import search
from cmapss.data import PROCESSED

RESULTS = PROCESSED / "train_results.json"
JOURNAL = ROOT / "viz" / "entrainement" / "journal.jsonl"
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
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
    from cmapss.train import RunData, predict, run_data_from_fold, train_config, train_run
    from cmapss.windows import FEATURES, make_windows

    torch.set_num_threads(spec["threads"])
    feats = FEATURES if spec["sensors"] == "17" else features_without(SENSORS_UNSTABLE)
    train = _CACHE.get("train")
    if train is None:
        train = _CACHE["train"] = load_all("train")
    cfg = train_config(spec["overrides"], preset=spec["cid"], seed=spec["seed"], max_epochs=spec["max_epochs"], patience=spec["patience"])
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


SEC_PER_EPOCH_4X2 = 6.9      # débit mesuré avec 4 entraînements × 2 threads (docs/reseau.md) : secondes par époque, en équivalent


def run_specs(specs, workers, results, dry_run=False, est_epochs=20):
    """Lance les entraînements (en parallèle si workers > 1), enregistre chaque résultat dès qu'il est fini.

    Avec dry_run, n'entraîne rien : affiche le nombre d'entraînements et une durée ESTIMÉE (débit mesuré × époques supposées)."""
    todo = [s for s in specs if s["key"] not in results["runs"]]
    print(f"{len(specs)} entraînements demandés, {len(specs) - len(todo)} déjà faits, {len(todo)} à faire "
          f"({workers} en parallèle, {todo[0]['threads'] if todo else 0} threads chacun)", flush=True)
    if dry_run:
        minutes = len(todo) * est_epochs * SEC_PER_EPOCH_4X2 / 60
        print(f"  [simulation] rien n'est lancé. Durée estimée si chaque entraînement dure {est_epochs} époques : "
              f"{minutes:.0f} min ({minutes / 60:.1f} h), d'après {SEC_PER_EPOCH_4X2} s par époque mesurées avec 4 × 2 threads "
              f"(estimation : le vrai nombre d'époques n'est connu qu'après les premiers entraînements).")
        return
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
def make_spec(stage, cid, overrides, fold, seed, args, journal=False, fixed_epochs=None):
    return {"key": search.run_key(stage, args.sensors, cid, fold, seed), "stage": stage, "cid": cid, "overrides": overrides,
            "fold": fold, "seed": seed, "sensors": args.sensors, "device": args.device, "threads": args.threads,
            "max_epochs": args.epochs, "patience": args.patience, "fixed_epochs": fixed_epochs,
            "journal": str(JOURNAL) if journal else None}


def reuse(results, from_stage, to_stage, cid, sensors, folds=range(N_FOLDS), seed=0, skip_fold0=False):
    """Reprend les entraînements déjà faits d'une étape précédente (même réglage, même graine) : pas de calcul en double."""
    n = 0
    for k in folds:
        if skip_fold0 and k == 0:
            continue
        a = results["runs"].get(search.run_key(from_stage, sensors, cid, k, seed))
        dst = search.run_key(to_stage, sensors, cid, k, seed)
        if a is not None and dst not in results["runs"]:
            results["runs"][dst] = a
            n += 1
    return n


def overrides_of(cid, results):
    """Réglage (écarts au défaut) d'un identifiant : réglage nommé, ou tiré par --search (enregistré dans les résultats)."""
    if cid in search.NAMED:
        return dict(search.NAMED[cid])
    cfgs = results.get("search_configs", {})
    if cid in cfgs:
        return dict(cfgs[cid])
    sys.exit(f"Réglage inconnu : {cid}. Noms : {', '.join(search.NAMED)} ; identifiants tirés : voir --search.")


def parse_set(text):
    out = {}
    for item in filter(None, (text or "").split(",")):
        k, v = item.split("=")
        if k not in search.DEFAULT:
            sys.exit(f"paramètre inconnu : {k} (connus : {', '.join(search.DEFAULT)})")
        t = type(search.DEFAULT[k])
        out[k] = v if t is str else t(float(v) if t is float else int(float(v)))
    return out


def stage_smoke(args, results):
    cid = args.preset or "aucune"
    overrides = {**overrides_of(cid, results), **parse_set(args.set)}
    seeds = list(range(args.workers))
    specs = [make_spec("smoke", cid, overrides, 0, s, args, journal=(s == 0)) for s in seeds]
    for s in specs:
        results["runs"].pop(s["key"], None)                  # un essai est toujours refait
    print(f"Essai : réglage {cid} ({search.describe(overrides)})")
    t0 = time.time()
    run_specs(specs, args.workers, results, args.dry_run, args.est_epochs)
    if args.dry_run:
        return
    wall = time.time() - t0
    for s in specs:
        r = results["runs"][s["key"]]
        h = r["history"]
        print(f"  graine {s['seed']} : {len(h)} époques, {np.mean([x['seconds'] for x in h]):.1f} s par époque "
              f"(préparation des données {r['prep_seconds']} s)")
    print(f"Temps total {wall:.0f} s pour {len(specs)} entraînement(s) en parallèle ; journal : {JOURNAL}")


def print_ranking(rows, configs, title, baseline=None, limit=15):
    print(f"\n=== {title} ===")
    if baseline:
        print(f"(baseline, mêmes plis : RMSE {baseline:.2f})")
    print(f"{'rang':>4} {'réglage':<9}{'RMSE':>14}{'écart à « aucune » (pli par pli)':>36}{'régime':>8}{'époque':>8}  paramètres")
    for i, r in enumerate(rows[:limit], 1):
        d = f"{r['diff_mean']:+.2f} ± {r['diff_std']:.2f}" if "diff_mean" in r else "(référence)"
        print(f"{i:>4} {r['cid']:<9}{r['rmse_mean']:>8.2f} ± {r['rmse_std']:<4.2f}{d:>36}{r['regime_acc']:>8.1%}{r['best_epoch']:>8.0f}  "
              f"{search.describe(configs[r['cid']])}")


def stage_search(args, results):
    screen = [int(x) for x in args.screen_folds.split(",")]
    rest = [k for k in range(N_FOLDS) if k not in screen]
    configs = search.sample_configs(args.search_n, args.search_seed)
    cfg_map = {cid: o for cid, o in configs}
    results.setdefault("search_configs", {}).update({c: o for c, o in cfg_map.items() if c not in search.NAMED})
    if not args.dry_run:
        save_results(results)
    cids = list(cfg_map)
    base = None
    b = PROCESSED / "baseline_results.json"
    if b.exists():
        s_ = next((x for x in json.loads(b.read_text(encoding="utf-8"))["cv"]["summary"] if x["sensors"] == args.sensors and x["cap"] == 150), None)
        base = s_["rmse_mean"] if s_ else None

    print(f"Recherche : {len(cids)} réglages ({len(search.NAMED)} nommés + {args.search_n} tirés, graine {args.search_seed}), "
          f"tri sur les plis {screen}, puis les {args.top} meilleurs sur les plis {rest}.")
    # phase 1 : tri (les runs d'un même pli se suivent : les données du pli restent en mémoire dans chaque processus)
    specs1 = [make_spec("search", c, cfg_map[c], k, 0, args) for k in screen for c in cids]
    run_specs(specs1, args.workers, results, args.dry_run, args.est_epochs)
    if args.dry_run:
        n2 = (args.top + 1) * len(rest)
        print(f"Phase 2 (après le tri) : jusqu'à {n2} entraînements ({args.top} meilleurs + « aucune » × plis {rest}).")
        print("Réglages tirés :")
        for cid in cids:
            print(f"  {cid:<9} {search.describe(cfg_map[cid])}")
        return
    rows1 = search.ranking(results["runs"], "search", args.sensors, cids, screen)
    print_ranking(rows1, cfg_map, f"Phase 1 : tri sur les plis {screen}", limit=args.top + 5)
    top = [r["cid"] for r in rows1[:args.top]]
    if "aucune" not in top:
        top.append("aucune")
    # phase 2 : confirmation sur les plis restants
    specs2 = [make_spec("search", c, cfg_map[c], k, 0, args) for k in rest for c in top]
    run_specs(specs2, args.workers, results, args.dry_run, args.est_epochs)
    rows2 = search.ranking(results["runs"], "search", args.sensors, top, range(N_FOLDS))
    if not rows2:
        print("Phase 2 incomplète.")
        return
    print_ranking(rows2, cfg_map, "Phase 2 : classement sur les 5 plis (graine 0)", baseline=base, limit=len(rows2))
    best = rows2[0]
    close = [r["cid"] for r in rows2[1:] if r["rmse_mean"] - best["rmse_mean"] < best["rmse_std"] / np.sqrt(N_FOLDS)]
    print(f"\nMeilleur réglage : {best['cid']} ({search.describe(cfg_map[best['cid']])}), RMSE {best['rmse_mean']:.2f}.")
    if close:
        print(f"Attention : {', '.join(close)} sont à moins d'une erreur-type du meilleur ({best['rmse_std'] / np.sqrt(N_FOLDS):.2f}) : "
              "indiscernables avec une seule graine.")
    results["search"] = {"best": best["cid"], "overrides": cfg_map[best["cid"]], "screen_folds": screen, "top": top,
                         "ranking": rows2, "n_configs": len(cids), "seed": args.search_seed}
    save_results(results)


def chosen_config(args, results):
    cid = args.preset or results.get("search", {}).get("best")
    if not cid:
        sys.exit("Lancer d'abord --search, ou donner --preset (un réglage nommé ou un identifiant rNNN).")
    return cid, overrides_of(cid, results)


def stage_cv(args, results):
    cid, overrides = chosen_config(args, results)
    print(f"Validation croisée du réglage {cid} : {search.describe(overrides)}")
    specs = [make_spec("cv", cid, overrides, k, s, args, journal=(args.journal and k == 0 and s == 0)) for s in SEEDS for k in range(N_FOLDS)]
    n = reuse(results, "search", "cv", cid, args.sensors, skip_fold0=args.journal)       # la graine 0 existe déjà dans --search
    if n:
        print(f"{n} entraînement(s) de la graine 0 repris de la recherche.")
    run_specs(specs, args.workers, results, args.dry_run, args.est_epochs)
    if args.dry_run:
        return
    f = search.by_fold(results["runs"], "cv", args.sensors, cid)
    if len(f) < N_FOLDS or any(len(v) < len(SEEDS) for v in f.values()):
        print("Validation croisée incomplète.")
        return
    rm = np.array([[r["val"]["rmse"] for r in f[k]] for k in range(N_FOLDS)])        # plis × graines
    sc = np.array([[r["val"]["score"] for r in f[k]] for k in range(N_FOLDS)])
    ac = np.array([[r["val"]["regime_acc"] for r in f[k]] for k in range(N_FOLDS)])
    ep = np.array([[r["best_epoch"] for r in f[k]] for k in range(N_FOLDS)])
    print(f"\n=== Validation croisée du réseau (réglage {cid}, 5 plis × {len(SEEDS)} graines) ===")
    print(f"RMSE   {rm.mean():.2f}   écart entre plis {rm.mean(1).std():.2f}   écart entre graines {rm.std(1).mean():.2f}")
    print(f"score  {sc.mean():.0f} ± {sc.mean(1).std():.0f}   exactitude du régime {ac.mean():.1%}   meilleure époque (médiane) {np.median(ep):.0f}")
    bf = PROCESSED / "baseline_results.json"
    base = json.loads(bf.read_text(encoding="utf-8"))["cv"]["summary"] if bf.exists() else []
    b = next((s for s in base if s["sensors"] == args.sensors and s["cap"] == 150), None)
    if b:
        print(f"baseline (mêmes plis, 17 capteurs, plafond 150) : RMSE {b['rmse_mean']:.2f} ± {b['rmse_std']:.2f}")
    results["cv_summary"] = {"cid": cid, "overrides": overrides, "sensors": args.sensors, "rmse_mean": float(rm.mean()),
                             "rmse_fold_std": float(rm.mean(1).std()), "rmse_seed_std": float(rm.std(1).mean()),
                             "median_best_epoch": float(np.median(ep))}
    save_results(results)


def stage_final(args, results):
    if "cv_summary" not in results:
        sys.exit("Lancer d'abord --cv.")
    cs = results["cv_summary"]
    cid, overrides, n_ep = cs["cid"], cs["overrides"], int(round(cs["median_best_epoch"]))
    print(f"Entraînement final : réglage {cid} ({search.describe(overrides)}), {n_ep} époques (médiane des meilleures époques de la validation croisée)")
    specs = [make_spec("final", cid, overrides, None, s, args, journal=(args.journal and s == 0), fixed_epochs=n_ep) for s in SEEDS]
    run_specs(specs, args.workers, results, args.dry_run, args.est_epochs)
    if args.dry_run:
        return
    rs = [results["runs"][s["key"]] for s in specs if s["key"] in results["runs"]]
    if len(rs) < len(SEEDS):
        return
    print("\n=== Test (une fenêtre par moteur, évalué une seule fois par graine) ===")
    from cmapss.metrics import evaluate_by_subset
    true, subs = np.array(rs[0]["test_true"]), np.array(rs[0]["test_subsets"])
    ens = evaluate_by_subset(np.mean([r["test_pred"] for r in rs], axis=0), true, subs)
    bf = PROCESSED / "baseline_results.json"
    base = json.loads(bf.read_text(encoding="utf-8"))["final"]["test"] if bf.exists() else {}
    print(f"{'jeu':<7}{'RMSE (5 graines)':>22}{'RMSE moyenne des 5':>20}{'score':>10}{'baseline':>10}{'régime exact':>14}")
    reg = np.array(rs[0]["test_regime"])
    for fd in ["FD001", "FD002", "FD003", "FD004", "total"]:
        v = np.array([r["test"][fd]["rmse"] for r in rs])
        acc = np.mean([((np.array(r["test_logit"]) > 0).astype(int) == reg)[(subs == fd) if fd != "total" else slice(None)].mean() for r in rs])
        print(f"{fd:<7}{v.mean():>14.2f} ± {v.std():<5.2f}{ens[fd]['rmse']:>20.2f}{np.mean([r['test'][fd]['score'] for r in rs]):>10.0f}"
              f"{base.get(fd, {}).get('rmse', float('nan')):>10.2f}{acc:>14.1%}")
    results["final_summary"] = {"cid": cid, "epochs": n_ep, "ensemble": ens}
    save_results(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("smoke", "search", "cv", "final", "journal"):
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--preset", default=None, help="réglage : nom (aucune, dropout, bruit, poids, petit, combinee) ou identifiant rNNN d'une recherche")
    ap.add_argument("--set", default=None, help="avec --smoke : valeurs à changer, ex. dropout=0.3,lr=0.003")
    ap.add_argument("--search-n", type=int, default=60, help="nombre de réglages tirés au hasard en plus des réglages nommés")
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--screen-folds", default="0,1", help="plis de la phase de tri")
    ap.add_argument("--top", type=int, default=8, help="nombre de réglages confirmés sur les 5 plis")
    ap.add_argument("--dry-run", action="store_true", help="n'entraîne rien : affiche le plan, le nombre d'entraînements et une durée estimée")
    ap.add_argument("--est-epochs", type=int, default=20, help="avec --dry-run : époques supposées par entraînement pour l'estimation de durée")
    ap.add_argument("--epochs", type=int, default=None, help="nombre maximal d'époques (défaut : 50 ; 3 pour --smoke)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sensors", choices=SENSOR_SETS, default="17")
    ap.add_argument("--device", default="cpu", help="cpu (défaut) ou cuda : à ne choisir que sur décision de l'utilisateur")
    ap.add_argument("--workers", type=int, default=None, help="entraînements en parallèle (défaut : 4, mesuré le plus efficace avec 8 cœurs ; 1 pour --smoke)")
    ap.add_argument("--threads", type=int, default=None, help="threads de calcul par entraînement (défaut : cœurs / workers)")
    args = ap.parse_args()
    if not (args.smoke or args.search or args.cv or args.final):
        ap.print_help()
        return
    args.workers = args.workers or (1 if args.smoke else 4)
    args.threads = args.threads or max(1, (os.cpu_count() or 4) // args.workers)
    args.epochs = args.epochs or (3 if args.smoke else 50)
    if args.smoke:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    results = load_results()
    if args.smoke:
        stage_smoke(args, results)
    if args.search:
        stage_search(args, results)
    if args.cv:
        stage_cv(args, results)
    if args.final:
        stage_final(args, results)


if __name__ == "__main__":
    main()
