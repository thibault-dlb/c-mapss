"""Entraînement du réseau CNN 1D + GRU : recherche évolutive d'hyperparamètres, validation croisée, entraînement final.

Usage (CPU par défaut ; CUDA seulement avec --device cuda, à la seule décision de l'utilisateur ; rien ne se lance seul) :
    .\\.venv\\Scripts\\python.exe scripts\\train.py --search --cv --final --live      # tout le protocole, suivi en direct
    .\\.venv\\Scripts\\python.exe scripts\\train.py --search --dry-run                 # le plan et la durée estimée, sans rien lancer
    .\\.venv\\Scripts\\python.exe scripts\\train.py --search --fake --live             # SIMULATION (aucun entraînement) pour l'interface
    .\\.venv\\Scripts\\python.exe scripts\\train.py --smoke [--epochs 3] [--preset dropout] [--set dropout=0.3,lr=0.003]

Recherche évolutive (--search), réglable : --generations G (défaut 8), --population P (défaut 8), --elite E (défaut 2),
--mutation M (défaut 0,15), --top K (défaut 4).
  - Génération 0 : les 6 réglages nommés puis des tirages au hasard. Chaque réglage est entraîné sur les plis de tri
    (--screen-folds, défaut 0 et 1) ; sa note est son RMSE de validation moyen.
  - Génération suivante : les E meilleurs sont conservés (pas réentraînés), les autres sont fabriqués en croisant deux bons
    réglages puis en mutant quelques paramètres. Jamais un réglage déjà essayé.
  - Confirmation : les K meilleurs de toutes les générations (et « aucune », la référence) sont entraînés sur les plis restants ;
    le classement final se fait sur les 5 plis, comparé pli par pli à « aucune ».
Paramètres cherchés (cmapss/search.py) : dropout, bruit sur les entrées, décroissance des poids, taille du réseau, taille du noyau,
taux d'apprentissage, taille des lots, poids λ de la tête régime, plafond de RUL.
Puis --cv (5 plis × 5 graines avec le meilleur réglage) et --final (tout l'entraînement, test évalué une fois).

Options communes : --threads N (threads par entraînement, défaut : cœurs / --workers), --workers K (entraînements en parallèle,
défaut 4 : 4 × 2 threads donne environ 2,3 fois le débit d'un seul entraînement sur 8 threads, mesuré), --sensors 17|14,
--live (un journal par entraînement, un pointeur par worker et un journal global dans viz/entrainement/runs/ ; à suivre dans
viz/entrainement/live.html), --journal (écrit viz/entrainement/journal.jsonl pour le pli 0 et la graine 0, vue détaillée).

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

from cmapss import evolution, search
from cmapss.data import PROCESSED

RESULTS = PROCESSED / "train_results.json"
JOURNAL = ROOT / "viz" / "entrainement" / "journal.jsonl"
LIVE_DIR = ROOT / "viz" / "entrainement" / "runs"        # un journal par entraînement (--live) ; en simulation : demo/
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
SENSOR_SETS = ("17", "14")
SEC_PER_EPOCH_4X2 = 6.9      # débit mesuré avec 4 entraînements × 2 threads (docs/reseau.md) : secondes par époque, en équivalent
_CACHE = {}
_SLOT = 0                    # numéro du worker (0 à workers-1), fixé par _init_worker

# état de l'orchestration (processus principal seulement)
GJ = None                    # journal global suivi par live.html
PROG = {"total": 0, "done": 0, "done_new": 0, "t0": None, "prior": 120.0}
GEN_OF = {}                  # identifiant de réglage -> génération où il est apparu
PHASES = []                  # [(nom, entraînements)] pour la barre d'avancement


def glog(type_, **fields):
    if GJ is not None:
        GJ.log(type_, **fields)


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


def baseline_on_folds(sensors, folds):
    """RMSE moyen de la baseline sur les plis donnés (comparaison équitable avec la note de tri)."""
    f = PROCESSED / "baseline_results.json"
    if not f.exists():
        return None
    rows = [x for x in json.loads(f.read_text(encoding="utf-8")).get("cv", {}).get("rows", [])
            if x["sensors"] == sensors and x["cap"] == 150 and x["fold"] in folds]
    return float(np.mean([x["metrics"]["rmse"] for x in rows])) if rows else None


# ---------------------------------------------------------------------------------------------------------------------
# Travail d'un entraînement (exécuté dans un processus à part quand --workers > 1)
def _init_worker(counter):
    global _SLOT
    with counter.get_lock():
        _SLOT = counter.value
        counter.value += 1


def run_one(spec):
    from cmapss.journal import Journal, publish_latest

    live_dir = Path(spec["live_dir"])
    if spec.get("live"):                                     # journal propre à cet entraînement, annoncé par le pointeur du worker
        jpath = live_dir / (spec["key"].replace("|", "_") + ".jsonl")
        journal = Journal(jpath)
        publish_latest(live_dir, jpath, name=f"slot{_SLOT}.json")
    else:
        journal = Journal(spec["journal"]) if spec.get("journal") else None
    where = f"pli {spec['fold']}" if spec["fold"] is not None else "tout l'entraînement"
    label = f"{spec['stage']} · réglage {spec['cid']} · {where} · graine {spec['seed']} — {search.describe(spec['overrides'])}"
    run_info = {"run": {"key": spec["key"], "stage": spec["stage"], "cid": spec["cid"], "fold": spec["fold"], "slot": _SLOT,
                        "gen": spec.get("gen"), "label": label[:110]}}
    meta_extra = {**(baseline_reference(spec["sensors"], spec["fold"]) or {}), **run_info}

    if spec.get("fake"):                                     # SIMULATION : rien n'est entraîné
        from cmapss.simulate import simulate_run
        res = simulate_run(spec["overrides"], spec["fold"], spec["seed"], journal=journal, sec_per_epoch=spec["fake_speed"],
                           max_epochs=spec["max_epochs"], patience=spec["patience"], fixed_epochs=spec["fixed_epochs"],
                           meta_extra=meta_extra, preset=spec["cid"])
        res["prep_seconds"] = 0.0
        return spec["key"], res

    import torch
    from cmapss.data import load_all
    from cmapss.folds import make_folds
    from cmapss.loader import features_without, fold_data, load_test_data, SENSORS_UNSTABLE
    from cmapss.metrics import evaluate_by_subset
    from cmapss.normalisation import ConditionScaler
    from cmapss.train import RunData, predict, run_data_from_fold, train_config, train_run
    from cmapss.windows import FEATURES, make_windows

    torch.set_num_threads(spec["threads"])
    feats = FEATURES if spec["sensors"] == "17" else features_without(SENSORS_UNSTABLE)
    train = _CACHE.get("train")
    if train is None:
        train = _CACHE["train"] = load_all("train")
    cfg = train_config(spec["overrides"], preset=spec["cid"], seed=spec["seed"], max_epochs=spec["max_epochs"],
                       patience=spec["patience"], log_kernels=not spec.get("live"))

    t0 = time.perf_counter()
    if spec["fold"] is not None:
        if _CACHE.get("fold") != (spec["fold"], spec["sensors"]):
            _CACHE.pop("data", None)
            fd = fold_data(spec["fold"], features=feats, train=train, folds=make_folds(train, n_splits=N_FOLDS))
            _CACHE["data"], _CACHE["fold"] = run_data_from_fold(fd), (spec["fold"], spec["sensors"])
            del fd
        data = _CACHE["data"]
        prep = time.perf_counter() - t0
        res, _ = train_run(data, cfg, journal=journal, device=spec["device"], meta_extra=meta_extra)
    else:                                                    # entraînement final : tout l'entraînement, test évalué une fois
        scaler = ConditionScaler().fit(train)
        w = make_windows(scaler.transform(train), "train", features=feats)
        data = RunData(X_tr=w.X, rul_tr=w.rul, reg_tr=w.regime)
        prep = time.perf_counter() - t0
        res, model = train_run(data, cfg, journal=journal, device=spec["device"], fixed_epochs=spec["fixed_epochs"], meta_extra=meta_extra)
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


# ---------------------------------------------------------------------------------------------------------------------
# Avancement global (affiché par live.html)
def progress_snapshot():
    elapsed = time.time() - PROG["t0"] if PROG["t0"] else 0.0
    remaining = max(0, PROG["total"] - PROG["done"])
    eta = evolution.eta_seconds(PROG["done_new"], remaining, elapsed, PROG["prior"])
    return {"done": PROG["done"], "total": PROG["total"], "elapsed": round(elapsed, 1), "eta_seconds": round(eta, 1)}


def record_done(key, res, replay=False):
    """Un entraînement est terminé (ou repris d'une exécution précédente) : avancement + événement du journal global."""
    p = search.parse_key(key)
    PROG["done"] += 1
    if not replay:
        PROG["done_new"] += 1
    v = res.get("val") or {}
    t = (res.get("test") or {}).get("total", {})
    glog("run_done", key=key, stage=p["stage"], cid=p["cid"], fold=p["fold"], seed=p["seed"], gen=GEN_OF.get(p["cid"]),
         rmse=v.get("rmse", t.get("rmse")), score=v.get("score", t.get("score")), regime_acc=v.get("regime_acc"),
         best_epoch=res.get("best_epoch"), epochs_run=res.get("epochs_run"), seconds=res.get("seconds"), replay=replay,
         journal=f"{LIVE_DIR.name}/{key.replace('|', '_')}.jsonl", progress=progress_snapshot())


def set_phases(phases):
    """Met à jour le total d'entraînements prévu (barre d'avancement) ; `phases` = [(nom, entraînements)]."""
    global PHASES
    PHASES = list(phases)
    PROG["total"] = sum(n for _, n in PHASES)
    glog("plan_update", phases=[{"name": n, "runs": r} for n, r in PHASES], total=PROG["total"], progress=progress_snapshot())


def run_specs(specs, workers, results, dry_run=False, est_epochs=20, on_done=None, skip_replay=()):
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
    if on_done:
        for s in specs:
            if s["key"] in results["runs"] and s["key"] not in skip_replay:
                on_done(s["key"], results["runs"][s["key"]], True)
    if not todo:
        return
    t0, done = time.time(), 0

    def finish(key, res):
        nonlocal done
        done += 1
        results["runs"][key] = res
        save_results(results)
        if on_done:
            on_done(key, res, False)
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
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(counter,)) as ex:
            futs = [ex.submit(run_one, s) for s in todo]
            for f in cf.as_completed(futs):
                finish(*f.result())


# ---------------------------------------------------------------------------------------------------------------------
def make_spec(stage, cid, overrides, fold, seed, args, journal=False, fixed_epochs=None):
    return {"key": search.run_key(stage, args.sensors, cid, fold, seed), "stage": stage, "cid": cid, "overrides": overrides,
            "fold": fold, "seed": seed, "sensors": args.sensors, "device": args.device, "threads": args.threads,
            "max_epochs": args.epochs, "patience": args.patience, "fixed_epochs": fixed_epochs,
            "journal": str(JOURNAL) if journal else None, "live": args.live or args.fake, "live_dir": str(LIVE_DIR),
            "fake": args.fake, "fake_speed": args.fake_speed, "gen": GEN_OF.get(cid)}


def reuse(results, from_stage, to_stage, cid, sensors, folds=range(N_FOLDS), seed=0, skip_fold0=False):
    """Reprend les entraînements déjà faits d'une étape précédente (même réglage, même graine) : pas de calcul en double."""
    keys = set()
    for k in folds:
        if skip_fold0 and k == 0:
            continue
        a = results["runs"].get(search.run_key(from_stage, sensors, cid, k, seed))
        dst = search.run_key(to_stage, sensors, cid, k, seed)
        if a is not None and dst not in results["runs"]:
            results["runs"][dst] = a
            keys.add(dst)
    return keys


def overrides_of(cid, results):
    """Réglage (écarts au défaut) d'un identifiant : réglage nommé, ou fabriqué par --search (enregistré dans les résultats)."""
    if cid in search.NAMED:
        return dict(search.NAMED[cid])
    cfgs = results.get("search_configs", {})
    if cid in cfgs:
        return dict(cfgs[cid])
    sys.exit(f"Réglage inconnu : {cid}. Noms : {', '.join(search.NAMED)} ; identifiants gNN-MM ou rNNN : voir --search.")


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
    phases = evolution.plan_phases(args.generations, args.population, args.elite, len(screen), args.top, len(rest), N_FOLDS, len(SEEDS))
    total = sum(n for _, n in phases)
    print(f"Recherche évolutive : {args.generations} générations de {args.population} réglages (dont {args.elite} élites conservées), "
          f"tri sur les plis {screen}, puis les {args.top} meilleurs sur les plis {rest}.")
    print(f"Entraînements prévus : {total} au plus (dont validation croisée et final). "
          + "  ".join(f"{n} : {r}" for n, r in phases[:3]) + " ...")
    pop0 = evolution.initial_population(args.population, args.search_seed)
    if args.dry_run:
        minutes = total * args.est_epochs * SEC_PER_EPOCH_4X2 / 60
        print(f"  [simulation] rien n'est lancé. Durée estimée si chaque entraînement dure {args.est_epochs} époques : "
              f"{minutes:.0f} min ({minutes / 60:.1f} h), d'après {SEC_PER_EPOCH_4X2} s par époque mesurées avec 4 × 2 threads "
              "(estimation : le vrai nombre d'époques n'est connu qu'après les premiers entraînements).")
        print("Génération 0 :")
        for m in pop0:
            print(f"  {m['cid']:<9} {m['label']}")
        return
    cfg_map = {}
    for m in pop0:
        GEN_OF[m["cid"]] = 0
        cfg_map[m["cid"]] = m["overrides"]
    set_phases(phases)
    base_screen = baseline_on_folds(args.sensors, screen)
    glog("baseline", screen_rmse=base_screen, **{k: v for k, v in (baseline_reference(args.sensors, None) or {}).get("reference", {}).items()})

    def on_event(type_, **f):
        if type_ == "generation_start":
            for m in f["members"]:
                cfg_map.setdefault(m["cid"], m["overrides"])
                GEN_OF.setdefault(m["cid"], f["gen"])
            glog("generation_start", gen=f["gen"], members=[{k: m[k] for k in ("cid", "label", "parents", "elite")} for m in f["members"]],
                 n_new_runs=sum(1 for m in f["members"] if not m["elite"]) * len(screen))
        elif type_ == "generation_end":
            slim = [{"cid": r["cid"], "fitness": r["fitness"], "label": r["label"], "elite": r["elite"], "parents": r["parents"]}
                    for r in f["ranking"]]
            glog("generation_end", gen=f["gen"], ranking=slim, best={"cid": f["best"]["cid"], "fitness": f["best"]["fitness"],
                 "label": f["best"]["label"]}, mean=f["mean"], delta=f["delta"], best_so_far=f["best_so_far"], progress=progress_snapshot())
            print(f"  génération {f['gen']} : meilleur {f['best']['cid']} {f['best']['fitness']:.2f}"
                  + ("" if f["delta"] is None else f" ({f['delta']:+.2f} par rapport au meilleur précédent)") + f", moyenne {f['mean']:.2f}", flush=True)

    def evaluate(members, gen):
        new = [m for m in members if not m["elite"]]            # les élites ont déjà été entraînées
        specs = [make_spec("search", m["cid"], m["overrides"], k, 0, args) for k in screen for m in new]
        run_specs(specs, args.workers, results, False, args.est_epochs, on_done=record_done)
        fit = {}
        for m in members:
            f = search.by_fold(results["runs"], "search", args.sensors, m["cid"], seeds=[0])
            fit[m["cid"]] = float(np.mean([f[k][0]["val"]["rmse"] for k in screen])) if all(k in f for k in screen) else float("inf")
        return fit

    glog("phase", name="tri", detail="recherche évolutive")
    history = evolution.run_evolution(evaluate, args.generations, args.population, args.elite, args.mutation, args.search_seed, on_event)
    results["search_configs"] = {**results.get("search_configs", {}), **{c: o for c, o in cfg_map.items() if c not in search.NAMED}}
    save_results(results)

    # confirmation : les meilleurs de toutes les générations sur les plis restants
    best_by_cid = {}
    for h in history:
        for r in h["ranking"]:
            best_by_cid.setdefault(r["cid"], r["fitness"])
    top = [c for c, _ in sorted(best_by_cid.items(), key=lambda kv: (kv[1], kv[0]))[:args.top]]
    if "aucune" not in top:
        top.append("aucune")
    specs2 = [make_spec("search", c, cfg_map[c], k, 0, args) for k in rest for c in top]
    set_phases(phases[:args.generations] + [("confirmation", len(specs2)), phases[-2], phases[-1]])
    glog("phase", name="confirmation", detail=f"{len(top)} réglages sur les plis {rest}")
    print(f"\nConfirmation : {len(top)} réglages ({', '.join(top)}) sur les plis {rest}.")
    run_specs(specs2, args.workers, results, False, args.est_epochs, on_done=record_done)
    rows2 = search.ranking(results["runs"], "search", args.sensors, top, range(N_FOLDS))
    if not rows2:
        print("Confirmation incomplète.")
        return
    base = baseline_on_folds(args.sensors, range(N_FOLDS))
    print_ranking(rows2, cfg_map, "Classement sur les 5 plis (graine 0)", baseline=base, limit=len(rows2))
    best = rows2[0]
    close = [r["cid"] for r in rows2[1:] if r["rmse_mean"] - best["rmse_mean"] < best["rmse_std"] / np.sqrt(N_FOLDS)]
    print(f"\nMeilleur réglage : {best['cid']} ({search.describe(cfg_map[best['cid']])}), RMSE {best['rmse_mean']:.2f}.")
    if close:
        print(f"Attention : {', '.join(close)} sont à moins d'une erreur-type du meilleur ({best['rmse_std'] / np.sqrt(N_FOLDS):.2f}) : "
              "indiscernables avec une seule graine.")
    glog("confirm_end", ranking=[{**r, "label": search.describe(cfg_map[r["cid"]])} for r in rows2], best=best["cid"],
         baseline=base, close=close, progress=progress_snapshot())
    results["search"] = {"best": best["cid"], "overrides": cfg_map[best["cid"]], "screen_folds": screen, "top": top,
                         "ranking": rows2, "generations": [{"gen": h["gen"], "best": h["best"]["cid"], "best_rmse": h["best"]["fitness"],
                                                            "mean": h["mean"], "delta": h["delta"]} for h in history], "seed": args.search_seed}
    save_results(results)


def chosen_config(args, results):
    cid = args.preset or results.get("search", {}).get("best")
    if not cid:
        sys.exit("Lancer d'abord --search, ou donner --preset (un réglage nommé ou un identifiant gNN-MM).")
    return cid, overrides_of(cid, results)


def stage_cv(args, results):
    cid, overrides = chosen_config(args, results)
    print(f"Validation croisée du réglage {cid} : {search.describe(overrides)}")
    specs = [make_spec("cv", cid, overrides, k, s, args, journal=(args.journal and k == 0 and s == 0)) for s in SEEDS for k in range(N_FOLDS)]
    reused = reuse(results, "search", "cv", cid, args.sensors, skip_fold0=args.journal)      # la graine 0 existe déjà dans --search
    if reused:
        print(f"{len(reused)} entraînement(s) de la graine 0 repris de la recherche.")
    if not args.dry_run:
        n_cv = sum(1 for s in specs if s["key"] not in reused)                                # entraînements propres à cette étape
        fixed = [(n, r) for n, r in PHASES if n not in ("validation croisée", "final")]
        set_phases(fixed + [("validation croisée", n_cv), ("final", len(SEEDS))])
        glog("phase", name="validation croisée", detail=f"réglage {cid}, 5 plis × 5 graines")
    run_specs(specs, args.workers, results, args.dry_run, args.est_epochs, on_done=record_done, skip_replay=reused)
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
    glog("cv_end", cid=cid, rmse_mean=float(rm.mean()), rmse_fold_std=float(rm.mean(1).std()), rmse_seed_std=float(rm.std(1).mean()),
         baseline=(b or {}).get("rmse_mean"), progress=progress_snapshot())
    save_results(results)


def stage_final(args, results):
    if "cv_summary" not in results:
        sys.exit("Lancer d'abord --cv.")
    cs = results["cv_summary"]
    cid, overrides, n_ep = cs["cid"], cs["overrides"], int(round(cs["median_best_epoch"]))
    print(f"Entraînement final : réglage {cid} ({search.describe(overrides)}), {n_ep} époques (médiane des meilleures époques de la validation croisée)")
    specs = [make_spec("final", cid, overrides, None, s, args, journal=(args.journal and s == 0), fixed_epochs=n_ep) for s in SEEDS]
    if not args.dry_run:
        glog("phase", name="final", detail=f"réglage {cid}, {n_ep} époques, test évalué une fois")
    run_specs(specs, args.workers, results, args.dry_run, args.est_epochs, on_done=record_done)
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
    table = {}
    for fd in ["FD001", "FD002", "FD003", "FD004", "total"]:
        v = np.array([r["test"][fd]["rmse"] for r in rs])
        acc = np.mean([((np.array(r["test_logit"]) > 0).astype(int) == reg)[(subs == fd) if fd != "total" else slice(None)].mean() for r in rs])
        print(f"{fd:<7}{v.mean():>14.2f} ± {v.std():<5.2f}{ens[fd]['rmse']:>20.2f}{np.mean([r['test'][fd]['score'] for r in rs]):>10.0f}"
              f"{base.get(fd, {}).get('rmse', float('nan')):>10.2f}{acc:>14.1%}")
        table[fd] = {"rmse_mean": float(v.mean()), "rmse_std": float(v.std()), "rmse_ensemble": float(ens[fd]["rmse"]),
                     "baseline": base.get(fd, {}).get("rmse"), "regime_acc": float(acc)}
    results["final_summary"] = {"cid": cid, "epochs": n_ep, "ensemble": ens}
    glog("final_end", cid=cid, table=table, progress=progress_snapshot())
    save_results(results)


def main():
    global RESULTS, LIVE_DIR, GJ
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("smoke", "search", "cv", "final", "journal"):
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--live", action="store_true", help="écrit un journal par entraînement, un pointeur par worker et un journal global dans viz/entrainement/runs/ (suivi : live.html)")
    ap.add_argument("--fake", action="store_true", help="SIMULATION : n'entraîne rien, produit de faux journaux dans viz/entrainement/demo/ (pour valider l'interface)")
    ap.add_argument("--fake-speed", type=float, default=0.4, help="avec --fake : secondes simulées par époque")
    ap.add_argument("--preset", default=None, help="réglage : nom (aucune, dropout, bruit, poids, petit, combinee) ou identifiant gNN-MM d'une recherche")
    ap.add_argument("--set", default=None, help="avec --smoke : valeurs à changer, ex. dropout=0.3,lr=0.003")
    ap.add_argument("--generations", type=int, default=8, help="nombre de générations de la recherche évolutive")
    ap.add_argument("--population", type=int, default=8, help="réglages par génération")
    ap.add_argument("--elite", type=int, default=2, help="réglages conservés tels quels d'une génération à la suivante")
    ap.add_argument("--mutation", type=float, default=0.15, help="probabilité de mutation de chaque paramètre")
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--screen-folds", default="0,1", help="plis de la phase de tri")
    ap.add_argument("--top", type=int, default=4, help="nombre de réglages confirmés sur les 5 plis")
    ap.add_argument("--dry-run", action="store_true", help="n'entraîne rien : affiche le plan, le nombre d'entraînements et une durée estimée")
    ap.add_argument("--est-epochs", type=int, default=20, help="époques supposées par entraînement pour l'estimation de durée")
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
    if args.fake:                                            # la simulation a ses propres résultats et son propre dossier
        RESULTS = PROCESSED / "train_results_fake.json"
        LIVE_DIR = ROOT / "viz" / "entrainement" / "demo"
    if args.smoke:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    results = load_results()

    if (args.live or args.fake) and not args.dry_run and (args.search or args.cv or args.final):
        from cmapss.journal import Journal
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        GJ = Journal(LIVE_DIR / "search.jsonl")
        per_run = (args.est_epochs * (args.fake_speed if args.fake else SEC_PER_EPOCH_4X2)) / (args.workers if args.fake else 1)
        PROG.update(t0=time.time(), prior=per_run, done=0, done_new=0)
        glog("plan", demo=bool(args.fake), mode="evolution", generations=args.generations, population=args.population, elite=args.elite,
             mutation=args.mutation, screen_folds=[int(x) for x in args.screen_folds.split(",")], top=args.top, workers=args.workers,
             threads=args.threads, sensors=args.sensors, est_epochs=args.est_epochs, prior_per_run=per_run,
             stages=[s for s in ("search", "cv", "final") if getattr(args, s)], space={k: v for k, v in search.SPACE.items()})
    try:
        if args.smoke:
            stage_smoke(args, results)
        if args.search:
            stage_search(args, results)
        if args.cv:
            stage_cv(args, results)
        if args.final:
            stage_final(args, results)
    finally:
        if GJ is not None:
            glog("phase", name="terminé")
            GJ.close()


if __name__ == "__main__":
    main()
