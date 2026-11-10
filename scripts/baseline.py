"""Baseline Gradient Boosting : choix du plafond de RUL et des capteurs par validation croisée, puis évaluation du test.

Usage :
    .\\.venv\\Scripts\\python.exe scripts\\baseline.py --cv       # 2 jeux de capteurs × 5 plafonds (100 à 200) × 5 plis = 50 ajustements
    .\\.venv\\Scripts\\python.exe scripts\\baseline.py --final    # meilleure configuration sur tout l'entraînement, test évalué UNE fois
    .\\.venv\\Scripts\\python.exe scripts\\baseline.py --journal  # journaux des deux vidéos (métriques et baseline)
    .\\.venv\\Scripts\\python.exe scripts\\baseline.py --all

Validation : fenêtres des moteurs de validation dont la vraie RUL est entre 6 et 195 (plage du test), mesurées contre la vraie
RUL non plafonnée. Choix de la configuration par RMSE moyen sur les 5 plis (le score PHM08 est dominé par quelques valeurs
extrêmes : il est rapporté, pas utilisé pour choisir). Le test n'est touché que par --final.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from cmapss.baseline import columns_for, make_classifier, make_regressor, window_features
from cmapss.data import PROCESSED, load_all
from cmapss.folds import make_folds
from cmapss.loader import (N_FOLDS, SENSORS_UNSTABLE, features_without, fold_data, load_test_data, subset_of)
from cmapss.metrics import evaluate, evaluate_by_subset
from cmapss.normalisation import ConditionScaler
from cmapss.windows import FEATURES, cap_rul, make_windows

CAPS = [100, 125, 150, 175, 200]   # 175 et 200 ajoutés car l'optimum était au bord de la grille (150)
RUL_BRACKETS = [(6, 50), (50, 100), (100, 150), (150, 196)]   # tranches de vraie RUL (borne haute exclue) pour comprendre l'effet du plafond
SENSOR_SETS = {"17": FEATURES, "14": features_without(SENSORS_UNSTABLE)}   # 17 capteurs, ou 14 sans s6, s10, s16
RESULTS = PROCESSED / "baseline_results.json"


def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def save_results(res):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1), encoding="utf-8")


def run_cv(train):
    folds = make_folds(train, n_splits=N_FOLDS)
    rows, t_all = [], time.time()
    # Les ajustements déjà calculés (mêmes capteurs, plafond, pli) sont réutilisés : les calculs sont déterministes
    # (graine fixe), seuls les nouveaux plafonds sont ajustés.
    old = load_results().get("cv", {})
    done = {(x["sensors"], x["cap"], x["fold"]): x for x in old.get("rows", [])} if old.get("n_folds") == N_FOLDS else {}
    for k in range(N_FOLDS):
        t0 = time.time()
        fd = fold_data(k, train=train, folds=folds)
        F_tr = window_features(fd.train.X)
        y_tr = fd.train.rul.copy()
        m = fd.val_in_range
        F_va, y_va = window_features(fd.val.X[m]), fd.val.rul[m]
        subs_va = subset_of(fd.val.engine_id[m])
        n_tr, n_va = len(y_tr), int(m.sum())
        del fd                                              # libère ~340 Mo de fenêtres avant les ajustements
        print(f"\npli {k} : {n_tr} fenêtres d'entraînement, {n_va} de validation dans la plage du test "
              f"(préparation {time.time() - t0:.0f} s)", flush=True)
        for name, feats in SENSOR_SETS.items():
            cols = columns_for(FEATURES, feats)
            for cap in CAPS:
                if (name, cap, k) in done:
                    rows.append(done[(name, cap, k)])
                    print(f"  {name} capteurs, plafond {cap} : déjà calculé (RMSE {rows[-1]['metrics']['rmse']:.2f})", flush=True)
                    continue
                t1 = time.time()
                model = make_regressor().fit(F_tr[:, cols], cap_rul(y_tr, cap))
                pred = model.predict(F_va[:, cols])
                res = evaluate_by_subset(pred, y_va, subs_va)
                by_rul = {f"{a}-{b - 1}": {"n": int(((y_va >= a) & (y_va < b)).sum()),
                                           "rmse": float(np.sqrt(np.mean((pred - y_va)[(y_va >= a) & (y_va < b)] ** 2)))}
                          for a, b in RUL_BRACKETS if ((y_va >= a) & (y_va < b)).any()}
                rows.append({"sensors": name, "cap": cap, "fold": k, "fit_seconds": round(time.time() - t1, 1),
                             "metrics": res["total"], "by_subset": {s: res[s] for s in res if s != "total"},
                             "by_rul": by_rul})
                print(f"  {name} capteurs, plafond {cap} : RMSE {res['total']['rmse']:.2f}  score {res['total']['score']:.0f}"
                      f"  biais {res['total']['bias']:+.1f}  ({rows[-1]['fit_seconds']} s)", flush=True)
    summary = []
    for name in SENSOR_SETS:
        for cap in CAPS:
            r = [x["metrics"] for x in rows if x["sensors"] == name and x["cap"] == cap]
            f = lambda key: [x[key] for x in r]
            summary.append({"sensors": name, "cap": cap,
                            "rmse_mean": float(np.mean(f("rmse"))), "rmse_std": float(np.std(f("rmse"))),
                            "score_mean": float(np.mean(f("score"))), "score_std": float(np.std(f("score"))),
                            "bias_mean": float(np.mean(f("bias"))), "late_rate_mean": float(np.mean(f("late_rate")))})
    for s_ in summary:
        rr = [x["by_rul"] for x in rows if x["sensors"] == s_["sensors"] and x["cap"] == s_["cap"]]
        s_["rmse_by_rul"] = {b: float(np.mean([r_[b]["rmse"] for r_ in rr if b in r_])) for b in rr[0]}
    best = min(summary, key=lambda s: s["rmse_mean"])
    print("\n=== Validation croisée (moyenne ± écart-type entre plis) ===")
    print(f"{'capteurs':>9} {'plafond':>8} {'RMSE':>14} {'score PHM08':>20} {'biais':>8}")
    for s in summary:
        mark = "  <- retenue" if s is best else ""
        print(f"{s['sensors']:>9} {s['cap']:>8} {s['rmse_mean']:>8.2f} ± {s['rmse_std']:<4.2f} "
              f"{s['score_mean']:>12.0f} ± {s['score_std']:<6.0f} {s['bias_mean']:>+8.1f}{mark}")
    print("\n=== RMSE par tranche de vraie RUL (moyenne sur les plis) ===")
    brs = list(summary[0]["rmse_by_rul"])
    print(f"{'capteurs':>9} {'plafond':>8} " + " ".join(f"{b:>9}" for b in brs))
    for s in summary:
        print(f"{s['sensors']:>9} {s['cap']:>8} " + " ".join(f"{s['rmse_by_rul'][b]:>9.1f}" for b in brs))
    res = load_results()
    res["cv"] = {"caps": CAPS, "sensor_sets": {k: v for k, v in SENSOR_SETS.items()}, "n_folds": N_FOLDS,
                 "rul_range": [6, 195], "rows": rows, "summary": summary,
                 "best": {"sensors": best["sensors"], "cap": best["cap"]}, "seconds": round(time.time() - t_all)}
    save_results(res)
    print(f"\nRésultats écrits dans {RESULTS} ({time.time() - t_all:.0f} s au total)")


def run_final(train):
    res = load_results()
    if "cv" not in res:
        sys.exit("Lancer d'abord --cv.")
    best = res["cv"]["best"]
    feats, cap = SENSOR_SETS[best["sensors"]], best["cap"]
    print(f"Configuration retenue par la validation croisée : {best['sensors']} capteurs, plafond {cap}")
    scaler = ConditionScaler().fit(train)
    w_tr = make_windows(scaler.transform(train), "train")
    F_tr, y_tr = window_features(w_tr.X), w_tr.rul.copy()
    reg_tr = w_tr.regime.copy()
    del w_tr
    cols = columns_for(FEATURES, feats)
    model = make_regressor().fit(F_tr[:, cols], cap_rul(y_tr, cap))
    clf = make_classifier().fit(F_tr[:, cols], reg_tr)

    td = load_test_data(train=train)                         # le test n'est touché qu'ici, une seule fois
    F_te = window_features(td.windows.X)[:, cols]
    pred, true = model.predict(F_te), td.windows.rul
    by = evaluate_by_subset(pred, true, td.subsets)
    const = float(np.mean(cap_rul(y_tr, cap)))                # point de repère : prédire toujours la RUL moyenne
    by_const = evaluate_by_subset(np.full_like(true, const, dtype=float), true, td.subsets)
    reg_pred = clf.predict(F_te)
    acc = {fd: float((reg_pred[td.subsets == fd] == td.windows.regime[td.subsets == fd]).mean()) for fd in by if fd != "total"}
    acc["total"] = float((reg_pred == td.windows.regime).mean())

    print("\n=== Test (une fenêtre par moteur ; la baseline contre la prédiction constante) ===")
    print(f"{'jeu':<7}{'n':>5}{'RMSE':>9}{'score PHM08':>14}{'biais':>8}{'RMSE const.':>13}{'régime exact':>14}")
    for fd, m in by.items():
        print(f"{fd:<7}{m['n']:>5}{m['rmse']:>9.2f}{m['score']:>14.0f}{m['bias']:>+8.1f}"
              f"{by_const[fd]['rmse']:>13.2f}{acc[fd]:>14.1%}")
    res["final"] = {"config": best, "test": by, "test_constant": by_const, "constant": const,
                    "regime_accuracy": acc, "n_train_windows": int(len(y_tr))}
    save_results(res)
    print(f"\nRésultats écrits dans {RESULTS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--journal", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not (args.cv or args.final or args.journal or args.all):
        ap.print_help()
        return
    train = load_all("train")
    if args.cv or args.all:
        run_cv(train)
    if args.final or args.all:
        run_final(train)
    if args.journal or args.all:
        from baseline_journal import write_journals     # scripts/baseline_journal.py
        write_journals(train, SENSOR_SETS, RESULTS)


if __name__ == "__main__":
    main()
