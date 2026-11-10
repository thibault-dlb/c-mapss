"""Journaux des deux vidéos (métriques et baseline), à partir des résultats de scripts/baseline.py.

Appelé par `baseline.py --journal`. Les valeurs viennent de vrais calculs :
- les résultats de la validation croisée et du test (data/processed/baseline_results.json) ;
- un ajustement de démonstration de la configuration retenue sur le pli 0, **vérifié identique à la ligne correspondante de
  la validation croisée** (événement `check`) ;
- pour la vidéo de la baseline, les prédictions de ce modèle après 0, 1, 2... itérations (`staged_predict`).
"""
import json
from pathlib import Path

import numpy as np

from cmapss.baseline import columns_for, make_regressor, window_features
from cmapss.folds import make_folds
from cmapss.journal import Journal
from cmapss.loader import fold_data, subset_of
from cmapss.metrics import A_EARLY, A_LATE, SUBSETS, evaluate, evaluate_by_subset, phm_penalty
from cmapss.windows import FEATURES, WINDOW, cap_rul

ROOT = Path(__file__).resolve().parent.parent
METRIQUES_PATH = ROOT / "viz" / "metriques" / "journal.jsonl"
BASELINE_PATH = ROOT / "viz" / "baseline" / "journal.jsonl"
STAGES = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 150, 200]   # itérations montrées dans la vidéo
DEMO_CHANNEL = "s11"                                          # capteur de la scène « résumer une fenêtre »
N_ENGINES_PER_SUBSET = 3


def r(a, n=2):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def write_journals(train, sensor_sets, results_path):
    res = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if "cv" not in res or "final" not in res:
        raise SystemExit("Lancer d'abord --cv et --final.")
    cv, fin = res["cv"], res["final"]
    best = cv["best"]
    feats, cap = sensor_sets[best["sensors"]], best["cap"]
    cols = columns_for(FEATURES, feats)

    # ---- ajustement de démonstration : pli 0, configuration retenue ----
    fd = fold_data(0, train=train, folds=make_folds(train))
    F_tr = window_features(fd.train.X)[:, cols]
    y_tr = fd.train.rul.copy()
    reg = make_regressor().fit(F_tr, cap_rul(y_tr, cap))
    F_va = window_features(fd.val.X)[:, cols]
    pred = reg.predict(F_va)
    m = fd.val_in_range
    met = evaluate(pred[m], fd.val.rul[m])
    cv_row = next(x for x in cv["rows"] if x["sensors"] == best["sensors"] and x["cap"] == cap and x["fold"] == 0)
    diff = abs(met["rmse"] - cv_row["metrics"]["rmse"])
    assert diff < 1e-3, f"l'ajustement de démonstration ne redonne pas la validation croisée (écart {diff})"
    val_by = evaluate_by_subset(pred[m], fd.val.rul[m], subset_of(fd.val.engine_id[m]))
    engines_all = np.asarray(fd.val.engine_id)

    write_metriques(res, fd, pred, m, best, met, val_by, diff)
    write_baseline(res, fd, reg, F_va, m, cols, y_tr, best, cap, cv, fin, met, diff)


# ---------------------------------------------------------------------------------------------------------------------
def write_metriques(res, fd, pred, m, best, met, val_by, diff):
    fin = res["final"]
    j = Journal(METRIQUES_PATH)
    j.log("meta", a_early=A_EARLY, a_late=A_LATE, config=best, fold=0, rul_range=[6, 195], script="scripts/baseline.py")

    # La courbe de pénalité, calculée par la vraie fonction phm_penalty
    d = np.arange(-60, 61)
    j.log("penalty_curve", d=d.tolist(), penalty=r(phm_penalty(d), 4))

    # Quelques moteurs de validation réels : une fenêtre tirée au hasard (graine 0) dans la plage du test
    rng = np.random.default_rng(0)
    ids_in_order = list(dict.fromkeys(fd.val.engine_id))
    engines = []
    for fdn in SUBSETS:
        for e in [x for x in ids_in_order if x.startswith(fdn + "_")][:3]:
            idx = np.where((np.asarray(fd.val.engine_id) == e) & m)[0]
            i = int(rng.choice(idx))
            engines.append({"engine": e, "subset": fdn, "end_cycle": int(fd.val.end_cycle[i]),
                            "true": float(fd.val.rul[i]), "pred": float(pred[i])})
    t = np.array([x["true"] for x in engines]); p = np.array([x["pred"] for x in engines])
    for x, dd, pen in zip(engines, p - t, phm_penalty(p - t)):
        x["error"] = float(dd); x["penalty"] = float(pen)
    j.log("engines", engines=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in engines],
          metrics=evaluate(p, t))

    j.log("validation", fold=0, total=val_by, by_subset={k: v for k, v in val_by.items() if k != "total"})
    j.log("test", by_subset={k: v for k, v in fin["test"].items()}, constant=fin["test_constant"])
    j.log("check", name="demo_fit_matches_cross_validation", max_abs_diff_rmse=float(diff), ok=bool(diff < 1e-3))
    j.close()
    print(f"Journal écrit : {METRIQUES_PATH} ({j.seq} événements)")


def write_baseline(res, fd, reg, F_va, m, cols, y_tr, best, cap, cv, fin, met, diff):
    j = Journal(BASELINE_PATH)
    j.log("meta", window=WINDOW, stats=["moyenne", "écart-type", "minimum", "maximum", "dernière valeur", "pente"],
          config=best, channel=DEMO_CHANNEL, fold=0, script="scripts/baseline.py")

    # 1. Une vraie fenêtre de validation, une colonne, ses 6 statistiques (calculées par window_features)
    ids_in_order = list(dict.fromkeys(fd.val.engine_id))
    demo = next(x for x in ids_in_order if x.startswith("FD001_"))
    eng_idx = np.where(np.asarray(fd.val.engine_id) == demo)[0]
    eng_idx = eng_idx[np.argsort(fd.val.end_cycle[eng_idx])]
    mid = int(eng_idx[len(eng_idx) // 2])
    k = FEATURES.index(DEMO_CHANNEL)
    vals = fd.val.X[mid, :, k]
    st = window_features(fd.val.X[mid:mid + 1, :, k:k + 1])[0]
    mean, std, mn, mx, last, slope = (float(v) for v in st)
    j.log("window_stats", engine=demo, end_cycle=int(fd.val.end_cycle[mid]), rul=float(fd.val.rul[mid]),
          values=r(vals, 3), mean=mean, std=std, min=mn, max=mx, last=last, slope=slope,
          line=[mean - slope * (WINDOW - 1) / 2, mean + slope * (WINDOW - 1) / 2])

    # 2. L'apprentissage étape par étape : prédiction du moteur de démonstration après 0, 1, 2... itérations
    F_e = F_va[eng_idx]
    staged = list(reg.staged_predict(F_e))
    rmse_curve = [float(np.sqrt(np.mean((p - fd.val.rul[m]) ** 2))) for p in reg.staged_predict(F_va[m])]
    start = float(np.mean(cap_rul(y_tr, cap)))                     # départ du boosting : la moyenne des cibles
    start_rmse = float(np.sqrt(np.mean((start - fd.val.rul[m]) ** 2)))
    j.log("learning_begin", engine=demo, n_windows=int(len(eng_idx)), cap=cap, n_iter=len(staged),
          end_cycle=fd.val.end_cycle[eng_idx].tolist(), true=r(fd.val.rul[eng_idx], 1),
          true_capped=r(cap_rul(fd.val.rul[eng_idx], cap), 1), start_value=start, rmse_curve=r(rmse_curve, 3),
          start_rmse=start_rmse)
    for it in STAGES:
        pe = np.full(len(eng_idx), start) if it == 0 else staged[it - 1]
        j.log("stage", iter=it, pred=r(pe, 1), rmse_val=r(start_rmse if it == 0 else rmse_curve[it - 1], 3))
    assert abs(rmse_curve[-1] - met["rmse"]) < 1e-3
    j.log("check", name="last_stage_equals_final_model", max_abs_diff_rmse=abs(rmse_curve[-1] - met["rmse"]), ok=True)

    # 3. Le choix des réglages : résultats réels de la validation croisée
    j.log("choice", summary=cv["summary"], best=best, n_folds=cv["n_folds"], rul_range=cv["rul_range"])
    # 4. Le score à battre : test, une fenêtre par moteur
    j.log("final", config=fin["config"], test=fin["test"], constant=fin["test_constant"],
          regime_accuracy=fin["regime_accuracy"], n_train_windows=fin["n_train_windows"])
    j.log("check", name="demo_fit_matches_cross_validation", max_abs_diff_rmse=float(diff), ok=bool(diff < 1e-3))
    j.close()
    print(f"Journal écrit : {BASELINE_PATH} ({j.seq} événements)")
