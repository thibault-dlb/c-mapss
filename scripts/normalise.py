"""Normalisation par condition de vol : plis de validation, statistiques par pli, contrôles.

Usage :
    .\\.venv\\Scripts\\python.exe scripts\\normalise.py            # plis, scalers, contrôles
    .\\.venv\\Scripts\\python.exe scripts\\normalise.py --journal  # + journal d'événements pour la page viz/normalisation/

Écrit dans data/processed/ : folds.csv (un pli par moteur) et scalers.json (statistiques par pli
et « full » = tout l'entraînement, pour le test final). Aucune copie normalisée des données n'est
enregistrée : l'entraînement appliquera le scaler du pli à la volée.

Avec --journal, le scaler « full » est ajusté en journalisant chaque étape réelle (lecture, μ et σ
bloc par bloc, états d'un moteur). La page rejoue ce journal sans rien recalculer.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cmapss.data import CONDITIONS, PROCESSED, SENSORS, SENSORS_KEPT, SUBSETS, load_all
from cmapss.folds import make_folds
from cmapss.journal import Journal
from cmapss.normalisation import BLOCK, ConditionScaler

N_FOLDS = 5
JOURNAL_PATH = ROOT / "viz" / "normalisation" / "journal.jsonl"
DEMO_ENGINE = "FD002_1"   # moteur montré dans la vidéo (6 conditions : le bruit des conditions se voit)
DEMO_SENSOR = "s7"        # capteur des scènes « calcul de μ et σ », « un moteur » et « distributions »
N_SETTINGS_POINTS = 3000
HIST_BINS = 80
TREND_WINDOW = 11


def r(a, n=4):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def check_scalers(train, folds, scalers, test):
    """Contrôles chiffrés : moyenne ~ 0, écart-type ~ 1 sur l'entraînement, proches sur la validation."""
    print("\n=== Contrôles par pli ===")
    print(f"{'pli':>4} {'moteurs ent.':>13} {'moteurs val.':>13} {'|moy| ent.':>11} {'ect ent.':>9} "
          f"{'|moy| val.':>11} {'ect val. (min-max)':>20}")
    for k in range(N_FOLDS):
        val_ids = set(folds.loc[folds["fold"] == k, "engine_id"])
        is_val = train["engine_id"].isin(val_ids)
        tr, va = train[~is_val], train[is_val]
        s_tr, s_va = scalers[k].transform(tr), scalers[k].transform(va)
        assert not s_tr[SENSORS_KEPT].isna().any().any() and not s_va[SENSORS_KEPT].isna().any().any()
        m_tr = s_tr.groupby("cond")[SENSORS_KEPT].mean().abs().to_numpy().max()
        m_va = s_va.groupby("cond")[SENSORS_KEPT].mean().abs().to_numpy().max()
        sd_tr = s_tr.groupby("cond")[SENSORS_KEPT].std(ddof=0).to_numpy()
        sd_va = s_va.groupby("cond")[SENSORS_KEPT].std(ddof=0).to_numpy()
        sd_tr, sd_va = sd_tr[sd_tr > 1e-8], sd_va[sd_va > 1e-8]  # on exclut les capteurs constants
        print(f"{k:>4} {tr['engine_id'].nunique():>13} {va['engine_id'].nunique():>13} {m_tr:>11.1e} "
              f"{sd_tr.mean():>9.3f} {m_va:>11.3f} {sd_va.min():>9.2f} – {sd_va.max():<8.2f}")

    # Test : le scaler « full » ne doit rencontrer que des conditions connues, sans NaN
    s_te = scalers["full"].transform(test)
    assert not s_te[SENSORS_KEPT].isna().any().any()
    print(f"\nTest : {len(test)} lignes, {test['engine_id'].nunique()} moteurs, "
          f"conditions toutes connues, aucun NaN.")

    flat = scalers["full"].flat
    pairs = sorted((s, c) for c, ss in flat.items() for s in ss)
    print(f"Capteurs constants dans certaines conditions (normalisés à 0 là) : {len(pairs)} couples")
    for s in sorted({s for s, _ in pairs}):
        conds = [CONDITIONS.index(c) for ss, c in pairs if ss == s]
        print(f"  {s} : conditions n°{conds}")


def fit_full_with_journal(train, journal):
    """Ajuste le scaler « full » (celui utilisé pour le test) en journalisant chaque étape réelle."""
    cond_of = {c: i for i, c in enumerate(CONDITIONS)}
    j = SENSORS_KEPT.index(DEMO_SENSOR)

    # Lecture des données, jeu par jeu
    for fd in SUBSETS:
        part = train[train["subset"] == fd]
        journal.log("read", subset=fd, rows=int(len(part)), engines=int(part["engine_id"].nunique()))

    # Les 6 conditions de vol et les jeux qui les utilisent
    journal.log("conditions", conditions=[
        {"key": c, "altitude": float(c.split("|")[0]), "mach": float(c.split("|")[1]),
         "tra": float(c.split("|")[2]), "n": int((train["cond"] == c).sum()),
         "subsets": sorted(train.loc[train["cond"] == c, "subset"].unique().tolist())}
        for c in CONDITIONS])

    # Échantillon des réglages (aléatoire reproductible) : [altitude, Mach, TRA, indice de condition, indice du jeu]
    sample = train.sample(N_SETTINGS_POINTS, random_state=0)
    journal.log("settings_sample", n=N_SETTINGS_POINTS, points=[
        [round(float(a), 4), round(float(b), 4), round(float(c), 2), cond_of[k], SUBSETS.index(fd)]
        for a, b, c, k, fd in sample[["op1", "op2", "op3", "cond", "subset"]].to_numpy()])

    # Calcul incrémental de μ et σ : une ligne de journal par bloc, valeurs du capteur de démonstration
    journal.log("stats_begin", sensor=DEMO_SENSOR, block=BLOCK, n_rows=int(len(train)))

    def on_block(k, n_blocks, rows_seen, block_df, snap):
        journal.log("stats_update", block=k, n_blocks=n_blocks, rows_seen=rows_seen,
                    subsets=sorted(block_df["subset"].unique().tolist()),
                    cond=[[int(snap[c][0]), round(float(snap[c][1][j]), 4), round(float(snap[c][2][j]), 4)]
                          if c in snap else None for c in CONDITIONS])

    scaler = ConditionScaler().fit(train, block=BLOCK, on_block=on_block)

    journal.log("stats_final", sensors=scaler.sensors, conditions=[
        {"key": c, "n": scaler.n_rows[c], "mean": r(scaler.mean[c], 5), "std": r(scaler.std[c], 5),
         "flat": scaler.flat.get(c, [])} for c in CONDITIONS])

    # Vérification : le calcul par blocs égale le calcul direct de pandas
    diff_mean = diff_std = 0.0
    for c in CONDITIONS:
        rows = train.loc[train["cond"] == c, SENSORS_KEPT]
        diff_mean = max(diff_mean, float(np.abs(scaler.mean[c] - rows.mean().to_numpy()).max()))
        sd = rows.std(ddof=0).to_numpy()
        ok = sd > 1e-8
        diff_std = max(diff_std, float(np.abs(scaler.std[c][ok] - sd[ok]).max()))
    assert diff_mean < 1e-9 and diff_std < 1e-9, (diff_mean, diff_std)
    journal.log("check", name="incremental_vs_pandas", max_abs_diff_mean=diff_mean, max_abs_diff_std=diff_std)
    return scaler


def _linear_fit(x, y):
    """Droite y = pente·x + ordonnée, par moindres carrés, avec le R² pour dire à quel point elle colle."""
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - float((resid**2).sum() / ((y - y.mean()) ** 2).sum())
    return {"method": "moindres carrés (np.polyfit, degré 1)", "slope": float(slope),
            "intercept": float(intercept), "x0": float(x[0]), "x1": float(x[-1]), "r2": round(r2, 4)}


def journal_engine_and_hists(train, scaler_full, journal):
    """États réels (brut, x − μ, (x − μ) / σ) d'un moteur pour les 21 capteurs, puis histogrammes."""
    eng = train[train["engine_id"] == DEMO_ENGINE].sort_values("cycle")
    scaler_all = ConditionScaler(sensors=SENSORS).fit(train)   # mêmes calculs, 21 capteurs
    centered, scaled = scaler_all.transform_steps(eng)

    # Cohérence avec le scaler « full » (17 capteurs) : même valeur normalisée pour le capteur de démo
    jd = SENSORS.index(DEMO_SENSOR)
    ref = scaler_full.transform(eng)[DEMO_SENSOR].to_numpy()
    assert np.allclose(scaled[:, jd], ref, atol=1e-9)

    always_flat = [s for s in SENSORS if all(s in scaler_all.flat.get(c, []) for c in CONDITIONS)]
    cond_of = {c: i for i, c in enumerate(CONDITIONS)}
    kd = SENSORS_KEPT.index(DEMO_SENSOR)
    journal.log(
        "engine_states", engine=DEMO_ENGINE, demo_sensor=DEMO_SENSOR,
        cycle=eng["cycle"].tolist(), cond=[cond_of[c] for c in eng["cond"]],
        mu=r([scaler_full.mean[c][kd] for c in CONDITIONS], 4),
        sigma=r([scaler_full.std[c][kd] for c in CONDITIONS], 4),
        constant_sensors=always_flat,
        # droite d'approximation de la valeur normalisée du capteur de démo (moindres carrés), calculée ici
        fit=_linear_fit(eng["cycle"].to_numpy(dtype=float), scaled[:, jd]),
        trend_window=TREND_WINDOW,
        # tendance du capteur de démo : moyenne glissante de la valeur normalisée (calculée ici, pas dans la page)
        trend=r(pd.Series(scaled[:, jd]).rolling(TREND_WINDOW, min_periods=1).mean().to_numpy()),
        sensors={s: {"raw": r(eng[s].to_numpy()), "centered": r(centered[:, i]), "scaled": r(scaled[:, i])}
                 for i, s in enumerate(SENSORS)})

    norm_all = scaler_full.transform(train)
    for kind, series, lo, hi in (
        ("raw", train[DEMO_SENSOR], train[DEMO_SENSOR].min(), train[DEMO_SENSOR].max()),
        ("scaled", norm_all[DEMO_SENSOR], -4.0, 4.0),
    ):
        edges = np.linspace(lo, hi, HIST_BINS + 1)
        dens = [r(np.histogram(series[train["cond"] == c], bins=edges, density=True)[0], 5) for c in CONDITIONS]
        journal.log("histogram", kind=kind, sensor=DEMO_SENSOR, edges=r(edges, 5), density=dens)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", action="store_true",
                    help=f"écrit le journal d'événements ({JOURNAL_PATH.relative_to(ROOT)})")
    args = ap.parse_args()

    train, test = load_all("train"), load_all("test")
    print(f"Entraînement : {len(train)} lignes, {train['engine_id'].nunique()} moteurs")
    unknown = sorted(set(test["cond"]) - set(CONDITIONS))
    if unknown:
        raise SystemExit(f"Conditions de vol inconnues dans le test : {unknown}")

    folds = make_folds(train, n_splits=N_FOLDS)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    folds.to_csv(PROCESSED / "folds.csv", index=False)
    print("\n=== Moteurs par pli et par jeu ===")
    print(folds.pivot_table(index="fold", columns="subset", values="unit", aggfunc="count").to_string())

    scalers = {}
    for k in range(N_FOLDS):
        val_ids = set(folds.loc[folds["fold"] == k, "engine_id"])
        scalers[k] = ConditionScaler().fit(train[~train["engine_id"].isin(val_ids)])

    journal = Journal(JOURNAL_PATH) if args.journal else None
    if journal:
        journal.log("meta", n_rows=int(len(train)), n_engines=int(train["engine_id"].nunique()),
                    sensors_all=SENSORS, sensors_kept=SENSORS_KEPT, subsets=SUBSETS,
                    demo_engine=DEMO_ENGINE, demo_sensor=DEMO_SENSOR, script="scripts/normalise.py")
        scalers["full"] = fit_full_with_journal(train, journal)
    else:
        scalers["full"] = ConditionScaler().fit(train)

    payload = {"n_folds": N_FOLDS, "folds": {str(k): scalers[k].to_dict() for k in range(N_FOLDS)},
               "full": scalers["full"].to_dict()}
    (PROCESSED / "scalers.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nÉcrit : {PROCESSED / 'folds.csv'} et {PROCESSED / 'scalers.json'}")

    check_scalers(train, folds, scalers, test)
    if journal:
        journal_engine_and_hists(train, scalers["full"], journal)
        journal.close()
        print(f"\nJournal écrit : {JOURNAL_PATH} ({journal.seq} événements, "
              f"{JOURNAL_PATH.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
