"""Fenêtres glissantes : fabrication sur les données réelles, contrôles chiffrés, journal pour la vidéo.

Usage :
    .\\.venv\\Scripts\\python.exe scripts\\fenetres.py            # fabrique toutes les fenêtres et affiche les contrôles
    .\\.venv\\Scripts\\python.exe scripts\\fenetres.py --journal  # + journal d'événements pour la page viz/fenetres/

Les fenêtres sont fabriquées avec le scaler « full » (tout l'entraînement). À l'entraînement réel, chaque pli utilisera son
propre scaler (voir docs/normalisation.md). Avec --journal, les fenêtres d'un moteur de démonstration sont créées par le vrai
code (`make_windows`) en journalisant chacune : la page rejoue ce journal sans rien recalculer.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from cmapss.data import SUBSETS, load_all
from cmapss.folds import make_folds
from cmapss.journal import Journal
from cmapss.normalisation import ConditionScaler
from cmapss.windows import FEATURES, REGIME_OF_SUBSET, WINDOW, make_windows, n_windows

JOURNAL_PATH = ROOT / "viz" / "fenetres" / "journal.jsonl"
DEMO_ENGINE = "FD002_1"   # moteur d'entraînement montré dans la vidéo
DEMO_SENSOR = "s11"       # capteur dont la tendance de dégradation est la plus nette (voir docs/exploration.md)
VAL_FOLD = 0              # pli montré comme « validation »


def r(a, n=3):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def subset_of(ids):
    return np.array([e.split("_")[0] for e in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", action="store_true", help=f"écrit le journal ({JOURNAL_PATH.relative_to(ROOT)})")
    args = ap.parse_args()

    train, test = load_all("train"), load_all("test")
    scaler = ConditionScaler().fit(train)
    tr, te = scaler.transform(train), scaler.transform(test)

    w_train = make_windows(tr, "train")
    w_test = make_windows(te, "test")
    lengths = train.groupby("engine_id", sort=False).size()
    assert len(w_train) == sum(n_windows(n) for n in lengths) == len(train) - len(lengths) * (WINDOW - 1)
    assert len(w_test) == te["engine_id"].nunique()
    assert not np.isnan(w_train.X).any() and not np.isnan(w_test.X).any()

    print(f"Fenêtre : {WINDOW} cycles × {len(FEATURES)} valeurs ({w_train.X.shape[2]} = 17 capteurs + 3 réglages)")
    print(f"\n{'jeu':<7}{'moteurs ent.':>13}{'fenêtres ent.':>15}{'moteurs test':>14}{'fenêtres test':>15}   régime")
    counts = {}
    for fd in SUBSETS:
        mt, ms = subset_of(w_train.engine_id) == fd, subset_of(w_test.engine_id) == fd
        counts[fd] = {"train_engines": int((subset_of(lengths.index) == fd).sum()), "train_windows": int(mt.sum()),
                      "test_engines": int(ms.sum()), "test_windows": int(ms.sum())}
        c = counts[fd]
        print(f"{fd:<7}{c['train_engines']:>13}{c['train_windows']:>15}{c['test_engines']:>14}{c['test_windows']:>15}"
              f"   {REGIME_OF_SUBSET[fd]}")
    print(f"{'total':<7}{len(lengths):>13}{len(w_train):>15}{len(w_test):>14}{len(w_test):>15}")
    short = (w_test.n_padded > 0)
    print(f"\nMoteurs de test complétés par répétition du premier cycle : {int(short.sum())} "
          f"(jusqu'à {int(w_test.n_padded.max())} cycles ajoutés)")
    print(f"Mémoire des fenêtres d'entraînement : {w_train.X.nbytes / 1e6:.0f} Mo (float32)")
    print(f"RUL au dernier cycle : entraînement {w_train.rul.min():.0f}–{w_train.rul.max():.0f}, "
          f"test {w_test.rul.min():.0f}–{w_test.rul.max():.0f}")

    if args.journal:
        write_journal(train, test, tr, te, w_train, w_test, lengths, counts)


def write_journal(train, test, tr, te, w_train, w_test, lengths, counts):
    j = Journal(JOURNAL_PATH)
    k_sensor = FEATURES.index(DEMO_SENSOR)
    j.log("meta", window=WINDOW, features=FEATURES, n_features=len(FEATURES), demo_engine=DEMO_ENGINE,
          demo_sensor=DEMO_SENSOR, regime_of_subset=REGIME_OF_SUBSET, subsets=SUBSETS, script="scripts/fenetres.py")

    # 1. Le moteur d'entraînement de démonstration : sa série normalisée et sa RUL réelle
    eng = tr[tr["engine_id"] == DEMO_ENGINE].sort_values("cycle")
    j.log("engine", engine=DEMO_ENGINE, subset=eng["subset"].iloc[0], n_cycles=int(len(eng)),
          cycle=eng["cycle"].tolist(), x=r(eng[DEMO_SENSOR]), rul=r(eng["rul"], 0))

    # 2. Ses fenêtres, créées une à une par le vrai code (chaque appel de on_window = un événement)
    made = []

    def on_window(engine_id, start, end, rul, n_padded, win):
        made.append(win)
        j.log("window", k=len(made) - 1, engine=engine_id, start=start, end=end, rul=rul, padded=n_padded,
              x=r(win[:, k_sensor]))

    wins = make_windows(eng, "train", on_window=on_window)
    assert len(wins) == n_windows(len(eng)) == len(made)

    # 3. Contrôle : fenêtres voisines = 29 cycles partagés ; chaque fenêtre est bien la tranche de la série
    shared_ok = all(np.array_equal(made[i + 1][:-1], made[i][1:]) for i in range(len(made) - 1))
    slice_ok = all(np.array_equal(made[i][:, k_sensor], eng[DEMO_SENSOR].to_numpy(np.float32)[i:i + WINDOW])
                   for i in range(len(made)))
    assert shared_ok and slice_ok
    j.log("check", name="overlap", shared_cycles=WINDOW - 1, pairs=len(made) - 1, ok=bool(shared_ok))
    j.log("check", name="window_is_slice_of_series", windows=len(made), ok=bool(slice_ok))

    # 4. Le moteur de test le plus court : sa fenêtre est complétée par la répétition du premier cycle
    sizes = te.groupby("engine_id", sort=False).size()
    short_id = sizes.idxmin()
    sh = te[te["engine_id"] == short_id].sort_values("cycle")
    pad_events = []
    make_windows(sh, "test", on_window=lambda e, a, b, rul, npad, win: pad_events.append((npad, win)))
    pad_n, pad_win = pad_events[0]
    j.log("short_engine", engine=short_id, subset=sh["subset"].iloc[0], n_cycles=int(len(sh)), n_padded=int(pad_n),
          x_engine=r(sh[DEMO_SENSOR]), x_window=r(pad_win[:, k_sensor]), rul=float(sh["rul"].iloc[-1]))
    assert np.all(pad_win[:pad_n] == pad_win[pad_n]) and pad_n == WINDOW - len(sh)

    # 5. Un moteur de test « normal » : une seule fenêtre, les 30 derniers cycles, avec la vraie RUL du fichier RUL_FD00x.txt
    tt = te[te["engine_id"] == DEMO_ENGINE].sort_values("cycle")
    test_events = []
    make_windows(tt, "test", on_window=lambda e, a, b, rul, npad, win: test_events.append((a, b, rul, npad, win)))
    a, b, rul, npad, win = test_events[0]
    j.log("test_window", engine=DEMO_ENGINE, subset=tt["subset"].iloc[0], n_cycles=int(len(tt)), start=a, end=b,
          rul=rul, padded=int(npad), cycle=tt["cycle"].tolist(), x_engine=r(tt[DEMO_SENSOR]), x_window=r(win[:, k_sensor]))

    # 6. Les effectifs réels
    j.log("counts", per_subset=counts, train_engines=int(len(lengths)), train_windows=int(len(w_train)),
          test_engines=int(len(w_test)), test_windows=int(len(w_test)),
          test_padded=int((w_test.n_padded > 0).sum()))

    # 7. Découpage par moteur (pli de validation 0) : un moteur n'est que d'un côté
    folds = make_folds(train)
    val_ids = set(folds.loc[folds["fold"] == VAL_FOLD, "engine_id"])
    train_ids = set(folds["engine_id"]) - val_ids
    assert not (train_ids & val_ids)
    in_val = np.isin(w_train.engine_id, list(val_ids))
    # 6 moteurs réels de chaque côté (1 de FD001, 2 de FD002, 1 de FD003, 2 de FD004) : les premiers dans l'ordre des plis
    per_subset = {"FD001": 1, "FD002": 2, "FD003": 1, "FD004": 2}
    examples = []
    for side, ids in (("entrainement", train_ids), ("validation", val_ids)):
        for fd, k in per_subset.items():
            for eid in folds.loc[(folds["subset"] == fd) & folds["engine_id"].isin(ids), "engine_id"].head(k):
                examples.append({"engine": eid, "subset": fd, "side": side, "windows": n_windows(int(lengths[eid]))})
    j.log("split", fold=VAL_FOLD, n_folds=5, train_engines=len(train_ids), val_engines=len(val_ids),
          train_windows=int((~in_val).sum()), val_windows=int(in_val.sum()), examples=examples)
    j.log("check", name="engines_disjoint", ok=True, train_engines=len(train_ids), val_engines=len(val_ids))
    j.close()
    print(f"\nJournal écrit : {JOURNAL_PATH} ({j.seq} événements, {JOURNAL_PATH.stat().st_size / 1024:.0f} Ko)")
    print(f"  moteur de démo {DEMO_ENGINE} : {len(eng)} cycles -> {len(made)} fenêtres ; "
          f"moteur de test le plus court {short_id} : {len(sh)} cycles ({pad_n} ajoutés)")


if __name__ == "__main__":
    main()
