"""Tests de la recherche évolutive : populations, opérateurs, déterminisme, amélioration, estimation de durée."""
import numpy as np
import pytest

from cmapss.evolution import (config_key, crossover, eta_seconds, initial_population, mutate, next_generation, plan_phases,
                              run_evolution)
from cmapss.search import DEFAULT, NAMED, SPACE, full


def test_population_initiale_nommes_d_abord_et_distincte():
    pop = initial_population(10, seed=1)
    assert len(pop) == 10 and [m["cid"] for m in pop[:len(NAMED)]] == list(NAMED)
    assert len({config_key(m["overrides"]) for m in pop}) == 10
    assert [m["cid"] for m in initial_population(4)] == list(NAMED)[:4]        # population plus petite que les nommés


def test_mutation_reste_dans_l_espace_et_taux_zero_ne_change_rien():
    rng = np.random.RandomState(0)
    base = full({})
    assert mutate(base, 0.0, rng) == base
    for _ in range(200):
        m = mutate(base, 0.5, rng)
        assert all(m[k] in SPACE[k] for k in SPACE)
    changed = sum(mutate(base, 1.0, np.random.RandomState(i)) != base for i in range(50))
    assert changed > 40                                                      # taux 1 : presque toujours différent


def test_croisement_prend_chaque_valeur_chez_un_parent():
    a, b = full({"dropout": 0.5, "lr": 3e-3}), full({"dropout": 0.0, "lr": 3e-4})
    child = crossover(a, b, np.random.RandomState(2))
    assert child["dropout"] in (0.5, 0.0) and child["lr"] in (3e-3, 3e-4)
    assert all(child[k] in (a[k], b[k]) for k in SPACE)


def _ranked(n=8):
    pop = initial_population(n, seed=0)
    return [{**m, "fitness": 25.0 + i} for i, m in enumerate(pop)]


def test_generation_suivante_garde_les_elites_et_ne_rejoue_rien():
    ranked = _ranked()
    seen = {config_key(m["overrides"]) for m in ranked}
    nxt = next_generation(ranked, seen, population=8, elite=2, mutation=0.3, seed=0, gen=1)
    assert len(nxt) == 8
    assert [m["cid"] for m in nxt[:2]] == [ranked[0]["cid"], ranked[1]["cid"]] and all(m["elite"] for m in nxt[:2])
    new = nxt[2:]
    keys = [config_key(m["overrides"]) for m in new]
    assert len(set(keys)) == len(keys) and not (set(keys) & {config_key(m["overrides"]) for m in ranked})   # nouveaux et distincts
    assert all(len(m["parents"]) == 2 and m["cid"].startswith("g01-") for m in new)


def test_generation_deterministe_selon_graine_et_generation():
    ranked = _ranked()
    def make(seed, gen):
        return next_generation(ranked, {config_key(m["overrides"]) for m in ranked}, 8, 2, 0.3, seed, gen)
    assert make(0, 1) == make(0, 1)
    assert make(0, 1) != make(0, 2) and make(0, 1) != make(1, 1)


def _jouet(members, gen):
    """Fitness jouet : distance à une cible connue (dropout 0,3, bruit 0,1, décroissance 1e-2, grande taille)."""
    target = full({"dropout": 0.3, "input_noise": 0.1, "weight_decay": 1e-2, "size": "grand"})
    out = {}
    for m in members:
        f = full(m["overrides"])
        out[m["cid"]] = 22.0 + sum(f[k] != target[k] for k in SPACE) * 0.8
    return out


def test_l_evolution_ameliore_le_meilleur_et_ne_le_degrade_jamais():
    improved = 0
    for seed in range(8):
        events = []
        hist = run_evolution(_jouet, generations=8, population=10, elite=2, mutation=0.15, seed=seed,
                             on_event=lambda t, **k: events.append((t, k)))
        best = [h["best_so_far"] for h in hist]
        assert all(b2 <= b1 for b1, b2 in zip(best, best[1:]))              # élitisme : jamais pire
        improved += best[-1] < best[0]
        assert hist[0]["delta"] is None and all(h["delta"] is not None for h in hist[1:])
        assert [t for t, _ in events].count("generation_start") == 8 == [t for t, _ in events].count("generation_end")
        keys = [config_key(m["overrides"]) for h in hist for m in h["ranking"] if not m["elite"]]
        assert len(keys) == len(set(keys))                                  # aucun réglage essayé deux fois
    assert improved >= 6                                                     # sur le problème jouet, l'évolution trouve mieux que la génération 0


def test_l_evolution_est_reproductible():
    a = run_evolution(_jouet, 5, 8, seed=3)
    b = run_evolution(_jouet, 5, 8, seed=3)
    assert [h["best"]["cid"] for h in a] == [h["best"]["cid"] for h in b]
    assert [h["mean"] for h in a] == [h["mean"] for h in b]


def test_plan_phases_totaux():
    ph = plan_phases(generations=8, population=8, elite=2, n_screen=2, top=4, n_rest=3)
    assert ph[0] == ("gén 0", 16) and ph[1] == ("gén 1", 12) and len(ph) == 8 + 3
    assert [n for n, _ in ph[-3:]] == ["confirmation", "validation croisée", "final"]
    assert sum(r for _, r in ph) == 16 + 7 * 12 + 15 + 20 + 5


def test_estimation_de_duree():
    assert eta_seconds(0, 100, 0, prior_per_run=60) == pytest.approx(6000)           # début : a priori seul
    assert eta_seconds(1000, 100, 50000, prior_per_run=60, prior_weight=4) == pytest.approx(100 * (240 + 50000) / 1004)
    assert eta_seconds(500, 0, 9999, 60) == 0                                         # rien à faire : zéro
