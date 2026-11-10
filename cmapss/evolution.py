"""Recherche évolutive d'hyperparamètres : populations, sélection, croisement, mutation, générations.

Une **génération** est une population de réglages (voir `cmapss.search`). Les meilleurs (les « élites ») passent tels quels à la
génération suivante ; le reste est fabriqué en croisant deux bons parents (le meilleur de deux tirages dans la moitié haute du
classement) puis en mutant quelques paramètres. Un réglage déjà essayé n'est jamais rejoué.

Le module ne dépend pas de PyTorch : l'entraînement est fourni par une fonction `evaluate(membres, generation)` (le script en
fait un vrai entraînement ; les tests, une fonction jouet). Le tirage ne dépend que de la graine et de la génération : relancer
après une interruption redonne exactement les mêmes populations.
"""
import numpy as np

from .search import NAMED, SPACE, describe, full, minimal, sample_configs


def config_key(overrides):
    """Forme canonique d'un réglage (pour savoir s'il a déjà été essayé)."""
    return tuple(sorted(minimal(overrides).items()))


def initial_population(population, seed=0):
    """Génération 0 : les réglages nommés d'abord (« aucune » est la référence), puis des tirages aléatoires distincts."""
    cfgs = sample_configs(max(0, population - len(NAMED)), seed)[:population]
    return [{"cid": cid, "overrides": dict(o), "label": describe(o), "parents": [], "elite": False} for cid, o in cfgs]


def crossover(a, b, rng):
    """Croisement uniforme : pour chaque paramètre, la valeur de l'un ou l'autre parent."""
    fa, fb = full(a), full(b)
    return {k: (fa[k] if rng.rand() < 0.5 else fb[k]) for k in SPACE}


def mutate(cfg, rate, rng):
    """Chaque paramètre change avec la probabilité `rate` : le plus souvent vers une valeur voisine, parfois au hasard."""
    out = dict(cfg)
    for k, values in SPACE.items():
        if rng.rand() < rate:
            i = values.index(out[k])
            if rng.rand() < 0.7:
                j = min(len(values) - 1, max(0, i + (1 if rng.rand() < 0.5 else -1)))
            else:
                j = int(rng.randint(len(values)))
            out[k] = values[j]
    return out


def _tournament(pool, rng):
    i, j = rng.randint(len(pool), size=2)
    return pool[i] if pool[i]["fitness"] <= pool[j]["fitness"] else pool[j]


def next_generation(prev_ranked, seen, population, elite, mutation, seed, gen):
    """Génération `gen` à partir du classement de la précédente (`prev_ranked` : dicts avec cid, overrides, fitness, meilleur d'abord).

    `seen` (ensemble de `config_key`) est mis à jour. Les élites gardent leur identifiant : leurs entraînements existent déjà.
    """
    members = [{"cid": e["cid"], "overrides": dict(e["overrides"]), "label": describe(e["overrides"]), "parents": [], "elite": True}
               for e in prev_ranked[:elite]]
    pool = prev_ranked[:max(2, (len(prev_ranked) + 1) // 2)]
    rng = np.random.RandomState(seed * 10007 + gen)
    n, tries = 0, 0
    while len(members) < population and tries < 5000:
        tries += 1
        a, b = _tournament(pool, rng), _tournament(pool, rng)
        child = mutate(crossover(a["overrides"], b["overrides"], rng), mutation, rng)
        key = config_key(child)
        if key in seen:
            continue
        seen.add(key)
        n += 1
        o = minimal(child)
        members.append({"cid": f"g{gen:02d}-{n:02d}", "overrides": o, "label": describe(o), "parents": [a["cid"], b["cid"]], "elite": False})
    return members


def run_evolution(evaluate, generations, population, elite=2, mutation=0.15, seed=0, on_event=None):
    """Lance `generations` générations. `evaluate(membres, génération) -> {cid: fitness}` (plus petit = meilleur).

    Événements envoyés à `on_event(type, **champs)` : `generation_start` (membres) et `generation_end` (classement, meilleur,
    moyenne, écart du meilleur avec le meilleur de toutes les générations précédentes). Retourne la liste des bilans.
    """
    emit = on_event or (lambda *a, **k: None)
    seen = set()
    members = initial_population(population, seed)
    for m in members:
        seen.add(config_key(m["overrides"]))
    history, ranked, best_so_far = [], None, None
    for gen in range(generations):
        if gen > 0:
            members = next_generation(ranked, seen, population, elite, mutation, seed, gen)
        emit("generation_start", gen=gen, members=members)
        fit = evaluate(members, gen)
        ranked = sorted(({**m, "fitness": float(fit[m["cid"]])} for m in members), key=lambda r: (r["fitness"], r["cid"]))
        best = ranked[0]
        finite = [r["fitness"] for r in ranked if np.isfinite(r["fitness"])]
        mean = float(np.mean(finite)) if finite else float("nan")
        delta = None if best_so_far is None else best["fitness"] - best_so_far        # < 0 : amélioration
        best_so_far = best["fitness"] if best_so_far is None else min(best_so_far, best["fitness"])
        record = {"gen": gen, "ranking": ranked, "best": best, "mean": mean, "delta": delta, "best_so_far": best_so_far}
        history.append(record)
        emit("generation_end", gen=gen, ranking=ranked, best=best, mean=mean, delta=delta, best_so_far=best_so_far)
    return history


# ---------------------------------------------------------------------------------------------------------------------
def plan_phases(generations, population, elite, n_screen, top, n_rest, n_folds=5, n_seeds=5):
    """Phases et nombre d'entraînements de chacune : [(nom, entraînements)]. `top` et la confirmation sont des maxima."""
    phases = [(f"gén {g}", (population if g == 0 else population - elite) * n_screen) for g in range(generations)]
    phases.append(("confirmation", (top + 1) * n_rest))
    phases.append(("validation croisée", n_folds * n_seeds - n_folds))          # la graine 0 est reprise de la recherche
    phases.append(("final", n_seeds))
    return phases


def eta_seconds(done, remaining, elapsed, prior_per_run, prior_weight=4):
    """Temps restant estimé : `remaining` × durée moyenne par entraînement (durée a priori au début, mesurée ensuite).

    La durée moyenne mêle l'a priori (poids `prior_weight` entraînements) et la mesure `elapsed / done` ; elle tient compte du
    parallélisme puisque `elapsed` est le temps réel écoulé.
    """
    per_run = (prior_per_run * prior_weight + elapsed) / (prior_weight + done)
    return remaining * per_run
