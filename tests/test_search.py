"""Tests de la recherche d'hyperparamètres : tirage reproductible, clés, classement pli par pli."""
import pytest

from cmapss.search import DEFAULT, NAMED, SPACE, by_fold, describe, full, minimal, parse_key, ranking, run_key, sample_configs


def test_les_reglages_nommes_viennent_en_premier_et_aucune_est_le_defaut():
    cfgs = sample_configs(5)
    assert [c for c, _ in cfgs[:len(NAMED)]] == list(NAMED)
    assert minimal(NAMED["aucune"]) == {}


def test_tirage_reproductible_et_prefixe_stable():
    a, b = sample_configs(20, seed=3), sample_configs(20, seed=3)
    assert a == b
    assert sample_configs(10, seed=3) == a[:len(NAMED) + 10]          # augmenter N ne change pas les premiers tirages
    assert sample_configs(20, seed=4) != a


def test_reglages_tires_distincts_valides_et_dans_l_espace():
    cfgs = sample_configs(80)
    forms = [tuple(sorted(minimal(o).items())) for _, o in cfgs]
    assert len(set(forms)) == len(forms)                               # aucun doublon, y compris avec les réglages nommés
    for cid, o in cfgs[len(NAMED):]:
        assert cid.startswith("r")
        f = full(o)
        assert all(f[k] in SPACE[k] for k in SPACE)


def test_full_refuse_un_parametre_inconnu():
    with pytest.raises(ValueError):
        full({"dropuot": 0.3})
    assert full({"dropout": 0.3})["lr"] == DEFAULT["lr"]


def test_describe():
    assert describe({}) == "défaut"
    assert describe({"dropout": 0.3, "input_noise": 0.1}) == "dropout 0,3 · bruit 0,1"


def test_cles_aller_retour_et_anciens_formats_ignores():
    k = run_key("search", "17", "r012", 3, 1)
    assert parse_key(k) == {"stage": "search", "sensors": "17", "cid": "r012", "fold": 3, "seed": 1}
    assert parse_key(run_key("final", "17", "aucune", None, 0))["fold"] is None
    assert parse_key("smoke|s17|lam0.05|fold0|seed0") is None           # ancien format


def _runs(table):
    """table : {cid: [rmse par pli]} -> faux résultats d'entraînement."""
    runs = {}
    for cid, vals in table.items():
        for fold, v in enumerate(vals):
            runs[run_key("search", "17", cid, fold, 0)] = {"val": {"rmse": v, "score": 1e6, "regime_acc": 0.6}, "best_epoch": 5}
    return runs


def test_classement_trie_et_ecart_apparie_a_la_reference():
    runs = _runs({"aucune": [25, 26, 27], "bon": [23, 24, 25], "moyen": [24.5, 26, 27]})
    rows = ranking(runs, "search", "17", ["aucune", "bon", "moyen"], range(3))
    assert [r["cid"] for r in rows] == ["bon", "moyen", "aucune"]
    bon = rows[0]
    assert bon["rmse_mean"] == pytest.approx(24.0)
    assert bon["diff_mean"] == pytest.approx(-2.0) and bon["diff_std"] == pytest.approx(0.0)   # -2 à chaque pli
    assert "diff_mean" not in rows[2]                                                           # la référence n'a pas d'écart


def test_classement_ignore_les_reglages_incomplets():
    runs = _runs({"aucune": [25, 26, 27], "partiel": [20, 20]})       # « partiel » n'a pas son 3ᵉ pli
    rows = ranking(runs, "search", "17", ["aucune", "partiel"], range(3))
    assert [r["cid"] for r in rows] == ["aucune"]


def test_by_fold_filtre_etape_capteurs_et_graine():
    runs = _runs({"a": [1, 2]})
    runs[run_key("search", "14", "a", 0, 0)] = {"x": 1}
    runs[run_key("cv", "17", "a", 0, 0)] = {"x": 2}
    assert sorted(by_fold(runs, "search", "17", "a")) == [0, 1]
    assert by_fold(runs, "search", "17", "a", seeds=[3]) == {}
