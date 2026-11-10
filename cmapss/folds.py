"""Plis de validation croisée : découpage par moteur, jamais par ligne."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(df, n_splits=5, seed=42):
    """Attribue un pli (0 à n_splits-1) à chaque moteur.

    df : lignes d'entraînement avec `engine_id` et `subset`. Un moteur n'est que dans un seul pli.
    La stratification par jeu garantit que chaque pli contient des moteurs des 4 jeux.
    Retourne un DataFrame (engine_id, subset, unit, fold), une ligne par moteur.
    """
    engines = df.drop_duplicates("engine_id")[["engine_id", "subset", "unit"]].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    engines["fold"] = -1
    for k, (_, val_idx) in enumerate(
        splitter.split(engines, y=engines["subset"], groups=engines["engine_id"])
    ):
        engines.loc[val_idx, "fold"] = k
    return engines
