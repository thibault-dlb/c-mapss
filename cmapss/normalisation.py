"""Normalisation des capteurs par condition de vol (z-score) et des réglages (min-max)."""
import numpy as np
import pandas as pd

from .data import CONDITIONS, SENSORS_KEPT, SETTINGS, add_condition

BLOCK = 2000  # lignes par bloc pour le calcul incrémental de μ et σ


class ConditionScaler:
    """z-score des capteurs retenus, par condition de vol ; min-max des 3 réglages.

    Les statistiques sont calculées sur les données passées à `fit` (tous jeux confondus : au test,
    on ne connaît pas le jeu). Appeler `fit` uniquement sur les moteurs d'entraînement du pli.

    μ et σ sont calculés **par blocs** de lignes, dans l'ordre des données, avec la fusion de
    statistiques partielles de Chan (équivalent à Welford par blocs). C'est le code réellement
    utilisé partout ; `on_block` permet de journaliser chaque bloc sans changer le calcul.
    """

    def __init__(self, sensors=None):
        self.sensors = list(sensors) if sensors is not None else list(SENSORS_KEPT)
        self.mean = {}      # condition -> moyenne de chaque capteur
        self.std = {}       # condition -> écart-type de chaque capteur
        self.n_rows = {}    # condition -> nombre de lignes utilisées
        self.flat = {}      # condition -> capteurs constants dans cette condition (écart-type nul)
        self.op_min = None  # minimum des 3 réglages
        self.op_max = None  # maximum des 3 réglages

    @staticmethod
    def _cond(df):
        return df["cond"] if "cond" in df.columns else add_condition(df)

    def fit(self, df, block=BLOCK, on_block=None):
        """Ajuste le scaler.

        on_block(k, n_blocks, rows_seen, block_df, snapshot) est appelé après chaque bloc ;
        snapshot = {condition: (n, moyenne, écart-type)} pour les conditions déjà vues.
        """
        cond = self._cond(df)
        unknown = sorted(set(cond) - set(CONDITIONS))
        if unknown:
            raise ValueError(f"Conditions de vol inconnues à l'ajustement : {unknown}")

        k_sens = len(self.sensors)
        n = {c: 0 for c in CONDITIONS}
        mean = {c: np.zeros(k_sens) for c in CONDITIONS}
        m2 = {c: np.zeros(k_sens) for c in CONDITIONS}

        values = df[self.sensors].to_numpy(dtype=float)
        cond_arr = cond.to_numpy()
        n_blocks = -(-len(df) // block)
        for b in range(n_blocks):
            sl = slice(b * block, (b + 1) * block)
            vb, cb = values[sl], cond_arr[sl]
            for c in CONDITIONS:
                sub = vb[cb == c]
                nb = len(sub)
                if nb == 0:
                    continue
                mean_b = sub.mean(axis=0)
                m2_b = ((sub - mean_b) ** 2).sum(axis=0)
                tot = n[c] + nb
                delta = mean_b - mean[c]
                mean[c] = mean[c] + delta * nb / tot
                m2[c] = m2[c] + m2_b + delta**2 * n[c] * nb / tot
                n[c] = tot
            if on_block is not None:
                snap = {c: (n[c], mean[c].copy(), np.sqrt(m2[c] / n[c])) for c in CONDITIONS if n[c] > 0}
                on_block(b, n_blocks, min((b + 1) * block, len(df)), df.iloc[sl], snap)

        self.mean, self.std, self.n_rows, self.flat = {}, {}, {}, {}
        for c in CONDITIONS:
            if n[c] == 0:
                continue
            std = np.sqrt(m2[c] / n[c])
            flat = std < 1e-8
            if flat.any():
                # Capteur constant dans cette condition : écart-type remplacé par 1, la valeur normalisée vaut 0
                self.flat[c] = [s for s, f in zip(self.sensors, flat) if f]
                std = np.where(flat, 1.0, std)
            self.mean[c] = mean[c]
            self.std[c] = std
            self.n_rows[c] = n[c]
        self.op_min = df[SETTINGS].min().to_numpy()
        self.op_max = df[SETTINGS].max().to_numpy()
        return self

    def transform_steps(self, df):
        """Les deux étapes du z-score : (x − μ) puis (x − μ) / σ. Retourne deux tableaux (lignes × capteurs)."""
        if not self.mean:
            raise RuntimeError("ConditionScaler non ajusté : appeler fit() d'abord")
        cond = self._cond(df)
        unknown = sorted(set(cond) - set(self.mean))
        if unknown:
            raise ValueError(f"Conditions de vol inconnues du scaler : {unknown}")
        values = df[self.sensors].to_numpy(dtype=float)
        uniq = cond.unique()
        mean = np.vstack([self.mean[c] for c in uniq])
        std = np.vstack([self.std[c] for c in uniq])
        idx = pd.Index(uniq).get_indexer(cond)  # ligne -> indice de condition dans mean/std
        centered = values - mean[idx]
        return centered, centered / std[idx]

    def transform(self, df):
        out = df.copy()
        _, scaled = self.transform_steps(df)
        out[self.sensors] = scaled
        span = np.where(self.op_max - self.op_min == 0, 1.0, self.op_max - self.op_min)
        out[SETTINGS] = (df[SETTINGS].to_numpy(dtype=float) - self.op_min) / span
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def to_dict(self):
        return {
            "sensors": self.sensors,
            "conditions": {
                c: {
                    "mean": self.mean[c].tolist(),
                    "std": self.std[c].tolist(),
                    "n_rows": self.n_rows[c],
                    "flat": self.flat.get(c, []),
                }
                for c in self.mean
            },
            "op_min": self.op_min.tolist(),
            "op_max": self.op_max.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(sensors=d["sensors"])
        for c, v in d["conditions"].items():
            s.mean[c] = np.array(v["mean"])
            s.std[c] = np.array(v["std"])
            s.n_rows[c] = v["n_rows"]
            if v.get("flat"):
                s.flat[c] = v["flat"]
        s.op_min = np.array(d["op_min"])
        s.op_max = np.array(d["op_max"])
        return s
