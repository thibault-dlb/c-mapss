"""Journal d'événements (JSON Lines) : une ligne par étape réellement exécutée.

Le journal est la source de vérité des visualisations : la page ne calcule rien, elle rejoue
exactement ces événements, dans l'ordre. `seq` est contigu (0, 1, 2, ...) pour que la page
détecte un journal tronqué ou modifié.
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _git_hash():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self._t0 = time.perf_counter()
        self.seq = 0
        self.log("journal", started_at=datetime.now().isoformat(timespec="seconds"), git=_git_hash())

    def log(self, type_, **fields):
        """Écrit un événement : seq, t (secondes depuis le début du calcul), type, puis les champs."""
        event = {"seq": self.seq, "t": round(time.perf_counter() - self._t0, 4), "type": type_, **fields}
        self._f.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._f.flush()
        self.seq += 1

    def close(self):
        self.log("end", n_events=self.seq + 1)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._f.closed:
            self.close()
