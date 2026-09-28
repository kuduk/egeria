"""Temperature scaling per tipo di domanda, fittato minimizzando la NLL.

La NLL è convessa nella temperatura inversa beta = 1/T, quindi una ricerca
della sezione aurea su beta trova l'ottimo globale. Dividere i logit per T non
cambia l'argmax: l'accuratezza resta identica, cambia solo la confidenza.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .confidence import log_softmax

BETA_RANGE = (0.05, 20.0)


def mean_nll(logits: list[np.ndarray], targets: list[np.ndarray], beta: float) -> float:
    """Cross-entropy media verso target soft (o one-hot)."""
    total = 0.0
    for z, t in zip(logits, targets):
        total -= float((np.asarray(t) * log_softmax(beta * np.asarray(z))).sum())
    return total / len(logits)


def fit_temperature(logits: list[np.ndarray], targets: list[np.ndarray], tolerance: float = 1e-5) -> float:
    if not logits:
        raise ValueError("servono esempi per fittare la temperatura")
    lo, hi = BETA_RANGE
    ratio = (math.sqrt(5) - 1) / 2
    a, b = lo + (1 - ratio) * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = mean_nll(logits, targets, a), mean_nll(logits, targets, b)
    while hi - lo > tolerance:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = lo + (1 - ratio) * (hi - lo)
            fa = mean_nll(logits, targets, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = mean_nll(logits, targets, b)
    return 1.0 / ((lo + hi) / 2)


def fit_by_type(records: list[dict]) -> dict[str, float]:
    """Una temperatura per tipo di domanda. Ogni record ha type, option_logits e gold."""
    temperatures = {}
    for qtype in ("noul", "choice", "score"):
        subset = [r for r in records if r["type"] == qtype]
        if subset:
            temperatures[qtype] = fit_temperature(
                [np.asarray(r["option_logits"]) for r in subset],
                [np.asarray(r["gold"]) for r in subset],
            )
    return temperatures


def load_temperatures(path: str | Path | None) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Legge il file delle temperature: (finali per tipo, per tipo e layer di uscita)."""
    if path is None:
        return {}, {}
    data = json.loads(Path(path).read_text())
    exits = {
        key: [int(v) for v in values] if key == "layers" else [float(t) for t in values]
        for key, values in data.get("exits", {}).items()
    }
    return {key: float(value) for key, value in data.items() if key != "exits"}, exits
