"""Distribuzioni, temperatura e confidenza con le stesse formule dell'API Jev.

Le formule di `confidence` sono statistiche della distribuzione, non una stima
appresa della correttezza (vedi documentazione/01-stato-dell-arte.md, §3.2).
"""

from __future__ import annotations

import numpy as np


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    weights = np.exp(z)
    return weights / weights.sum()


def log_softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def choice_confidence(probabilities) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    n = len(p)
    if n == 1:
        return 1.0
    return float((n * p.max() - 1) / (n - 1))


def score_confidence(probabilities) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    n = len(p)
    levels = np.arange(n)
    mode = int(p.argmax())
    spread = np.abs(levels - (n - 1) / 2).mean()
    return float(max(0.0, 1.0 - (p * np.abs(levels - mode)).sum() / spread))


def decision_confidence(qtype: str, probabilities) -> float:
    """Confidenza corretta per il caso: 0 = distribuzione uniforme, 1 = certezza.

    È la `confidence` di Jev: per choice (n·p_max − 1)/(n − 1), per score la formula
    basata sulla distanza dalla moda. Per noul si usa la formula di choice con n = 2,
    cioè 2·p_max − 1. È la quantità confrontata con `min_confidence`.
    """
    if qtype in ("score", "number"):
        return score_confidence(probabilities)
    return choice_confidence(probabilities)


def answer(question, probabilities) -> dict:
    """Risposta nel formato Jev per una domanda, data la distribuzione sulle opzioni."""
    p = np.asarray(probabilities, dtype=np.float64)
    keys = question.keys
    rounded = {key: round(float(value), 6) for key, value in zip(keys, p)}
    if question.type == "noul":
        # `confidence` per noul è un'estensione Egeria: Jev non la restituisce.
        return {"type": "noul", "noul": rounded["true"], "confidence": round(choice_confidence(p), 6)}
    if question.type == "choice":
        return {
            "type": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": rounded,
            "confidence": round(choice_confidence(p), 6),
        }
    if question.type == "rank":
        order = np.argsort(-p, kind="stable")
        return {
            "type": "rank",
            "ranking": [keys[i] for i in order],
            "probabilities": rounded,
            "confidence": round(choice_confidence(p), 6),
        }
    if question.type == "number":
        return number_answer(question, p)
    return {
        "type": "score",
        "score": round(float((np.arange(len(p)) * p).sum()), 6),
        "legend": {option.key: option.description for option in question.options},
        "probabilities": rounded,
        "confidence": round(score_confidence(p), 6),
    }


def _bin_bounds(edges: list) -> list[tuple[float, float]]:
    """Estremi finiti di ogni intervallo: un intervallo aperto collassa sul suo estremo finito."""
    bounds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bounds.append((hi if lo is None else lo, lo if hi is None else hi))
    return bounds


def _quantile(p: np.ndarray, bounds: list[tuple[float, float]], q: float) -> float:
    """Quantile di una distribuzione uniforme a tratti sugli intervalli."""
    cumulative = 0.0
    for mass, (lo, hi) in zip(p, bounds):
        if cumulative + mass >= q and mass > 0:
            return float(lo + (hi - lo) * (q - cumulative) / mass)
        cumulative += mass
    return float(bounds[-1][1])


def number_answer(question, probabilities) -> dict:
    """Stima numerica da una distribuzione sugli intervalli: valore atteso e intervallo 10–90%."""
    p = np.asarray(probabilities, dtype=np.float64)
    bounds = _bin_bounds(question.params["edges"])
    midpoints = np.array([(lo + hi) / 2 for lo, hi in bounds])
    result = {
        "type": "number",
        "value": round(float((p * midpoints).sum()), 6),
        "interval": [round(_quantile(p, bounds, 0.1), 6), round(_quantile(p, bounds, 0.9), 6)],
        "range": question.keys[int(p.argmax())],
        "probabilities": {key: round(float(v), 6) for key, v in zip(question.keys, p)},
        "confidence": round(score_confidence(p), 6),
    }
    if question.params.get("unit"):
        result["unit"] = question.params["unit"]
    return result


def merge_candidates(candidates: list[tuple[str, float]], top_k: int) -> list[dict]:
    """Unisce i token che decodificano allo stesso testo (spazi e maiuscole a parte)."""
    merged: dict[str, list] = {}
    for text, prob in candidates:
        cleaned = text.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in merged:
            merged[key][1] += prob
        else:
            merged[key] = [cleaned, prob]
    ranked = sorted(merged.values(), key=lambda item: -item[1])[:top_k]
    return [{"text": text, "p": round(float(prob), 6)} for text, prob in ranked]


def open_answer(candidates: list[dict]) -> dict:
    top = candidates[0] if candidates else {"text": "", "p": 0.0}
    return {"type": "open", "answer": top["text"], "candidates": candidates, "confidence": top["p"]}
