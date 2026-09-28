"""Metriche per decisioni tipizzate con gold soft (distribuzioni) o hard.

Ogni record ha:
- `type`: noul | choice | score
- `probs`: distribuzione predetta (nell'ordine originale delle opzioni)
- `gold`: distribuzione gold (soft o one-hot)
- `label`: indice dell'opzione gold (argmax dichiarato dal dataset)

Definizioni:
- accuracy: argmax predetto == label
- nll: -log p[label]; cross_entropy: -sum gold * log p (target soft)
- kl: KL(gold || pred); tv: 0.5 * sum |pred - gold|
- brier: sum_k (p_k - onehot_k)^2, in [0, 2]; brier_mean: la stessa divisa per il numero di opzioni
- ece: confidenza top-1 (max p) contro correttezza, 10 bin di ampiezza uguale
- score_mae: |E_pred[livello] - E_gold[livello]| per le domande score
- coverage_at_5: frazione massima di decisioni, ordinate per confidenza, con errore <= 5%
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

EPS = 1e-12


def _record_stats(record: dict) -> dict:
    p = np.clip(np.asarray(record["probs"], dtype=np.float64), EPS, 1.0)
    g = np.asarray(record["gold"], dtype=np.float64)
    label = int(record["label"])
    onehot = np.zeros_like(p)
    onehot[label] = 1.0
    stats = {
        "correct": float(int(np.argmax(p)) == label),
        "confidence": float(p.max()),
        "nll": float(-np.log(p[label])),
        "cross_entropy": float(-(g * np.log(p)).sum()),
        "kl": float((g * (np.log(np.clip(g, EPS, 1.0)) - np.log(p))).sum()),
        "tv": float(0.5 * np.abs(p - g).sum()),
        "brier": float(((p - onehot) ** 2).sum()),
        "brier_mean": float(((p - onehot) ** 2).mean()),
    }
    if record["type"] == "score":
        levels = np.arange(len(p))
        stats["score_mae"] = float(abs((levels * p).sum() - (levels * g).sum()))
        stats["within_1"] = float(abs(int(np.argmax(p)) - label) <= 1)
    return stats


def ece(confidences, correct, bins: int = 10) -> float:
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidences > low) & (confidences <= high)
        if inside.any():
            total += inside.mean() * abs(confidences[inside].mean() - correct[inside].mean())
    return float(total)


def coverage_at_error(confidences, correct, max_error: float = 0.05) -> float:
    """Copertura massima rispondendo solo alle decisioni più confidenti con errore <= max_error."""
    order = np.argsort(-np.asarray(confidences), kind="stable")
    errors = np.cumsum(1.0 - np.asarray(correct)[order])
    counts = np.arange(1, len(order) + 1)
    ok = np.nonzero(errors / counts <= max_error)[0]
    return float((ok.max() + 1) / len(order)) if len(ok) else 0.0


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    stats = [_record_stats(r) for r in records]
    result = {"n": len(records)}
    for key in ("correct", "nll", "cross_entropy", "kl", "tv", "brier", "brier_mean"):
        result["accuracy" if key == "correct" else key] = float(np.mean([s[key] for s in stats]))
    confidences = [s["confidence"] for s in stats]
    correct = [s["correct"] for s in stats]
    result["ece"] = ece(confidences, correct)
    result["mean_confidence"] = float(np.mean(confidences))
    result["coverage_at_5"] = coverage_at_error(confidences, correct, 0.05)
    score_stats = [s for s in stats if "score_mae" in s]
    if score_stats:
        result["score_mae"] = float(np.mean([s["score_mae"] for s in score_stats]))
        result["within_1"] = float(np.mean([s["within_1"] for s in score_stats]))
    return result


def summarize_by(records: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return {name: summarize(items) for name, items in sorted(groups.items())}


def paired_bootstrap(records_a: list[dict], records_b: list[dict], resamples: int = 2000, seed: int = 0) -> dict:
    """Differenze B - A su accuracy, NLL e Brier, appaiate per decisione.

    Il ricampionamento è per caso (cluster bootstrap): le domande dello stesso
    stato non sono indipendenti. I record devono avere `probs`, `label`, `case`
    e `question`, e riferirsi alle stesse decisioni.
    """
    key = lambda r: (r["case"], r["question"])
    b_by_key = {key(r): r for r in records_b}
    pairs = [(a, b_by_key[key(a)]) for a in records_a if key(a) in b_by_key]
    if not pairs:
        raise ValueError("nessuna decisione in comune fra i due insiemi di predizioni")
    names = ("accuracy", "nll", "brier")
    diffs = {name: [] for name in names}
    cases = []
    for a, b in pairs:
        sa, sb = _record_stats(a), _record_stats(b)
        diffs["accuracy"].append(sb["correct"] - sa["correct"])
        diffs["nll"].append(sb["nll"] - sa["nll"])
        diffs["brier"].append(sb["brier"] - sa["brier"])
        cases.append(a["case"])
    case_ids = sorted(set(cases))
    index = {case: i for i, case in enumerate(case_ids)}
    groups = np.array([index[c] for c in cases])
    rng = np.random.default_rng(seed)
    result = {"n": len(pairs), "cases": len(case_ids)}
    for name in names:
        values = np.asarray(diffs[name])
        sums = np.bincount(groups, weights=values, minlength=len(case_ids))
        counts = np.bincount(groups, minlength=len(case_ids))
        boot = []
        for _ in range(resamples):
            pick = rng.integers(0, len(case_ids), len(case_ids))
            boot.append(sums[pick].sum() / counts[pick].sum())
        low, high = np.percentile(boot, [2.5, 97.5])
        result[name] = {"diff": float(values.mean()), "ci95": [float(low), float(high)]}
    return result


def flip_rate(records: list[dict]) -> float | None:
    """Frazione di decisioni in cui le permutazioni delle opzioni non concordano sull'argmax."""
    repeated = [r["order_argmax"] for r in records if len(r.get("order_argmax", [])) > 1]
    if not repeated:
        return None
    return float(np.mean([len(set(argmaxes)) > 1 for argmaxes in repeated]))
