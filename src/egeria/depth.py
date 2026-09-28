"""Profondità dinamica per asserzione: simulazione dell'early exit sui readout intermedi.

Per ogni domanda si hanno i logit delle opzioni a vari layer di uscita (confini
di blocco). La politica di uscita si ferma al primo layer in cui la confidenza
calibrata raggiunge la soglia; altrimenti arriva all'ultimo layer.

Il calcolo è misurato in frazione di layer eseguiti: uscire dopo il layer l
costa (l + 1) / L del forward completo. In un solo forward pass non c'è KV-cache
da propagare, quindi il risparmio è reale per ogni domanda che esce prima.

Le temperature vanno fittate per (tipo, layer) su un set di calibrazione diverso
da quello di valutazione: i layer bassi hanno logit di scala molto diversa.
"""

from __future__ import annotations

import numpy as np

from .calibration import fit_temperature
from .confidence import decision_confidence, softmax
from .metrics import summarize


def fit_exit_temperatures(records: list[dict]) -> dict[str, list[float]]:
    """Una temperatura per tipo e per layer di uscita."""
    result = {}
    for qtype in ("noul", "choice", "score"):
        subset = [r for r in records if r["type"] == qtype]
        if not subset:
            continue
        exits = len(subset[0]["exit_logits"])
        result[qtype] = [
            fit_temperature([np.asarray(r["exit_logits"][e]) for r in subset], [np.asarray(r["gold"]) for r in subset])
            for e in range(exits)
        ]
    return result


def exit_probs(record: dict, temperatures: dict[str, list[float]]) -> list[np.ndarray]:
    temps = temperatures.get(record["type"])
    return [
        softmax(logits, temps[e] if temps else 1.0) for e, logits in enumerate(record["exit_logits"])
    ]


def per_layer_summary(records: list[dict], temperatures: dict[str, list[float]]) -> list[dict]:
    """Qualità del readout a ogni layer di uscita, come se si uscisse sempre lì."""
    layers = records[0]["exit_layers"]
    rows = []
    for e, layer in enumerate(layers):
        probs = [exit_probs(r, temperatures)[e] for r in records]
        summary = summarize([{**r, "probs": p} for r, p in zip(records, probs)])
        rows.append({"layer": layer, **{k: summary[k] for k in ("accuracy", "nll", "brier", "ece", "coverage_at_5")}})
    return rows


def simulate(records: list[dict], temperatures: dict[str, list[float]], threshold: float, num_layers: int) -> dict:
    """Early exit alla prima uscita con confidenza corretta per il caso >= soglia (come `min_confidence`)."""
    layers = records[0]["exit_layers"]
    chosen, cost, depth, agree = [], [], [], []
    for record in records:
        probs = exit_probs(record, temperatures)
        index = len(layers) - 1
        for e in range(len(layers)):
            if decision_confidence(record["type"], probs[e]) >= threshold:
                index = e
                break
        chosen.append({**record, "probs": probs[index]})
        cost.append((layers[index] + 1) / num_layers)
        depth.append(layers[index])
        # Consistenza con il modello completo (il criterio di CATs).
        agree.append(int(np.argmax(probs[index])) == int(np.argmax(probs[-1])))
    summary = summarize(chosen)
    return {
        "threshold": threshold,
        "compute": float(np.mean(cost)),
        "agree_full": float(np.mean(agree)),
        "accuracy": summary["accuracy"],
        "nll": summary["nll"],
        "brier": summary["brier"],
        "ece": summary["ece"],
        "exit_histogram": {layer: int(np.sum(np.asarray(depth) == layer)) for layer in layers},
    }


def oracle(records: list[dict], num_layers: int) -> dict:
    """Risparmio massimo con un criterio d'uscita perfetto.

    Per ogni decisione si esce al primo layer da cui l'argmax resta uguale a
    quello finale fino all'ultimo layer: la decisione non cambia, il calcolo sì.
    """
    layers = records[0]["exit_layers"]
    costs, stable = [], []
    for record in records:
        argmaxes = [int(np.argmax(logits)) for logits in record["exit_logits"]]
        k = len(argmaxes) - 1
        while k > 0 and argmaxes[k - 1] == argmaxes[-1]:
            k -= 1
        costs.append((layers[k] + 1) / num_layers)
        stable.append(layers[k])
    return {
        "compute": float(np.mean(costs)),
        "stable_from": {layer: int(np.sum(np.asarray(stable) == layer)) for layer in layers},
    }


def sweep(records: list[dict], temperatures: dict[str, list[float]], num_layers: int, thresholds=None) -> list[dict]:
    # Soglie sulla confidenza corretta per il caso (0 = a caso, 1 = certezza); 1.01 = mai uscire prima.
    thresholds = thresholds or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    return [simulate(records, temperatures, t, num_layers) for t in thresholds]
