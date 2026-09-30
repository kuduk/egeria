"""Correzioni a basso costo per la posizione delle opzioni e per la tendenza al "Sì".

    .venv/bin/python scripts/esperimento_debiasing.py Qwen/Qwen3.5-2B-Base            # passata del modello + analisi
    .venv/bin/python scripts/esperimento_debiasing.py Qwen/Qwen3.5-2B-Base --solo-analisi

Esperimento sul test di typed-decisions (400 casi, 2000 decisioni, con le etichette).

Fase 1, con il modello: per ogni domanda si calcolano i logit delle lettere per TUTTE le rotazioni
cicliche delle opzioni (più l'ordine inverso per le score), la stessa domanda su uno stato vuoto
("N/A", per la calibrazione a input vuoto) e, per le domande Sì/No, l'affermazione negata: a mano
(examples/controlli/negazioni.json) e con una negazione automatica. Tutto in runs/debiasing/.

Fase 2, senza modello: si simulano le varianti sugli stessi numeri.
- 1, 2 o tutte le permutazioni (come lo scorer: rotazioni cicliche; per le score diretto e inverso);
- preferenza a priori per la lettera (PriDe, Zheng et al. 2024): stimata su 20 casi di taratura con
  tutte le rotazioni, poi sottratta dai logit delle lettere; costo in inferenza nullo;
- calibrazione a input vuoto (Zhao et al. 2021): si divide per la previsione sullo stato "N/A";
  costo: una passata per tipo di domanda, riusabile;
- calibrazione di lotto (Zhou et al. 2024): si divide per la media delle previsioni sul lotto;
  costo nullo, ma serve un lotto dello stesso carico di lavoro;
- simmetrizzazione della polarità, solo per le Sì/No: logit P*(A) = (logit P(A) − logit P(non A)) / 2;
  costo doppio sulle Sì/No (riferimento).
I 20 casi di taratura sono esclusi dalla valutazione di tutte le varianti.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egeria.confidence import log_softmax
from egeria.datasets import gold_vector, load_typed_decisions
from egeria.metrics import summarize, summarize_by
from egeria.prompt import ORDINAL, orderings
from egeria.schema import parse_question, parse_request

CALIBRATION_CASES = 20
TEMPLATE_NEGATION = "It is not the case that the following holds: {statement}"


def rotations(n: int) -> list[list[int]]:
    return [[(i + shift) % n for i in range(n)] for shift in range(n)]


def all_orders(question) -> list[list[int]]:
    """Tutte le rotazioni cicliche, più l'ordine inverso per le domande ordinali."""
    n = len(question.options)
    orders = rotations(n)
    if question.type in ORDINAL:
        orders.append(list(range(n))[::-1])
    return orders


def negated_body(question: dict, statement: str) -> dict:
    body = {"type": "noul", "instructions": statement}
    if question.get("criteria"):
        body["criteria"] = {"true": question["criteria"].get("false"), "false": question["criteria"].get("true")}
    return body


# ------------------------------------------------------------------------------------ fase 1: modello

def collect(model: str, out: Path) -> None:
    from egeria.scorer import DecisionScorer

    scorer = DecisionScorer(model)
    negations = json.loads(Path("examples/controlli/negazioni.json").read_text())["typed-decisions"]
    cases = list(load_typed_decisions("test", "all", None))
    empty_done = set()
    with out.open("w") as handle:
        for number, case in enumerate(cases):
            state, questions = parse_request(case["body"])
            rows = []  # (chiave, domanda, ordine, stato)
            for question in questions:
                raw = case["body"]["questions"][question.id]
                variants = [("base", question, state)]
                template = f"{case['workflow']}/{question.id}"
                if template not in empty_done:
                    variants.append(("vuoto", question, "N/A"))
                    empty_done.add(template)
                if question.type == "noul":
                    if template in negations:
                        variants.append(("neg_mano", parse_question(question.id, negated_body(raw, negations[template])), state))
                    automatic = TEMPLATE_NEGATION.format(statement=raw["instructions"])
                    variants.append(("neg_auto", parse_question(question.id, negated_body(raw, automatic)), state))
                for kind, q, st in variants:
                    for order in all_orders(q):
                        rows.append((kind, q, order, st))
            logits = {}
            for st in {id(r[3]): r[3] for r in rows}.values():
                subset = [r for r in rows if r[3] is st]
                sequences = [scorer.encode(st, q, order) for _, q, order, _ in subset]
                values = scorer.slot_logits(sequences, [len(order) for _, _, order, _ in subset])
                for (kind, q, order, _), value in zip(subset, values):
                    logits.setdefault(q.id, {}).setdefault(kind, []).append([order, [round(float(x), 5) for x in value]])
            record = {"case": case["id"], "workflow": case["workflow"], "questions": {}}
            for question in questions:
                gold, label = gold_vector(question, case["gold"][question.id])
                record["questions"][question.id] = {"type": question.type, "n": len(question.options),
                                                    "gold": gold, "label": label, "logits": logits[question.id]}
            handle.write(json.dumps(record) + "\n")
            if (number + 1) % 50 == 0:
                print(f"{number + 1}/{len(cases)} casi", flush=True)


# ------------------------------------------------------------------------------------ fase 2: analisi

def option_logprobs(entries, orders, bias=None) -> np.ndarray:
    """Media dei log-prob per opzione sugli ordini scelti; `bias` (per posizione) sottratto prima."""
    n = len(orders[0])
    total = np.zeros(n)
    table = {tuple(order): np.asarray(values) for order, values in entries}
    for order in orders:
        slots = np.asarray(table[tuple(order)])
        if bias is not None:
            slots = slots - bias
        back = np.empty(n)
        back[order] = log_softmax(slots)
        total += back / len(orders)
    return log_softmax(total)


def scorer_orders(qtype: str, n: int, count: int) -> list[list[int]]:
    class Fake:  # orderings() vuole solo type e options
        pass
    fake = Fake()
    fake.type, fake.options = qtype, [None] * n
    return orderings(fake, count)


def prior_from(entries_list, qtype: str, n: int) -> np.ndarray:
    """PriDe: per ogni esempio, media sulle rotazioni del log-prob di ogni posizione; poi media sugli esempi."""
    estimates = []
    for entries in entries_list:
        table = {tuple(order): np.asarray(values) for order, values in entries}
        slots = [log_softmax(table[tuple(order)]) for order in rotations(n) if tuple(order) in table]
        estimates.append(log_softmax(np.mean(slots, axis=0)))
    return log_softmax(np.mean(estimates, axis=0))


def analyse(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    stride = max(1, len(records) // CALIBRATION_CASES)
    calibration = {i for i in range(0, len(records), stride)}
    # Preferenza a priori per (tipo, n) dai casi di taratura, con tutte le rotazioni.
    priors = {}
    for index in calibration:
        for q in records[index]["questions"].values():
            priors.setdefault((q["type"], q["n"]), []).append(q["logits"]["base"])
    priors = {key: prior_from(entries, *key) for key, entries in priors.items()}
    # Stato vuoto: un record per domanda-modello (workflow/id).
    empty = {}
    for record in records:
        for qid, q in record["questions"].items():
            if "vuoto" in q["logits"]:
                empty[f"{record['workflow']}/{qid}"] = q["logits"]["vuoto"]

    def variants(q, key):
        qtype, n, entries = q["type"], q["n"], q["logits"]["base"]
        full = rotations(n) if qtype not in ORDINAL else scorer_orders(qtype, n, 2)
        bias = priors.get((qtype, n))
        empty_bias = prior_from([empty[key]], qtype, n)  # preferenza per posizione sullo stato vuoto, per questa domanda
        result = {
            "1 permutazione": option_logprobs(entries, scorer_orders(qtype, n, 1)),
            "2 permutazioni (oggi)": option_logprobs(entries, scorer_orders(qtype, n, 2)),
            "tutte le rotazioni": option_logprobs(entries, full),
            "1 perm. + priore": option_logprobs(entries, scorer_orders(qtype, n, 1), bias),
            "2 perm. + priore": option_logprobs(entries, scorer_orders(qtype, n, 2), bias),
            "1 perm. + priore da input vuoto": option_logprobs(entries, scorer_orders(qtype, n, 1), empty_bias),
        }
        if qtype in ORDINAL:
            result["score: tutte le rotazioni cicliche"] = option_logprobs(entries, rotations(n))
        cf = empty[key]
        for name in ("2 permutazioni (oggi)", "tutte le rotazioni", "1 perm. + priore"):
            orders = scorer_orders(qtype, n, 2) if "2" in name else (full if "tutte" in name else scorer_orders(qtype, n, 1))
            base_bias = bias if "priore" in name else None
            result[f"{name} + input vuoto"] = log_softmax(result[name] - option_logprobs(cf, orders, base_bias))
        if qtype == "noul":
            two = scorer_orders(qtype, n, 2)
            p_a = result["2 permutazioni (oggi)"]  # [true, false]
            for kind, label in (("neg_mano", "Sì/No simmetrizzata (negazione a mano)"),
                                ("neg_auto", "Sì/No simmetrizzata (negazione automatica)")):
                if kind in q["logits"]:
                    p_n = option_logprobs(q["logits"][kind], two)
                    logit = ((p_a[0] - p_a[1]) - (p_n[0] - p_n[1])) / 2
                    result[label] = log_softmax(np.array([logit, 0.0]))
        return result

    rows: dict[str, list] = {}
    for index, record in enumerate(records):
        if index in calibration:
            continue
        for qid, q in record["questions"].items():
            key = f"{record['workflow']}/{qid}"
            for name, logp in variants(q, key).items():
                rows.setdefault(name, []).append({"case": record["case"], "probs": np.exp(logp), "gold": q["gold"], "label": q["label"],
                                                  "type": q["type"], "workflow": record["workflow"], "question": key})
    # Calibrazione di lotto: si divide per la media delle previsioni della stessa domanda-modello.
    # - "lotto": media sugli stessi casi valutati (come nell'articolo; senza etichette ma sugli stessi dati);
    # - "lotto, stima su altri casi": casi divisi in due metà, la media di una corregge l'altra;
    # - "lotto, storico di 20 casi": media sui soli 20 casi di taratura (~5 per domanda-modello).
    calibration_rows: dict[str, dict[str, list]] = {}
    for index in calibration:
        record = records[index]
        for qid, q in record["questions"].items():
            key = f"{record['workflow']}/{qid}"
            for name, logp in variants(q, key).items():
                calibration_rows.setdefault(name, {}).setdefault(key, []).append(np.exp(logp))

    def divide(row, mean):
        probs = row["probs"] / mean
        return {**row, "probs": probs / probs.sum()}

    for name in ("1 permutazione", "2 permutazioni (oggi)", "tutte le rotazioni"):
        by_question: dict[str, list] = {}
        for row in rows[name]:
            by_question.setdefault(row["question"], []).append(row["probs"])
        means = {k: np.mean(v, axis=0) for k, v in by_question.items()}
        rows[f"{name} + lotto"] = [divide(row, means[row["question"]]) for row in rows[name]]
        halves = [{}, {}]
        for position, row in enumerate(rows[name]):
            halves[position % 2].setdefault(row["question"], []).append(row["probs"])
        half_means = [{k: np.mean(v, axis=0) for k, v in half.items()} for half in halves]
        rows[f"{name} + lotto, stima su altri casi"] = [
            divide(row, half_means[1 - position % 2][row["question"]]) for position, row in enumerate(rows[name])]
        history = {k: np.mean(v, axis=0) for k, v in calibration_rows[name].items()}
        rows[f"{name} + lotto, storico di 20 casi"] = [divide(row, history[row["question"]]) for row in rows[name]]
    report = {}
    for name, items in rows.items():
        by_type = summarize_by(items, "type")
        report[name] = {"tutte": summarize(items), **{t: by_type[t] for t in by_type}}
    reference = "2 permutazioni (oggi)"
    report["_confronti"] = {name: paired(rows[reference], rows[name]) for name in (
        "tutte le rotazioni", "1 perm. + priore da input vuoto", "1 permutazione + lotto, stima su altri casi",
        "2 permutazioni (oggi) + lotto, stima su altri casi", "tutte le rotazioni + lotto, stima su altri casi")}
    return report


def paired(a: list, b: list, resamples: int = 2000, seed: int = 0) -> dict:
    """Differenza di accuratezza B − A, con intervallo al 95% da bootstrap per caso."""
    cases = sorted({row["case"] for row in a})
    index = {case: i for i, case in enumerate(cases)}
    diff = np.zeros(len(cases))
    count = np.zeros(len(cases))
    for row_a, row_b in zip(a, b):
        i = index[row_a["case"]]
        diff[i] += float(np.argmax(row_b["probs"]) == row_b["label"]) - float(np.argmax(row_a["probs"]) == row_a["label"])
        count[i] += 1
    rng = np.random.default_rng(seed)
    samples = [diff[pick].sum() / count[pick].sum() for pick in (rng.integers(0, len(cases), len(cases)) for _ in range(resamples))]
    return {"diff": round(float(diff.sum() / count.sum()), 4),
            "ci95": [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)]}


COST = {  # righe per domanda rispetto a 1 permutazione, su typed-decisions (choice con 4–5 opzioni)
    "1 permutazione": "1×", "2 permutazioni (oggi)": "2×", "tutte le rotazioni": "choice 4–5×",
    "1 perm. + priore": "1×", "2 perm. + priore": "2×",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--solo-analisi", action="store_true")
    args = parser.parse_args()
    out = Path("runs/debiasing") / f"{args.model.split('/')[-1]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.solo_analisi:
        collect(args.model, out)
    report = analyse(out)
    (out.with_suffix(".report.json")).write_text(json.dumps(report, indent=2, default=float))
    print(f"\n{args.model}: typed-decisions test senza i {CALIBRATION_CASES} casi di taratura, senza temperatura")
    print(f"{'variante':52s} {'acc':>6s} {'noul':>6s} {'choice':>6s} {'score':>6s} {'NLL':>6s} {'ECE':>6s}")
    for name, row in report.items():
        if name.startswith("_"):
            continue
        total = row["tutte"]
        cells = [f"{total['accuracy']:.3f}"] + [f"{row[t]['accuracy']:.3f}" if t in row else "  -  " for t in ("noul", "choice", "score")]
        print(f"{name:52s} {' '.join(f'{c:>6s}' for c in cells)} {total['nll']:6.3f} {total['ece']:6.3f}")
    print("\naccuratezza rispetto a 2 permutazioni (oggi), bootstrap per caso:")
    for name, result in report["_confronti"].items():
        print(f"  {name:52s} {result['diff']:+.3f} [{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
