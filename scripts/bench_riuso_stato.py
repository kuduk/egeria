"""Riuso dello stato: le risposte restano le stesse? Quanto tempo si risparmia, e quando?

    .venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-2B-Base            # GPU, testo e immagini
    .venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-0.8B-Base --cpu    # CPU (float32), solo testo

Per ogni richiesta delle suite confronta
`share_state="never"` (ogni prompt per intero) con `"always"` (prefisso una volta, poi le code),
oppure con `"auto"` (--mode auto: condiviso solo oltre la soglia):
- accordo sulla risposta e differenza massima di probabilità;
- tempo mediano di `decide` (3 ripetizioni dopo un riscaldamento);
- token evitati = prefisso × (righe − 1), per scegliere la soglia della modalità auto.

Richieste: le 12 di examples/suite_it.jsonl (testo) e, raggruppate per immagine, le domande di
examples/suite_immagini.jsonl (una richiesta per immagine, a ogni lato indicato con --sides).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from egeria.schema import is_multimodal
from egeria.scorer import DecisionScorer



def load_cases(sides: list[int]) -> list[dict]:
    cases = [json.loads(line) for line in open("examples/suite_it.jsonl") if line.strip()]
    by_image: dict[str, dict] = {}
    for line in open("examples/suite_immagini.jsonl"):
        if line.strip():
            item = json.loads(line)
            questions = by_image.setdefault(item["image"], {})
            questions[f"q{len(questions)}"] = {"type": "choice", "instructions": item["question"],
                                               "criteria": {option: None for option in item["options"]}}
    for side in sides:
        for image, questions in by_image.items():
            state = [{"type": "text", "text": "Immagine:"}, {"type": "image", "path": image}]
            name = image.rsplit("/", 1)[-1]
            cases.append({"id": f"{name} {side}px", "body": {"state": state, "image_max_side": side, "questions": questions}})
    return cases


def distribution(result: dict) -> tuple[str, np.ndarray]:
    """Risposta scelta e probabilità, per confrontare i due percorsi."""
    if result["type"] == "noul":
        p = np.array([result["noul"], 1 - result["noul"]])
        return ("true" if p[0] >= 0.5 else "false"), p
    if result["type"] == "short_answer":
        return result["answer"], np.array([result["confidence"]])
    probabilities = result["probabilities"]
    return max(probabilities, key=probabilities.get), np.array(list(probabilities.values()))


def timed(scorer, body, permutations, repeats=3):
    scorer.decide(body, permutations=permutations)  # riscaldamento
    times, response = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        response = scorer.decide(body, permutations=permutations)
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times), response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--cpu", action="store_true", help="CPU in float32, solo le richieste di testo")
    parser.add_argument("--permutations", type=int, default=2)
    parser.add_argument("--sides", default="448,800", help="lati delle immagini, separati da virgole")
    parser.add_argument("--mode", default="always", choices=["always", "auto"],
                        help="percorso da confrontare con never: always (sempre condiviso) o auto (soglia)")
    args = parser.parse_args()
    device, dtype = ("cpu", "float32") if args.cpu else ("cuda", "bfloat16")
    scorer = DecisionScorer(args.model, device=device, dtype=dtype, vision=not args.cpu)

    shared = {}
    original = scorer._worth_sharing

    def recording(prefix, rows):  # annota prefisso e righe dell'ultima decisione
        shared.update(prefix=prefix, rows=rows)
        return original(prefix, rows)

    scorer._worth_sharing = recording
    cases = load_cases([int(side) for side in args.sides.split(",")])
    rows, agree, total, max_diff = [], 0, 0, 0.0
    for case in cases:
        body = case["body"]
        kind = "immagini" if is_multimodal(body["state"]) else "testo"
        if args.cpu and kind == "immagini":
            continue
        scorer.share_state = "never"
        t_never, r_never = timed(scorer, body, args.permutations)
        scorer.share_state = args.mode
        shared.clear()
        t_always, r_always = timed(scorer, body, args.permutations)
        saved = shared.get("prefix", 0) * (shared.get("rows", 1) - 1)
        diffs = []
        for qid, result in r_never["answers"].items():
            a, pa = distribution(result)
            b, pb = distribution(r_always["answers"][qid])
            agree += a == b
            total += 1
            diffs.append(float(np.abs(pa - pb).max()))
        max_diff = max(max_diff, max(diffs))
        rows.append((kind, saved, t_never, t_always))
        print(f"{case['id'][:24]:24s} {kind:8s} token evitati {saved:6d}  intero {t_never:7.1f} ms  "
              f"condiviso {t_always:7.1f} ms  ({(t_always / t_never - 1) * 100:+5.0f}%)  Δp max {max(diffs):.4f}")
    print(f"\n{args.model} su {device}, {args.permutations} permutazioni")
    print(f"risposte uguali: {agree}/{total}; differenza massima di probabilità: {max_diff:.4f}")
    for kind in ("testo", "immagini"):
        subset = [r for r in rows if r[0] == kind]
        if subset:
            ratios = [a / n for _, _, n, a in subset]
            print(f"{kind}: tempo condiviso / intero, mediana {statistics.median(ratios):.2f} "
                  f"(min {min(ratios):.2f}, max {max(ratios):.2f}) su {len(subset)} richieste")
    print("soglia: rapporto medio per fascia di token evitati")
    for low, high in ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 10**9)):
        subset = [a / n for _, saved, n, a in rows if low <= saved < high]
        if subset:
            print(f"  {low:5d}–{high if high < 10**9 else '∞'}: {statistics.mean(subset):.2f} ({len(subset)} richieste)")


if __name__ == "__main__":
    main()
