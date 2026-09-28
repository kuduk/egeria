"""Interfaccia a riga di comando.

    egeria info
    egeria decide   --model M richiesta.json
    egeria predict  --model M --split test --out pred.jsonl
    egeria calibrate --predictions pred-train.jsonl --out temperature.json
    egeria evaluate --predictions pred-test.jsonl [--temperatures temperature.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def _scorer(args):
    from .scorer import DecisionScorer

    return DecisionScorer(
        args.model, device=args.device, dtype=args.dtype, quantize=args.quantize,
        prompt_style=args.prompt_style, batch_tokens=args.batch_tokens, vision=getattr(args, "vision", False),
    )


def cmd_info(_args) -> int:
    import importlib.util

    import torch
    import transformers

    print(f"torch {torch.__version__}, transformers {transformers.__version__}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name(0)}: {free / 1e9:.1f} GB liberi su {total / 1e9:.1f} GB")
    else:
        print("GPU CUDA non disponibile")
    for module, package in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        found = importlib.util.find_spec(module) is not None
        status = "installato" if found else "MANCANTE: fallback PyTorch lento (vedi documentazione/03-baseline-f0.md)"
        print(f"{package}: {status}")
    return 0


def _memory(args):
    from .memory import MemoryStore

    return MemoryStore.load(args.memory) if getattr(args, "memory", None) else None


def cmd_decide(args) -> int:
    from .calibration import load_temperatures

    from .schema import is_multimodal

    body = json.loads(Path(args.request).read_text() if args.request != "-" else sys.stdin.read())
    # Con immagini nello stato serve il modello completo con la torre visiva.
    args.vision = args.vision or is_multimodal(body.get("state"))
    if args.recall and "memory" not in body:
        body["memory"] = {"recall": args.recall}
    response = _scorer(args).decide(
        body, permutations=args.permutations, temperatures=load_temperatures(args.temperatures),
        default_min_confidence=args.min_confidence, memory=_memory(args),
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


def cmd_predict(args) -> int:
    from .datasets import gold_vector, load_typed_decisions
    from .schema import parse_request

    scorer = _scorer(args)
    cases = list(load_typed_decisions(args.split, args.config, args.limit))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_decisions = 0
    started = time.perf_counter()
    with out.open("w") as handle:
        for number, case in enumerate(cases, 1):
            state, questions = parse_request(case["body"])
            case_start = time.perf_counter()
            scores = scorer.score(state, questions, args.permutations)
            case_ms = (time.perf_counter() - case_start) * 1000
            for s in scores:
                gold, label = gold_vector(s.question, case["gold"][s.question.id])
                record = {
                    "case": case["id"],
                    "workflow": case["workflow"],
                    "question": s.question.id,
                    "type": s.question.type,
                    "options": s.question.keys,
                    "option_logits": s.option_logits.tolist(),
                    "order_argmax": s.order_argmax,
                    "gold": gold,
                    "label": label,
                    "input_tokens": s.input_tokens,
                    "case_ms": case_ms,
                }
                handle.write(json.dumps(record) + "\n")
                total_decisions += 1
            if number % 20 == 0 or number == len(cases):
                elapsed = time.perf_counter() - started
                print(f"{number}/{len(cases)} casi, {total_decisions} decisioni, {elapsed:.0f} s", file=sys.stderr)
    meta = {
        "model": args.model, "dtype": args.dtype, "quantize": args.quantize, "prompt_style": args.prompt_style,
        "permutations": args.permutations, "split": args.split, "config": args.config, "cases": len(cases),
        "decisions": total_decisions, "seconds": time.perf_counter() - started,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


def _read_predictions(path: str) -> tuple[list[dict], dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta_path = Path(path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return records, meta


def cmd_calibrate(args) -> int:
    from .calibration import fit_by_type

    records, _ = _read_predictions(args.predictions)
    temperatures = fit_by_type(records)
    Path(args.out).write_text(json.dumps(temperatures, indent=2))
    print(json.dumps(temperatures, indent=2))
    return 0


def _fmt(row: dict, keys) -> str:
    return " ".join(f"{key}={row[key]:.3f}" if isinstance(row.get(key), float) else f"{key}={row.get(key)}" for key in keys)


def cmd_evaluate(args) -> int:
    from .calibration import load_temperatures
    from .confidence import softmax
    from .metrics import flip_rate, summarize, summarize_by

    records, meta = _read_predictions(args.predictions)
    temperatures = load_temperatures(args.temperatures)
    for record in records:
        record["probs"] = softmax(record["option_logits"], temperatures.get(record["type"], 1.0))
    keys = ("n", "accuracy", "nll", "kl", "brier_mean", "ece", "mean_confidence", "coverage_at_5")
    report = {
        "model": meta.get("model"),
        "temperatures": temperatures,
        "overall": summarize(records),
        "by_type": summarize_by(records, "type"),
        "by_workflow": summarize_by(records, "workflow"),
        "flip_rate": flip_rate(records),
        "ms_per_case": float(np.mean(list({r["case"]: r["case_ms"] for r in records}.values()))),
    }
    print(f"modello: {report['model']}  temperature: {temperatures or 'nessuna (T=1)'}")
    print("totale  ", _fmt(report["overall"], keys + ("score_mae",)))
    for name, row in report["by_type"].items():
        print(f"{name:8s}", _fmt(row, keys))
    for name, row in report["by_workflow"].items():
        print(f"{name[:8]:8s}", _fmt(row, keys))
    print(f"flip rate permutazioni: {report['flip_rate']}  ms/caso: {report['ms_per_case']:.0f}")
    if args.prior:
        # Riferimento che non legge lo stato: distribuzione gold media per (workflow, domanda) sul train.
        train, _ = _read_predictions(args.prior)
        groups: dict = {}
        for r in train:
            groups.setdefault((r["workflow"], r["question"]), []).append(np.asarray(r["gold"]))
        means = {key: np.mean(values, axis=0) for key, values in groups.items()}
        prior = [{**r, "probs": means[(r["workflow"], r["question"])]} for r in records]
        report["prior"] = summarize(prior)
        print("prior   ", _fmt(report["prior"], keys))

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    return 0


def cmd_compare(args) -> int:
    """Tabella markdown da più report di `evaluate --report`."""
    header = "| Modello | T | Acc | noul | choice | score | NLL | KL | Brier/k | ECE | Flip | ms/caso |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for path in args.reports:
        report = json.loads(Path(path).read_text())
        overall, by_type = report["overall"], report["by_type"]
        calibrated = "fit" if report.get("temperatures") else "1"
        flip = report.get("flip_rate")
        cells = [
            str(report.get("model", path)).split("/")[-1], calibrated, f"{overall['accuracy']:.3f}",
            *(f"{by_type[t]['accuracy']:.3f}" if t in by_type else "-" for t in ("noul", "choice", "score")),
            f"{overall['nll']:.3f}", f"{overall['kl']:.3f}", f"{overall['brier_mean']:.3f}", f"{overall['ece']:.3f}",
            f"{flip:.2f}" if flip is not None else "-", f"{report.get('ms_per_case', 0):.0f}",
        ]
        print("| " + " | ".join(cells) + " |")
    return 0


def _calibrated(path: str, temperatures_path: str | None) -> list[dict]:
    from .calibration import load_temperatures
    from .confidence import softmax

    records, _ = _read_predictions(path)
    temperatures = load_temperatures(temperatures_path)
    for record in records:
        record["probs"] = softmax(record["option_logits"], temperatures.get(record["type"], 1.0))
    return records


def cmd_paired(args) -> int:
    """Confronto appaiato B - A con intervalli bootstrap per caso."""
    from .metrics import paired_bootstrap

    a = _calibrated(args.a, args.temperatures_a)
    b = _calibrated(args.b, args.temperatures_b)
    print(f"B - A  (A={args.a}, B={args.b}); accuracy: >0 meglio B; nll e brier: <0 meglio B")
    groups = [("totale", lambda r: True)]
    groups += [(t, lambda r, t=t: r["type"] == t) for t in ("noul", "choice", "score")]
    groups += [(w, lambda r, w=w: r["workflow"] == w) for w in sorted({r["workflow"] for r in a})]
    report = {}
    for name, keep in groups:
        result = paired_bootstrap([r for r in a if keep(r)], [r for r in b if keep(r)], args.resamples)
        report[name] = result
        cells = []
        for metric in ("accuracy", "nll", "brier"):
            diff, (low, high) = result[metric]["diff"], result[metric]["ci95"]
            mark = "*" if low > 0 or high < 0 else " "
            cells.append(f"{metric}={diff:+.3f} [{low:+.3f},{high:+.3f}]{mark}")
        print(f"{name[:12]:12s} n={result['n']:4d}  " + "  ".join(cells))
    print("* = intervallo al 95% che esclude lo zero")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


def cmd_suite(args) -> int:
    """Suite di richieste con risposte attese: una riga JSON per caso {id, body, expected}."""
    from .calibration import load_temperatures
    from .schema import parse_request

    temperatures = load_temperatures(args.temperatures)
    scorer = _scorer(args)
    cases = [json.loads(line) for line in Path(args.suite).read_text().splitlines() if line.strip()]
    rows = []
    elapsed = 0.0
    for case in cases:
        body = case["body"]
        started = time.perf_counter()
        response = scorer.decide(body, args.permutations, temperatures, default_min_confidence=args.min_confidence)
        elapsed += time.perf_counter() - started
        _, questions = parse_request(body, args.min_confidence)
        for q in questions:
            result = response["answers"][q.id]
            probs = result.get("probabilities") or {"true": result.get("noul"), "false": 1 - result.get("noul", 0)}
            expected = str(case["expected"][q.id])
            chosen = max(probs, key=probs.get)
            rows.append({
                "case": case["id"], "question": q.id, "type": q.type, "expected": expected, "chosen": chosen,
                "p_expected": float(probs[expected]), "correct": chosen == expected, "status": result.get("status"),
            })
            mark = "ok " if chosen == expected else "NO "
            extra = f" {result['status']}" if result.get("status") else ""
            print(f"{mark} {case['id'][:22]:22s} {q.id[:18]:18s} atteso={expected:16s} scelto={chosen:16s} "
                  f"p(atteso)={rows[-1]['p_expected']:.2f}{extra}")
    correct = np.mean([r["correct"] for r in rows])
    p_expected = np.mean([r["p_expected"] for r in rows])
    print(f"\n{args.model}: {int(sum(r['correct'] for r in rows))}/{len(rows)} corrette "
          f"({correct:.0%}), p(atteso) media {p_expected:.2f}, tempo {elapsed:.2f} s")
    if args.min_confidence is not None:
        uncertain = sum(r["status"] == "uncertain" for r in rows)
        print(f"  min_confidence={args.min_confidence}: {uncertain}/{len(rows)} incerte (sotto soglia)")
    for qtype in ("noul", "choice", "score"):
        subset = [r for r in rows if r["type"] == qtype]
        if subset:
            print(f"  {qtype:7s} {sum(r['correct'] for r in subset)}/{len(subset)}")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return 0


def _gold_decisions(question, gold_answer) -> str:
    """Decisione confermata da salvare: la chiave dell'opzione (per le score, il livello)."""
    return str(gold_answer["label"])


def cmd_memory(args) -> int:
    from .memory import Memory, MemoryStore
    from .schema import is_multimodal

    state = None
    if args.action != "build":
        state = json.loads(Path(args.state).read_text() if args.state != "-" else sys.stdin.read())
        state = state.get("state", state) if isinstance(state, dict) else state
        # Con immagini nello stato serve il modello completo con la torre visiva.
        args.vision = args.vision or is_multimodal(state)
    scorer = _scorer(args)
    if args.action == "build":
        from .datasets import load_typed_decisions
        from .schema import parse_request

        store = MemoryStore()
        cases = list(load_typed_decisions(args.split, args.config, args.limit))
        for number, case in enumerate(cases, 1):
            state, questions = parse_request(case["body"])
            vector = scorer.analyze_state(state)[0]["embedding"]
            decisions = {q.id: _gold_decisions(q, case["gold"][q.id]) for q in questions}
            store.add(vector, Memory(case["id"], state, decisions, meta={"workflow": case["workflow"]}))
            if number % 100 == 0 or number == len(cases):
                print(f"{number}/{len(cases)} ricordi", file=sys.stderr)
        store.save(args.memory)
        print(f"archivio {args.memory}: {len(store)} ricordi")
        return 0
    store = MemoryStore.load(args.memory)
    vector = scorer.analyze_state(state, image_max_side=args.image_max_side)[0]["embedding"]
    if args.action == "add":
        decisions = json.loads(args.decisions) if args.decisions else {}
        store.add(vector, Memory(args.id, state, decisions, note=args.note or ""))
        store.save(args.memory)
        print(f"aggiunto {args.id!r}: {len(store)} ricordi in {args.memory}")
    else:
        for score, item in store.search(vector, args.k):
            print(f"{score:+.3f}  {item.id}  {json.dumps(item.decisions, ensure_ascii=False)[:120]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egeria", description="Modello decisionale System 1 su Qwen3.5")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--device", default="cuda", help="cuda oppure cpu (su cpu usare --dtype float32)")
        p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        p.add_argument("--quantize", default=None, choices=["4bit"])
        p.add_argument("--prompt-style", default="chat", choices=["chat", "plain"])
        p.add_argument("--batch-tokens", type=int, default=12288)
        p.add_argument("--permutations", type=int, default=1)
        p.add_argument("--vision", action="store_true", help="carica il modello con la torre visiva (stati con immagini)")

    sub.add_parser("info", help="ambiente, GPU e kernel veloci").set_defaults(func=cmd_info)

    p = sub.add_parser("decide", help="risponde a una richiesta /v1/systemone")
    model_args(p)
    p.add_argument("request", help="file JSON della richiesta, - per stdin")
    p.add_argument("--temperatures")
    p.add_argument("--memory", help="cartella dell'archivio dei ricordi")
    p.add_argument("--recall", type=int, default=0, help="ricordi da richiamare se la richiesta non lo indica")
    p.add_argument("--min-confidence", type=float,
                   help="default del server per min_confidence; la richiesta e le singole domande lo sovrascrivono")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("predict", help="predizioni su typed-decisions")
    model_args(p)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--config", default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("memory", help="archivio dei ricordi: build, add, search")
    model_args(p)
    p.add_argument("action", choices=["build", "add", "search"])
    p.add_argument("--memory", required=True, help="cartella dell'archivio")
    p.add_argument("--split", default="train", choices=["train", "test"], help="build: split di typed-decisions")
    p.add_argument("--config", default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--state", help="add/search: file JSON con lo stato (o una richiesta), - per stdin")
    p.add_argument("--id", help="add: identificativo del ricordo")
    p.add_argument("--decisions", help='add: JSON delle decisioni confermate, es. {"reparto": "pagamenti"}')
    p.add_argument("--note", help="add: nota libera")
    p.add_argument("-k", type=int, default=3, help="search: quanti ricordi")
    p.add_argument("--image-max-side", type=int, help="lato massimo delle immagini (usare lo stesso valore delle richieste)")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("calibrate", help="fitta le temperature per tipo di domanda")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate", help="metriche sulle predizioni")
    p.add_argument("--predictions", required=True)
    p.add_argument("--temperatures")
    p.add_argument("--prior", help="predizioni del train: stampa il riferimento che non legge lo stato")
    p.add_argument("--report", help="salva il report completo in JSON")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="tabella markdown da più report JSON")
    p.add_argument("reports", nargs="+")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("model-server", help="server del modello: solo inferenza (/v1/systemone, /v1/embed), senza stato")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quantize", default=None, choices=["4bit"])
    p.add_argument("--text-only", action="store_true", help="senza torre visiva (niente immagini)")
    p.add_argument("--temperatures", help="file delle temperature (calibrazione, mai applicata alle immagini)")
    p.add_argument("--permutations", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.5, help="soglia di default sotto cui una risposta è 'incerta'")
    p.add_argument("--image-max-side", type=int, default=448, help="lato massimo delle immagini, se la richiesta non lo indica")
    p.add_argument("--allow-paths", action="store_true", help="accetta immagini indicate con un percorso locale")
    p.add_argument("--token", help="richiede 'Authorization: Bearer <token>' (anche da EGERIA_TOKEN)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.set_defaults(func=lambda args: __import__("egeria.model_server", fromlist=["serve_model"]).serve_model(args))

    p = sub.add_parser("serve", help="interfaccia web (domande, storico, ricordi); usa il server del modello")
    p.add_argument("--model-url", default="http://127.0.0.1:8100", help="indirizzo di egeria model-server")
    p.add_argument("--model-token", help="token del server del modello (anche da EGERIA_TOKEN)")
    p.add_argument("--memory", default="runs/memoria-console", help="cartella dell'archivio dei ricordi")
    p.add_argument("--data", default="runs/console", help="cartella dei casi e delle immagini caricate")
    p.add_argument("--image-max-side", type=int, default=448, help="lato massimo delle immagini se l'interfaccia non lo indica")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=lambda args: __import__("egeria.server", fromlist=["serve"]).serve(args))

    p = sub.add_parser("paired", help="confronto appaiato fra due modelli con bootstrap per caso")
    p.add_argument("--a", required=True, help="predizioni del modello A")
    p.add_argument("--b", required=True, help="predizioni del modello B")
    p.add_argument("--temperatures-a")
    p.add_argument("--temperatures-b")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--report")
    p.set_defaults(func=cmd_paired)

    p = sub.add_parser("suite", help="suite di richieste con risposte attese")
    model_args(p)
    p.add_argument("suite", help="file JSONL: {id, body, expected}")
    p.add_argument("--temperatures")
    p.add_argument("--min-confidence", type=float,
                   help="default per min_confidence: sotto la soglia la risposta è 'uncertain'")
    p.add_argument("--out")
    p.set_defaults(func=cmd_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
