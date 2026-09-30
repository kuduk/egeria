"""Controlli senza etichette: posizione delle opzioni, tendenza al "Sì", coerenza fra lingue.

    .venv/bin/python scripts/controlli_senza_etichette.py Qwen/Qwen3.5-2B-Base [--limit 200]

1. Posizione. Domande `choice` con opzioni tutte identiche ("Opzione", "Opzione", ...), una sola
   permutazione: un modello senza preferenze di posizione dà probabilità uguali alle lettere.
   Misura: distanza di variazione totale (TV) media dalla distribuzione uniforme, per 2–5 opzioni,
   su stati di testo (typed-decisions, suite italiana) e immagini, con istruzioni in italiano e inglese.
2. Tendenza al "Sì". Ogni affermazione Sì/No (`noul`) è posta anche negata, con le descrizioni di
   true/false scambiate. Se il modello è coerente P(Sì|A) + P(Sì|non A) ≈ 1; l'eccesso sopra 1 è la
   tendenza al "Sì". Contraddizione = stessa risposta ad affermazione e negazione. Variante: la stessa
   affermazione come `choice` con l'opzione "lo stato non permette di dirlo".
3. Lingua. La stessa domanda sullo stesso stato in italiano e in inglese; si confrontano le
   distribuzioni per posizione delle opzioni: accordo sulla risposta e TV media.

Soglie fissate PRIMA di vedere i risultati (29/09/2026), in SOGLIE. Dati: examples/controlli/.
Rapporto completo in runs/controlli/<modello>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from egeria.confidence import softmax
from egeria.datasets import load_typed_decisions
from egeria.images import load_images
from egeria.schema import Option, Question
from egeria.scorer import DecisionScorer

SOGLIE = {
    "posizione_tv": 0.05,  # TV media dalla distribuzione uniforme, per ogni numero di opzioni
    "si_eccesso": 0.05,  # media di P(Sì|A) + P(Sì|non A) - 1
    "si_contraddizioni": 0.10,  # quota di coppie con la stessa risposta ad A e a non A
    "lingua_accordo": 0.90,  # quota di risposte uguali fra italiano e inglese
    "lingua_tv": 0.10,  # TV media fra le distribuzioni nelle due lingue
}
IMAGE_SIDE = 448
DATA = Path("examples/controlli")


def tv(p, q) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def distribution(result: dict) -> np.ndarray:
    """Probabilità nell'ordine delle opzioni della domanda (noul: [true, false])."""
    if result["type"] == "noul":
        return np.array([result["noul"], 1 - result["noul"]])
    return np.array(list(result["probabilities"].values()))


def image_state(path: str, header: str | None = None) -> list:
    parts = [{"type": "text", "text": header}] if header else []
    return parts + [{"type": "image", "path": path}]


# ------------------------------------------------------------------------------------------ posizione

def check_position(scorer, text_states, image_paths, permutations=(1, 2, 5)) -> dict:
    """Per ogni numero di permutazioni: con 1 si misura la preferenza grezza per le lettere; con più
    permutazioni (rotazioni cicliche, come nello scorer) quanto ne resta dopo la media. Con 5 ogni
    opzione passa per tutte le posizioni (fino a 5 opzioni)."""
    labels = {"it": ("Quale opzione scegli?", "Opzione"), "en": ("Which option do you choose?", "Option")}
    rows = {}  # (permutazioni, lingua, n) -> lista di distribuzioni
    questions = [
        (lang, n, Question(f"{lang}{n}", "choice", instruction, tuple(Option(label, None) for _ in range(n))))
        for lang, (instruction, label) in labels.items() for n in range(2, 6)
    ]
    plain = [q for _, _, q in questions]
    for count in permutations:
        for state in text_states:
            for (lang, n, _), score in zip(questions, scorer.score(state, plain, count)):
                rows.setdefault((count, lang, n), []).append(softmax(score.option_logits))
        for path in image_paths:
            state = image_state(path)
            images = load_images(state, IMAGE_SIDE)
            for (lang, n, _), score in zip(questions, scorer.score_images(state, plain, count, images)):
                rows.setdefault((count, lang, n), []).append(softmax(score.option_logits))
    report = {}
    for (count, lang, n), dists in sorted(rows.items()):
        uniform = np.full(n, 1 / n)
        mean = np.mean(dists, axis=0)
        report[f"{count} perm., {lang} n={n}"] = {
            "permutazioni": count, "stati": len(dists),
            "tv_media": round(statistics.mean(tv(d, uniform) for d in dists), 4),
            "profilo": [round(float(x), 3) for x in mean],  # probabilità media per opzione (lettera con 1 perm.)
        }
    return report


# ----------------------------------------------------------------------------------------- tendenza Sì

def negated(question: dict, statement: str) -> dict:
    """La stessa domanda noul con l'affermazione negata e le descrizioni di true/false scambiate."""
    result = {"type": "noul", "instructions": statement}
    criteria = question.get("criteria")
    if criteria:
        result["criteria"] = {"true": criteria.get("false"), "false": criteria.get("true")}
    return result


def unknown_choice(statement: str, lang: str) -> dict:
    options = {
        "it": {"si": "Sì, vale per lo stato", "no": "No, non vale per lo stato", "non_si_sa": "Lo stato non permette di dirlo"},
        "en": {"yes": "Yes, it holds for the state", "no": "No, it does not hold for the state",
               "unknown": "The state does not make it possible to tell"},
    }[lang]
    return {"type": "choice", "instructions": statement, "criteria": options}


def yes_pairs(cases, suite) -> list[dict]:
    data = json.loads((DATA / "negazioni.json").read_text())
    pairs = []
    for case in cases:
        for qid, question in case["body"]["questions"].items():
            negation = data["typed-decisions"].get(f"{case['workflow']}/{qid}")
            if question["type"] == "noul" and negation:
                pairs.append({"fonte": "typed-decisions", "lingua": "en", "state": case["body"]["state"],
                              "a": question, "non_a": negated(question, negation)})
    for case in suite:
        for question in case["body"]["questions"].values():
            entry = data["suite_it"].get(question["instructions"])
            if question["type"] == "noul" and entry:
                state = case["body"]["state"]
                pairs.append({"fonte": "suite italiana", "lingua": "it", "state": state, "a": question,
                              "non_a": negated(question, entry["negazione"])})
                english = {"type": "noul", "instructions": entry["en"][0]}
                pairs.append({"fonte": "suite italiana", "lingua": "en", "state": state, "a": english,
                              "non_a": negated(english, entry["en"][1])})
    for item in data["immagini"]:
        for lang, header in (("it", "Immagine:"), ("en", "Image:")):
            statement, negation = item[lang]
            question = {"type": "noul", "instructions": statement}
            pairs.append({"fonte": "immagini", "lingua": lang, "state": image_state(item["image"], header),
                          "a": question, "non_a": negated(question, negation)})
    return pairs


def check_yes(scorer, pairs, permutations) -> dict:
    groups: dict[str, list] = {}
    for pair in pairs:
        yes = "si" if pair["lingua"] == "it" else "yes"
        no = "no"
        body = {"state": pair["state"], "image_max_side": IMAGE_SIDE, "questions": {
            "a": pair["a"], "non_a": pair["non_a"],
            "a_scelta": unknown_choice(pair["a"]["instructions"], pair["lingua"]),
            "non_a_scelta": unknown_choice(pair["non_a"]["instructions"], pair["lingua"]),
        }}
        answers = scorer.decide(body, permutations=permutations)["answers"]
        pa, pn = answers["a"]["noul"], answers["non_a"]["noul"]

        def normalized(result):
            p = result["probabilities"]
            return p[yes] / max(p[yes] + p[no], 1e-9), 1 - p[yes] - p[no]

        ca, unknown_a = normalized(answers["a_scelta"])
        cn, unknown_n = normalized(answers["non_a_scelta"])
        row = {"somma": pa + pn, "contraddizione": (pa >= 0.5) == (pn >= 0.5), "entrambe_si": pa >= 0.5 and pn >= 0.5,
               "entrambe_no": pa < 0.5 and pn < 0.5, "affermazione": pair["a"]["instructions"], "p_a": pa, "p_non_a": pn,
               "somma_scelta": ca + cn, "contraddizione_scelta": (ca >= 0.5) == (cn >= 0.5),
               "non_si_sa": (unknown_a + unknown_n) / 2}
        for key in (f"{pair['fonte']} ({pair['lingua']})", "tutte"):
            groups.setdefault(key, []).append(row)
    statements: dict[str, list] = {}
    for row in groups["tutte"]:
        statements.setdefault(row["affermazione"], []).append(row)
    report = {"per_affermazione": {
        text: {"coppie": len(rows), "p_si_affermazione": round(statistics.mean(r["p_a"] for r in rows), 3),
               "p_si_negazione": round(statistics.mean(r["p_non_a"] for r in rows), 3),
               "entrambe_si": round(statistics.mean(r["entrambe_si"] for r in rows), 3),
               "entrambe_no": round(statistics.mean(r["entrambe_no"] for r in rows), 3)}
        for text, rows in statements.items()}}
    for key, rows in groups.items():
        report[key] = {
            "coppie": len(rows),
            "eccesso": round(statistics.mean(r["somma"] for r in rows) - 1, 4),
            "contraddizioni": round(statistics.mean(r["contraddizione"] for r in rows), 4),
            "di_cui_entrambe_si": round(statistics.mean(r["entrambe_si"] for r in rows), 4),
            "di_cui_entrambe_no": round(statistics.mean(r["entrambe_no"] for r in rows), 4),
            "scelta_eccesso": round(statistics.mean(r["somma_scelta"] for r in rows) - 1, 4),
            "scelta_contraddizioni": round(statistics.mean(r["contraddizione_scelta"] for r in rows), 4),
            "scelta_massa_non_si_sa": round(statistics.mean(r["non_si_sa"] for r in rows), 4),
        }
    return report


# ---------------------------------------------------------------------------------------------- lingua

def language_items(cases, suite) -> list[dict]:
    data = json.loads((DATA / "lingue.json").read_text())
    items = []
    for case in cases:
        questions = {}
        for qid, question in case["body"]["questions"].items():
            italian = data["typed-decisions (italiano)"].get(f"{case['workflow']}/{qid}")
            if italian:
                questions[f"en:{qid}"] = question
                questions[f"it:{qid}"] = {"type": question["type"], **italian}
        items.append({"fonte": "typed-decisions", "state": case["body"]["state"], "questions": questions})
    for case in suite:
        questions = {}
        for qid, question in case["body"]["questions"].items():
            english = data["suite_it (inglese)"].get(question["instructions"])
            if english:
                questions[f"it:{qid}"] = question
                questions[f"en:{qid}"] = {"type": question["type"], **english}
        items.append({"fonte": "suite italiana", "state": case["body"]["state"], "questions": questions})
    by_image: dict[str, dict] = {}
    for line in open("examples/suite_immagini.jsonl"):
        if not line.strip():
            continue
        item = json.loads(line)
        english = data["immagini (inglese)"][item["question"]]
        questions = by_image.setdefault(item["image"], {})
        n = len(questions) // 2
        questions[f"it:q{n}"] = {"type": "choice", "instructions": item["question"],
                                 "criteria": {option: None for option in item["options"]}}
        questions[f"en:q{n}"] = {"type": "choice", "instructions": english["question"],
                                 "criteria": {option: None for option in english["options"]}}
    for path, questions in by_image.items():
        items.append({"fonte": "immagini", "state": image_state(path), "questions": questions})
    return items


def check_language(scorer, items, permutations) -> dict:
    groups: dict[str, list] = {}
    for item in items:
        body = {"state": item["state"], "image_max_side": IMAGE_SIDE, "questions": item["questions"]}
        answers = scorer.decide(body, permutations=permutations)["answers"]
        for key in item["questions"]:
            if not key.startswith("it:"):
                continue
            other = "en:" + key[3:]
            pi, pe = distribution(answers[key]), distribution(answers[other])
            row = {"accordo": int(np.argmax(pi) == np.argmax(pe)), "tv": tv(pi, pe), "tipo": answers[key]["type"]}
            names = [item["fonte"], f"tipo {row['tipo']}", "tutte"]
            if max(pi.max(), pe.max()) >= 0.7:  # almeno una delle due lingue è sicura della risposta
                names.append("tutte, risposte sicure (p ≥ 0.7)")
            for group in names:
                groups.setdefault(group, []).append(row)
    return {
        key: {"domande": len(rows), "accordo": round(statistics.mean(r["accordo"] for r in rows), 4),
              "tv_media": round(statistics.mean(r["tv"] for r in rows), 4)}
        for key, rows in groups.items()
    }


# ---------------------------------------------------------------------------------------------- main

def verdict(ok: bool) -> str:
    return "superata" if ok else "NON superata"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--limit", type=int, default=200, help="casi di typed-decisions (test), campionati su tutti i flussi")
    parser.add_argument("--permutations", type=int, default=2, help="per i controlli sul Sì e sulla lingua")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device, dtype = ("cpu", "float32") if args.cpu else ("cuda", "bfloat16")
    scorer = DecisionScorer(args.model, device=device, dtype=dtype, vision=True)
    cases = list(load_typed_decisions("test", "all", args.limit))
    suite = [json.loads(line) for line in open("examples/suite_it.jsonl") if line.strip()]
    images = sorted({json.loads(line)["image"] for line in open("examples/suite_immagini.jsonl") if line.strip()})
    report = {"modello": args.model, "dispositivo": device, "permutazioni": args.permutations, "soglie": SOGLIE}

    started = time.perf_counter()
    text_states = [case["body"]["state"] for case in cases[:60]] + [case["body"]["state"] for case in suite]
    report["posizione"] = check_position(scorer, text_states, images)
    print("\n1. POSIZIONE (opzioni identiche): TV media dalla distribuzione uniforme")
    for key, row in report["posizione"].items():
        print(f"  {key:20s} stati {row['stati']:3d}  TV {row['tv_media']:.3f}  profilo {row['profilo']}")
    for count in (1, 2, 5):
        worst = max(row["tv_media"] for row in report["posizione"].values() if row["permutazioni"] == count)
        print(f"  {count} permutazioni: peggiore {worst:.3f} (soglia {SOGLIE['posizione_tv']}): "
              f"{verdict(worst <= SOGLIE['posizione_tv'])}")

    report["si"] = check_yes(scorer, yes_pairs(cases, suite), args.permutations)
    print("\n2. TENDENZA AL SÌ: eccesso = media di P(Sì|A) + P(Sì|non A) - 1")
    for key, row in report["si"].items():
        if key == "per_affermazione":
            continue
        print(f"  {key:26s} coppie {row['coppie']:3d}  eccesso {row['eccesso']:+.3f}  contraddizioni {row['contraddizioni']:.2f} "
              f"(entrambe Sì {row['di_cui_entrambe_si']:.2f}, entrambe No {row['di_cui_entrambe_no']:.2f})  | con 'non si può dire': eccesso "
              f"{row['scelta_eccesso']:+.3f}, contraddizioni {row['scelta_contraddizioni']:.2f}, massa {row['scelta_massa_non_si_sa']:.2f}")
    print("  per affermazione: P(Sì) media dell'affermazione / della negazione, entrambe Sì, entrambe No")
    for text, row in report["si"]["per_affermazione"].items():
        print(f"    {text[:70]:70s} n={row['coppie']:3d}  {row['p_si_affermazione']:.2f} / {row['p_si_negazione']:.2f}  "
              f"{row['entrambe_si']:.2f}  {row['entrambe_no']:.2f}")
    total = report["si"]["tutte"]
    ok = total["eccesso"] <= SOGLIE["si_eccesso"] and total["contraddizioni"] <= SOGLIE["si_contraddizioni"]
    print(f"  tutte: eccesso {total['eccesso']:+.3f} (soglia {SOGLIE['si_eccesso']}), contraddizioni "
          f"{total['contraddizioni']:.2f} (soglia {SOGLIE['si_contraddizioni']}): {verdict(ok)}")

    report["lingua"] = check_language(scorer, language_items(cases, suite), args.permutations)
    print("\n3. LINGUA: stessa domanda in italiano e in inglese")
    for key, row in report["lingua"].items():
        print(f"  {key:34s} domande {row['domande']:4d}  accordo {row['accordo']:.2f}  TV media {row['tv_media']:.3f}")
    total = report["lingua"]["tutte"]
    ok = total["accordo"] >= SOGLIE["lingua_accordo"] and total["tv_media"] <= SOGLIE["lingua_tv"]
    print(f"  tutte: accordo {total['accordo']:.2f} (soglia {SOGLIE['lingua_accordo']}), TV {total['tv_media']:.3f} "
          f"(soglia {SOGLIE['lingua_tv']}): {verdict(ok)}")

    report["secondi"] = round(time.perf_counter() - started, 1)
    out = Path("runs/controlli") / f"{args.model.split('/')[-1]}-{device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nrapporto completo: {out} ({report['secondi']} s)")


if __name__ == "__main__":
    main()
