"""Verifica della calibrazione a input vuoto su immagini ed eventi rari, prima di renderla default.

    .venv/bin/python scripts/verifica_calibrazione.py Qwen/Qwen3.5-2B-Base

La calibrazione a input vuoto (Zhao et al. 2021) divide la previsione per quella sullo stato vuoto.
Su typed-decisions migliora le Sì/No (documentazione/09-controlli-senza-etichette.md §8), ma lì le
Sì/No diverse sono solo 6, tutte su testo. Qui si controlla dove potrebbe fare danni:

1. immagini: le 21 immagini di examples/immagini (con nuove/ ed estranee/) × 6 domande Sì/No su
   eventi presenti in 2 immagini su 21 (incendio, incidente, gatto, email, stop, scontrino), in
   italiano e in inglese;
2. testi: 30 messaggi scritti a mano, con domande su incendio ed emergenza medica (3 positivi su 15
   per lingua) e alcuni negativi vicini all'evento (examples/controlli/eventi_rari.json);
3. la suite immagini (examples/suite_immagini.jsonl): le sì/no come noul, la scala come score, le
   altre come choice, per confrontare sulle immagini anche 2 permutazioni e tutte le rotazioni;
4. pose di yoga (examples/immagini/pose): la tendenza al Sì vista sulla posa dell'albero, su 9 pose
   dell'albero, 5 negative vicine, 7 altre pose e 3 immagini senza persone. Tre Sì/No ("posa
   dell'albero?", "in equilibrio su una gamba?", "piede appoggiato all'interno dell'altra gamba?") e
   due scelte: generica ("albero / un'altra posa / nessuna") e con le pose descritte a parole.

Stato vuoto: il testo "N/A" e, per le immagini, anche un'immagine grigia uniforme.
Correzioni della Sì/No, dato lo spostamento s = logit P(Sì | stato vuoto):
- piena: si toglie s;
- tetto: si toglie s limitato a ±1;
- verso il No: si toglie s solo se è positivo (lo stato vuoto pende verso il Sì).
Rapporto in runs/calibrazione/<modello>.json.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np

from egeria.confidence import log_softmax
from egeria.images import load_images
from egeria.metrics import ece
from egeria.schema import MAX_OPTIONS, parse_question
from egeria.scorer import DecisionScorer

IMAGE_SIDE = 448
EMPTY_TEXT = "N/A"
CAP = 1.0
PERMUTATIONS = 2
IMAGES = Path("examples/immagini")
DATA = Path("examples/controlli/eventi_rari.json")
SUITE = Path("examples/suite_immagini.jsonl")
MODES = ("grezza", "piena", "tetto", "verso il No")


def image_state(name: str) -> list:
    return [{"type": "image", "path": str(IMAGES / name)}]


def gray_state() -> list:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), (128, 128, 128)).save(buffer, format="PNG")
    return [{"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode()}]


def log_probs(scorer, state, questions, permutations: int = PERMUTATIONS) -> dict[str, np.ndarray]:
    """Log-prob per opzione, nell'ordine originale, mediate sulle permutazioni."""
    if isinstance(state, list):
        scores = scorer.score_images(state, questions, permutations, load_images(state, IMAGE_SIDE))
    else:
        scores = scorer.score(state, questions, permutations)
    return {s.question.id: s.option_logits for s in scores}


def noul(qid: str, text: str):
    return parse_question(qid, {"type": "noul", "instructions": text})


def correct_noul(lp: np.ndarray, empty: np.ndarray, mode: str) -> float:
    """P(Sì) corretta con lo stato vuoto; lp ed empty sono log-prob [Sì, No]."""
    shift = float(empty[0] - empty[1])
    if mode == "grezza":
        shift = 0.0
    elif mode == "tetto":
        shift = float(np.clip(shift, -CAP, CAP))
    elif mode == "verso il No":
        shift = max(shift, 0.0)
    return float(np.exp(log_softmax(lp - np.array([shift, 0.0]))[0]))


def summarize_noul(rows: list[dict]) -> dict:
    y = np.array([r["positivo"] for r in rows], dtype=bool)
    p = np.clip(np.array([r["p_si"] for r in rows]), 1e-9, 1 - 1e-9)
    said_yes = p >= 0.5
    hit = said_yes[y].mean() if y.any() else float("nan")
    rejection = (~said_yes[~y]).mean() if (~y).any() else float("nan")
    correct = said_yes == y
    return {
        "domande": len(rows), "positive": int(y.sum()),
        "accuratezza": round(float(correct.mean()), 4),
        "bilanciata": round(float((hit + rejection) / 2), 4),
        "falsi_allarmi": round(float(1 - rejection), 4),
        "mancati": round(float(1 - hit), 4),
        "nll": round(float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))), 4),
        "ece": round(ece(np.maximum(p, 1 - p), correct), 4),
        "p_si_negative": round(float(p[~y].mean()), 4) if (~y).any() else None,
        "p_si_positive": round(float(p[y].mean()), 4) if y.any() else None,
    }


# ------------------------------------------------------------------------------------ raccolta

def collect_images(scorer, data: dict) -> tuple[list[dict], dict]:
    """Domande Sì/No su eventi rari, per immagine; stati vuoti N/A e grigio."""
    questions_by_event = data["immagini"]["domande"]
    names = sorted(p.relative_to(IMAGES).as_posix() for p in IMAGES.rglob("*")
                   if p.suffix.lower() in {".jpg", ".png"} and not p.relative_to(IMAGES).as_posix().startswith("pose/"))
    all_questions = [noul(f"{lang}:{event}", spec[lang]) for event, spec in questions_by_event.items() for lang in ("it", "en")]
    empty = {"N/A": log_probs(scorer, EMPTY_TEXT, all_questions), "grigia": log_probs(scorer, gray_state(), all_questions)}
    rows = []
    for name in names:
        values = log_probs(scorer, image_state(name), all_questions)
        for q in all_questions:
            lang, event = q.id.split(":")
            spec = questions_by_event[event]
            if name in spec.get("escluse", []):
                continue
            rows.append({"gruppo": f"immagini {lang}", "domanda": q.id, "immagine": name,
                         "positivo": name in spec["positive"], "lp": values[q.id]})
    return rows, empty


def collect_texts(scorer, data: dict) -> tuple[list[dict], dict]:
    questions = {lang: [noul(f"{lang}:{event}", text) for event, text in qs.items()] for lang, qs in data["domande"].items()}
    empty = {"N/A": log_probs(scorer, EMPTY_TEXT, [q for qs in questions.values() for q in qs])}
    rows = []
    for message in data["messaggi"]:
        lang = message["lingua"]
        values = log_probs(scorer, message["testo"], questions[lang])
        for q in questions[lang]:
            rows.append({"gruppo": f"testi {lang}", "domanda": q.id, "testo": message["testo"],
                         "positivo": q.id.split(":")[1] in message["eventi"], "lp": values[q.id]})
    return rows, empty


def suite_question(qid: str, item: dict):
    options = item["options"]
    if options == ["sì", "no"]:
        return noul(qid, item["question"]), item["expected"] == "sì"
    if "(scala crescente)" in item["question"]:
        return parse_question(qid, {"type": "score", "instructions": item["question"], "criteria": options}), options.index(item["expected"])
    body = {"type": "choice", "instructions": item["question"], "criteria": {option: None for option in options}}
    return parse_question(qid, body), options.index(item["expected"])


def collect_suite(scorer) -> list[dict]:
    """Suite immagini: per ogni domanda log-prob con 2 permutazioni e con tutte le rotazioni, più lo stato N/A."""
    by_image: dict[str, list] = {}
    for line in SUITE.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            by_image.setdefault(item["image"], []).append(item)
    rows = []
    for path, items in by_image.items():
        pairs = [suite_question(f"q{k}", item) for k, item in enumerate(items)]
        questions = [q for q, _ in pairs]
        state = [{"type": "image", "path": path}]
        two = log_probs(scorer, state, questions, PERMUTATIONS)
        every = log_probs(scorer, state, questions, MAX_OPTIONS)
        empty_two = log_probs(scorer, EMPTY_TEXT, questions, PERMUTATIONS)
        empty_every = log_probs(scorer, EMPTY_TEXT, questions, MAX_OPTIONS)
        for q, gold in pairs:
            rows.append({"tipo": q.type, "immagine": path, "domanda": q.instructions, "gold": gold,
                         "due": two[q.id], "tutte": every[q.id], "vuoto_due": empty_two[q.id], "vuoto_tutte": empty_every[q.id]})
    return rows


def collect_poses(scorer, data: dict) -> dict:
    """Pose: log-prob delle Sì/No e della scelta per immagine, con la lettura di default (tutte le rotazioni)."""
    spec = data["pose"]
    noul_questions = [noul(f"{lang}:{key}", q[lang]) for key, q in spec["domande"].items() for lang in ("it", "en")]
    choice_questions = [parse_question(f"{lang}:{name}", {"type": "choice", **choice[lang]})
                        for name, choice in spec["scelte"].items() for lang in ("it", "en")]
    questions = noul_questions + choice_questions
    empty = log_probs(scorer, EMPTY_TEXT, questions, "auto")
    rows = []
    for group, files in spec["gruppi"].items():
        for name in files:
            values = log_probs(scorer, image_state(f"pose/{name}"), questions, "auto")
            for q in questions:
                lang, key = q.id.split(":")
                row = {"gruppo": group, "immagine": name, "lingua": lang, "domanda": key, "lp": values[q.id], "vuoto": empty[q.id]}
                if key in spec["scelte"]:
                    gold = spec["scelte"][key]["gold"][group]
                    row["gold"] = q.keys.index(gold) if gold else None
                else:
                    row["positivo"] = group in spec["domande"][key]["positivi"]
                rows.append(row)
    return {"righe": rows, "vuoto": {q.id: np.exp(empty[q.id]).round(3).tolist() for q in questions}}


def analyse_poses(poses: dict) -> dict:
    rows = poses["righe"]
    report: dict = {"stato vuoto": poses["vuoto"], "sì/no": {}, "scelta": {}}
    for key in dict.fromkeys(r["domanda"] for r in rows if "positivo" in r):
        for mode in ("grezza", "piena", "verso il No"):
            items = [{**r, "p_si": correct_noul(r["lp"], r["vuoto"], mode)} for r in rows if r["domanda"] == key]
            result = {"tutte": summarize_noul(items)}
            for group in dict.fromkeys(r["gruppo"] for r in items):
                subset = [r for r in items if r["gruppo"] == group]
                result[group] = {"p_si_media": round(float(np.mean([r["p_si"] for r in subset])), 3),
                                 "detti_si": round(float(np.mean([r["p_si"] >= 0.5 for r in subset])), 3)}
            report["sì/no"][f"{key}, {mode}"] = result
    for name in dict.fromkeys(r["domanda"] for r in rows if "gold" in r):
        choice = [r for r in rows if r["domanda"] == name]
        report["scelta"][name] = {}
        for group in dict.fromkeys(r["gruppo"] for r in choice):
            subset = [r for r in choice if r["gruppo"] == group]
            probs = np.exp([r["lp"] for r in subset])
            graded = [int(np.argmax(r["lp"])) == r["gold"] for r in subset if r["gold"] is not None]
            report["scelta"][name][group] = {"immagini": len(subset),
                                             "corrette": round(float(np.mean(graded)), 3) if graded else None,
                                             "p_albero_media": round(float(probs[:, 0].mean()), 3),
                                             "detto_albero": round(float(np.mean(probs.argmax(1) == 0)), 3)}
    return report


# ------------------------------------------------------------------------------------ analisi

def analyse(image_rows, image_empty, text_rows, text_empty, suite_rows) -> dict:
    report: dict = {"sì/no": {}, "stato vuoto": {}, "suite": {}}
    variants: dict[str, list] = {}
    for rows, empties in ((image_rows, image_empty), (text_rows, text_empty)):
        for row in rows:
            for empty_name, empty in empties.items():
                for mode in MODES:
                    if mode == "grezza" and empty_name != "N/A":
                        continue
                    name = mode if mode == "grezza" else f"{mode} ({empty_name})"
                    p = correct_noul(row["lp"], empty[row["domanda"]], mode)
                    variants.setdefault(name, []).append({**row, "p_si": p})
    for name, rows in variants.items():
        groups: dict[str, list] = {}
        for row in rows:
            groups.setdefault(row["gruppo"], []).append(row)
            groups.setdefault("immagini" if row["gruppo"].startswith("immagini") else "testi", []).append(row)
            groups.setdefault("tutte", []).append(row)
        report["sì/no"][name] = {group: summarize_noul(items) for group, items in groups.items()}
    for kind, empties in (("immagini", image_empty), ("testi", text_empty)):
        for empty_name, empty in empties.items():
            report["stato vuoto"][f"{kind}, {empty_name}"] = {
                qid: round(float(np.exp(values[0])), 3) for qid, values in empty.items()}

    # Suite: choice con 2 permutazioni / tutte le rotazioni, con e senza N/A; noul e score con e senza N/A.
    for kind in ("choice", "noul", "score"):
        rows = [r for r in suite_rows if r["tipo"] == kind]
        if not rows:
            continue
        result = {}
        for orders in (("due", "vuoto_due"), ("tutte", "vuoto_tutte")) if kind == "choice" else (("due", "vuoto_due"),):
            for use_empty in (False, True):
                name = ("2 permutazioni" if orders[0] == "due" else "tutte le rotazioni") + (" + N/A" if use_empty else "")
                correct, nll, conf = [], [], []
                for r in rows:
                    lp = r[orders[0]] - (r[orders[1]] if use_empty else 0.0)
                    lp = log_softmax(lp)
                    gold = (0 if r["gold"] else 1) if kind == "noul" else r["gold"]
                    correct.append(int(np.argmax(lp)) == gold)
                    nll.append(-float(lp[gold]))
                    conf.append(float(np.exp(lp.max())))
                result[name] = {"domande": len(rows), "corrette": int(sum(correct)),
                                "nll": round(float(np.mean(nll)), 4), "ece": round(ece(conf, correct), 4)}
        report["suite"][kind] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device, dtype = ("cpu", "float32") if args.cpu else ("cuda", "bfloat16")
    scorer = DecisionScorer(args.model, device=device, dtype=dtype, vision=True)
    data = json.loads(DATA.read_text())
    started = time.perf_counter()
    image_rows, image_empty = collect_images(scorer, data)
    text_rows, text_empty = collect_texts(scorer, data)
    suite_rows = collect_suite(scorer)
    poses = collect_poses(scorer, data)
    report = {"modello": args.model, "permutazioni": PERMUTATIONS, "tetto": CAP,
              **analyse(image_rows, image_empty, text_rows, text_empty, suite_rows),
              "pose": analyse_poses(poses), "secondi": round(time.perf_counter() - started, 1)}
    out = Path("runs/calibrazione") / f"{args.model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{args.model}: Sì/No (acc, bilanciata, falsi allarmi, mancati, NLL, ECE, P(Sì) medio su negative / positive)")
    for group in ("immagini it", "immagini en", "testi it", "testi en", "immagini", "testi", "tutte"):
        print(f"  {group}")
        for name, groups in report["sì/no"].items():
            s = groups.get(group)
            if s:
                print(f"    {name:22s} {s['accuratezza']:.3f}  {s['bilanciata']:.3f}  FA {s['falsi_allarmi']:.3f}  "
                      f"mancati {s['mancati']:.3f}  NLL {s['nll']:.3f}  ECE {s['ece']:.3f}  "
                      f"P(Sì) {s['p_si_negative']:.2f} / {s['p_si_positive']:.2f}  (n={s['domande']}, pos={s['positive']})")
    print("  P(Sì | stato vuoto) per domanda")
    for key, values in report["stato vuoto"].items():
        print(f"    {key:20s} " + "  ".join(f"{k} {v:.2f}" for k, v in values.items()))
    print("  suite immagini")
    for kind, result in report["suite"].items():
        for name, s in result.items():
            print(f"    {kind:7s} {name:28s} {s['corrette']}/{s['domande']}  NLL {s['nll']:.3f}  ECE {s['ece']:.3f}")
    print("  pose di yoga: Sì/No (P(Sì) media e quota di Sì per gruppo; tutte: acc, bilanciata, falsi allarmi, mancati)")
    print("    P(risposta | stato vuoto): " + "  ".join(f"{k} {v}" for k, v in report["pose"]["stato vuoto"].items()))
    for name, result in report["pose"]["sì/no"].items():
        t = result["tutte"]
        groups = "  ".join(f"{g} {v['p_si_media']:.2f}/{v['detti_si']:.2f}" for g, v in result.items() if g != "tutte")
        print(f"    {name:26s} acc {t['accuratezza']:.3f} bil {t['bilanciata']:.3f} FA {t['falsi_allarmi']:.3f} "
              f"mancati {t['mancati']:.3f} | {groups}")
    print("  pose di yoga: scelte (corrette, P(albero) media, quota detta «albero»)")
    for name, groups in report["pose"]["scelta"].items():
        for group, v in groups.items():
            graded = f"{v['corrette']:.2f}" if v["corrette"] is not None else "  - "
            print(f"    {name:9s} {group:15s} n={v['immagini']:2d}  corrette {graded}  P(albero) {v['p_albero_media']:.2f}  "
                  f"detto albero {v['detto_albero']:.2f}")
    print(f"\nrapporto: {out} ({report['secondi']} s)")


if __name__ == "__main__":
    main()
