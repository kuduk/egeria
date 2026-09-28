"""Verifica del recupero (RAG) con stati che contengono immagini.

    .venv/bin/python scripts/verifica_rag_immagini.py Qwen/Qwen3.5-2B-Base

Archivio: le 6 foto reali di examples/immagini/, ognuna con categoria e azione decisa.
Interrogazioni: 6 foto nuove delle stesse categorie (examples/immagini/nuove/), diverse da quelle
in archivio. Soglia decisa in anticipo: il ricordo più simile è della categoria giusta in almeno 5 casi su 6.
"""

import sys

from egeria.memory import Memory, MemoryStore
from egeria.scorer import DecisionScorer

MAX_SIDE = 448
ARCHIVE = {  # foto in archivio -> decisioni confermate
    "scontrino.jpg": {"categoria": "scontrino", "azione": "archiviare_spesa"},
    "email_phishing.png": {"categoria": "phishing", "azione": "segnalare_sicurezza"},
    "incidente_auto.jpg": {"categoria": "incidente", "azione": "aprire_sinistro"},
    "stop_italia.jpg": {"categoria": "segnale_stradale", "azione": "nessuna"},
    "casa_incendio.jpg": {"categoria": "incendio", "azione": "chiamare_vigili"},
    "gatto.jpg": {"categoria": "animale", "azione": "nessuna"},
}
QUERIES = {  # foto nuove -> categoria attesa
    "nuove/scontrino2.jpg": "scontrino",
    "nuove/phishing2.png": "phishing",
    "nuove/incidente2.jpg": "incidente",
    "nuove/stop2.jpg": "segnale_stradale",
    "nuove/incendio2.jpg": "incendio",
    "nuove/gatto2.jpg": "animale",
}
QUESTIONS = {
    "categoria": {"type": "choice", "instructions": "Che cosa mostra l'immagine?",
                  "criteria": {"scontrino": "Uno scontrino o una ricevuta", "phishing": "Un'email di phishing o truffa",
                               "incidente": "Un incidente stradale", "segnale_stradale": "Un segnale stradale",
                               "incendio": "Un incendio", "animale": "Un animale"}},
    "azione": {"type": "choice", "instructions": "Quale azione è appropriata?",
               "criteria": {"archiviare_spesa": "Archiviare la spesa in contabilità",
                            "segnalare_sicurezza": "Segnalare al team di sicurezza informatica",
                            "aprire_sinistro": "Aprire una pratica di sinistro", "chiamare_vigili": "Chiamare i vigili del fuoco",
                            "nessuna": "Nessuna azione"}},
}
EXPECTED_ACTION = {c: d["azione"] for d in ARCHIVE.values() for c in [d["categoria"]]}


def state(path: str) -> list:
    return [{"type": "text", "text": "Immagine ricevuta:"}, {"type": "image", "path": f"examples/immagini/{path}"}]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B-Base"
    scorer = DecisionScorer(model, vision=True)
    store = MemoryStore()
    for path, decisions in ARCHIVE.items():
        vector = scorer.analyze_state(state(path), image_max_side=MAX_SIDE)[0]["embedding"]
        store.add(vector, Memory(path, state(path), decisions))

    hits = votes_ok = model_ok = 0
    for path, category in QUERIES.items():
        body = {"state": state(path), "image_max_side": MAX_SIDE, "memory": {"recall": 3}, "questions": QUESTIONS}
        response = scorer.decide(body, permutations=2, memory=store)
        top = response["memories"][0]
        found = ARCHIVE[top["id"]]["categoria"]
        hits += found == category
        vote = response["answers"]["azione"]["memory"]["answer"]
        votes_ok += vote == EXPECTED_ACTION[category]
        model_ok += response["answers"]["categoria"]["choice"] == category
        ranking = ", ".join(f"{m['id']} {m['similarity']:.2f}" for m in response["memories"])
        print(f"{'✓' if found == category else '✗'} {path:22s} ricordo più simile: {top['id']:20s} | "
              f"voto azione: {vote:20s} | modello categoria: {response['answers']['categoria']['choice']:16s} | {ranking}")
    print(f"\nricordo più simile della categoria giusta: {hits}/{len(QUERIES)}  -> {'TENERE' if hits >= 5 else 'NON SUPERATA'}")
    print(f"voto dei ricordi sull'azione corretto: {votes_ok}/{len(QUERIES)}; modello zero-shot sulla categoria: {model_ok}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
