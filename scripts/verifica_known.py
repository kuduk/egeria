"""Verifica di `known`: "è qualcosa che hai in memoria?".

    .venv/bin/python scripts/verifica_known.py Qwen/Qwen3.5-2B-Base

Tre gruppi di interrogazioni, per immagini e testo:
- stesso: un ricordo già in archivio, leggermente modificato (foto ritagliata; testo con un dettaglio in più);
- simile: casi nuovi dello stesso tipo di quelli in archivio;
- nuovo: casi di tipo assente dall'archivio.
Misure: similarità con il ricordo più vicino e percentile rispetto ai vicini interni dell'archivio.
Soglia decisa in anticipo: AUROC >= 0.85 sia per stesso/simile sia per simile/nuovo.
"""

import json
import sys

import numpy as np
from PIL import Image

from egeria.datasets import load_typed_decisions
from egeria.memory import Memory, MemoryStore
from egeria.schema import parse_request
from egeria.scorer import DecisionScorer

MAX_SIDE = 448
ARCHIVE = ["scontrino.jpg", "email_phishing.png", "incidente_auto.jpg", "stop_italia.jpg", "casa_incendio.jpg", "gatto.jpg"]
SIMILAR = ["nuove/scontrino2.jpg", "nuove/phishing2.png", "nuove/incidente2.jpg", "nuove/stop2.jpg",
           "nuove/incendio2.jpg", "nuove/gatto2.jpg"]
NOVEL = ["estranee/pizza.jpg", "estranee/montagna.jpg", "estranee/portatile.jpg", "estranee/rastrelliera_bici.jpg",
         "estranee/scoglio_mare.jpg", "estranee/barca.jpg"]
OFF_TOPIC = [
    "Per la carbonara servono guanciale, uova, pecorino e pepe nero: niente panna.",
    "Domani cielo sereno al nord, qualche nuvola al centro e temperature in lieve aumento.",
    "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura.",
    "La Juventus ha vinto due a uno in trasferta con un gol nei minuti di recupero.",
    "Il Colosseo fu inaugurato nell'anno 80 dopo Cristo sotto l'imperatore Tito.",
    "Per rinvasare il basilico usa terriccio fresco e annaffia senza ristagni.",
    "La fotosintesi trasforma luce, acqua e anidride carbonica in zuccheri e ossigeno.",
    "Il treno regionale per Firenze parte dal binario 7 alle 8 e 12.",
    "Il romanzo racconta l'estate di due fratelli in un paese di montagna.",
    "Per il mal di testa riposa al buio e bevi molta acqua.",
    "Il concerto di Vivaldi apre con un allegro in mi maggiore.",
    "La maratona di Roma si corre ogni anno a marzo lungo le strade del centro.",
    "Il gatto di casa dorme sul divano tutto il pomeriggio.",
    "Le rondini tornano in primavera e nidificano sotto le grondaie.",
    "La nuova versione del videogioco aggiunge tre mappe e una modalità cooperativa.",
    "Il museo resta chiuso il lunedì e apre alle 9 negli altri giorni.",
    "La torta di mele richiede un'ora di forno a 180 gradi.",
    "Il lago di Como è circondato da montagne e ville storiche.",
    "Il corso di yoga del mercoledì è spostato alle 19.",
    "La luna piena di settembre è detta luna del raccolto.",
]


def auroc(low, high) -> float:
    """Probabilità che un elemento di `high` superi uno di `low` (pareggi a metà)."""
    return float(np.mean([(h > l) + 0.5 * (h == l) for h in high for l in low]))


def image_state(path, image=None):
    part = {"type": "image", "path": f"examples/immagini/{path}"}
    return [{"type": "text", "text": "Immagine ricevuta:"}, part]


def cropped(path):
    image = Image.open(f"examples/immagini/{path}").convert("RGB")
    w, h = image.size
    image = image.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def report(name, groups):
    for measure in ("similarity", "percentile"):
        values = {g: [r[measure] for r in rows] for g, rows in groups.items()}
        same_sim = auroc(values["simile"], values["stesso"])
        sim_new = auroc(values["nuovo"], values["simile"])
        print(f"  {name:8s} {measure:10s} stesso {np.round(values['stesso'], 3)}\n"
              f"  {'':8s} {'':10s} simile {np.round(values['simile'], 3)}\n"
              f"  {'':8s} {'':10s} nuovo  {np.round(values['nuovo'], 3)}\n"
              f"  {'':8s} {'':10s} AUROC stesso/simile {same_sim:.2f}   simile/nuovo {sim_new:.2f}")


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B-Base"
    scorer = DecisionScorer(model, vision=True)

    # Immagini: archivio di 6 foto.
    store = MemoryStore()
    for path in ARCHIVE:
        store.add(scorer.analyze_state(image_state(path), image_max_side=MAX_SIDE)[0]["embedding"], Memory(path, image_state(path)))
    groups = {"stesso": [], "simile": [], "nuovo": []}
    for path in ARCHIVE:
        vector = scorer.analyze_state(image_state(path), images=[cropped(path)])[0]["embedding"]
        groups["stesso"].append(store.familiarity(vector))
    for group, paths in (("simile", SIMILAR), ("nuovo", NOVEL)):
        for path in paths:
            vector = scorer.analyze_state(image_state(path), image_max_side=MAX_SIDE)[0]["embedding"]
            groups[group].append(store.familiarity(vector))
    report("immagini", groups)

    # Testo: archivio typed-decisions (1.200 casi di train).
    store = MemoryStore.load(f"runs/memoria-{model.split('/')[-1]}")
    rng = np.random.default_rng(0)
    groups = {"stesso": [], "simile": [], "nuovo": []}
    for index in rng.choice(len(store), 20, replace=False):
        state = store.items[int(index)].state
        edited = dict(state, nota="Pratica ricontrollata dall'operatore.") if isinstance(state, dict) else state + " Grazie."
        groups["stesso"].append(store.familiarity(scorer.analyze_state(edited)[0]["embedding"]))
    test = list(load_typed_decisions("test", limit=20))
    for case in test:
        groups["simile"].append(store.familiarity(scorer.analyze_state(parse_request(case["body"])[0])[0]["embedding"]))
    for text in OFF_TOPIC:
        groups["nuovo"].append(store.familiarity(scorer.analyze_state(text)[0]["embedding"]))
    report("testo", groups)


if __name__ == "__main__":
    main()
