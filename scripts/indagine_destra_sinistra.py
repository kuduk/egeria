"""Indagine su destra e sinistra nelle immagini.

    .venv/bin/python scripts/indagine_destra_sinistra.py Qwen/Qwen3.5-2B-Base

Tre parti, in italiano e in inglese, con la lettura di default (tutte le rotazioni delle opzioni):

1. Figure sintetiche (nessuna ambiguità): due figure colorate una accanto all'altra. Domande:
   - relativa: "Il cerchio rosso è a sinistra o a destra del quadrato blu?";
   - assoluta: "In quale metà dell'immagine si trova il cerchio rosso?";
   - Sì/No: "Il cerchio rosso è a sinistra del quadrato blu?".
   Ogni scena anche specchiata: la risposta deve invertirsi. Controllo: le stesse figure una sopra
   l'altra, con "sopra o sotto?" (se sopra/sotto va e destra/sinistra no, il difetto è orizzontale).
2. Foto di pose (examples/immagini/pose): la gamba sollevata, chiesta in due cornici:
   - dell'immagine: "Nell'immagine, la gamba sollevata è a sinistra o a destra?";
   - della persona: "La persona ha sollevato la gamba destra o la sinistra?". Per una persona vista di
     fronte la sua destra sta a sinistra nell'immagine. Si conta anche quante volte la risposta
     "della persona" coincide con il lato nell'immagine (confusione delle cornici).
   Anche qui ogni foto è specchiata: nell'immagine specchiata cambiano sia il lato sia la gamba.
3. Foto di mani alzate (examples/immagini/mani), persona vista di fronte come davanti a una webcam:
   "La mano alzata è la destra?" (la domanda dell'utente), "ha alzato la sua mano destra o la sinistra?",
   "nell'immagine la mano alzata è sul lato sinistro o destro?", "in quale metà dell'immagine è la mano alzata?".
4. Chiralità (examples/immagini/mani/primi_piani): primi piani di mani destre e sinistre, palmo o dorso.
   La mano specchiata è l'altra mano: se il modello usasse la chiralità, la risposta si invertirebbe.
   Si chiede anche ciò da cui la chiralità si ricava (palmo o dorso, lato del pollice, dita in su o in giù)
   e si ricompone la mano nel codice ("composta").
5. Riepilogo: accuratezza, coerenza con lo specchio, preferenza per "sinistra".

Rapporto in runs/destra_sinistra/<modello>.json.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from egeria.scorer import DecisionScorer

SIDE = 448
POSE = Path("examples/immagini/pose")
SHAPES = {  # nome italiano, nome inglese, colore RGB
    "cerchio_rosso": ("il cerchio rosso", "the red circle", (220, 30, 30)),
    "quadrato_blu": ("il quadrato blu", "the blue square", (30, 60, 220)),
    "triangolo_verde": ("il triangolo verde", "the green triangle", (30, 160, 60)),
    "cerchio_giallo": ("il cerchio giallo", "the yellow circle", (230, 190, 20)),
}
# Gamba sollevata: lato nell'immagine e, per chi si vede di fronte o di schiena, la gamba della persona.
# Etichette fatte guardando le foto (29/09/2026 → 05/10/2026). Di lato: solo il lato nell'immagine.
POSES = {
    "kavya_vrikshasan": ("sinistra", "fronte"), "free_souls": ("destra", "fronte"),
    "tree_pose_at_home": ("destra", "schiena"), "vriksasana_yoga_asana_nina_mel": ("destra", "fronte"),
    "tree_pose_with_the_trees": ("sinistra", "fronte"), "tree_pose": ("destra", "fronte"),
    "a_style_of_vrikshasana_1": ("destra", "fronte"), "a_style_of_vrikshasana_2": ("sinistra", "fronte"),
    "woman_yoga_pose": ("destra", "fronte"), "balance": ("sinistra", "lato"), "dancer_s_pose": ("sinistra", "lato"),
    "kavya_devduttasan": ("sinistra", "lato"), "mr_yoga_lord_of_dance_1": ("sinistra", "lato"),
}
OTHER = {"sinistra": "destra", "destra": "sinistra", "sopra": "sotto", "sotto": "sopra"}
HANDS = Path("examples/immagini/mani")
# Primi piani: (mano, lato visto, lato del pollice nell'immagine, dita); None = non valutabile.
CLOSEUPS = {"mano_1": ("sinistra", "dorso", "destra", "su"), "mano_2": ("sinistra", "dorso", "destra", "su"),
            "mano_3": ("sinistra", "palmo", "sinistra", "su"), "mano_4": ("sinistra", "dorso", None, None),
            "mano_5": ("sinistra", "palmo", "sinistra", "su"), "mano_6": ("destra", "dorso", "destra", "giù"),
            "mano_7": ("destra", "palmo", "destra", "su")}
CLOSEUP_QUESTIONS = {
    "it:mano": {"type": "choice", "instructions": "Questa è una mano destra o una mano sinistra?",
                "criteria": {"destra": "una mano destra", "sinistra": "una mano sinistra"}},
    "en:mano": {"type": "choice", "instructions": "Is this a right hand or a left hand?",
                "criteria": {"destra": "a right hand", "sinistra": "a left hand"}},
    "it:palmo_dorso": {"type": "choice", "instructions": "Della mano si vede il palmo o il dorso?",
                       "criteria": {"palmo": "il palmo", "dorso": "il dorso"}},
    "it:pollice": {"type": "choice", "instructions": "Nell'immagine, il pollice è sul lato sinistro o sul lato destro della mano?",
                   "criteria": {"sinistra": "sul lato sinistro", "destra": "sul lato destro"}},
    "it:dita": {"type": "choice", "instructions": "Le dita della mano puntano verso l'alto o verso il basso?",
                "criteria": {"su": "verso l'alto", "giù": "verso il basso"}},
}
SAME_UNDER_MIRROR = ("verticale", "palmo_dorso", "dita")  # specchiando in orizzontale non cambiano


def hand_from_cues(side: str, thumb: str, fingers: str) -> str:
    """Chiralità: con le dita in su, palmo e pollice a destra (o dorso e pollice a sinistra) = mano destra."""
    right = (side == "palmo") == (thumb == "destra")
    if fingers == "giù":
        right = not right
    return "destra" if right else "sinistra"
# Mano alzata della persona; tutte viste di fronte, quindi nell'immagine sta dall'altro lato.
RAISED_HANDS = {"bessie_smith": "sinistra", "poliziotta_giuramento": "destra", "statuetta_saluto": "destra",
                "giuramento_ritaglio": "destra"}
HAND_QUESTIONS = {  # (domanda, cornice, chiave della risposta "Sì" per le Sì/No)
    "it:è_la_destra": ({"type": "noul", "instructions": "La mano alzata è la destra?"}, "persona", "destra"),
    "en:è_la_destra": ({"type": "noul", "instructions": "Is the raised hand the right one?"}, "persona", "destra"),
    "it:persona": ({"type": "choice", "instructions": "La persona ha alzato la sua mano destra o la sua mano sinistra?",
                    "criteria": {"destra": "la sua mano destra", "sinistra": "la sua mano sinistra"}}, "persona", None),
    "it:immagine": ({"type": "choice", "instructions": "Nell'immagine, la mano alzata è sul lato sinistro o sul lato destro?",
                     "criteria": {"sinistra": "sul lato sinistro dell'immagine", "destra": "sul lato destro dell'immagine"}}, "immagine", None),
    "it:metà": ({"type": "choice", "instructions": "In quale metà dell'immagine si trova la mano alzata?",
                 "criteria": {"sinistra": "nella metà sinistra", "destra": "nella metà destra"}}, "immagine", None),
}


def as_state(image: Image.Image) -> list:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return [{"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode()}]


def draw_shape(draw, name, center, radius):
    x, y = center
    color = SHAPES[name][2]
    if name.startswith("cerchio"):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif name.startswith("quadrato"):
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
    else:
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)


def scenes(count: int, seed: int = 0):
    """Coppie di figure affiancate (orizzontali) o una sopra l'altra (verticali), con la verità."""
    rng = random.Random(seed)
    names = list(SHAPES)
    for number in range(count):
        a, b = rng.sample(names, 2)
        vertical = number % 2 == 1
        first = rng.random() < 0.5  # a a sinistra (o sopra)
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        radius = rng.randint(40, 70)
        if vertical:
            x = rng.randint(200, 440)
            ya, yb = (120, 360) if first else (360, 120)
            draw_shape(draw, a, (x + rng.randint(-30, 30), ya), radius)
            draw_shape(draw, b, (x + rng.randint(-30, 30), yb), radius)
            truth = "sopra" if first else "sotto"
        else:
            y = rng.randint(160, 320)
            xa, xb = (160, 480) if first else (480, 160)
            draw_shape(draw, a, (xa, y + rng.randint(-40, 40)), radius)
            draw_shape(draw, b, (xb, y + rng.randint(-40, 40)), radius)
            truth = "sinistra" if first else "destra"
        yield {"id": f"scena{number}", "a": a, "b": b, "vertical": vertical, "truth": truth, "image": image}


def synthetic_questions(scene: dict) -> dict:
    a_it, a_en, _ = SHAPES[scene["a"]]
    b_it, b_en, _ = SHAPES[scene["b"]]
    if scene["vertical"]:
        return {
            "it:verticale": {"type": "choice", "instructions": f"{a_it.capitalize()} è sopra o sotto {b_it}?",
                             "criteria": {"sopra": "sopra", "sotto": "sotto"}},
            "en:verticale": {"type": "choice", "instructions": f"Is {a_en} above or below {b_en}?",
                             "criteria": {"sopra": "above", "sotto": "below"}},
        }
    return {
        "it:relativa": {"type": "choice", "instructions": f"{a_it.capitalize()} è a sinistra o a destra di {b_it}?",
                        "criteria": {"sinistra": "a sinistra", "destra": "a destra"}},
        "en:relativa": {"type": "choice", "instructions": f"Is {a_en} to the left or to the right of {b_en}?",
                        "criteria": {"sinistra": "to the left", "destra": "to the right"}},
        "it:assoluta": {"type": "choice", "instructions": f"In quale metà dell'immagine si trova {a_it}?",
                        "criteria": {"sinistra": "nella metà sinistra", "destra": "nella metà destra"}},
        "en:assoluta": {"type": "choice", "instructions": f"In which half of the image is {a_en}?",
                        "criteria": {"sinistra": "in the left half", "destra": "in the right half"}},
        "it:sì/no": {"type": "noul", "instructions": f"{a_it.capitalize()} è a sinistra di {b_it}?"},
        "en:sì/no": {"type": "noul", "instructions": f"Is {a_en} to the left of {b_en}?"},
    }


POSE_QUESTIONS = {
    "it:immagine": {"type": "choice", "instructions": "Nell'immagine, la gamba sollevata è sul lato sinistro o sul lato destro?",
                    "criteria": {"sinistra": "sul lato sinistro dell'immagine", "destra": "sul lato destro dell'immagine"}},
    "en:immagine": {"type": "choice", "instructions": "In the image, is the raised leg on the left side or on the right side?",
                    "criteria": {"sinistra": "on the left side of the image", "destra": "on the right side of the image"}},
    "it:persona": {"type": "choice", "instructions": "La persona ha sollevato la sua gamba destra o la sua gamba sinistra?",
                   "criteria": {"destra": "la sua gamba destra", "sinistra": "la sua gamba sinistra"}},
    "en:persona": {"type": "choice", "instructions": "Has the person raised their right leg or their left leg?",
                   "criteria": {"destra": "their right leg", "sinistra": "their left leg"}},
}


def answer_of(result: dict) -> tuple[str, float]:
    """Chiave scelta e probabilità di "sinistra" (o "sopra"; per le Sì/No la probabilità del Sì = sinistra)."""
    if result["type"] == "noul":
        return ("sinistra" if result["noul"] >= 0.5 else "destra"), result["noul"]
    p = result["probabilities"]
    first = "sopra" if "sopra" in p else "sinistra"
    return result["choice"], p[first]


def run(scorer, count: int) -> dict:
    rows = []
    for scene in scenes(count):
        for mirrored in (False, True):
            image = ImageOps.mirror(scene["image"]) if mirrored else scene["image"]
            truth = scene["truth"] if not mirrored or scene["vertical"] else OTHER[scene["truth"]]
            questions = synthetic_questions(scene)
            answers = scorer.decide({"state": as_state(image), "image_max_side": SIDE, "questions": questions})["answers"]
            for qid, result in answers.items():
                chosen, p_first = answer_of(result)
                rows.append({"parte": "sintetiche", "caso": scene["id"], "specchiata": mirrored, "domanda": qid,
                             "vero": truth, "scelta": chosen, "p_sinistra": round(p_first, 4)})
    for name, (side, view) in POSES.items():
        original = Image.open(POSE / f"{name}.jpg").convert("RGB")
        for mirrored in (False, True):
            image = ImageOps.mirror(original) if mirrored else original
            image_side = OTHER[side] if mirrored else side
            # gamba della persona: di fronte la sua destra è a sinistra nell'immagine; di schiena coincide
            body = None if view == "lato" else (OTHER[image_side] if view == "fronte" else image_side)
            answers = scorer.decide({"state": as_state(image), "image_max_side": SIDE, "questions": POSE_QUESTIONS})["answers"]
            for qid, result in answers.items():
                chosen, p_first = answer_of(result)
                truth = image_side if qid.endswith("immagine") else body
                rows.append({"parte": "pose", "caso": name, "vista": view, "specchiata": mirrored, "domanda": qid,
                             "cornice": "immagine" if qid.endswith("immagine") else "persona",
                             "vero": truth, "lato_immagine": image_side, "scelta": chosen, "p_sinistra": round(p_first, 4)})
    for name, hand in RAISED_HANDS.items():
        original = Image.open(HANDS / f"{name}.jpg").convert("RGB")
        for mirrored in (False, True):
            image = ImageOps.mirror(original) if mirrored else original
            person = OTHER[hand] if mirrored else hand
            image_side = OTHER[person]
            questions = {qid: q for qid, (q, _, _) in HAND_QUESTIONS.items()}
            answers = scorer.decide({"state": as_state(image), "image_max_side": SIDE, "questions": questions})["answers"]
            for qid, (_, frame, yes_key) in HAND_QUESTIONS.items():
                result = answers[qid]
                chosen = (yes_key if result["noul"] >= 0.5 else OTHER[yes_key]) if result["type"] == "noul" else result["choice"]
                rows.append({"parte": "mani", "caso": name, "vista": "fronte", "specchiata": mirrored, "domanda": qid,
                             "cornice": frame, "vero": person if frame == "persona" else image_side,
                             "lato_immagine": image_side, "scelta": chosen})
    for name, (hand, side, thumb, fingers) in CLOSEUPS.items():
        original = Image.open(HANDS / "primi_piani" / f"{name}.jpg").convert("RGB")
        for mirrored in (False, True):
            image = ImageOps.mirror(original) if mirrored else original
            truth = {"it:mano": OTHER[hand] if mirrored else hand, "en:mano": OTHER[hand] if mirrored else hand,
                     "it:palmo_dorso": side, "it:pollice": (OTHER[thumb] if mirrored else thumb) if thumb else None,
                     "it:dita": fingers}
            answers = scorer.decide({"state": as_state(image), "image_max_side": SIDE, "questions": CLOSEUP_QUESTIONS})["answers"]
            chosen = {qid: answers[qid]["choice"] for qid in CLOSEUP_QUESTIONS}
            chosen["composta (codice)"] = hand_from_cues(chosen["it:palmo_dorso"], chosen["it:pollice"], chosen["it:dita"])
            truth["composta (codice)"] = truth["it:mano"] if fingers else None  # mano di lato: la regola non vale
            for qid, value in chosen.items():
                rows.append({"parte": "chiralità", "caso": name, "specchiata": mirrored, "domanda": qid,
                             "vero": truth[qid], "scelta": value})
    return {"righe": rows}


def summarize(rows: list[dict]) -> dict:
    report = {}
    for part in ("sintetiche", "pose", "mani", "chiralità"):
        for qid in dict.fromkeys(r["domanda"] for r in rows if r["parte"] == part):
            items = [r for r in rows if r["parte"] == part and r["domanda"] == qid]
            graded = [r for r in items if r["vero"] is not None]
            pairs = {}
            for r in items:
                pairs.setdefault(r["caso"], {})[r["specchiata"]] = r["scelta"]
            # specchiando in orizzontale destra e sinistra si invertono, sopra e sotto no
            same = qid.endswith(SAME_UNDER_MIRROR)
            flips = [(p[False] == p[True]) if same else (OTHER.get(p[False]) == p[True]) for p in pairs.values() if len(p) == 2]
            entry = {
                "domande": len(items),
                "accuratezza": round(float(np.mean([r["scelta"] == r["vero"] for r in graded])), 3) if graded else None,
                "coerente_con_lo_specchio": round(float(np.mean(flips)), 3),
                "sceglie_sinistra_o_sopra": round(float(np.mean([r["scelta"] in ("sinistra", "sopra") for r in items])), 3),
            }
            if items[0].get("cornice") == "persona":
                # confusione delle cornici: la risposta "della persona" è il lato nell'immagine
                facing = [r for r in items if r["vista"] == "fronte"]
                entry["di_fronte_risponde_col_lato_immagine"] = round(float(np.mean(
                    [r["scelta"] == r["lato_immagine"] for r in facing])), 3)
            report[f"{part} · {qid}"] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--scene", type=int, default=48, help="scene sintetiche (metà orizzontali, metà verticali)")
    args = parser.parse_args()
    scorer = DecisionScorer(args.model, vision=True)
    started = time.perf_counter()
    data = run(scorer, args.scene)
    report = {"modello": args.model, "lato_immagini": SIDE, "riepilogo": summarize(data["righe"]),
              "secondi": round(time.perf_counter() - started, 1), **data}
    out = Path("runs/destra_sinistra") / f"{args.model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\n{args.model} (accuratezza, coerenza con lo specchio, quota «sinistra»/«sopra»)")
    for key, entry in report["riepilogo"].items():
        extra = (f"  di fronte risponde col lato dell'immagine: {entry['di_fronte_risponde_col_lato_immagine']:.2f}"
                 if "di_fronte_risponde_col_lato_immagine" in entry else "")
        accuracy = f"{entry['accuratezza']:.3f}" if entry["accuratezza"] is not None else "  -  "
        print(f"  {key:28s} n={entry['domande']:3d}  acc {accuracy}  specchio {entry['coerente_con_lo_specchio']:.2f}  "
              f"sinistra/sopra {entry['sceglie_sinistra_o_sopra']:.2f}{extra}")
    print(f"\nrapporto: {out} ({report['secondi']} s)")


if __name__ == "__main__":
    main()
