"""Verifica delle primitive zero-shot con casi a risposta nota e soglie decise in anticipo.

    .venv/bin/python scripts/verifica_primitive.py Qwen/Qwen3.5-2B-Base

Si tiene una primitiva solo se supera la sua soglia (documentazione/05-primitive-di-lettura.md).
`surprise` (perplessità dello stato come segnale di anomalia) non ha superato la verifica
(AUROC 0.50 su 6 testi normali contro 6 anomali) ed è stata tolta.
"""

import json
import sys
import unicodedata

from egeria.scorer import DecisionScorer

TICKET = json.load(open("examples/ticket_it.json"))["state"]

PRIORITA = [  # (stato, domanda, opzioni, scelta attesa): "cosa va fatto per primo" come choice
    (TICKET, "Quale azione va fatta per prima?",
     {"riparare_pagamenti": "Risolvere il problema dei pagamenti", "offrire_sconto": "Offrire uno sconto commerciale",
      "inviare_newsletter": "Inviare la newsletter mensile"}, "riparare_pagamenti"),
    ("Allarme: fumo nero dal magazzino B, le fiamme sono visibili dalla strada.", "Quale azione va fatta per prima?",
     {"aprire_ticket": "Aprire un ticket di manutenzione", "chiamare_vigili": "Chiamare i vigili del fuoco",
      "ignorare": "Ignorare l'allarme"}, "chiamare_vigili"),
    ("Ciao Marco, ti confermo la riunione di domani alle 10 in sala B. Giulia", "Come rispondere a questa email?",
     {"segnalare_phishing": "Segnalarla come phishing", "inoltrare_legale": "Inoltrarla all'ufficio legale",
      "confermare": "Rispondere confermando la presenza"}, "confermare"),
    ("Il server di produzione è irraggiungibile da 20 minuti e i clienti non riescono a pagare.",
     "Quale attività ha la priorità?",
     {"documentazione": "Aggiornare la documentazione", "ripristino": "Ripristinare il server",
      "riunione": "Pianificare una riunione per la prossima settimana"}, "ripristino"),
]

NUMBER = [  # (stato, domanda, estremi, unità, intervallo atteso)
    (TICKET, "Da quanti giorni falliscono i pagamenti?", [0, 1, 2, 4, 7, None], "giorni", "2-4"),
    ("Riunione di progetto: 12 persone in sala A e altre 3 collegate da remoto.",
     "Quante persone partecipano in totale, contando anche chi è collegato?", [0, 5, 10, 14, 16, 20, None], "persone", "14-16"),
    ("Il preventivo per la ristrutturazione è di 1.250 euro più IVA.", "A quanto ammonta il preventivo, IVA esclusa?",
     [0, 500, 1000, 1500, 2000, None], "euro", "1000-1500"),
    ("Il pacco pesa circa 7 kg e misura 40x30x20 cm.", "Quanto pesa il pacco?", [0, 2, 5, 10, 20, None], "kg", "5-10"),
    ("Il corriere ha scritto che la consegna arriverà tra 45 minuti.", "Tra quanto arriva la consegna?",
     [0, 15, 30, 60, 120, None], "minuti", "30-60"),
]

OPEN = [  # (stato, domanda, risposte accettate)
    (TICKET, "Entro quale giorno della settimana vanno pagati gli stipendi?", ["venerdi", "friday"]),
    (TICKET, "In che lingua è scritto il messaggio?", ["italiano", "italian"]),
    ("Il volo AZ610 parte da Roma alle 18:40 ed è diretto a New York.", "Da quale città parte il volo?", ["roma", "rome"]),
    ("Mario ha ordinato tre biciclette rosse per i suoi figli.", "Di che colore sono le biciclette?", ["rosse", "rosso", "red"]),
    ("Il contratto di affitto scade il 31 dicembre e va rinnovato entro novembre.", "In quale mese scade il contratto?",
     ["dicembre", "december"]),
    ("Buongiorno, sono il dottor Bianchi, primario del reparto di cardiologia.", "Qual è il cognome di chi scrive?",
     ["bianchi"]),
]

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().strip(" .,!?\"'"))
    return "".join(c for c in text if not unicodedata.combining(c))


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B-Base"
    scorer = DecisionScorer(model)
    report = {}

    ok = 0
    for state, question, criteria, expected in PRIORITA:
        result = scorer.decide({"state": state, "questions": {"q": {"type": "choice", "instructions": question,
                                                                      "criteria": criteria}}}, permutations=2)
        chosen = result["answers"]["q"]["choice"]
        ok += chosen == expected
        print(f"prima  {'✓' if chosen == expected else '✗'} {question[:45]:45s} -> {chosen}")
    report["priorita"] = (ok, len(PRIORITA), ok >= 3)

    ok = 0
    for state, question, edges, unit, expected in NUMBER:
        result = scorer.decide({"state": state, "questions": {"q": {"type": "number", "instructions": question,
                                                                      "criteria": {"bins": edges, "unit": unit}}}},
                               permutations=2)["answers"]["q"]
        ok += result["range"] == expected
        print(f"number {'✓' if result['range'] == expected else '✗'} {question[:45]:45s} -> {result['range']:10s} "
              f"(atteso {expected}, valore {result['value']:.1f})")
    report["number"] = (ok, len(NUMBER), ok >= 4)

    ok = 0
    for state, question, accepted in OPEN:
        result = scorer.decide({"state": state, "questions": {"q": {"type": "open", "instructions": question,
                                                                      "top_k": 3}}})["answers"]["q"]
        good = normalize(result["answer"]) in accepted
        ok += good
        print(f"open   {'✓' if good else '✗'} {question[:45]:45s} -> {result['answer']!r} (p={result['confidence']:.2f})")
    report["open"] = (ok, len(OPEN), ok >= 5)

    print("\nesito:")
    for name, (value, total, passed) in report.items():
        print(f"  {name:8s} {value}/{total}  {'TENERE' if passed else 'TOGLIERE'}")


if __name__ == "__main__":
    main()
