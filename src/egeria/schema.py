"""Validazione delle richieste compatibili con l'API Jev (`POST /v1/systemone`).

Una richiesta ha uno `state` (stringa, oggetto o array JSON) e una mappa
`questions` id -> domanda. Ogni domanda ha `type`, `instructions` e, a seconda
del tipo, `criteria`.

Tipi di Jev: noul, choice, score. Estensioni Egeria (senza training):
- rank: ordinamento completo delle opzioni;
- number: stima numerica a intervalli (valore atteso + intervallo);
- open: risposta di una parola dall'intero vocabolario;
- recall: "cosa ti ricorda?", i ricordi più simili allo stato (richiede una memoria).

Altre estensioni: `min_confidence` per domanda o globale (profondità dinamica);
`embed` a livello di richiesta (vettore semantico dello stato).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score", "rank", "number", "open", "recall")
# Readout usato per ogni tipo: temperature, ordinamenti e formula di confidenza.
READOUT = {"noul": "noul", "choice": "choice", "rank": "choice", "score": "score", "number": "score",
           "open": "open", "recall": "memory"}
# Il readout usa una lettera maiuscola per opzione, ognuna un singolo token.
MAX_OPTIONS = 26
MAX_SCORE_LEVELS = 10
MAX_TOP_K = 20
MAX_RECALL = 10

DEFAULT_NOUL_CRITERIA = {
    "true": "The statement holds for the state.",
    "false": "The statement does not hold for the state.",
}


class RequestError(ValueError):
    """Richiesta non valida (l'equivalente di un 422 dell'API Jev)."""


@dataclass(frozen=True)
class Option:
    key: str
    description: str | None


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    options: tuple[Option, ...]
    # Confidenza minima, sulla stessa scala del campo `confidence` della risposta
    # (corretta per il caso: 0 = a caso, 1 = certezza). Con la profondità dinamica il
    # modello esce al primo blocco che la raggiunge; se nemmeno l'ultimo layer ci
    # arriva, la risposta è marcata come incerta.
    min_confidence: float | None = None
    # Parametri specifici del tipo: number {edges, unit}; open {top_k}.
    params: dict = field(default_factory=dict, compare=False)

    @property
    def keys(self) -> list[str]:
        return [option.key for option in self.options]

    @property
    def readout(self) -> str:
        return READOUT[self.type]


def render_value(value: Any) -> str:
    """Le stringhe restano tali, il resto diventa JSON compatto."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _description(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, dict, list)):
        return render_value(value)
    raise RequestError(f"{where}: la descrizione deve essere stringa, oggetto, array o null")


def _min_confidence(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
        raise RequestError(f"{where}: min_confidence deve essere un numero in (0, 1]")
    return float(value)


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _interval_key(lo: float | None, hi: float | None) -> str:
    if lo is None:
        return f"<{_number(hi)}"
    if hi is None:
        return f">={_number(lo)}"
    return f"{_number(lo)}-{_number(hi)}"


def _interval_text(lo: float | None, hi: float | None, unit: str) -> str:
    suffix = f" {unit}" if unit else ""
    if lo is None:
        return f"less than {_number(hi)}{suffix}"
    if hi is None:
        return f"{_number(lo)}{suffix} or more"
    return f"from {_number(lo)} to {_number(hi)}{suffix}"


def _number_edges(qid: str, criteria: Any) -> tuple[list[float | None], str]:
    """`criteria` di number: lista di estremi crescenti, oppure {"bins": [...], "unit": "..."}.

    Il primo estremo può essere null (aperto in basso), l'ultimo null (aperto in alto).
    """
    unit = ""
    if isinstance(criteria, dict):
        unit = criteria.get("unit", "")
        criteria = criteria.get("bins")
        if not isinstance(unit, str):
            raise RequestError(f"domanda {qid!r}: unit deve essere una stringa")
    if not isinstance(criteria, list) or not 3 <= len(criteria) <= MAX_OPTIONS + 1:
        raise RequestError(f"domanda {qid!r}: number richiede da 3 a {MAX_OPTIONS + 1} estremi (almeno 2 intervalli)")
    for position, value in enumerate(criteria):
        if value is None and 0 < position < len(criteria) - 1:
            raise RequestError(f"domanda {qid!r}: solo il primo e l'ultimo estremo possono essere null")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise RequestError(f"domanda {qid!r}: gli estremi devono essere numeri finiti")
    finite = [value for value in criteria if value is not None]
    if not finite or any(b <= a for a, b in zip(finite, finite[1:])):
        raise RequestError(f"domanda {qid!r}: gli estremi devono essere strettamente crescenti")
    return list(criteria), unit


def parse_question(qid: str, body: Any, default_min_confidence: float | None = None) -> Question:
    if not isinstance(body, dict):
        raise RequestError(f"domanda {qid!r}: deve essere un oggetto")
    qtype = body.get("type")
    if qtype not in QUESTION_TYPES:
        raise RequestError(f"domanda {qid!r}: type deve essere uno di {QUESTION_TYPES}")
    instructions = body.get("instructions")
    if qtype == "recall" and instructions in (None, "", [], {}):
        instructions = "What does this remind you of?"  # il recupero usa lo stato, non il testo della domanda
    if instructions in (None, "", [], {}):
        raise RequestError(f"domanda {qid!r}: instructions mancante")
    criteria = body.get("criteria")
    params: dict = {}

    if qtype == "noul":
        merged = dict(DEFAULT_NOUL_CRITERIA)
        if criteria is not None:
            if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                raise RequestError(f"domanda {qid!r}: criteria di noul ammette solo le chiavi true/false")
            for key, value in criteria.items():
                description = _description(value, f"domanda {qid!r}")
                if description:
                    merged[key] = description
        options = (Option("true", merged["true"]), Option("false", merged["false"]))
    elif qtype in ("choice", "rank"):
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise RequestError(f"domanda {qid!r}: {qtype} richiede criteria con almeno 2 opzioni")
        if len(criteria) > MAX_OPTIONS:
            raise RequestError(f"domanda {qid!r}: al massimo {MAX_OPTIONS} opzioni (sono {len(criteria)})")
        options = tuple(
            Option(str(key), _description(value, f"domanda {qid!r}, opzione {key!r}"))
            for key, value in criteria.items()
        )
    elif qtype == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise RequestError(f"domanda {qid!r}: score richiede da 2 a {MAX_SCORE_LEVELS} livelli")
        options = tuple(
            Option(str(level), _description(value, f"domanda {qid!r}, livello {level}"))
            for level, value in enumerate(criteria)
        )
    elif qtype == "number":
        edges, unit = _number_edges(qid, criteria)
        params = {"edges": edges, "unit": unit}
        options = tuple(
            Option(_interval_key(lo, hi), _interval_text(lo, hi, unit)) for lo, hi in zip(edges[:-1], edges[1:])
        )
    elif qtype == "recall":
        k = body.get("k", 3)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_RECALL:
            raise RequestError(f"domanda {qid!r}: k deve essere un intero fra 1 e {MAX_RECALL}")
        params = {"k": k}
        options = ()
    else:  # open
        if criteria is not None:
            raise RequestError(f"domanda {qid!r}: open non ammette criteria")
        top_k = body.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise RequestError(f"domanda {qid!r}: top_k deve essere un intero fra 1 e {MAX_TOP_K}")
        params = {"top_k": top_k}
        options = ()
    # Chiave presente (anche null) = scelta esplicita della domanda; assente = default ereditato.
    if "min_confidence" in body:
        threshold = _min_confidence(body["min_confidence"], f"domanda {qid!r}")
    else:
        threshold = default_min_confidence
    return Question(qid, qtype, render_value(instructions), options, threshold, params)


def parse_request(body: Any, default_min_confidence: float | None = None) -> tuple[Any, list[Question]]:
    """Valida un body `/v1/systemone` e restituisce (state, domande).

    Soglia `min_confidence` di ogni domanda, dalla più specifica alla più generale:
    1. `min_confidence` della domanda (`null` esplicito = profondità completa);
    2. `min_confidence` globale della richiesta (`null` esplicito = profondità completa);
    3. `default_min_confidence` del server;
    4. nessuna soglia: default prudente, modello completo senza uscite anticipate.
    """
    if not isinstance(body, dict):
        raise RequestError("il body deve essere un oggetto JSON")
    state = body.get("state")
    if state in (None, "", [], {}):
        raise RequestError("state mancante")
    if not isinstance(state, (str, dict, list)):
        raise RequestError("state deve essere stringa, oggetto o array")
    if is_multimodal(state):
        validate_parts(state)
    questions = body.get("questions", {})
    analysis = state_analysis(body)
    if not isinstance(questions, dict) or not (questions or any(analysis.values())):
        raise RequestError("questions deve essere una mappa non vuota id -> domanda (o chiedere embed)")
    if "min_confidence" in body:
        default = _min_confidence(body["min_confidence"], "richiesta")
    else:
        default = _min_confidence(default_min_confidence, "default del server")
    return state, [parse_question(str(qid), question, default) for qid, question in questions.items()]


IMAGE_SOURCES = ("path", "url", "base64")


def is_multimodal(state: Any) -> bool:
    """Lo stato è una lista di parti con almeno un'immagine: [{"type": "image", "path": ...}, ...]."""
    return isinstance(state, list) and any(isinstance(p, dict) and p.get("type") == "image" for p in state)


def validate_parts(state: list) -> None:
    for number, part in enumerate(state):
        if not isinstance(part, dict) or part.get("type") not in ("text", "image"):
            raise RequestError(f"state[{number}]: ogni parte deve essere {{'type': 'text'|'image', ...}}")
        if part["type"] == "text" and not isinstance(part.get("text"), str):
            raise RequestError(f"state[{number}]: una parte di testo richiede 'text' stringa")
        if part["type"] == "image" and sum(isinstance(part.get(k), str) for k in IMAGE_SOURCES) != 1:
            raise RequestError(f"state[{number}]: un'immagine richiede uno solo fra {IMAGE_SOURCES}")


def image_max_side(body: dict) -> int | None:
    value = body.get("image_max_side")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 64 <= value <= 4096:
        raise RequestError("image_max_side deve essere un intero fra 64 e 4096")
    return value


def state_analysis(body: dict) -> dict[str, bool]:
    """Opzioni di analisi dello stato a livello di richiesta: `embed`."""
    result = {}
    for key in ("embed",):
        value = body.get(key, False)
        if not isinstance(value, bool):
            raise RequestError(f"{key} deve essere true o false")
        result[key] = value
    return result



def memory_options(body: dict) -> dict:
    """`"memory": {"recall": k, "min_similarity": x, "vote": true, "inject": false}`.

    - recall: quanti ricordi restituire in output;
    - vote: voto dei ricordi per le domande che coprono (sui 10 più simili);
    - inject: mettere i ricordi nel prompt. Spento di default: zero-shot non migliora le decisioni.
    """
    options = body.get("memory", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise RequestError("memory deve essere un oggetto")
    recall = options.get("recall", 0)
    if isinstance(recall, bool) or not isinstance(recall, int) or not 0 <= recall <= MAX_RECALL:
        raise RequestError(f"memory.recall deve essere un intero fra 0 e {MAX_RECALL}")
    threshold = options.get("min_similarity")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not -1.0 <= threshold <= 1.0):
        raise RequestError("memory.min_similarity deve essere un numero fra -1 e 1")
    flags = {}
    for key, default in (("vote", True), ("inject", False)):
        value = options.get(key, default)
        if not isinstance(value, bool):
            raise RequestError(f"memory.{key} deve essere true o false")
        flags[key] = value
    return {"recall": recall, "min_similarity": threshold, **flags}


MAX_PROMPT_MEMORIES = 10
MAX_MEMORY_CHARS = 4000


def prompt_memories(body: dict) -> list[str] | None:
    """`"memories": ["...", ...]`: ricordi già in forma di testo da mettere nel prompt.

    Li usa il server web quando la memoria sta da lui e il modello è un servizio separato.
    """
    value = body.get("memories")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > MAX_PROMPT_MEMORIES or not all(isinstance(v, str) for v in value):
        raise RequestError(f"memories deve essere una lista di al massimo {MAX_PROMPT_MEMORIES} testi")
    return [v[:MAX_MEMORY_CHARS] for v in value] or None
