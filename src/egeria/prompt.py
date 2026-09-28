"""Costruzione dei prompt con opzioni etichettate da lettere (readout stile SemIf).

Ogni domanda diventa un prompt indipendente: stato, domanda, opzioni
etichettate A, B, C, ... Il modello non genera nulla: si leggono i logit delle
lettere nell'ultima posizione, dopo il prompt di generazione dell'assistente.
"""

from __future__ import annotations

import string

from .schema import Question, is_multimodal, render_value

LETTERS = string.ascii_uppercase
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def render_state(state) -> str:
    """Testo dello stato; le immagini diventano il segnaposto che il processore espande in token visivi."""
    if is_multimodal(state):
        return "\n".join(part["text"] if part["type"] == "text" else IMAGE_PLACEHOLDER for part in state)
    return render_value(state)

SYSTEM_PROMPT = (
    "You are a decision engine. Read the state, apply the question to it, and choose "
    "exactly one of the listed options. Answer with only the uppercase letter of the "
    "chosen option, with no explanation or reasoning."
)

OPEN_SYSTEM_PROMPT = (
    "You are a decision engine. Read the state and answer the question with a single word, "
    "with no explanation or reasoning."
)

TYPE_HINTS = {
    "noul": "Decide whether the statement or question below holds for the state.",
    "choice": "Choose the single option that best answers the question for the state.",
    "rank": "Choose the single option that best answers the question for the state.",
    "score": "Place the state on the ordered scale below, from the lowest level to the highest.",
    "number": "Estimate the quantity asked and choose the range that contains it.",
}

ORDINAL = {"score", "number"}


def orderings(question: Question, permutations: int) -> list[list[int]]:
    """Ordini di presentazione delle opzioni, come indici nell'ordine originale.

    Per noul e choice si usano rotazioni cicliche equispaziate, così ogni opzione
    occupa posizioni diverse. Per score l'ordine della scala conta, quindi si usa al
    massimo l'ordine originale e quello inverso.
    """
    n = len(question.options)
    if permutations < 1:
        raise ValueError("permutations deve essere >= 1")
    identity = list(range(n))
    if question.type in ORDINAL:
        return [identity] if permutations == 1 else [identity, identity[::-1]]
    count = min(permutations, n)
    result = []
    for k in range(count):
        shift = round(k * n / count)
        result.append([(i + shift) % n for i in range(n)])
    return result


def _option_line(letter: str, question: Question, index: int) -> str:
    option = question.options[index]
    if question.type == "number":
        return f"{letter}. {option.description}"
    if question.type == "score":
        label = f"level {option.key} of {len(question.options) - 1}"
    else:
        label = option.key
    if option.description:
        return f"{letter}. {label}: {option.description}"
    return f"{letter}. {label}"


MEMORY_HEADER = (
    "Past cases similar to the current state, with the decisions taken. "
    "Use them only as reference: evaluate the current state."
)


def memory_block(memories: list[str] | None) -> list[str]:
    """Ricordi richiamati dalla memoria, prima dello stato corrente."""
    if not memories:
        return []
    lines = ["<memories>", MEMORY_HEADER]
    for number, text in enumerate(memories, 1):
        lines += [f"[{number}]", text]
    return lines + ["</memories>", ""]


def user_content(state, question: Question, order: list[int], memories: list[str] | None = None) -> str:
    lines = memory_block(memories) + [
        "<state>",
        render_state(state),
        "</state>",
        "",
        TYPE_HINTS[question.type],
        f"Question: {question.instructions}",
        "",
        "Options:",
    ]
    lines += [_option_line(LETTERS[slot], question, index) for slot, index in enumerate(order)]
    return "\n".join(lines)


def build_messages(state, question: Question, order: list[int], memories: list[str] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(state, question, order, memories)},
    ]


def build_plain(state, question: Question, order: list[int], memories: list[str] | None = None) -> str:
    """Formato senza chat template, per i modelli Base."""
    return f"{SYSTEM_PROMPT}\n\n{user_content(state, question, order, memories)}\n\nAnswer:"


def open_content(state, question: Question, memories: list[str] | None = None) -> str:
    return "\n".join(memory_block(memories) + [
        "<state>", render_state(state), "</state>", "", f"Question: {question.instructions}", "",
        "Answer with a single word.",
    ])


def build_open_messages(state, question: Question, memories: list[str] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": OPEN_SYSTEM_PROMPT},
        {"role": "user", "content": open_content(state, question, memories)},
    ]


def build_state_messages(state) -> list[dict]:
    """Solo lo stato, per embed."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(["<state>", render_state(state), "</state>"])},
    ]
