import math

import numpy as np
import pytest

from egeria.calibration import fit_temperature, mean_nll
from egeria.confidence import answer, choice_confidence, score_confidence, softmax
from egeria.datasets import gold_vector
from egeria.metrics import coverage_at_error, ece, flip_rate, summarize
from egeria.prompt import LETTERS, build_messages, orderings
from egeria.schema import MAX_OPTIONS, RequestError, parse_request

JEV_EXAMPLE = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


# ------------------------------------------------------------------ schema


def test_parse_jev_example():
    state, questions = parse_request(JEV_EXAMPLE)
    assert state.startswith("Help!")
    by_id = {q.id: q for q in questions}
    assert by_id["is_urgent"].keys == ["true", "false"]
    assert by_id["department"].keys == ["billing", "technical", "sales"]
    assert by_id["frustration"].keys == ["0", "1", "2"]
    assert by_id["frustration"].options[2].description == "Very angry"


def test_noul_default_criteria():
    _, [question] = parse_request({"state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    assert all(option.description for option in question.options)


def test_structured_instructions_and_state():
    body = {"state": {"a": 1}, "questions": {"q": {"type": "noul", "instructions": {"check": "field `a`"}}}}
    state, [question] = parse_request(body)
    assert state == {"a": 1}
    assert question.instructions == '{"check":"field `a`"}'


@pytest.mark.parametrize(
    "body",
    [
        {"questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"q": {"type": "maybe", "instructions": "x"}}},
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}}},
        {"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}},
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}},
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": ""}}},
    ],
)
def test_invalid_requests(body):
    with pytest.raises(RequestError):
        parse_request(body)


def test_too_many_options():
    criteria = {f"o{i}": None for i in range(MAX_OPTIONS + 1)}
    with pytest.raises(RequestError):
        parse_request({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": criteria}}})


# ------------------------------------------------------------------ prompt


def test_orderings_rotate_choice_and_reverse_score():
    _, questions = parse_request(JEV_EXAMPLE)
    by_id = {q.id: q for q in questions}
    assert orderings(by_id["department"], 1) == [[0, 1, 2]]
    rotations = orderings(by_id["department"], 3)
    assert rotations == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    # ogni opzione occupa ogni posizione una volta
    assert all(sorted(column) == [0, 1, 2] for column in zip(*rotations))
    assert orderings(by_id["is_urgent"], 2) == [[0, 1], [1, 0]]
    assert orderings(by_id["frustration"], 3) == [[0, 1, 2], [2, 1, 0]]


def test_messages_list_options_with_letters():
    state, questions = parse_request(JEV_EXAMPLE)
    department = next(q for q in questions if q.id == "department")
    content = build_messages(state, department, [2, 0, 1])[1]["content"]
    assert "A. sales: Pricing, upgrades, new accounts" in content
    assert "C. technical: Bugs, outages, integrations" in content
    assert state in content
    assert len(LETTERS) == MAX_OPTIONS


# -------------------------------------------------------------- confidence


def test_jev_confidence_examples():
    # Dalla documentazione Jev: billing 0.88 su 3 opzioni -> ~0.81; score [0, 0.95, 0.05] -> 0.92
    assert choice_confidence([0.875, 0.125, 0.0]) == pytest.approx(0.8125)
    assert score_confidence([0.0, 0.95, 0.05]) == pytest.approx(0.925)
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)


def test_answer_format():
    _, questions = parse_request(JEV_EXAMPLE)
    by_id = {q.id: q for q in questions}
    assert answer(by_id["is_urgent"], [0.9, 0.1]) == {"type": "noul", "noul": 0.9, "confidence": 0.8}
    choice = answer(by_id["department"], [0.2, 0.7, 0.1])
    assert choice["choice"] == "technical" and set(choice["probabilities"]) == {"billing", "technical", "sales"}
    score = answer(by_id["frustration"], [0.0, 0.95, 0.05])
    assert score["score"] == pytest.approx(1.05)
    assert score["legend"]["2"] == "Very angry"


# ------------------------------------------------------------- calibration


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    logits, targets = [], []
    for _ in range(2000):
        true = rng.normal(size=4)
        label = rng.choice(4, p=softmax(true))
        logits.append(true * 3.0)  # modello 3 volte troppo sicuro
        targets.append(np.eye(4)[label])
    temperature = fit_temperature(logits, targets)
    assert 2.5 < temperature < 3.5
    assert mean_nll(logits, targets, 1 / temperature) < mean_nll(logits, targets, 1.0)


def test_temperature_preserves_argmax():
    z = np.array([0.3, 2.0, -1.0])
    assert np.argmax(softmax(z, 0.2)) == np.argmax(softmax(z, 5.0))


# ----------------------------------------------------------------- metrics


def test_summarize_perfect_and_uniform():
    perfect = [{"type": "choice", "probs": [1.0, 0.0], "gold": [1.0, 0.0], "label": 0}] * 4
    result = summarize(perfect)
    assert result["accuracy"] == 1.0 and result["brier"] == pytest.approx(0.0, abs=1e-9)
    assert result["coverage_at_5"] == 1.0
    uniform = [{"type": "noul", "probs": [0.5, 0.5], "gold": [0.0, 1.0], "label": 1}]
    assert summarize(uniform)["nll"] == pytest.approx(math.log(2))
    assert summarize(uniform)["brier"] == pytest.approx(0.5)


def test_score_metrics():
    record = {"type": "score", "probs": [0.0, 1.0, 0.0], "gold": [0.0, 0.0, 1.0], "label": 2}
    result = summarize([record])
    assert result["score_mae"] == pytest.approx(1.0)
    assert result["within_1"] == 1.0


def test_ece_and_coverage():
    assert ece([0.9, 0.9], [1, 0]) == pytest.approx(0.4)
    assert coverage_at_error([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1], 0.05) == pytest.approx(0.5)
    assert flip_rate([{"order_argmax": [0, 0]}, {"order_argmax": [0, 1]}]) == 0.5


def test_gold_vector_uses_question_order():
    _, questions = parse_request(JEV_EXAMPLE)
    department = next(q for q in questions if q.id == "department")
    gold = {"label": "sales", "probabilities": {"sales": 0.6, "billing": 0.4}}
    vector, label = gold_vector(department, gold)
    assert vector == [0.4, 0.0, 0.6] and label == 2


# ------------------------------------------------------ min_confidence


def _thresholds(body, server_default=None):
    _, questions = parse_request(body, server_default)
    return {q.id: q.min_confidence for q in questions}


def test_min_confidence_precedence():
    body = {
        "state": "s",
        "questions": {
            "propria": {"type": "noul", "instructions": "x", "min_confidence": 0.95},
            "forzata_completa": {"type": "noul", "instructions": "x", "min_confidence": None},
            "eredita": {"type": "noul", "instructions": "x"},
        },
    }
    # Nessuna soglia da nessuna parte: default prudente (None = modello completo).
    assert _thresholds(body) == {"propria": 0.95, "forzata_completa": None, "eredita": None}
    # Default del server.
    assert _thresholds(body, 0.7)["eredita"] == 0.7
    # Il campo globale della richiesta sovrascrive il default del server.
    assert _thresholds({**body, "min_confidence": 0.8}, 0.7) == {
        "propria": 0.95, "forzata_completa": None, "eredita": 0.8,
    }
    # null globale esplicito: profondità completa anche se il server ha un default.
    assert _thresholds({**body, "min_confidence": None}, 0.7)["eredita"] is None


@pytest.mark.parametrize("value", [0, 1.5, -0.1, "0.9", True])
def test_min_confidence_validation(value):
    with pytest.raises(RequestError):
        parse_request({"state": "s", "min_confidence": value, "questions": {"q": {"type": "noul", "instructions": "x"}}})


def test_decision_confidence_is_chance_corrected():
    from egeria.confidence import decision_confidence

    # Una noul a 0.5 è a caso: confidenza 0, quindi nessuna soglia > 0 la fa uscire.
    assert decision_confidence("noul", [0.5, 0.5]) == pytest.approx(0.0)
    assert decision_confidence("noul", [0.9, 0.1]) == pytest.approx(0.8)
    assert decision_confidence("choice", [0.25] * 4) == pytest.approx(0.0)
    assert decision_confidence("choice", [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert decision_confidence("score", [0.0, 1.0, 0.0]) == pytest.approx(1.0)


# ------------------------------------------------------- nuove primitive


def _one(question_body):
    _, [question] = parse_request({"state": "s", "questions": {"q": question_body}})
    return question


def test_new_types_parse():
    rank = _one({"type": "rank", "instructions": "x", "criteria": {"a": None, "b": None}})
    assert rank.readout == "choice"
    number = _one({"type": "number", "instructions": "x", "criteria": {"bins": [0, 1, 3, None], "unit": "persone"}})
    assert number.keys == ["0-1", "1-3", ">=3"] and number.params["unit"] == "persone"
    assert number.options[2].description == "3 persone or more"
    opened = _one({"type": "open", "instructions": "x", "top_k": 3})
    assert opened.options == () and opened.params == {"top_k": 3}


@pytest.mark.parametrize(
    "body",
    [
        {"type": "number", "instructions": "x", "criteria": [0, 1]},  # un solo intervallo
        {"type": "number", "instructions": "x", "criteria": [0, 5, 3]},  # non crescente
        {"type": "number", "instructions": "x", "criteria": [0, None, 3]},  # null interno
        {"type": "open", "instructions": "x", "criteria": {"a": None}},
        {"type": "open", "instructions": "x", "top_k": 0},
        {"type": "multi", "instructions": "x", "criteria": {"a": None, "b": None}},  # tipo rimosso
    ],
)
def test_new_types_validation(body):
    with pytest.raises(RequestError):
        parse_request({"state": "s", "questions": {"q": body}})


def test_embed_only_request_needs_no_questions():
    from egeria.schema import state_analysis

    body = {"state": "s", "embed": True}
    state, questions = parse_request(body)
    assert questions == [] and state_analysis(body) == {"embed": True}
    with pytest.raises(RequestError):
        parse_request({"state": "s"})


def test_rank_and_number_answers():
    rank = _one({"type": "rank", "instructions": "x", "criteria": {"a": None, "b": None, "c": None}})
    assert answer(rank, [0.2, 0.5, 0.3])["ranking"] == ["b", "c", "a"]
    number = _one({"type": "number", "instructions": "x", "criteria": [0, 10, 20]})
    result = answer(number, [0.5, 0.5])
    assert result["value"] == pytest.approx(10.0)  # media dei punti medi 5 e 15
    assert result["interval"] == [pytest.approx(2.0), pytest.approx(18.0)]  # quantili 10% e 90%
    assert result["range"] == "0-10"


def test_open_answers():
    from egeria.confidence import merge_candidates, open_answer

    merged = merge_candidates([(" Bianca", 0.5), ("bianca", 0.2), (" ", 0.1), ("grigia", 0.1)], top_k=2)
    assert merged == [{"text": "Bianca", "p": 0.7}, {"text": "grigia", "p": 0.1}]
    assert open_answer(merged)["answer"] == "Bianca"


# ---------------------------------------------------------------- memoria


def test_memory_store_search_save_load(tmp_path):
    from egeria.memory import Memory, MemoryStore

    store = MemoryStore()
    store.add([1.0, 0.0, 0.0], Memory("a", "stato a", {"q": "x"}))
    store.add([0.9, 0.1, 0.0], Memory("b", "stato b", {"q": "y"}, note="confermato"))
    store.add([0.0, 1.0, 0.0], Memory("c", "stato c"))
    found = store.search([1.0, 0.05, 0.0], k=2)
    assert [item.id for _, item in found] == ["a", "b"]
    assert [item.id for _, item in store.search([1.0, 0.05, 0.0], k=2, exclude={"a"})] == ["b", "c"]
    assert store.search([1.0, 0.0, 0.0], k=3, min_similarity=0.5)[-1][1].id == "b"
    store.save(tmp_path / "m")
    again = MemoryStore.load(tmp_path / "m")
    assert len(again) == 3 and again.items[1].note == "confermato"
    assert "Decisions: q = y" in again.items[1].render() and "Note: confermato" in again.items[1].render()
    with pytest.raises(ValueError):
        store.add([0.0, 0.0, 1.0], Memory("a", "doppione"))


def test_memory_block_in_prompt_and_options():
    from egeria.prompt import build_messages
    from egeria.schema import memory_options

    state, questions = parse_request(JEV_EXAMPLE)
    content = build_messages(state, questions[0], [0, 1], memories=["State: vecchio caso\nDecisions: q = x"])[1]["content"]
    assert content.index("<memories>") < content.index("<state>") and "[1]\nState: vecchio caso" in content
    assert memory_options({"memory": {"recall": 3}}) == {"recall": 3, "min_similarity": None, "vote": True, "inject": False}
    with pytest.raises(RequestError):
        memory_options({"memory": {"recall": 50}})


# --------------------------------------------------------------- immagini


def test_multimodal_state_parts_and_prompt(tmp_path):
    from PIL import Image

    from egeria.images import load_images
    from egeria.prompt import IMAGE_PLACEHOLDER, build_messages
    from egeria.schema import image_max_side, is_multimodal

    path = tmp_path / "rosso.png"
    Image.new("RGB", (800, 400), "red").save(path)
    body = {
        "state": [{"type": "text", "text": "Foto:"}, {"type": "image", "path": str(path)}],
        "image_max_side": 200,
        "questions": {"q": {"type": "noul", "instructions": "È rossa?"}},
    }
    state, [question] = parse_request(body)
    assert is_multimodal(state) and image_max_side(body) == 200
    content = build_messages(state, question, [0, 1])[1]["content"]
    assert f"<state>\nFoto:\n{IMAGE_PLACEHOLDER}\n</state>" in content
    [image] = load_images(state, 200)
    assert image.size == (200, 100)


@pytest.mark.parametrize(
    "state",
    [
        [{"type": "image"}],  # nessuna sorgente
        [{"type": "image", "path": "a.png", "url": "http://x"}],  # due sorgenti
        [{"type": "image", "path": "a.png"}, {"type": "video", "path": "b.mp4"}],
        [{"type": "image", "path": "a.png"}, {"type": "text"}],
    ],
)
def test_multimodal_state_validation(state):
    with pytest.raises(RequestError):
        parse_request({"state": state, "questions": {"q": {"type": "noul", "instructions": "x"}}})


def test_memory_votes_cover_only_known_questions():
    from egeria.memory import Memory, memory_key, memory_votes

    state, questions = parse_request(JEV_EXAMPLE)
    by_id = {q.id: q for q in questions}
    found = [(0.9, Memory("a", "s", {"department": "billing", "frustration": "level 2 of 2"})),
             (0.8, Memory("b", "s", {"department": "billing", "frustration": "1"})),
             (0.7, Memory("c", "s", {"department": "sconosciuto"}))]
    assert memory_key(by_id["frustration"], "level 2 of 2") == "2"
    votes = memory_votes(questions, found)
    assert set(votes) == {"department", "frustration"}  # is_urgent non è coperta dai ricordi
    assert votes["department"]["answer"] == "billing" and votes["department"]["support"] == 2
    assert votes["department"]["probabilities"]["billing"] == pytest.approx(0.9 + 0.1 / 3)
    assert votes["frustration"]["support"] == 2


def test_memory_render_hides_image_data():
    from egeria.memory import Memory

    memory = Memory("foto-1", [{"type": "text", "text": "Sinistro:"}, {"type": "image", "base64": "AAAA" * 1000}],
                    {"danno": "moderato"})
    rendered = memory.render()
    assert "[image: base64]" in rendered and "AAAA" not in rendered
    assert "[image: examples/x.jpg]" in Memory("f", [{"type": "image", "path": "examples/x.jpg"}]).render()


def test_recall_question_parse():
    q = _one({"type": "recall", "k": 5})
    assert q.readout == "memory" and q.params == {"k": 5} and q.instructions
    with pytest.raises(RequestError):
        parse_request({"state": "s", "questions": {"q": {"type": "recall", "k": 0}}})
