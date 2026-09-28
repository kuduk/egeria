"""Test d'integrazione con un modello vero: EGERIA_TEST_MODEL=Qwen/Qwen3.5-0.8B pytest -m model"""

import os

import numpy as np
import pytest

from egeria.schema import parse_request

from .test_core import JEV_EXAMPLE

MODEL = os.environ.get("EGERIA_TEST_MODEL")
pytestmark = [pytest.mark.model, pytest.mark.skipif(not MODEL, reason="EGERIA_TEST_MODEL non impostata")]


@pytest.fixture(scope="module")
def scorer():
    from egeria.scorer import DecisionScorer

    return DecisionScorer(MODEL, dtype="float32")


def test_decide_returns_jev_shape(scorer):
    response = scorer.decide(JEV_EXAMPLE, permutations=2)
    answers = response["answers"]
    assert set(answers) == {"is_urgent", "department", "frustration"}
    assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0
    assert sum(answers["department"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= answers["frustration"]["score"] <= 2.0


def test_batched_equals_single_in_fp32(scorer):
    state, questions = parse_request(JEV_EXAMPLE)
    sequences = [scorer.encode(state, q, list(range(len(q.options)))) for q in questions]
    counts = [len(q.options) for q in questions]
    batched = scorer.slot_logits(sequences, counts)
    # Su CPU batch e singolo coincidono (~1e-5). Su GPU i kernel veloci (flash-linear-attention,
    # causal-conv1d) non sono esatti al bit con padding diversi: ~3e-3 sui logit, irrilevante per le decisioni.
    tolerance = 1e-3 if scorer.device.type == "cpu" else 1e-2
    for seq, n, together in zip(sequences, counts, batched):
        [alone] = scorer.slot_logits([seq], [n])
        assert np.abs(alone - together).max() < tolerance


def test_status_and_open_answer(scorer):
    body = {**JEV_EXAMPLE, "min_confidence": 0.99,
            "questions": {**JEV_EXAMPLE["questions"], "word": {"type": "open", "instructions": "Main topic, one word?"}}}
    answers = scorer.decide(body)["answers"]
    assert answers["department"]["status"] in ("decided", "uncertain") and answers["department"]["min_confidence"] == 0.99
    assert answers["word"]["type"] == "open" and answers["word"]["answer"]
