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
    batched, _ = scorer.slot_logits(sequences, counts)
    for seq, n, together in zip(sequences, counts, batched):
        [alone], _ = scorer.slot_logits([seq], [n])
        assert np.abs(alone - together).max() < 1e-3


def test_last_exit_matches_final_readout(scorer):
    state, questions = parse_request(JEV_EXAMPLE)
    sequences = [scorer.encode(state, q, list(range(len(q.options)))) for q in questions]
    counts = [len(q.options) for q in questions]
    finals, exits = scorer.slot_logits(sequences, counts, exit_layers=[scorer.num_layers - 1])
    for final, inter in zip(finals, exits):
        assert np.abs(final - inter[0]).max() < 1e-3
