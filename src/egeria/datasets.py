"""Loader del benchmark typed-decisions (HF `LocalLLaMA/typed-decisions`).

Ogni riga ha `state` e `questions`, che insieme sono esattamente un body
`/v1/systemone`, e `gold` con la distribuzione completa per ogni domanda.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

TYPED_DECISIONS = "LocalLLaMA/typed-decisions"
CONFIGS = ("all", "agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


def _maybe_json(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def load_typed_decisions(split: str = "test", config: str = "all", limit: int | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if config not in CONFIGS:
        raise ValueError(f"config deve essere uno di {CONFIGS}")
    path = hf_hub_download(TYPED_DECISIONS, f"{config}/{split}-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["id", "workflow", "state", "questions", "gold"]).to_pylist()
    if limit is not None and limit < len(rows):
        # Le righe sono ordinate per workflow: un campione equispaziato li copre tutti.
        rows = [rows[round(i * len(rows) / limit)] for i in range(limit)]
    for row in rows:
        yield {
            "id": row["id"],
            "workflow": row["workflow"],
            "body": {"state": _maybe_json(row["state"]), "questions": json.loads(row["questions"])},
            "gold": json.loads(row["gold"]),
        }


def gold_vector(question, gold_answer: dict) -> tuple[list[float], int]:
    """Distribuzione gold nell'ordine delle opzioni della domanda e indice della label."""
    probabilities = gold_answer["probabilities"]
    vector = [float(probabilities.get(key, 0.0)) for key in question.keys]
    total = sum(vector)
    vector = [value / total for value in vector]
    return vector, question.keys.index(str(gold_answer["label"]))
