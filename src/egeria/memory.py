"""Memoria a lungo termine: ricordi richiamati per somiglianza.

Un ricordo è uno stato passato con le decisioni prese (idealmente confermate da un
esito o da una correzione umana) e una nota libera opzionale. Si indicizza con
l'embedding dello stato (`DecisionScorer.analyze_state`, media sui token) e si
richiama per similarità coseno. I ricordi richiamati tornano nella risposta, e le loro
decisioni votano sulle domande che coprono. Non entrano nel prompt: zero-shot non
migliorava le decisioni (documentazione/06-memoria.md).

Gli hidden state di un LLM sono anisotropi: tutte le coppie hanno coseno alto.
Con abbastanza ricordi si sottrae la media dell'archivio prima del coseno.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import render_value

CENTER_MIN_ITEMS = 20
# Il voto usa fino a 10 ricordi: con meno è troppo sicuro (typed-decisions: NLL 1.22 con 3, 0.96 con 10).
VOTE_K = 10
VOTE_SMOOTHING = 0.1
# Pesi del voto: softmax della similarità / 0.05. Su typed-decisions equivale ai pesi uniformi
# (0.560 contro 0.562), ma con pochi ricordi privilegia quello davvero simile.
VOTE_TEMPERATURE = 0.05


@dataclass
class Memory:
    id: str
    state: Any
    decisions: dict[str, str] = field(default_factory=dict)
    note: str = ""
    meta: dict = field(default_factory=dict)


def describe_state(state: Any) -> str:
    """Stato in forma di testo; le immagini diventano un riferimento (percorso o url), mai i dati."""
    from .schema import is_multimodal

    if not is_multimodal(state):
        return render_value(state)
    parts = []
    for part in state:
        if part["type"] == "text":
            parts.append(part["text"])
        else:
            source = part.get("path") or part.get("url") or "base64"
            parts.append(f"[image: {source}]")
    return " ".join(parts)


class MemoryStore:
    def __init__(self):
        self.items: list[Memory] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._profile: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.items)

    def add(self, vector, memory: Memory) -> None:
        vector = np.asarray(vector, dtype=np.float32)[None, :]
        if any(item.id == memory.id for item in self.items):
            raise ValueError(f"ricordo {memory.id!r} già presente")
        self.vectors = vector if not len(self.items) else np.concatenate([self.vectors, vector])
        self.items.append(memory)
        self._profile = None

    def remove(self, memory_id: str) -> None:
        index = next((i for i, item in enumerate(self.items) if item.id == memory_id), None)
        if index is None:
            raise KeyError(memory_id)
        del self.items[index]
        self.vectors = np.delete(self.vectors, index, axis=0)
        self._profile = None

    def get(self, memory_id: str) -> Memory | None:
        return next((item for item in self.items if item.id == memory_id), None)

    def _normalized(self, vectors: np.ndarray) -> np.ndarray:
        if len(self.items) >= CENTER_MIN_ITEMS:
            vectors = vectors - self.vectors.mean(0)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)

    def similarities(self, vector) -> np.ndarray:
        """Similarità coseno fra un vettore e tutti i ricordi (centrate come in `search`)."""
        query = self._normalized(np.asarray(vector, dtype=np.float32)[None, :])[0]
        return self._normalized(self.vectors) @ query

    def neighbor_profile(self) -> np.ndarray:
        """Per ogni ricordo, la similarità con il suo vicino più simile nell'archivio.

        Serve da riferimento relativo: la stessa similarità significa cose diverse
        in archivi diversi (testo o immagini, centrati o no).
        """
        if self._profile is None:
            if len(self.items) < 2:
                self._profile = np.zeros(0)
            else:
                normalized = self._normalized(self.vectors)
                matrix = normalized @ normalized.T
                np.fill_diagonal(matrix, -np.inf)
                self._profile = matrix.max(1)
        return self._profile

    def familiarity(self, vector) -> dict:
        """Quanto lo stato somiglia a ciò che c'è in memoria.

        - similarity: similarità con il ricordo più vicino;
        - percentile: frazione dei ricordi il cui vicino più simile è *meno* simile di così
          (0 = più lontano di qualsiasi coppia in archivio, 1 = più vicino di tutte).
        """
        scores = self.similarities(vector)
        best = int(np.argmax(scores))
        profile = self.neighbor_profile()
        percentile = float((profile < scores[best]).mean()) if len(profile) else 0.0
        return {"similarity": float(scores[best]), "percentile": percentile, "closest": self.items[best]}

    def search(self, vector, k: int = 3, min_similarity: float | None = None, exclude: set[str] = frozenset()):
        """I k ricordi più simili: lista di (similarità, ricordo)."""
        if not self.items or k <= 0:
            return []
        query = self._normalized(np.asarray(vector, dtype=np.float32)[None, :])[0]
        scores = self._normalized(self.vectors) @ query
        found = []
        for index in np.argsort(-scores):
            memory = self.items[int(index)]
            if memory.id in exclude:
                continue
            if min_similarity is not None and scores[index] < min_similarity:
                break
            found.append((float(scores[index]), memory))
            if len(found) == k:
                break
        return found

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "memories.jsonl").open("w") as handle:
            for item in self.items:
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryStore":
        directory = Path(directory)
        store = cls()
        if not (directory / "memories.jsonl").exists():
            return store
        store.vectors = np.load(directory / "vectors.npy")
        store.items = [Memory(**json.loads(line)) for line in (directory / "memories.jsonl").read_text().splitlines() if line]
        if len(store.items) != len(store.vectors):
            raise ValueError(f"archivio incoerente in {directory}: {len(store.items)} ricordi, {len(store.vectors)} vettori")
        return store


def decisions_from_answers(answers: dict) -> dict[str, str]:
    """Riassunto testuale delle risposte di `decide`, per salvarle come ricordo."""
    summary = {}
    for qid, result in answers.items():
        kind = result.get("type")
        if kind == "noul":
            summary[qid] = "true" if result["noul"] >= 0.5 else "false"
        elif kind == "choice":
            summary[qid] = result["choice"]
        elif kind == "score":
            summary[qid] = f"level {round(result['score'])}"
        elif kind == "number":
            summary[qid] = f"{result['value']:g}"
        elif kind == "open":
            summary[qid] = result["answer"]
    return summary


def memory_key(question, stored: str | None) -> str | None:
    """Chiave dell'opzione corrispondente a una decisione salvata ("level 2 of 3" vale anche per le score)."""
    if stored is None:
        return None
    if stored in question.keys:
        return stored
    parts = stored.split()
    if question.readout == "score" and len(parts) >= 2 and parts[0] == "level" and parts[1] in question.keys:
        return parts[1]
    return None


def memory_votes(questions, found, smoothing: float = VOTE_SMOOTHING) -> dict[str, dict]:
    """Voto dei ricordi per ogni domanda che i ricordi coprono (stessa domanda, opzione valida).

    Pesi = softmax(similarità / VOTE_TEMPERATURE), più una piccola smussatura verso l'uniforme.
    Su typed-decisions questo voto (10 ricordi) arriva a 0.56 di accuratezza contro 0.47 del
    modello zero-shot (documentazione/06-memoria.md).
    """
    votes = {}
    for question in questions:
        if question.type == "open":
            continue
        hits = [(score, memory_key(question, item.decisions.get(question.id))) for score, item in found]
        hits = [(score, key) for score, key in hits if key is not None]
        if not hits:
            continue
        scores = np.array([score for score, _ in hits]) / VOTE_TEMPERATURE
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        n = len(question.keys)
        dist = np.zeros(n)
        for weight, (_, key) in zip(weights, hits):
            dist[question.keys.index(key)] += weight
        dist = (1 - smoothing) * dist + smoothing / n
        votes[question.id] = {
            "answer": question.keys[int(dist.argmax())],
            "probabilities": {key: round(float(v), 6) for key, v in zip(question.keys, dist)},
            "support": len(hits),
        }
    return votes


# ---------------------------------------------------------------------------------------------
# Ricordi in una decisione: usato sia dallo scorer locale sia dal server web (modello remoto).


def memory_entry(score: float, item: "Memory") -> dict:
    return {"id": item.id, "similarity": round(float(score), 4), "decisions": item.decisions, "note": item.note}


def needs_embedding(body: dict, store) -> bool:
    """Serve il vettore dello stato per consultare la memoria?"""
    from .schema import memory_options

    if store is None or not len(store):
        return False
    return memory_options(body)["recall"] > 0


def memory_context(body: dict, questions, store, embedding) -> dict:
    """Ciò che la memoria aggiunge a una decisione, dato il vettore dello stato.

    - recalled: i ricordi da restituire (`memory.recall`);
    - votes: il voto dei ricordi per ogni domanda che coprono (`memory.vote`).
    """
    from .schema import memory_options

    options = memory_options(body)
    context = {"recalled": [], "votes": {}}
    if store is None or options["recall"] == 0 or not len(store):
        return context
    found = store.search(embedding, max(options["recall"], VOTE_K), options["min_similarity"])
    context["recalled"] = found[: options["recall"]]
    if options["vote"]:
        context["votes"] = memory_votes(questions, found[:VOTE_K])
    return context


def apply_memory_context(response: dict, context: dict, body: dict, questions) -> dict:
    """Aggiunge a una risposta del modello il voto dei ricordi e i ricordi richiamati."""
    from .schema import memory_options

    answers = response.setdefault("answers", {})
    for qid, vote in context["votes"].items():
        if qid in answers:
            answers[qid]["memory"] = vote
    response["answers"] = {q.id: answers[q.id] for q in questions if q.id in answers}  # ordine della richiesta
    if memory_options(body)["recall"] > 0:
        response["memories"] = [memory_entry(score, item) for score, item in context["recalled"]]
    return response
