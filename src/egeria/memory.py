"""Memoria a lungo termine: ricordi richiamati per somiglianza.

Un ricordo è uno stato passato con le decisioni prese (idealmente confermate da un
esito o da una correzione umana) e una nota libera opzionale. Si indicizza con
l'embedding dello stato (`DecisionScorer.analyze_state`, media sui token) e si
richiama per similarità coseno. I ricordi richiamati tornano nella risposta, e le loro
decisioni votano sulle domande che coprono. Non entrano nel prompt: zero-shot non
migliorava le decisioni (documentazione/06-memoria.md).

Gli hidden state di un LLM sono anisotropi: tutte le coppie hanno coseno alto.
Con abbastanza ricordi si sottrae la media dell'archivio prima del coseno.

I vettori dipendono dal modello che li ha calcolati. L'archivio registra quel modello
(`store.json`): se il modello cambia, i vettori si ricalcolano dagli stati dei ricordi
(`ensure_model`). I ricordi che non si possono ricalcolare (per esempio un'immagine
cancellata) passano in `memories-sospese.jsonl`, fuori dalla ricerca ma non persi.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import render_value

CENTER_MIN_ITEMS = 20
STORE_FILE = "store.json"
SUSPENDED_FILE = "memories-sospese.jsonl"
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
        self.model: str | None = None  # modello che ha calcolato i vettori (None: archivio senza registrazione)
        self.suspended: list[dict] = []  # sospesi in questa sessione, da aggiungere al file al prossimo save
        self.suspended_count = 0  # sospesi in tutto, file compreso

    def __len__(self) -> int:
        return len(self.items)

    @property
    def dim(self) -> int:
        """Dimensione dei vettori (0 se l'archivio è vuoto)."""
        return int(self.vectors.shape[1]) if self.items else 0

    def add(self, vector, memory: Memory) -> None:
        vector = np.asarray(vector, dtype=np.float32)[None, :]
        if any(item.id == memory.id for item in self.items):
            raise ValueError(f"ricordo {memory.id!r} già presente")
        if self.items and vector.shape[1] != self.dim:
            raise ValueError(f"vettore da {vector.shape[1]} dimensioni in un archivio da {self.dim}: "
                             "l'archivio va ricalcolato con ensure_model")
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

    def reforge(self, embed, model: str | None) -> dict:
        """Ricalcola il vettore di ogni ricordo con `embed(stato)`, cioè con un altro modello.

        I ricordi il cui stato non si può più leggere (immagine cancellata, stato non valido)
        passano fra i sospesi. Se `embed` fallisce per altri motivi (per esempio il modello non
        risponde), l'errore si propaga e l'archivio resta com'era.
        """
        from .schema import RequestError

        vectors, kept, suspended = [], [], []
        for item in self.items:
            try:
                vectors.append(np.asarray(embed(item.state), dtype=np.float32))
                kept.append(item)
            except (RequestError, OSError, ValueError) as error:
                suspended.append({"memory": asdict(item), "reason": str(error), "model": model})
        self.items = kept
        self.vectors = np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
        self.model = model
        self._profile = None
        self.suspended.extend(suspended)
        self.suspended_count += len(suspended)
        return {"action": "reforged", "model": model, "reforged": len(kept), "suspended": len(suspended)}

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "memories.jsonl").open("w") as handle:
            for item in self.items:
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        (directory / STORE_FILE).write_text(json.dumps({"model": self.model, "dim": self.dim}, indent=2))
        if self.suspended:
            with (directory / SUSPENDED_FILE).open("a") as handle:
                for entry in self.suspended:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.suspended = []

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryStore":
        directory = Path(directory)
        store = cls()
        if (directory / STORE_FILE).exists():
            store.model = json.loads((directory / STORE_FILE).read_text()).get("model")
        if (directory / SUSPENDED_FILE).exists():
            store.suspended_count = sum(1 for line in (directory / SUSPENDED_FILE).read_text().splitlines() if line)
        if not (directory / "memories.jsonl").exists():
            return store
        store.vectors = np.load(directory / "vectors.npy")
        store.items = [Memory(**json.loads(line)) for line in (directory / "memories.jsonl").read_text().splitlines() if line]
        if len(store.items) != len(store.vectors):
            raise ValueError(f"archivio incoerente in {directory}: {len(store.items)} ricordi, {len(store.vectors)} vettori")
        return store


def ensure_model(store: MemoryStore, model: str | None, dim: int, embed) -> dict:
    """Rende l'archivio confrontabile con i vettori di `model` (dimensione `dim`).

    - archivio vuoto, oppure registrato senza modello ma con la stessa dimensione: si registra `model`;
    - stesso modello e stessa dimensione: niente da fare;
    - modello o dimensione diversi: si ricalcolano tutti i vettori con `embed(stato)`.

    Restituisce che cosa è stato fatto (`action`: none, adopted, reforged); il chiamante salva.
    """
    if store.items and (dim != store.dim or store.model not in (None, model)):
        return store.reforge(embed, model)
    if store.model != model and model is not None:
        store.model = model
        return {"action": "adopted", "model": model}
    return {"action": "none", "model": store.model}


def decisions_from_answers(answers: dict) -> dict[str, str]:
    """Riassunto testuale delle risposte di `decide`, per salvarle come ricordo."""
    summary = {}
    from .schema import canonical_type

    for qid, result in answers.items():
        kind = canonical_type(result.get("type"))
        if kind == "noul":
            summary[qid] = "true" if result["noul"] >= 0.5 else "false"
        elif kind == "choice":
            summary[qid] = result["choice"]
        elif kind == "score":
            summary[qid] = f"level {round(result['score'])}"
        elif kind == "estimate":
            summary[qid] = f"{result['value']:g}"
        elif kind == "short_answer":
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
        if question.type == "short_answer":
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
