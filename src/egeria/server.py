"""Server web: interfaccia, storico, ricordi e API compatibile Jev. Il modello è un servizio a parte.

    .venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base      # il modello, porta 8100
    .venv/bin/egeria serve --model-url http://127.0.0.1:8100        # l'interfaccia, porta 8000

Questo server non carica il modello e non importa torch: chiama il server del modello via HTTP
(`client.ModelClient`). Tiene lui lo storico (SQLite), le immagini caricate e i ricordi: cerca i
ricordi con il vettore dello stato (`/v1/embed`), manda al modello solo le domande, aggiunge alla
risposta i ricordi simili e il voto dei ricordi. Le immagini indicate con un percorso vengono
lette qui e inviate al modello in base64.

Calibrazione sullo storico, facoltativa per domanda (`"batch_calibration": true`): la risposta si
divide per la media delle risposte alla stessa domanda nei casi precedenti dello storico.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .client import ModelUnavailable
from .memory import (Memory, MemoryStore, apply_memory_context, describe_state, ensure_model, memory_context,
                     needs_embedding)
from .schema import RequestError, canonical_type, is_multimodal, parse_question, parse_request, parse_state

WEB_DIR = Path(__file__).parent / "web"
PROJECT_ROOT = Path.cwd()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Calibrazione sullo storico (calibrazione di lotto, Zhou et al. 2024): la risposta del modello si
# divide per la media delle sue risposte alla stessa domanda nei casi precedenti, così la distorsione
# propria di quella domanda si cancella. Si applica da BATCH_MIN_CASES casi (con meno, sulle domande a
# scelta peggiora) e usa gli ultimi BATCH_WINDOW. Presume che le risposte vere siano varie: non va usata
# per gli eventi rari, dove spingerebbe verso falsi allarmi, né dal vivo, dove i fotogrammi si somigliano.
# Misure in documentazione/09-controlli-senza-etichette.md §8.
BATCH_MIN_CASES = 50
BATCH_WINDOW = 200
BATCH_TYPES = ("noul", "choice", "score")


def question_signature(question: dict) -> str:
    """Identità di una domanda nello storico: tipo, testo e opzioni."""
    return json.dumps([question.get("type"), question.get("instructions"), question.get("criteria")],
                      ensure_ascii=False, sort_keys=True)


def raw_distribution(result: dict) -> dict | None:
    """Distribuzione del modello prima della calibrazione sullo storico."""
    batch = result.get("batch_calibration") or {}
    if batch.get("raw"):
        return batch["raw"]
    if result.get("type") == "noul":
        return {"true": result["noul"], "false": 1 - result["noul"]}
    return result.get("probabilities")


def batch_questions(questions: dict) -> set:
    """Id delle domande con la calibrazione sullo storico; errore se il campo non è valido."""
    chosen = set()
    for qid, question in questions.items():
        if not isinstance(question, dict) or "batch_calibration" not in question:
            continue
        flag = question["batch_calibration"]
        if not isinstance(flag, bool):
            raise RequestError(f"domanda {qid!r}: batch_calibration deve essere true o false")
        if flag and question.get("type") not in BATCH_TYPES:
            raise RequestError(f"domanda {qid!r}: batch_calibration vale solo per noul, choice e score")
        if flag:
            chosen.add(qid)
    return chosen


def without_batch_flag(questions: dict) -> dict:
    return {qid: {k: v for k, v in q.items() if k != "batch_calibration"} if isinstance(q, dict) else q
            for qid, q in questions.items()}


def answer_key(result: dict) -> str | None:
    """Risposta del modello come chiave di opzione, per confrontarla con il voto dei ricordi."""
    kind = canonical_type(result.get("type"))  # lo storico può avere i nomi vecchi (number, open)
    if kind == "noul":
        return "true" if result["noul"] >= 0.5 else "false"
    if kind == "choice":
        return result["choice"]
    if kind == "score":
        probabilities = result["probabilities"]
        return max(probabilities, key=probabilities.get)
    if kind == "estimate":
        return result["range"]
    if kind == "short_answer":
        return result["answer"]
    return None


def review_flags(response: dict) -> dict:
    """Perché un caso va rivisto: risposte incerte e disaccordi fra modello e ricordi."""
    uncertain, disagreements = [], []
    for qid, result in response.get("answers", {}).items():
        if result.get("status") == "uncertain":
            uncertain.append(qid)
        vote = result.get("memory")
        if vote and answer_key(result) is not None and vote["answer"] != answer_key(result):
            disagreements.append(qid)
    return {"uncertain": uncertain, "disagreements": disagreements,
            "status": "da_rivedere" if uncertain or disagreements else "ok"}


class CaseStore:
    """Casi della console in SQLite: richiesta, risposta, esito della revisione."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY, created REAL, title TEXT, request TEXT, response TEXT,
                status TEXT, flags TEXT, review TEXT, final TEXT, note TEXT, reviewed REAL)"""
        )
        self.db.commit()

    def _row(self, row) -> dict:
        keys = ["id", "created", "title", "request", "response", "status", "flags", "review", "final", "note", "reviewed"]
        case = dict(zip(keys, row))
        for key in ("request", "response", "flags", "final"):
            case[key] = json.loads(case[key]) if case[key] else None
        return case

    def add(self, title: str, request: dict, response: dict) -> dict:
        flags = review_flags(response)
        case_id = f"caso-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        with self.lock:
            self.db.execute(
                "INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, time.time(), title, json.dumps(request), json.dumps(response), flags["status"],
                 json.dumps(flags), "in_attesa", None, "", None),
            )
            self.db.commit()
        return self.get(case_id)

    def get(self, case_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, view: str = "da_rivedere", limit: int = 200) -> list[dict]:
        where = {
            "da_rivedere": "WHERE review = 'in_attesa' AND status = 'da_rivedere'",
            "in_attesa": "WHERE review = 'in_attesa'",
            "rivisti": "WHERE review != 'in_attesa'",
            "tutti": "",
        }[view]
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM cases {where} ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def counts(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT review, status, COUNT(*) FROM cases GROUP BY review, status").fetchall()
        counts = {"da_rivedere": 0, "in_attesa": 0, "rivisti": 0, "tutti": 0}
        for review, status, n in rows:
            counts["tutti"] += n
            if review == "in_attesa":
                counts["in_attesa"] += n
                if status == "da_rivedere":
                    counts["da_rivedere"] += n
            else:
                counts["rivisti"] += n
        return counts

    def history(self, signature: str, model: str | None, limit: int = BATCH_WINDOW) -> list[dict]:
        """Distribuzioni grezze delle risposte passate alla stessa domanda con lo stesso modello, dalla più recente."""
        found: list[dict] = []
        with self.lock:
            for request_text, response_text in self.db.execute("SELECT request, response FROM cases ORDER BY created DESC"):
                response = json.loads(response_text)
                if response.get("model") != model:
                    continue
                for qid, question in (json.loads(request_text).get("questions") or {}).items():
                    result = response.get("answers", {}).get(qid)
                    raw = raw_distribution(result) if result and question_signature(question) == signature else None
                    if raw:
                        found.append(raw)
                if len(found) >= limit:
                    break
        return found[:limit]

    def review(self, case_id: str, review: str, final: dict, note: str) -> dict:
        with self.lock:
            self.db.execute(
                "UPDATE cases SET review = ?, final = ?, note = ?, reviewed = ? WHERE id = ?",
                (review, json.dumps(final), note, time.time(), case_id),
            )
            self.db.commit()
        return self.get(case_id)


class Engine:
    """Storico, immagini e ricordi; il modello è raggiunto con un client HTTP (`client.ModelClient`)."""

    def __init__(self, client, memory_dir: Path, data_dir: Path, image_max_side: int = 448):
        self.client = client
        self.memory_dir = memory_dir
        self.memory = MemoryStore.load(memory_dir)
        self.data_dir = data_dir
        self.media_dir = data_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.cases = CaseStore(data_dir / "console.db")
        self.image_max_side = image_max_side
        self.lock = threading.Lock()  # archivio dei ricordi (la GPU la protegge il server del modello)

    # ------------------------------------------------------------- immagini

    def inline_images(self, state: Any) -> Any:
        """Le immagini indicate con un percorso diventano base64: il server del modello non legge file.

        Solo file dentro il progetto o dentro la cartella dei dati (le immagini caricate).
        """
        if not is_multimodal(state):
            return state
        roots = [PROJECT_ROOT.resolve(), self.data_dir.resolve()]
        parts = []
        for number, part in enumerate(state):
            if part.get("type") == "image" and "path" in part:
                target = (PROJECT_ROOT / part["path"]).resolve()
                if not any(target.is_relative_to(root) for root in roots) or not target.is_file():
                    raise RequestError(f"state[{number}]: immagine non trovata o fuori dal progetto ({part['path']})")
                part = {"type": "image", "base64": base64.b64encode(target.read_bytes()).decode("ascii")}
            parts.append(part)
        return parts

    def embed(self, state: Any) -> np.ndarray:
        side = self.image_max_side if is_multimodal(state) else None
        return self._embed_inlined(self.inline_images(state), side)

    def _embed_inlined(self, state: Any, side: int | None) -> np.ndarray:
        """Vettore dello stato (immagini già in base64), con l'archivio reso coerente con il modello."""
        vector, model = self.client.embed(state, side)
        self._ensure_model(model, len(vector))
        return vector

    def _ensure_model(self, model: str | None, dim: int) -> None:
        """Se il server del modello ora usa un altro modello, ricalcola i vettori dei ricordi.

        Succede quando si riavvia `model-server` con un altro `--model`: i vettori vecchi hanno un'altra
        dimensione, o comunque non sono confrontabili con quelli nuovi.
        """
        with self.lock:
            if self.memory.model == model and (not self.memory.items or self.memory.dim == dim):
                return

            def embed(state):
                side = self.image_max_side if is_multimodal(state) else None
                return self.client.embed(self.inline_images(state), side)[0]

            report = ensure_model(self.memory, model, dim, embed)
            if report["action"] == "none":
                return
            self.memory.save(self.memory_dir)
            if report["action"] == "reforged":
                suspended = (f", {report['suspended']} sospesi in {self.memory_dir}/memories-sospese.jsonl"
                             if report["suspended"] else "")
                print(f"Egeria: ricordi ricalcolati con il modello {model}: {report['reforged']} ricordi{suspended}",
                      flush=True)

    # ----------------------------------------------------------- decisioni

    def decide(self, body: dict, use_memory: bool = True, calibrated: bool = True) -> dict:
        """Decisione: i ricordi li consulta questo server, le domande le risponde il server del modello.

        `calibrated=False` chiede le probabilità grezze. Con immagini nello stato il server del modello
        non applica mai la calibrazione (le temperature sono fittate su testo).
        """
        state, questions = parse_request(body)
        batch = batch_questions(body["questions"])
        body = {**body, "state": self.inline_images(state)}
        body.setdefault("image_max_side", self.image_max_side)
        store = self.memory if use_memory else None
        embedding = None
        if needs_embedding(body, store):
            side = body["image_max_side"] if is_multimodal(state) else None
            embedding = self._embed_inlined(body["state"], side)
        with self.lock:
            context = memory_context(body, questions, store, embedding)

        model_body = {k: v for k, v in body.items() if k != "memory"}
        model_body["questions"] = without_batch_flag(body["questions"])
        if not calibrated:
            model_body["calibrated"] = False
        response = self.client.decide(model_body)
        if batch:
            self.calibrate_on_history(response, body["questions"], batch)
        return apply_memory_context(response, context, body, questions)

    def calibrate_on_history(self, response: dict, questions: dict, qids: set) -> None:
        """Calibrazione sullo storico delle domande indicate; ogni risposta riporta `batch_calibration`."""
        from .confidence import answer, decision_confidence

        for qid in sorted(qids):
            result = response.get("answers", {}).get(qid)
            raw = raw_distribution(result) if result else None
            if raw is None:
                continue
            history = self.cases.history(question_signature(questions[qid]), response.get("model"))
            info = {"cases": len(history), "min_cases": BATCH_MIN_CASES, "applied": False}
            if len(history) >= BATCH_MIN_CASES:
                question = parse_question(qid, without_batch_flag(questions)[qid])
                mean = np.array([np.mean([h.get(key, 0.0) for h in history]) for key in question.keys])
                p = np.array([raw[key] for key in question.keys]) / np.maximum(mean, 1e-9)
                p = p / p.sum()
                result.update(answer(question, p))
                if result.get("min_confidence") is not None:
                    decided = decision_confidence(question.type, p) >= result["min_confidence"]
                    result["status"] = "decided" if decided else "uncertain"
                info.update(applied=True, raw={key: raw[key] for key in question.keys})
            result["batch_calibration"] = info

    # ----------------------------------------------------------------- media

    def save_media(self, data_url: str) -> dict:
        """Salva un'immagine inviata come data URL (o base64) e restituisce percorso e url."""
        header, _, payload = data_url.partition(",") if data_url.startswith("data:") else ("", "", data_url)
        suffix = ".png" if "png" in header else ".jpg"
        name = f"{uuid.uuid4().hex}{suffix}"
        target = (self.media_dir / name).resolve()
        target.write_bytes(base64.b64decode(payload, validate=True))
        try:
            shown = str(target.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            shown = str(target)
        return {"path": shown, "url": f"/media/{name}"}

    # --------------------------------------------------------------- memoria

    def remember(self, state: Any, decisions: dict, note: str = "", memory_id: str | None = None,
                 source: str = "console", readable: dict | None = None) -> Memory:
        vector = self.embed(state)
        meta = {"created": time.time(), "source": source, **(readable or {})}
        memory = Memory(memory_id or f"ricordo-{uuid.uuid4().hex[:8]}", state, decisions, note, meta=meta)
        with self.lock:
            self.memory.add(vector, memory)
            self.memory.save(self.memory_dir)
        return memory

    def forget(self, memory_id: str) -> None:
        with self.lock:
            self.memory.remove(memory_id)
            self.memory.save(self.memory_dir)

    def search(self, state: Any, k: int = 8) -> list[tuple[float, Memory]]:
        if not len(self.memory):
            return []
        vector = self.embed(state)
        with self.lock:
            return self.memory.search(vector, k)

    def model_info(self) -> dict:
        """Stato del server del modello, per l'interfaccia: raggiungibile o no, e con quale modello."""
        try:
            return {**self.client.info(), "model_status": "ok"}
        except ModelUnavailable as error:
            return {"model": None, "model_status": "non_raggiungibile", "model_error": str(error)}


def plain_text(state: Any) -> str:
    """Solo il testo dello stato, senza riferimenti alle immagini."""
    if is_multimodal(state):
        return " ".join(part["text"] for part in state if part.get("type") == "text").strip()
    return describe_state(state)


def image_urls(state: Any) -> list[str]:
    """Url per mostrare nel browser le immagini di uno stato."""
    if not is_multimodal(state):
        return []
    urls = []
    for part in state:
        if part.get("type") != "image":
            continue
        if "url" in part:
            urls.append(part["url"])
        elif "path" in part:
            urls.append(f"/files?path={part['path']}")
    return urls


def readable_labels(questions: dict) -> dict:
    """Testo delle domande e delle opzioni, per mostrare i ricordi in italiano invece che con i codici."""
    texts, labels = {}, {}
    for qid, question in (questions or {}).items():
        if not isinstance(question, dict):
            continue
        texts[qid] = str(question.get("instructions", qid))
        criteria = question.get("criteria")
        if question.get("type") == "noul":
            labels[qid] = {"true": "Sì", "false": "No"}
        elif isinstance(criteria, dict) and question.get("type") == "choice":
            labels[qid] = {str(k): str(v or k) for k, v in criteria.items()}
        elif isinstance(criteria, list) and question.get("type") == "score":
            labels[qid] = {str(i): str(v) for i, v in enumerate(criteria)}
    return {"questions": texts, "labels": labels}


def memory_view(item: Memory, similarity: float | None = None) -> dict:
    texts = item.meta.get("questions", {})
    labels = item.meta.get("labels", {})
    generic = {"true": "Sì", "false": "No"}  # ricordi senza etichette salvate
    # "value" è la decisione grezza (es. "true"): l'interfaccia in inglese mostra Yes/No al posto di Sì/No.
    view = {
        "id": item.id, "text": describe_state(item.state)[:400], "plain": plain_text(item.state)[:400],
        "images": image_urls(item.state), "decisions": item.decisions, "note": item.note, "meta": item.meta,
        "readable": [{"question": texts.get(qid, qid.replace("_", " ")),
                      "answer": labels.get(qid, {}).get(value, generic.get(value, value)), "value": value}
                     for qid, value in item.decisions.items()],
    }
    if similarity is not None:
        view["similarity"] = round(similarity, 4)
    return view


def create_app(engine: Engine, info: dict | None = None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Egeria", version="0.1.0")
    info = info or {}

    def guarded(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RequestError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ModelUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/")
    def root():
        return RedirectResponse("/ui/")

    @app.get("/api/info")
    def api_info():
        return {**engine.model_info(), **info, "memories": len(engine.memory), "memory_model": engine.memory.model,
                "memories_suspended": engine.memory.suspended_count, "cases": engine.cases.counts(),
                "image_max_side": engine.image_max_side}

    # API compatibile Jev (con le estensioni Egeria).
    @app.post("/v1/systemone")
    def systemone(body: dict = Body(...)):
        return guarded(engine.decide, body)

    # ------------------------------------------------------------ console

    @app.get("/api/cases")
    def list_cases(view: str = "da_rivedere", limit: int = 200):
        if view not in ("da_rivedere", "in_attesa", "rivisti", "tutti"):
            raise HTTPException(status_code=422, detail="view non valida")
        cases = engine.cases.list(view, limit)
        for case in cases:
            state = case["request"]["state"]
            case["thumb"] = (image_urls(state) or [None])[0]
            case["images"] = len(image_urls(state))
            case["text"] = plain_text(state)[:160]
            case["questions"] = len(case["request"].get("questions", {}))
        return {"counts": engine.cases.counts(), "cases": cases}

    @app.post("/api/cases")
    def create_case(payload: dict = Body(...)):
        request = payload.get("request")
        if not isinstance(request, dict):
            raise HTTPException(status_code=422, detail="serve 'request': un body /v1/systemone")
        request = dict(request)
        request.setdefault("memory", {"recall": 3})
        response = guarded(engine.decide, request)
        title = payload.get("title") or describe_state(request["state"])[:80]
        return engine.cases.add(title, request, response)

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        case = engine.cases.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="caso non trovato")
        return {**case, "images": image_urls(case["request"]["state"])}

    @app.post("/api/cases/{case_id}/review")
    def review_case(case_id: str, payload: dict = Body(...)):
        case = engine.cases.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="caso non trovato")
        decisions = payload.get("decisions")
        if not isinstance(decisions, dict) or not decisions:
            raise HTTPException(status_code=422, detail="servono le decisioni confermate")
        decisions = {str(k): str(v) for k, v in decisions.items()}
        model = {qid: answer_key(result) for qid, result in case["response"]["answers"].items()}
        review = "confermato" if all(model.get(qid) == value for qid, value in decisions.items()) else "corretto"
        note = str(payload.get("note", ""))
        if payload.get("remember", True):
            if engine.memory.get(case_id) is not None:
                engine.forget(case_id)  # una revisione successiva sostituisce il ricordo
            engine.remember(case["request"]["state"], decisions, note, memory_id=case_id, source="storico",
                            readable=readable_labels(case["request"].get("questions")))
        return engine.cases.review(case_id, review, decisions, note)

    # ------------------------------------------------------------ monitor

    @app.post("/api/monitor/frame")
    def monitor_frame(payload: dict = Body(...)):
        """Un fotogramma (data URL JPEG) e le domande: decisione senza memoria, per la velocità.

        Probabilità non calibrate: le temperature disponibili sono fittate su decisioni testuali
        (typed-decisions) e appiattirebbero le risposte sulle immagini, che grezze sono nette.
        """
        frame = payload.get("frame", "")
        questions = payload.get("questions")
        if not frame or not isinstance(questions, dict) or not questions:
            raise HTTPException(status_code=422, detail="servono 'frame' e 'questions'")
        body = {
            "state": [{"type": "text", "text": payload.get("context", "Fotogramma della telecamera:")},
                      {"type": "image", "base64": frame.partition(",")[2] if frame.startswith("data:") else frame}],
            "questions": without_batch_flag(questions),  # mai dal vivo: i fotogrammi si somigliano
            "image_max_side": int(payload.get("max_side", 336)),
        }
        started = time.perf_counter()
        response = guarded(engine.decide, body, use_memory=False, calibrated=False)
        response["server_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    # ------------------------------------------------------------- memoria

    @app.get("/api/memory")
    def list_memory(offset: int = 0, limit: int = 60):
        items = list(reversed(engine.memory.items))  # più recenti prima
        return {"total": len(items), "items": [memory_view(item) for item in items[offset: offset + limit]]}

    @app.post("/api/memory")
    def add_memory(payload: dict = Body(...)):
        state = payload.get("state")
        if state in (None, "", []):
            raise HTTPException(status_code=422, detail="serve lo stato")
        guarded(parse_state, state)  # stessa validazione delle richieste
        decisions = {str(k): str(v) for k, v in (payload.get("decisions") or {}).items()}
        memory_id = payload.get("id") or None
        if memory_id and engine.memory.get(memory_id) is not None:
            raise HTTPException(status_code=409, detail=f"esiste già un ricordo {memory_id!r}")
        memory = guarded(engine.remember, state, decisions, str(payload.get("note", "")), memory_id,
                         str(payload.get("source", "memoria")), readable_labels(payload.get("questions")))
        return memory_view(memory)

    @app.post("/api/memory/search")
    def search_memory(payload: dict = Body(...)):
        state = payload.get("state")
        if state in (None, "", []):
            raise HTTPException(status_code=422, detail="serve lo stato da cercare")
        guarded(parse_state, state)
        found = guarded(engine.search, state, int(payload.get("k", 8)))
        return {"items": [memory_view(item, score) for score, item in found]}

    @app.delete("/api/memory/{memory_id}")
    def delete_memory(memory_id: str):
        if engine.memory.get(memory_id) is None:
            raise HTTPException(status_code=404, detail="ricordo non trovato")
        engine.forget(memory_id)
        return {"deleted": memory_id}

    # -------------------------------------------------------------- media

    @app.post("/api/media")
    def upload_media(payload: dict = Body(...)):
        data = payload.get("data", "")
        if not data:
            raise HTTPException(status_code=422, detail="serve 'data' (data URL dell'immagine)")
        try:
            return engine.save_media(data)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail="immagine non valida") from error

    @app.get("/files")
    def project_file(path: str):
        """Immagini del progetto (es. examples/immagini/...), solo dentro la cartella del progetto."""
        target = (PROJECT_ROOT / path).resolve()
        if not target.is_relative_to(PROJECT_ROOT.resolve()) or target.suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=404, detail="file non disponibile")
        if not target.exists():
            raise HTTPException(status_code=404, detail="file non trovato")
        return FileResponse(target)

    app.mount("/media", StaticFiles(directory=engine.media_dir), name="media")
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
    return app


def serve(args) -> int:
    """Avvio da CLI: server web leggero, il modello è raggiunto su --model-url."""
    import os

    import uvicorn

    from .client import ModelClient

    token = args.model_token or os.environ.get("EGERIA_TOKEN") or None
    client = ModelClient(args.model_url, token)
    engine = Engine(client, Path(args.memory), Path(args.data), image_max_side=args.image_max_side)
    info = {"model_url": args.model_url}
    status = engine.model_info()
    if status["model_status"] == "ok":
        print(f"Egeria: modello {status['model']} su {args.model_url}", flush=True)
    else:
        print(f"Egeria: attenzione, {status['model_error']}", flush=True)
    app = create_app(engine, info)
    print(f"Egeria: apri http://{args.host}:{args.port}/ nel browser", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
