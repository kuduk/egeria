"""Server del modello e server web con un modello finto: casi, revisione, memoria, separazione, sicurezza."""

import base64
import hashlib
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from egeria.client import ModelClient  # noqa: E402
from egeria.confidence import answer, softmax  # noqa: E402
from egeria.model_server import ModelService, create_model_app  # noqa: E402
from egeria.schema import parse_request  # noqa: E402
from egeria.server import Engine, answer_key, create_app, review_flags  # noqa: E402


class FakeScorer:
    """Risponde in modo deterministico: sceglie sempre la prima opzione, con probabilità decrescenti.

    Ricorda ogni chiamata, per verificare che cosa il server web manda al modello.
    """

    model_id = "finto"

    def __init__(self):
        self.calls = []

    def decide(self, body, permutations=1, temperatures=None, default_min_confidence=None, memory=None,
               yes_correction=True):
        self.calls.append({"body": body, "permutations": permutations, "temperatures": temperatures,
                           "yes_correction": yes_correction})
        state, questions = parse_request(body, default_min_confidence)
        answers = {}
        for q in questions:
            p = softmax(np.linspace(1.0, 0.0, len(q.options)) * (3 if "sicuro" in str(state) else 0.3))
            if "varia" in str(state):  # distorsione verso la prima opzione, ma la risposta cambia da caso a caso
                p = softmax(np.eye(len(q.options))[len(state) % len(q.options)] + np.linspace(1.5, 0.0, len(q.options)))
            result = answer(q, p)
            if q.min_confidence is not None:
                result["status"] = "decided" if p.max() - 1 / len(p) > q.min_confidence / 2 else "uncertain"
            answers[q.id] = result
        return {"model": "finto", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}

    def analyze_state(self, state, image_max_side=None, images=None):
        digest = hashlib.sha256(str(state).encode()).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)[:16]
        return {"embedding": (vector / np.linalg.norm(vector)).tolist(), "embedding_dim": 16}, 1


def model_client(scorer, token=None, client_token=None, **options):
    """Un ModelClient vero che parla con il server del modello in-process (niente rete)."""
    service = ModelService(scorer, {"choice": 2.0}, **options)
    app = create_model_app(service, {"model": "finto", "device": "cpu"}, token=token)
    client = ModelClient("http://modello", client_token)
    client.http = TestClient(app, base_url="http://modello", headers=dict(client.http.headers))
    return client


QUESTIONS = {
    "reparto": {"type": "choice", "instructions": "Quale reparto?", "criteria": {"pagamenti": None, "tecnico": None}},
    "urgente": {"type": "noul", "instructions": "È urgente?"},
}


@pytest.fixture()
def scorer():
    return FakeScorer()


@pytest.fixture()
def client(tmp_path, monkeypatch, scorer):
    import egeria.server as server

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    engine = Engine(model_client(scorer), tmp_path / "memoria", tmp_path / "dati")
    return TestClient(create_app(engine, {"model_url": "http://modello"}))


def test_systemone_and_info(client):
    response = client.post("/v1/systemone", json={"state": "ticket sicuro", "questions": QUESTIONS})
    assert response.status_code == 200 and response.json()["answers"]["reparto"]["choice"] == "pagamenti"
    assert client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "boh"}}}).status_code == 422
    info = client.get("/api/info").json()
    assert info["model"] == "finto" and info["model_status"] == "ok" and info["memories"] == 0


def test_case_review_creates_memory(client):
    uncertain = client.post("/api/cases", json={"request": {"state": "ticket dubbio", "questions": QUESTIONS}}).json()
    assert uncertain["status"] == "da_rivedere" and set(uncertain["flags"]["uncertain"]) == {"reparto", "urgente"}
    assert client.get("/api/cases").json()["counts"]["da_rivedere"] == 1
    reviewed = client.post(f"/api/cases/{uncertain['id']}/review",
                           json={"decisions": {"reparto": "tecnico", "urgente": "true"}, "note": "verificato"}).json()
    assert reviewed["review"] == "corretto" and reviewed["final"]["reparto"] == "tecnico"
    memories = client.get("/api/memory").json()
    assert memories["total"] == 1 and memories["items"][0]["id"] == uncertain["id"]
    # una nuova revisione sostituisce il ricordo invece di duplicarlo
    client.post(f"/api/cases/{uncertain['id']}/review", json={"decisions": {"reparto": "pagamenti", "urgente": "true"}})
    assert client.get("/api/memory").json()["total"] == 1
    assert client.get("/api/cases?view=rivisti").json()["counts"]["rivisti"] == 1


def test_memory_crud_and_search(client):
    added = client.post("/api/memory", json={"state": "bonifico respinto", "decisions": {"reparto": "pagamenti"},
                                             "note": "blocco antifrode"}).json()
    assert added["decisions"] == {"reparto": "pagamenti"}
    assert client.post("/api/memory", json={"id": added["id"], "state": "altro"}).status_code == 409
    found = client.post("/api/memory/search", json={"state": "bonifico respinto"}).json()["items"]
    assert found[0]["id"] == added["id"] and found[0]["similarity"] == pytest.approx(1.0)
    assert client.delete(f"/api/memory/{added['id']}").status_code == 200
    assert client.get("/api/memory").json()["total"] == 0
    assert client.delete("/api/memory/nessuno").status_code == 404


def test_media_upload_monitor_and_path_safety(client, tmp_path):
    data_url = image_data_url()
    media = client.post("/api/media", json={"data": data_url}).json()
    assert media["url"].startswith("/media/") and client.get(media["url"]).status_code == 200
    frame = client.post("/api/monitor/frame", json={"frame": data_url, "questions": {"fuoco": {"type": "noul", "instructions": "Fuoco?"}}})
    assert frame.status_code == 200 and "fuoco" in frame.json()["answers"] and "server_ms" in frame.json()
    (tmp_path / "segreto.txt").write_text("no")
    assert client.get("/files", params={"path": "segreto.txt"}).status_code == 404
    assert client.get("/files", params={"path": "../../etc/passwd"}).status_code == 404
    assert client.get("/files", params={"path": media["path"]}).status_code == 200


def test_review_flags_and_answer_key():
    response = {"answers": {
        "a": {"type": "choice", "choice": "x", "memory": {"answer": "y"}},
        "b": {"type": "noul", "noul": 0.9, "status": "uncertain"},
        "c": {"type": "noul", "noul": 0.2, "memory": {"answer": "false"}},
    }}
    flags = review_flags(response)
    assert flags == {"uncertain": ["b"], "disagreements": ["a"], "status": "da_rivedere"}
    assert answer_key({"type": "score", "probabilities": {"0": 0.2, "1": 0.8}}) == "1"


def test_memory_is_readable(client):
    request = {"state": "ticket dubbio", "questions": QUESTIONS}
    case = client.post("/api/cases", json={"request": request}).json()
    client.post(f"/api/cases/{case['id']}/review", json={"decisions": {"reparto": "tecnico", "urgente": "true"}})
    item = client.get("/api/memory").json()["items"][0]
    # con la decisione grezza ("value"), per mostrare Sì/No tradotti nell'interfaccia in inglese
    assert {"question": "È urgente?", "answer": "Sì", "value": "true"} in item["readable"]
    assert {"question": "Quale reparto?", "answer": "tecnico", "value": "tecnico"} in item["readable"]
    # ricordo senza etichette salvate: sì/no generici, sempre con il valore grezzo
    bare = client.post("/api/memory", json={"state": "altro ticket", "decisions": {"urgente": "false"}}).json()
    assert bare["readable"] == [{"question": "urgente", "answer": "No", "value": "false"}]
    listed = client.get("/api/cases?view=tutti").json()["cases"][0]
    assert listed["text"] == "ticket dubbio" and listed["questions"] == 2 and listed["thumb"] is None and listed["images"] == 0


# ------------------------------------------------------------------ separazione modello / web


def image_data_url(color="red"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_model_server_is_stateless_and_guarded():
    scorer = FakeScorer()
    service = ModelService(scorer, {"choice": 2.0})
    app = TestClient(create_model_app(service, {"model": "finto"}, token="segreto"))
    body = {"state": "ticket sicuro", "questions": QUESTIONS}
    assert app.post("/v1/systemone", json=body).status_code == 401
    auth = {"Authorization": "Bearer segreto"}
    assert app.post("/v1/systemone", json=body, headers={"Authorization": "Bearer altro"}).status_code == 401
    assert app.post("/v1/systemone", json=body, headers=auth).json()["answers"]["reparto"]["choice"] == "pagamenti"
    assert scorer.calls[-1]["temperatures"] == {"choice": 2.0} and scorer.calls[-1]["permutations"] == "auto"
    assert scorer.calls[-1]["yes_correction"] is True
    # la memoria non sta qui; i percorsi locali sono rifiutati; permutazioni e calibrazione per richiesta
    assert app.post("/v1/systemone", json={**body, "memory": {"recall": 3}}, headers=auth).status_code == 422
    image = {"state": [{"type": "image", "path": "/etc/hosts"}], "questions": QUESTIONS}
    assert app.post("/v1/systemone", json=image, headers=auth).status_code == 422
    assert app.post("/v1/embed", json={"state": image["state"]}, headers=auth).status_code == 422
    for wrong in ({"permutations": 0}, {"permutations": "tutte"}, {"permutations": True}, {"yes_correction": "no"}):
        assert app.post("/v1/systemone", json={**body, **wrong}, headers=auth).status_code == 422
    app.post("/v1/systemone", json={**body, "permutations": 3, "calibrated": False, "yes_correction": False}, headers=auth)
    assert scorer.calls[-1]["temperatures"] == {} and scorer.calls[-1]["permutations"] == 3
    assert scorer.calls[-1]["yes_correction"] is False
    assert not {"permutations", "calibrated", "yes_correction"} & set(scorer.calls[-1]["body"])
    # con immagini la calibrazione non si applica mai
    picture = [{"type": "image", "base64": image_data_url().partition(",")[2]}]
    app.post("/v1/systemone", json={"state": picture, "questions": QUESTIONS}, headers=auth)
    assert scorer.calls[-1]["temperatures"] == {} and scorer.calls[-1]["body"]["image_max_side"] == 448
    embedded = app.post("/v1/embed", json={"state": "ciao"}, headers=auth).json()
    assert embedded["embedding_dim"] == 16 and "latency_ms" in embedded
    info = app.get("/v1/info", headers=auth).json()
    assert info["calibrated"] is True and info["permutations"] == "auto" and info["yes_correction"] is True


def test_client_token_and_unreachable_model(tmp_path, monkeypatch):
    import egeria.server as server

    wrong = model_client(FakeScorer(), token="segreto", client_token="altro")
    with pytest.raises(server.ModelUnavailable, match="token"):
        wrong.info()
    assert model_client(FakeScorer(), token="segreto", client_token="segreto").info()["model"] == "finto"

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    engine = Engine(ModelClient("http://127.0.0.1:9", timeout=2), tmp_path / "memoria", tmp_path / "dati")
    web = TestClient(create_app(engine, {"model_url": "http://127.0.0.1:9"}))
    info = web.get("/api/info").json()
    assert info["model_status"] == "non_raggiungibile" and info["memories"] == 0
    failed = web.post("/api/cases", json={"request": {"state": "x", "questions": QUESTIONS}})
    assert failed.status_code == 503 and "model-server" in failed.json()["detail"]
    assert web.get("/api/memory").status_code == 200  # storico e ricordi restano consultabili


def test_web_handles_memory(client, scorer):
    client.post("/api/memory", json={"state": "bonifico respinto", "decisions": {"reparto": "tecnico"}, "note": "n"})
    body = {"state": "bonifico respinto", "questions": QUESTIONS, "memory": {"recall": 1}}
    response = client.post("/v1/systemone", json=body).json()
    sent = scorer.calls[-1]["body"]
    assert "memory" not in sent and "memories" not in sent  # al modello solo stato e domande
    assert list(response["answers"]) == ["reparto", "urgente"]
    assert response["answers"]["reparto"]["memory"]["answer"] == "tecnico"
    assert response["memories"][0]["decisions"] == {"reparto": "tecnico"}
    assert response["memories"][0]["similarity"] == pytest.approx(1.0)
    # senza memory.recall la memoria non viene consultata
    plain = client.post("/v1/systemone", json={"state": "bonifico respinto", "questions": QUESTIONS}).json()
    assert "memories" not in plain and "memory" not in plain["answers"]["reparto"]


def test_memory_built_with_another_model_is_reforged(tmp_path, monkeypatch):
    """Archivio creato con un altro modello (vettori da 8): niente errore 500, i vettori si ricalcolano."""
    import json

    import egeria.server as server
    from egeria.memory import Memory, MemoryStore

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    old = MemoryStore()
    old.model = "altro-modello"
    old.add(np.ones(8), Memory("bonifico", "bonifico respinto", {"reparto": "tecnico"}))
    old.add(np.ones(8), Memory("foto", [{"type": "image", "path": "dati/media/cancellata.jpg"}], {"reparto": "tecnico"}))
    old.save(tmp_path / "memoria")

    engine = Engine(model_client(FakeScorer()), tmp_path / "memoria", tmp_path / "dati")
    web = TestClient(create_app(engine, {"model_url": "http://modello"}))
    response = web.post("/v1/systemone", json={"state": "bonifico respinto", "questions": QUESTIONS,
                                               "memory": {"recall": 1}})
    assert response.status_code == 200
    assert response.json()["memories"][0]["id"] == "bonifico"
    assert response.json()["memories"][0]["similarity"] == pytest.approx(1.0)
    info = web.get("/api/info").json()
    assert info["memory_model"] == "finto" and info["memories"] == 1 and info["memories_suspended"] == 1
    saved = json.loads((tmp_path / "memoria" / "store.json").read_text())
    assert saved == {"model": "finto", "dim": 16}
    # il ricordo con l'immagine cancellata non è perso: è fra i sospesi
    suspended = json.loads((tmp_path / "memoria" / "memories-sospese.jsonl").read_text().splitlines()[0])
    assert suspended["memory"]["id"] == "foto"


def test_web_sends_images_inline(client, scorer, tmp_path):
    media = client.post("/api/media", json={"data": image_data_url()}).json()
    state = [{"type": "text", "text": "foto"}, {"type": "image", "path": media["path"]}]
    case = client.post("/api/cases", json={"request": {"state": state, "questions": QUESTIONS}}).json()
    sent = scorer.calls[-1]["body"]["state"][1]
    assert "path" not in sent and base64.b64decode(sent["base64"])[:2] == b"\xff\xd8"
    assert case["request"]["state"][1] == {"type": "image", "path": media["path"]}  # lo storico tiene il percorso
    outside = [{"type": "image", "path": "/etc/hosts"}]
    assert client.post("/v1/systemone", json={"state": outside, "questions": QUESTIONS}).status_code == 422


def test_calibration_on_history(client, scorer):
    """Calibrazione sullo storico: niente sotto i 50 casi, poi la distorsione della domanda si cancella."""
    import egeria.server as server

    question = {"type": "choice", "instructions": "Quale reparto?", "batch_calibration": True,
                "criteria": {"pagamenti": None, "tecnico": None, "vendite": None}}
    request = {"state": "varia", "questions": {"reparto": question}, "memory": {"recall": 0}}
    first = client.post("/api/cases", json={"request": request}).json()
    assert first["response"]["answers"]["reparto"]["batch_calibration"] == {"cases": 0, "min_cases": 50, "applied": False}
    assert "batch_calibration" not in scorer.calls[-1]["body"]["questions"]["reparto"]  # il modello non lo vede
    for n in range(server.BATCH_MIN_CASES):
        client.post("/api/cases", json={"request": {**request, "state": "varia" + "x" * n}})
    # una domanda diversa (altro testo) non conta nello storico di questa
    other = {"reparto": {**question, "instructions": "Quale ufficio?"}}
    assert client.post("/api/cases", json={"request": {**request, "questions": other}}).json()[
        "response"]["answers"]["reparto"]["batch_calibration"]["cases"] == 0
    result = client.post("/api/cases", json={"request": {**request, "state": "varia" + "x" * 9}}).json()
    answer = result["response"]["answers"]["reparto"]
    info = answer["batch_calibration"]
    assert info["applied"] is True and info["cases"] == server.BATCH_MIN_CASES + 1
    # grezza: vince la prima opzione per la distorsione; calibrata: vince quella del caso
    assert max(info["raw"], key=info["raw"].get) == "pagamenti" and answer["choice"] == "vendite"
    assert abs(sum(answer["probabilities"].values()) - 1) < 1e-6
    # nello storico conta la distribuzione grezza, non quella già calibrata
    assert server.raw_distribution(answer) == info["raw"]
    assert server.raw_distribution({"type": "noul", "noul": 0.8}) == {"true": 0.8, "false": 1 - 0.8}
    # campo non valido o tipo non ammesso
    wrong = {"reparto": {**question, "batch_calibration": "sì"}}
    assert client.post("/v1/systemone", json={"state": "x", "questions": wrong}).status_code == 422
    words = {"parola": {"type": "short_answer", "instructions": "Che giorno?", "batch_calibration": True}}
    assert client.post("/v1/systemone", json={"state": "x", "questions": words}).status_code == 422


def test_monitor_never_uses_history(client, scorer):
    question = {"type": "noul", "instructions": "C'è fumo?", "batch_calibration": True}
    response = client.post("/api/monitor/frame", json={"frame": image_data_url(), "questions": {"fumo": question}})
    assert response.status_code == 200
    assert "batch_calibration" not in response.json()["answers"]["fumo"]
    assert "batch_calibration" not in scorer.calls[-1]["body"]["questions"]["fumo"]
