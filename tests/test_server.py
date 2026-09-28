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

    Come lo scorer vero, senza `memory` rifiuta le domande recall; ricorda l'ultima chiamata.
    """

    def __init__(self):
        self.calls = []

    def decide(self, body, permutations=1, temperatures=None, exit_temperatures=None,
               default_min_confidence=None, memory=None):
        from egeria.schema import RequestError

        self.calls.append({"body": body, "permutations": permutations, "temperatures": temperatures})
        state, questions = parse_request(body, default_min_confidence)
        if any(q.type == "recall" for q in questions) and memory is None:
            raise RequestError("le domande recall richiedono una memoria (--memory)")
        answers = {}
        for q in questions:
            p = softmax(np.linspace(1.0, 0.0, len(q.options)) * (3 if "sicuro" in str(state) else 0.3))
            result = answer(q, p)
            if q.min_confidence is not None:
                result["status"] = "decided" if p.max() - 1 / len(p) > q.min_confidence / 2 else "uncertain"
            answers[q.id] = result
        return {"model": "finto", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}

    def analyze_state(self, state, embed=True, image_max_side=None, images=None):
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
    assert {"question": "È urgente?", "answer": "Sì"} in item["readable"]
    assert {"question": "Quale reparto?", "answer": "tecnico"} in item["readable"]
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
    assert scorer.calls[-1]["temperatures"] == {"choice": 2.0} and scorer.calls[-1]["permutations"] == 2
    # la memoria non sta qui; i percorsi locali sono rifiutati; permutazioni e calibrazione per richiesta
    assert app.post("/v1/systemone", json={**body, "memory": {"recall": 3}}, headers=auth).status_code == 422
    image = {"state": [{"type": "image", "path": "/etc/hosts"}], "questions": QUESTIONS}
    assert app.post("/v1/systemone", json=image, headers=auth).status_code == 422
    assert app.post("/v1/embed", json={"state": image["state"]}, headers=auth).status_code == 422
    assert app.post("/v1/systemone", json={**body, "permutations": 0}, headers=auth).status_code == 422
    app.post("/v1/systemone", json={**body, "permutations": 3, "calibrated": False}, headers=auth)
    assert scorer.calls[-1]["temperatures"] == {} and scorer.calls[-1]["permutations"] == 3
    assert "permutations" not in scorer.calls[-1]["body"]
    # con immagini la calibrazione non si applica mai
    picture = [{"type": "image", "base64": image_data_url().partition(",")[2]}]
    app.post("/v1/systemone", json={"state": picture, "questions": QUESTIONS}, headers=auth)
    assert scorer.calls[-1]["temperatures"] == {} and scorer.calls[-1]["body"]["image_max_side"] == 448
    embedded = app.post("/v1/embed", json={"state": "ciao"}, headers=auth).json()
    assert embedded["embedding_dim"] == 16 and "latency_ms" in embedded
    assert app.get("/v1/info", headers=auth).json()["calibrated"] is True


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


def test_web_handles_memory_and_recall(client, scorer):
    client.post("/api/memory", json={"state": "bonifico respinto", "decisions": {"reparto": "tecnico"}, "note": "n"})
    questions = {**QUESTIONS, "simili": {"type": "recall", "k": 2}}
    body = {"state": "bonifico respinto", "questions": questions, "memory": {"recall": 1, "inject": True}}
    response = client.post("/v1/systemone", json=body).json()
    sent = scorer.calls[-1]["body"]
    assert "memory" not in sent and "simili" not in sent["questions"]  # al modello solo le sue domande
    assert sent["memories"] and "bonifico respinto" in sent["memories"][0]
    assert list(response["answers"]) == ["reparto", "urgente", "simili"]
    assert response["answers"]["simili"]["memories"][0]["similarity"] == pytest.approx(1.0)
    assert response["answers"]["reparto"]["memory"]["answer"] == "tecnico"
    assert response["memories"][0]["decisions"] == {"reparto": "tecnico"}
    # solo domande recall: il modello non viene chiamato
    calls = len(scorer.calls)
    only = client.post("/v1/systemone", json={"state": "x", "questions": {"simili": {"type": "recall"}}}).json()
    assert len(scorer.calls) == calls and only["answers"]["simili"]["type"] == "recall"


def test_web_sends_images_inline(client, scorer, tmp_path):
    media = client.post("/api/media", json={"data": image_data_url()}).json()
    state = [{"type": "text", "text": "foto"}, {"type": "image", "path": media["path"]}]
    case = client.post("/api/cases", json={"request": {"state": state, "questions": QUESTIONS}}).json()
    sent = scorer.calls[-1]["body"]["state"][1]
    assert "path" not in sent and base64.b64decode(sent["base64"])[:2] == b"\xff\xd8"
    assert case["request"]["state"][1] == {"type": "image", "path": media["path"]}  # lo storico tiene il percorso
    outside = [{"type": "image", "path": "/etc/hosts"}]
    assert client.post("/v1/systemone", json={"state": outside, "questions": QUESTIONS}).status_code == 422
