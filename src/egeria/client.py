"""Client HTTP del server del modello (`egeria model-server`)."""

from __future__ import annotations

import numpy as np

from .schema import RequestError


class ModelUnavailable(RuntimeError):
    """Il server del modello non risponde, rifiuta il token o restituisce un errore interno."""


class ModelClient:
    def __init__(self, url: str = "http://127.0.0.1:8100", token: str | None = None, timeout: float = 300.0):
        import httpx

        self.url = url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        import httpx

        try:
            response = self.http.request(method, path, json=body)
        except httpx.HTTPError as error:
            raise ModelUnavailable(f"server del modello non raggiungibile su {self.url} "
                                   "(avvialo con: egeria model-server)") from error
        if response.status_code == 422:
            detail = response.json().get("detail", "richiesta non valida")
            raise RequestError(detail if isinstance(detail, str) else str(detail))
        if response.status_code == 401:
            raise ModelUnavailable("il server del modello rifiuta il token (--model-token)")
        if response.status_code >= 400:
            raise ModelUnavailable(f"errore del server del modello ({response.status_code}): {response.text[:200]}")
        return response.json()

    def info(self) -> dict:
        return self._call("GET", "/v1/info")

    def decide(self, body: dict) -> dict:
        return self._call("POST", "/v1/systemone", body)

    def embed(self, state, image_max_side: int | None = None) -> np.ndarray:
        body = {"state": state}
        if image_max_side:
            body["image_max_side"] = image_max_side
        return np.asarray(self._call("POST", "/v1/embed", body)["embedding"], dtype=np.float32)
