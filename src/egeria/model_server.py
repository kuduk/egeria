"""Server del modello: solo inferenza, senza stato.

    .venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base          # porta 8100

Tiene caricato il modello e risponde a tre chiamate:
- `GET  /v1/info`: modello, dispositivo, parte visiva, calibrazione;
- `POST /v1/systemone`: la decisione (API compatibile Jev, con le estensioni Egeria);
- `POST /v1/embed`: il vettore semantico di uno stato (per la memoria).

Niente storico, niente ricordi, niente file: li tiene il server web (`egeria serve`), che
chiama questo server via HTTP. Le immagini arrivano in base64 (o url); i percorsi locali
sono rifiutati, a meno di `--allow-paths`.
"""

from __future__ import annotations

import hmac
import threading
import time

from .schema import RequestError, image_max_side, is_multimodal, parse_state

MAX_PERMUTATIONS = 8


class ModelService:
    """Il modello, le temperature e un lock che serializza l'uso della GPU."""

    def __init__(self, scorer, temperatures=None, permutations: int = 2,
                 min_confidence: float | None = 0.5, image_max_side: int = 448, allow_paths: bool = False):
        self.scorer = scorer
        self.temperatures = temperatures or {}
        self.permutations = permutations
        self.min_confidence = min_confidence
        self.image_max_side = image_max_side
        self.allow_paths = allow_paths
        self.lock = threading.Lock()

    def check_images(self, state) -> None:
        if self.allow_paths or not is_multimodal(state):
            return
        for number, part in enumerate(state):
            if part.get("type") == "image" and "path" in part:
                raise RequestError(f"state[{number}]: il server del modello non legge file locali, "
                                   "invia l'immagine in base64 (oppure avvialo con --allow-paths)")

    def decide(self, body: dict) -> dict:
        """Decisione. Oltre ai campi di `/v1/systemone` accetta:

        - `"permutations": n` (1–8), per cambiare il numero di permutazioni delle opzioni;
        - `"calibrated": false`, per avere le probabilità grezze anche con le temperature caricate.

        La calibrazione non si applica mai agli stati con immagini: le temperature sono fittate su testo.
        """
        if not isinstance(body, dict):
            raise RequestError("il body deve essere un oggetto JSON")
        if "memory" in body:
            raise RequestError("la memoria sta nel server web (egeria serve): il server del modello non la consulta")
        permutations = body.get("permutations", self.permutations)
        if not isinstance(permutations, int) or isinstance(permutations, bool) or not 1 <= permutations <= MAX_PERMUTATIONS:
            raise RequestError(f"permutations deve essere un intero fra 1 e {MAX_PERMUTATIONS}")
        calibrated = body.get("calibrated", True)
        if not isinstance(calibrated, bool):
            raise RequestError("calibrated deve essere true o false")
        calibrated = calibrated and not is_multimodal(body.get("state"))
        self.check_images(body.get("state"))
        body = {k: v for k, v in body.items() if k not in ("permutations", "calibrated")}
        body.setdefault("image_max_side", self.image_max_side)
        with self.lock:
            return self.scorer.decide(
                body, permutations=permutations, temperatures=self.temperatures if calibrated else {},
                default_min_confidence=self.min_confidence,
            )

    def embed(self, body: dict) -> dict:
        """Vettore semantico dello stato: `{"state": ..., "image_max_side": 448}`."""
        if not isinstance(body, dict):
            raise RequestError("il body deve essere un oggetto JSON")
        state = parse_state(body.get("state"))
        self.check_images(state)
        side = image_max_side({"image_max_side": body.get("image_max_side", self.image_max_side)})
        started = time.perf_counter()
        with self.lock:
            result, tokens = self.scorer.analyze_state(state, image_max_side=side if is_multimodal(state) else None)
        return {**result, "input_tokens": tokens, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def create_model_app(service: ModelService, info: dict | None = None, token: str | None = None):
    from fastapi import Body, Depends, FastAPI, Header, HTTPException

    app = FastAPI(title="Egeria model server", version="0.1.0")
    info = info or {}

    def authorized(authorization: str | None = Header(default=None)):
        if token is None:
            return
        expected = f"Bearer {token}"
        if authorization is None or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="token mancante o non valido")

    def guarded(fn, *args):
        try:
            return fn(*args)
        except RequestError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/v1/info", dependencies=[Depends(authorized)])
    def model_info():
        return {**info, "permutations": service.permutations, "min_confidence": service.min_confidence,
                "image_max_side": service.image_max_side, "calibrated": bool(service.temperatures)}

    @app.post("/v1/systemone", dependencies=[Depends(authorized)])
    def systemone(body: dict = Body(...)):
        return guarded(service.decide, body)

    @app.post("/v1/embed", dependencies=[Depends(authorized)])
    def embed(body: dict = Body(...)):
        return guarded(service.embed, body)

    return app


def serve_model(args) -> int:
    """Avvio da CLI: carica il modello (con la parte visiva, salvo --text-only) e avvia uvicorn."""
    import os

    import torch
    import uvicorn

    from .calibration import load_temperatures
    from .scorer import DecisionScorer

    scorer = DecisionScorer(args.model, device=args.device, dtype=args.dtype, quantize=args.quantize,
                            vision=not args.text_only)
    service = ModelService(scorer, load_temperatures(args.temperatures), permutations=args.permutations,
                           min_confidence=args.min_confidence, image_max_side=args.image_max_side,
                           allow_paths=args.allow_paths)
    info = {"model": args.model, "device": str(scorer.device), "vision": scorer.processor is not None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    token = args.token or os.environ.get("EGERIA_TOKEN") or None
    app = create_model_app(service, info, token)
    print(f"Egeria, server del modello: http://{args.host}:{args.port}/v1/info"
          f"{' (con token)' if token else ''}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
