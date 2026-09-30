"""Server del modello: solo inferenza, senza stato.

    .venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base          # porta 8100

Tiene caricato il modello e risponde a tre chiamate:
- `GET  /v1/info`: modello, dispositivo, parte visiva, permutazioni, correzioni;
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

from .prompt import AUTO
from .schema import MAX_OPTIONS, RequestError, image_max_side, is_multimodal, parse_state

# Fino a una rotazione per opzione: con tutte le rotazioni sparisce la preferenza per la posizione
# (documentazione/09-controlli-senza-etichette.md §2). "auto" (il default) = tutte le rotazioni.
MAX_PERMUTATIONS = MAX_OPTIONS


def check_permutations(value) -> int | str:
    """`"auto"` oppure un intero fra 1 e MAX_PERMUTATIONS."""
    if value == AUTO:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_PERMUTATIONS:
        raise RequestError(f"permutations deve essere 'auto' o un intero fra 1 e {MAX_PERMUTATIONS}")
    return value


class ModelService:
    """Il modello, le temperature e un lock che serializza l'uso della GPU."""

    def __init__(self, scorer, temperatures=None, permutations: int | str = AUTO,
                 min_confidence: float | None = 0.5, image_max_side: int = 448, allow_paths: bool = False,
                 yes_correction: bool = True):
        self.scorer = scorer
        self.temperatures = temperatures or {}
        self.permutations = permutations
        self.yes_correction = yes_correction
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

        - `"permutations": n` (1–26) o `"auto"` (tutte le rotazioni; diretto e inverso per le scale);
        - `"yes_correction": false`, per togliere la correzione della tendenza al Sì;
        - `"calibrated": false`, per avere le probabilità senza le temperature caricate.

        La calibrazione non si applica mai agli stati con immagini: le temperature sono fittate su testo.
        """
        if not isinstance(body, dict):
            raise RequestError("il body deve essere un oggetto JSON")
        if "memory" in body:
            raise RequestError("la memoria sta nel server web (egeria serve): il server del modello non la consulta")
        permutations = check_permutations(body.get("permutations", self.permutations))
        calibrated = body.get("calibrated", True)
        yes_correction = body.get("yes_correction", self.yes_correction)
        for name, value in (("calibrated", calibrated), ("yes_correction", yes_correction)):
            if not isinstance(value, bool):
                raise RequestError(f"{name} deve essere true o false")
        calibrated = calibrated and not is_multimodal(body.get("state"))
        self.check_images(body.get("state"))
        body = {k: v for k, v in body.items() if k not in ("permutations", "calibrated", "yes_correction")}
        body.setdefault("image_max_side", self.image_max_side)
        with self.lock:
            return self.scorer.decide(
                body, permutations=permutations, temperatures=self.temperatures if calibrated else {},
                default_min_confidence=self.min_confidence, yes_correction=yes_correction,
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
        return {**result, "model": getattr(self.scorer, "model_id", None), "input_tokens": tokens,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


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
        return {**info, "permutations": service.permutations, "yes_correction": service.yes_correction,
                "min_confidence": service.min_confidence, "image_max_side": service.image_max_side,
                "calibrated": bool(service.temperatures)}

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
                            vision=not args.text_only, share_state=args.share_state)
    service = ModelService(scorer, load_temperatures(args.temperatures), permutations=args.permutations,
                           min_confidence=args.min_confidence, image_max_side=args.image_max_side,
                           allow_paths=args.allow_paths, yes_correction=not args.no_yes_correction)
    info = {"model": args.model, "device": str(scorer.device), "vision": scorer.processor is not None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    token = args.token or os.environ.get("EGERIA_TOKEN") or None
    app = create_model_app(service, info, token)
    print(f"Egeria, server del modello: http://{args.host}:{args.port}/v1/info"
          f"{' (con token)' if token else ''}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
