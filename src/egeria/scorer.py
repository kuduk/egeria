"""Scorer decisionale su Qwen3.5: un forward pass, logit delle lettere delle opzioni.

Il modello non genera token. Per ogni prompt si prende l'hidden state finale
dell'ultima posizione e lo si moltiplica solo per le righe dell'lm_head delle
lettere A..Z: la softmax è ristretta alle opzioni dichiarate (readout di SemIf).

Riuso dello stato: tutte le domande e le permutazioni su uno stesso stato iniziano con lo
stesso prefisso (istruzioni + stato, immagini comprese). Il prefisso si calcola una volta con
la cache; la cache ibrida (stato ricorrente e convoluzione dei layer DeltaNet, KV dei layer di
attenzione) si duplica per le domande, e si passano in batch solo le code (domanda + opzioni).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

from .confidence import log_softmax
from .images import load_images
from .prompt import LETTERS, build_messages, build_plain, orderings
from .schema import Question, parse_request


SHARE_MODES = ("auto", "always", "never")
# Riuso dello stato in modalità auto: si condivide il prefisso se evita almeno questi token
# (prefisso × righe in più). Misure con scripts/bench_riuso_stato.py (RTX 4070 Laptop, 2 permutazioni,
# documentazione/08-riuso-dello-stato.md): su GPU il passaggio in più per il prefisso ha un costo fisso
# (lanci di kernel) che si recupera solo evitando abbastanza token, con pareggio a ~500 token sul 2B e
# ~1000 sul 0.8B, dove ogni token costa meno. Su CPU il calcolo domina e condividere conviene sempre.
SHARE_THRESHOLD_GPU_SMALL = 1000  # modelli testuali sotto SMALL_MODEL_PARAMETERS
SHARE_THRESHOLD_GPU = 500
SHARE_THRESHOLD_CPU = 1
SMALL_MODEL_PARAMETERS = 1.2e9

_DISPATCH_INSTALLED = False
_KERNEL_FUNCTIONS = (
    "causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
)


def install_device_dispatch() -> None:
    """Kernel veloci (causal-conv1d, flash-linear-attention) solo sui tensori CUDA.

    transformers sceglie l'implementazione all'importazione: con i kernel installati li
    userebbe anche su CPU, dove falliscono. Il dispatcher usa il kernel veloce su CUDA e
    l'implementazione PyTorch di riferimento (conservata in `__wrapped__`) altrove.
    """
    global _DISPATCH_INSTALLED
    if _DISPATCH_INSTALLED:
        return
    import functools
    import inspect

    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    for name in _KERNEL_FUNCTIONS:
        fast = getattr(qwen, name, None)
        reference = getattr(fast, "__wrapped__", None)
        if fast is None or reference is None:
            continue
        parameters = inspect.signature(reference).parameters
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

        @functools.wraps(reference)
        def dispatch(*args, _fast=fast, _reference=reference, _parameters=parameters, _any=accepts_any, **kwargs):
            first = next((v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)), None)
            if first is not None and first.is_cuda:
                return _fast(*args, **kwargs)
            if not _any:
                kwargs = {k: v for k, v in kwargs.items() if k in _parameters}
            return _reference(*args, **kwargs)

        setattr(qwen, name, dispatch)
    _DISPATCH_INSTALLED = True


@dataclass
class QuestionScore:
    question: Question
    option_logits: np.ndarray  # log-prob medie sulle permutazioni, nell'ordine originale delle opzioni
    order_argmax: list[int]  # argmax (indice originale) per ogni permutazione
    input_tokens: int


class DecisionScorer:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        quantize: str | None = None,
        prompt_style: str = "chat",
        max_tokens: int = 8192,
        batch_tokens: int = 12288,
        revision: str | None = None,
        vision: bool = False,
        image_batch: int = 6,
        share_state: str = "auto",
    ):
        import torch
        import transformers

        install_device_dispatch()
        if prompt_style not in {"chat", "plain"}:
            raise ValueError("prompt_style deve essere chat o plain")
        if share_state not in SHARE_MODES:
            raise ValueError(f"share_state deve essere uno di {SHARE_MODES}")
        self.share_state = share_state
        self.torch = torch
        self.model_id = model_id
        self.prompt_style = prompt_style
        self.max_tokens = max_tokens
        self.batch_tokens = batch_tokens

        config = transformers.AutoConfig.from_pretrained(model_id, revision=revision)
        cls = transformers.AutoModelForCausalLM
        self.image_batch = image_batch
        if vision:
            # Modello completo, con la torre visiva, per gli stati con immagini.
            if config.model_type != "qwen3_5":
                raise ValueError("la modalità vision è supportata solo per i modelli Qwen3.5")
            cls = transformers.Qwen3_5ForConditionalGeneration
        elif config.model_type in {"qwen3_5", "qwen3_5_moe"}:
            # Carica solo il modello testuale: la torre visiva non serve.
            config = config.get_text_config()
            cls = transformers.Qwen3_5ForCausalLM if config.model_type == "qwen3_5_text" else cls
        kwargs = {"config": config, "revision": revision, "device_map": {"": device}}
        if quantize == "4bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype)
            )
        elif quantize is not None:
            raise ValueError("quantize supporta solo 4bit")
        else:
            kwargs["dtype"] = getattr(torch, dtype)
        if vision:
            kwargs.pop("config")
        self.model = cls.from_pretrained(model_id, **kwargs).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.device = next(self.model.parameters()).device

        self.processor = None
        if vision:
            self.processor = transformers.AutoProcessor.from_pretrained(model_id, revision=revision)
            self.processor.tokenizer.padding_side = "left"  # l'ultima posizione è sempre l'ultimo token reale
            self.multimodal = self.model.model
            self.body = self.model.model.language_model
        else:
            self.body = self.model.model
        self.body_parameters = sum(p.numel() for p in self.body.parameters())
        self.embedding_dim = int(self.body.config.hidden_size)  # dimensione dei vettori di analyze_state
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

        prefix = " " if prompt_style == "plain" else ""
        self.slot_ids = []
        for letter in LETTERS:
            ids = self.tokenizer.encode(prefix + letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"lo slot {prefix + letter!r} non è un singolo token")
            self.slot_ids.append(ids[0])
        weight = self.model.get_output_embeddings().weight
        self.slot_weight = weight[self.slot_ids].detach().float()  # [26, d]
        self._check_boundary()

    # ------------------------------------------------------------------ prompt

    def _prompt_text(self, state, question: Question, order: list[int]) -> str:
        if self.prompt_style == "plain":
            return build_plain(state, question, order)
        return self.tokenizer.apply_chat_template(
            build_messages(state, question, order),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def _check_boundary(self) -> None:
        """La lettera appesa al prompt deve diventare esattamente il token dello slot."""
        probe = Question("probe", "noul", "probe", ())
        text = self._prompt_text("probe state", probe, [])
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        prefix = " " if self.prompt_style == "plain" else ""
        for letter, slot in zip(LETTERS, self.slot_ids):
            if self.tokenizer.encode(text + prefix + letter, add_special_tokens=False) != ids + [slot]:
                raise ValueError(f"la tokenizzazione cambia al confine della risposta per lo slot {letter}")

    def encode(self, state, question: Question, order: list[int]) -> list[int]:
        ids = self.tokenizer.encode(self._prompt_text(state, question, order), add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f"domanda {question.id!r}: {len(ids)} token superano il limite {self.max_tokens}")
        return ids

    # ----------------------------------------------------------------- forward

    def slot_logits(self, sequences: list[list[int]], n_options: list[int]) -> list[np.ndarray]:
        """Logit degli slot (lettere) all'ultima posizione di ogni sequenza.

        Se conviene, il prefisso comune a tutte le sequenze si calcola una volta sola (riuso dello stato).
        """
        prefix = self._shared_prefix(sequences)
        if prefix:
            with self.torch.inference_mode():
                cache = self.body(input_ids=self.torch.tensor([sequences[0][:prefix]], device=self.device),
                                  use_cache=True).past_key_values
            return self._suffix_logits(self.body, cache, prefix, [seq[prefix:] for seq in sequences], n_options)
        return self._full_logits(sequences, n_options)

    def _shared_prefix(self, sequences: list[list[int]]) -> int:
        """Token iniziali comuni a tutte le sequenze, se conviene calcolarli una volta sola; altrimenti 0."""
        if self.share_state == "never" or len(sequences) < 2:
            return 0
        first, limit = sequences[0], min(map(len, sequences)) - 1  # a ogni sequenza resta almeno un token di coda
        prefix = 0
        while prefix < limit and all(seq[prefix] == first[prefix] for seq in sequences):
            prefix += 1
        return prefix if self._worth_sharing(prefix, len(sequences)) else 0

    def _worth_sharing(self, prefix: int, rows: int) -> bool:
        """Il prefisso condiviso costa un passaggio in più: conviene solo se evita abbastanza token."""
        if self.share_state == "always":
            return prefix > 0
        return self.share_state == "auto" and prefix * (rows - 1) >= self.share_threshold()

    def share_threshold(self) -> int:
        """Token evitati a partire dai quali, in modalità auto, conviene condividere il prefisso."""
        if self.device.type != "cuda":
            return SHARE_THRESHOLD_CPU
        return SHARE_THRESHOLD_GPU_SMALL if self.body_parameters < SMALL_MODEL_PARAMETERS else SHARE_THRESHOLD_GPU

    def _suffix_logits(self, model, cache, prefix: int, suffixes: list[list[int]], n_options: list[int],
                       rope_delta=None) -> list[np.ndarray]:
        """Logit degli slot per le code, tutte a partire dallo stesso prefisso già in `cache`.

        Per ogni batch la cache del prefisso si copia e si duplica sulle righe. Right padding: con un
        modello causale le posizioni dopo l'ultimo token reale non cambiano quelle lette.
        Con le immagini le posizioni M-RoPE delle code sono indice + `rope_delta`.
        """
        torch = self.torch
        order = sorted(range(len(suffixes)), key=lambda i: len(suffixes[i]))
        finals: list[np.ndarray | None] = [None] * len(suffixes)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * len(suffixes[order[end]]) <= self.batch_tokens:
                end += 1
            batch = order[start:end]
            rows, width = len(batch), max(len(suffixes[i]) for i in batch)
            ids = torch.full((rows, width), self.pad_id, dtype=torch.long, device=self.device)
            mask = torch.zeros((rows, prefix + width), dtype=torch.long, device=self.device)
            mask[:, :prefix] = 1
            for row, index in enumerate(batch):
                ids[row, : len(suffixes[index])] = torch.tensor(suffixes[index], device=self.device)
                mask[row, prefix : prefix + len(suffixes[index])] = 1
            kwargs = {}
            if rope_delta is not None:
                positions = torch.arange(prefix, prefix + width, device=self.device) + rope_delta
                kwargs["position_ids"] = positions.view(1, 1, -1).expand(3, rows, -1)
            with torch.inference_mode():
                branch = copy.deepcopy(cache) if end < len(order) else cache  # l'ultimo batch usa l'originale
                branch.reorder_cache(torch.zeros(rows, dtype=torch.long, device=self.device))
                out = model(input_ids=ids, attention_mask=mask, past_key_values=branch, use_cache=True, **kwargs)
                last = torch.tensor([len(suffixes[i]) - 1 for i in batch], device=self.device)
                hidden = out.last_hidden_state[torch.arange(rows, device=self.device), last].float()
                logits = (hidden @ self.slot_weight.T).cpu().numpy()
            for row, index in enumerate(batch):
                finals[index] = logits[row, : n_options[index]]
            start = end
        return finals

    def _full_logits(self, sequences: list[list[int]], n_options: list[int]) -> list[np.ndarray]:
        """Ogni sequenza per intero, a batch per lunghezza."""
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        finals: list[np.ndarray | None] = [None] * len(sequences)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * len(sequences[order[end]]) <= self.batch_tokens:
                end += 1
            batch = order[start:end]
            final = self._forward_batch([sequences[i] for i in batch])
            for row, index in enumerate(batch):
                finals[index] = final[row, : n_options[index]]
            start = end
        return finals

    def _forward_batch(self, sequences: list[list[int]]) -> np.ndarray:
        torch = self.torch
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        last = torch.tensor([len(seq) - 1 for seq in sequences], device=self.device)
        rows = torch.arange(len(sequences), device=self.device)
        with torch.inference_mode():
            out = self.body(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device), use_cache=False, return_dict=True
            )
            hidden = out.last_hidden_state[rows, last].float()
            return (hidden @ self.slot_weight.T).cpu().numpy()

    # --------------------------------------------------------------- decisioni

    def score(self, state, questions: list[Question], permutations: int = 1) -> list[QuestionScore]:
        """Punteggi per tutte le domande su uno stato, combinando le permutazioni delle opzioni."""
        plan, sequences, n_options = [], [], []
        for qi, question in enumerate(questions):
            for order in orderings(question, permutations):
                plan.append((qi, order))
                sequences.append(self.encode(state, question, order))
                n_options.append(len(order))
        finals = self.slot_logits(sequences, n_options)
        return self._combine(questions, plan, [len(seq) for seq in sequences], finals)

    def _combine(self, questions, plan, lengths, finals) -> list[QuestionScore]:
        """Media delle log-probabilità sulle permutazioni, riportate all'ordine originale delle opzioni."""
        results = []
        for qi, question in enumerate(questions):
            n = len(question.options)
            parts = [k for k, (q, _) in enumerate(plan) if q == qi]
            combined = np.zeros(n)
            argmaxes = []
            for k in parts:
                order = plan[k][1]
                lp = log_softmax(finals[k][: len(order)])
                back = np.empty(n)
                back[order] = lp  # slot s -> opzione originale order[s]
                combined += back / len(parts)
                argmaxes.append(int(order[int(np.argmax(lp))]))
            results.append(QuestionScore(question=question, option_logits=combined, order_argmax=argmaxes,
                                         input_tokens=sum(lengths[k] for k in parts)))
        return results

    def score_images(self, state, questions: list[Question], permutations: int, images: list):
        """Readout a lettere con immagini nello stato: processore di Qwen3.5 + modello multimodale.

        Se conviene, istruzioni e stato (immagini comprese) si calcolano una volta sola e si
        passano in batch solo le code; altrimenti ogni prompt porta la sua copia delle immagini.
        """
        torch = self.torch
        plan, texts = [], []
        for qi, question in enumerate(questions):
            for order in orderings(question, permutations):
                plan.append((qi, order))
                texts.append(self._prompt_text(state, question, order))
        cut = self._state_boundary(texts)
        if cut is not None:
            self.multimodal.rope_deltas = None  # niente spostamenti M-RoPE rimasti da una richiesta precedente
            prefix_inputs = self.processor(text=[texts[0][:cut]], images=images, return_tensors="pt").to(self.device)
            prefix = int(prefix_inputs["input_ids"].shape[1])
            suffixes = [self.tokenizer.encode(text[cut:], add_special_tokens=False) for text in texts]
            if self._worth_sharing(prefix, len(texts)):
                with torch.inference_mode():
                    cache = self.multimodal(**prefix_inputs, use_cache=True).past_key_values
                delta = self.multimodal.rope_deltas.reshape(-1)[0]
                finals = self._suffix_logits(self.multimodal, cache, prefix, suffixes,
                                             [len(order) for _, order in plan], rope_delta=delta)
                return self._combine(questions, plan, [prefix + len(s) for s in suffixes], finals)
        finals, lengths = [], []
        for start in range(0, len(texts), self.image_batch):
            chunk = texts[start : start + self.image_batch]
            inputs = self.processor(
                text=chunk, images=[image for _ in chunk for image in images], return_tensors="pt", padding=True
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.multimodal(**inputs, use_cache=False).last_hidden_state[:, -1].float()
            logits = (hidden @ self.slot_weight.T).cpu().numpy()
            finals += list(logits)
            lengths += inputs["attention_mask"].sum(1).tolist()
        return self._combine(questions, plan, lengths, finals)

    def _state_boundary(self, texts: list[str]) -> int | None:
        """Posizione, uguale in tutti i prompt, subito dopo il blocco dello stato; None se non si può tagliare lì.

        Il taglio non deve cambiare la tokenizzazione: si verifica su ogni prompt.
        """
        if self.share_state == "never" or len(texts) < 2:
            return None
        marker = "</state>\n\n"
        cut = texts[0].find(marker)
        if cut < 0:
            return None
        cut += len(marker)
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        head = encode(texts[0][:cut])
        for text in texts:
            if text[:cut] != texts[0][:cut] or encode(text) != head + encode(text[cut:]):
                return None
        return cut

    def decide(
        self,
        body: dict,
        permutations: int = 1,
        temperatures: dict[str, float] | None = None,
        default_min_confidence: float | None = None,
        memory=None,
    ) -> dict:
        """Risposta nel formato dell'API Jev per un body `/v1/systemone`.

        Ogni domanda con una soglia (`min_confidence`) riporta `status`: `decided`, oppure
        `uncertain` se la confidenza resta sotto la soglia. Le domande `open` leggono tutto il
        vocabolario. Con una `memory` (MemoryStore) e `"memory": {"recall": k}` nella richiesta,
        la risposta riporta i k ricordi più simili e, per le domande che coprono, il voto dei
        ricordi (`answers[q]["memory"]`).
        """
        from .confidence import answer, decision_confidence, softmax
        from .memory import apply_memory_context, memory_context, needs_embedding
        from .schema import RequestError, image_max_side, is_multimodal

        state, questions = parse_request(body, default_min_confidence)
        multimodal = is_multimodal(state)
        if multimodal and self.processor is None:
            raise RequestError("lo stato contiene immagini: serve il modello con la parte visiva (--vision)")
        started = time.perf_counter()
        temperatures = temperatures or {}
        images = load_images(state, image_max_side(body)) if multimodal else None

        # Vettore dello stato, solo per consultare la memoria.
        embedding, analysis_tokens = None, 0
        if needs_embedding(body, memory):
            result, analysis_tokens = self.analyze_state(state, images=images)
            embedding = result["embedding"]
        context = memory_context(body, questions, memory, embedding)

        # Domande con readout a lettere e domande aperte (vocabolario intero).
        lettered = [q for q in questions if q.type != "open"]
        opens = [q for q in questions if q.type == "open"]
        answers: dict[str, dict] = {}
        input_tokens = analysis_tokens
        if lettered:
            scores = self.score_images(state, lettered, permutations, images) if multimodal \
                else self.score(state, lettered, permutations)
            for s in scores:
                q = s.question
                p = softmax(s.option_logits, temperatures.get(q.readout, 1.0))
                result = answer(q, p)
                if q.min_confidence is not None:
                    result["min_confidence"] = q.min_confidence
                    decided = decision_confidence(q.type, p) >= q.min_confidence
                    result["status"] = "decided" if decided else "uncertain"
                answers[q.id] = result
                input_tokens += s.input_tokens
        for q, result, tokens in self.open_answers(state, opens, images):
            answers[q.id] = result
            input_tokens += tokens
        response = {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        apply_memory_context(response, context, body, questions)
        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    # ------------------------------------------------------------------ open

    def open_answers(self, state, questions: list[Question], images=None):
        """Risposta breve (una parola o un valore) dall'intero vocabolario, anche con immagini nello stato.

        Per ogni domanda: un prefill del prompt (con le immagini, se ci sono) che dà insieme i
        logit dell'ultima posizione e la cache; poi le candidate migliori si completano fino
        allo spazio successivo, così si leggono anche date, importi e codici ("25.03.1980", "14,21").
        """
        from .confidence import merge_candidates, open_answer
        from .prompt import build_open_messages
        from .schema import is_multimodal

        torch = self.torch
        special = self._special_ids()
        results = []
        for q in questions:
            text = self.tokenizer.apply_chat_template(
                build_open_messages(state, q), tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            if is_multimodal(state):
                inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
            else:
                inputs = {"input_ids": torch.tensor([self.tokenizer.encode(text, add_special_tokens=False)], device=self.device)}
            if self.processor is not None:
                self.multimodal.rope_deltas = None  # niente spostamenti M-RoPE rimasti da una richiesta precedente
            with torch.inference_mode():
                out = self.model(**inputs, use_cache=True, logits_to_keep=1)
                logits = out.logits[0, -1].float()
                logits[special] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                top = torch.topk(probs, q.params["top_k"] + 3)
            ids = [int(i) for i in top.indices.tolist() if self.tokenizer.decode([int(i)]).strip()]
            texts = self._complete_words(out.past_key_values, ids)
            candidates = [(texts[i], float(probs[i])) for i in ids]
            results.append((q, open_answer(merge_candidates(candidates, q.params["top_k"])), int(inputs["input_ids"].shape[1])))
        return results

    def _complete_words(self, cache, first_ids: list[int], max_tokens: int = 16) -> dict[int, str]:
        """Completa ogni primo token in greedy fino allo spazio successivo (al massimo `max_tokens` token:
        il tokenizer di Qwen spezza le cifre una per una, e una data come 25.03.1980 ne occupa 10).

        Le parole lunghe e i valori occupano più token ("Pag" + "amenti", "25" + "." + "03"…):
        senza completamento si leggerebbe solo il primo pezzo. La probabilità resta quella del
        primo token. La cache del prefill (KV + stato DeltaNet) si duplica per le candidate,
        che avanzano insieme di un token alla volta.
        """
        torch = self.torch
        if not first_ids:
            return {}
        special = set(self._special_ids().tolist())
        pieces = [[i] for i in first_ids]
        open_rows = list(range(len(first_ids)))
        with torch.inference_mode():
            cache.reorder_cache(torch.zeros(len(first_ids), dtype=torch.long, device=self.device))
            step = torch.tensor([[i] for i in first_ids], device=self.device)
            for _ in range(max_tokens):
                out = self.model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                following = out.logits[:, -1].argmax(-1).tolist()
                for row in list(open_rows):
                    token = following[row]
                    piece = self.tokenizer.decode([token])
                    # Fine della risposta: token speciale o spazio (la punteggiatura interna resta: date, importi).
                    if token in special or not piece or piece[0].isspace():
                        open_rows.remove(row)
                    else:
                        pieces[row].append(token)
                if not open_rows:
                    break
                step = torch.tensor([[t] for t in following], device=self.device)
        return {first: self.tokenizer.decode(tokens).strip().rstrip(".,;:!?»\"'").strip()
                for first, tokens in zip(first_ids, pieces)}

    def _special_ids(self):
        if not hasattr(self, "_special_cache"):
            ids = set(self.tokenizer.all_special_ids) | set(self.tokenizer.added_tokens_decoder.keys())
            self._special_cache = self.torch.tensor(sorted(ids), device=self.device)
        return self._special_cache

    # ------------------------------------------------------- vettore dello stato

    def analyze_state(self, state, image_max_side: int | None = None, images=None) -> tuple[dict, int]:
        """Un passaggio sul solo stato: embedding = media degli hidden finali sui token dello stato, normalizzata L2.

        Con immagini nello stato si usa il modello multimodale e la media include i token visivi.
        Nei test (testo) recupera casi pertinenti quanto Qwen3-Embedding-0.6B (documentazione/06-memoria.md).
        """
        from .prompt import build_state_messages
        from .schema import RequestError, is_multimodal

        torch = self.torch
        text = self.tokenizer.apply_chat_template(build_state_messages(state), tokenize=False, add_generation_prompt=False)
        head = text[: text.index("<state>\n") + len("<state>\n")]
        tail = text[text.rindex("\n</state>"):]
        with torch.inference_mode():
            if is_multimodal(state):
                if self.processor is None:
                    raise RequestError("lo stato contiene immagini: serve il modello con la parte visiva (--vision)")
                images = images if images is not None else load_images(state, image_max_side)
                inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
                ids = inputs["input_ids"][0].tolist()
                hidden = self.multimodal(**inputs, use_cache=False).last_hidden_state[0]
            else:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                hidden = self.body(input_ids=torch.tensor([ids], device=self.device), use_cache=False).last_hidden_state[0]

        # Confini dello stato dal testo che lo precede e da quello che lo segue (nessuna immagine lì dentro).
        # Token fusi (es. ".\n") possono spostare il confine: si usa la parte comune più lunga.
        enc = self.tokenizer.encode(head, add_special_tokens=False)
        start = min(len(enc), len(ids))
        while start > 0 and enc[:start] != ids[:start]:
            start -= 1
        enc = self.tokenizer.encode(tail, add_special_tokens=False)
        common = min(len(enc), len(ids))
        while common > 0 and enc[-common:] != ids[-common:]:
            common -= 1
        end = len(ids) - common
        vector = hidden[start:end].float().mean(0)
        vector = vector / vector.norm()
        embedding = [round(v, 6) for v in vector.cpu().tolist()]
        return {"embedding": embedding, "embedding_dim": len(embedding)}, len(ids)

