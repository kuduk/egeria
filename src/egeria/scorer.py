"""Scorer decisionale su Qwen3.5: un forward pass, logit delle lettere delle opzioni.

Il modello non genera token. Per ogni prompt si prende l'hidden state finale
dell'ultima posizione e lo si moltiplica solo per le righe dell'lm_head delle
lettere A..Z: la softmax è ristretta alle opzioni dichiarate (readout di SemIf).

Readout intermedi (per lo studio della profondità dinamica): gli hidden state
all'uscita dei layer richiesti passano per la norm finale e per le stesse righe
dell'lm_head (logit lens). Di default si leggono i confini di blocco, cioè i
layer full-attention che chiudono ogni gruppo di 3 Gated DeltaNet + 1 attention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .confidence import log_softmax
from .images import load_images
from .prompt import LETTERS, build_messages, build_plain, orderings
from .schema import Question, parse_request


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
    exit_logits: np.ndarray | None = None  # [n_exit, n_opzioni], stessa combinazione
    exit_layers: list[int] = field(default_factory=list)


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
    ):
        import torch
        import transformers

        install_device_dispatch()
        if prompt_style not in {"chat", "plain"}:
            raise ValueError("prompt_style deve essere chat o plain")
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
            config = config.get_text_config()
        else:
            self.body = self.model.model
        self.layer_types = list(getattr(config, "layer_types", []) or [])
        self.num_layers = len(self.body.layers)
        self.block_exits = [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]
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

    def _prompt_text(self, state, question: Question, order: list[int], memories: list[str] | None = None) -> str:
        if self.prompt_style == "plain":
            return build_plain(state, question, order, memories)
        return self.tokenizer.apply_chat_template(
            build_messages(state, question, order, memories),
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

    def encode(self, state, question: Question, order: list[int], memories: list[str] | None = None) -> list[int]:
        ids = self.tokenizer.encode(self._prompt_text(state, question, order, memories), add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(f"domanda {question.id!r}: {len(ids)} token superano il limite {self.max_tokens}")
        return ids

    # ----------------------------------------------------------------- forward

    def slot_logits(self, sequences: list[list[int]], n_options: list[int], exit_layers: list[int] | None = None):
        """Logit degli slot per ogni sequenza: finali [n] ed eventualmente intermedi [n_exit, n]."""
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        finals: list[np.ndarray | None] = [None] * len(sequences)
        exits: list[np.ndarray | None] = [None] * len(sequences)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * len(sequences[order[end]]) <= self.batch_tokens:
                end += 1
            batch = order[start:end]
            final, inter = self._forward_batch([sequences[i] for i in batch], exit_layers)
            for row, index in enumerate(batch):
                n = n_options[index]
                finals[index] = final[row, :n]
                if inter is not None:
                    exits[index] = inter[:, row, :n]
            start = end
        return finals, (exits if exit_layers else None)

    def _forward_batch(self, sequences: list[list[int]], exit_layers: list[int] | None):
        torch = self.torch
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        last = torch.tensor([len(seq) - 1 for seq in sequences], device=self.device)
        rows = torch.arange(len(sequences), device=self.device)

        captured: dict[int, object] = {}
        handles = []
        for layer in exit_layers or []:
            def hook(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[rows, last]
            handles.append(self.body.layers[layer].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                out = self.body(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device), use_cache=False, return_dict=True
                )
                hidden = out.last_hidden_state[rows, last].float()
                final = (hidden @ self.slot_weight.T).cpu().numpy()
                inter = None
                if exit_layers:
                    stacked = torch.stack([self.body.norm(captured[layer]).float() for layer in exit_layers])
                    inter = (stacked @ self.slot_weight.T).cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        return final, inter

    # ------------------------------------------------------ profondità dinamica

    def _prepare(self, hidden, mask):
        """Maschere e rotary per il batch corrente, come nel forward di Qwen3_5TextModel."""
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        torch = self.torch
        batch, width = mask.shape
        positions = torch.arange(width, device=self.device).view(1, 1, -1).expand(4, batch, -1)
        text_positions = positions[0]
        kwargs = {
            "config": self.body.config,
            "inputs_embeds": hidden,
            "attention_mask": mask,
            "past_key_values": None,
            "position_ids": text_positions,
        }
        masks = {
            "full_attention": create_causal_mask(**kwargs),
            "linear_attention": create_recurrent_attention_mask(**kwargs),
        }
        return masks, self.body.rotary_emb(hidden, positions[1:]), text_positions

    def _adaptive_batch(self, items: list[dict], temperatures: dict, exit_layers: list[int]) -> dict[int, tuple]:
        """Esegue il modello blocco per blocco e toglie dal batch le domande già decise.

        `items`: per domanda, le sequenze (una per permutazione), gli ordini, il tipo,
        il numero di opzioni e la soglia. Tutte le permutazioni di una domanda escono
        insieme, sulla distribuzione combinata e calibrata per quel layer.
        Restituisce {indice domanda: (log-prob combinate, layer d'uscita)}.
        """
        from .confidence import decision_confidence, softmax

        torch = self.torch
        rows = [(qi, k) for qi, item in enumerate(items) for k in range(len(item["sequences"]))]
        sequences = [items[qi]["sequences"][k] for qi, k in rows]
        width = max(map(len, sequences))
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        for r, seq in enumerate(sequences):
            ids[r, : len(seq)] = torch.tensor(seq, device=self.device)
            mask[r, : len(seq)] = 1
        lengths = torch.tensor([len(seq) for seq in sequences], device=self.device)
        calibrated_layers = temperatures.get("layers", exit_layers)
        last = self.num_layers - 1
        early = {layer for layer in exit_layers if layer != last}

        active = list(range(len(rows)))  # righe originali ancora nel batch
        done: dict[int, tuple] = {}
        with torch.inference_mode():
            hidden = self.body.embed_tokens(ids)
            masks, rope, positions = self._prepare(hidden, mask)
            for i, layer in enumerate(self.body.layers):
                hidden = layer(
                    hidden, position_embeddings=rope, attention_mask=masks[self.layer_types[i]], position_ids=positions
                )
                if i not in early and i != last:
                    continue
                current = lengths[active]
                picked = hidden[torch.arange(len(active), device=self.device), current - 1]
                slots = (self.body.norm(picked).float() @ self.slot_weight.T).cpu().numpy()
                parts: dict[int, list[tuple[int, int]]] = {}
                for position, r in enumerate(active):
                    parts.setdefault(rows[r][0], []).append((position, rows[r][1]))
                finished = set()
                for qi, members in parts.items():
                    item = items[qi]
                    combined = np.zeros(item["n"])
                    for position, k in members:
                        back = np.empty(item["n"])
                        back[item["orders"][k]] = log_softmax(slots[position, : item["n"]])
                        combined += back / len(members)
                    if i == last:
                        done[qi] = (combined, i)
                        continue
                    if item["threshold"] is None or i not in calibrated_layers or item["readout"] not in temperatures:
                        continue
                    t = temperatures[item["readout"]][calibrated_layers.index(i)]
                    if decision_confidence(item["type"], softmax(combined, t)) >= item["threshold"]:
                        done[qi] = (combined, i)
                        finished.add(qi)
                if i == last:
                    break
                if finished:
                    keep = [position for position, r in enumerate(active) if rows[r][0] not in finished]
                    active = [active[position] for position in keep]
                    if not active:
                        break
                    # Right padding + modello causale: si possono tagliare le colonne oltre la sequenza più lunga.
                    width = int(lengths[active].max())
                    index = torch.tensor(keep, device=self.device)
                    hidden = hidden[index, :width]
                    mask = mask[index, :width]
                    masks, rope, positions = self._prepare(hidden, mask)
        return done

    def score_adaptive(self, state, questions: list[Question], permutations: int, temperatures: dict,
                       exit_layers: list[int], memories: list[str] | None = None) -> list[tuple[QuestionScore, int]]:
        """Come `score`, ma ogni domanda esce al primo layer che raggiunge la sua min_confidence."""
        items = []
        for question in questions:
            orders = orderings(question, permutations)
            items.append({
                "question": question, "orders": orders, "type": question.type, "readout": question.readout,
                "n": len(question.options),
                "threshold": question.min_confidence,
                "sequences": [self.encode(state, question, order, memories) for order in orders],
            })
        # Batch di domande intere (le permutazioni di una domanda escono insieme), per lunghezza.
        order = sorted(range(len(items)), key=lambda qi: max(map(len, items[qi]["sequences"])))
        results: dict[int, tuple] = {}
        start = 0
        while start < len(order):
            end, rows, width = start, 0, 0
            while end < len(order):
                item = items[order[end]]
                new_rows = rows + len(item["sequences"])
                new_width = max(width, max(map(len, item["sequences"])))
                if end > start and new_rows * new_width > self.batch_tokens:
                    break
                rows, width, end = new_rows, new_width, end + 1
            batch = [items[qi] for qi in order[start:end]]
            for local, (logits, layer) in self._adaptive_batch(batch, temperatures, exit_layers).items():
                results[order[start + local]] = (logits, layer)
            start = end
        output = []
        for qi, item in enumerate(items):
            logits, layer = results[qi]
            score = QuestionScore(
                question=item["question"], option_logits=logits, order_argmax=[],
                input_tokens=sum(map(len, item["sequences"])),
            )
            output.append((score, layer))
        return output

    # --------------------------------------------------------------- decisioni

    def score(self, state, questions: list[Question], permutations: int = 1, exit_layers: list[int] | None = None,
              memories: list[str] | None = None):
        """Punteggi per tutte le domande su uno stato, combinando le permutazioni delle opzioni."""
        plan, sequences, n_options = [], [], []
        for qi, question in enumerate(questions):
            for order in orderings(question, permutations):
                plan.append((qi, order))
                sequences.append(self.encode(state, question, order, memories))
                n_options.append(len(order))
        finals, exits = self.slot_logits(sequences, n_options, exit_layers)
        return self._combine(questions, plan, [len(seq) for seq in sequences], finals, exits, exit_layers)

    def _combine(self, questions, plan, lengths, finals, exits, exit_layers):
        """Media delle log-probabilità sulle permutazioni, riportate all'ordine originale delle opzioni."""
        results = []
        for qi, question in enumerate(questions):
            n = len(question.options)
            parts = [k for k, (q, _) in enumerate(plan) if q == qi]
            combined = np.zeros(n)
            combined_exit = np.zeros((len(exit_layers), n)) if exit_layers else None
            argmaxes = []
            for k in parts:
                order = plan[k][1]
                lp = log_softmax(finals[k][: len(order)])
                back = np.empty(n)
                back[order] = lp  # slot s -> opzione originale order[s]
                combined += back / len(parts)
                argmaxes.append(int(order[int(np.argmax(lp))]))
                if exit_layers:
                    for e in range(len(exit_layers)):
                        lpe = log_softmax(exits[k][e])
                        backe = np.empty(n)
                        backe[order] = lpe
                        combined_exit[e] += backe / len(parts)
            results.append(
                QuestionScore(
                    question=question,
                    option_logits=combined,
                    order_argmax=argmaxes,
                    input_tokens=sum(lengths[k] for k in parts),
                    exit_logits=combined_exit,
                    exit_layers=list(exit_layers or []),
                )
            )
        return results

    def score_images(self, state, questions: list[Question], permutations: int, images: list):
        """Readout a lettere con immagini nello stato: processore di Qwen3.5 + modello multimodale."""
        torch = self.torch
        plan, texts = [], []
        for qi, question in enumerate(questions):
            for order in orderings(question, permutations):
                plan.append((qi, order))
                texts.append(self._prompt_text(state, question, order))
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
        return self._combine(questions, plan, lengths, finals, None, None)

    def default_exit_layers(self) -> list[int]:
        """Confini di blocco da circa metà profondità in su: prima la risposta non è ancora formata (F0 §6.2)."""
        return [layer for layer in self.block_exits if layer >= int(0.45 * self.num_layers)] or [self.num_layers - 1]

    def decide(
        self,
        body: dict,
        permutations: int = 1,
        temperatures: dict[str, float] | None = None,
        exit_temperatures: dict | None = None,
        exit_layers: list[int] | None = None,
        default_min_confidence: float | None = None,
        memory=None,
    ) -> dict:
        """Risposta nel formato dell'API Jev per un body `/v1/systemone`.

        Se almeno una domanda ha `min_confidence` e ci sono le temperature per layer,
        si usa la profondità dinamica: ogni risposta riporta `depth` (layer usati) e
        `status` (`decided`, oppure `uncertain` se nemmeno l'ultimo layer raggiunge la soglia).
        Le domande `open` leggono tutto il vocabolario;
        `embed` analizza lo stato in un passaggio a parte.
        Con una `memory` (MemoryStore) e `"memory": {"recall": k}` nella richiesta, la risposta
        riporta i k ricordi più simili e, per le domande che coprono, il voto dei ricordi
        (`answers[q]["memory"]`). Con `"inject": true` i ricordi entrano anche nel prompt.
        """
        from .confidence import answer, decision_confidence, softmax
        from .memory import apply_memory_context, memory_context, needs_embedding
        from .schema import RequestError, image_max_side, is_multimodal, prompt_memories, state_analysis

        state, questions = parse_request(body, default_min_confidence)
        analysis = state_analysis(body)
        if is_multimodal(state):
            if self.processor is None:
                raise RequestError("lo stato contiene immagini: serve il modello con la parte visiva (--vision)")
        recalls = [q for q in questions if q.type == "recall"]
        if recalls and memory is None:
            raise RequestError("le domande recall richiedono una memoria (--memory)")
        started = time.perf_counter()
        temperatures = temperatures or {}

        # Vettore dello stato: per `embed` e per consultare la memoria.
        state_result, analysis_tokens = None, 0
        images = load_images(state, image_max_side(body)) if is_multimodal(state) else None
        if analysis["embed"] or needs_embedding(body, questions, memory):
            state_result, analysis_tokens = self.analyze_state(state, images=images)
        context = memory_context(body, questions, memory, state_result["embedding"] if state_result else None)
        # Ricordi nel prompt: dalla memoria locale, oppure già in testo dalla richiesta (server web).
        memories = context["prompt"] or prompt_memories(body)

        # Domande con readout a lettere, domande aperte (vocabolario intero) e recall (memoria).
        lettered = [q for q in questions if q.type not in ("open", "recall")]
        opens = [q for q in questions if q.type == "open"]

        multimodal = is_multimodal(state)
        # Con le immagini niente uscita anticipata (posizioni M-RoPE): modello completo, ma status calcolato.
        adaptive = not multimodal and bool(exit_temperatures) and any(q.min_confidence is not None for q in lettered)
        results: dict[str, dict] = {}
        input_tokens = 0
        if lettered and multimodal:
            for s in self.score_images(state, lettered, permutations, images):
                q = s.question
                p = softmax(s.option_logits, temperatures.get(q.readout, 1.0))
                result = answer(q, p)
                if q.min_confidence is not None:
                    result["min_confidence"] = q.min_confidence
                    decided = decision_confidence(q.type, p) >= q.min_confidence
                    result["status"] = "decided" if decided else "uncertain"
                results[q.id] = result
                input_tokens += s.input_tokens
        elif lettered and adaptive:
            layers = exit_layers or self.default_exit_layers()
            calibrated = exit_temperatures.get("layers", self.block_exits)
            for s, layer in self.score_adaptive(state, lettered, permutations, exit_temperatures, layers, memories):
                q = s.question
                t = exit_temperatures[q.readout][calibrated.index(layer)] if q.readout in exit_temperatures else 1.0
                p = softmax(s.option_logits, t)
                result = answer(q, p)
                result["depth"] = layer + 1
                if q.min_confidence is not None:
                    result["min_confidence"] = q.min_confidence
                    decided = decision_confidence(q.type, p) >= q.min_confidence
                    result["status"] = "decided" if decided else "uncertain"
                results[q.id] = result
                input_tokens += s.input_tokens
        elif lettered:
            for s in self.score(state, lettered, permutations, memories=memories):
                q = s.question
                p = softmax(s.option_logits, temperatures.get(q.readout, 1.0))
                result = answer(q, p)
                if q.min_confidence is not None:
                    result["min_confidence"] = q.min_confidence
                    decided = decision_confidence(q.type, p) >= q.min_confidence
                    result["status"] = "decided" if decided else "uncertain"
                results[q.id] = result
                input_tokens += s.input_tokens

        answers: dict[str, dict] = dict(results)
        if opens:
            for q, result, tokens in self.open_answers(state, opens, memories, images):
                answers[q.id] = result
                input_tokens += tokens
        response = {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        if analysis["embed"]:
            response["state"] = state_result
        response["usage"]["input_tokens"] += analysis_tokens
        apply_memory_context(response, context, body, questions)
        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if adaptive:
            response["layers"] = self.num_layers
        return response

    # ------------------------------------------------------------------ open

    def open_answers(self, state, questions: list[Question], memories: list[str] | None = None, images=None):
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
                build_open_messages(state, q, memories), tokenize=False, add_generation_prompt=True, enable_thinking=False
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

    # ------------------------------------------------------------------ embed

    def analyze_state(self, state, embed: bool = True, image_max_side: int | None = None, images=None) -> tuple[dict, int]:
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
        result: dict = {}
        if embed:
            vector = hidden[start:end].float().mean(0)
            vector = vector / vector.norm()
            result["embedding"] = [round(v, 6) for v in vector.cpu().tolist()]
            result["embedding_dim"] = len(result["embedding"])
        return result, len(ids)

