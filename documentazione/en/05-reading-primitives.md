[Italiano](../05-primitive-di-lettura.md) · **English**

# Reading primitives beyond noul, choice and score

> Status: `rank`, `number`, `open` and `embed` are implemented and **verified** (§4). **Removed** because they did not pass the test: `multi` (yes-bias; it will come back with F1 training) and `surprise` (AUROC 0.50). The ones that need training (`span`, `locate`, `value`, `tags`, `why`) are still to be done. Usage examples in §5.

## 1. What can be read in a single forward pass

The same pass makes available:
- the logits of the letters at the last position, already used;
- the hidden states of **all** positions and **all** layers;
- the attention towards the state and towards the image tokens.

From these, other primitives can be derived without generating text.

### Zero-shot (no training)

| Primitive | Returns | How it is read | Example |
|---|---|---|---|
| `multi` (removed, see §4) | an independent probability for each option | one `noul` per option, in the same batch | issues present in a ticket |
| `rank` | full ranking with probabilities | distribution of the `choice` | reordering documents or actions |
| `number` (binned) | expected value + interval | `score` with numeric bins as levels | how many people are in the image |
| `open` | top-k over the whole vocabulary | softmax over the full lm_head at the last position | color of the car, without listing the options (single-token answers) |
| `embed` | semantic vector of the state | final hidden state (like Qwen3-Embedding) | similar-case search, deduplication, scene change in video |
| `recall` | the most similar memories (with decisions and notes) | memory search with the state's `embed` | "cosa ti ricorda?" ("what does it remind you of?") ([06-memory.md](06-memory.md) §7) |
| `known` (not kept) | already seen / similar / new | thresholds on similarity with the memory | "è qualcosa che hai in memoria?" ("is it something you have in memory?") ([06-memory.md](06-memory.md) §7) |
| `surprise` (removed, see §4) | how unexpected the state is | the model's log-probability of the state tokens | anomalies in emails or frames |
| uncertainty | entropy, margin, exit depth, agreement across permutations | partly already computed (`depth`, `status`) | difficulty of the decision |

### With a trained head

| Primitive | Returns | How | Example |
|---|---|---|---|
| `span` | a stretch of text from the state | start/end pointer over the positions (extractive QA, GLiNER) | invoice number, without generating |
| `locate` | a region of the image | pointer over the image tokens (32×32 px patches) | "dov'è l'incendio?" ("where is the fire?") |
| `value` | continuous number with quantiles | regression head | amounts, durations, ages |
| `tags` | one label per token | per-token classification (like Qwen3Guard-Stream) | personal data, offensive phrases |
| `why` | the decisive parts of the state | attention or gradients from the reading position | explanation without text |

## 2. Comparison with Jev and Laya (checked on 2026-09-26)

Jev (`jev-1.13.0`, docs.typesafe.ai) has **only** `noul`, `choice` and `score`, on a single `/v1/systemone` endpoint. The state is text or JSON only. By stated design it returns neither strings nor sequential structures, so that all outputs can be computed in parallel. It has no per-question confidence thresholds: abstention has to be done in the client's code.

| Capability | Jev | Laya | Egeria today | Egeria potential |
|---|---|---|---|---|
| `noul`, `choice`, `score` | ✓ | ✓ | ✓ | ✓ |
| Per-question threshold / abstention | ✗ | act/escalate head not working | ✓ `min_confidence` + `status` | ✓ |
| Dynamic depth | ✗ | ✗ | ✓ | ✓ |
| Image state | ✗ | ✗ | prototype | ✓ |
| `multi`, `rank`, `number`, `open` | ✗ (only the expected value of `score`) | ✗ | – | ✓ zero-shot |
| `embed`, `surprise` | ✗ | ✗ | – | ✓ zero-shot |
| `span`, `locate`, `value`, `tags` | ✗ | ✗ | – | with training |
| Customer fine-tuning | ✗ | ✓ | ✓ | ✓ |
| Local execution | ✗ (cloud) | ✓ | ✓ | ✓ |

## 3. Proposed priorities

1. **`embed` for real-time video.** The frame vector comes for free in the same pass: the questions are asked only when the scene changes.
2. **`locate` on images.** "Yes, here" is much more useful than "yes".
3. **`span`** to extract values from documents without generation.
4. **`open`**: cheap and zero-shot. (`multi` postponed to F1.)

## 4. Implementation and verification (zero-shot)

| Type | How it is built | Response |
|---|---|---|
| `rank` | letter readout, like `choice` | `ranking`, `probabilities`, `confidence` |
| `number` | ordinal readout like `score`, over the bins; `null` = open end | `value` (expected value over the midpoints), `interval` (10–90%, piecewise uniform), `range`, `probabilities`, `unit` |
| `open` | softmax over the whole vocabulary (special tokens excluded). Candidates are completed up to the word boundary: one prefill + ≤4 greedy tokens, with a duplicated cache | `answer`, `candidates` with `p` (probability of the first token), `confidence` |
| `embed` | mean of the final hidden states over the state tokens, normalized | `state.embedding`, `state.embedding_dim` |

**Verification** ([scripts/verifica_primitive.py](../../scripts/verifica_primitive.py), 2B-Base, thresholds fixed before seeing the results). A primitive is kept only if it passes its threshold.

| Primitive | Test | Threshold | Result | Outcome |
|---|---|---|---|---|
| `rank` | 4 situations with an obvious priority action | ≥ 3/4 | **4/4** | kept |
| `number` | 5 quantities written in the text (days, people, euros, kg, minutes) | ≥ 4/5 within the bin | **5/5** (15.4 people, €1,303, 7.7 kg, 39 min) | kept |
| `open` | 6 one-word answers (day, language, city, color, month, surname) | ≥ 5/6 | **6/6** | kept |
| `embed` | retrieval of cases with the same decisions (kNN on typed-decisions) | better than the Prior | **0.564** vs 0.478, on par with Qwen3-Embedding-0.6B | kept |
| `surprise` | 6 normal texts vs 6 anomalous ones | AUROC ≥ 0.85 | **0.50** | **removed** |
| `multi` | tickets and photos | – | 0.8B: "yes" to almost everything (yes-bias, not fixable with temperature) | **removed** |

**Why `surprise` does not work.** Perplexity measures how *predictable* a text is, not how *anomalous* it is for the use case:
- *Lorem ipsum* has a perplexity of 2.9, because the model knows it extremely well;
- a prompt injection written in good Italian scores 12.7, lower than many normal tickets (18–70);
- only random text and shuffled words come out as "surprising".

**Known limits of the primitives we kept:**
- `open` answers with a single "word" or value. The probability refers to the first token; the completion continues up to the first space (within 16 tokens), so dates, amounts and codes come out whole (a full date in "dd.mm.yyyy" format, the amount "14,21"). It also works with images ([04-image-states.md](04-image-states.md) §7).
- `embed` costs one extra pass over the state.

## 5. Usage examples

All the examples are in [examples/primitive/](../../examples/primitive/). The outputs below are **real** (Qwen3.5-2B-Base, 2 permutations, RTX 4070 Laptop). A single request can mix questions of different types ([examples/primitive_it.json](../../examples/primitive_it.json)).

### `rank`: ordering the options

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/primitive/rank.json
```

Request (the state is the ticket from [examples/ticket_it.json](../../examples/ticket_it.json)):

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ai nostri fornitori falliscono ...",
 "questions": {"priorita": {"type": "rank", "instructions": "Quale azione va fatta per prima?",
   "criteria": {"riparare_pagamenti": "Risolvere il problema dei pagamenti",
                "offrire_sconto": "Offrire uno sconto commerciale",
                "inviare_newsletter": "Inviare la newsletter mensile"}}}}
```

Response:

```json
"priorita": {"type": "rank",
  "ranking": ["riparare_pagamenti", "inviare_newsletter", "offrire_sconto"],
  "probabilities": {"riparare_pagamenti": 0.9948, "offrire_sconto": 0.0023, "inviare_newsletter": 0.0029},
  "confidence": 0.9922}
```

**How to read it.** `ranking` is the order from the most to the least appropriate. The probabilities tell how clear-cut the first choice is; here the other two are almost tied, so their relative order carries no information.

### `number`: estimating a quantity

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/primitive/number.json
```

```json
{"state": "Riunione di progetto: 12 persone in sala A e altre 3 collegate da remoto. Due colleghi hanno avvisato che arriveranno in ritardo.",
 "questions": {"partecipanti": {"type": "number",
   "instructions": "Quante persone partecipano in totale alla riunione, contando anche chi è collegato?",
   "criteria": {"bins": [0, 5, 10, 14, 16, 20, null], "unit": "persone"}}}}
```

- `bins` are the bin edges, in increasing order. `null` as the last (or first) edge means "or more" (or "less than").
- You can also pass just the list: `"criteria": [0, 5, 10, 14, 16, 20, null]`.

```json
"partecipanti": {"type": "number", "value": 14.59, "interval": [10.86, 18.97], "range": "14-16",
  "probabilities": {"0-5": 0.014, "5-10": 0.028, "10-14": 0.273, "14-16": 0.460, "16-20": 0.168, ">=20": 0.057},
  "confidence": 0.565, "unit": "persone"}
```

**How to read it:**
- `range` is the most likely bin (the right answer is 15: correct);
- `value` is the point estimate (expected value);
- `interval` contains the answer with 80% probability.

Choose the bins according to the precision you need: the narrower they are, the harder the question.

### `open`: one-word answer, without listing the options

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base examples/primitive/open.json
```

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ... Abbiamo stipendi da pagare venerdì. ...",
 "questions": {
   "giorno": {"type": "open", "instructions": "Entro quale giorno della settimana vanno pagati gli stipendi?", "top_k": 3},
   "lingua": {"type": "open", "instructions": "In che lingua è scritto il messaggio?", "top_k": 3}}}
```

```json
"giorno": {"type": "open", "answer": "Venerdì",
  "candidates": [{"text": "Venerdì", "p": 0.779}, {"text": "domenica", "p": 0.061}, {"text": "Giovedì", "p": 0.034}],
  "confidence": 0.779},
"lingua": {"type": "open", "answer": "Italiano",
  "candidates": [{"text": "Italiano", "p": 0.913}, {"text": "Italieno", "p": 0.040}, {"text": "IT", "p": 0.018}],
  "confidence": 0.913}
```

**How to read it.** `answer` is the most likely word; `candidates` are the alternatives with their probabilities (`top_k`, from 1 to 20).
- **Good for:** extracting a name, a day, a city, a color, a language.
- **Not suited to:** multi-word answers or abstract concepts. For those, use `choice` with the options.

### `embed`: semantic vector of the state

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base examples/primitive/embed.json
```

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ...", "embed": true}
```

```json
"answers": {},
"state": {"embedding": [0.0239, -0.0038, -0.0358, -0.0114, ...], "embedding_dim": 2048}
```

**How to use it:**
- `questions` can be omitted if only `embed` is requested;
- the vector is normalized: the similarity between two states is their dot product;
- for a case archive, subtract the mean vector first (anisotropy).

The case memory uses exactly this vector ([06-memory.md](06-memory.md)).

### From Python

```python
import json
from egeria.calibration import load_temperatures
from egeria.scorer import DecisionScorer

scorer = DecisionScorer("Qwen/Qwen3.5-2B-Base")  # device="cpu", dtype="float32" for CPU
temperatures, exit_temperatures = load_temperatures("runs/Qwen3.5-2B-Base/temperature.json")
body = json.load(open("examples/primitive_it.json"))
response = scorer.decide(body, permutations=2, temperatures=temperatures)
print(response["answers"]["priorita"]["ranking"], response["answers"]["giorni"]["value"])
```

Temperatures are applied per readout: `rank` uses the one for `choice`, `number` the one for `score`. `open` is not calibrated.
