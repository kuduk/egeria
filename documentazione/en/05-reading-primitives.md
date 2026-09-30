[Italiano](../05-primitive-di-lettura.md) · **English**

# Reading primitives beyond noul, choice and score

> Status: `estimate` and `short_answer` are implemented and **verified** (§4), plus the semantic vector of the state, which today is requested only through `POST /v1/embed` on the model server. Until 29/09/2026 `estimate` and `short_answer` were called `number` and `open`: the old names are still accepted as aliases.
>
> **Removed:**
> - on 29/09/2026, to simplify: `rank`, which is a `choice` with the options sorted by probability, and `embed` inside `/v1/systemone`;
> - earlier, because they did not pass the test: `multi` (yes-bias) and `surprise` (AUROC 0.50). On 29/09/2026 `multi` was also dropped as a training goal: it is equivalent to N Yes/No questions.
>
> The ones that need training (`span`, `locate`, `value`, `tags`, `why`) are still to be done. Usage examples in §5.

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
| `rank` (removed on 29/09/2026) | full ranking with probabilities | distribution of the `choice`: just sort it | reordering documents or actions |
| `estimate` (binned) | expected value + interval | `score` with numeric bins as levels | how many people are in the image |
| `short_answer` | top-k over the whole vocabulary | softmax over the full lm_head at the last position | color of the car, without listing the options (single-token answers) |
| state vector (`/v1/embed`) | semantic vector of the state | final hidden state (like Qwen3-Embedding) | similar-case search, deduplication, scene change in video |
| similar memories (`memory.recall`) | the most similar memories (with decisions and notes) | memory search with the state vector | "cosa ti ricorda?" ("what does it remind you of?") ([06-memory.md](06-memory.md) §7). As a question type, `recall` was removed on 29/09/2026 |
| `known` (not kept, abandoned on 29/09/2026) | already seen / similar / new | thresholds on similarity with the memory | "è qualcosa che hai in memoria?" ("is it something you have in memory?") ([06-memory.md](06-memory.md) §7) |
| `surprise` (removed, see §4) | how unexpected the state is | the model's log-probability of the state tokens | anomalies in emails or frames |
| uncertainty | entropy, margin, agreement across permutations | partly already computed (`status` against `min_confidence`) | difficulty of the decision |

### With a trained head

`span`, `locate` and `why` are also on the roadmap in a **zero-shot** version, without a trained head (§3 and [02-implications-and-proposal.md](02-implications-and-proposal.md) §6).

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
| Dynamic depth | ✗ | ✗ | removed on 29/09/2026 (real zero-shot gain 0.5–4%) | – |
| Image state | ✗ | ✗ | ✓ | ✓ |
| `estimate`, `short_answer` | ✗ (only the expected value of `score`) | ✗ | ✓ zero-shot | ✓ |
| State vector and memory | ✗ | ✗ | ✓ (`/v1/embed`, `memory`) | ✓ |
| `span`, `locate`, `value`, `tags` | ✗ | ✗ | – | with training |
| Customer fine-tuning | ✗ | ✓ | ✓ | ✓ |
| Local execution | ✗ (cloud) | ✓ | ✓ | ✓ |

## 3. Proposed priorities

1. **The state vector for real-time video.** The frame vector comes for free in the same pass: the questions are asked only when the scene changes.
2. **`span`** (on the roadmap, right after F0.5): values copied verbatim from the state, without training. The `short_answer` completion accepts only tokens that continue a piece present in the state, so the answer cannot be made up.
3. **`locate`** (on the roadmap, together with the left/right investigation): "yes, here" is much more useful than "yes". Zero-shot, from the coordinates that Qwen vision models can produce.
4. **`why`** (on the roadmap, together with the operator interface): the sentences of the state that drove the decision, found by removing one at a time.
5. **`short_answer`**: cheap and zero-shot. (`multi` abandoned on 29/09/2026: use N Yes/No questions.)

## 4. Implementation and verification (zero-shot)

| Type | How it is built | Response |
|---|---|---|
| `estimate` | ordinal readout like `score`, over the bins; `null` = open end | `value` (expected value over the midpoints), `interval` (10–90%, piecewise uniform), `range`, `probabilities`, `unit` |
| `short_answer` | softmax over the whole vocabulary (special tokens excluded). Candidates are completed up to the word boundary: one prefill + ≤4 greedy tokens, with a duplicated cache | `answer`, `candidates` with `p` (probability of the first token), `confidence` |
| state vector (`POST /v1/embed`) | mean of the final hidden states over the state tokens, normalized | `embedding`, `embedding_dim` |

**Verification** ([scripts/verifica_primitive.py](../../scripts/verifica_primitive.py), 2B-Base, thresholds fixed before seeing the results). A primitive is kept only if it passes its threshold.

| Primitive | Test | Threshold | Result | Outcome |
|---|---|---|---|---|
| `rank` | 4 situations with an obvious priority action | ≥ 3/4 | **4/4** | kept, then removed on 29/09/2026: the same test is now done with `choice` |
| `estimate` | 5 quantities written in the text (days, people, euros, kg, minutes) | ≥ 4/5 within the bin | **5/5** (15.4 people, €1,303, 7.7 kg, 39 min) | kept |
| `short_answer` | 6 one-word answers (day, language, city, color, month, surname) | ≥ 5/6 | **6/6** | kept |
| `embed` | retrieval of cases with the same decisions (kNN on typed-decisions) | better than the Prior | **0.564** vs 0.478, on par with Qwen3-Embedding-0.6B | kept (now through `/v1/embed` and the memory) |
| `surprise` | 6 normal texts vs 6 anomalous ones | AUROC ≥ 0.85 | **0.50** | **removed** |
| `multi` | tickets and photos | – | 0.8B: "yes" to almost everything (yes-bias, not fixable with temperature) | **removed** |

**Why `surprise` does not work.** Perplexity measures how *predictable* a text is, not how *anomalous* it is for the use case:
- *Lorem ipsum* has a perplexity of 2.9, because the model knows it extremely well;
- a prompt injection written in good Italian scores 12.7, lower than many normal tickets (18–70);
- only random text and shuffled words come out as "surprising".

**Known limits of the primitives we kept:**
- `short_answer` answers with a single "word" or value. The probability refers to the first token; the completion continues up to the first space (within 16 tokens), so dates, amounts and codes come out whole (a full date in "dd.mm.yyyy" format, the amount "14,21"). It also works with images ([04-image-states.md](04-image-states.md) §7).
- the state vector costs one extra pass over the state.

## 5. Usage examples

All the examples are in [examples/primitive/](../../examples/primitive/). The outputs below are **real** (Qwen3.5-2B-Base, 2 permutations, RTX 4070 Laptop). A single request can mix questions of different types ([examples/primitive_it.json](../../examples/primitive_it.json)).

> **`rank` is gone** (29/09/2026). To order options, use `choice`: it already returns the probability of each one, and sorting them is one line of code. The question "Quale azione va fatta per prima?" ("Which action should be taken first?") in [examples/primitive_it.json](../../examples/primitive_it.json) is now a `choice`.

### `estimate`: estimating a quantity (*An estimate* in the web UI)

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/primitive/estimate.json
```

```json
{"state": "Riunione di progetto: 12 persone in sala A e altre 3 collegate da remoto. Due colleghi hanno avvisato che arriveranno in ritardo.",
 "questions": {"partecipanti": {"type": "estimate",
   "instructions": "Quante persone partecipano in totale alla riunione, contando anche chi è collegato?",
   "criteria": {"bins": [0, 5, 10, 14, 16, 20, null], "unit": "persone"}}}}
```

- `bins` are the bin edges, in increasing order. `null` as the last (or first) edge means "or more" (or "less than").
- You can also pass just the list: `"criteria": [0, 5, 10, 14, 16, 20, null]`.

```json
"partecipanti": {"type": "estimate", "value": 14.59, "interval": [10.86, 18.97], "range": "14-16",
  "probabilities": {"0-5": 0.014, "5-10": 0.028, "10-14": 0.273, "14-16": 0.460, "16-20": 0.168, ">=20": 0.057},
  "confidence": 0.565, "unit": "persone"}
```

**How to read it:**
- `range` is the most likely bin (the right answer is 15: correct);
- `value` is the point estimate (expected value);
- `interval` contains the answer with 80% probability.

Choose the bins according to the precision you need: the narrower they are, the harder the question.

**`estimate` estimates, it does not read** (test of 29/09/2026, 2B, default readout). On the total of the test receipt (14.21 €), with 10 € bins from 0 to 50:
- at 448 px the most likely bin is wrong (20–30, 30%), with `value` 27.6;
- at 800 px it is right (10–20, 45%), but `value` is 23.4.

The distribution is wide, and the open top bin ("50 or more", counted as 50) drags the mean outside the most likely bin. The same question as `short_answer` reads **14,21** with confidence 0.98–0.99 at both resolutions. So:
- `short_answer` for a number written in the state (totals, dates, codes);
- `estimate` for a quantity to estimate (people, age, distances).

The web UI says so under "An estimate" questions ([07](07-web-interface.md) §3).

### `short_answer`: short answer, without listing the options (*A short answer* in the web UI)

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base examples/primitive/short_answer.json
```

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ... Abbiamo stipendi da pagare venerdì. ...",
 "questions": {
   "giorno": {"type": "short_answer", "instructions": "Entro quale giorno della settimana vanno pagati gli stipendi?", "top_k": 3},
   "lingua": {"type": "short_answer", "instructions": "In che lingua è scritto il messaggio?", "top_k": 3}}}
```

```json
"giorno": {"type": "short_answer", "answer": "Venerdì",
  "candidates": [{"text": "Venerdì", "p": 0.779}, {"text": "domenica", "p": 0.061}, {"text": "Giovedì", "p": 0.034}],
  "confidence": 0.779},
"lingua": {"type": "short_answer", "answer": "Italiano",
  "candidates": [{"text": "Italiano", "p": 0.913}, {"text": "Italieno", "p": 0.040}, {"text": "IT", "p": 0.018}],
  "confidence": 0.913}
```

**How to read it.** `answer` is the most likely word; `candidates` are the alternatives with their probabilities (`top_k`, from 1 to 20).
- **Good for:** extracting a name, a day, a city, a color, a language.
- **Not suited to:** multi-word answers or abstract concepts. For those, use `choice` with the options.

### Semantic vector of the state: `POST /v1/embed`

Since 29/09/2026 it is no longer requested inside `/v1/systemone` (`"embed": true`), but with a separate call to the model server:

```bash
curl -s localhost:8100/v1/embed -H 'Content-Type: application/json' \
  -d '{"state": "Buongiorno, sono tre giorni che i pagamenti ..."}'
```

```json
{"embedding": [0.0239, -0.0038, -0.0358, -0.0114, ...], "embedding_dim": 2048, "input_tokens": ..., "latency_ms": ...}
```

**How to use it:**
- the state can contain images (in base64), with optional `image_max_side`;
- the vector is normalized: the similarity between two states is their dot product;
- for a case archive, subtract the mean vector first (anisotropy).

The case memory uses exactly this vector ([06-memory.md](06-memory.md)). From Python, the same vector is obtained with `scorer.analyze_state(stato)`.

### From Python

```python
import json
from egeria.calibration import load_temperatures
from egeria.scorer import DecisionScorer

scorer = DecisionScorer("Qwen/Qwen3.5-2B-Base")  # device="cpu", dtype="float32" for CPU
temperatures = load_temperatures("runs/Qwen3.5-2B-Base/temperature.json")
body = json.load(open("examples/primitive_it.json"))
response = scorer.decide(body, permutations=2, temperatures=temperatures)
print(response["answers"]["priorita"]["choice"], response["answers"]["giorni"]["value"])
```

Temperatures are applied per readout: `estimate` uses the one for `score`. `short_answer` is not calibrated.
