[Italiano](../08-riuso-dello-stato.md) · **English**

# State reuse

> Status: **implemented** on 29/09/2026 (F0.5, item 1 of the roadmap in [02-implications-and-proposal.md](02-implications-and-proposal.md) §6). On by default, in `auto` mode.

## 1. Idea

All the questions and permutations about the same state start with the same prefix: system instructions and state, images included. Before, every prompt reprocessed everything, state and images, once per question and per permutation. Now:

1. the prefix is computed **once**, with the cache;
2. the hybrid cache is **duplicated** for the questions. It holds the recurrent state and the convolution of the Gated DeltaNet layers, and the keys/values of the attention layers;
3. **only the tails** go through in a batch: question type, question and options.

transformers can already continue several tokens from a Qwen3.5 cache: the convolution restarts from its own state and the delta rule from the saved recurrent state. The model does not need any changes.

## 2. Where the cut happens

- **Text.** The prefix is the leading part that is identical in all the token sequences. The tails are exactly the rest of each sequence, so the tokenization is identical by construction.
- **Images.** The Qwen processor expands images into tokens, so the text is cut right after the state block (`</state>` and a blank line). For every prompt we check that the cut does not change the tokenization; if it does, we fall back to full prompts. The M-RoPE positions of the tails are index + the prefix's `rope_deltas`.
- **Padding.** The batched tails are right-padded. With a causal model, positions after the last real token do not change the ones that are read.

## 3. Verification

Script: [scripts/bench_riuso_stato.py](../../scripts/bench_riuso_stato.py). For each request it compares full prompts (`never`) with the shared prefix (`always` or `auto`).

Requests used:
- the 12 of the Italian suite ([examples/suite_it.jsonl](../../examples/suite_it.jsonl));
- the 9 images of [examples/suite_immagini.jsonl](../../examples/suite_immagini.jsonl), with their questions grouped into one request per image, at 448 and 800 px.

30 requests and 77 decisions in all, with 2 permutations, on an RTX 4070 Laptop.

**The answers do not change:**

| Model | Same answers | Largest probability difference |
|---|---|---|
| 2B-Base, GPU bf16 | 77/77 | 0.026 |
| 0.8B-Base, GPU bf16 | 77/77 | 0.030 |
| 0.8B-Base, CPU float32 (text only) | 25/25 | 0.0000 |

The GPU differences are bf16 noise: on CPU in float32 the two paths coincide. The integration test `test_state_reuse_matches_full_prompts` checks this on every run with the model.

**Time of the shared prefix relative to full prompts**, by band of avoided tokens (prefix × extra rows; below 1 = faster):

| Avoided tokens | 2B, GPU | 0.8B, GPU | 0.8B, CPU |
|---|---|---|---|
| fewer than 500 | 1.12 | 1.77 | 0.77 (243–535 tokens) |
| 500–1000 | 0.58 | 0.93 | – |
| 1000–2000 | 0.38 | 0.62 | – |
| 2000–4000 | 0.22 | 0.29 | – |

**What emerges:**
- **On GPU the shared prefix has a fixed cost:** one extra pass, i.e. a few tens of milliseconds of kernel launches. It pays off only when enough tokens are avoided. With the short texts of the suite, 2–3 questions and states of a few lines, sharing was slower (up to +33% on the 2B and +95% on the 0.8B).
- **With images the gain is large,** because there are many visual tokens: on the 2B an 800 px photo with its questions goes from 1.1–1.5 s to 0.27–0.32 s, i.e. **75% to 81% less**.
- **The break-even point depends on the size:** ~500 avoided tokens on the 2B, ~1000 on the 0.8B, where each token costs less.
- **On CPU compute dominates** and sharing always pays: 15% to 36% less even on short texts.

## 4. The `auto` mode rule

The prefix is shared if the avoided tokens (prefix × extra rows) exceed the threshold:

| Device | Threshold |
|---|---|
| GPU, text model under 1.2 billion parameters (0.8B) | 1000 tokens |
| GPU, larger models (2B and up) | 500 tokens |
| CPU | always, with at least 2 rows |

Result with `auto`, relative to full prompts (median):

| Model (GPU) | Text | Images |
|---|---|---|
| 2B-Base | 1.00 (no loss) | **0.39** (down to 0.20) |
| 0.8B-Base | 1.01 | **0.61** (down to 0.24) |

**Realistic text workload: typed-decisions.** 40 test cases, 5 questions each, 2 permutations, 2B on GPU, two runs per mode (`egeria predict --limit 40 --share-state never|auto`):

| | Full prompts | `auto` |
|---|---|---|
| Time per case | 500–516 ms | **233–247 ms** (−53%, about twice as fast) |
| Same answers | – | 197/200 |

The 3 different answers were near ties: for example 0.288 vs 0.281 for the top two options. bf16 noise tips them one way or the other. Here the prompts are long (~720 tokens per question) and there are many questions, so `auto` shares the prefix on text too.

The thresholds were measured on a single GPU (RTX 4070 Laptop). On different hardware the break-even point may move: the script measures it again.

## 5. Usage

- **By default** the scorer uses `auto`.
- **From the command line:** `--share-state auto|always|never` on `decide`, `predict`, `suite`, `memory` and `model-server`.
- **From Python:** `DecisionScorer(..., share_state="auto")`, or `scorer.share_state = "never"` to compare.

```bash
# measure again on another machine (GPU: text and images; --cpu: text only, float32)
.venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-2B-Base
.venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-2B-Base --mode auto
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-0.8B-Base --cpu
```

## 6. What is left out

- **`short_answer` questions** still each do their own full pass: first the whole prompt, then the candidates are completed. They could start from the same prefix cache.
- **The state vector for memory** (`/v1/embed`) is a separate pass over the same state. The web server asks for it before the decision, in two separate HTTP calls.
- **The fixed cost per pass** remains the limit on short texts on GPU. It is what CUDA graphs would address ([04-image-states.md](04-image-states.md) §4).
