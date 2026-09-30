[Italiano](../03-baseline-f0.md) · **English**

# F0 baseline: zero-shot readout on Qwen3.5 and dynamic depth diagnostics

> Status: **implemented**. Results are in §6 and are updated with every run.
>
> **Dynamic depth removed on 29/09/2026.** Intermediate readouts, early exit and per-layer temperatures (§1 item 6, §6.2, §6.4) are no longer in the code. The results are kept here for reference. `min_confidence` remains and flags uncertain answers (`status`), computed on the full model.

## 1. What it does

Phase F0 turns a Qwen3.5 into a decision model compatible with the Jev API, **with no training at all**. It is the same method as SemIf:

1. For each question, a prompt is built that contains the state, the question and the options labeled A, B, C, …, using the chat template in non-thinking mode.
2. **A single forward pass** per prompt, with no token generation.
3. The final hidden state at the last position is multiplied **only by the lm_head rows of the letters**. The softmax is restricted to the declared options.
4. With `--permutations P`, each question is evaluated under P different option orderings: cyclic rotations for `noul`/`choice`, forward and reversed order for `score`. Log-probabilities are averaged per option, which reduces position bias.
5. **Temperature scaling** per question type, fitted on separate data (the train split).
6. **Intermediate readouts** (removed on 29/09/2026). The hidden states at the output of the full-attention layers go through the final norm and the letter rows (logit lens). They are used to study per-assertion dynamic depth (see [02 §4bis](02-implications-and-proposal.md)).

## 2. Code structure

| File | Contents |
|---|---|
| [src/egeria/schema.py](../../src/egeria/schema.py) | Validation of the `/v1/systemone` body (noul, choice with up to 26 options, score with 2 to 10 levels) |
| [src/egeria/prompt.py](../../src/egeria/prompt.py) | Prompts with lettered options and orderings for the permutations |
| [src/egeria/scorer.py](../../src/egeria/scorer.py) | `DecisionScorer`: model loading (text only, no vision tower), batched forward with right padding, state reuse ([08](08-state-reuse.md)), final readout, `decide()` in Jev format |
| [src/egeria/confidence.py](../../src/egeria/confidence.py) | Softmax with temperature, Jev `confidence` formulas, response format |
| [src/egeria/calibration.py](../../src/egeria/calibration.py) | Temperature scaling (golden-section search on 1/T; the NLL is convex) |
| [src/egeria/metrics.py](../../src/egeria/metrics.py) | Accuracy, NLL, KL, Brier, ECE, coverage at 5% error, score MAE, flip rate |
| `depth.py` (removed on 29/09/2026) | Temperatures per (type, layer), per-layer quality, threshold-based early-exit simulation |
| [src/egeria/datasets.py](../../src/egeria/datasets.py) | Loader for `LocalLLaMA/typed-decisions` (`--limit` samples evenly spaced across all workflows) |
| [src/egeria/cli.py](../../src/egeria/cli.py) | `egeria` CLI |
| [scripts/baseline.sh](../../scripts/baseline.sh) | Full pipeline for one or more models (resumable) |
| [scripts/run_all.sh](../../scripts/run_all.sh) | Model download + baselines in sequence, meant to run detached from the session |

## 3. Installation

```bash
cd /home/kuduk/egeria
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[model,eval,dev]"
.venv/bin/egeria info        # GPU, free VRAM, fast-kernel status
```

### Fast kernels for the Gated DeltaNet layers (recommended)

Without these packages, transformers uses the PyTorch reference implementation. It is correct but much slower: more than 10× on `chunk_gated_delta_rule`, according to transformers.

```bash
# flash-linear-attention: Triton kernels, no compilation
uv pip install --python .venv/bin/python "flash-linear-attention>=0.2.2"

# causal-conv1d: CUDA extension. It must see the venv's torch (--no-build-isolation).
# Without a prebuilt wheel for the installed torch version, it compiles from source (~10-15 min).
uv pip install --python .venv/bin/python ninja packaging wheel setuptools
CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=4 \
  uv pip install --python .venv/bin/python --no-build-isolation "causal-conv1d>=1.5"
```

`TORCH_CUDA_ARCH_LIST="8.9"` corresponds to the RTX 4070 (Ada). Adjust it for other GPUs: 8.0 for A100, 9.0 for H100.

**CPU with the kernels installed.** transformers selects the kernel at import time and, with causal-conv1d installed, would also use it on CPU tensors, where it fails. The scorer installs a dispatcher (`install_device_dispatch`): fast kernels on CUDA, the PyTorch reference implementation elsewhere. Both CPU and GPU therefore work.

Times observed on 26/09/2026: flash-linear-attention 0.5.2 installs in a few seconds. causal-conv1d 1.7.0 compiles from source in about 6 minutes, because there is no prebuilt wheel for torch 2.11. Effect on the forward pass (0.8B, 224 px image, 3 questions): 77 → 68 ms with fla, → 45 ms with both; details in [04-image-states.md](04-image-states.md) §4.

### 4-bit quantization (for 4B and 9B on 8 GB)

```bash
uv pip install --python .venv/bin/python -e ".[quant]"
.venv/bin/egeria predict --model Qwen/Qwen3.5-4B --quantize 4bit ...
```

### Notes on WSL2

- **VRAM is shared with the Windows desktop**, which takes 1–4 GB of it. `egeria info` shows how much is actually free.
- **When VRAM runs out, the WSL driver can "spill over" into shared system memory** instead of raising an error, and becomes very slow. If a run suddenly slows down, reduce `--batch-tokens`.
- **By default WSL sees half of the host's RAM.** To offload large models, raise it with `memory=48GB` in the `[wsl2]` section of `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.

## 4. Usage

Ready-made examples in [examples/](../../examples/): `ticket_it.json` (a ticket in Italian) and `jev_esempio.json` (the example from the Jev documentation).

```bash
# A request in Jev format (on GPU)
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B --permutations 2 examples/ticket_it.json

# On CPU (GPU busy or absent): use float32. The 0.8B takes ~5 s for 3 questions × 2 permutations
.venv/bin/egeria decide --model Qwen/Qwen3.5-0.8B-Base --device cpu --dtype float32 --permutations 2 \
  --temperatures runs/Qwen3.5-0.8B-Base/temperature.json examples/ticket_it.json

# Comparison table across models
.venv/bin/egeria compare runs/*/report-cal.json

# Predictions on typed-decisions
.venv/bin/egeria predict --model Qwen/Qwen3.5-0.8B --split test --permutations 2 --out runs/x/test.jsonl

# Temperatures fitted on train (one per question type)
.venv/bin/egeria calibrate --predictions runs/x/train.jsonl --out runs/x/temperature.json

# Report
.venv/bin/egeria evaluate --predictions runs/x/test.jsonl --temperatures runs/x/temperature.json

# Full pipeline, detached from the session
setsid nohup scripts/run_all.sh > runs/run_all.log 2>&1 &

# Tests (the ones that need the model only on request)
.venv/bin/pytest -q
EGERIA_TEST_MODEL=Qwen/Qwen3.5-0.8B .venv/bin/pytest -q -m model
```

## 5. Technical details and verified pitfalls

- **Right padding with the DeltaNet layers is safe.** The model is causal, so padding after the last position does not affect earlier tokens. In fp32, batched and single-sample runs agree to 1e-5.
- **bf16 noise.** In bf16 the letter logits vary by up to ~0.2 when the batch shape changes (0.8B). This is numerical noise, not a bug. It can flip the argmax in borderline cases. For fine-grained comparisons use `--dtype float32`: the 0.8B in fp32 takes ~3.2 GB.
- **Answer boundary.** At startup we check that, for every letter, `tokenize(prompt + lettera) == tokenize(prompt) + [token lettera]` (*lettera* = letter). All letters A–Z are single tokens in the Qwen3.5 tokenizer.
- **Text-only model.** `Qwen3_5ForCausalLM` is loaded with the text config: the vision tower is not loaded.
- **Efficient readout.** The lm_head is not computed over the full vocabulary (248k): the final hidden state is multiplied only by the 26 letter rows, in fp32.
- **Hooks for the intermediate readouts.** They capture only the hidden state at the last position of each batch row, so the extra memory is negligible. The output of the last layer, after the norm, matches the final readout (verified).
- **Metric quality.** The typed-decisions gold is the average of 3 samples from a "4B-class" teacher, with a self-agreement ceiling of 0.735. Temperatures fitted on the train split make the result "calibrated on the workload" rather than purely zero-shot. Both T=1 and calibrated results are always reported.

## 6. Results (26/09/2026)

**Setup:**
- Benchmark: typed-decisions, full test split (400 cases, 2,000 decisions). 2 option permutations per question, in bf16 on an RTX 4070 Laptop, with the PyTorch fallback kernels.
- Temperatures fitted on 300 train cases (1,500 decisions, evenly spaced sample across the 4 workflows).
- Reproducible with `scripts/run_all.sh`.

### 6.1 Quality

| Model | T | Acc | noul | choice | score | NLL | KL | Brier/k | ECE | Flip | ms/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Prior** (train gold frequencies, does not read the state) | – | **0.497** | – | – | – | **1.025** | **0.327** | 0.175 | 0.033 | – | 0 |
| Qwen3.5-0.8B | 1 | 0.428 | 0.518 | 0.490 | 0.314 | 1.231 | 0.641 | 0.214 | 0.244 | 0.71 | 474 |
| Qwen3.5-0.8B | fit | 0.428 | 0.518 | 0.490 | 0.314 | 1.097 | 0.378 | 0.189 | 0.031 | 0.71 | 474 |
| Qwen3.5-0.8B-Base | 1 | 0.450 | 0.508 | 0.483 | 0.380 | 1.185 | 0.523 | 0.209 | 0.122 | 0.44 | 504 |
| Qwen3.5-0.8B-Base | fit | 0.450 | 0.508 | 0.483 | 0.380 | 1.121 | 0.392 | 0.192 | 0.053 | 0.44 | 504 |
| Qwen3.5-2B | 1 | 0.469 | 0.665 | 0.398 | 0.375 | 1.239 | 0.616 | 0.200 | 0.176 | 0.38 | 722 |
| Qwen3.5-2B | fit | 0.469 | 0.665 | 0.398 | 0.375 | 1.098 | 0.377 | 0.185 | 0.061 | 0.38 | 722 |
| Qwen3.5-2B-Base | 1 | 0.468 | 0.517 | 0.522 | 0.390 | 1.195 | 0.632 | 0.209 | 0.226 | 0.23 | 700 |
| Qwen3.5-2B-Base | fit | 0.468 | 0.517 | 0.522 | 0.390 | **1.049** | **0.346** | 0.183 | **0.029** | 0.23 | 700 |

**How to read the columns:**
- *Brier/k* is the mean per-option Brier score against the hard label.
- *Flip* is the fraction of decisions in which the two permutations give a different argmax.
- *ms/case* is the time for 5 questions × 2 permutations = 10 prompts.
- The Prior here (0.497) is computed on our train sample. The dataset reports 0.470 under its own definition.

**Findings:**
1. **Zero-shot, no small model beats the Prior.** Even the best one, the calibrated 2B-Base, falls just short of it on NLL and KL. The models read the state only marginally: the 2B-Base beats the Prior only on customer service (0.616). On invoices they all collapse (0.28–0.48), because these require numerical comparisons in a single pass. This is consistent with the literature: Laya base zero-shot ≈ 0.36, and SemIf uses the 4B.
2. **Temperature scaling always works.** ECE drops to 0.03–0.06. The models are overconfident (fitted T of 1.7–7). For `noul` on the 0.8B-Base the temperature hits the limit (T = 20): that readout carries no information, and calibration flattens it to 0.5.
3. **Base beats Instruct on robustness to option order** (flip 0.44 vs 0.71 on the 0.8B, 0.23 vs 0.38 on the 2B) and, after calibration, on NLL and KL for the 2B. This confirms the choice of starting training from the Base checkpoints.
4. **Order sensitivity decreases with model size** (0.71 → 0.38 Instruct; 0.44 → 0.23 Base). Permutations are still necessary.

### 6.2 Dynamic depth (readouts at block boundaries: layers 3, 7, 11, 15, 19, 23)

| Model | Compute with **oracle** exit | Potential savings | Threshold 0.4: compute / agreement with full model | Threshold 0.2: compute / agreement with full model |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.694 | 31% | 0.992 / 100% | 0.952 / 96.8% |
| Qwen3.5-0.8B-Base | 0.654 | 35% | 0.992 / 100% | 0.965 / 99.7% |
| Qwen3.5-2B | 0.727 | 27% | 0.989 / 100% | 0.829 / 78.3% |
| Qwen3.5-2B-Base | 0.643 | **36%** | 0.970 / 100% | 0.847 / 84.9% |

The **oracle** exits at the first layer from which the argmax no longer changes through the last layer: it is the maximum saving with unchanged decisions.

The thresholds apply to the **chance-corrected** confidence, the same scale as `min_confidence` (§6.4): 0 = chance, 1 = certainty.

**Findings:**
1. **The potential is there: 27–36% of compute can be saved** without changing a single decision. About half of the decisions are already final at layer 15 of 24; in the 2B-Base, 222 of 2,000 are already final at layer 3.
2. **The zero-shot confidence criterion does not capture it.** At intermediate layers the calibrated logit lens is unsure even when it already has the answer.
   - At threshold 0.4 only 1–3% is saved, but with 100% agreement with the full model: the criterion is safe but too conservative.
   - Lowering the threshold to 0.2 saves up to 17%, but some exits change the answer (78–85% agreement on the 2B models), at equal accuracy.
3. **Implication for F1.** **Trained exit heads** are needed at every block boundary to make the intermediate confidence reliable. The measurable goal is to close the gap between the savings achieved and the oracle.

**On the Italian suite** (2B-Base, cases more clear-cut than the benchmark's, per-layer temperatures fitted on typed-decisions):
- **Compute:** oracle **0.60** (40% can be saved); with threshold 0.6, **0.90** (10% saved).
- **Where decisions are made.** Clear-cut `choice` questions are decided at **layer 15 of 24** with already high confidence, sometimes higher than at the last layer:
  - "nuova spedizione" ("new shipment") 0.91 at layer 15 vs 0.83 at the end;
  - "logistica" ("logistics") 0.91 vs 0.89;
  - "cambio pagamento" ("payment method change") 0.79 vs 0.91;
  - "successo" ("success") 0.71 vs 0.74.
- **The Base model's `noul` answers stay at 0.50–0.63 at every layer**: confidence is never high enough to exit, even when the answer is stable from layer 15 on.
- **Early exits inherit the full model's errors, but add none.** For example, "newsletter" for the scam email is already at 0.76 at layer 15, and it is wrong at the end too.
- **Layers 3–11 are not reliable.** The answer changes often and confidence is low: in the 2B, the decision forms between block 3 (layer 11) and block 4 (layer 15).

### 6.3 2B vs 0.8B comparison

**Paired comparison on the benchmark.** `egeria paired`: difference B − A on the same 2,000 decisions, with fitted temperatures and a 95% bootstrap interval resampled by case. An asterisk marks an interval that excludes zero.

| Comparison | Δ accuracy | Δ NLL | Δ Brier | Where it changes most |
|---|---|---|---|---|
| 2B-Base − 0.8B-Base | +0.018 [−0.002, +0.039] | **−0.071*** | **−0.037*** | customer service +0.088*; security −0.054* |
| 2B − 0.8B (Instruct) | **+0.041*** | +0.000 | −0.006 | noul +0.147*, score +0.061*, invoices +0.200*; choice −0.092*, security −0.084* |
| 2B-Base − 2B (Instruct) | −0.002 | **−0.048*** | **−0.018*** | choice +0.123*; noul −0.148*, invoices −0.122* |

For NLL and Brier, a negative value is better.

**Italian suite** ([examples/suite_it.jsonl](../../examples/suite_it.jsonl)): 12 cases, 25 decisions with a clear-cut answer, including traps (negations, amounts that don't add up, destructive actions). Run it with `egeria suite`.

| Model | Correct | noul | choice | score | mean p(expected) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 12/25 | 4/10 | 7/11 | 1/4 | 0.46 |
| Qwen3.5-0.8B-Base | 13/25 | 4/10 | 7/11 | 2/4 | 0.45 |
| **Qwen3.5-2B** | **17/25** | **8/10** | 8/11 | 1/4 | **0.55** |
| **Qwen3.5-2B-Base** | **17/25** | **8/10** | 7/11 | 2/4 | 0.51 |

**Findings:**
1. **The 2B is better than the 0.8B, but in different ways depending on the metric.**
   - On the benchmark, accuracy rises only slightly (+2–4 points); the improvement is mostly in the probabilities (NLL and Brier).
   - On the Italian suite the jump is clear (+4–5 answers out of 25).
   - The largest gain is on `noul`: the 0.8B models answer yes/no almost at random (p ≈ 0.5), while the 2B goes from 4/10 to 8/10.
2. **The 2B handles negations** ("non vuole il rimborso" ("does not want the refund"), "non voglio disdire" ("I don't want to cancel"), with p up to 0.92). The 0.8B models do not.
3. **Instruct and Base have complementary profiles**: Instruct is stronger on `noul`, Base on `choice`, with better probabilities. Training starts from Base regardless (§6.1).
4. **Errors shared by all models** (they need training or a decomposition into atomic questions):
   - the invoice with a wrong amount gets "paid", even though the 2B correctly answers that the amount does **not** match. This is a cross-question inconsistency, the same problem documented for Jev. It confirms the "harness" pattern: atomic questions, with the logic in code;
   - the agent that deletes the database is judged a "success";
   - the harmless alert gets escalated;
   - the scam email: the 2B answers "not phishing".
5. **`score` questions remain weak for all models** (1–2 of 4): ordinal scales are the hardest primitive, as for Laya.

### 6.4 Real dynamic depth with per-assertion `min_confidence`

> **Removed on 29/09/2026.** This section describes block-by-block execution, which is no longer in the code. What remains today is the threshold part: `min_confidence`, with the same precedence and the same scale described below, compared against the full model's confidence. The response reports `min_confidence` and `status`, no longer `depth`.

Beyond the simulation, the scorer actually runs the model **block by block** (`DecisionScorer.score_adaptive`):
1. At each exit point it computes, for each question, the distribution combined across permutations, calibrated with that layer's temperature.
2. Questions that reach **their own** `min_confidence` exit. Their rows are removed from the batch, and the batch is trimmed to the longest remaining sequence: with right padding on a causal model this is safe.
3. Masks and rotary embeddings are recomputed for the reduced batch.

With no exits, the result matches the standard forward pass exactly (verified in fp32: difference 0.0).

**API** (an extension of the Jev format; without `min_confidence` the behavior is identical to before):

```json
{
  "state": "...",
  "min_confidence": 0.6,
  "questions": {
    "urgente":  {"type": "noul", "instructions": "...", "min_confidence": 0.55},
    "reparto":  {"type": "choice", "instructions": "...", "criteria": {"...": "..."}},
    "rischio":  {"type": "score", "instructions": "...", "criteria": ["..."], "min_confidence": null}
  }
}
```

**Threshold precedence**, from most specific to most general:
1. **Question:** the local value always overrides the global one. Explicit `null` = full model.
2. **Request:** the global `min_confidence` field. Explicit `null` = full model.
3. **Server:** `--min-confidence` on the CLI, or the `default_min_confidence` argument of `decide()`.
4. **None of the above:** **conservative default**, i.e. the full model with no early exits.

**Semantics and response:**
- **What the threshold measures.** `min_confidence` is compared against the **chance-corrected confidence**, the same value as the `confidence` field of the response: 0 = uniform distribution, 1 = certainty.
  - `choice`: (n·p_max − 1)/(n − 1), the Jev formula.
  - `noul`: 2·p_max − 1. The `confidence` field for `noul` is an Egeria extension.
  - `score`: the Jev formula based on distance from the mode.

  A first version compared the threshold against p_max. That was wrong: for `noul`, p_max ≥ 0.5 always, so a threshold of 0.5 would let a coin-flip answer exit. With the corrected scale, the same threshold means the same thing with 2 options or with 20.
- **Limitation for `score`.** For `score`, the Jev formula clips very flat distributions to 0. These never pass any positive threshold and always reach the last layer, marked `uncertain`.
- **Where exits happen.** Only at block boundaries from roughly half depth onward: layers 11, 15, 19 and 23 of 24. Before that, the answer has not formed yet (§6.2).
- **Response fields.** Each answer reports `depth` (layers used) and, if it has a threshold, `min_confidence` and `status`:
  - `decided`: threshold reached;
  - `uncertain`: not even the last layer reaches it. This is the signal to escalate to a larger model, to thinking mode or to a human.
- **Temperatures.** Per-layer temperatures are required (`egeria calibrate` on predictions made with `--exits blocks`). The file also stores the layers they refer to.

**First measurements** (2B-Base, Italian suite, 2 permutations, default threshold for all questions):

| `min_confidence` | Correct | Mean depth | `uncertain` | Time |
|---|---|---|---|---|
| none (conservative default) | 17/25 | 1.00 | – | 1.85 s |
| 0.6 | 17/25 | 0.93 | 20/25 | 1.89 s |
| 0.4 | 17/25 | 0.90 | 16/25 | 1.83 s |
| 0.2 | 16/25 | **0.83** | 9/25 | **1.73 s** |

**Latency vs compute: the request waits for its deepest question.** The questions of a request run in the same batch, so the response arrives when the deepest one finishes. Exiting early reduces the number of batch rows in the later layers, but the latency depends on how much a layer with fewer rows actually costs.

Measured on the ticket, 3 questions × 2 permutations, exit forced at layer 11 (12 of 24 layers used), median over 9 runs:

| Hardware / model | All full (24) | All exit (12) | Mixed: 2 exit, 1 full | Share of the final stretch still paid |
|---|---|---|---|---|
| GPU, 2B-Base bf16 | 188 ms | 106 ms (−44%) | 150 ms (−20%) | 54% |
| GPU, 0.8B-Base bf16 | 130 ms | 66 ms (−49%) | 93 ms (−29%) | 41% |
| CPU, 0.8B-Base fp32 | 4.046 s | 2.058 s (−49%) | 2.676 s (−34%) | **31%** ≈ 1/3 of the rows |

The last column is (mixed − all exit) / (full − all exit): how much of the cost of layers 12–24 is still paid when only 1 question out of 3 remains.

- **When all questions exit, latency really does halve**, on both GPU and CPU.
- **On CPU the cost is proportional to the number of rows**: with 1 question out of 3 in the final stretch, you pay about 1/3 (31%).
- **Not on GPU.** With small batches the GPU is underutilized: removing 2/3 of the rows saves only half of the final stretch (41–54% still paid). The deepest question dominates latency.
- **Design implications:**
  1. on GPU, the **latency** gain materializes when *all* the questions of the request exit;
  2. the **compute and throughput** gain is always real, and it matters on CPU and on a server handling many requests;
  3. to turn it into a latency gain as well, we need:
     - **progressive responses**: return each answer as soon as it exits, via streaming, so the caller can already act on the easy decisions;
     - **block-level continuous batching**: rows that have exited are replaced by rows from other requests (F4).

Full example: [examples/ticket_it_soglie.json](../../examples/ticket_it_soglie.json).

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 \
  --temperatures runs/Qwen3.5-2B-Base/temperature.json examples/ticket_it_soglie.json
.venv/bin/egeria suite --model Qwen/Qwen3.5-2B-Base --permutations 2 \
  --temperatures runs/Qwen3.5-2B-Base/temperature.json --min-confidence 0.4 examples/suite_it.jsonl
```

### 6.5 F0 conclusion

Zero-shot readout on the small Qwen3.5 models is not enough. The harness, metrics, calibration and depth diagnostics are ready and verified. The next step is **F1: training on the small models** (LoRA + readout + exit heads), see [02 §6](02-implications-and-proposal.md). The 4B and the 9B at 4 bits remain an optional data point.
