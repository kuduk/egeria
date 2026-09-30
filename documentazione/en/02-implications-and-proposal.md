[Italiano](../02-implicazioni-e-proposta.md) · **English**

# Implications for Egeria and starting proposal

> This document turns the [state of the art](01-state-of-the-art.md) into a concrete proposal. The decisions still open are at the end (§7).

## 0. Decisions made

| Date | Decision | Rationale |
|---|---|---|
| 26/09/2026 | **Backbone: Qwen3.5 family** (like SemIf), not Qwen3.8 | Qwen3.8 has no small sizes and no Base checkpoints. Qwen3.5 has the same architecture. Qwen3.8-27B remains a candidate teacher |
| 26/09/2026 | **Start from the zero-shot F0 baseline**: readout of the letter logits, no training | Sets the reference to beat with LoRA and distillation. Implemented, see [03-baseline-f0.md](03-baseline-f0.md) |
| 26/09/2026 | **Development hardware: RTX 4070 Laptop (8 GB), WSL2** | 0.8B and 2B fit in bf16. 4B and 9B only quantized to 4 bits, or on a cloud GPU |
| 26/09/2026 | **New direction: dynamic depth per assertion** (§4bis), abandoned on 29/09/2026 | The user's idea. It fits a single-forward-pass model well |
| 26/09/2026 | **Per-assertion `min_confidence` + global field** (§4bis) | The caller knows the cost of an error. The local value overrides the global one. Since 29/09/2026 it only flags uncertain answers (`status`) |
| 28/09/2026 | **Italian and English on an equal footing** | The documentation is in both languages too |
| 28/09/2026 | **Generalist, but the first LoRA on a specific image task** | Images are the project's strongest point; a single task makes it clear what worked |
| 28/09/2026 | **One model size at a time** | Because of the 8 GB GPU, and to avoid confounding the effects |
| 28/09/2026 | **Name: Egeria** (formerly semLMM) | Numa's advisor nymph: a small, quick advisor for whoever decides |
| 29/09/2026 | **Simplification:** removed dynamic depth, `rank`, the `recall` question, `embed` inside `/v1/systemone`, `inject` | Real gain too small (dynamic depth, `inject`) or duplicates of something that stays (`rank` = sorted `choice`; `recall` = `memory.recall`; `embed` = `/v1/embed`) |
| 29/09/2026 | **`multi` and `known` abandoned as primitives** | `multi` is equivalent to N Yes/No questions, which are cheap with state reuse. For `known`, what remains is the out-of-domain calibration measurement (F0.5) and `memory.min_similarity` |
| 29/09/2026 | **Three new zero-shot primitives on the roadmap: `span`, `locate`, `why`** (§6) | Each gives something that cannot be obtained by combining the existing ones: an answer anchored to the text of the state, a position in the image, the reason behind a decision |
| 29/09/2026 | **Left/right out of the checks and out of the first LoRA** | According to the user, something is fundamentally wrong there: the example needs to be re-examined in a separate investigation |

## 1. What changes compared with the initial idea ("start from Qwen 3.8")

| Initial assumption | Reality (September 2026) | Consequence |
|---|---|---|
| Qwen 3.8 exists in several sizes | Only 27B dense, 2.4T MoE and Flash-Next (125B MoE). **No small size** | The fast *student* must start from **Qwen3.5 Base** (0.8B–9B), which has the same architecture as 3.8 |
| We can start from a base checkpoint | Qwen3.8 has **only post-trained versions**, which are more overconfident | Qwen3.8-27B should be used as a **teacher** or as a "large" student, not as the ideal starting point for calibration |
| We can copy Laya (encoder + one `[MASK]` per option) | Qwen is **causal and hybrid** (DeltaNet), so it does not turn into a BERT | Markers **after** the options and a pointer head (Kev style). Prefix-fork for multiple questions |
| RLCD is the key | With known labels, RL ≈ a supervised proper loss, only noisier | **Supervised training with soft CE + Brier + RPS**; RL only for bandit or environment feedback |
| Nobody has done it on Qwen yet | Kev-27B, Open-Jev-27B, decider and JevK5 already exist | A clear **differentiation** is needed (§2) |

## 2. Possible differentiators

On general English quality, Jev, decider and Kev are already on par. Areas where the field is weak:

1. **Italian and European multilingual.** Jev is "English-first". Laya-multilingual is weak (MASSIVE 0.366). The Qwen replications do not measure Italian. A decision benchmark in Italian **does not exist** today.
2. **Honest abstention.** No system handles unknowable items and aleatoric uncertainty well (Jev bluffs in 50% of cases). An explicit "not determinable" option can be trained, with dedicated data and a declared risk–coverage.
3. **Order invariance and consistency.** Flip rates of 13–49% in current systems. Options: permutation augmentation, a symmetrized readout and a consistency loss between equivalent or negated questions.
4. **A vertical domain.** For example Italian public administration (PA), legal, healthcare or customer care, with our own data: TypeSafe itself says the advantage lies in the data.
5. **API compatibility with Jev** (`/v1/systemone`, same `confidence` formulas). Existing tools would work out of the box: SDK, JevBench, harness.

## 3. Proposed architecture (v0)

```
                ┌────────────────────────────────────── shared prefix (computed once) ──┐
input:  <state> …state as text/JSON… </state>                                           │
                └───────────────────────────────────────────────────────────────────────┘
                     │ prefix-fork: copy of KV + conv/DeltaNet state for each question
      ┌──────────────┴──────────────┐
branch q1:  <q type=choice> instructions </q> <opt> A: … </opt> <opt> B: … </opt> … <decide>
branch q2:  <q type=noul> … </q> <opt> false: … </opt> <opt> true: … </opt> <decide>
      │
      ▼
pointer head:   logit_i = f(h(</opt>_i), h(<decide>))   →  restricted softmax / T_type
```

- **Backbone.**
  - Student: **Qwen3.5-4B-Base** (trade-off) or 2B-Base (latency), with LoRA r16–32 and the added special tokens.
  - Teacher: **Qwen3.8-27B** (4-bit locally, or via API).
- **Readout.** Kev-style pointer head: the hidden state of the end-of-option marker compared with that of the `<decide>` marker. The readout runs in fp32.
  - Zero-shot baseline: SemIf-style **letter logits**, no training.
- **Primitives.**
  - `choice`: softmax over the options.
  - `noul`: two false/true slots.
  - `score`: softmax over the levels, with expected value and RPS loss.
- **Multiple questions per state.** Prefix-fork of the hybrid state (see qwen-rlcd and Kev). In training we can start with one row per question, which is simpler, and optimize later.
- **Optional experiment.** Bidirectional mask on the full-attention layers only (dQwen3.5 style), as an ablation.

## 4. Proposed training

1. **Data.**
   - Public classification, NLI, safety and intent datasets, converted into the three primitives.
   - Synthetic states with **soft** labels from the Qwen3.8-27B teacher, with N samples and a calibrated temperature.
   - Dedicated Italian set.
   - Augmentation: option permutation, instruction paraphrases, state as text or as JSON, distractor questions, unknowable items.
2. **Loss.** Soft CE (log-score) + λ·Brier + RPS for `score`. Mix of hard (human) and soft (teacher) labels. Optional: consistency across permutations. This is also the recipe Laya actually uses: full-weight soft CE is the main term and the noisy policy gradient is an add-on ([01](01-state-of-the-art.md) §4.6).
3. **Post-hoc calibration.** Temperature per type and per option-count band (as in Laya: 2, 3–5, 6–10, 11+), fitted on a split held out from training **before** starting, **never** on the train set. Temperature does not correct a systematic bias (for example a yes-bias): that is corrected with balanced data, or with a bias term (shift) fitted on labels.
4. **Data design against shortcuts** (lessons from fine-tuning Laya as a web agent):
   - hard negatives (for "Yes": cases that look like the positive but are not) and reweighted rare classes;
   - "reverse" labels (answer first, then an LLM writes the state or the question that calls for it);
   - corrections on the model's own trajectories (DAgger) for uses in which the model chooses actions.
5. **RL (later phase, only if needed).** Bounded reward (Brier or spherical), leave-one-out baseline **without std normalization**, on real bandit-type feedback or on environments.

## 4bis. Dynamic depth per assertion

> **Abandoned on 29/09/2026.** Zero-shot, the real gain was only 0.5–4% (step 1 of the plan), and block-wise execution complicated the scorer: the code has been removed. `min_confidence` remains and flags uncertain answers. The section is kept as a record of the idea.

**Idea.** Each assertion (question) uses only the depth it needs. After each block of the model, the distribution over the options is read out. If the confidence *calibrated for that layer* exceeds a threshold, we exit; otherwise we continue. Easy questions cost a few layers, hard ones the whole model.

**Why it fits this project well.**
- **No KV-cache propagation.** In token-by-token generation, early exit is complicated (CALM, LayerSkip) because later tokens need the states of the skipped layers. Here there is **a single forward pass per question**, so exiting early is a net saving.
- **The architecture already has natural boundaries.** Qwen3.5 is organized in `3 × Gated DeltaNet + 1 × Attention` blocks: the natural exits are after each full-attention layer (6 exits in the 0.8B/2B, 8 in the 4B/9B).
- **In a batch, it is enough to drop the rows already decided** between one block and the next: no dynamic padding is needed.
- **It combines with the model cascade.** Dynamic depth inside the small model, then escalation to the large model or to "thinking" only if even the last layer remains uncertain. In the literature (NYU, arXiv 2609.24574) the System 1 → LLM cascade reaches frontier quality at 25–50% of the cost.

**Known risks** (from the early-exit literature):
- **Lower layers are overconfident or have logits on a different scale.** A **per-layer** temperature is needed (ACT, "annealed confidence threshold", 2026). Alternatively, a "patience" criterion: the same answer for k consecutive exits (PABEE).
- **The logit lens on intermediate layers is weak without training.** Training (F1) needs **trained exit heads** (tuned lens / LayerSkip), with a loss on every exit weighted by depth.
- **A quality guarantee is needed.** The threshold can be chosen with **conformal risk control** (CATs, Schuster et al. 2021), so that the early decision matches the full model's with probability ≥ 1 − ε.

**Plan.**
1. **F0 (done).** Zero-shot diagnostics: readout at every block boundary, temperatures per (type, layer), simulation of different thresholds → compute/accuracy curve. Result: the potential (oracle) saving is 27–36%, while the saving achieved with zero-shot confidence is only 0.5–4%, always with 100% agreement with the full model. See [03-baseline-f0.md](03-baseline-f0.md) §6.2.
2. **API and real exit (done, then removed on 29/09/2026; `min_confidence` with `status` remains).** Per-assertion `min_confidence` parameter, with a global field in the request and a server default. The local value overrides the global one; with no threshold, the conservative default applies, i.e. the full model. Block-by-block execution with removal of the questions already decided; the response carries `depth` and `status` (`decided`/`uncertain`). The threshold applies to the chance-corrected confidence (0 = chance, 1 = certainty). Latency measurement: the request waits for the deepest question. On GPU the latency gain only materializes if all questions exit; on CPU it is proportional. See [03-baseline-f0.md](03-baseline-f0.md) §6.4.
3. **F1 (no longer planned).** Trained exit heads at every block boundary, with a proper loss on all exits. Threshold chosen with conformal risk control, so that `min_confidence` becomes an empirical accuracy guarantee.
4. **F3 (no longer planned in its block-wise form).** Multi-level cascade: early exit → full model → larger model or thinking → human escalation. `status: uncertain` is the entry signal.

## 5. Proposed evaluation

- **Generalist (zero-shot):** typed-decisions (test, without training on the train split), JevBench public, Decision Index, BTZSC.
- **Robustness:** CLINC150 with OOS (abstention), Banking77 (cardinality), permutation flip rate, negated questions.
- **Label-free checks** (in the style of Laya's `research/eval`, plus two of our own):
  - identical options, to measure position preference and decide whether 2 permutations are needed or 1 is enough;
  - negation consistency: P(Yes | question) + P(Yes | negated question) ≈ 1, which measures the yes-bias;
  - mirrored image, for left and right.
- **Confidence:** AUROC against correctness and accuracy on the most confident half, in addition to ECE.
- **Multilingual:** XNLI-it, MASSIVE-it, plus a **new Italian decision set** to be built.
- **Metrics:** accuracy, NLL, Brier, ECE (declared binning), coverage at 5% error, p50/p95 latency of the model alone.
- **Baselines to beat:** SemIf zero-shot (same backbone), Kev-4B/9B, decider-4b, Laya-multilingual, Jev (via API, if there is budget).

## 5bis. Current limitations of Egeria (updated on 29/09/2026)

- **Zero-shot on typed-decisions we are at the level of the prior:** 0.468 vs 0.478. Above Laya base (0.362), but below a trivial baseline. What beats the prior is the memory vote (0.562).
- **Cost per question and per permutation:** partly solved on 29/09/2026 with state reuse ([08-state-reuse.md](08-state-reuse.md)). With images the time drops by up to 80%; on short texts on GPU the fixed cost of each pass remains, and `open` questions do not share the prefix yet.
- **Calibration does not transfer.** The typed-decisions temperatures make generic questions and images worse, there is no bias term (shift), and the yes-bias is not measured.
- **The latency profile differs from Laya's.** A 0.8–2B LLM costs more per token than a 421M encoder; state reuse partly compensates (done, [08](08-state-reuse.md)).
- **At most 26 options** per question (one letter per option); `known` and `surprise` have no effect zero-shot.

**Priorities for improving the model, cheapest first:**
1. **State reuse across questions** (prefix-fork, including images).
2. **Label-free checks** (§5), to decide whether one permutation is enough and to measure the yes-bias.
3. **Temperature + bias term (shift) calibration per option-count band**, evaluated out of domain.
4. **F1** with a supervised proper loss, anti-shortcut data, permutations with a consistency loss, soft human labels.

## 6. Roadmap (revised on 29/09/2026)

| Phase | What | Where | Criterion for "done" |
|---|---|---|---|
| F0 | Setup and zero-shot baseline: letter logits on Qwen3.5 (0.8B/2B in bf16, 4B/9B in 4 bits) and dynamic-depth diagnostics | Locally | **Completed**: no small model beats the Prior zero-shot (see [03-baseline-f0.md](03-baseline-f0.md) §6) |
| Simplification | Removed dynamic depth, `rank`, the `recall` question, `embed` inside `/v1/systemone`, `inject` | Locally | **Completed on 29/09/2026**: tests green, interface re-tested, model runs on text, images and memories |
| **F0.5**, measure and speed up without training | 1) **State reuse** across questions and permutations, text and images: **done on 29/09/2026** ([08-state-reuse.md](08-state-reuse.md)). 2) **Label-free checks:** position (identical options), "Yes" (a question and its negation), language (the same question in Italian and in English). 3) **Calibration** temperature + bias term (shift) per type, separate for text and images, evaluated out of domain. 4) **Evaluation sets** in Italian and English: the one for the first LoRA plus a small generic text set | Locally | Same answers with lower latency. One number for each check. Decision on 1 or 2 permutations |
| **New primitives** (zero-shot) | 1) **`span`**, right after F0.5: a piece of text copied verbatim from the state (invoice number, IBAN, name, date). It is the `open` completion restricted to tokens that continue a piece present in the state, so it cannot make things up. Text states only. 2) **`locate`**, together with the left/right investigation: a box or a point in the image, from the coordinates that Qwen vision models can produce (about twenty generated tokens). 3) **`why`**, when working on the operator interface: the sentences of the state that drove the decision, found by removing one at a time and measuring how much the probability changes (N passes in one batch; text only at first) | Locally | Each has a test set and a threshold fixed **before** the results, and is kept only if it passes. For `span`: accuracy at least equal to `open` on the same set, and zero answers that are not in the state |
| **F1**, first LoRA on an image task | Task to be chosen (§7). Letter readout unchanged; soft CE + Brier loss; augmentations: permutations, negations; Italian and English. A single size: 2B if it fits in 8 GB, otherwise 0.8B | Locally | Better than zero-shot on the task. Fewer spurious "Yes" answers. No regression on text or on `suite_immagini` |
| **F2**, generalist LoRA | Text and images, Italian and English. Human labels (including multi-annotator ones, e.g. ChaosNLI) and the typed-decisions train split. Anti-shortcut data (§4) | Locally | Above the prior (0.478) and the memory vote (0.562), towards 0.766. No out-of-domain regression |
| **F3**, scale | 4B/9B models, labels from a large teacher alongside the human ones | Rented GPU | Only if F2 shows a gain |

**Parked:** trained dynamic depth, the model as an actuator (games, robots), streaming and continuous batching in the server.

**Usage examples to add to the documentation** (no new code):
- **"It cannot be told":** a `choice` with an explicit "the state does not say" option instead of a `noul`. It gives the model a way out instead of a forced "Yes"; the yes-bias check in F0.5 will show how much it helps.
- **Two frames or two cases in the same state,** with a `noul` "has anything changed?" or "is it the same case?". To be verified.

**Separate investigation:** left/right in images, starting from the user's example. Leads to check:
- `load_images` does not apply the EXIF orientation;
- "right" is ambiguous (the person's right or the viewer's);
- the webcam frame compared with what the user sees.

## 7. Open decisions

1. **GPU for training (F1+).** The user is willing to rent a GPU. Plan: local development and debugging on 0.8B/2B; rent only for the heavy steps:
   - labeling with the Qwen3.8-27B teacher via vLLM: 80 GB in bf16, or 48 GB with the FP8 version;
   - LoRA on 4B/9B: 48–80 GB.

   To be decided: provider and budget.
2. **Target language.** Decided on 28/09/2026: Italian and English on an equal footing.
3. **Scope.** Decided on 28/09/2026: generalist, with the first LoRA on a specific image task.
4. **Jev API compatibility.** Done: `/v1/systemone` is exposed by both the model server and the web server.
5. **License and publication.** Open release (Apache-2.0) or internal use?
6. **Task for the first LoRA (F1).** Candidates:
   - **dance poses:** AIST++, with labels computed from the body keypoints and phrased without left/right; the classic named poses as the evaluation set;
   - **Galaxy Zoo:** multiple-choice questions with the volunteers' vote fractions, i.e. true human probabilities;
   - **spatial relations (VSR):** only after the left/right investigation.

   To check for all of them: license and size.
7. **`multi` and `known`.** Decided on 29/09/2026: abandoned as primitives (§0).
