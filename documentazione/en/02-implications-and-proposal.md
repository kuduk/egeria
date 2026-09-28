[Italiano](../02-implicazioni-e-proposta.md) · **English**

# Implications for Egeria and starting proposal

> This document turns the [state of the art](01-state-of-the-art.md) into a concrete proposal. The decisions still open are at the end (§7).

## 0. Decisions made

| Date | Decision | Rationale |
|---|---|---|
| 26/09/2026 | **Backbone: Qwen3.5 family** (like SemIf), not Qwen3.8 | Qwen3.8 has no small sizes and no Base checkpoints. Qwen3.5 has the same architecture. Qwen3.8-27B remains a candidate teacher |
| 26/09/2026 | **Start from the zero-shot F0 baseline**: readout of the letter logits, no training | Sets the reference to beat with LoRA and distillation. Implemented, see [03-baseline-f0.md](03-baseline-f0.md) |
| 26/09/2026 | **Development hardware: RTX 4070 Laptop (8 GB), WSL2** | 0.8B and 2B fit in bf16. 4B and 9B only quantized to 4 bits, or on a cloud GPU |
| 26/09/2026 | **New direction: dynamic depth per assertion** (§4bis) | The user's idea. It fits a single-forward-pass model well |
| 26/09/2026 | **Per-assertion `min_confidence` + global field** (§4bis) | The caller knows the cost of an error. The local value overrides the global one; by default the full model is used |

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
2. **API and real exit (done).** Per-assertion `min_confidence` parameter, with a global field in the request and a server default. The local value overrides the global one; with no threshold, the conservative default applies, i.e. the full model. Block-by-block execution with removal of the questions already decided; the response carries `depth` and `status` (`decided`/`uncertain`). The threshold applies to the chance-corrected confidence (0 = chance, 1 = certainty). Latency measurement: the request waits for the deepest question. On GPU the latency gain only materializes if all questions exit; on CPU it is proportional. See [03-baseline-f0.md](03-baseline-f0.md) §6.4.
3. **F1.** Trained exit heads at every block boundary, with a proper loss on all exits. Threshold chosen with conformal risk control, so that `min_confidence` becomes an empirical accuracy guarantee.
4. **F3.** Multi-level cascade: early exit → full model → larger model or thinking → human escalation. `status: uncertain` is the entry signal.

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

## 5bis. Current limitations of Egeria (September 2026)

- **Zero-shot on typed-decisions we are at the level of the prior:** 0.468 vs 0.478. Above Laya base (0.362), but below a trivial baseline. What beats the prior is the memory vote (0.562).
- **Cost linear in the number of questions, and doubled for permutations.** The state is reprocessed for every question and every permutation, even though the format (state before the question) would allow computing it once.
- **Calibration does not transfer.** The typed-decisions temperatures make generic questions and images worse, there is no bias term (shift), and the yes-bias is not measured.
- **The latency profile differs from Laya's.** A 0.8–2B LLM costs more per token than a 421M encoder; the possible compensation is state reuse.
- **Dynamic depth** with little real gain zero-shot; **at most 26 options**; `inject`, `known` and `surprise` have no effect zero-shot.

**Priorities for improving the model, cheapest first:**
1. **State reuse across questions** (prefix-fork, including images).
2. **Label-free checks** (§5), to decide whether one permutation is enough and to measure the yes-bias.
3. **Temperature + bias term (shift) calibration per option-count band**, evaluated out of domain.
4. **F1** with a supervised proper loss, anti-shortcut data, permutations with a consistency loss, soft human labels.

## 6. Indicative roadmap

| Phase | Goal | Output |
|---|---|---|
| F0 | Setup and zero-shot baseline: letter logits on Qwen3.5 (0.8B/2B in bf16, 4B/9B in 4 bits) and dynamic-depth diagnostics | **Completed**: no small model beats the Prior zero-shot; dynamic depth with 27–36% potential saving (see [03-baseline-f0.md](03-baseline-f0.md) §6) |
| F1 | LoRA + pointer head on public data, exit heads for dynamic depth. Plus the deferred primitives: `multi` (proper loss per option) and `known` (proper loss on same/similar/new pairs + contrastive embedding with LoRA), see [06-memory.md](06-memory.md) §7 | First checkpoint and comparison with Kev/decider |
| F2 | Soft distillation from Qwen3.8-27B and Italian set | Multilingual v1 checkpoint |
| F3 | Calibration, abstention, order invariance | Risk–coverage report |
| F4 | Serving: prefix-fork, Jev-compatible API, quantization, progressive responses (per-question streaming) and continuous batching at block level | Inference server |
| F5 (opt.) | RL with real feedback; LLM-JEPA auxiliary loss | Ablation |

## 7. Open decisions

1. **GPU for training (F1+).** The user is willing to rent a GPU. Plan: local development and debugging on 0.8B/2B; rent only for the heavy steps:
   - labeling with the Qwen3.8-27B teacher via vLLM: 80 GB in bf16, or 48 GB with the FP8 version;
   - LoRA on 4B/9B: 48–80 GB.

   To be decided: provider and budget.
2. **Target language.** Italian first, or English?
3. **Scope.** Generalist model (like Jev) or vertical on a specific domain?
4. **Jev API compatibility.** F0 is already compatible with the request and response format. Still to decide: whether to also expose an HTTP server `/v1/systemone`.
5. **License and publication.** Open release (Apache-2.0) or internal use?
