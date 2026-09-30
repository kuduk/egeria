[Italiano](../04-profondita-dinamica.md) · **English**

# Per-assertion dynamic depth

> **Proposal.** Each assertion (question) uses only as much compute depth as it needs. Easy decisions exit early; uncertain ones keep going.
>
> **Status.** Feasibility measured in F0 with "logit lens" readouts at block boundaries. Results in [03-baseline-f0.md](03-baseline-f0.md).
>
> **Abandoned on 29/09/2026.** The potential (oracle) saving was 27–36%, but the saving achieved zero-shot with the confidence was only 0.5–4%, and the block-wise execution path complicated the scorer. The code (early exit, intermediate readouts, per-layer temperatures, `depth.py`, `--exits`, `--depth`) has been removed. `min_confidence` remains: today it only serves to flag uncertain answers (`status`). The document is kept as a record of the idea and of the measurements.

## 1. Why it fits a System 1 model well

- **No KV cache to propagate.** A decision is **a single forward pass** with a readout at the last position. In early exit for generation (CALM, LayerSkip), when a token exits early the following tokens cannot find the KVs of the skipped layers, which then have to be copied over or recomputed. That problem does not arise here.
- **The savings are real for every question.** Exiting after layer `l` costs `(l+1)/L` of the full forward pass.
  - In a batch, the rows that exit are removed and the batch is compacted before the next block.
- **The signal is already calibrated.** The confidence of a decision model is by construction a calibrated probability, so it is the natural signal for deciding when to stop.
- **It is new.** None of the existing decision models (Jev, Laya, Kev, decider, SemIf) uses adaptive depth.

## 2. The possible "depths"

| Level | What adapts | Cost of an easy decision | Notes |
|---|---|---|---|
| **A. Layers (intra-model)** | Exit at a Qwen3.5 block boundary | From 4/L of the forward pass upward | Subject of the F0 measurement |
| **B. Permutations** | Additional permutations are evaluated only if the option order changes the answer | 1 forward pass instead of P | Reduces order bias only where needed |
| **C. Model (cascade)** | 0.8B → 4B → Qwen3.8-27B | Only the small model | Like the Jev → LLM cascade in the NYU paper (frontier quality at 25–50% of the cost) |
| **D. System 2** | If even the large model is uncertain, hand off to reasoning ("thinking") or to a human | – | Honest abstention |

The levels compose: for example, exit at layer 12 of the 0.8B, or the full 0.8B, or the 4B, or thinking.

## 3. Constraints of the Qwen3.5 architecture

- **Exits only at block boundaries.** The layers are organized in blocks of `3 × Gated DeltaNet + 1 × full attention`. The full-attention layer is the only one that mixes the whole sequence globally. Exiting mid-block means giving up the last global mixing step, so the sensible exits are layers 3, 7, 11, … (0-based indices).

  | Models | Layers | Possible exits |
  |---|---|---|
  | 0.8B / 2B | 24 | 6 |
  | 4B / 9B | 32 | 8 |

- **Intermediate readouts have different scales.** Applying the final norm and the lm_head to intermediate hidden states (logit lens) yields logits with very different magnitudes across layers: in the 0.8B they range from ~1 at block 1 to ~29 at the last one. This calls for **per-layer temperatures** (as in the Annealed Confidence Threshold) or **trained exit heads**.
- **State shared across questions.** With the state prefix-fork (several questions on the same state), the prefix too only needs to be computed up to the maximum depth required by the questions still active. It can be computed "lazily", block by block.

## 4. Exit criteria

| Criterion | Rule | Pros and cons |
|---|---|---|
| **Confidence threshold** | Exit if `max p_T(l) ≥ τ`, where `T(l)` is the layer's temperature | Simple. Requires per-layer calibration |
| **Margin** | Exit if `p1 − p2 ≥ δ` | More robust with many options |
| **Patience** (PABEE) | Exit if the argmax is the same for `k` consecutive exits | Needs no calibration, but kicks in late |
| **Conformal guarantee** (CATs) | `τ` chosen on a calibration set so that `P(pred_uscita ≠ pred_completa) ≤ ε` (*uscita* = exit, *completa* = full model) | Statistical guarantee of consistency with the full model |

## 5. Reference literature

- **DeeBERT** (Xin et al., 2020) and **PABEE** (Zhou et al., NeurIPS 2020): early exit for BERT, based on entropy and on patience respectively.
- **CATs, "Consistent Accelerated Inference via Confident Adaptive Transformers"** (Schuster et al., EMNLP 2021): thresholds with a conformal guarantee of consistency with the full model.
- **CALM, "Confident Adaptive Language Modeling"** (Schuster et al., NeurIPS 2022): per-token early exit in generation, with calibrated confidence measures. Not to be confused with the 2025 "continuous autoregressive" CALM.
- **LayerSkip** (Elhoushi et al., ACL 2024): layer dropout plus an **early-exit loss** on all layers during training, with a shared head. It is the template for training the exits.
- **SimLens** (arXiv 2507.17618): accurate latent predictions from intermediate layers with one extra token.
- **Annealed Confidence Threshold**: higher temperature in the lower layers to counter their overconfidence.
  - https://www.emergentmind.com/topics/early-exit-large-language-models
- **Two-dimensional early exit** (arXiv 2604.18592): layers per sentence. 1.4–2.3× speedup on sentiment classification.
  - https://arxiv.org/html/2604.18592
- **HELIOS** (MLSys 2026): dynamic model selection and early exit in serving.
  - https://arxiv.org/abs/2504.10724
- **System 1 → LLM cascade** (NYU Abu Dhabi, arXiv 2609.24574): frontier quality at 25–50% of the cost.

## 6. Plan

| Phase | Activity |
|---|---|
| **F0** (done) | Logit-lens readouts at block boundaries (`egeria predict --exits blocks`). Temperatures per (type, layer) fitted on train. Threshold simulation (`egeria evaluate --depth`): mean compute vs accuracy, NLL and ECE |
| **F1** | Trained exit heads: LoRA + early-exit loss (proper, soft CE + Brier) at every block boundary, with weights increasing with depth. Alternative: self-distillation from the final exit to the intermediate ones |
| **F3** | Calibrated exit policy: per-type threshold with a conformal guarantee (CATs). Level B (adaptive permutations) and level C (0.8B → 4B → 27B cascade) |
| **F4** | Actual implementation in serving: block-wise execution with batch compaction and lazy computation of the shared prefix. Measurement of real latency, not just of the layer fraction |

**Metrics to report:**
- mean compute (fraction of layers) vs accuracy, NLL, Brier and ECE;
- histogram of exits;
- disagreement rate with the full model;
- real p50/p95 latency.
