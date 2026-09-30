[Italiano](../01-stato-dell-arte.md) · **English**

# State of the art: "System 1" semantic decision models

> Research current as of **26 September 2026**; CLM (§5.5) was added on 29 September. The category emerged in mid-September 2026: almost all numbers are vendor claims or come from community leaderboards only a few days old.
>
> Labels used:
> - **[vendor]**: claimed by whoever built the model.
> - **[community]**: third-party measurement, not peer-reviewed.
> - **[uncertain]**: sources disagree or the data has not been verified.
>
> The full list of sources is in [sources.md](sources.md).

## Contents

1. [Summary](#1-summary)
2. [What is a "System 1" decision model](#2-what-is-a-system-1-decision-model)
3. [Jev (TypeSafe AI)](#3-jev-typesafe-ai)
4. [Laya (Convai Innovations)](#4-laya-convai-innovations)
5. [SemIf and the other open replications](#5-semif-and-the-other-open-replications)
6. [Benchmarks and metrics](#6-benchmarks-and-metrics)
7. [The Qwen 3.8 family](#7-the-qwen-38-family)
8. [Techniques: from decoder to classifier in a single forward pass](#8-techniques-from-decoder-to-classifier-in-a-single-forward-pass)
9. [Calibration, scoring rules and RLCD](#9-calibration-scoring-rules-and-rlcd)
10. [Other "semantic" paradigms (JEPA, LCM, CALM…)](#10-other-semantic-paradigms-jepa-lcm-calm)
11. [Open problems in the field](#11-open-problems-in-the-field)
12. [Uncertain points](#12-uncertain-points)

---

## 1. Summary

- **What they are.** Jev, Laya and SemIf are not generative LLMs. They are **"System 1" decision models**: they take a *state* (text or JSON) and *typed questions* with declared options. In a single forward pass they return a **calibrated probability distribution** over the options. They do not generate text.
- **Jev** (TypeSafe AI, 15/09/2026) is closed and available only through an API. It is the quality reference. Black-box measurements point to a **large decoder** (MMLU-Pro-level knowledge ≈ 84%), not a small encoder.
- **Laya** (Convai, 18/09/2026) is open (Apache-2.0) and uses a 421M ModernBERT encoder.
  - It is fast: about 33 ms on a T4.
  - Zero-shot, however, it is **close to random** (0.36 on typed-decisions) and always has to be fine-tuned.
  - Its "RLCD" is in effect a noisy version of supervised training with proper scoring rules.
- **Open replications on Qwen already exist**, and several of them match or beat Jev on community benchmarks:
  - Kev (also on **Qwen3.8-27B**);
  - Open-Jev-27B (on **Qwen3.8-27B**);
  - decider;
  - JevK5;
  - SemIf.

  The winning recipe today is: Qwen3.5 **Base** + LoRA, readout of the option logits (letter or pointer head), restricted softmax, soft distillation from a larger Qwen teacher, and a fitted temperature.
- **CLM** (Stanford + NVIDIA, 23/09/2026) is open and takes a different route: a **contrastive dual encoder**, that is a frozen Qwen3-8B plus two small heads.
  - It is very fast and has no position bias.
  - In its own tests, however, it agrees with the right answer much less often than Jev (§5.5).
- **Qwen 3.8 has no small models and no Base checkpoints.** Only the 27B dense model, the 2.4T MoE and Flash-Next (125B MoE) exist. The 27B is realistic as a **teacher**, or as a student only with an 80 GB GPU.
- **Hybrid architecture.** Qwen 3.5/3.6/3.8 use a hybrid architecture (75% linear Gated DeltaNet). These parts **cannot be made bidirectional** by changing the mask, and **tree attention masks do not work**. Dedicated techniques are needed, such as the state "prefix-fork".
- **The other "semantic" paradigms** (JEPA, Large Concept Models, CALM, latent reasoning) are interesting but **not needed** for a decision model. At most, an LLM-JEPA auxiliary loss is a cheap experiment.

> **A note on names.** In the original request, "semif laya" was interpreted as **SemIf** + **Laya**, two separate projects described below.

---

## 2. What is a "System 1" decision model

The name echoes Kahneman's "System 1": fast, intuitive decisions, as opposed to the slow reasoning of "System 2".

| | Generative LLM (System 2) | Decision model (System 1) |
|---|---|---|
| Output | Free text (token by token) | Closed types: probabilities over the declared options |
| Forward passes | Many (one per token) | One per request (or per question) |
| Typical latency | Seconds | 5–500 ms |
| "Hallucinations" | Made-up text, malformed JSON | *Syntactically* impossible, but the answer can still be wrong |
| Confidence | Verbalized, unreliable | Explicit distribution, can be recalibrated |
| Typical use | Chat, generation, reasoning | Routing, triage, guardrails, agent gating, moderation, reranking |

### The three primitives (de facto standard, introduced by Jev)

| Primitive | Semantics | Output |
|---|---|---|
| `noul` (from *Bernoulli*) | Yes/no question | `P(true)` |
| `choice` | One of N options (Jev goes up to 255) | Distribution over the options, plus the argmax |
| `score` | Ordinal scale (2–10 levels) | Distribution over the levels and expected value |

The usage pattern is a **code harness**: business logic (sums, dates, rules) lives in code, while the model only answers atomic questions. TypeSafe shows that even frontier models improve when used this way. For example, Opus 5 goes from 64.8% in prompt mode to 73.1% in workflow mode.

---

## 3. Jev (TypeSafe AI)

### 3.1 Facts

- **Company.** TypeSafe AI. The founder is Diogo Almeida (formerly at Google Brain and OpenAI, co-author of InstructGPT and GPT-4); the co-founders are Erik Gafni and Sasha Sheng.
  - $40M seed round led by DCVC; $200M valuation according to Forbes.
- **Launch.** 15/09/2026, after about 2 years in stealth. The current model is `jev-1.13.0`.
- **Distribution.** Closed, API-only, US cloud. No fine-tuning: "the same weights serve every account". Zero data retention is reserved for enterprise customers.
- **Price.** $0.042 per million input tokens; output is free.
- **Context limits.** 64k tokens per request; 32k for the state plus the longest question.
- **Languages.** Mainly English. CJK is handled "but not as well".
- **Latency.**
  - Claimed [vendor]: 70–500 ms end-to-end.
  - Measured [community]: 236–276 ms p50 from the client.
  - Server-side time: about 57–218 ms as the state grows.

### 3.2 API (`POST /v1/systemone`)

```json
{"state": "Help! My payouts have been failing for 3 days.",
 "model": "jev-latest",
 "questions": {
   "is_urgent":  {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
   "department": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Payments, invoicing, refunds",
                               "technical": "Bugs, outages, integrations",
                               "sales": "Pricing, upgrades, new accounts"}},
   "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                  "criteria": ["Calm", "Frustrated", "Very angry"]}}}
```

For each question, the response contains the argmax, the `probabilities` and a `confidence`.

**`confidence` formulas.** They are statistics of the distribution, not a learned estimate of correctness.
- choice: `c = (n·p_max − 1)/(n − 1)`
- score: `c = max(0, 1 − Σ p_i·|i − mode| / D)`, where `D = (1/n)·Σ|i − (n−1)/2|`

**Abstention.** There is no abstention field: it has to be done client-side with thresholds, for example act, confirm or escalate. The official cookbook sends cases with p_max < 0.60 to human review, achieving 99.2% consistency with 74.2% of cases automated.

Official SDKs: Python `typesafe-sdk`, JavaScript `@typesafe-ai/sdk`. Integrations with LangChain and Vercel AI Gateway exist.

### 3.3 Architecture: what is known

**Official statements:**
- "New architecture + parallel sampler + RLCD". FAQ: "Jev is neither small nor an LLM".
- No string output: all outputs can be computed in parallel.
- "We are primarily a data research lab." The stated competitive advantage is the data, not the architecture.

**Black-box analysis** (Archer Hume, about 10,000 calls) [community]:
- **Shared state, isolated questions.** A secret placed in a sibling question is invisible to the other questions; placed in the state, it is found. This is consistent with a shared prefix KV-cache and separate per-question suffixes.
- **Listwise options.** Adding an irrelevant option shifts the log-odds among the others, violating IIA (independence of irrelevant alternatives). Option order matters as well.
- **Tokenizer.** It matches none of the 192 public tokenizers tested; the closest is **Qwen**.
- **Knowledge.** MMLU-Pro ≈ 84.6%. This suggests a large decoder, probably an MoE with about 10B active parameters [uncertain].

### 3.4 RLCD ("Reinforcement Learning for Calibrated Decisions")

An **unpublished** technique. The only official description: it optimizes for "answers with epistemically honest probabilities", as opposed to RLHF (sycophancy) and RLVR ("spiky intelligence").

Third-party interpretations converge on a **proper-scoring-rule objective** (log or Brier) applied to the option slots.

### 3.5 Performance and weaknesses

**TypeSafe Workflow Evals** [vendor]:
- Four workflows: security, agent traces, invoices, customer service.
- The reference labels are the average of GPT-6 Astra and Fable 5.1, so the benchmark measures **agreement with frontier LLMs**, not ground truth.

| Model | Mean accuracy | $/case | s/case |
|---|---|---|---|
| Jev | 67.8% | 0.0004 | 0.4 |
| GPT-5.6 Sol | 74.1% | 0.0836 | 23.3 |
| Opus 5 | 73.1% | 0.1761 | 37.8 |
| Sonnet 5 | 67.8% | – | – |
| Haiku 4.5 | 53.6% | – | – |

**Stated weaknesses** ("Jev 1.13 jaggedness" page):
- overly literal reading;
- arithmetic, counting and dates;
- multi-hop questions;
- long states full of noise;
- prompt injection in the state;
- no structural consistency across questions. For example, `noul` "refund" = 0.72 and `noul` "not a refund" = 0.47, which add up to 1.19.

**Third-party measurements** [community]:
- **Bluffing.** It admits not knowing in only 49.7% of impossible cases, versus 97–100% for LLMs.
- **Order sensitivity.** Permuting the options changes the answer in 13–14.6% of cases.
- **Aleatoric uncertainty.** On a fair coin it gives P(heads) = 0.92.
- **typed-decisions.** Accuracy 0.727, but a KL of 1.442 from the gold distribution: it is overconfident.
- **NYU Abu Dhabi** (arXiv 2609.24574, 18 computational social science tasks):
  - behind the best LLM on 14 of 15 tasks (median −11.6 F1);
  - median ECE 0.157;
  - a Jev → LLM **cascade** on the uncertain cases reaches frontier quality at 25–50% of the cost.

---

## 4. Laya (Convai Innovations)

### 4.1 Facts

- **Author and license.** Nandakishor M (Convai Innovations), an independent researcher. Released on 18/09/2026, **Apache-2.0**, `pip install laya`, weights on HF `convaiinnovations/laya`.
- **Checkpoints:**

| Checkpoint | Encoder | Parameters | Context (default) |
|---|---|---|---|
| `laya` | ModernBERT-large (395M, 28 layers) | 421M | 512 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 (up to 8192) |
| `laya-typed-decisions` | ModernBERT-large, fine-tuned | 421M | 1024 |

### 4.2 Architecture (from the code)

**Input packing**, one sequence per question:
```
[CLS] "<tipo> question: <istruzioni>" [SEP] [MASK] opz0 [MASK] opz1 … [SEP] <stato> [SEP]
```

**Decision head:**
1. A question-type embedding is added to the encoder output.
2. This is followed by 2 additional transformer layers (about 25M parameters).
3. The hidden states at the `[MASK]` positions are gathered.
4. An MLP produces one logit per option, followed by temperature and softmax.

**Primitives:**
- `noul` always uses two slots, `[false, true]`.
- `score` returns the expected value `Σ i·p_i`.

**Act/Escalate head.** An MLP over CLS, p_top1, margin, entropy and k. In the released version **it is broken** (issue #185): it always returns ≈1.0 and is anti-correlated with correctness.

**Multilingual router.** It detects script and language in under 0.5 ms and picks the checkpoint. It is needed because the English checkpoint scores 0.000 accuracy on Khmer with 0.952 confidence.

### 4.3 Training

**Data.** Public human-labeled datasets: triage, NLI, safety, jailbreak, rubrics, sales conversations. The exact list has not been published.

**Laya-style "RLCD":**
1. Zero-mean Gaussian noise on the logits, with G = 8 samples and σ decreasing from 1.0 to 0.3.
2. Reward = log-score + 0.5·spherical − 1.0·RPS. RPS is used only for `score`.
3. Group-normalized advantage, then REINFORCE. For multi-turn, TD(λ = 1) is used.

**Public fine-tuning** (notebook for 2×T4 on Kaggle):
- The loss is RL plus **1.0 × soft cross-entropy** towards the gold distributions, so it is not "pure" RL.
- The temperature is fitted per question type with L-BFGS on held-out data.

> **Analysis** (shared by several commenters). With a Gaussian policy over the logits, REINFORCE is a zeroth-order estimate (evolution-strategies style) of the scoring-rule gradient. As σ → 0 it coincides with the supervised gradient, which can be computed exactly with backprop. The log-score is equivalent to soft cross-entropy.
>
> **When the rewards are derived from labels, the "RL" only adds variance.** RL makes sense only with environment or bandit feedback (see §9.3).

### 4.4 Results [vendor, unless stated otherwise]

| Metric | Value |
|---|---|
| typed-decisions, base zero-shot | **0.362** (random 0.318; majority 0.461) |
| typed-decisions, after fine-tuning on the train split | **0.766** (Jev 0.727) |
| Banking77 (77 classes) | **0.425** (Jev 0.870); the token budget for the options runs out |
| AG News / DAIR Emotion | 0.950 / 0.595 (Jev 0.910 / 0.480) |
| ECE before and after the temperature refit | 0.466 → 0.081 |
| Latency on T4, 1 question | 32.8–39.5 ms; 103–332 questions/s batched |
| MASSIVE (51 languages, 20 options) | 0.227 (EN) / 0.366 (multilingual) |
| Flip rate under option permutation | 0.15–0.23 |

### 4.5 Criticisms

- **The 0.766 vs 0.727 comparison is not like for like.** Laya is a *specialist* trained on the train split; Jev is a zero-shot *generalist*. According to the dataset card itself, exceeding the teacher's "ceiling" (0.735) indicates that the model has learned the teacher's idiosyncrasies.
- **The "2.4× better" Brier is a metric mismatch.** With an independent harness, Laya has a Brier of 0.213 versus Jev's 0.148.
- **The latencies are not comparable.** Laya's is measured locally; Jev's includes the network round-trip.
- **Independent stress test** (gazelle93):

  | | Jev | Laya |
  |---|---|---|
  | Accuracy at 128 candidates | 60% | 39% |
  | Answers changed just by reordering the options | 14.6% | 49.4% |

- **The "prior art" claims** (the author's 2025 arXiv paper) have been disputed: that paper is not about schema-based typed decisions.

### 4.6 From the code: what Laya actually does and what we need (verified on 28/09/2026)

Sources read: the TypeScript runtime ([receptron/laya](https://github.com/receptron/laya)), the Python code ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya): `laya/common.py`, `laya/shortlist.py`, `research/`), the fine-tuning guide and the limitations stated in the README.

**Architecture (inference).**
- The sequence is `[CLS] <tipo> question: istruzioni [SEP] [MASK] opzione0 [MASK] opzione1 … [SEP] stato [SEP]`. One logit per `[MASK]`, then a softmax over the question's options.
- `noul` is a choice between two described options: `false: no, the statement does not hold` and `true: yes, the statement holds`. The descriptions can be replaced with `criteria` (`{"false": "la recensione è negativa", "true": "…positiva"}`, i.e. "the review is negative" / "…positive"), and the labels shown to the model with `labels` (for example `B`/`A`).
- **Inference-time calibration is just one temperature per type and per number of options** (`temperature_by_options`):

  | Bucket | Temperature |
  |---|---|
  | `noul:2` | 1.98 |
  | `choice:2` | 1.91 |
  | `choice:3-5` | 1.76 |
  | `choice:6-10` | 1.00 |
  | `choice:11+` | 0.10 |
  | `score:3-5` | 1.25 |

  There is no bias term (shift): a systematic lean towards one answer (for example a yes-bias) cannot be corrected at inference time, only through training.
- **The `act` head ("act or hand off") does not work:** it is ~1.0 almost always, with an AUROC of 0.30 against correctness (issue #185). Confidence, by contrast, reaches an AUROC of 0.77. Unless a dedicated head is trained with its own objective, confidence alone is enough to decide when to trust the model.

**Training (RLCD in practice).**
- **Targets:** the teacher's (soft) distributions, not hard labels.
- **Loss:** full-weight soft cross-entropy is the main term. A policy-gradient term on noisy logits is added, GRPO-style: 4 samples, noise from 0.4 down to 0.1, reward = log-score + spherical (0.5–0.75) − RPS for `score`.
- **Post-hoc calibration:** on a slice of data removed from training *before* it starts (up to 400 examples or 10%). One temperature per type is fitted with L-BFGS on log T, bounded to [0.1, 10]. The calibration must be saved together with the weights: otherwise the old per-bucket temperatures would take precedence and cancel the new one.
- **Cost:** on 2×T4 it takes minutes for 6,000 decisions and about 4–5 hours for 30,000 questions.
- **TD(λ) in conversations:** every prefix of the conversation gets the final outcome as its target. With λ = 1 this coincides with Monte Carlo.

**Fine-tuning as the decision head of a web agent** (`docs/finetune_browser_agent.md`, a single 16 GB RTX 4070 Ti SUPER, no paid APIs). This is the case closest to the "model = actuator" use (games, robots):

| | Zero-shot | Fine-tuned |
|---|---|---|
| Right element on unseen pages (~45 candidates) | 0.10 | **0.66** (421M) |
| Right operation (click / type / select / done) | 0.54 | 0.88–0.89 |
| Real tasks completed | 0% | 50–62% |
| Latency per step | | 17–50 ms |

Lessons:
- **The input format matters more than the data.** Moving the candidates out of the state and into the options gave +7 points and 10 tasks out of 16 instead of 6.
- **Clean labels "in reverse":** you pick the answer first, then a local LLM writes the goal that requires it. No teacher has to solve the task.
- **Shortcuts to avoid.** Goals written from a fixed template teach the model the phrasing, not the task. "There is a history" ends up meaning "I'm done". You need mid-task negatives and reweighted rare actions (×3–×4).
- **Corrections on its own trajectories** (DAgger): run the model and correct it where it goes wrong.
- **Handing the uncertain cases to an 8B or 27B LLM made results worse:** on these tasks the trained 322M decides better, and in 21 ms instead of 4.7 s.

**Label-free evaluations** (`research/eval/`):
- **Metamorphic:** permute the options or replace the labels with opaque letters, then compare the distributions (rate of changed answers).
- **Identical options:** every option has the same text. A model with no position preference gives a uniform distribution; on the multilingual checkpoint this revealed a preference against the first `score` level (#131).
- **Confidence:** measured with AUROC against correctness and with accuracy on the more confident half, because ECE alone depends on the scale.

**Known checkpoint defects:**
- `noul` can follow the `false`/`true` labels instead of the state, giving confident "no" answers on clearly positive cases (#156). The workaround is a two-option `choice` with neutral keys `A`/`B`.
- In negated questions the choice follows the question, not the state (#377).
- In `choice`, the labels `yes`/`no` and `true`/`false` should be avoided.
- Beyond ~20 options the token budget runs out. The remedy is `predict_shortlist`: an embedding picks the `k` closest options, and the decision is then made among those.

**Comparison with Egeria on the same split** (typed-decisions test, 400 cases, 2,000 decisions):

| Model | Accuracy |
|---|---|
| Laya base, zero-shot | 0.362 (below the majority baseline, 0.461) |
| Egeria, Qwen3.5-2B-Base zero-shot | 0.468 |
| Egeria, memory vote | 0.562 |
| Laya fine-tuned on the train split | 0.766 |

The gap to 0.766 is what phase F1 has to close.

**Limitations visible in the code, and what can be done better:**

| Laya's limitation | Better |
|---|---|
| **The state is reprocessed for every question.** The question comes *before* the state and the encoder is bidirectional, so nothing is reused: 1 question 39.5 ms, 10 questions 159 ms, 50 questions 771 ms | In a causal model with the state **before** the question (our format), the state is computed once and each question only costs its own tail. With images the gain is even larger |
| **It depends on option order:** 13–23% of answers changed in their own test, 49% in the independent stress test. Nothing is done about it at inference time | We average over 2 permutations, at the cost of twice the compute. Even better: train with permutations and a consistency loss, so that a single one is enough |
| **The RL term is redundant.** With known targets, soft CE already gives the exact gradient; the noisy policy gradient is a higher-variance estimate of it that costs 4 extra samples. If the advantage is normalized by the group standard deviation, it pushes towards overconfidence, and indeed the model comes out overconfident (ECE 0.466 before the refit) | A pure proper supervised loss. RL only with real feedback (environments, bandits), with a baseline and no standard-deviation normalization |
| **The teacher is the ceiling.** The targets are an LLM's distributions, and typed-decisions measures agreement with that teacher, not the truth. Beating its ceiling (0.766 > 0.735) means having learned its idiosyncrasies | Mix in soft human labels (multiple annotators: ChaosNLI, Galaxy Zoo) and also evaluate on human-labeled sets |
| **Fragile calibration.** Temperature only, no shift. Fitted in the training domain (400 examples). The `choice:11+` bucket is at 0.10, i.e. the lower bound of the fit: a degenerate fit. Out of domain, confidence does not hold up (Khmer: accuracy 0 with confidence 0.95) | Temperature plus shift per bucket; discard fits that hit the bounds; measure calibration **out of** domain; thresholds with guarantees (conformal risk control) |
| **The "act or hand off" head is not trained** | Either train it with an objective (was the answer right?), or use confidence with guaranteed thresholds |
| **The option token budget is fixed:** 48 tokens per option, ~20 options at most, state at 512 tokens | In our format the options follow the state with no fixed budget; the limit is the 26 letters, which can be overcome with a preliminary selection via `embed` |
| **It can only choose:** it does not read values (dates, amounts) and does not see images | Already covered by `short_answer`, `estimate` and image states |

---

## 5. SemIf and the other open replications

### 5.1 SemIf (TheoLeeCJ, MIT)

- **Idea.** No training: it takes **Qwen3.5-4B** (or MiniCPM5-2B, Qwen3-0.6B, or an EXL3-quantized 27B). It puts the state, the question and the letter-labeled options in the prompt, and reads **the letter logits at the last position**. The softmax is restricted to the declared options.
- **Calibration.** A per-workload temperature, added recently [uncertain: earlier versions were declared "uncalibrated"].
- **Numbers** [community]:
  - 21 binary criteria on an RTX 3090: 1.02 s with direct logits versus 5.33 s when generating JSON.
  - Balanced accuracy 0.845 on a TypeSafe subset (Jev 0.883).
  - With a shared state and parallel suffixes: 20 decisions/s.
- **Execution.** CUDA, Apple MLX, CPU (llama.cpp/GGUF) and WebGPU in the browser.
- **Why it matters.** It is the simplest demonstration that **a Qwen LLM is already a decision model** with nothing more than a logit readout.

### 5.2 Decoder replications (the closest to our project)

| Project | Base | Readout | Training | Results [community] |
|---|---|---|---|---|
| **Kev** (J. Palmer, Apache-2.0) | Qwen3.5-Base 0.8B/4B/9B and **Qwen3.8-27B** | **Pointer head**: hidden state of `</opt>` compared with the `<decide>` token | LoRA r16 + CE; T = 2.30 | Kev-9B OOD 0.852 (Jev 0.857), ECE 0.042. Kev-27B dev/test 0.848/0.896. Kev-4B: 18 ms for 6 questions on H100 |
| **Open-Jev** (Zefan Cai, MIT) | **Qwen3.8-27B** | Scalar head initialized from Yes − No | LoRA; soft CE + 0.1·Brier; 148k rows | JevBench public 85.3% (Jev +3 answers) |
| **decider** (Mapika) | Qwen3.5-2B/4B/35B-A3B-Base | Letter logits in one slot per question | CE on about 95 datasets + labels from a Qwen3.5-27B teacher (1.47M examples); v10 adds calibrated RL on games and browser tasks | **#1 JevBench v1.4.2** (decider-4b 64.13 vs Jev 63.29). Best ECE on the Decision Index. 18 ms (2B) on B300 |
| **JevK5** | Qwen3.5-4B + LoRA r16 | Letters, T = 1.532 | Distilled from Qwen3.6-27B thinking | #2 JevBench v1.4; ECE 0.066; ~13 ms on H100 |
| **open-alternative-jev** | Any LLM | Packed multi-question, 2 permutations | Temperature scaling only | Qwen3.6-27B **zero-shot** on typed-decisions: 73.7% acc, ECE 0.020 (Jev 72.7% / 0.144) |
| **"Your LM is already a decision model"** | Qwen3.5-9B zero-shot | Rotated letters | None | JevBench 80.5% vs Jev 86.1%; beats Jev on PhishNChips and MetaTool |
| **qwen3-0.6b-rlcd** | Qwen3-0.6B-Base + LoRA r32 | Per-option hidden state | CE on soft frequencies (507k questions, 2,456 tasks) | Held-out: acc 0.783, ECE 0.031 |
| **eve-rlcd** | Qwen3-0.6B-Base | 26 letters | RL with reward `c − p(a)` (= ½ the Brier gradient) | See §9.3: supervised training is on par or better |
| **AnyJev** (Nokia) | Any LLM | Training-free readout + closed-form head fitted on 100–500 labels | – | Order flips 0.230 → 0.073; ECE 0.240 → 0.095 |
| **qwen-rlcd** | Qwen3.5-0.8B | Prototype of a DeltaNet state **prefix-fork** | – | – |
| **SCX Router** (Knowledgator, arXiv 2609.02292) | Causal Qwen3-0.6B + lightweight bidirectional scorer + GLiClass head | Labels scored on top of a persistent KV-cache | 150k tasks | Hybrid decoder + encoder blueprint |

### 5.3 Encoder replications and services

- **GLiNER2.5-Decide** (Fastino, 24–25/09/2026):
  - 340M DeBERTa-v3-large, schema-conditioned.
  - "Fast Decisions" (17 domains × 300 examples): 60.2%, versus JevK5 57.6, SemIf 56.4, Laya 46.6.
  - Latency 38 ms on V100.
- **meraGPT Decider 1**: closed, Jev-compatible API. typed-decisions 0.768 zero-shot, Brier 0.052.
- **Featherless "Simple Jev"**: logit readout on open models (for example Qwen3.6-35B-A3B) with a shared prompt cache; "RFDT" trainer.

### 5.4 What the replications tell us

1. **A 4–9B Qwen with LoRA and a logit readout reaches Jev** on community benchmarks.
2. **The main gap is the backbone's knowledge**, not the head architecture. decider scores 0.51 versus Jev's 0.69 on the "knowledge" area of the Decision Index.
3. **Two open baselines already exist on Qwen3.8-27B**, Kev-27B and Open-Jev-27B. To be useful, a new model has to **differentiate itself**: for example on language, abstention, order invariance or domain.

### 5.5 CLM (Stanford + NVIDIA, 23/09/2026): a contrastive dual encoder

Code read on 29/09/2026 (commit bb42c6c), together with the Hugging Face cards.

- **What it is.** CLM-8B (*Contrastive Language Model*): J. Kwok, A. Mirhoseini, C. Ré, M. Pavone and others, Stanford with NVIDIA. It is Apache-2.0 (package `contrastive-lm`). It uses the same format as Jev: `POST /v1/systemone` with `noul`, `choice` and `score` (verified in the code). It also has a `rank` over free-form candidates.
- **The model.**
  - Encoder: **frozen Qwen3-8B**. It is not a Qwen3.5: no DeltaNet, no images. It is served by vLLM in *pooling* mode and uses the **last-token** embedding (4096 dimensions). States are truncated to 2048 tokens.
  - On top sit **two MLP heads**, one for the state and one for the action: 4096 → 1536 → 1536 → 512, with LayerNorm. Together they have about 19M parameters (75 MB in fp32) and are the only trained part.
  - Score: `exp(logit_scale) · cos(state_head(s), action_head(c))`, then a softmax over the question's candidates.
- **How it reads questions.**
  - The state, followed by the `instructions`, goes into the state head.
  - Each option is a separate text in the action head: its description, or else its key.
  - For `noul` the two candidates are `true: Yes. This is true: {question}` and `false: No. This is false: {question}`.
  - There is no calibration, only a temperature passed by the caller.
- **Training** [vendor], in three stages:
  1. 60M Nemotron question–answer pairs, with bidirectional InfoNCE.
  2. 30M synthetic hard negatives generated with Gemini 2.5 Flash-Lite. Top-1 over 10 negatives goes from 52.1% to 69.2%. Used from the start instead, hard negatives stop at 62.4%.
  3. 1M agent trajectories (ADP, Endless-Terminals, LiteCoder). Each step is a (context, action taken) pair. The mix contains 40% Nemotron replay: without replay, top-1 on Nemotron drops from 69% to 56.2%.
- **Speed.** Actions are encoded once and stay in the cache. With repeated candidates, a question costs one state embedding plus a dot product. On T-Rex: 2.6 ms of model time and 16.5 ms p50 client-side. Jev, via the API, takes 150.

**How it handles the agent benchmarks (from the code and the published files):**

| Benchmark | What it actually does | Critical note |
|---|---|---|
| T-Rex (Chrome dinosaur game, 5 courses × 60 s) | A physics planner works out which actions are safe and writes it into the options: `Safe. … Best.`, `Safe. …`, `Unsafe. … Collision.`. The model only has to read the label. A "shield" replaces unsafe answers | Both survive 5/5, but CLM agrees with the planner on only **65.8%** of decisions (Jev 98.7%). The shield steps in **4,883 times** for CLM and 28 for Jev: the shield makes the tie. The test measures latency, not the decision |
| DeepSWE (best-of-4 verifier) | A head **trained for the purpose** on 59 tasks (22,576 pairs taken from successful trajectories). Each step gets the state–action cosine and a trajectory scores the mean of its last 12 steps. The trajectory with the highest score is picked | 38 held-out tasks, only **13** of them decidable. Random: 28/38 (73.7%). CLM: **31/38** (81.6%). Oracle: 34/38. That is **3 tasks** above chance. Jev, instead, is used zero-shot. The evaluation dataset cited in the README is not public (29/09/2026) |
| Terminal-Bench 2.1 (87.6%) | Same scheme, with candidates generated by Fable 5 | Head and data not published |
| BFCL v4, WikiRacing, Super Mario | Only in the README chart. BFCL 95.2% (Jev 99.2%), WikiRacing 26/30 (Jev 30/30) [vendor] | No code in the repository |

The fine-tuning guide (`docs/FINETUNING.md`) runs an agent that edits the trainer in a loop and keeps a change if it improves the "held-out" best-of-N rate. That turns the test set into the selection criterion. It is not stated whether the published numbers come from that loop; if they do, they are optimistic.

**What is useful to us:**
1. **Order invariance by construction.** Each option is encoded on its own, so the position bias measured in [09](09-label-free-checks.md) does not exist. The price is that state and options never meet inside the model. Negations and near-identical labels (`Safe`/`Unsafe`) become hard, as the 65.8% on T-Rex shows.
2. **Many options at low cost.** Our letter readout stops at 26 options; CLM keeps thousands in its cache (WikiRacing). If we ever need to choose among many candidates, a contrastive head on top of our `/v1/embed` is the cheapest route.
3. **An alternative to LoRA for F1.** The heads train on embeddings precomputed by a frozen backbone: minutes, not hours. They can be tried as a baseline for the first image task, to compare with the LoRA.
4. **A data recipe for F2.** Broad data first, then hard negatives (+7 points over using them from the start), plus 40% replay to avoid forgetting. This applies to our LoRAs too.
5. **Things not to copy:**
   - comparisons between heads trained on the benchmark and zero-shot Jev;
   - benchmarks with the answer written into the options;
   - selection done on the test set.

**Hardware.** Qwen3-8B in bf16 takes about 16 GB. On our 8 GB GPU it would have to be quantized, but the head was trained on bf16 embeddings and quantization shifts them: this needs checking. A multimodal CLM-35B-A3B is announced for early October 2026 [vendor, from articles].

---

## 6. Benchmarks and metrics

### 6.1 Benchmarks specific to decision models (all from September 2026)

| Benchmark | Content | Notes |
|---|---|---|
| **typed-decisions** (HF `LocalLLaMA/typed-decisions`, Apache-2.0) | 4 synthetic workflows. 1,200 train cases and 400 test cases (2,000 decisions) | Gold labels from a "4B-class" teacher (3 samples). The teacher's self-agreement ceiling is 0.735. **Do not compare specialists with generalists** |
| **TypeSafe Workflow Evals** | The same 4 workflows | Measures agreement with GPT-6 Astra + Fable 5.1 |
| **JevBench v1.4.2** | 534 decisions (hard, standard, easy, judge) | Composite: geometric mean of accuracy, calibration, speed and cost |
| **Decision Index v0.1** | 132k requests, 37 benchmarks (BFCL, RouterBench, MMLU, GPQA, GSM8K, ForecastBench…) | Abstentions score 0 |
| **Fast Decisions** (Fastino) | 17 domains × 300 examples | – |
| **nibzard DMB**, **decision-models-under-pressure** | Stress tests: cardinality, distractors, order | – |

### 6.2 Useful classic benchmarks

- **BTZSC** (22 zero-shot classification datasets): the best is Qwen3-Reranker-8B with macro-F1 0.72.
- **Banking77**: 77 classes, stress-tests high cardinality.
- **CLINC150**: 150 intents plus **out-of-scope** requests, which makes it the natural test for abstention.
- **AG News**, **XNLI** and **MASSIVE**, the last two for multilingual evaluation.
- **MTEB**: careful, its classification tasks use logistic regression on embeddings, so they are not zero-shot.

### 6.3 Metrics to always report

- **Accuracy.** Keep the generalist (zero-shot) and specialist settings separate.
- **Probabilistic quality.** NLL, **Brier**, **ECE** (stating the binning), KL from the gold distribution if it is soft.
- **Risk–coverage.** For example coverage at a 5% error budget, and accuracy at p ≥ 0.9.
- **Robustness.** Flip rate under option permutation, consistency across equivalent or negated questions, confidence on "unknowable" items.
- **Latency.** Separate model time from the network round-trip.

---

## 7. The Qwen 3.8 family

### 7.1 Open checkpoints (verified via the Hugging Face API on 26/09/2026)

| Repo | Type | Parameters | License | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` (+FP8) | Dense, multimodal | 27.78B (including the vision tower) | **Apache-2.0** | Post-trained only ("thinking" can be turned off). **No Base** |
| `Qwen/Qwen3.8-2.4T-A95B` (+FP8) | MoE | 2.446T, 95B active | Custom "Qwen3.8-Max" license | Needs B300/GB300-class hardware |
| `Qwen/Qwen3.8-Flash-Next` (+FP8) | MoE + n-gram embedding ("Qwen4 preview") | 125B, 6B active | qwen-community-1.0 | Requires the new `qwen4_exp` code |

- **There are no 0.6–9B Qwen3.8 models.** Small models with a **Base** variant exist only in **Qwen3.5**: 0.8B, 2B, 4B, 9B and 35B-A3B, all Apache-2.0, with **the same architecture** and the same tokenizer family as 3.8.
- **Qwen3.8-27B is architecturally identical to Qwen3.5-27B and 3.6-27B** (`model_type: qwen3_5`, same parameter count). Only the post-training differs.

### 7.2 Hybrid architecture and its implications

**Qwen3.8-27B:**
- 64 layers, organized as `16 × (3 × Gated DeltaNet → 1 × Gated Attention)`.
- hidden 5120, vocabulary 248,320.
- Native context of 262k tokens (about 1M with YaRN), 1 MTP layer.

**Small Qwen3.5 models**, same 3:1 pattern:

| Model | Layers | Hidden |
|---|---|---|
| 0.8B | 24 | 1024 |
| 2B | 24 | 2048 |
| 4B | 32 | 2560 |
| 9B | 32 | 4096 |

**Practical consequences:**

1. **No BERT-style bidirectionality.** The Gated DeltaNet layers are causal by construction: recurrence plus a causal conv1d, not a mask. **Only the 25% of full-attention layers** can be made bidirectional. transformers allows this with a dictionary mask `{"full_attention": …, "linear_attention": …}`.
   - The only published precedent: **dQwen3.5** (arXiv 2609.20751), which does exactly this.
   - No one has published classification results with this configuration yet.
2. **Laya's `[MASK]`-before-the-option pattern does not work on a causal model**, because the marker cannot see the option text. The markers must go **after** the options, ideally in a final block.
3. **Tree attention masks and packing do not isolate branches.** The DeltaNet recurrence cannot be masked per branch.
   - For multiple questions on the same state you need a **prefix-fork**: compute the state once and copy KV + conv state + recurrent state for each question.
   - Backpropagating through the forked state is not trivial, because the DeltaNet state is updated in place.
4. **Known bugs:**
   - In transformers 5.2–5.8.1, packing leaked state across samples (fixed in 5.9.0).
   - Without the `fla`/`causal_conv1d` kernels, it silently falls back to slow PyTorch.
   - vLLM: prefix caching on hybrids is experimental (NaN on Qwen3.8-27B, open issue), and there are problems with LoRA on the GDN projections.
5. **The readout must be done in fp32.** In bf16, discrepancies of up to 2.3e-2 were observed across different batch shapes.

### 7.3 Hardware (Unsloth estimates for bf16 LoRA on Qwen3.5; the 27B figures also apply to 3.8)

| Model | bf16 LoRA | Full FT (estimate) | Inference |
|---|---|---|---|
| Qwen3.5-0.8B | 3 GB | ~15 GB | – |
| Qwen3.5-2B | 5 GB | ~36 GB | – |
| Qwen3.5-4B | 10 GB | ~75 GB | ~60–80 ms per 1k tokens on a 4090 (estimate) |
| Qwen3.5-9B | 22 GB | ~155 GB | – |
| Qwen3.8-27B | **56 GB** | ~450 GB | 4-bit: 16–19 GB (4090/5090); ~100–150 ms per 1k tokens on H100 (estimate) |

- **Unsloth advises against 4-bit QLoRA** on Qwen3.5 models.
- transformers v5 is required.
- Qwen3.8 fine-tuning support in Unsloth is not documented, but the architecture is identical.

### 7.4 Tooling

- **transformers 5.17.** Already ships `Qwen3_5ForSequenceClassification` and `Qwen3_5ForTokenClassification`, with last-token pooling.
- **PEFT/LoRA.** Target modules:
  - DeltaNet: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
  - attention: `q/k/v/o_proj`
  - MLP: `gate/up/down_proj`
- **vLLM 0.30.** Supports classification pooling with `classifier_from_token`, as for Qwen3-Reranker. It has no native `SequenceClassification` class for Qwen3.5 [uncertain about the generic conversion].
- **ms-swift.** Supports Qwen3.8 with classification, reranker and embedding tasks.

### 7.5 Qwen precedents for "a decoder used as a scorer"

- **Qwen3-Embedding** (0.6B/4B/8B): last-token pooling, #1 on MMTEB in June 2025.
- **Qwen3-Reranker**: softmax over the "yes"/"no" logits at the last position. It is effectively a `noul`.
- **Qwen3Guard-Stream**: a **classification head on the hidden states** of Qwen3. It is a direct model for our heads.
- There are no official 3.5/3.8 versions of these models yet.

---

## 8. Techniques: from decoder to classifier in a single forward pass

### 8.1 Encoder vs decoder (evidence)

- **At equal size, the encoder wins.** Ettin (ICLR 2026) on MNLI: 400M encoder 91.3 versus 400M decoder 88.2. The 400M encoder also beats the 1B decoder.
  - Converting a decoder into an encoder (50B MNTP tokens) only reaches 87.6.
- **Larger fine-tuned decoders win.**
  - Gemma Encoder, GLUE with a bidirectional mask: 84.5 → 89.4 (2B) and 87.5 → 90.9 (9B).
  - A causal LLM of up to 8B with a **last-token head** and 4-bit LoRA matches or beats fine-tuned BERT (arXiv 2512.12677).
- **Bidirectional conversion requires training.**
  - LLM2Vec: bidirectional mask + MNTP + SimCSE.
  - **BidirLM** (2026): bidirectionalized Qwen3-0.6B/1.7B. Conclusion: "enabling bidirectionality with a masking objective is critical".
  - It has a `[MASK]` token, so Laya's design can be reused on a Qwen3 backbone (not 3.5).
- **Zero-shot versus fine-tuned.** Fine-tuned encoders beat zero-shot LLMs on closed tasks, at 30–100× lower cost. LLMs, on the other hand, clearly win in zero-shot and on knowledge.

### 8.2 Single-pass readout with a causal decoder

| Technique | How it works | Precedents | Notes |
|---|---|---|---|
| **Letter logits** | Options labeled A/B/C…; restricted softmax over the letter tokens at the last position | SemIf, decider, JevK5 | Has a position bias: mitigate it with PriDe or permutations. The letters must be single tokens |
| **Yes/No** | Softmax over {yes, no} | Qwen3-Reranker, Open-Jev | Natural for `noul` |
| **Last-token head** | Linear layer or MLP on the final hidden state | `*ForSequenceClassification`, Qwen3Guard-Stream | Requires a fixed number of classes: poorly suited to dynamic options |
| **Marker after each option + head** | A special token after each candidate; head on the marker hidden states | **Kev** (pointer `</opt>` × `<decide>`), jina-reranker-v3, YOJO | It is the causal equivalent of Laya. YOJO: 3.9× lower latency, +10 points |
| **Prefix sharing / tree mask** | Shared state, questions in isolated branches | Jev (hypothesis), Hydragen, DeFT | **Does not work on GDN**: requires the state prefix-fork |
| **FIRST** | Ranking from the logits of the first generated token | FIRST (2024) | For reranking |

---

## 9. Calibration, scoring rules and RLCD

### 9.1 Foundations

- **Strictly proper scoring rules.** Log (NLL), Brier (quadratic), spherical, RPS/CRPS (for ordinal outcomes). They are maximized in expectation only by the true distribution (Gneiting & Raftery 2007).
- **ECE is incomplete:** it ignores the "grouping loss". Always report it together with NLL and Brier.
- **Temperature scaling** (Guo et al. 2017). All decision models use it:

  | Model | Temperature |
  |---|---|
  | Kev | T = 2.30 |
  | JevK5 | T = 1.53 |
  | Laya | one per question type and number of options |

- **Base vs post-trained.** **Base** models are better calibrated than post-trained ones (RLHF, thinking), which are overconfident (BaseCal, ACL 2026). This is a strong argument for starting from a **Base** model, which Qwen3.8 does not have.
- **Option bias.** PriDe (ICLR 2024) estimates and removes the preference for certain letters or positions. Batch Calibration (ICLR 2024) corrects contextual bias.
- **Guard models and general-purpose LLMs used as moderators** are overconfident: the optimal temperature ranges from 2.7 to 9.0 (arXiv 2609.19072).

### 9.2 Scoring rules as RL rewards

- **RLCR, "Beyond Binary Rewards"** (ICLR 2026). Reward = correctness − Brier(confidence).
  - *Bounded* rules (Brier) incentivize accuracy and calibration jointly.
  - The unbounded log-score can reward confident wrong answers.
  - HotpotQA: ECE 0.37 → 0.03 at equal accuracy.
- **GRPO is overconfident** on stochastic outcomes. The cause is **normalization by the group standard deviation**: removing it fixes the problem (Bereket & Leskovec 2025). Laya uses it.
- **Other work:**
  - Rewarding Doubt (log-score + PPO);
  - ConfTuner (tokenized Brier);
  - C2GSPG and CAPO;
  - "Confidence reward hacking" (2026): the best reward depends on the dataset, so the choice should be treated as a hyperparameter.

### 9.3 Is RL really needed? Evidence from eve-rlcd (Qwen3-0.6B)

| Training | Accuracy | Brier | ECE | Notes |
|---|---|---|---|---|
| RLCD (reward `c − p(a)`) | 0.823 | 0.267 | 0.023 | On an ambiguous probe (truth 0.5): 0.593 |
| RLVR (reward = correctness) | 0.778 | – | 0.213 | Collapses to 0.99 confidence |
| **Supervised** | 0.817 | **0.176** | **0.014** | – |

**Shared conclusion.** When labels or teacher distributions are available, it is better to **directly minimize a proper loss** (soft CE, optionally + Brier, + RPS for `score`).

RL with scoring rules only makes sense with:
- **bandit feedback**, when only the outcome of the chosen option is observed;
- **delayed or multi-turn outcomes**;
- **real environments**, such as the games and browser tasks of decider v10.

### 9.4 Abstention, cascades, conformal prediction

- **Selective prediction.** Accuracy at a given coverage; different thresholds for different risk levels.
- **System 1 → LLM cascades.** Sending only the uncertain cases to the LLM yields frontier quality at 25–50% of the cost (NYU, 2609.24574).
- **Conformal prediction.** Option sets with guaranteed coverage, which can be built on top of any calibrated softmax. Useful when errors are costly.

### 9.5 Distillation from LLMs

- Classifiers trained on LLM labels perform about as well as those trained on human labels.
- **Calibrate the teacher before distilling**: teacher calibration correlates with student accuracy.
- **A mix of hard and soft labels** is better than either one alone (ICML 2026).
- Teachers used in practice: Qwen3.5-27B (decider) and Qwen3.6-27B thinking (JevK5). **Qwen3.8-27B is the natural candidate for us.**

---

## 10. Other "semantic" paradigms (JEPA, LCM, CALM…)

These represent the broader meaning of "semantic model": models that reason or predict in a **representation space** rather than in token space.

| Work | Idea | Key result | Feasible on Qwen? |
|---|---|---|---|
| **LLM-JEPA** (Huang, LeCun, Balestriero; ICLR 2026) | LM loss + λ·distance between the predicted embedding of one "view" and the embedding of the other view | Llama-3.2-1B: GSM8K 32.4 → 36.4. Training cost about 2× | **Yes**, as an auxiliary loss during fine-tuning. Requires pairs of views (for example paraphrases of the state) |
| **Semantic Tube Prediction** (2026) | JEPA-style regularizer on hidden-state trajectories | Same accuracy with 16× less data | Yes, as a regularizer |
| **VL-JEPA** (Meta, Dec 2025) | Predicts the *embedding* of the target text; decodes only when needed | −50% trainable parameters; beats CLIP/SigLIP2 on video classification | Only the idea: predict an embedding and compare it with the embeddings of the options |
| **The JEPA Paradox in Language** (Jul 2026) | Masked text has many valid continuations, with no latent center | Pure T-JEPA collapses systematically | A **warning** against latent-only prediction on text |
| **Large Concept Models** (Meta, Dec 2024) | Autoregression over SONAR sentence embeddings | Excellent multilingual zero-shot | **No**: different space, requires SONAR and pretraining |
| **CALM** (WeChat AI, Oct 2025) | Autoencoder that compresses 4 tokens into one vector; energy-score head | −44% training FLOPs at equal quality | **No**: pretraining from scratch |
| **Beyond Tokens** (EACL 2026) | Targets as *concept sets* (paraphrases) instead of the exact token | Improves perplexity and 7 tasks on Llama-3-8B | Yes, as cheap post-training. Pairs well with soft targets |
| **Coconut** and latent reasoning | Reasoning happens in the hidden state instead of in the CoT | Good on planning, mixed on math | Of little relevance to System 1 |

**Verdict.** None of these paradigms is necessary for a decision model.
- Cheap, optional experiments: the **LLM-JEPA/STP** auxiliary loss with paraphrases of the state, and **soft or concept-level targets**.
- Out of reach because they require pretraining: LCM, CALM and latent reasoning.

---

## 11. Open problems in the field

1. **Calibration is not portable.** It depends on the data distribution: it always has to be recalibrated on your own traffic with a few hundred labels.
2. **Aleatoric uncertainty and "I don't know".** Coins, dice and unknowable items are handled poorly by almost every system.
3. **Listwise or independent.** Comparing the options against each other (listwise) introduces order sensitivity and IIA violations. Independent scores are order-invariant but cannot handle "none of the above".
4. **Consistency across questions.** No system guarantees invariants such as P(A) + P(¬A) = 1 across different questions.
5. **High cardinality.** Beyond 20–77 options encoders collapse; every system degrades when the number of options doubles.
6. **Knowledge ceiling.** Multi-step math, dates and multi-hop cannot be done in a single pass: they have to be delegated to code or to System 2.
7. **Long states and prompt injection.** A single injected line can make accuracy collapse (openjev: 0.833 → 0.467). Training runs use short states (384–4k tokens).
8. **Evaluation hygiene.** Gold labels from teachers, specialists mixed with generalists, test splits used in training, metrics defined differently, non-comparable latencies.
9. **Data is the real competitive advantage** (TypeSafe says so explicitly). Open replications inherit their teachers' biases.

---

## 12. Uncertain points

- Jev's architecture and size: not disclosed, only black-box hypotheses.
- Jev's launch date: 15, 18 or 19 September depending on the source. This document uses the 15th, from the official blog and Forbes.
- Jev's ECE: ranges from 0.107 to 0.246 depending on task, binning and source.
- JevBench rankings: they differ across versions (v1.4 and v1.4.2) and list 36, 76 or 93 systems.
- Qwen3.5/3.8 quality in **Italian**: Qwen claims 201 languages, but there are no per-language numbers.
- Classification quality with only 25% bidirectional layers on a hybrid Qwen: never published.
- Latency and memory estimates for full fine-tuning: theoretical calculations, still to be measured.
- Frontier model names (GPT-6 Astra, Fable 5.1, GPT-5.6 Sol, Opus 5…) are reported as they appear in the sources.
