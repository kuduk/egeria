[Italiano](../09-controlli-senza-etichette.md) · **English**

# Label-free checks

> Status: **run** on 29/09/2026 (F0.5, item 2 of the roadmap in [02-implications-and-proposal.md](02-implications-and-proposal.md) §6), on the 2B and the 0.8B. None of the three thresholds is met, but the results change two decisions: the number of permutations and step 3 of the roadmap (§5). The chosen corrections (§8–9) are **in the code since 29/09/2026**: all rotations and the yes-bias correction by default, the calibration on the history as an option.

## 1. What they measure

Three flaws that can be measured without labels, by comparing the model with itself:

1. **Option position.** `choice` questions whose options are all identical ("Option", "Option", …): a model with no position preference gives them equal probabilities. We measure the mean total variation (TV) distance from the uniform distribution, for 2 to 5 options, with 1, 2 and all permutations (cyclic rotations, as in the scorer).
2. **Yes-bias.** Every Yes/No statement is also asked negated, with the true/false descriptions swapped. If the model is consistent, P(Yes | A) + P(Yes | not A) ≈ 1: the **excess** above 1 is the yes-bias. A **contradiction** is the same answer to A and to not A. Variant: the same statement as a `choice` with the option "the state does not make it possible to tell".
3. **Language.** The same question on the same state in Italian and in English: share of identical answers (**agreement**) and mean TV between the two distributions.

**Data** ([examples/controlli/](../../examples/controlli/)), all generic and with no personal data:
- **negations** ([negazioni.json](../../examples/controlli/negazioni.json)): 348 pairs. They are 300 in English on 200 typed-decisions cases (6 statements), 20 on the Italian suite (6 statements, in Italian and English) and 28 on the images (14 statements, 6 of them deliberately false, in both languages);
- **translations** ([lingue.json](../../examples/controlli/lingue.json)): 1051 questions compared. They are the 20 typed-decisions questions in Italian (on 200 cases), the 15 of the Italian suite in English and the 26 image questions in English;
- **position:** 81 states (60 from typed-decisions, 12 from the suite, 9 images).

**Thresholds fixed before the results:**

| Check | Threshold |
|---|---|
| Position | mean TV ≤ 0.05 for every number of options |
| "Yes" | excess ≤ 0.05 and contradictions ≤ 10% |
| Language | agreement ≥ 90% and mean TV ≤ 0.10 |

Script: [scripts/controlli_senza_etichette.py](../../scripts/controlli_senza_etichette.py). The full report goes to `runs/controlli/`.

## 2. Position: a strong preference for A, and 2 permutations are not enough

**With identical options and a single permutation, the model picks A:**

| | 2 options | 3 | 4 | 5 |
|---|---|---|---|---|
| 2B, mean probability of A | 0.76–0.82 | 0.81–0.87 | 0.82–0.86 | 0.84–0.88 |
| 0.8B, mean probability of A | 0.93–0.96 | 0.88–0.92 | 0.86–0.91 | 0.86–0.91 |

(The ranges go from the Italian to the English instructions.)

**Mean TV from the uniform distribution** (the worse of Italian and English):

| Permutations | 2B | 0.8B |
|---|---|---|
| 1 | 0.71 | 0.71 |
| 2 (today's default) | **0.41** | **0.38** |
| all rotations | **0.00** | **0.00** |

With 2 options, 2 permutations are enough. **From 3 options up they are not:** with 2 rotations one option never reaches position A. With 3 identical options the 2B gives on average 0.47 / 0.10 / 0.42, and the middle one is penalized for no reason. With all rotations every option goes through every position, and the distortion disappears by construction.

**Check against labels** (typed-decisions test, 400 cases, 2000 decisions, 2B, no calibration):

| Permutations | Accuracy | `choice` | NLL | ECE | Time per case |
|---|---|---|---|---|---|
| 1 | 0.458 | 0.478 | 1.385 | 0.243 | 150 ms |
| 2 | 0.469 | 0.525 | 1.193 | 0.224 | 239 ms |
| **all** | **0.493** | **0.603** | 1.197 | **0.204** | 331 ms |

Paired comparison, per-case bootstrap with 95% interval:
- **all vs 2, on the 2B:** accuracy +2.4 points [+1.5, +3.2], on `choice` questions **+7.8 points [+5.2, +10.6]**, better Brier, NLL unchanged. `noul` and `score` questions do not change: with 2 options, or with the forward and reverse order of scales, 2 permutations are already complete;
- **1 vs 2:** significantly worse (accuracy −1.2, NLL +0.19);
- **0.8B, all vs 2:** no difference (−0.4 points, an interval that includes zero), time from 145 to 188 ms per case.

**Conclusions:**
- **A single permutation is ruled out.**
- **With all rotations the zero-shot 2B reaches 0.493 and beats the prior (0.478) for the first time.** The preference for A was hiding answers the model "knows".
- **The cost,** with state reuse, is +38% time on the 2B and +30% on the 0.8B, not double or triple.

## 3. "Yes": two different flaws, not one

**Summary** (excess, contradictions, and among them the share of "both Yes" or "both No"):

| Source | 2B: excess | 2B: contradictions | 0.8B: excess | 0.8B: contradictions |
|---|---|---|---|---|
| typed-decisions (en) | +0.36 | 73% (70% both Yes) | +0.43 | 100% (all both Yes) |
| Italian suite (it) | −0.15 | 20% (both No) | +0.36 | 100% (both Yes) |
| Italian suite (en) | −0.06 | 30% | +0.40 | 100% (both Yes) |
| images (it) | −0.28 | 43% (both No) | +0.22 | 79% (both Yes) |
| images (en) | −0.23 | 36% (both No) | +0.26 | 86% (both Yes) |

**What emerges:**
- **The 0.8B says "Yes" to everything,** negations included: this is the actual yes-bias.
- **The 2B on abstract states (typed-decisions JSON) says "Yes" to both the statement and its negation.** For example "This trace requires human review" 0.90 and "does not require" 0.78, and likewise for 4 statements out of 6.
- **The 2B on concrete cases (images, suite) handles true statements and their false negations well:** "there is a fire in progress" 0.98, "there is no fire" 0.01. **But it struggles to say "Yes" to a true negation:** "the shape is *not* a square" 0.35, "the receipt is *not* from a pharmacy" 0.22. Here all the contradictions are "both No".
- **The "the state does not make it possible to tell" option helps little and not everywhere:** on the 2B, contradictions on typed-decisions drop from 73% to 56%, on Italian images from 43% to 29%, but on the suite they increase. It is not a fix.

**Consequence for step 3 of the roadmap** (calibration with a bias term): **a single bias term is not enough.** The direction changes with the model (0.8B towards Yes) and with the domain (2B towards Yes on abstract states, towards No on true negations). A per-domain bias needs labels from that domain.

**Practical advice, starting now:** write statements in positive form ("the parcel has arrived", not "the parcel has not arrived"). Both the 2B and the 0.8B are unreliable on negations.

## 4. Language: disagreement comes from uncertainty more than from language

| | 2B: agreement | 2B: TV | 0.8B: agreement | 0.8B: TV |
|---|---|---|---|---|
| all (1051 questions) | 0.76 | 0.17 | 0.74 | 0.14 |
| typed-decisions (1000) | 0.75 | 0.17 | 0.73 | 0.14 |
| Italian suite (25) | **0.96** | 0.08 | **0.96** | 0.08 |
| images (26) | **1.00** | 0.03 | **0.96** | 0.05 |
| **confident answers** (p ≥ 0.7 in at least one language) | **0.91** (614) | 0.16 | **0.98** (311) | 0.11 |

**What emerges:**
- **The threshold is not met overall,** but where the model is confident agreement is above 90%, and on clear cases it is 96–100%.
- **Disagreement is concentrated on typed-decisions,** where the model is close to chance even in a single language.
- **Uncertainty weighs more than language.** The distributions still differ on confident answers (TV 0.11–0.16): language shifts the confidence more than the answer.

## 5. Decisions and next steps

1. **Permutations.** *(Since 29/09/2026 the default is `auto`, all rotations: §9.)* The default was 2. Proposal: **all rotations** for `choice` questions, since on the 2B they give +7.8 points and better calibration, at +38% time. This is already possible today with `--permutations 26` (26 is the maximum number of options; `noul` and `score` stay at 2). Since 29/09/2026 a single request can also ask the model server for up to 26 permutations (`"permutations"`).
2. **Step 3 (calibration with a bias term):** to be revised, because a global bias cannot fix flaws that change direction with the model and the domain. Negations become training data: statement/negation pairs with opposite labels, already planned among the F1 augmentations.
3. **This script becomes a regression test:** it is run again after every LoRA (F1, F2) to see whether position, "Yes" and language improve.
4. **Left/right** stays out, for the separate investigation.

```bash
# run the checks again (GPU; about 2–4 minutes)
.venv/bin/python scripts/controlli_senza_etichette.py Qwen/Qwen3.5-2B-Base
# permutations on typed-decisions, against labels
.venv/bin/egeria predict --model Qwen/Qwen3.5-2B-Base --split test --permutations 26 --out runs/perm-tutte/test.jsonl
.venv/bin/egeria paired --a runs/Qwen3.5-2B-Base/test.jsonl --b runs/perm-tutte/test.jsonl
```

## 6. What Laya and SemIf do (code read on 29/09/2026)

| | Laya ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), commit 9d95567) | SemIf ([TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), commit 23cf1f3) | Egeria |
|---|---|---|---|
| **Option position** | **Measures** it: identical options for `score` (`research/eval/presentation_checks.py`) and shuffled order with opaque labels (`metamorphic.py`). The English checkpoint prefers the early slots, more so with more options; the multilingual one avoids the first (#131). Answers changed by order: 28% and 10.6% on two tasks. The planned fix is a position-balanced **retrain**. At inference a single pass, no averaging | **Measures** it only: reversed options, 10 changed answers out of 36 cases, "not solved" in their report. An alternative (a reranker with an independent Yes/No per option) is order-invariant by construction but less accurate (0.53 vs 0.72) | Averaging over permutations at inference; with **all rotations** the distortion disappears (§2) |
| **Negations and "Yes"** | Documented and not fixed: with negated requests the choice follows the question, not the state (#377). `noul` can follow its `false`/`true` labels instead of the state (#156): the remedy is opaque labels (`labels`: `A`/`B`) | Every decision has three fixed options: supported, **insufficient**, contradicted. Pairs with the criterion reversed **and labels**: both must be right (threshold 20 pairs out of 24). States without the information: it must pick "insufficient" | Statement/negation pairs without labels (§3); `noul` uses the letters A/B in both orders, so no "true/false" labels |
| **Language** | Routing by language to a multilingual checkpoint. Consistency measured between simplified and traditional Chinese: 12.8% of answers differ | Not addressed | Same question in Italian and English (§4) |
| **Calibration** | One temperature per type and number of options | One temperature per workload, fitted on its data | One temperature per type; the bias term is to be revised (§3) |

**What we can borrow:**
- **From SemIf, three missing checks:** irrelevant context appended to the state, the criterion reworded with the same meaning, and states lacking the information, where the right answer is "it cannot be told". The first two are measured without labels like the others; the third has its label by construction.
- **From Laya, the opaque-label test:** replace the option keys (for example `spam_truffa`) with neutral letters, to separate the effect of the label words from the effect of position.
- **For training (F1):** position-balanced data, as Laya plans, and pairs with a reversed criterion and labels, as in SemIf.

**Where Egeria is ahead:** neither of them corrects position at inference. Laya leaves it to retraining, SemIf only measures it. Averaging over all rotations removes it today, and with state reuse it costs +38% time instead of a retrain.

## 7. State of the art and mathematical tools (research of 29/09/2026)

**Who has worked on these problems:**

| Problem | Work | What they propose |
|---|---|---|
| Option position ("selection bias") | Zheng et al., ICLR 2024 (**PriDe**) | The distortion comes mostly from a preference for the letter tokens. The prior preference for each letter is estimated by permuting the options on a few examples, then divided out of the other predictions: a single pass, no labels |
| | Choi et al., ACL 2025 | **Bias node pruning** (0.002% of the output projection parameters) and an auxiliary "I don't know" option. A new metric, Choice KL Divergence |
| | Guda et al., IJCNLP-AACL 2026 | A label-free permutation bias metric, unsupervised LoRA, and **voting over all permutations with a shared question-and-context cache**: the same principle as our state reuse |
| | Zheng et al., ACL 2026 (**PA-GRPO**) | Training with permutation groups and a reward for consistency across orders |
| "Yes"/"No" and negations | Braun, EMNLP Findings 2025 | 37,975 variants in English, German and Polish: in English LLMs lean towards **"No"**, the opposite of human acquiescence. This matches our 2B on concrete cases |
| | Huang, July 2026 | "Crossed symmetrization": answer order, words and verdict are flipped in a balanced way. The bias comes from the form (order, the word "no"), not from the judgment. Model P = σ((θ ± m)/s): m measures sensitivity to wording, s how decisive the model is |
| | Burns et al., ICLR 2023 (**CCS**) | Statement/negation pairs and the consistency constraint P(A) ≈ 1 − P(not A), used to find a "true/false" direction in the hidden states without labels |
| Language | Qi et al., EMNLP 2023 (**RankC**); 2025–2026 work on cross-lingual consistency | A consistency metric on ranked answers; multi-language ensembles with voting; training with a cross-lingual consistency reward |
| Label-free calibration | Zhao et al., ICML 2021 (Calibrate Before Use); Zhou et al., ICLR 2024 (**Batch Calibration**) | The "contextual" preference is estimated with an empty input or with the mean prediction over a batch of the same workload, and divided out |

**Mathematical tools and how they apply to us:**

1. **Averaging over a group of transformations (Latin square).** If the logit of an option is `content(option) + b(position)`, averaging the log-probabilities over all cyclic rotations makes every option visit every position once: the `b` term becomes the same for all options and cancels **exactly**. With 2 rotations and 3 or more options this does not happen, which is the result of §2. It costs n passes (the tails, with state reuse).
2. **A prior preference estimated once (PriDe).** Estimate `b(letter)` once, per number of options and language, from the identical options of §2 (they are exactly a "content-free" input), then use **a single permutation** and subtract `b` from the letter logits. In our readout this is one line: the letter logits are `hidden @ slot_weight.T`, and it is enough to subtract a vector. It promises the quality of all rotations at the cost of one.
3. **Polarity symmetrization** (from psychometrics: balanced items against acquiescence; the same idea as CCS). Ask the question and its negation and combine them: `logit P*(A) = (logit P(A) − logit P(not A)) / 2`. If the bias is an additive term in the logits, it cancels exactly, whatever its direction. This is what a flaw that changes direction with the domain needs (§3). It costs twice as much; the negation must be written, or generated with a fixed template ("It is not true that…").
4. **Batch or contextual calibration** (Batch Calibration). Divide out the mean prediction over the workload, without labels. It corrects a bias that differs per domain, if the batch is large and balanced enough.
5. **Auxiliary "I don't know" option.** In the literature it helps on some models; for us the results were mixed (§3).
6. **In training (F1):** consistency across permutations (PA-GRPO, LoRA driven by the permutation metric), position-balanced data, pairs with flipped polarity and labels, and cross-lingual consistency as an objective.

**Experiments possible right away, with the data we have** (typed-decisions labels):
- the prior preference (2) vs all rotations (1): accuracy and time;
- polarity symmetrization (3) on the 600 Yes/No questions, with the hand-written negations and with an automatic negation;
- batch calibration (4) on the yes-bias;
- an Italian plus English ensemble for language.

## 8. Low-cost corrections: results (29/09/2026)

Script: [scripts/esperimento_debiasing.py](../../scripts/esperimento_debiasing.py). A single model pass over the typed-decisions test computes the letter logits for all rotations, for the empty state ("N/A") and for the negated Yes/No questions; the variants are then simulated on the same numbers. Evaluation on 380 cases (the 20 calibration cases excluded), no temperature. Times per case come from the 2B measurements of §2.

**2B:**

| Variant | Accuracy | Yes/No | Choice | Scale | NLL | ECE | Time per case |
|---|---|---|---|---|---|---|---|
| 2 permutations (today) | 0.471 | 0.511 | 0.519 | 0.405 | 1.193 | 0.220 | ~240 ms |
| all rotations | 0.494 | 0.511 | 0.596 | 0.405 | 1.197 | 0.201 | ~330 ms |
| 1 permutation + fixed per-letter prior (PriDe) | 0.422 | 0.507 | 0.449 | 0.338 | 1.456 | 0.320 | ~150 ms |
| 1 permutation + prior from the empty state | 0.485 | 0.535 | 0.498 | 0.437 | 1.287 | 0.202 | ~150 ms + 1 pass per question |
| 2 permutations + empty input | 0.473 | 0.533 | 0.511 | 0.399 | **1.065** | **0.153** | ~240 ms + 1 pass per question |
| **1 permutation + batch (estimated on other cases)** | **0.529** | 0.605 | 0.528 | 0.474 | 1.023 | **0.053** | **~150 ms** |
| 2 permutations + batch (estimated on other cases) | 0.540 | 0.625 | 0.505 | 0.503 | 1.005 | 0.073 | ~240 ms |
| all rotations + batch (estimated on other cases) | **0.545** | 0.625 | 0.523 | 0.503 | **1.000** | 0.080 | ~330 ms |
| all rotations + batch, history of only 20 cases | 0.516 | 0.635 | 0.451 | 0.475 | 1.084 | 0.065 | ~330 ms |
| Yes/No symmetrized, hand-written negation | – | 0.556 | – | – | – | – | Yes/No at twice the cost |
| Yes/No symmetrized, automatic negation | – | 0.505 | – | – | – | – | Yes/No at twice the cost |

Accuracy relative to today, per-case bootstrap (95% interval): all rotations +2.3 [+1.5, +3.2]; prior from the empty state +1.4 [+0.1, +2.7]; **1 permutation + batch +5.8 [+3.0, +8.6]**; 2 permutations + batch +6.9 [+4.3, +9.5]; all rotations + batch +7.4 [+4.7, +10.1].

**0.8B:**

| Variant | Accuracy | Yes/No | NLL | ECE |
|---|---|---|---|---|
| 2 permutations (today) | 0.446 | 0.504 | 1.184 | 0.123 |
| **2 permutations + empty input** | **0.459** | 0.525 | 1.097 | **0.035** |
| 1 permutation + batch (estimated on other cases) | 0.426 | 0.568 | 1.112 | 0.028 |
| all rotations + batch (estimated on other cases) | 0.446 | 0.558 | 1.117 | 0.058 |
| Yes/No symmetrized, hand-written negation | – | **0.602** | – | – |

On the 0.8B, rotations and batch calibration do not improve accuracy (differences within ±3 points, intervals that include zero). Calibration improves a lot: ECE from 0.12 to 0.03–0.06.

**What emerges:**
1. **The fixed per-letter prior (PriDe) makes things worse** on both models: the position distortion of our small models changes from case to case, it is not a constant vector. To be dropped.
2. **Batch calibration is the strongest correction and costs nothing at inference.** On the 2B with a single permutation it gives +5.8 points while being 37% faster than today. It recovers almost all the benefit of the rotations, because the distortion is fairly stable *within the same question*.
   - **It needs a history per question:** with ~50 cases it works as well as on the same data, with ~5 choice questions get worse.
   - **It must not be used for rare events.** It pushes the mean prediction towards balance: on "is there a fire?", where the answer is almost always "No", it would create false alarms.
3. **Empty-input calibration** costs one pass per question, which can be cached, and needs no history. It improves calibration on both models (ECE: 2B from 0.22 to 0.15, 0.8B from 0.12 to 0.035) with a small accuracy gain. It is the safest default.
4. **Polarity symmetrization** works only with negations written by a person: +4.5 points on the 2B and +10 on the 0.8B on Yes/No questions. With the automatic negation there is no gain. It doubles the cost of Yes/No questions.
5. **For scales, all cyclic rotations make things worse** (0.404 vs 0.405 on the 2B, NLL from 1.20 to 1.54): scales stay with the forward and reverse order.

**By question type** (same data, accuracy and ECE; analysis of 29/09/2026). Empty input does not help every type equally:

| Type | Variant | 2B | 0.8B |
|---|---|---|---|
| Yes/No | 2 permutations (today) | 0.511 · ECE 0.240 | 0.504 · ECE 0.216 |
| Yes/No | **+ empty input** | **0.533 · ECE 0.131** | **0.525 · ECE 0.040** |
| Choice | 2 permutations (today) | 0.519 · ECE 0.170 | 0.481 · ECE 0.139 |
| Choice | **all rotations** | **0.596 · ECE 0.107** | 0.461 · ECE 0.118 |
| Choice | all rotations + empty input | 0.565 · ECE 0.074 | 0.472 · ECE 0.065 |
| Scale | forward and reverse order (today) | 0.405 · NLL 1.50 · ECE 0.249 | 0.378 · NLL 1.38 · ECE 0.075 |
| Scale | **+ empty input** | 0.399 · **NLL 1.33 · ECE 0.217** | 0.388 · **NLL 1.29 · ECE 0.062** |

- **Yes/No:** empty input improves accuracy and calibration on both models, because it corrects exactly the lean towards Yes. With the empty state the model says Yes with median probability 0.64 (2B) and 0.72 (0.8B), so the correction pushes towards No. However, typed-decisions has only 6 distinct Yes/No questions: the result rests on few questions.
- **Choice:** on the 2B, empty input takes 3 points off the rotations (0.596 → 0.565).
- **Scale:** accuracy does not change, but NLL and ECE improve.
- **Mixed policy** (Yes/No and scale with empty input, choice with all rotations): 2B from 0.471 to 0.498, 0.8B from 0.446 to 0.451, with no history.

**Proposal** (then checked and corrected in §9):
- **empty-input calibration by default for Yes/No and scales,** in the model server, with the empty-state prediction cached per question;
- **all rotations by default for choice questions,** without empty input;
- **optional batch calibration, off by default,** in the web server, which already has the answer history per question:
  - it turns on per question from ~50 cases;
  - never for rare events, nor for the live stream, where the same question receives many near-identical frames;
  - when it is on, 1 permutation is enough and it replaces the empty input (the two together have not been measured);
- **three checks before the default:**
  1. empty input on questions with images, and on a rare-event question where the empty state says No: there the correction would push towards Yes, so it needs to be assessed whether to limit it;
  2. the temperatures must be refitted on top of the new calibration;
  3. at the same `min_confidence`, the `decided`/`uncertain` outcomes change.

**Consequence for the roadmap:** step 3 of F0.5 (calibration with a bias term) becomes "empty-input calibration + batch calibration on the history", both label-free.

## 9. Check and final choice (29/09/2026)

Before making empty-input calibration the default, we tried it where it could do harm: questions on images and on rare events. Script: [scripts/verifica_calibrazione.py](../../scripts/verifica_calibrazione.py); data: [examples/controlli/eventi_rari.json](../../examples/controlli/eventi_rari.json). 2B model, 2 permutations.
- **Images:** the 21 images of `examples/immagini` × 6 Yes/No questions about events present in 2 images out of 21 (fire, accident, cat, email, stop sign, receipt), in Italian and English: 250 questions, 24 positive.
- **Texts:** 30 hand-written messages (15 per language) with questions about a fire and a medical emergency, 3 positives out of 15 per language. Some negatives are close to the event: a lit fireplace, some candles, a headache.

| Yes/No correction (310 questions) | Accuracy | False alarms | NLL | ECE |
|---|---|---|---|---|
| none | 1.000 | 0.000 | 0.066 | 0.060 |
| full, empty state "N/A" | 0.987 | 0.015 | 0.148 | 0.112 |
| full, empty state = grey image (only the 250 on images) | 0.912 | 0.097 | 0.381 | 0.184 |
| capped at ±1 in logit | 0.990 | 0.011 | 0.112 | 0.087 |
| **towards No only** | **1.000** | **0.000** | **0.066** | **0.060** |

**Why the full correction does harm.**
- On these questions the empty state leans towards **No**. P(Yes | "N/A") ranges from 0.10 to 0.29 on images and from 0.22 to 0.44 on texts; with the grey image, from 0.03 to 0.10.
- On typed-decisions it leaned towards Yes instead (median 0.64 on the 2B).
- The full correction also removes the lean towards No, which here is right: with no information, the answer to "is there a fire?" is No. Removing it pushes towards Yes and creates false alarms.

**The chosen rule: correct towards No only.** The shift is removed only when the empty state leans towards Yes.
- On typed-decisions it gives the same result as the full correction: on the 2B Yes/No 0.533 with ECE 0.130; on the 0.8B identical. There almost all questions lean towards Yes.
- On rare events it changes nothing.
- The ±1 cap reduces the harm, but on typed-decisions it loses part of the gain (2B 0.519).

**Scales: no correction.**
- On typed-decisions the empty state leans towards the highest level. The questions are about urgency and severity, and on the 2B the last level gets between 0.58 and 0.89.
- On the only scale of the image suite ("How serious is the situation?") the correction worsens the NLL, from 0.28 to 0.56.
- On typed-decisions it improves only calibration, not accuracy.
- For scales there is no "right" direction like No for Yes/No questions.

**Choice:** all rotations, without empty input. On images the 18 choice questions of the suite are right in every variant, and all rotations give the lowest NLL (0.036 vs 0.043).

**Final check, with the code.** `egeria predict` with the new defaults on the full typed-decisions test set (400 cases, 2,000 decisions). The comparison is paired against the previous predictions with 2 permutations, with a per-case bootstrap:

| | Accuracy | NLL | ECE | ms/case |
|---|---|---|---|---|
| 2B, before (2 permutations) | 0.468 | 1.195 | 0.226 | |
| **2B, new defaults** | **0.496** | **1.161** | **0.169** | 251 |
| 0.8B, before (2 permutations) | 0.450 | 1.185 | 0.122 | |
| 0.8B, new defaults | 0.441 | **1.156** | **0.086** | 163 |

- **2B:**
  - accuracy +2.9 [+1.7, +4.1];
  - choice +8.2 [+5.5, +11.0];
  - Yes/No +1.0 [−2.1, +4.1], with NLL −0.119 [−0.164, −0.074];
  - scales unchanged.
- **0.8B:**
  - accuracy −0.9 [−3.1, +1.3], not significant;
  - NLL −0.029 and Brier −0.022, both significant;
  - Yes/No ECE from 0.215 to 0.064.

**What was implemented:**
- **All rotations by default.** `permutations="auto"` applies everywhere: model server, `decide`, `predict` and `suite`. That is all rotations for Yes/No and choice, forward and reverse order for scales and numbers. An integer from 1 to 26 is still possible.
- **Yes-bias correction.** It applies to Yes/No questions and is on by default. It is turned off with `"yes_correction": false` in the request or `--no-yes-correction` at startup.
  - The shift is computed on the "N/A" state with the same permutations, once per question, and stays in the cache (up to 4,096 questions).
  - `egeria predict` applies it too, so temperatures are fitted on top of the correction.
- **Calibration on the history.** It lives in the web server and is optional per question (`"batch_calibration": true`, the *Calibrate on the history* checkbox). It turns on from 50 cases and never live ([07](07-web-interface.md) §1bis).

**What remained** (done in §10): refitting the temperatures, measuring how the `decided`/`uncertain` outcomes change, checking the lean towards Yes on poses.

```bash
# check of the correction on images and rare events (GPU, ~20 s after loading)
.venv/bin/python scripts/verifica_calibrazione.py Qwen/Qwen3.5-2B-Base
# final check on typed-decisions with the new defaults
.venv/bin/egeria predict --model Qwen/Qwen3.5-2B-Base --split test --out runs/calibrazione/2B-default/test.jsonl
.venv/bin/egeria paired --a runs/Qwen3.5-2B-Base/test.jsonl --b runs/calibrazione/2B-default/test.jsonl
```

## 10. Temperatures, threshold and poses (29/09/2026)

### 10.1 Temperatures refitted

Predictions on the typed-decisions train split (1,200 cases) with the new defaults, then `egeria calibrate`, evaluated on the test split:

| | T Yes/No | T choice | T scale | Test NLL | Test ECE |
|---|---|---|---|---|---|
| 2B, old settings | 7.08 | 2.34 | 3.91 | | |
| **2B, new defaults** | **3.85** | 2.58 | 3.81 | 1.161 → **1.051** | 0.169 → **0.050** |
| 0.8B, old settings | 20 (the maximum) | 2.19 | 2.73 | | |
| 0.8B, new defaults | 20 (the maximum) | 2.35 | 2.70 | 1.156 → 1.127 | 0.086 → 0.046 |

- **2B:** with the yes-bias correction the Yes/No temperature halves, from 7.08 to 3.85. It is still high: on typed-decisions the 2B is only slightly above chance on Yes/No questions (0.527).
- **0.8B:** the Yes/No temperature stays at the maximum. On typed-decisions its Yes/No answers carry no information, and the temperature flattens them to 0.5.
- **Own data only:** the temperatures are fitted on typed-decisions. As before ([07](07-web-interface.md) §1), use them only on similar data and never on images. They are in `runs/calibrazione/<model>-default/temperature.json`; the old ones in `runs/*/temperature.json` are superseded.

### 10.2 Threshold: how many answers become uncertain

`egeria evaluate --min-confidence 0.5` (the model server's default threshold) counts the answers below the threshold and the accuracy of the decided ones. 2B, typed-decisions test split:

| | Uncertain | Accuracy of decided | Yes/No uncertain | Yes/No: accuracy of decided | Choice: accuracy of decided |
|---|---|---|---|---|---|
| before (2 permutations) | 46% | 0.570 | 47% | 0.647 | 0.608 |
| **new defaults** | 58% | **0.590** | 86% | **0.819** | **0.667** |
| new defaults + temperatures | 95% | **0.905** | 100% | – | 0.905 |

- **Without temperatures:** with the new defaults, typed-decisions Yes/No answers are uncertain much more often. Those that stay decided are much more reliable: 0.819 vs 0.647.
- **Clear-cut cases:** they stay decided. On the image questions of §9 the model gives P(Yes) 0.04 on negatives and 0.92 on positives.
- **With temperatures:** the 0.5 threshold leaves only the clearest choices decided, right 90% of the time. That is honest behaviour on a task where the zero-shot model knows little.

### 10.3 The lean towards Yes on poses

**Data.** 24 photos from Wikimedia Commons in `examples/immagini/pose/` (sources and licences in `FONTI.md`), in Italian and English:
- 9 tree poses;
- 4 other one-leg poses (dancer and similar);
- 1 pose with arms raised and both feet on the ground;
- 7 other poses (warrior, chair);
- 3 images with no people: a real tree, a tiger among trees, a bird.

2B model, default readout. Script: `scripts/verifica_calibrazione.py`, part 4.

| Question | Tree (said yes / tree) | On one leg | Arms raised | Other poses | No people |
|---|---|---|---|---|---|
| Yes/No "Is the person doing the yoga tree pose?" | 100% | 62% | 50% | 64% | 0% |
| Yes/No "Is the person balancing on one leg?" | 100% | 100% | 0% | 14% | 0% |
| Yes/No "Is the foot of the raised leg resting against the inside of the other leg?" | 50% | 0% | 0% | 0% | 0% |
| Generic choice: tree / another pose / no person | 100% | 100% | 100% | 100% | 0% |
| **Choice with the poses described** (standing on one leg with the foot against the inside of the other; on one leg with the other raised behind; legs wide apart and one knee bent; bent knees as if on a chair; standing on both feet; no person) | 100% | 100% | **0%** | **14%** | **0%** |

**What emerges:**
1. **It is not the empty-state lean towards Yes.** For the tree pose question the empty state leans towards No (P(Yes) 0.31–0.37), so the correction does not step in. The full one would make things worse: false alarms from 50% to 67%.
2. **It is a content confusion.** The 2B recognises "a person doing yoga", not which pose: with Yes/No it says "tree" even to two warriors out of three. Its Yes answers are not a generic lean: it answers well whether a person is balancing on one leg (95.8%).
3. **Describing the options in words helps.** Other poses taken for the tree drop from 64% to 14%, and raised arms to 0.
4. **The confusion between the tree and the other one-leg poses remains.** The 2B cannot tell a foot resting against the inside of the leg from a leg raised behind. The question about the foot alone is too hard: no false alarms, but it misses half of the tree poses.

**Advice for users (also in [07](07-web-interface.md) §10):**
- for poses, a **choice with the poses described** works better than a Yes/No with the pose name;
- telling one-leg poses apart needs **training**. It is a good argument for the first LoRA on dance poses (F1, [02](02-implications-and-proposal.md) §7).

```bash
# temperatures with the new defaults
.venv/bin/egeria predict --model Qwen/Qwen3.5-2B-Base --split train --out runs/calibrazione/2B-default/train.jsonl
.venv/bin/egeria calibrate --predictions runs/calibrazione/2B-default/train.jsonl --out runs/calibrazione/2B-default/temperature.json
.venv/bin/egeria evaluate --predictions runs/calibrazione/2B-default/test.jsonl --temperatures runs/calibrazione/2B-default/temperature.json --min-confidence 0.5
```
