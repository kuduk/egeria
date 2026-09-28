[Italiano](../04-stato-con-immagini.md) · **English**

# Image states

> Status: **integrated into the API** (`egeria decide` with the state as a list of text/image parts, §6). **26/26** on the 2B models in the test suite, real photos included ([scripts/demo_immagini.py](../../scripts/demo_immagini.py)).

## 1. Why it is possible

Qwen3.5 models, even the 0.8B and 2B, are **natively multimodal**:
- the architecture is `Qwen3_5ForConditionalGeneration`;
- the vision tower is a 24-layer ViT with hidden size 1024, already included in the weights;
- a processor converts images into visual tokens that are inserted into the sequence.

The decision readout does not change: the image enters the state, and at the end the logits of the letters are read at the last position.

Jev and Laya accept text only. Among the open replicas, only decider has a "vision" variant (2B). An image state (scanned documents, invoices, photos of products or damage, screenshots, signs) is therefore a **differentiator**.

## 2. How the prototype works

1. Load the full model and `AutoProcessor`. GPU memory for the 2B in bf16 is 4.4 GB.
2. The prompt uses the same text format as F0. The state contains the placeholder `<|vision_start|><|image_pad|><|vision_end|>`, and the text is built with the **tokenizer's** chat template: the processor of the Base models has no chat template.
3. `processor(text=[prompt], images=[img])` expands the placeholder into visual tokens and prepares `pixel_values` and `image_grid_thw`.
4. `model.model(**inputs)` returns the final hidden state. The readout uses the lm_head rows of the letters, with 2 permutations.

## 3. Results

**Suite:** [examples/suite_immagini.jsonl](../../examples/suite_immagini.jsonl), 26 questions on 9 images, with 2 permutations of the options.
- 3 synthetic images: a colored shape, a danger sign, an invoice.
- 6 **real** photos and screenshots from Wikimedia Commons, under free licenses; authors and licenses in [examples/immagini/FONTI.md](../../examples/immagini/FONTI.md):
  - a Dutch receipt, folded and photographed;
  - a phishing email (domain slamming) in Gmail;
  - a crashed car;
  - a stop sign with a street-name sign;
  - a house on fire;
  - a sleeping cat, used as a "harmless" control with the **same action options** as the fire.

| Model | Correct | Errors |
|---|---|---|
| **Qwen3.5-2B-Base** | **26/26** | – |
| **Qwen3.5-2B** | **26/26** | – |
| Qwen3.5-0.8B-Base | 21/26 | invoice "corrisponde" ("matches"; wrong), receipt "supera 50 €" ("exceeds €50": yes), crash "ribaltamento" ("rollover"), cat "pericolo sì" ("danger: yes") and "monitorare" ("monitor") |

**Examples from the 2B-Base** (probability of the answer):
- **Extraction from a real receipt, folded and photographed:**
  - total "14,21": 1.00;
  - payment by PIN: 1.00;
  - city "Rotterdam": 1.00;
  - "la spesa supera 50 €? no" ("does the purchase exceed €50? no"): 0.68. The numeric comparison is less confident than the extraction.
- **Email:**
  - subject: 1.00;
  - "chiede di cliccare un link per aggiornare dati? sì" ("does it ask you to click a link to update your data? yes"): 0.97.
- **Crash:** "urto contro un ostacolo fisso" ("collision with a fixed obstacle"): 0.90; color "bianca" ("white"): 1.00.
- **Stop:** sign 1.00; street name "via XXV Aprile" read from the sign: 1.00.
- **Same action question, two images:**
  - fire → "chiamare subito i vigili del fuoco" ("call the fire brigade immediately") (0.97), severity "alta" ("high") (0.99);
  - cat → "nessuna azione" ("no action") (0.87), "c'è pericolo? no" ("is there any danger? no") (0.99).

**Takeaways:**
1. **On images the 2B models are very strong zero-shot**, unlike on the typed-decisions text benchmark, where they stayed at the level of the Prior. These questions are about **perception and extraction** (reading a value, recognizing an object or an obvious situation), which vision-language models are trained on extensively. typed-decisions, instead, asks the model to align with an implicit decision policy on ambiguous cases.
2. **The 0.8B breaks down on comparisons and judgments** (mismatched amount, €50 threshold, danger), not on reading: it reads total, vendor, city and street correctly.
3. **Latency depends on resolution.** At 800 px the portrait receipt becomes ~1,400 visual tokens: 549 ms on the 2B, 289 ms on the 0.8B. Small images cost 60–100 ms. Lowering the processor's maximum resolution (`max_pixels`) and computing the vision tower once per request are the two main optimizations.
4. **Caveat:** 26 questions picked to have a clear-cut answer are not a benchmark. We still need ambiguous cases, long documents and low-quality images, plus calibration on images.

## 4. Towards real time

**Scenario.** One frame at a time, 3 questions per frame in the same batch (fire in progress? action? severity?), 1 permutation, bf16, RTX 4070 Laptop. The time includes image preprocessing on the CPU (resizing and normalization) and the forward pass.

**Resolution** (before the fast kernels):

| Max side | Visual tokens | 0.8B-Base | 2B-Base |
|---|---|---|---|
| 800 px | 475 | 274 ms (3.6 fps) | 476 ms (2.1 fps) |
| 448 px | 140 | 101 ms (9.9 fps) | 166 ms (6.0 fps) |
| 224 px | 70 | 80 ms (12.5 fps) | 122 ms (8.2 fps) |

The answers are correct at every resolution. Below 448 px the time barely drops any further: it is a "floor" caused by the number of kernels, over 3,000 per forward pass, not by the amount of compute.

**Where the floor came from** (profiler, 0.8B, 224 px, 62 ms forward pass):
- the vision tower takes 15 ms, because **the same image is encoded 3 times**, once per question;
- the language part takes 47 ms, dominated by the kernels of the PyTorch fallback for DeltaNet: triangular solves `batch_trsm` and `sgemm` in fp32, depthwise convolution, hundreds of elementwise kernels.

**Effect of the fast kernels** (forward pass, 224 px):

| | PyTorch fallback | + flash-linear-attention | + causal-conv1d | + `torch.compile` (reduce-overhead) |
|---|---|---|---|---|
| 0.8B-Base | 77 ms | 68 ms | **45 ms** | 49 ms |
| 2B-Base | 116 ms | 101 ms | **91 ms** | 94 ms |

**Takeaways:**
- **With both fast kernels:** about 51 ms per frame on the 0.8B (~20 fps) and about 97 ms on the 2B (~10 fps), preprocessing included.
- **`torch.compile` with CUDA graphs** helped without the fast kernels (77 → 50 ms), but with the custom kernels the graph breaks and it no longer brings any gain. Compilation is a one-off cost of 15–140 s.
- **Building causal-conv1d** means compiling from source, about 6 minutes for torch 2.11 with `TORCH_CUDA_ARCH_LIST="8.9"`: no prebuilt wheel exists.

**Image once vs. image per question** ([scripts/bench_prefisso_immagine.py](../../scripts/bench_prefisso_immagine.py)). In the measurements above, **every question reprocessed the image**: a batch of 3 full prompts, each with its own copy. The shared variant:
1. computes instructions + image **once**, with the cache;
2. duplicates the hybrid cache for the questions: DeltaNet state + conv + KV of the attention layers;
3. batches only the tails (question + options), with M-RoPE positions = index + `rope_deltas`.

The answers are identical and the probabilities match within bf16 noise.

| Model, image side | 3 full prompts (image ×3) | Image + prefix once, 3 tails | Single question |
|---|---|---|---|
| 0.8B, 224 px | **51 ms** | 80 ms | 42 ms |
| 0.8B, 448 px | **64 ms** | 78 ms | 44 ms |
| 0.8B, 800 px | 181 ms | **97 ms** (−46%) | 61 ms |
| 2B, 224 px | 96 ms | **92 ms** | 54 ms |
| 2B, 448 px | 133 ms | **96 ms** (−28%) | 59 ms |
| 2B, 800 px | 392 ms | **166 ms** (−58%) | 130 ms |

**Takeaways:**
- **With small images, repetition is cheap.** On the 0.8B at 224 px the 2 extra questions, image included, cost 9 ms (42 → 51 ms). The shared variant makes **two sequential passes** (prefix, then tails), so it pays the fixed per-pass cost twice, about 35–40 ms of small kernels on the 0.8B.
- **With large images or more questions, sharing wins clearly**, up to −58%, because each extra question costs only its tail. The multimodal scorer will have to pick the strategy based on visual tokens × questions.
- **The real limit is the fixed per-pass cost**: 42 ms for one question on the 0.8B, of which only 8 ms is the vision tower.

**Next steps towards real time:**
1. **CUDA graphs on our layer loop.** `torch.compile` breaks on the custom kernels. In streaming, however, shapes are fixed, so the scorer's block-by-block loop can be captured by hand, with precomputed masks and rotary embeddings, for both the prefix and the tails. This attacks the fixed cost directly.
2. **Dynamic depth.** In a video stream most frames are "nothing is happening", with very confident answers (the cat: "nessuna azione" ("no action"), "pericolo: no" ("danger: no") at 0.99). These are ideal cases for exiting early.
3. **Recurrent state across frames.** The Gated DeltaNet layers have a fixed-size state. In principle only the tokens of the new frame could be processed, keeping the state: the cache of the full-attention layers would still need to be handled with a window. To be explored.
4. **System 1 → System 2 cascade.** Per-frame decisions run at 10–20 fps on the small model, and only `uncertain` or relevant events go to a large model.

## 5. API integration

- **State.** It is a **list of parts**: `{"type": "text", "text": ...}` and `{"type": "image", "path" | "url" | "base64": ...}`. Multiple images are allowed and their order is preserved in the prompt.
- **`image_max_side`** (optional, 64–4096): downscales the images, so the number of visual tokens and the latency go down (§4).
- **Model loading.** `egeria decide` detects images in the state and loads the full model with the vision tower on its own (`--vision` forces it). Memory for the 2B in bf16 is 4.4 GB.
- **Supported question types:** `noul`, `choice`, `score`, `rank`, `number`, with permutations and `min_confidence` (the `status` is computed at the last layer).
- **`embed` and `memory` also work with images:** the vector includes the visual tokens. Check: 5/6 in [06-memory.md](06-memory.md) §6.
- **`open` (one word or a value) also works with images** and reads dates, amounts and codes (§7).
- **Not yet supported with images:**
  - early exit, because the M-RoPE positions of the visual tokens require the adapted layer loop.
- **Efficiency.** The prompt with the image is repeated for every question and permutation. With small images this costs little; with large images the shared prefix (§4) pays off, but it is still to be integrated.

## 6. Usage examples

### Car accident claim ([examples/immagini_sinistro.json](../../examples/immagini_sinistro.json))

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/immagini_sinistro.json
```

```json
{
  "state": [
    {"type": "text", "text": "Foto allegata dal cliente alla denuncia di sinistro n. 2026-0913:"},
    {"type": "image", "path": "examples/immagini/incidente_auto.jpg"}
  ],
  "image_max_side": 448,
  "questions": {
    "danneggiato": {"type": "noul", "instructions": "Il veicolo nella foto è danneggiato?"},
    "dinamica": {"type": "choice", "instructions": "Che tipo di incidente mostra la foto?",
                 "criteria": {"urto_ostacolo": "Urto contro un ostacolo fisso", "tamponamento": "Tamponamento fra veicoli",
                              "ribaltamento": "Ribaltamento del veicolo", "nessun_incidente": "Nessun incidente visibile"}},
    "gravita": {"type": "score", "instructions": "Quanto è grave il danno visibile?",
                "criteria": ["Nessun danno", "Lieve (graffi)", "Moderato (carrozzeria deformata)", "Grave (veicolo distrutto)"]}
  }
}
```

Actual response (2B-Base, 2.3 s including image loading):

```json
"danneggiato": {"type": "noul", "noul": 0.984, "confidence": 0.968},
"dinamica": {"type": "choice", "choice": "tamponamento",
  "probabilities": {"urto_ostacolo": 0.477, "tamponamento": 0.478, "ribaltamento": 0.015, "nessun_incidente": 0.030},
  "confidence": 0.305},
"gravita": {"type": "score", "score": 2.11,
  "probabilities": {"0": 0.014, "1": 0.043, "2": 0.760, "3": 0.182}, "confidence": 0.746}
```

**How to read it:**
- `danneggiato` ("damaged"): yes, with high confidence; correct.
- `gravita` ("severity"): moderate, correct (deformed bodywork).
- `dinamica` ("how it happened"): a tie between "urto contro un ostacolo fisso" ("collision with a fixed obstacle") and "tamponamento" ("rear-end collision"), with low confidence (0.30). The ambiguity is real: in the background, behind the hedges, there is another car. The model *signals* that it cannot tell them apart. With `"min_confidence": 0.5` that question would come out as `uncertain`, i.e. to be handed to a human.

**Atomic questions instead of one compound question.** This is the "harness" pattern that TypeSafe also recommends: simple questions to the model, logic in the code. On the same photo:

| Atomic question (`noul`) | P(yes) | Confidence |
|---|---|---|
| L'auto bianca in primo piano ha urtato con il muso un palo o un oggetto fisso? ("Did the white car in the foreground hit a pole or a fixed object with its front end?") | 0.84 | 0.67 |
| Ha danni visibili nella parte posteriore? ("Does it have visible damage at the rear?") | 0.06 | 0.87 |
| Si vede un contatto fra l'auto bianca e un altro veicolo? ("Is there visible contact between the white car and another vehicle?") | 0.48 | 0.03 |
| Nella foto è visibile anche un altro veicolo, sullo sfondo? ("Is another vehicle also visible in the photo, in the background?") | 0.86 | 0.73 |

**What comes out of it:**
- The model sees the other car, but recognizes the frontal collision with a fixed object and the absence of rear damage.
- A rule in the code ("frontal collision with a fixed object and no rear damage → collision with an obstacle") yields the right answer for how the accident happened.
- The only real doubt, contact between the two vehicles, stays explicit, with confidence 0.03.

### Security camera frame ([examples/immagini_incendio.json](../../examples/immagini_incendio.json))

```json
{
  "state": [{"type": "text", "text": "Fotogramma dalla telecamera 3:"},
            {"type": "image", "path": "examples/immagini/casa_incendio.jpg"}],
  "image_max_side": 336,
  "min_confidence": 0.6,
  "questions": {
    "incendio": {"type": "noul", "instructions": "C'è un incendio in corso?"},
    "azione": {"type": "rank", "instructions": "Quale azione è più appropriata?",
               "criteria": {"chiamare_vigili": "Chiamare subito i vigili del fuoco",
                            "monitorare": "Monitorare la situazione", "nessuna": "Nessuna azione"}}
  }
}
```

```json
"incendio": {"type": "noul", "noul": 0.969, "confidence": 0.938, "min_confidence": 0.6, "status": "decided"},
"azione": {"type": "rank", "ranking": ["chiamare_vigili", "nessuna", "monitorare"],
  "probabilities": {"chiamare_vigili": 0.942, "monitorare": 0.028, "nessuna": 0.030},
  "confidence": 0.913, "min_confidence": 0.6, "status": "decided"}
```

**Images from a URL or in base64** (instead of `path`):

```json
{"type": "image", "url": "https://upload.wikimedia.org/.../foto.jpg"}
{"type": "image", "base64": "iVBORw0KGgoAAAANSUhEUgAA..."}
```

**Test suite with expected answers.** A simpler JSONL format, one question per line, for comparing models:

```bash
.venv/bin/python scripts/demo_immagini.py Qwen/Qwen3.5-2B-Base examples/suite_immagini.jsonl
```

```json
{"image": "examples/immagini/incidente_auto.jpg", "question": "Il veicolo è danneggiato?", "options": ["sì", "no"], "expected": "sì"}
```

## 7. Reading vs reasoning

A test on a photographed **identity card** (the personal data is not reported here) and on a receipt. Model: Qwen3.5-2B-Base, no calibration.

| Question | Type | Result |
|---|---|---|
| "Vi è una persona nella foto?" ("Is there a person in the photo?") | Yes/No | ✅ Yes, 93–94% |
| "È un documento di identità?" ("Is it an identity document?") | Yes/No | ✅ Yes, 94–95% |
| Year of birth (choice among nearby years) | choice | ✅ correct, 99% |
| Date of birth | value (`open`) | ✅ read in full, 92% at 448 px, 97% at 896 px |
| Expiry date | value (`open`) | ✅ read in full, 98–99% |
| Receipt total | value (`open`) | ✅ "14,21", 98–99% |
| "La persona ha più di 21 anni?" ("Is the person over 21?") | Yes/No | ❌ No at 31% (448 px); 50% at 896 px |
| Same question, with today's date in the text | Yes/No | ⚠️ 57–62% |
| "È nata prima del 2005?" ("Was she born before 2005?") | Yes/No | ❌ No at 39%, even though it reads the year correctly |

**The model reads very well but cannot do comparisons and calculations** in a single pass. For "più di 21 anni?" ("over 21?") it would have to:
1. read the date of birth;
2. know today's date, which it does not;
3. compute the age;
4. compare it.

This is the same limitation documented for Jev (dates, arithmetic).

**Rule of thumb:** ask the model for the **value** ("Qual è la data di nascita?" ("What is the date of birth?"), question type *Una parola o un valore* ("One word or a value")) and do the comparison in code or by hand. To read small text, use the *Massima* ("Maximum") image quality (896 px).

**Fixes made after this test:**
- *Una parola o un valore* now also works with images.
- Reading no longer stops at punctuation: the completion continues up to the first space, within 16 tokens, because the Qwen tokenizer splits digits one by one.
- Calibration is no longer applied to images, and the server does not use it by default. With the calibration fitted on the text tickets, "Vi è una persona nella foto?" dropped to 59%.
