[Italiano](../11-modelli-e-strumenti.md) · **English**

# Before training: models, tools and literature

> **Status:** web research of 07/10/2026. Starting question: does any existing model or tool already solve the flaws measured in [10](10-left-right.md), with no training?
> - Measured flaws: the person's own left and right, hand chirality, body parts with the 2B-Base, similar-looking poses.
> - Full sources in [sources.md](sources.md). The labels [vendor], [third party] and [independent] say who measured.

## 1. In short

1. **No small open model solves the person's own left and right, and neither do the large ones.** It is an open research problem, not a flaw of ours (§2).
2. **Our observations match the literature.** Fixed answers when the visual signal is weak; ingredients seen but not combined (chirality is an XOR between palm/back and thumb side); no help from resolution.
3. **A model with our very method already exists:** **Intern-Decision** (InternLM, 26/09/2026, Apache-2.0), a fine-tune of Qwen3.5 in 0.8B, 2B and 4B. It should be compared right away (§3).
4. **Body keypoint detectors already give the person's own left and right:** MediaPipe Pose and Hand Landmarker, Apache-2.0, on the CPU. There is a trap: MediaPipe hands assume a mirrored image (§4).
5. **If training is needed, how and how much is known:** a LoRA on mirrored pairs (mirrored image, reversed answer), a few thousand examples per task. AIST++, however, does not fit: the videos are for academic research only and there are no yoga poses (§5).

## 2. What the literature says

**The person's own right is an open problem even for large models** [independent]:

| Benchmark | What it measures | Results |
|---|---|---|
| **MirrorQA** (ACL 2026) | The subject's own left/right ("which hand holds…", "which eye is open"), 5,549 questions | All open models zero-shot 41.9–48.1% (random 42.5%). The best LoRA 65.4%, humans 99.3%. 51–55% of errors use the viewer's frame |
| **SPACE** (ACM MM 2026) | The subject's left/right arm and hand | Several open models just above 50%. Qwen3-VL-8B 66.6–70.4, Gemini-3.1-Pro 81.1 |
| **FlipSet** (2026) | What another person sees | 91% of 103 models (1–90B) are below chance; the best 33.9% |
| **OmniSpatial** (ICLR 2026) | Another observer's frame | The best (o3) 48.4%, humans 94.3% |
| **3DSRBench** (ICCV 2025) | Orientation, with "FlipEval" on mirrored images | The best about 40%, humans 97.7% |

**Fixed answers:** they are documented.
- **Isle-Brick-V2:** small Qwen3-VL models fall back on a default direction when asked which way a figure faces. Zooming in does not help; the authors speak of "linguistic priors over visual facts".
- **ENACT:** finds a right-hand prior.

**Proposed causes** that fit our measurements:
1. **Unstated frame in the training text.** Captions say "on the left" almost always from the viewer's side. That is why objects work and the person's own side does not.
2. **Composition.** The model sees the pieces but does not combine them:
   - the right hand is "palm XOR thumb on the right";
   - the person's right is "facing the camera XOR image half".

   An XOR cannot be read linearly from the two ingredients separately.
3. **Falling back on a habitual answer when the visual signal is weak.** It explains the fixed answers, the mirror not changing them, and resolution not helping.
4. **Base vs Instruct:** Qwen3-VL's instruction tuning contains relation phrases ("the cup to the left of the laptop"). That is the likely reason only the Instruct finds the raised leg.

**Mirroring during vision-encoder training is probably not the cause.**
- CLIP, OpenCLIP and SigLIP 2 do not use it in their main losses.
- The encoder of the small Qwen3.5 models has SigLIP2's dimensions (inferred from the configurations: Qwen does not document it).
- Meta Perception Encoder does use it (in Perception-LM and Step3-VL-10B): to avoid for chirality.

**The fixes with the most evidence**, in order of cost:
1. **Decompose the question and combine the answers in code**, with no training: what already works for us ("in which half of the image…").
2. **Tools:** body keypoint detectors (§4).
3. **LoRA on mirrored pairs:**
   - **CounterCurate:** left/right from 61.8 to 95.7%.
   - **"Art of Interrogation" (2026):** LoRA r=32 on Qwen2.5-VL-3B with 2.4–7.7 thousand examples per task; a **label-free** version, rewarding only consistency with the mirror, reaches 74.1 vs 76.4 for the supervised one.
   - **Plateau:** around 4 thousand examples (AlloEgo-VLM).
4. **"Thinking" or larger models:** modest gains (+3.8 points between Instruct and Thinking on SPACE). Step-by-step reasoning often **worsens** spatial questions.

**Before a LoRA on chirality**, a cheap test: a linear probe on the vision encoder's features, with held-out mirrored pairs. If it cannot separate the right hand from the left, a LoRA on the language model alone will not be enough.

## 3. Models to try

No small model has published numbers on the person's own right, on hands or on poses. Those to measure on our checks, in order of cost:

| Model | Why | Cost to try it |
|---|---|---|
| **Intern-Decision-2B / 4B** (Apache-2.0, on Qwen3.5) | Same method as Egeria: one pass, letter logits, shared state, up to 8 images, calibration. Claims typed-decisions 79.35 (2B) and 80.55 (4B) [vendor], vs our 0.496 zero-shot. No numbers on images | None: same architecture as our scorer. Its prompt format (a JSON skeleton with one placeholder per question) differs from ours, and should be tried in both formats |
| **Qwen3.5-4B and 9B**, Base and Instruct, in 4-bit | Tells whether size alone fixes body parts and poses. The 9B has a larger vision encoder (SigLIP2-So400m dimensions). The 4B beats the 2B on BLINK and CV-Bench [independent and third party] | Low: same code, needs `bitsandbytes` for 4-bit |
| **Qwen3-VL-4B-Instruct** and **Spatial-SSRL-Qwen3VL-4B** | Qwen3-VL injects features from several encoder layers into the language model (DeepStack) and has spatial-relation data. SpinBench 61.4 [independent]. Spatial-SSRL is also trained to recognise mirrored image patches | Medium: different architecture, the scorer needs adapting |
| **InternVL3.5-2B / 4B** (also `-Pretrained`) | The best small model on SpinBench (4B: 65.8) [independent]. Encoder not trained with mirroring. The pretrained/final pair repeats our Base/Instruct comparison | Medium: different architecture |
| **HY-Embodied-0.5 (2B active)** or **VST-3B** | The best numbers under 4B on perspective (ViewSpatial and 3DSRBench, 53–57) [vendor] | Medium-high: pinned transformers version, licence to check |

Excluded:
- **for licence:** FastVLM, Isaac, Perception-LM (non-commercial);
- **for mirroring in the encoder:** Perception-LM and Step3-VL;
- **for architecture:** Florence-2 (no reasoning).

**Other benchmarks useful as evaluation sets:**
- **MirrorQA:** data on GitHub. The paper states CC BY 4.0, GitHub says "Other": to check.
- SpinBench, ViewSpatial-Bench, 3DSRBench.
- For hand chirality and for zero-shot poses **there is no benchmark**: our own sets remain necessary.

## 4. Tools: body keypoint detectors

| Tool | Left/right | Licence | Where it runs | Notes |
|---|---|---|---|---|
| **MediaPipe Pose Landmarker** | 33 points named left/right, 3D coordinates | Apache-2.0 | Python and browser, 25–50 ms on CPU | The docs do not say whose left it is: needs checking. Weak when the head is not visible (person seen from behind) |
| **MediaPipe Hand Landmarker** | 21 points per hand, a "Left/Right" label | Apache-2.0 | Python and browser, ~17 ms on CPU | **Assumes a mirrored (selfie) image**: on our frames, which are not mirrored, the label must be **swapped**. Accuracy not published; known errors near the edges and with a single hand |
| **RTMW / RTMPose** (via `rtmlib`, ONNX) | 133 points, hands included | Apache-2.0 code; weights from datasets with mixed licences | CPU, 11–45 ms | Second opinion for people seen from behind or several people |
| To avoid | | | | YOLO-pose (AGPL), OpenPose and Sapiens v1 (non-commercial), WiLoR (NC-ND) |

**A possible hybrid design:**
- **Where:** MediaPipe Pose and Hand on the server, on the CPU, so the GPU stays with the model.
- **Hands:** each is assigned to the nearest skeleton wrist; MediaPipe's own label, swapped, is used only for close-ups with no body.
- **Two uses:**
  1. fixed questions ("raising the right hand?") are answered directly from the points;
  2. for the others, the facts go into the state as text: "the person's right hand is above the head", "the person faces the camera".
- **Mirror check:** on the mirrored frame the labels must swap, otherwise the fact is uncertain. Same principle as our label-free checks.
- **Resolution:** frames are 448 px today and hands come out small. The detector should get the larger frame, or a crop around the wrists.
- **Risks:**
  - the 2B may ignore facts in the state, to be measured;
  - people seen from behind and several people;
  - points jittering from one frame to the next;
  - 25–45 ms of CPU per model per frame.
- **Yoga poses:** tree and dancer can be told apart with rules on the 3D points (raised foot against the standing leg, or behind the hips). No published accuracy exists for the rules: to measure on our photos.

## 5. Data, if a LoRA is needed

- **AIST++:** the annotations are CC BY 4.0, but **videos and images are for academic research only**. Commercial use needs AIST's written consent and redistribution is prohibited. Besides, it is street dance, with no yoga. **Not suitable** for the first LoRA.
- **Better candidates:**
  - the **MirrorQA** train split (~3.9 thousand examples, licence to check);
  - **our own mirrored pairs**, which give free labels;
  - freely licensed photos labelled automatically by MediaPipe and checked with the mirror.
- **For our letter readout,** the cheap equivalent of "consistency with the mirror" is a symmetric KL loss between P(right | photo) and P(left | mirrored photo). It needs no labels (our proposal, not tried in the literature).

## 6. Proposed plan, before training

Updated on 08/10/2026 with the ideas of §7:
1. **The "is the model looking?" signal** (§7.1). The comparison needs it too, to see which model really looks.
2. **Model comparison (local, ~1 day).**
   - First Intern-Decision-2B/4B, then Qwen3.5-4B/9B (Base and Instruct, in 4-bit; needs `bitsandbytes`), on our checks: [left/right investigation](../../scripts/indagine_destra_sinistra.py), [calibration and pose check](../../scripts/verifica_calibrazione.py), typed-decisions and the image suite.
   - Then, with a small scorer adaptation, Qwen3-VL-4B and InternVL3.5-4B.
3. **Derived questions** (§7.2).
4. **Only if 1–3 are not enough:** the linear probe on the encoder, then a LoRA on mirrored pairs (MirrorQA + ours) with label-free consistencies (§7.3).
5. **At low priority:** the support tools (§4 and §7.4).

## 7. Reusable ideas, beyond left and right (08/10/2026)

Decided with the user and put in the roadmap ([02](02-implications-and-proposal.md) §6).

1. **Is the model looking?** (high priority) The most useful lesson of the research: when it does not know, the model always gives the same answer and does not say so. Three label-free tests:
   - **empty state:** the answer on the real state is almost the same as on "N/A" or on a grey image (we already compute the empty state for the yes-bias correction);
   - **rotations:** the option rotations disagree (MirrorQA calls this "circular" evaluation);
   - **mirror**, for side questions: the answer does not change.

   Together they give a signal in the response and the `uncertain` status.
2. **Derived questions** (high priority). The model perceives simple things well and fails when it has to combine them (the XOR of §2). In the API one declares the simple questions and a rule (not, and, or, XOR, thresholds); the probability of the derived question is computed from the simple ones. Examples:
   - "is it the person's right hand" = "in the left half" XOR "seen from behind";
   - "is it a right hand" = palm XOR "thumb on the right" (fingers up);
   - "is the car driving on the wrong side" = side of the road XOR direction of travel.
3. **Symmetries as free data** (in the first LoRA). The label-free checks become label-free training losses (symmetric KL between the two answers):
   - **mirror:** P(right | photo) = P(left | mirrored photo);
   - **negation:** P(Yes | A) = P(No | not A);
   - **language:** the same answer in Italian and English;
   - **rotations:** the same distribution for any option order.

   They also allow adapting the model to one's own data (for example webcam frames) without labelling them.
4. **Tools that give the model facts** (low priority):
   - MediaPipe, also to send only the skeleton in live mode with no images, for pose memories (keypoint similarity, like MediaPipe's k-NN) and for gestures;
   - OCR for small text;
   - a detector for counting;
   - barcodes and QR codes.

**What to take from Intern-Decision** (to check in the model comparison):
- **A possible shortcut for text.** It is Qwen3.5 already trained with our method, under Apache-2.0. If its claimed 79 on typed-decisions holds on our checks, against our 50 zero-shot, its weights could become Egeria's base for text questions and save a good part of F2.
  - Caveats: the numbers are its own; it has probably seen the typed-decisions train split; it has no numbers on images.
- **Speed.** It puts all the questions in a single pass: a JSON skeleton with one placeholder per question, and the logits read before each. We do one pass per question and per rotation, even if we reuse the state. Whether it is faster, and whether questions interfere, has to be measured.
- **Minor details:** 62 symbols for options instead of 26 (A–Z, a–z, 0–9); calibration shipped together with the checkpoint.

## 8. Documented for later

Research leads that do not go into the roadmap for now:
- **Step-by-step reasoning worsens spatial questions** (13 benchmarks, https://arxiv.org/abs/2604.16060; FlipSet). It confirms the "System 1" choice: an argument for the README.
- **Small heads trained on encoder features** as a light alternative to LoRA for perception tasks. The information is often in the encoder but not read out (VLMs are blind, Theodoridis et al. 2026); it is the same idea as CLM's heads ([01](01-state-of-the-art.md) §5.5).
- **Pointing models** (Molmo2, Moondream): a ready-made route for the `locate` primitive.
- **External evaluation sets:** MirrorQA (licence to check), SpinBench, ViewSpatial-Bench, 3DSRBench, with "circular" evaluation (right only if right for every option order).
