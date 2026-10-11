[Italiano](../10-destra-sinistra.md) · **English**

# Left and right in images

> **Status:** investigation of 05/10/2026. On 29/09 left/right had been taken out of the checks and of the first LoRA because "something there doesn't work at all" ([02](02-implications-and-proposal.md) §0). Script: [scripts/indagine_destra_sinistra.py](../../scripts/indagine_destra_sinistra.py). Reports: `runs/destra_sinistra/<model>.json`.

**In short:**
- **No left/right swap.** On simple figures the 2B is right 100% of the time.
- **The flaw is with body parts.** The 2B-**Base** does not look: asked about the raised leg it always gives the same answer. The 2B-**Instruct** finds it (88.5%).
- **Neither knows the *person's* own right.** "Is the raised hand the right one?" always gets No (the user's case, §3.1). Asking "in which half of the image is the raised hand?" works: 8/8 with both models. Even on close-ups the model does not use the hand's chirality (§3.2).
- **There was a bug in the image path** (EXIF orientation), now fixed. The webcam in the browser is clean.

## 1. The image path

**EXIF orientation: a bug, fixed.**
- **The flaw:** a phone often saves a photo with rotated pixels and an EXIF tag that says how to turn it. `load_images` did not apply it. A photo taken with the phone sideways reached the model rotated by 90° (the circle the user sees on the left ended up at the bottom). With tags 2, 4, 5 and 7, used by some front cameras, it arrived **mirrored**, that is with left and right swapped.
- **Who was affected:** images read from disk or sent by a program (command line, API with a path, url or base64).
- **Who was not:** images uploaded from the web UI (the browser applies the orientation when it resizes them) and webcam frames.
- **The fix:** `ImageOps.exif_transpose` in [images.py](../../src/egeria/images.py), with a test. No image in `examples/immagini` had an EXIF tag other than "normal".

**Webcam: clean.** I used Chromium with a fake webcam that streams a purpose-made video (`--use-file-for-fake-video-capture`): a red circle on the left and a blue square on the right. The frame the web UI sends to the server was intercepted.

| | Red position (x) | Blue position (x) |
|---|---|---|
| Webcam video | 0.22 | 0.78 |
| Preview in the page | 0.22 | 0.78 |
| Frame sent to the model | 0.22 | 0.78 |

- **Consistency:** the preview is not mirrored (no CSS `transform`), and what you see is what the model sees.
- **Webcam driver:** it may send an already mirrored image, which cannot be checked from here.
- **How to tell:** raise your right hand in front of the webcam. In a normal image it appears **on the left** of the preview, as in a photo; if it appears on the right, as in a mirror, the image is mirrored.

## 2. The model on simple figures: perfect

48 synthetic scenes, half with two figures side by side and half with one figure above the other, each also mirrored. 2B-Base, default readout, in Italian and English:

| Question | Accuracy | Consistent with the mirror |
|---|---|---|
| "Is the red circle to the left or to the right of the blue square?" | 1.000 | 100% |
| "In which half of the image is the red circle?" | 1.000 | 100% |
| Yes/No "Is the red circle to the left of the blue square?" | 1.000 | 100% |
| Control: "above or below?" | 1.000 | 100% |

The 0.8B-Base is much weaker: 0.67–0.75, only partly consistent with the mirror.

## 3. The model on body parts

**The data.** 13 photos of poses with one leg raised ([examples/immagini/pose](../../examples/immagini/pose)), plus their mirrored versions (26 images):
- 8 tree poses seen from the front;
- 1 seen from behind;
- 4 seen from the side.

**The questions**, in the two frames:
- **of the image:** "In the image, is the raised leg on the left side or on the right side?";
- **of the person:** "Has the person raised their right leg or their left leg?". For someone seen from the front, their right is on the left of the image.

| Model | Image frame: accuracy (consistency with the mirror) | Person frame: accuracy | Person seen from the front: answers with the image side |
|---|---|---|---|
| 0.8B-Base | it 0.50 (0%), en 0.81 (61%) | 0.50 | 50% |
| **2B-Base** (default) | it **0.50 (0%: always "left")**, en 0.65 (31%) | 0.50 (always "right") | 50% |
| **2B-Instruct** | **it 0.885 (92%), en 0.885 (77%)** | it 0.44, en 0.33 | 56–62% |

**Variants tried on the 2B-Base** (image frame):

| Variant | Accuracy | Consistency with the mirror |
|---|---|---|
| Image at 800 px instead of 448 | 0.50 | 0% |
| "Does the knee of the bent leg stick out towards the left or the right side?" | 0.50 | 0% |
| Two Yes/No: "Is there a bent leg in the left / right half?" | 0.50 | 0% |
| **"In which half of the image is the foot lifted off the ground?"** | **0.77** | 54% |
| **Image cut in two**, "is there a bent leg?" on each half | **0.73** | **92%** |

**What emerges:**
1. **"Something that doesn't work at all" is the 2B-Base on body parts.** It does not swap left and right: it does not look, and gives a fixed answer. More resolution does not help.
2. **Asking where a specific object is works better** (the foot: 0.77) than asking which side a body part is on. Cutting the image forces the model to look (92% consistency).
3. **The 2B-Instruct locates body parts** (0.885). On this it is much stronger than the Base.
4. **The person's own right is hard for both:** about chance, and the Instruct tends to answer with the image side. It needs training, or converting in code.

### 3.1 Raised hands: the user's case

The case reported by the user: in front of the webcam (image not mirrored) they ask whether the raised hand is the right one, and the model gets it wrong both with the right and with the left. Reproduced with 4 photos of people seen from the front raising one hand ([examples/immagini/mani](../../examples/immagini/mani)), plus the mirrored versions (8 images):

| Question | 2B-Base | 2B-Instruct |
|---|---|---|
| Yes/No **"Is the raised hand the right one?"** (and in Italian) | 4/8: **always No** | 4/8: **always No** |
| "Has the person raised their right hand or their left hand?" | 4/8: always "left" | 4/8: always "left" |
| "In the image, is the raised hand on the left side or on the right side?" | 7/8 | **8/8** |
| **"In which half of the image is the raised hand?"** | **8/8** | **8/8** |

- **"Is it the right one?" always gets No**, whichever hand is raised. If the question was "is it the right one?" with the right hand and "is it the left one?" with the left, the answer was No both times: wrong in both cases, as observed.
- **Asking which half of the image the hand is in works with both models.** With a front-facing, non-mirrored webcam, the **left half** of the image is the person's **right hand**.

### 3.2 Chirality: the hand alone should be enough

**The user's objection:** hands are chiral, so a hand on its own tells whether it is right or left, whatever the frame. With the fingers up, a right palm has the thumb on the right and a right back of the hand has it on the left. **The mirror puts this to the test:** mirrored, a right hand becomes a left hand. If the model used chirality, the answer would flip.

**Data.** 7 close-ups ([examples/immagini/mani/primi_piani](../../examples/immagini/mani/primi_piani)), with palms and backs, fingers up and down, plus the mirrored versions.

| Question (14 images) | 2B-Base | 2B-Instruct |
|---|---|---|
| **"Is this a right hand or a left hand?"** | 0.50: always "right" in Italian (almost always "left" in English) | 0.50: same |
| …answer flipped by the mirror | **0 out of 7** | **0 out of 7** |
| "Do you see the palm or the back of the hand?" | 0.71 | 0.79 |
| "Is the thumb on the left or the right side of the hand?" | 0.50 (always "left") | 0.67 at 448 px, 0.83 at 800 px |
| "Do the fingers point up or down?" | 0.83 | 0.83 |
| **Hand recomposed in code** from the three cues | 0.58 | **0.75** at 448 px (0.58 at 800 px) |

**What emerges:**
- **The model does not use chirality.** To the direct question it gives a fixed answer even on close-ups, at any resolution, and the mirror never changes it.
- **The Instruct partly sees the ingredients.** Palm/back and finger direction are fine; the thumb side is fragile and changes with resolution.
- **Recomposing in code** ("palm with fingers up and thumb on the right = right hand") reaches 0.75: better than chance, not enough to rely on.
- **It is an ideal training target.** Every photo of a hand, mirrored, gives for free the example with the opposite answer. Labels can even be skipped, by requiring the model's P(right | photo) to equal P(left | mirrored photo).

## 4. 2B-Base or 2B-Instruct?

With today's defaults (all rotations, yes-bias correction), on the typed-decisions test split:

| | Total | Yes/No | Choice | Scale |
|---|---|---|---|---|
| 2B-Base | **0.496** | 0.527 | **0.603** | 0.393 |
| 2B-Instruct | 0.475 | **0.647** | 0.432 | 0.380 |
| Difference (Instruct − Base) | −2.1 [−3.9, −0.2] | **+12.0** [+7.2, +16.8] | **−17.2** [−20.8, −13.6] | −1.3 (n.s.) |

**On images** (the check of §9–10 in [09](09-label-free-checks.md)):
- rare events: the Instruct matches the Base (0.987 vs 1.000);
- poses described in words: the Instruct takes the other one-leg poses for the tree 50% of the time (the Base 100%).

Neither wins everywhere. On images and Yes/No questions the Instruct is stronger, on text with choices the Base. Two 2B models in bf16 do not fit together in the 8 GB of the local GPU.

## 5. What to do

1. **For left and right of objects and figures:** it already works, with the Base too.
2. **For body parts:**
   - ask **where** the part is ("in which half of the image is the raised hand?", 8/8; for the raised foot 0.77) instead of "which side";
   - or use the 2B-Instruct as the model server for image tasks.
3. **For the person's own right:** ask in the image frame and convert in code. With a front-facing, non-mirrored webcam, the person's right is the left side of the image.
4. **For training (F1):** the mirror gives **free labels**. A mirrored image has the answer reversed, and in the person's frame the leg changes too. Mirrored poses are a ready-made augmentation to teach both frames.
5. **Before training:** the research on existing models and tools is in [11-models-and-tools.md](11-models-and-tools.md).
6. **To decide:**
   - which model to use for images (Base or Instruct);
   - whether to put left/right back among the label-free checks, with consistency under mirroring as the measure;
   - whether the first LoRA should also teach the person's frame.

```bash
# full investigation (GPU, ~20 s after loading)
.venv/bin/python scripts/indagine_destra_sinistra.py Qwen/Qwen3.5-2B-Base
.venv/bin/python scripts/indagine_destra_sinistra.py Qwen/Qwen3.5-2B
```
