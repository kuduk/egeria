[Italiano](../07-interfaccia.md) · **English**

# Web interface

> Status: **implemented and acceptance-tested** in a headless browser ([scripts/collaudo_ui.py](../../scripts/collaudo_ui.py)). Second version: the first one (separate console, monitor and memory pages, questions written as JSON, task-specific question templates) was hard to use and has been replaced.

## 1. Startup

The model and the web interface are **two separate servers** (§1bis). Start them in two terminals:

```bash
cd /home/kuduk/egeria
uv pip install --python .venv/bin/python -e ".[model,eval,vision,server]"   # first time only

# 1) the model: loads Qwen onto the GPU, port 8100
.venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base

# 2) the web interface: lightweight, no torch, port 8000
.venv/bin/egeria serve --model-url http://127.0.0.1:8100
```

Then, in the browser (from Windows too, if the server runs in WSL): **http://localhost:8000/**.

- **Startup order:** it does not matter. If the model does not respond, the UI flags it at the top ("Model unreachable" (*Modello non raggiungibile*)) and, on the first question, explains how to start it. History and memories remain available.
- **Switching model without closing the UI:** stop and restart only `model-server`, for example with `--model Qwen/Qwen3.5-0.8B-Base`. History and memories do not change: on the first question with memories, their vectors are recomputed for the new model ([06-memory.md](06-memory.md) §5).
- **Stopping them:** `Ctrl+C` in each terminal. If they were started in the background:
  - `pgrep -f '^/home/kuduk/egeria/.venv/bin/python .venv/bin/egeria model-server' | xargs kill`
  - `pgrep -f '^/home/kuduk/egeria/.venv/bin/python .venv/bin/egeria serve' | xargs kill`
- **Where the data goes** (web server only):
  - the question history and the uploaded images in `runs/console/` (`--data`);
  - the memories in `runs/memoria-console/` (`--memory`).
- **Default readout** (since 29/09/2026, [09](09-label-free-checks.md) §9):
  - the options go through every position (`--permutations auto`: all rotations for Yes/No and choice, forward and reverse order for scales);
  - Yes/No questions get the **yes-bias correction**: the question is also asked on the empty state "N/A", only once, and if the model leans towards Yes there, that shift is removed. It is turned off with `--no-yes-correction`.
- **Calibration with temperatures:** it is an option of the model server (`model-server --temperatures`). By default it is not used.
  - The available temperatures (`runs/*/temperature.json`) are fitted on typed-decisions cases. On generic questions and on images they flatten the answers: "Il veicolo è danneggiato? Sì" ("Is the vehicle damaged? Yes") went from 97% to 62%.
  - Use `--temperatures` only with a calibration fitted on your own data. Even then it is **never applied to states with images**, nor to live frames.

> **Before the split** the command was `egeria serve --model ... --temperatures ...`: now `--model`, `--temperatures`, `--permutations`, `--min-confidence`, `--text-only` go to `model-server`.

## 1bis. Architecture: separate model and web server

```
browser ──HTTP──▶ egeria serve (port 8000)             ──HTTP──▶ egeria model-server (port 8100)
                  web UI, history (SQLite),                      Qwen3.5 on the GPU, GPU lock
                  uploaded images, memories,                     stateless: no history,
                  memory vote                                    no memories, no files
```

| | `egeria model-server` | `egeria serve` |
|---|---|---|
| What it does | inference only: decisions and semantic vectors | web UI, history, uploaded images, memories |
| Dependencies | torch, transformers, GPU | fastapi, uvicorn, httpx, numpy: **no torch** |
| State | none | `runs/console/`, `runs/memoria-console/` |
| Default port | 8100 | 8000 |
| Restart it to | change model, calibration, permutations, yes-bias correction, state reuse (`--share-state`, [08](08-state-reuse.md)) | update the web UI |

**How a question flows:**
1. The web server validates the request and, if the state contains images given as a path, reads them and converts them to base64. It only accepts files inside the project or inside the data folder.
2. If memories are needed (`memory.recall` > 0) and the store is not empty, it asks the model for the state vector (`POST /v1/embed`) and searches for the most similar memories.
3. It sends the model (`POST /v1/systemone`) the request **without** `memory`: the model receives only the state and the questions.
4. To the questions with `"batch_calibration": true` it applies the calibration on the history (below).
5. To the model's response it adds the memory vote (`answers[q].memory`) and the recalled memories (`memories`).

### Calibration on the history (optional, per question)

A question with `"batch_calibration": true` (types `noul`, `choice`, `score`) is corrected with the answers given to the **same question** (same type, text and options) by the same model in the earlier cases of the history:
- the model's distribution is divided by the mean of the distributions of the last 200 cases, then renormalized;
- it applies **from 50 cases**: before that the answer stays as it is and reports how many cases are missing;
- every answer reports `batch_calibration`: `{"cases": n, "min_cases": 50, "applied": true|false}`, and once applied also `raw`, the distribution before the correction. The history always uses `raw`, so corrections do not pile up;
- **it must not be used for rare events** (for example "is there a fire?"): it assumes that the true answers vary, and on a question whose answer is almost always "No" it would push towards false alarms. **It is never applied live**, where frames look alike.

On the 2B, on typed-decisions, it is the strongest correction: 0.529 accuracy with a single permutation, 0.545 with all rotations ([09](09-label-free-checks.md) §8). On the Ask page it is the **Calibrate on the history** checkbox of each question.

### Model server API

| Method and path | Body | Response |
|---|---|---|
| `GET /v1/info` | | model, device, GPU, vision component, permutations, yes-bias correction, default threshold, calibration |
| `POST /v1/systemone` | Jev body (`state`, `questions`, `min_confidence`, `image_max_side`) plus `permutations` (`"auto"`, the default, or 1–26, [09](09-label-free-checks.md) §2), `yes_correction` (`false` = without the yes-bias correction) and `calibrated` (`false` = without temperatures) | Jev response with the Egeria extensions |
| `POST /v1/embed` | `{"state": ..., "image_max_side": 448}` | `embedding` (L2-normalized), `embedding_dim`, `model` (the model that computed it), `input_tokens`, `latency_ms` |

Example, directly against the model server:

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Salve, il mio ordine 4471 non è ancora arrivato. Vorrei il rimborso.",
  "questions": {
    "rimborso": {"type": "noul", "instructions": "Il cliente chiede un rimborso?"},
    "ordine":   {"type": "short_answer", "instructions": "Qual è il numero d ordine?"}
  },
  "permutations": 1,
  "yes_correction": false
}'
```

With `"permutations": 1` and `"yes_correction": false` it is the fastest readout, with no corrections.

With an image, from disk to base64:

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d "{
  \"state\": [{\"type\": \"image\", \"base64\": \"$(base64 -w0 examples/immagini/gatto.jpg)\"}],
  \"questions\": {\"gatto\": {\"type\": \"noul\", \"instructions\": \"Nella foto c'è un gatto?\"}}
}"
```

**Model server security:**
- it listens on `127.0.0.1` by default;
- it rejects images given as a local path, because it would be reading arbitrary files: images arrive as base64 or url. `--allow-paths` re-enables them, for local testing;
- it rejects `memory`, because memories live in the web server: the error message explains this;
- `--token SEGRETO` ("secret"; or the `EGERIA_TOKEN` variable) requires `Authorization: Bearer SEGRETO` on every call. The web server passes it with `--model-token` (or the same variable). This is needed if the model runs on another machine, for example a rented GPU reached through an SSH tunnel:

```bash
# on the GPU machine
EGERIA_TOKEN=... .venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base
# locally: tunnel and web interface
ssh -N -L 8100:127.0.0.1:8100 utente@macchina-gpu &
EGERIA_TOKEN=... .venv/bin/egeria serve --model-url http://127.0.0.1:8100
```

**Performance:** the split adds a local HTTP hop, a few milliseconds. Images travel as base64 (about +33% in bytes). For memories the model makes one extra pass over the state (the vector), as it did before, when everything ran in a single process.

## 2. Idea

The web interface does one thing only: **look at something and answer my questions, telling me how confident you are.** The technical format is never shown. Questions are built with fields and menus, and the examples live only in the *Try an example* (*Prova un esempio*) menu.

At the top there are three entries:
- **Ask** (*Chiedi*), the main page;
- **History** (*Storico*), with the number of analyses to review;
- **Memories** (*Ricordi*), with the number of memories.

Next to them: the server status, the settings (gear icon), the light/dark theme toggle and the language: a button that shows "EN" when the UI is in Italian and "IT" when it is in English (§6).

The UI is in **Italian and English**, with the same care for both languages. The labels below are the English ones, with the Italian label in parentheses where it helps.

## 3. Ask (Chiedi)

**On the left, "What to look at"** (*Cosa guardare*), with two modes. The page opens in **live** mode (since 29/09/2026); the switch at the top of the panel changes mode:
- **Live** (*Dal vivo*). You choose *Camera*, *Screen* or *Video from file*; the source is handled by the browser, so it also works with the server in WSL. You set how many frames per second to look at (default *As many as possible*: the next frame is sent as soon as the answer to the previous one arrives, one at a time) and the confidence above which to raise an alert (slider, default 80%). The **Alerts** show image, time and answer, with *Save to memories* and *Ignore*.
- **Text and images** (*Testo e immagini*). A box to type or paste text into. Images are added by dragging them in, pasting them (Ctrl+V) or with *choose a file*, and each one can be removed with its ×.

Below: *Try an example* (a customer message, a photo of an accident, a photo of a receipt, live fire watch) and *Start over*. With the English UI the examples are in English: same images, with the texts and questions translated.

**On the right, "What to ask"** (*Cosa chiedere*): one card per question.
1. Write the question, in English or in Italian.
2. Choose the answer type:

| Answer type | What you set | What you get |
|---|---|---|
| **Yes / No** (*Sì / No*) | nothing | Yes or No, and how confident it is |
| **One of several options** (*Una tra più opzioni*) | the options: type one and press Enter; the × removes it | the chosen option, with the probabilities of all of them |
| **A scale** (*Una scala*) | the levels from lowest to highest (default Low, Medium, High) | the level, with the probabilities of all of them |
| **A short answer (a name, a number, a date…)** (*Una risposta breve*) | nothing (also works with images) | a word or a value read from the input (name, date, amount, code), with the alternatives |
| **An estimate (how many, how much…)** (*Una stima*) | "usually between … and …" and the unit | "about N", the most likely range and the near-certain interval. If the mean falls outside the most likely range, the range is shown in large type and the mean goes in the line below. It estimates quantities that are not written: to read a written number use *A short answer* (the hint under the question says so) |

Until 29/09/2026 these two types were called *A word or a value* and *A number*, and `open` and `number` in the API: the old API names are still accepted, and the history and the questions saved in the browser reopen with the new names.

3. For Yes/No, choice and scale, in text-and-images mode, the **Calibrate on the history** checkbox corrects the answer using those given to the same question in earlier cases (from 50 cases; §1bis). The explanation, with the warning not to use it for rare events, appears when the box is ticked.
4. In live mode, each card also has *Alert me when the answer is…*.

*Add a question* adds a card; the trash icon removes it. *Save these questions* saves them under a name, and *Saved questions…* loads them back: these are the user's own questions, stored in the browser.

**Ask** (or Ctrl+Enter) shows the answer inside each card:
- **the answer**, in large type, for example "Yes · 97% sure" ("Sì · sicuro al 97%" in Italian);
- **the bars** of the probabilities;
- **"Not sure"** (*Non sono sicuro*) if the confidence is below the threshold chosen in the settings;
- **the memories**, if there are similar cases: "In the similar cases I remember (n) you had decided: … — same / different";
- with the calibration on the history, a line saying whether it was applied ("Calibrated on the history (n earlier cases)") or how many cases are missing ("Not calibrated on the history yet: n of 50 cases").

Then:
- **Correct** (*Correggi*) next to each answer: you pick the right one.
- **Save to memories** (*Salva nei ricordi*), at the bottom, with an optional note: the case, with its corrections, is remembered for the next similar questions.

If you change the text or a question after asking, the answers fade out and "press “Ask” to update the answers" appears.

## 4. History (Storico)

All the analyses made with *Ask*. The *To review* (*Da controllare*) filter shows those with uncertain answers or answers that differ from the memories, not yet saved. A click reopens the analysis in the *Ask* page, with text, images, questions and answers, ready to be corrected and saved to memories.

## 5. Memories (Ricordi)

The saved cases, in readable form: text or image, "question → answer", note, date. Questions and options stay as the user wrote them; Yes/No answers are shown in the UI language (§6).
- **Search** (*Cerca*): describe something and get the most similar memories.
- **Search with an image** (*Cerca con un'immagine*): the same, starting from a photo.
- **Forget** (*Dimentica*): deletes a memory, after confirmation.

## 6. Settings (gear icon)

- **Language · Lingua**: English or Italian (the first entry). The change is immediate, without waiting for *Done*.
- **When to flag an answer as uncertain**: *often*, *normal* or *rarely*, i.e. a confidence threshold of 0.7, 0.5 or 0.3.
- **Use memories**: shows what was decided in similar cases.
- **Image quality**: *fast*, *normal*, *detailed* or *maximum*, i.e. a longest side of 336, 448, 672 or 896 px. *Maximum* is for reading small text in documents.

Settings are kept in the browser.

### Language

- **Where to change it:** in the settings (*Language · Lingua*) or with the button at the top next to the theme toggle, which shows the code of the language it switches to ("EN" or "IT") and has a label for screen readers ("Switch to Italian (Italiano)", "Passa all'inglese (English)").
- **Initial language:** the one chosen last time; if there is none, the browser's (`navigator.languages`): Italian if it starts with `it`, English in every other case.
- **Where it is kept:** in the browser, under the `localStorage` key `egeria-lang` (`it` or `en`), like the theme. If the browser does not allow saving it, the browser language is used again. `<html lang>` follows the language too.
- **What happens on a switch:** the whole UI is redrawn (texts, question cards, answers already shown, history, memories, live alerts, notifications) without losing the work in progress: text, images, questions, answers and corrections stay. Focus stays on the button or menu that was used.
- **What changes:** the answer types, the default levels of a scale (Low, Medium, High ↔ Basso, Medio, Alto), Yes/No, "97% sure" ↔ "sicuro al 97%", "Not sure" ↔ *Non sono sicuro*, the examples, relative times ("5 minutes ago"), dates and numbers (decimal point or comma).
- **What does not change:** the texts written by the user (questions, options, levels already visible in a card, notes, names of saved questions). The default levels (Low, Medium, High) take the current language when a card becomes *A scale*; if they are already visible they stay as they are. Question identifiers are always derived from the question text (§9), in either language.
- **Memories:** the server stores the readable labels in Italian ("Sì"/"No"). For every answer it also returns the raw value (`readable[].value`, for example `"true"`), so the English UI shows Yes/No. The text "Immagine dal vivo:" in the state of memories saved in live mode stays in Italian, because it goes into the memory vector; the memory cards show it translated ("Live image:").
- **Server errors:** the details sent by the server stay **in Italian** (for example those of `422` responses). With the English UI only the ones that can be recognized reliably are translated: model server unreachable, token rejected, model server error, case or memory not found, invalid image, missing decisions, memory already existing. The others are shown as they arrive.

## 7. Web server API (for other programs)

| Method and path | Use |
|---|---|
| `POST /v1/systemone` | Jev-compatible API, with the Egeria extensions, **memories** (`memory.recall`, `memory.min_similarity`, `memory.vote`) and the per-question calibration on the history (`"batch_calibration": true`, §1bis) |
| `GET /api/info` | model and model server status (`model_status`: `ok` or `non_raggiungibile`), number of memories and of analyses, model of the memory vectors (`memory_model`) and suspended memories (`memories_suspended`) |
| `GET /api/cases?view=da_rivedere\|tutti` | history |
| `POST /api/cases` | `{"request": <body /v1/systemone>}`: analyzes and saves to history |
| `GET /api/cases/{id}`, `POST /api/cases/{id}/review` | detail; saving to memories with `{"decisions": {...}, "note": ...}` |
| `POST /api/monitor/frame` | a frame (data URL) and the questions, for live mode (without calibration on the history) |
| `GET/POST /api/memory`, `POST /api/memory/search`, `DELETE /api/memory/{id}` | memories; each one has `readable`: question, readable answer (`answer`: the option text, or Sì/No) and raw value (`value`) |
| `POST /api/media` | uploads an image (data URL) and returns `path` and `url` |

Errors: `422` for an invalid request (including when it is the model that rejects it), `503` if the model server is unreachable or rejects the token.

## 8. Acceptance test

`scripts/collaudo_ui.py` walks through the UI in headless Chromium, on a separate instance so as not to touch the real data, and saves screenshots. Headless Chromium has an English browser language: the Italian pass uses a browser with the `it-IT` locale.
1. checks that the page opens in live mode, switches to *Text and images*, writes two questions by hand, with options added from the keyboard, and ticks *Calibrate on the history* on the second one;
2. asks (the second answer says how many cases are missing for the calibration, the first does not), corrects an answer and saves to memories;
3. loads the photo example and asks;
4. switches to live mode;
5. opens history and memories;
6. switches theme;
7. with an `en-US` browser, checks that the UI starts in English;
8. English pass: starts in Italian with text and a question already written, switches to English with the top button (the work stays, and so does the focus), checks the main labels, asks, switches to Italian and back to English (the answers stay, translated), corrects and saves, changes language from the settings, opens the dialog to save the questions, loads an English example, switches to live mode, opens history and memories (Sì/No of the memories shown as Yes/No), reloads the page (the language stays).

On every page it looks for UI words left in the other language ("Chiedi", "Storico", "Ricordi", "Non sono sicuro", "Salva nei ricordi", "Impostazioni"… and, in the Italian pass, "Ask", "History"…), excluding the user's texts, and for `{name}` placeholders left unreplaced. It also checks that nothing sticks out of the page at 1440 and 390 px width. It exits with 1 if it finds problems or console errors.

Result of the first version: no errors in the browser console. Answers on the hand-written case: "Il cliente chiede un rimborso? Sì, 90%" ("Is the customer asking for a refund? Yes, 90%"); "Qual è il problema? Consegna in ritardo, 99%" ("What is the problem? Late delivery, 99%").

After the split (September 2026) the acceptance test was repeated with `model-server` (Qwen3.5-0.8B-Base, port 8100) and a test web instance (port 8001): no console errors. The model being down was also tested (at the time the UI was in Italian only): "Modello non raggiungibile" ("Model unreachable") appears at the top and *Chiedi* (*Ask*) shows "server del modello non raggiungibile su http://127.0.0.1:8100 (avvialo con: egeria model-server)" ("model server unreachable at http://127.0.0.1:8100 (start it with: egeria model-server)").

Manual tests along the full path:
- text: "4471" read as the order number at 93%;
- photo given as a path: reaches the model as base64; the saved memory is recalled with similarity 1.0 and votes on the answer;
- the same photo sent as a path directly to the model server: rejected with `422`.

The automated tests ([tests/test_server.py](../../tests/test_server.py)) make the real HTTP client talk to the model server running in-process. They cover the token, rejected paths, calibration never applied to images, memories and vote computed by the web server, base64 images, and an unreachable model (`503`).

Test of the two languages (29/09/2026), with the Qwen3.5-0.8B-Base model on CPU (port 8110) and a test web instance (port 8011): no console errors, no words left in the other language, nothing outside the page at 1440 and 390 px. Also tested by hand: a card error and notifications that get re-translated, live alerts with a fake camera (`--use-fake-device-for-media-stream`) saved to memories, the model being down ("model server unreachable at …" in English, the original Italian message in Italian).

Acceptance test of the calibration on the history (29/09/2026), with Qwen3.5-2B-Base on GPU (port 8101) and the test web instance (port 8001): no console errors, no words left in the other language, nothing off the page. Fixed after looking at the screenshots: "1 casi su 50" (now singular and plural) and the long explanation on every card (now it appears only when the box is ticked).

**Bugs found and fixed through the acceptance test:**
- empty probability bars (inline element without `display: block`);
- duplicate options when pressing Enter (the blur of the redrawn field added them again);
- answers flattened by a calibration fitted on a different kind of data;
- at 390 px width the page stuck out sideways (469 px wide): top bar too full, menus with a fixed minimum width. Now below 480 px the top bar keeps only the emblem, menus shrink, and in the history the badges go below the text;
- in live mode each new alert erased the previous ones, and the *Saved* button of an alert became clickable again (found while rewriting the alerts for the two languages).

## 9. Design choices

- **No build step:** static HTML, CSS and JS served by FastAPI. The DOM is built only with `textContent` (no XSS).
- **Two languages without libraries:** [i18n.js](../../src/egeria/web/i18n.js) holds the two dictionaries and `t(key, values)`, with `{name}` interpolation and the plural chosen with `Intl.PluralRules` (`{one, other}`). In `index.html` static texts carry the `data-i18n` (text), `data-i18n-placeholder`, `data-i18n-aria-label` and `data-i18n-title` attributes; all dynamic texts in `app.js` go through `t()`. Translations too reach the DOM only through `textContent`. To add a text: one key in both dictionaries (`it` is the fallback).
- **Video from file, in live mode:** only MP4, WebM, Ogg, MOV and MKV are accepted. The file becomes a `blob:` URL whose type comes from a fixed table (not from the file), and the previous URL is released when the video changes. This fixed, on 29/09/2026, the CodeQL alert `js/xss-through-dom`, verified locally with CodeQL 2.27.1: zero alerts with the code scanning suite and with `security-extended`.
- **Style:** dark or light theme, Fira Sans/Fira Code, SVG icons.
  - **Logo:** [img/logo.png](../../src/egeria/web/img/logo.png). The top bar shows the emblem on a light rounded tile, identical in both themes, next to the name in letter-spaced capitals like the lettering in the logo. The light tile is needed because the face is drawn with the white of the background: on a dark background the emblem would turn into a negative.
  - **Derived images:** emblem, favicon, iPhone icon and the two 1280×640 social previews for GitHub (centered logo, or emblem and lettering side by side) are generated with `.venv/bin/python scripts/genera_icone.py`, to be rerun whenever the logo changes.
  - Status colors are always paired with text and an icon: amber = not sure, purple = differs from memories, red = alert.
  - Visible focus, and `prefers-reduced-motion` is respected.
- **Model separate from the UI** (§1bis): a single model loaded in the `model-server`, with a lock on the GPU; the web server does not import torch. Both listen only on `127.0.0.1` by default, and `/files` only serves images inside the project.
- **"Not sure" (*Non sono sicuro*) on text too:** every answer with a threshold reports `status`. Before the split, on text without calibration `status` was missing and the warning only appeared on images; now it is always there.
- **Question identifiers** derived from the question text. The same question asked again gets the same identifier, so the memory vote recognizes it.

## 10. Usage tips from the tests

- **Statements in positive form.** For Yes/No questions write "the parcel has arrived", not "the parcel has not arrived": the model is unreliable on negations ([09-label-free-checks.md](09-label-free-checks.md) §3).
- **A written number is read with *A short answer*,** not with *An estimate*, which estimates by ranges. On a receipt total (14.21 €) *A short answer* reads 14,21 at 97–99%; *An estimate* gives "about 23–28" and at 448 px gets even the range wrong ([05](05-reading-primitives.md) §5).
- **Ask it to read, not to reason.** The model reads values and text well, but gets comparisons and arithmetic wrong: "ha più di 21 anni?" ("is the person over 21?") fails, "qual è la data di nascita?" ("what is the date of birth?") works. Ask for the value with *A short answer* (*Una risposta breve*) and do the comparison by hand or in code ([04-image-states.md](04-image-states.md) §7).
- **Documents and small text:** *Maximum* image quality in the settings.
- **Simple questions:** one thing per question. See the insurance claim example in [04-image-states.md](04-image-states.md) §6.
- **Poses and similar-looking figures:** prefer *One of several options* with the poses **described in words** ("standing on one leg, with the foot resting against the inside of the other") over a Yes/No with the pose name. With "Is the person doing the tree pose?" the 2B says Yes even to two warriors out of three; with the described options the poses taken for the tree drop to 14%. One-leg poses remain confused with each other ([09](09-label-free-checks.md) §10.3).
