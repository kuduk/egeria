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

- **Startup order:** it does not matter. If the model does not respond, the UI flags it at the top ("Modello non raggiungibile", "Model unreachable") and, on the first question, explains how to start it. History and memories remain available.
- **Switching model without closing the UI:** stop and restart only `model-server`, for example with `--model Qwen/Qwen3.5-0.8B-Base`. History and memories do not change.
- **Stopping them:** `Ctrl+C` in each terminal. If they were started in the background:
  - `pgrep -f '^/home/kuduk/egeria/.venv/bin/python .venv/bin/egeria model-server' | xargs kill`
  - `pgrep -f '^/home/kuduk/egeria/.venv/bin/python .venv/bin/egeria serve' | xargs kill`
- **Where the data goes** (web server only):
  - the question history and the uploaded images in `runs/console/` (`--data`);
  - the memories in `runs/memoria-console/` (`--memory`).
- **Calibration:** it is an option of the model server (`model-server --temperatures`). By default it is not used, and the model's probabilities are shown as they are (with 2 permutations of the options).
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
| Restart it to | change model, calibration, permutations, state reuse (`--share-state`, [08](08-state-reuse.md)) | update the web UI |

**How a question flows:**
1. The web server validates the request and, if the state contains images given as a path, reads them and converts them to base64. It only accepts files inside the project or inside the data folder.
2. If memories are needed (`memory.recall` > 0) and the store is not empty, it asks the model for the state vector (`POST /v1/embed`) and searches for the most similar memories.
3. It sends the model (`POST /v1/systemone`) the request **without** `memory`: the model receives only the state and the questions.
4. To the model's response it adds the memory vote (`answers[q].memory`) and the recalled memories (`memories`).

### Model server API

| Method and path | Body | Response |
|---|---|---|
| `GET /v1/info` | | model, device, GPU, vision component, permutations, default threshold, calibration |
| `POST /v1/systemone` | Jev body (`state`, `questions`, `min_confidence`, `image_max_side`) plus `permutations` (1–8) and `calibrated` (`false` = raw probabilities) | Jev response with the Egeria extensions |
| `POST /v1/embed` | `{"state": ..., "image_max_side": 448}` | `embedding` (L2-normalized), `embedding_dim`, `input_tokens`, `latency_ms` |

Example, directly against the model server:

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Salve, il mio ordine 4471 non è ancora arrivato. Vorrei il rimborso.",
  "questions": {
    "rimborso": {"type": "noul", "instructions": "Il cliente chiede un rimborso?"},
    "ordine":   {"type": "open", "instructions": "Qual è il numero d ordine?"}
  },
  "permutations": 1
}'
```

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

The web interface does one thing only: **look at something and answer my questions, telling me how confident you are.** The technical format is never shown. Questions are built with fields and menus, and the examples live only in the *Prova un esempio* ("Try an example") menu.

At the top there are three entries:
- **Chiedi** ("Ask"), the main page;
- **Storico** ("History"), with the number of analyses to review;
- **Ricordi** ("Memories"), with the number of memories.

Next to them: the server status, the settings (gear icon) and the light/dark theme toggle.

## 3. Chiedi (Ask)

**On the left, "Cosa guardare"** ("What to look at"), with two modes:
- **Testo e immagini** ("Text and images"). A box to type or paste text into. Images are added by dragging them in, pasting them (Ctrl+V) or with *scegli un file* ("choose a file"), and each one can be removed with its ×.
- **Dal vivo** ("Live"). You choose *Telecamera* ("Camera"), *Schermo* ("Screen") or *Video da file* ("Video from file"); the source is handled by the browser, so it also works with the server in WSL. You set how many frames per second to look at and the confidence above which to raise an alert (slider, default 80%). The **Avvisi** ("Alerts") show image, time and answer, with *Salva nei ricordi* ("Save to memories") and *Ignora* ("Ignore").

Below: *Prova un esempio* (a customer message, a photo of an accident, a photo of a receipt, live fire monitoring) and *Ricomincia da zero* ("Start over").

**On the right, "Cosa chiedere"** ("What to ask"): one card per question.
1. Write the question in Italian.
2. Choose the answer type:

| Answer type | What you set | What you get |
|---|---|---|
| **Sì / No** ("Yes / No") | nothing | Sì or No, and how confident it is |
| **Una tra più opzioni** ("One of several options") | the options: type one and press Enter; the × removes it | the chosen option, with the probabilities of all of them |
| **Una scala** ("A scale") | the levels from lowest to highest (default Basso, Medio, Alto: "Low, Medium, High") | the level, with the probabilities of all of them |
| **Un numero** ("A number") | "di solito tra … e …" ("usually between … and …") and the unit | "circa N" ("about N"), the most likely range and the near-certain interval |
| **Una parola o un valore** ("A word or a value") | nothing (also works with images) | a word or a value read from the input (name, date, amount, code), with the alternatives |

3. In live mode, each card also has *Avvisami quando la risposta è…* ("Alert me when the answer is…").

*Aggiungi una domanda* ("Add a question") adds a card; the trash icon removes it. *Salva queste domande* ("Save these questions") saves them under a name, and *Domande salvate…* ("Saved questions…") loads them back: these are the user's own questions, stored in the browser.

**Chiedi** (or Ctrl+Enter) shows the answer inside each card:
- **the answer**, in large type, for example "Sì · sicuro al 97%" ("Yes · 97% sure");
- **the bars** of the probabilities;
- **«Non sono sicuro»** ("I'm not sure") if the confidence is below the threshold chosen in the settings;
- **the memories**, if there are similar cases: "Nei casi simili che ricordo (n) avevi deciso: … — uguale / diverso" ("In the similar cases I remember (n) you had decided: … — same / different").

Then:
- **Correggi** ("Correct") next to each answer: you pick the right one.
- **Salva nei ricordi**, at the bottom, with an optional note: the case, with its corrections, is remembered for the next similar questions.

If you change the text or a question after asking, the answers fade out and "premi «Chiedi» per aggiornare" ("press «Chiedi» to update") appears.

## 4. Storico (History)

All the analyses made with *Chiedi*. The *Da controllare* ("To review") filter shows those with uncertain answers or answers that differ from the memories, not yet saved. A click reopens the analysis in the *Chiedi* page, with text, images, questions and answers, ready to be corrected and saved to memories.

## 5. Ricordi (Memories)

The saved cases, in Italian: text or image, "question → answer", note, date.
- **Cerca** ("Search"): describe something and get the most similar memories.
- **Cerca con un'immagine** ("Search with an image"): the same, starting from a photo.
- **Dimentica** ("Forget"): deletes a memory, after confirmation.

## 6. Settings (gear icon)

- **Quando segnalare una risposta come incerta** ("When to flag an answer as uncertain"): *spesso*, *normale* or *raramente* ("often", "normal", "rarely"), i.e. a confidence threshold of 0.7, 0.5 or 0.3.
- **Usa i ricordi** ("Use memories"): shows what was decided in similar cases.
- **Qualità delle immagini** ("Image quality"): *veloce*, *normale*, *dettagliata* or *massima* ("fast", "normal", "detailed", "maximum"), i.e. a longest side of 336, 448, 672 or 896 px. *Massima* is for reading small text in documents.

Settings are kept in the browser.

## 7. Web server API (for other programs)

| Method and path | Use |
|---|---|
| `POST /v1/systemone` | Jev-compatible API, with the Egeria extensions **and memories** (`memory.recall`, `memory.min_similarity`, `memory.vote`) |
| `GET /api/info` | model and model server status (`model_status`: `ok` or `non_raggiungibile`), number of memories and of analyses |
| `GET /api/cases?view=da_rivedere\|tutti` | history |
| `POST /api/cases` | `{"request": <body /v1/systemone>}`: analyzes and saves to history |
| `GET /api/cases/{id}`, `POST /api/cases/{id}/review` | detail; saving to memories with `{"decisions": {...}, "note": ...}` |
| `POST /api/monitor/frame` | a frame (data URL) and the questions, for live mode |
| `GET/POST /api/memory`, `POST /api/memory/search`, `DELETE /api/memory/{id}` | memories |
| `POST /api/media` | uploads an image (data URL) and returns `path` and `url` |

Errors: `422` for an invalid request (including when it is the model that rejects it), `503` if the model server is unreachable or rejects the token.

## 8. Acceptance test

`scripts/collaudo_ui.py` walks through the UI in headless Chromium, on a separate instance so as not to touch the real data, and saves screenshots:
1. writes two questions by hand, with options added from the keyboard;
2. asks, corrects an answer and saves to memories;
3. loads the photo example and asks;
4. switches to live mode;
5. opens history and memories;
6. switches theme.

Latest result: no errors in the browser console. Answers on the hand-written case: "Il cliente chiede un rimborso? Sì, 90%" ("Is the customer asking for a refund? Yes, 90%"); "Qual è il problema? Consegna in ritardo, 99%" ("What is the problem? Late delivery, 99%").

After the split (September 2026) the acceptance test was repeated with `model-server` (Qwen3.5-0.8B-Base, port 8100) and a test web instance (port 8001): no console errors. The model being down was also tested: "Modello non raggiungibile" appears at the top and *Chiedi* shows "server del modello non raggiungibile su http://127.0.0.1:8100 (avvialo con: egeria model-server)" ("model server unreachable at http://127.0.0.1:8100 (start it with: egeria model-server)").

Manual tests along the full path:
- text: "4471" read as the order number at 93%;
- photo given as a path: reaches the model as base64; the saved memory is recalled with similarity 1.0 and votes on the answer;
- the same photo sent as a path directly to the model server: rejected with `422`.

The automated tests ([tests/test_server.py](../../tests/test_server.py)) make the real HTTP client talk to the model server running in-process. They cover the token, rejected paths, calibration never applied to images, memories and vote computed by the web server, base64 images, and an unreachable model (`503`).

**Bugs found and fixed through the acceptance test:**
- empty probability bars (inline element without `display: block`);
- duplicate options when pressing Enter (the blur of the redrawn field added them again);
- answers flattened by a calibration fitted on a different kind of data.

## 9. Design choices

- **No build step:** static HTML, CSS and JS served by FastAPI. The DOM is built only with `textContent` (no XSS).
- **Style:** dark or light theme, Fira Sans/Fira Code, SVG icons.
  - **Logo:** [img/logo.png](../../src/egeria/web/img/logo.png). The top bar shows the emblem on a light rounded tile, identical in both themes, next to the name in letter-spaced capitals like the lettering in the logo. The light tile is needed because the face is drawn with the white of the background: on a dark background the emblem would turn into a negative.
  - **Derived images:** emblem, favicon, iPhone icon and the two 1280×640 social previews for GitHub (centered logo, or emblem and lettering side by side) are generated with `.venv/bin/python scripts/genera_icone.py`, to be rerun whenever the logo changes.
  - Status colors are always paired with text and an icon: amber = not sure, purple = differs from memories, red = alert.
  - Visible focus, and `prefers-reduced-motion` is respected.
- **Model separate from the UI** (§1bis): a single model loaded in the `model-server`, with a lock on the GPU; the web server does not import torch. Both listen only on `127.0.0.1` by default, and `/files` only serves images inside the project.
- **«Non sono sicuro» on text too:** every answer with a threshold reports `status`. Before the split, on text without calibration `status` was missing and the warning only appeared on images; now it is always there.
- **Question identifiers** derived from the question text. The same question asked again gets the same identifier, so the memory vote recognizes it.

## 10. Usage tips from the tests

- **Ask it to read, not to reason.** The model reads values and text well, but gets comparisons and arithmetic wrong: "ha più di 21 anni?" ("is the person over 21?") fails, "qual è la data di nascita?" ("what is the date of birth?") works. Ask for the value with *Una parola o un valore* and do the comparison by hand or in code ([04-image-states.md](04-image-states.md) §7).
- **Documents and small text:** *Massima* quality in the settings.
- **Simple questions:** one thing per question. See the insurance claim example in [04-image-states.md](04-image-states.md) §6.
