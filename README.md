**English** · [Italiano](README.it.md)

# Egeria

<p align="center"><img src="src/egeria/web/img/logo.png" alt="Egeria" width="220"></p>

<p align="center">[![Test](https://github.com/kuduk/egeria/actions/workflows/tests.yml/badge.svg)](https://github.com/kuduk/egeria/actions/workflows/tests.yml)</p>

> Named after the nymph Egeria, advisor to Numa Pompilius, king of Rome: a small, quick advisor that gives its opinion, and says how sure it is, to whoever actually decides (a large LLM, an agent, a person). Until 2026-09-28 the project was called **semLMM**; the package, command and folder have been renamed to `egeria`.

A project to build a **"System 1" semantic decision model** on the **Qwen3.5** family, along the lines of Jev (TypeSafe), Laya (Convai) and SemIf.

A model of this kind:
- receives a **state**, as text or JSON (and, here, images too);
- receives **typed questions** (`noul` yes/no, `choice`, `score`);
- returns, in a single forward pass, **calibrated probabilities** over the declared options, without generating text.

## Project status

| Phase | Status |
|---|---|
| State-of-the-art research | Done (September 2026) |
| **F0** – zero-shot baseline (letter-logit readout, SemIf style) + per-assertion dynamic depth diagnostics | Done: harness ready; zero-shot, the small models do not beat the prior; dynamic depth with 27–36% potential savings |
| States with images (multimodal) | Integrated in the API (`decide`) |
| Zero-shot primitives `estimate` and `short_answer` (`number` and `open` until 2026-09-29, still accepted), plus the state vector (`/v1/embed`) | Implemented and verified (`multi` and `surprise` removed: they failed the test) |
| External memory (similar memories + memory vote in the output, `memory.recall`), images included | Implemented: memory vote 0.56 vs 0.47 for the zero-shot model; 5/6 with photos. `known` tested and not kept |
| Generic web interface (`egeria serve`): ask about text, images or a live camera; history; memories | Implemented and tested in the browser |
| Model separated from the web server (`egeria model-server`, stateless API, optional token) | Implemented and tested |
| Simplification: removed dynamic depth, `rank`, the `recall` question, `embed` inside the request, `inject` | Completed on 2026-09-29 |
| **F0.5** – measure and speed up without training: state reuse, label-free checks, calibration, evaluation sets | In progress: state reuse done (up to 80% less time with images); label-free checks done; label-free calibration done (all rotations and yes-bias correction by default: +2.9 points on the 2B, +8.2 on `choice`; optional calibration on the history) |
| **F1** – first LoRA on an image task, locally | After F0.5 (roadmap in [02](documentazione/en/02-implications-and-proposal.md) §6) |

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[model,eval,dev]"   # + ",vision" for states with images
.venv/bin/egeria info
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base examples/ticket_it.json
.venv/bin/pytest -q
```

The tests also run on GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) on every pull request to `main` and `develop` and after every merge into them, without torch or a model; the integration tests with a real model run locally with `EGERIA_TEST_MODEL=Qwen/Qwen3.5-0.8B-Base .venv/bin/pytest -q -m model`.

The request has the same format as a Jev `POST /v1/systemone` (`state` + `questions`), and so does the response. Examples in [examples/](examples/).

**Default readout.** The options go through every position (`--permutations auto`) and Yes/No questions get the yes-bias correction (`--no-yes-correction` removes it): [documentazione/en/09-label-free-checks.md](documentazione/en/09-label-free-checks.md) §9.

**Per-assertion confidence threshold.** Optional extension: `min_confidence` per question or globally in the request; the per-question value overrides the global one. Below the threshold the answer has `status: uncertain`, otherwise `decided`. Until 2026-09-29 the same threshold also made the model exit early (dynamic depth): this was removed because, zero-shot, the real gain was only 0.5–4% ([documentazione/en/03-baseline-f0.md](documentazione/en/03-baseline-f0.md) §6).

For the fast kernels of the Gated DeltaNet layers, 4-bit quantization and the typed-decisions evaluation, see [documentazione/en/03-baseline-f0.md](documentazione/en/03-baseline-f0.md).

## Web interface

The model and the interface are two separate servers, started in two terminals:

```bash
uv pip install --python .venv/bin/python -e ".[model,eval,vision,server]"
.venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base    # the model on the GPU, port 8100
.venv/bin/egeria serve --model-url http://127.0.0.1:8100      # the interface, port 8000 (no torch)
```

The model server is stateless: it answers `/v1/systemone` and `/v1/embed`. History, uploaded images and memories live in the web server, so you can switch models, or run the model on another machine, without touching the data.

Then open **http://localhost:8000/** in the browser (from Windows too, if the server runs in WSL). The interface is in English and Italian (language button in the top bar, or in the settings):
- on the left, the page opens in live mode (camera, screen or video file); switch to *Text and images* to type a text or add images;
- on the right, write the questions and pick the answer type: yes/no, one of several options, a scale, a number, a word or value;
- press *Ask*, correct the answers if needed, then *Save to memories*.

Details in [documentazione/en/07-web-interface.md](documentazione/en/07-web-interface.md).

## Documentation

The documentation is in English in [documentazione/en/](documentazione/en/) and in Italian in [documentazione/](documentazione/).

| Document | Contents |
|---|---|
| [01-state-of-the-art.md](documentazione/en/01-state-of-the-art.md) | Survey: Jev, Laya, SemIf, CLM, open replications on Qwen, benchmarks, Qwen 3.8, readout techniques, calibration, JEPA/LCM paradigms |
| [02-implications-and-proposal.md](documentazione/en/02-implications-and-proposal.md) | Decisions taken, proposed architecture, training, evaluation, current limitations, roadmap, open decisions |
| [03-baseline-f0.md](documentazione/en/03-baseline-f0.md) | F0 baseline: code, installation (fast kernels included), usage, verified pitfalls, results |
| [04-dynamic-depth.md](documentazione/en/04-dynamic-depth.md) | Per-assertion dynamic depth: proposal, plan and measurements (abandoned on 2026-09-29) |
| [04-image-states.md](documentazione/en/04-image-states.md) | States with images: JSON format, examples, results (26/26 on the 2B models), towards real time |
| [05-reading-primitives.md](documentazione/en/05-reading-primitives.md) | Primitives beyond noul/choice/score: verification, **usage examples** (number, open, state vector) and comparison with Jev and Laya |
| [06-memory.md](documentazione/en/06-memory.md) | External memory: memories recalled by similarity, memory vote, managing the store |
| [07-web-interface.md](documentazione/en/07-web-interface.md) | Web interface: startup, architecture (separate model and web servers, model API, token), the Ask page, live mode, history, memories, settings, API, acceptance test |
| [08-state-reuse.md](documentazione/en/08-state-reuse.md) | State reuse: prefix computed once for all questions, verification, measurements, the `auto` mode rule |
| [09-label-free-checks.md](documentazione/en/09-label-free-checks.md) | Label-free checks: option position, yes-bias with negations, Italian/English consistency; low-cost corrections and the default readout (all rotations, yes-bias correction, calibration on the history) |
| [sources.md](documentazione/en/sources.md) | Sources by area |

## Layout

```
src/egeria/        package: schema, prompt, scorer, calibration, metrics, memory, servers, CLI
src/egeria/web/    web interface (static HTML/CSS/JS); img/logo.png is the logo, the icons are generated with scripts/genera_icone.py
scripts/           evaluations (baseline.sh, run_all.sh, checks), interface acceptance test, icons from the logo
tests/             unit tests (+ optional integration tests with the model)
runs/              run outputs (predictions, temperatures, reports); not versioned
documentazione/    project documentation in Italian; documentazione/en/ in English
```
