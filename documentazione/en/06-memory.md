[Italiano](../06-memoria.md) · **English**

# Memory: memories recalled by similarity

> Status: **implemented**, for text and **images**. **Retrieval** works: the memory vote scores 0.56 versus 0.47 for the zero-shot model (text), and with photos the most similar memory belongs to the right category in 5 cases out of 6 (§6). Putting the memories **in the prompt** of the zero-shot model did not improve decisions: the `inject` option was removed on 29/09/2026.

## 1. Idea

Memory is **an object external to the model**: a store of past cases with the decisions taken (preferably confirmed by an outcome or a correction), notes and rules. Entries can be added, inspected and deleted **without retraining anything**.

Given a new state, the most similar memories are retrieved through the state vector, computed by the same model, and returned in the output. For every question the memories cover, the **memory vote** is returned as well.

## 2. Does retrieval work?

**Minimal test** (6 sentences): the last-token hidden state is anisotropic, with cosine 0.91–0.97 for any pair. The **mean over the state tokens** separates much better, so the state vector uses the mean.

**Test on decisions** (typed-decisions): for each test case, the 10 most similar train cases are retrieved and their decisions are used to predict.

| Vector / method | Accuracy | NLL | ECE |
|---|---|---|---|
| **Qwen3.5-2B-Base (mean over tokens), vote with gold distributions** | **0.564** | **0.923** | 0.057 |
| Qwen3.5-0.8B-Base (mean over tokens), same | 0.564 | 0.930 | 0.052 |
| Qwen3-Embedding-0.6B (dedicated model), same | 0.552 | 0.934 | 0.033 |
| 2B-Base, vote with **hard decisions**, 10 memories | **0.562** | 0.963 | 0.043 |
| 2B-Base, hard decisions, 5 memories | 0.554–0.560 | 1.07 | ~0.10 |
| 2B-Base, hard decisions, 3 memories | 0.538 | 1.22 | ~0.16 |
| Prior (no retrieval) | 0.478 | 1.034 | 0.034 |
| *2B-Base zero-shot, without memories* | *0.468* | *1.049* | *0.029* |

**What emerges:**
- **Our model's vector retrieves useful cases** as well as a dedicated embedding model does.
- **With hard decisions** (as in real use) accuracy holds, but honest probabilities require **10 memories**: with 3 the vote is overconfident.

## 3. Memories in the prompt: no help, for now

Experiment: 2B-Base, 3 memories with their gold decisions in the prompt before the state, 2 permutations, paired comparison against the same model without memories, on the same 2,000 decisions.

| | Δ accuracy | Δ NLL |
|---|---|---|
| total | +0.002 [−0.018, +0.022] | +0.009 |
| security incidents | **+0.074*** | **−0.041*** |
| choice | +0.050 [0.000, +0.101] | −0.011 |
| score | **−0.041*** | +0.024* |
| agent traces | **−0.058*** | +0.052* |

\* 95% interval that excludes zero.

**Combining the model with the memory vote**, with the weight fitted on train: the optimal model weight is **0**. The vote alone scores 0.562 versus 0.468, **+9.5 points** [+6.0, +12.6], and NLL −0.119.

**Conclusion.** On repetitive decisions like these, similar cases that were already decided are worth more than the zero-shot model, and the 2B cannot yet exploit them in context. Therefore:
- by default, memory **returns** the memories and their vote;
- prompt injection (`inject`) was removed on 29/09/2026. It can be retried after training, by teaching the model to use memories.

## 4. API

```json
{
  "state": "Salve, da due giorni i bonifici verso i nostri fornitori vengono respinti ...",
  "memory": {"recall": 3},
  "questions": {
    "reparto": {"type": "choice", "instructions": "Quale reparto deve gestire il ticket?",
                "criteria": {"pagamenti": "Pagamenti e bonifici", "tecnico": "Bug e malfunzionamenti", "commerciale": "Contratti e offerte"}},
    "urgente": {"type": "noul", "instructions": "Il ticket è urgente?"}
  }
}
```

**`memory` options:**
- `recall` (0–10): how many memories to return.
- `min_similarity`: similarity threshold.
- `vote` (default `true`): a vote for each covered question. It uses the 10 most similar memories that contain that question, weighted with softmax(similarity / 0.05), plus 10% smoothing.

**Real response** (2B-Base; test store with 3 memories, [examples/memoria_ticket.json](../../examples/memoria_ticket.json)):

```json
"answers": {
  "reparto": {"type": "choice", "choice": "pagamenti", "probabilities": {"pagamenti": 0.993, "...": "..."},
              "memory": {"answer": "pagamenti", "probabilities": {"pagamenti": 0.733, "tecnico": 0.119, "commerciale": 0.148}, "support": 3}},
  "urgente": {"type": "noul", "noul": 0.613, "confidence": 0.225,
              "memory": {"answer": "true", "probabilities": {"true": 0.835, "false": 0.165}, "support": 3}}
},
"memories": [
  {"id": "caso-101", "similarity": 0.9432, "decisions": {"reparto": "pagamenti", "urgente": "true"},
   "note": "IBAN corretto: era un blocco antifrode della banca"},
  {"id": "caso-103", "similarity": 0.8528, "decisions": {"reparto": "commerciale", "urgente": "false"}, "note": ""},
  {"id": "caso-102", "similarity": 0.8384, "decisions": {"reparto": "tecnico", "urgente": "true"}, "note": "CDN scaduta"}
]
```

**How to read it:**
- `memories` are the most similar cases, with decisions and notes. The first one (0.94) is the rejected bank transfers case, and its note ("blocco antifrode", "anti-fraud block") is information the model does not have on its own.
- `answers[q].memory` is the memory vote (`support` = how many memories cover the question), to be shown alongside the model's answer. When the two agree, the decision is more solid; when they diverge, it is a signal for a human.

## 5. Managing the store

```bash
M="--model Qwen/Qwen3.5-2B-Base --memory runs/mia-memoria"
# add a confirmed case (the state can be a JSON request file or just the state; - for stdin)
.venv/bin/egeria memory $M add --id caso-101 --state examples/ticket_it.json \
  --decisions '{"reparto": "pagamenti", "urgente": "true"}' --note "IBAN corretto: era un blocco antifrode della banca"
# search for the most similar ones
.venv/bin/egeria memory $M search --state examples/memoria_ticket.json -k 3
# decide with memory
.venv/bin/egeria decide $M --permutations 2 examples/memoria_ticket.json
# store built from a dataset (typed-decisions train, decisions = gold labels)
.venv/bin/egeria memory --model Qwen/Qwen3.5-2B-Base --memory runs/memoria-Qwen3.5-2B-Base build
```

The store is a folder with `memories.jsonl` (human-readable memories) and `vectors.npy` (vectors). The vector depends on the model: if you change model, the store must be rebuilt.

**What to save.** Only **confirmed** decisions (outcomes, human corrections). Saving the model's answers without verification would reinforce its errors; that is why `decide` never writes to memory.

**Evaluation of memories in the prompt:** the `scripts/eval_memoria.sh` script (with `--inject`) was removed on 29/09/2026 together with `inject`. The numbers in §3 are kept for reference.

## 6. Memory with images

States can contain images ([04-image-states.md](04-image-states.md)), and memory works the same way. The vector is the mean of the hidden states over the state tokens, **visual tokens included**, computed by the multimodal model. This makes it possible to store photos (insurance claims, documents, frames) together with the decisions taken, and to retrieve the ones most similar to a new photo.

```bash
M="--model Qwen/Qwen3.5-2B-Base --memory runs/memoria-foto --image-max-side 448"
echo '{"state": [{"type": "text", "text": "Immagine ricevuta:"}, {"type": "image", "path": "examples/immagini/casa_incendio.jpg"}]}' \
  | .venv/bin/egeria memory $M add --id foto-incendio --state - --decisions '{"azione": "chiamare_vigili"}'
```

Requests must use the same `image_max_side` the photos were stored with: the vector depends on the resolution.

**Verification** ([scripts/verifica_rag_immagini.py](../../scripts/verifica_rag_immagini.py), 2B-Base, threshold set in advance: ≥ 5/6).
- **Store:** the 6 real photos in `examples/immagini/`, each with a category and an action.
- **Queries:** 6 **new** photos from Wikimedia, of the same categories but different:

| New photo | Most similar memory | Similarity | Vote on the action | Zero-shot model (category) |
|---|---|---|---|---|
| receipt from another supermarket, in English | ✓ receipt | 0.83 | ✓ archive | ✓ |
| accident with a scooter on the ground | ✓ car accident | 0.88 | ✓ open a claim | ✓ |
| historic stop sign | ✓ stop | 0.85 | ✓ none | ✓ |
| another fire | ✓ fire | 0.86 | ✓ call the fire brigade | ✓ |
| another cat | ✓ cat | 0.91 | ✓ none | ✓ |
| phishing email in Slovenian, from a smartphone | ✗ receipt (0.62; phishing 0.58) | – | ✗ archive | ✓ phishing |

**Outcome: 5/6, kept.**

**What the error shows:**
- **An image's vector largely captures its visual appearance.** A white screenshot full of text "looks like" a receipt more than another email with a different visual layout.
- **The model, when queried directly, correctly recognizes all 6 photos.** Disagreement between the memory vote and the model's answer is therefore the signal to stop and ask a human.
- **With richer stores** (more examples per category, from different sources) the effect of visual appearance gets diluted.

**Limitation:** in the store, the memories' images appear only as a reference (`[image: percorso]`, where "percorso" is the path): the image data is not copied.

## 7. Questions about memory

### "What does this remind you of?": similar memories (kept)

It is requested with `"memory": {"recall": k}` in the request (k from 1 to 10): the response lists in `memories` the memories most similar to the state, with similarity, decisions and notes, and, for the questions they cover, the memory vote. Retrieval is the one verified above (§2 and §6). In the interface the same thing is done from the *Ricordi* ("Memories") page, with *Cerca* ("Search") or *Cerca con un'immagine* ("Search with an image").

Until 29/09/2026 there was also a `recall` question type that returned the same list: it was removed because it duplicated `memory.recall`.

```json
{
  "state": [{"type": "text", "text": "Immagine ricevuta:"}, {"type": "image", "path": "examples/immagini/nuove/incendio2.jpg"}],
  "image_max_side": 448,
  "memory": {"recall": 2},
  "questions": {
    "azione": {"type": "choice", "instructions": "Quale azione è appropriata?",
               "criteria": {"chiamare_vigili": "Chiamare i vigili del fuoco", "aprire_sinistro": "Aprire una pratica di sinistro", "nessuna": "Nessuna azione"}}
  }
}
```

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --memory runs/memoria-foto --permutations 2 examples/memoria_ricorda.json
```

Real response: a store of 4 photos with notes ([examples/memoria_ricorda.json](../../examples/memoria_ricorda.json)), queried with a **new** photo of a different fire.

```json
"answers": {
  "azione": {"type": "choice", "choice": "chiamare_vigili",
    "probabilities": {"chiamare_vigili": 0.857, "aprire_sinistro": 0.050, "nessuna": 0.093}, "confidence": 0.786,
    "memory": {"answer": "chiamare_vigili",
               "probabilities": {"chiamare_vigili": 0.694, "aprire_sinistro": 0.230, "nessuna": 0.076}, "support": 4}}
},
"memories": [
  {"id": "foto-incendio-0812", "similarity": 0.8591, "decisions": {"azione": "chiamare_vigili"},
   "note": "Incendio in via Verdi, 12 agosto: vigili arrivati in 9 minuti"},
  {"id": "foto-sinistro-2026-0913", "similarity": 0.7985, "decisions": {"azione": "aprire_sinistro"},
   "note": "Urto contro palo, pratica 2026-0913 liquidata"}]
```

### `known`: "is this something you have in memory?" (not kept)

The idea was to answer on three levels: the **same** case already seen, **similar** to something in memory, or **new**. Verification ([scripts/verifica_known.py](../../scripts/verifica_known.py), 2B-Base; threshold set in advance: AUROC ≥ 0.85 on both separations, for each data type):

| Separation | Images (store of 6 photos) | Text (store of 1,200 cases) |
|---|---|---|
| same (cropped copy, or with one extra detail) vs similar | **1.00**: copies ≥ 0.956, different photos ≤ 0.908 | 0.79 |
| similar vs new (category or topic not in the store) | 0.81 | **1.00**: same type ≥ 0.898, off-topic ≤ 0.29 |

A store-relative measure (percentile with respect to the neighbors within the store) does no better. Each data type succeeds at only one of the two questions, so `known` **was not kept**:
- **text, same vs similar:** the typed-decisions cases are generated by text models and resemble each other almost as much as the modified copies do;
- **images, similar vs new:** the errors caused by visual appearance come back (the phishing screenshot looks like a receipt; the colorful bike rack scores 0.83).

**What remains usable, with caution.** `memory.recall` returns the similarities, and the verification yields two practical rules:
- **text:** similarity of the nearest memory below ~0.6 → the topic is absent from memory (here there was nothing between 0.29 and 0.90);
- **images:** similarity ≥ ~0.94 → almost certainly the same photo, even when cropped (here between 0.908 and 0.956).

These thresholds were measured on few cases and on these particular stores. They depend on the model, on `image_max_side` and, above 20 memories, on centering. For exact text duplicates, comparing the hash of the normalized text is enough.

### Why `known` fails, and what remains

> **Abandoned on 29/09/2026**, including as a training goal. The ideas below stay as a record; for "is this case outside what I know?" what remains is the out-of-domain calibration measurement (F0.5 in [02-implications-and-proposal.md](02-implications-and-proposal.md) §6) and `memory.min_similarity`.

**The common cause: recall ranks, `known` uses a threshold.** Recall (`memory.recall`) only has to rank the memories, and the ranking stays correct even with "squashed" similarities. `known` requires an absolute threshold, but the similarity scale depends on the model, the data type, the resolution and the store size (centering beyond 20 memories).

**Text, same vs similar.** The vector is a mean over the whole state: it represents *what kind of thing* the state is, not *which instance*. Different cases from the same flow (same structure, a few different values) have similarity 0.90–0.996; modified copies 0.95–0.996.

**Images, similar vs new.** Visual tokens dominate the mean, so the vector mostly captures appearance (colors, composition, "a sheet with text"). The model, when queried directly, recognizes all 6 categories instead: the meaning is there, but it does not surface in the mean.

**How it could have come back:**
1. **Without training: retrieval + verification.** Retrieval finds the candidate; a `noul` with both states in the prompt asks "is it the same case?" / "is it the same kind of situation?". To be verified against the same thresholds.
2. **F1, a dedicated loss on pairs** (the supervised equivalent of RLCD). It makes the probability of that `noul` calibrated. Pairs that can be built: modified copies and crops (same), different cases from the same flow or category (similar), topics and categories not in the store (new).
3. **F1, contrastive embedding with LoRA.** Pulls vectors of the same type together and pushes the others apart, using hard examples (screenshot vs receipt). It also improves recall and the memory vote.
4. **Same case in text:** a content check (hash of the normalized text, MinHash/Jaccard, identifying fields), which is more reliable than any semantic vector.
