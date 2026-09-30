**Italiano** · [English](en/08-state-reuse.md)

# Riuso dello stato

> Stato: **implementato** il 29/09/2026 (F0.5, punto 1 della roadmap in [02-implicazioni-e-proposta.md](02-implicazioni-e-proposta.md) §6). Attivo di default in modalità `auto`.

## 1. Idea

Tutte le domande e le permutazioni su uno stesso stato iniziano con lo stesso prefisso: istruzioni di sistema e stato, immagini comprese. Prima ogni prompt rielaborava tutto, stato e immagini, una volta per domanda e per permutazione. Ora:

1. il prefisso si calcola **una volta**, con la cache;
2. la cache ibrida si **duplica** per le domande. Contiene lo stato ricorrente e la convoluzione dei layer Gated DeltaNet e le chiavi/valori dei layer di attenzione;
3. si passano in batch **solo le code**: tipo di domanda, domanda e opzioni.

transformers sa già continuare più token da una cache di Qwen3.5: la convoluzione riparte dal proprio stato e la regola delta dallo stato ricorrente salvato. Non serve toccare il modello.

## 2. Dove si taglia

- **Testo.** Il prefisso è la parte iniziale uguale in tutte le sequenze di token. Le code sono esattamente il resto di ogni sequenza, quindi la tokenizzazione è identica per costruzione.
- **Immagini.** Il processore di Qwen espande le immagini in token, quindi si taglia il testo subito dopo il blocco dello stato (`</state>` e riga vuota). Per ogni prompt si verifica che il taglio non cambi la tokenizzazione; se la cambia, si torna ai prompt interi. Le posizioni M-RoPE delle code sono indice + `rope_deltas` del prefisso.
- **Padding.** Le code in batch hanno padding a destra. Con un modello causale le posizioni dopo l'ultimo token reale non cambiano quelle lette.

## 3. Verifica

Script: [scripts/bench_riuso_stato.py](../scripts/bench_riuso_stato.py). Per ogni richiesta confronta i prompt interi (`never`) con il prefisso condiviso (`always` oppure `auto`).

Richieste usate:
- le 12 della suite italiana ([examples/suite_it.jsonl](../examples/suite_it.jsonl));
- le 9 immagini di [examples/suite_immagini.jsonl](../examples/suite_immagini.jsonl), con le loro domande raggruppate in una richiesta per immagine, a 448 e a 800 px.

In tutto 30 richieste e 77 decisioni, con 2 permutazioni, su RTX 4070 Laptop.

**Le risposte non cambiano:**

| Modello | Risposte uguali | Differenza massima di probabilità |
|---|---|---|
| 2B-Base, GPU bf16 | 77/77 | 0.026 |
| 0.8B-Base, GPU bf16 | 77/77 | 0.030 |
| 0.8B-Base, CPU float32 (solo testo) | 25/25 | 0.0000 |

Le differenze su GPU sono rumore del bf16: su CPU in float32 i due percorsi coincidono. Il test d'integrazione `test_state_reuse_matches_full_prompts` lo verifica a ogni esecuzione con il modello.

**Tempo del prefisso condiviso rispetto ai prompt interi**, per fascia di token evitati (prefisso × righe in più; sotto 1 = più veloce):

| Token evitati | 2B, GPU | 0.8B, GPU | 0.8B, CPU |
|---|---|---|---|
| meno di 500 | 1.12 | 1.77 | 0.77 (243–535 token) |
| 500–1000 | 0.58 | 0.93 | – |
| 1000–2000 | 0.38 | 0.62 | – |
| 2000–4000 | 0.22 | 0.29 | – |

**Cosa emerge:**
- **Su GPU il prefisso condiviso ha un costo fisso:** un passaggio in più, cioè qualche decina di millisecondi di lanci di kernel. Si recupera solo evitando abbastanza token. Con i testi brevi della suite, 2–3 domande e stati di poche righe, condividere era più lento (fino al +33% sul 2B e al +95% sullo 0.8B).
- **Con le immagini il guadagno è grande,** perché i token visivi sono tanti: sul 2B una foto a 800 px con le sue domande passa da 1.1–1.5 s a 0.27–0.32 s, cioè **dal 75% all'81% in meno**.
- **Il pareggio dipende dalla taglia:** ~500 token evitati sul 2B, ~1000 sullo 0.8B, dove ogni token costa meno.
- **Su CPU il calcolo domina** e condividere conviene sempre: dal 15% al 36% in meno anche sui testi brevi.

## 4. La regola della modalità `auto`

Si condivide il prefisso se i token evitati (prefisso × righe in più) superano la soglia:

| Dispositivo | Soglia |
|---|---|
| GPU, modello testuale sotto 1.2 miliardi di parametri (0.8B) | 1000 token |
| GPU, modelli più grandi (2B e oltre) | 500 token |
| CPU | sempre, con almeno 2 righe |

Risultato con `auto`, rispetto ai prompt interi (mediana):

| Modello (GPU) | Testo | Immagini |
|---|---|---|
| 2B-Base | 1.00 (nessuna perdita) | **0.39** (fino a 0.20) |
| 0.8B-Base | 1.01 | **0.61** (fino a 0.24) |

**Carico realistico sul testo: typed-decisions.** 40 casi del test, 5 domande ciascuno, 2 permutazioni, 2B su GPU, due esecuzioni per modalità (`egeria predict --limit 40 --share-state never|auto`):

| | Prompt interi | `auto` |
|---|---|---|
| Tempo per caso | 500–516 ms | **233–247 ms** (−53%, circa il doppio più veloce) |
| Risposte uguali | – | 197/200 |

Le 3 risposte diverse erano quasi pari: per esempio 0.288 contro 0.281 per le prime due opzioni. Il rumore del bf16 le sposta da una parte o dall'altra. Qui i prompt sono lunghi (~720 token per domanda) e le domande tante, quindi `auto` condivide il prefisso anche sul testo.

Le soglie sono misurate su una sola GPU (RTX 4070 Laptop). Su hardware diverso il pareggio può spostarsi: lo script le rimisura.

## 5. Uso

- **Di default** lo scorer usa `auto`.
- **Da riga di comando:** `--share-state auto|always|never` su `decide`, `predict`, `suite`, `memory` e `model-server`.
- **Da Python:** `DecisionScorer(..., share_state="auto")`, oppure `scorer.share_state = "never"` per confrontare.

```bash
# rimisurare su un'altra macchina (GPU: testo e immagini; --cpu: solo testo, float32)
.venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-2B-Base
.venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-2B-Base --mode auto
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_riuso_stato.py Qwen/Qwen3.5-0.8B-Base --cpu
```

## 6. Cosa resta fuori

- **Le domande `short_answer`** fanno ancora ciascuna il proprio passaggio completo: prima si calcola il prompt intero, poi si completano le candidate. Si potrebbe partire dalla stessa cache del prefisso.
- **Il vettore dello stato per la memoria** (`/v1/embed`) è un passaggio a parte sullo stesso stato. Il server web lo chiede prima della decisione, con due chiamate HTTP separate.
- **Il costo fisso per passaggio** resta il limite sui testi brevi su GPU. È l'obiettivo dei CUDA graphs ([04-stato-con-immagini.md](04-stato-con-immagini.md) §4).
