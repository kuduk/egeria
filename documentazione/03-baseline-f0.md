**Italiano** · [English](en/03-baseline-f0.md)

# Baseline F0: readout zero-shot su Qwen3.5 e diagnostica della profondità dinamica

> Stato: **implementata**. I risultati sono nella §6 e si aggiornano a ogni run.
>
> **Profondità dinamica tolta il 29/09/2026.** Letture intermedie, uscita anticipata e temperature per layer (§1 punto 6, §6.2, §6.4) non sono più nel codice. I risultati restano qui come riferimento. `min_confidence` è rimasto e marca le risposte incerte (`status`), calcolato sul modello completo.

## 1. Cosa fa

La F0 trasforma un Qwen3.5 **senza alcun training** in un modello decisionale compatibile con l'API Jev. È lo stesso metodo di SemIf:

1. Per ogni domanda si costruisce un prompt che contiene lo stato, la domanda e le opzioni etichettate A, B, C, …, con il chat template in modalità non-thinking.
2. **Un solo forward pass** per prompt, senza generare token.
3. Si prende l'hidden state finale dell'ultima posizione e lo si moltiplica **solo per le righe dell'lm_head delle lettere**. La softmax è ristretta alle opzioni dichiarate.
4. Con `--permutations P` ogni domanda viene valutata con P ordini diversi delle opzioni: rotazioni cicliche per `noul`/`choice`, ordine diretto e inverso per `score`. Le log-probabilità si mediano per opzione, così si riduce il bias di posizione.
5. **Temperature scaling** per tipo di domanda, fittato su dati separati (il train split).
6. **Readout intermedi** (tolti il 29/09/2026). Gli hidden state all'uscita dei layer full-attention passano per la norm finale e per le righe delle lettere (logit lens). Servono a studiare la profondità dinamica per asserzione (vedi [02 §4bis](02-implicazioni-e-proposta.md)).

## 2. Struttura del codice

| File | Contenuto |
|---|---|
| [src/egeria/schema.py](../src/egeria/schema.py) | Validazione del body `/v1/systemone` (noul, choice fino a 26 opzioni, score da 2 a 10 livelli) |
| [src/egeria/prompt.py](../src/egeria/prompt.py) | Prompt con opzioni a lettere e ordinamenti per le permutazioni |
| [src/egeria/scorer.py](../src/egeria/scorer.py) | `DecisionScorer`: caricamento del modello (solo testo, niente torre visiva), forward batch con right padding, readout finale, `decide()` in formato Jev |
| [src/egeria/confidence.py](../src/egeria/confidence.py) | Softmax con temperatura, formule di `confidence` di Jev, formato della risposta |
| [src/egeria/calibration.py](../src/egeria/calibration.py) | Temperature scaling (ricerca della sezione aurea su 1/T, la NLL è convessa) |
| [src/egeria/metrics.py](../src/egeria/metrics.py) | Accuratezza, NLL, KL, Brier, ECE, copertura al 5% d'errore, score MAE, flip rate |
| `depth.py` (tolto il 29/09/2026) | Temperature per (tipo, layer), qualità per layer, simulazione dell'early exit a soglia |
| [src/egeria/datasets.py](../src/egeria/datasets.py) | Loader di `LocalLLaMA/typed-decisions` (il `--limit` campiona in modo equispaziato su tutti i workflow) |
| [src/egeria/cli.py](../src/egeria/cli.py) | CLI `egeria` |
| [scripts/baseline.sh](../scripts/baseline.sh) | Pipeline completa per uno o più modelli (ripartibile) |
| [scripts/run_all.sh](../scripts/run_all.sh) | Download dei modelli + baseline in sequenza, da lanciare staccato dalla sessione |

## 3. Installazione

```bash
cd /home/kuduk/egeria
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[model,eval,dev]"
.venv/bin/egeria info        # GPU, VRAM libera, stato dei kernel veloci
```

### Kernel veloci per i layer Gated DeltaNet (consigliati)

Senza questi pacchetti transformers usa l'implementazione PyTorch di riferimento. È corretta, ma molto più lenta: oltre 10× su `chunk_gated_delta_rule` secondo transformers.

```bash
# flash-linear-attention: kernel Triton, nessuna compilazione
uv pip install --python .venv/bin/python "flash-linear-attention>=0.2.2"

# causal-conv1d: estensione CUDA. Deve vedere il torch del venv (--no-build-isolation).
# Senza wheel precompilata per la versione di torch, compila da sorgente (~10-15 min).
uv pip install --python .venv/bin/python ninja packaging wheel setuptools
CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=4 \
  uv pip install --python .venv/bin/python --no-build-isolation "causal-conv1d>=1.5"
```

`TORCH_CUDA_ARCH_LIST="8.9"` corrisponde alla RTX 4070 (Ada). Su altre GPU va adattato: 8.0 per A100, 9.0 per H100.

**CPU con i kernel installati.** transformers sceglie il kernel all'importazione e, con causal-conv1d installato, lo userebbe anche sui tensori CPU, dove fallisce. Lo scorer installa un dispatcher (`install_device_dispatch`): kernel veloci su CUDA, implementazione PyTorch di riferimento altrove. CPU e GPU funzionano quindi entrambi.

Tempi osservati il 26/09/2026: flash-linear-attention 0.5.2 si installa in pochi secondi. causal-conv1d 1.7.0 si compila da sorgente in circa 6 minuti, perché per torch 2.11 non c'è una wheel precompilata. Effetto sul forward (0.8B, immagine a 224 px, 3 domande): 77 → 68 ms con fla, → 45 ms con entrambi; dettagli in [04-stato-con-immagini.md](04-stato-con-immagini.md) §4.

### Quantizzazione a 4 bit (per 4B e 9B su 8 GB)

```bash
uv pip install --python .venv/bin/python -e ".[quant]"
.venv/bin/egeria predict --model Qwen/Qwen3.5-4B --quantize 4bit ...
```

### Note su WSL2

- **La VRAM è condivisa con il desktop di Windows**, che ne occupa 1–4 GB. `egeria info` mostra quella realmente libera.
- **Se la VRAM finisce, il driver WSL può "sforare" nella memoria condivisa di sistema** invece di dare errore, e diventa molto lento. Se un run rallenta all'improvviso, ridurre `--batch-tokens`.
- **WSL vede di default metà della RAM dell'host.** Per l'offload di modelli grandi si alza con `memory=48GB` nella sezione `[wsl2]` di `%UserProfile%\.wslconfig`, poi `wsl --shutdown`.

## 4. Uso

Esempi pronti in [examples/](../examples/): `ticket_it.json` (ticket in italiano) e `jev_esempio.json` (l'esempio della documentazione Jev).

```bash
# Una richiesta in formato Jev (su GPU)
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B --permutations 2 examples/ticket_it.json

# Sul CPU (GPU occupata o assente): usare float32. Il 0.8B impiega ~5 s per 3 domande × 2 permutazioni
.venv/bin/egeria decide --model Qwen/Qwen3.5-0.8B-Base --device cpu --dtype float32 --permutations 2 \
  --temperatures runs/Qwen3.5-0.8B-Base/temperature.json examples/ticket_it.json

# Tabella di confronto fra modelli
.venv/bin/egeria compare runs/*/report-cal.json

# Predizioni su typed-decisions
.venv/bin/egeria predict --model Qwen/Qwen3.5-0.8B --split test --permutations 2 --out runs/x/test.jsonl

# Temperature fittate sul train (una per tipo di domanda)
.venv/bin/egeria calibrate --predictions runs/x/train.jsonl --out runs/x/temperature.json

# Report
.venv/bin/egeria evaluate --predictions runs/x/test.jsonl --temperatures runs/x/temperature.json

# Pipeline completa, staccata dalla sessione
setsid nohup scripts/run_all.sh > runs/run_all.log 2>&1 &

# Test (quelli col modello solo se richiesti)
.venv/bin/pytest -q
EGERIA_TEST_MODEL=Qwen/Qwen3.5-0.8B .venv/bin/pytest -q -m model
```

## 5. Dettagli tecnici e insidie verificate

- **Right padding con i layer DeltaNet è sicuro.** Il modello è causale, quindi i pad dopo l'ultima posizione non influenzano i token precedenti. In fp32 batch e singolo coincidono a 1e-5.
- **Rumore bf16.** In bf16 i logit delle lettere variano fino a ~0.2 al cambiare della forma del batch (0.8B). È rumore numerico, non un bug. Può cambiare l'argmax nei casi al limite. Per confronti fini usare `--dtype float32`: il 0.8B in fp32 occupa ~3.2 GB.
- **Confine della risposta.** All'avvio si verifica che, per ogni lettera, `tokenize(prompt + lettera) == tokenize(prompt) + [token lettera]`. Tutte le lettere A–Z sono token singoli nel tokenizer di Qwen3.5.
- **Solo modello testuale.** Si carica `Qwen3_5ForCausalLM` con la text config: la torre visiva non viene caricata.
- **Readout efficiente.** Non si calcola l'lm_head su tutto il vocabolario (248k): si moltiplica l'hidden finale solo per le 26 righe delle lettere, in fp32.
- **Hook per i readout intermedi.** Catturano solo l'hidden dell'ultima posizione per ogni riga del batch, quindi la memoria extra è trascurabile. L'uscita dell'ultimo layer, dopo la norm, coincide con il readout finale (verificato).
- **Qualità delle metriche.** Il gold di typed-decisions è la media di 3 campioni di un teacher "classe 4B", con tetto di auto-accordo 0.735. Le temperature fittate sul train split rendono il risultato "calibrato sul workload", non puramente zero-shot. Si riportano sempre sia T=1 sia calibrato.

## 6. Risultati (26/09/2026)

**Setup:**
- Benchmark: typed-decisions, split test completo (400 casi, 2.000 decisioni). Ci sono 2 permutazioni delle opzioni per domanda, in bf16 su RTX 4070 Laptop, con i kernel di fallback PyTorch.
- Temperature fittate su 300 casi del train (1.500 decisioni, campione equispaziato sui 4 workflow).
- Riproducibile con `scripts/run_all.sh`.

### 6.1 Qualità

| Modello | T | Acc | noul | choice | score | NLL | KL | Brier/k | ECE | Flip | ms/caso |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Prior** (frequenze gold del train, non legge lo stato) | – | **0.497** | – | – | – | **1.025** | **0.327** | 0.175 | 0.033 | – | 0 |
| Qwen3.5-0.8B | 1 | 0.428 | 0.518 | 0.490 | 0.314 | 1.231 | 0.641 | 0.214 | 0.244 | 0.71 | 474 |
| Qwen3.5-0.8B | fit | 0.428 | 0.518 | 0.490 | 0.314 | 1.097 | 0.378 | 0.189 | 0.031 | 0.71 | 474 |
| Qwen3.5-0.8B-Base | 1 | 0.450 | 0.508 | 0.483 | 0.380 | 1.185 | 0.523 | 0.209 | 0.122 | 0.44 | 504 |
| Qwen3.5-0.8B-Base | fit | 0.450 | 0.508 | 0.483 | 0.380 | 1.121 | 0.392 | 0.192 | 0.053 | 0.44 | 504 |
| Qwen3.5-2B | 1 | 0.469 | 0.665 | 0.398 | 0.375 | 1.239 | 0.616 | 0.200 | 0.176 | 0.38 | 722 |
| Qwen3.5-2B | fit | 0.469 | 0.665 | 0.398 | 0.375 | 1.098 | 0.377 | 0.185 | 0.061 | 0.38 | 722 |
| Qwen3.5-2B-Base | 1 | 0.468 | 0.517 | 0.522 | 0.390 | 1.195 | 0.632 | 0.209 | 0.226 | 0.23 | 700 |
| Qwen3.5-2B-Base | fit | 0.468 | 0.517 | 0.522 | 0.390 | **1.049** | **0.346** | 0.183 | **0.029** | 0.23 | 700 |

**Come leggere le colonne:**
- *Brier/k* è il Brier medio per opzione rispetto alla label hard.
- *Flip* è la frazione di decisioni in cui le due permutazioni danno argmax diversi.
- *ms/caso* è il tempo per 5 domande × 2 permutazioni = 10 prompt.
- La Prior qui (0.497) è calcolata sul nostro campione di train. Il dataset riporta 0.470 con la sua definizione.

**Cosa emerge:**
1. **Zero-shot, nessun modello piccolo batte la Prior.** Anche il migliore, il 2B-Base calibrato, è appena sotto per NLL e KL. I modelli leggono lo stato solo marginalmente: il 2B-Base supera la Prior solo sul customer service (0.616). Sulle fatture tutti crollano (0.28–0.48), perché servono confronti numerici in un solo passaggio. È coerente con la letteratura: Laya base zero-shot ≈ 0.36, e SemIf usa il 4B.
2. **Il temperature scaling funziona sempre.** L'ECE scende a 0.03–0.06. I modelli sono sovraconfidenti (T fittate 1.7–7). Per `noul` del 0.8B-Base la temperatura va al limite (T = 20): quel readout non porta informazione e la calibrazione lo appiattisce a 0.5.
3. **Base meglio di Instruct per robustezza all'ordine** (flip 0.44 contro 0.71 sul 0.8B, 0.23 contro 0.38 sul 2B) e, dopo la calibrazione, per NLL e KL sul 2B. Conferma la scelta di partire dai checkpoint Base per il training.
4. **La sensibilità all'ordine cala con la taglia** (0.71 → 0.38 Instruct; 0.44 → 0.23 Base). Le permutazioni restano necessarie.

### 6.2 Profondità dinamica (readout ai confini di blocco: layer 3, 7, 11, 15, 19, 23)

| Modello | Calcolo con uscita **oracolo** | Risparmio potenziale | Soglia 0.4: calcolo / accordo col completo | Soglia 0.2: calcolo / accordo col completo |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.694 | 31% | 0.992 / 100% | 0.952 / 96.8% |
| Qwen3.5-0.8B-Base | 0.654 | 35% | 0.992 / 100% | 0.965 / 99.7% |
| Qwen3.5-2B | 0.727 | 27% | 0.989 / 100% | 0.829 / 78.3% |
| Qwen3.5-2B-Base | 0.643 | **36%** | 0.970 / 100% | 0.847 / 84.9% |

L'**oracolo** esce al primo layer da cui l'argmax non cambia più fino all'ultimo: è il risparmio massimo a decisioni invariate.

Le soglie sono sulla confidenza **corretta per il caso**, la stessa scala di `min_confidence` (§6.4): 0 = a caso, 1 = certezza.

**Cosa emerge:**
1. **Il potenziale c'è: 27–36% di calcolo risparmiabile** senza cambiare nessuna decisione. Circa metà delle decisioni è già definitiva al layer 15 su 24; nel 2B-Base 222 su 2.000 lo sono già al layer 3.
2. **Il criterio di confidenza zero-shot non lo coglie.** Ai layer intermedi il logit lens calibrato è poco sicuro anche quando ha già la risposta.
   - A soglia 0.4 si risparmia solo l'1–3%, ma con accordo del 100% col modello completo: il criterio è sicuro ma troppo prudente.
   - Abbassando la soglia a 0.2 si risparmia fino al 17%, ma alcune uscite cambiano risposta (accordo 78–85% sui 2B), a parità di accuratezza.
3. **Conseguenza per la F1.** Servono **teste di uscita addestrate** per ogni confine di blocco, che rendano affidabile la confidenza intermedia. L'obiettivo misurabile è chiudere il divario fra risparmio ottenuto e oracolo.

**Sulla suite italiana** (2B-Base, casi più netti del benchmark, temperature per layer fittate su typed-decisions):
- **Calcolo:** oracolo **0.60** (40% risparmiabile); con soglia 0.6 **0.90** (10% risparmiato).
- **Dove si decide.** Le `choice` chiare si decidono al **layer 15 su 24** con confidenza già alta, e a volte più alta che all'ultimo layer:
  - "nuova spedizione" 0.91 al layer 15 contro 0.83 finale;
  - "logistica" 0.91 contro 0.89;
  - "cambio pagamento" 0.79 contro 0.91;
  - "successo" 0.71 contro 0.74.
- **Le `noul` del Base restano a 0.50–0.63 a tutti i layer**: la confidenza non basta mai a uscire, anche quando la risposta è stabile dal layer 15.
- **Le uscite anticipate ereditano gli errori del modello completo, ma non ne aggiungono.** Per esempio "newsletter" per l'email truffa è 0.76 già al layer 15, ed è sbagliato anche alla fine.
- **I layer 3–11 non sono affidabili.** La risposta cambia spesso e la confidenza è bassa: nel 2B la decisione si forma fra il blocco 3 (layer 11) e il blocco 4 (layer 15).

### 6.3 Confronto 2B e 0.8B

**Confronto appaiato sul benchmark.** `egeria paired`: differenza B − A sulle stesse 2.000 decisioni, con temperature fittate e intervallo bootstrap al 95% ricampionando per caso. L'asterisco indica un intervallo che esclude lo zero.

| Confronto | Δ accuratezza | Δ NLL | Δ Brier | Dove cambia di più |
|---|---|---|---|---|
| 2B-Base − 0.8B-Base | +0.018 [−0.002, +0.039] | **−0.071*** | **−0.037*** | customer service +0.088*; security −0.054* |
| 2B − 0.8B (Instruct) | **+0.041*** | +0.000 | −0.006 | noul +0.147*, score +0.061*, fatture +0.200*; choice −0.092*, security −0.084* |
| 2B-Base − 2B (Instruct) | −0.002 | **−0.048*** | **−0.018*** | choice +0.123*; noul −0.148*, fatture −0.122* |

Per NLL e Brier un valore negativo è meglio.

**Suite italiana** ([examples/suite_it.jsonl](../examples/suite_it.jsonl)): 12 casi, 25 decisioni con risposta netta, incluse trappole (negazioni, importi che non tornano, azioni distruttive). Si lancia con `egeria suite`.

| Modello | Corrette | noul | choice | score | p(atteso) media |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 12/25 | 4/10 | 7/11 | 1/4 | 0.46 |
| Qwen3.5-0.8B-Base | 13/25 | 4/10 | 7/11 | 2/4 | 0.45 |
| **Qwen3.5-2B** | **17/25** | **8/10** | 8/11 | 1/4 | **0.55** |
| **Qwen3.5-2B-Base** | **17/25** | **8/10** | 7/11 | 2/4 | 0.51 |

**Cosa emerge:**
1. **Il 2B è meglio del 0.8B, ma in modo diverso a seconda della metrica.**
   - Sul benchmark l'accuratezza sale poco (+2–4 punti); migliorano soprattutto le probabilità (NLL e Brier).
   - Sulla suite italiana il salto è netto (+4–5 risposte su 25).
   - Il guadagno più grande è sulle `noul`: i 0.8B rispondono sì/no quasi a caso (p ≈ 0.5), il 2B da 4/10 passa a 8/10.
2. **Il 2B gestisce le negazioni** ("non vuole il rimborso", "non voglio disdire", con p fino a 0.92). I 0.8B no.
3. **Instruct e Base hanno profili complementari**: l'Instruct è più forte sulle `noul`, il Base sulle `choice`, con probabilità migliori. Per il training si parte comunque dal Base (§6.1).
4. **Errori comuni a tutti** (servono training o una scomposizione in domande atomiche):
   - la fattura con importo errato viene "pagata", anche se il 2B risponde correttamente che l'importo **non** corrisponde. È un'incoerenza fra domande, lo stesso problema documentato per Jev. Conferma il pattern "harness": domande atomiche e logica nel codice;
   - l'agente che cancella il database viene giudicato "successo";
   - l'alert innocuo viene escalato;
   - l'email truffa: il 2B risponde "non phishing".
5. **Le `score` restano deboli per tutti** (1–2 su 4): le scale ordinali sono la primitiva più difficile, come per Laya.

### 6.4 Profondità dinamica reale con `min_confidence` per asserzione

> **Tolta il 29/09/2026.** Questa sezione descrive l'esecuzione blocco per blocco, che non è più nel codice. Oggi resta la parte sulla soglia: `min_confidence` con la stessa precedenza e la stessa scala descritte sotto, confrontata con la confidenza del modello completo. La risposta riporta `min_confidence` e `status`, non più `depth`.

Oltre alla simulazione, lo scorer esegue davvero il modello **blocco per blocco** (`DecisionScorer.score_adaptive`):
1. A ogni punto d'uscita calcola, per ogni domanda, la distribuzione combinata sulle permutazioni, calibrata con la temperatura di quel layer.
2. Le domande che raggiungono la **propria** `min_confidence` escono. Le loro righe vengono tolte dal batch, e il batch si accorcia fino alla sequenza più lunga rimasta: con right padding su un modello causale è sicuro.
3. Maschere e rotary vengono ricalcolati per il batch ridotto.

Senza uscite, il risultato coincide esattamente con il forward standard (verificato in fp32: differenza 0.0).

**API** (estensione del formato Jev; senza `min_confidence` il comportamento è identico a prima):

```json
{
  "state": "...",
  "min_confidence": 0.6,
  "questions": {
    "urgente":  {"type": "noul", "instructions": "...", "min_confidence": 0.55},
    "reparto":  {"type": "choice", "instructions": "...", "criteria": {"...": "..."}},
    "rischio":  {"type": "score", "instructions": "...", "criteria": ["..."], "min_confidence": null}
  }
}
```

**Precedenza della soglia**, dalla più specifica alla più generale:
1. **Domanda:** il valore locale sovrascrive sempre quello globale. `null` esplicito = modello completo.
2. **Richiesta:** il campo globale `min_confidence`. `null` esplicito = modello completo.
3. **Server:** `--min-confidence` da CLI, oppure `default_min_confidence` di `decide()`.
4. **Nessuno dei precedenti:** **default prudente**, cioè modello completo senza uscite anticipate.

**Semantica e risposta:**
- **Cosa misura la soglia.** `min_confidence` si confronta con la **confidenza corretta per il caso**, la stessa del campo `confidence` della risposta: 0 = distribuzione uniforme, 1 = certezza.
  - `choice`: (n·p_max − 1)/(n − 1), la formula di Jev.
  - `noul`: 2·p_max − 1. Il campo `confidence` per `noul` è un'estensione Egeria.
  - `score`: la formula di Jev basata sulla distanza dalla moda.

  Una prima versione confrontava la soglia con p_max. Era sbagliato: per le `noul` p_max ≥ 0.5 sempre, quindi una soglia 0.5 faceva uscire una risposta a testa o croce. Con la scala corretta la stessa soglia ha lo stesso significato con 2 o con 20 opzioni.
- **Limite delle `score`.** Per le `score` la formula di Jev tronca a 0 le distribuzioni molto piatte. Queste non superano nessuna soglia positiva e arrivano sempre all'ultimo layer, marcate `uncertain`.
- **Dove si esce.** Solo ai confini di blocco da circa metà profondità in su: layer 11, 15, 19 e 23 su 24. Prima, la risposta non è ancora formata (§6.2).
- **Campi in risposta.** Ogni risposta riporta `depth` (layer usati) e, se ha una soglia, `min_confidence` e `status`:
  - `decided`: soglia raggiunta;
  - `uncertain`: nemmeno l'ultimo layer la raggiunge. È il segnale per escalare a un modello più grande, al thinking o a un umano.
- **Temperature.** Servono quelle per layer (`egeria calibrate` su predizioni con `--exits blocks`). Il file salva anche i layer a cui si riferiscono.

**Prime misure** (2B-Base, suite italiana, 2 permutazioni, soglia di default per tutte le domande):

| `min_confidence` | Corrette | Profondità media | `uncertain` | Tempo |
|---|---|---|---|---|
| nessuna (default prudente) | 17/25 | 1.00 | – | 1.85 s |
| 0.6 | 17/25 | 0.93 | 20/25 | 1.89 s |
| 0.4 | 17/25 | 0.90 | 16/25 | 1.83 s |
| 0.2 | 16/25 | **0.83** | 9/25 | **1.73 s** |

**Latenza contro calcolo: la richiesta aspetta la domanda più profonda.** Le domande di una richiesta girano nello stesso batch, quindi la risposta arriva quando finisce la più profonda. Uscire prima riduce le righe del batch nei layer successivi, ma la latenza dipende da quanto costa un layer con meno righe.

Misura sul ticket, 3 domande × 2 permutazioni, uscita forzata al layer 11 (12 layer usati su 24), mediana su 9 esecuzioni:

| Hardware / modello | Tutte complete (24) | Tutte escono (12) | Miste: 2 escono, 1 completa | Quota pagata del tratto finale |
|---|---|---|---|---|
| GPU, 2B-Base bf16 | 188 ms | 106 ms (−44%) | 150 ms (−20%) | 54% |
| GPU, 0.8B-Base bf16 | 130 ms | 66 ms (−49%) | 93 ms (−29%) | 41% |
| CPU, 0.8B-Base fp32 | 4.046 s | 2.058 s (−49%) | 2.676 s (−34%) | **31%** ≈ 1/3 delle righe |

L'ultima colonna è (miste − tutte escono) / (complete − tutte escono): quanto del costo dei layer 12–24 si paga ancora quando resta solo 1 domanda su 3.

- **Se tutte le domande escono, la latenza si dimezza** davvero, su GPU e su CPU.
- **Su CPU il costo è proporzionale alle righe**: con 1 domanda su 3 nel tratto finale si paga circa 1/3 (31%).
- **Su GPU no.** Con batch piccoli la GPU è sottoutilizzata: togliere 2/3 delle righe fa risparmiare solo metà del tratto finale (41–54% pagato). La domanda più profonda domina la latenza.
- **Conseguenze per il design:**
  1. su GPU il guadagno di **latenza** arriva quando escono *tutte* le domande della richiesta;
  2. il guadagno di **calcolo e throughput** è sempre reale e conta su CPU e su un server con molte richieste;
  3. per sfruttarlo anche in latenza servono:
     - **risposte progressive**: restituire ogni risposta appena esce, in streaming, così il chiamante può già agire sulle decisioni facili;
     - **batching continuo a livello di blocco**: le righe uscite vengono sostituite da righe di altre richieste (F4).

Esempio completo: [examples/ticket_it_soglie.json](../examples/ticket_it_soglie.json).

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 \
  --temperatures runs/Qwen3.5-2B-Base/temperature.json examples/ticket_it_soglie.json
.venv/bin/egeria suite --model Qwen/Qwen3.5-2B-Base --permutations 2 \
  --temperatures runs/Qwen3.5-2B-Base/temperature.json --min-confidence 0.4 examples/suite_it.jsonl
```

### 6.5 Conclusione della F0

Il readout zero-shot sui Qwen3.5 piccoli non basta. Harness, metriche, calibrazione e diagnostica della profondità sono pronti e verificati. Il passo successivo è la **F1: training sui modelli piccoli** (LoRA + readout + teste di uscita), vedi [02 §6](02-implicazioni-e-proposta.md). Il 4B e il 9B a 4 bit restano un dato opzionale.
