# Primitive di lettura oltre noul, choice e score

> Stato: implementate e **verificate** `rank`, `number`, `open` ed `embed` (§4). **Tolte** perché non hanno superato la prova: `multi` (bias verso il "sì", tornerà con il training F1) e `surprise` (AUROC 0.50). Quelle con training (`span`, `locate`, `value`, `tags`, `why`) sono da fare. Esempi d'uso nella §5.

## 1. Cosa si può leggere in un solo forward pass

Nello stesso passaggio sono disponibili:
- i logit delle lettere nell'ultima posizione, già usati;
- gli hidden state di **tutte** le posizioni e di **tutti** i layer;
- le attenzioni verso lo stato e verso i token dell'immagine.

Da qui si possono ricavare altre primitive senza generare testo.

### Zero-shot (senza training)

| Primitiva | Restituisce | Come si legge | Esempio |
|---|---|---|---|
| `multi` (tolta, vedi §4) | probabilità indipendente per ogni opzione | una `noul` per opzione, nello stesso batch | problemi presenti in un ticket |
| `rank` | ordinamento completo con probabilità | distribuzione della `choice` | riordinare documenti o azioni |
| `number` (a intervalli) | valore atteso + intervallo | `score` con intervalli numerici come livelli | quante persone nell'immagine |
| `open` | top-k su tutto il vocabolario | softmax sull'lm_head completo nell'ultima posizione | colore dell'auto, senza elencare le opzioni (risposte di un solo token) |
| `embed` | vettore semantico dello stato | hidden finale (come Qwen3-Embedding) | ricerca di casi simili, deduplica, cambio di scena nel video |
| `recall` | i ricordi più simili (con decisioni e note) | ricerca nella memoria con l'`embed` dello stato | "cosa ti ricorda?" ([06-memoria.md](06-memoria.md) §7) |
| `known` (non tenuta) | già visto / simile / nuovo | soglie sulla similarità con la memoria | "è qualcosa che hai in memoria?" ([06-memoria.md](06-memoria.md) §7) |
| `surprise` (tolta, vedi §4) | quanto lo stato è inatteso | log-probabilità del modello sui token dello stato | anomalie in email o fotogrammi |
| incertezza | entropia, margine, profondità d'uscita, accordo fra permutazioni | già calcolati in parte (`depth`, `status`) | difficoltà della decisione |

### Con una testa addestrata

| Primitiva | Restituisce | Come | Esempio |
|---|---|---|---|
| `span` | un tratto di testo dello stato | puntatore inizio/fine sulle posizioni (QA estrattivo, GLiNER) | numero di fattura, senza generare |
| `locate` | zona dell'immagine | puntatore sui token dell'immagine (riquadri di 32×32 px) | "dov'è l'incendio?" |
| `value` | numero continuo con quantili | testa di regressione | importi, tempi, età |
| `tags` | un'etichetta per token | classificazione per token (come Qwen3Guard-Stream) | dati personali, frasi offensive |
| `why` | parti dello stato decisive | attenzioni o gradienti dalla posizione di lettura | spiegazione senza testo |

## 2. Confronto con Jev e Laya (verificato il 26/09/2026)

Jev (`jev-1.13.0`, docs.typesafe.ai) ha **solo** `noul`, `choice` e `score` su un unico endpoint `/v1/systemone`. Lo stato è solo testo o JSON. Per scelta dichiarata non restituisce stringhe né strutture sequenziali, così tutti gli output sono calcolabili in parallelo. Non ha soglie di confidenza per domanda: l'astensione va fatta nel codice del cliente.

| Capacità | Jev | Laya | Egeria oggi | Egeria possibile |
|---|---|---|---|---|
| `noul`, `choice`, `score` | ✓ | ✓ | ✓ | ✓ |
| Soglia per domanda / astensione | ✗ | testa act/escalate non funzionante | ✓ `min_confidence` + `status` | ✓ |
| Profondità dinamica | ✗ | ✗ | ✓ | ✓ |
| Stato con immagini | ✗ | ✗ | prototipo | ✓ |
| `multi`, `rank`, `number`, `open` | ✗ (solo il valore atteso di `score`) | ✗ | – | ✓ zero-shot |
| `embed`, `surprise` | ✗ | ✗ | – | ✓ zero-shot |
| `span`, `locate`, `value`, `tags` | ✗ | ✗ | – | con training |
| Fine-tuning del cliente | ✗ | ✓ | ✓ | ✓ |
| Esecuzione locale | ✗ (cloud) | ✓ | ✓ | ✓ |

## 3. Priorità proposte

1. **`embed` per il video in tempo reale.** Il vettore del fotogramma è gratuito nello stesso passaggio: si fanno le domande solo quando la scena cambia.
2. **`locate` sulle immagini.** "Sì, qui" è molto più utile di "sì".
3. **`span`** per estrarre valori dai documenti senza generazione.
4. **`open`**: economica e zero-shot. (`multi` rinviata alla F1.)

## 4. Implementazione e verifica (zero-shot)

| Tipo | Come è fatto | Risposta |
|---|---|---|
| `rank` | readout a lettere come `choice` | `ranking`, `probabilities`, `confidence` |
| `number` | readout ordinale come `score`, sugli intervalli; `null` = estremo aperto | `value` (valore atteso sui punti medi), `interval` (10–90%, uniforme a tratti), `range`, `probabilities`, `unit` |
| `open` | softmax sull'intero vocabolario (token speciali esclusi). Le candidate si completano fino al confine di parola: un prefill + ≤4 token in greedy, cache duplicata | `answer`, `candidates` con `p` (probabilità del primo token), `confidence` |
| `embed` | media degli hidden finali sui token dello stato, normalizzata | `state.embedding`, `state.embedding_dim` |

**Verifica** ([scripts/verifica_primitive.py](../scripts/verifica_primitive.py), 2B-Base, soglie decise prima dei risultati). Si tiene una primitiva solo se supera la soglia.

| Primitiva | Prova | Soglia | Risultato | Esito |
|---|---|---|---|---|
| `rank` | 4 situazioni con azione prioritaria ovvia | ≥ 3/4 | **4/4** | tenuta |
| `number` | 5 quantità scritte nel testo (giorni, persone, euro, kg, minuti) | ≥ 4/5 nell'intervallo | **5/5** (15.4 persone, 1.303 €, 7.7 kg, 39 min) | tenuta |
| `open` | 6 risposte di una parola (giorno, lingua, città, colore, mese, cognome) | ≥ 5/6 | **6/6** | tenuta |
| `embed` | recupero di casi con le stesse decisioni (kNN su typed-decisions) | meglio della Prior | **0.564** contro 0.478, come Qwen3-Embedding-0.6B | tenuta |
| `surprise` | 6 testi normali contro 6 anomali | AUROC ≥ 0.85 | **0.50** | **tolta** |
| `multi` | ticket e foto | – | 0.8B: "sì" a quasi tutto (bias verso il sì, non correggibile con la temperatura) | **tolta** |

**Perché `surprise` non funziona.** La perplessità misura quanto un testo è *prevedibile*, non quanto è *anomalo* per il caso d'uso:
- il *Lorem ipsum* ha perplessità 2.9, perché il modello lo conosce benissimo;
- una prompt injection in buon italiano ha 12.7, meno di molti ticket normali (18–70);
- solo il testo casuale e le parole mescolate risultano "sorprendenti".

**Limiti noti delle primitive tenute:**
- `open` risponde con una sola "parola" o un valore. La probabilità riguarda il primo token; il completamento prosegue fino al primo spazio (entro 16 token), così date, importi e codici escono interi ("gg.mm.aaaa", "14,21"). Funziona anche con le immagini ([04-stato-con-immagini.md](04-stato-con-immagini.md) §7).
- `embed` costa un passaggio in più sullo stato.

## 5. Esempi d'uso

Tutti gli esempi sono in [examples/primitive/](../examples/primitive/). Gli output qui sotto sono **reali** (Qwen3.5-2B-Base, 2 permutazioni, RTX 4070 Laptop). Una richiesta può contenere domande di tipi diversi insieme ([examples/primitive_it.json](../examples/primitive_it.json)).

### `rank`: ordinare le opzioni

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/primitive/rank.json
```

Richiesta (lo stato è il ticket di [examples/ticket_it.json](../examples/ticket_it.json)):

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ai nostri fornitori falliscono ...",
 "questions": {"priorita": {"type": "rank", "instructions": "Quale azione va fatta per prima?",
   "criteria": {"riparare_pagamenti": "Risolvere il problema dei pagamenti",
                "offrire_sconto": "Offrire uno sconto commerciale",
                "inviare_newsletter": "Inviare la newsletter mensile"}}}}
```

Risposta:

```json
"priorita": {"type": "rank",
  "ranking": ["riparare_pagamenti", "inviare_newsletter", "offrire_sconto"],
  "probabilities": {"riparare_pagamenti": 0.9948, "offrire_sconto": 0.0023, "inviare_newsletter": 0.0029},
  "confidence": 0.9922}
```

**Come si legge.** `ranking` è l'ordine dalla più alla meno indicata. Le probabilità dicono quanto è netta la prima scelta; qui le altre due sono quasi a pari merito, quindi il loro ordine relativo non è informativo.

### `number`: stimare una quantità

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/primitive/number.json
```

```json
{"state": "Riunione di progetto: 12 persone in sala A e altre 3 collegate da remoto. Due colleghi hanno avvisato che arriveranno in ritardo.",
 "questions": {"partecipanti": {"type": "number",
   "instructions": "Quante persone partecipano in totale alla riunione, contando anche chi è collegato?",
   "criteria": {"bins": [0, 5, 10, 14, 16, 20, null], "unit": "persone"}}}}
```

- `bins` sono gli estremi degli intervalli, in ordine crescente. `null` come ultimo (o primo) estremo significa "o più" (o "meno di").
- Si può passare anche solo la lista: `"criteria": [0, 5, 10, 14, 16, 20, null]`.

```json
"partecipanti": {"type": "number", "value": 14.59, "interval": [10.86, 18.97], "range": "14-16",
  "probabilities": {"0-5": 0.014, "5-10": 0.028, "10-14": 0.273, "14-16": 0.460, "16-20": 0.168, ">=20": 0.057},
  "confidence": 0.565, "unit": "persone"}
```

**Come si legge:**
- `range` è l'intervallo più probabile (la risposta giusta è 15: corretto);
- `value` è la stima puntuale (valore atteso);
- `interval` contiene la risposta con l'80% di probabilità.

Gli intervalli vanno scelti in base alla precisione che serve: più sono stretti, più la domanda è difficile.

### `open`: risposta di una parola, senza elencare le opzioni

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base examples/primitive/open.json
```

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ... Abbiamo stipendi da pagare venerdì. ...",
 "questions": {
   "giorno": {"type": "open", "instructions": "Entro quale giorno della settimana vanno pagati gli stipendi?", "top_k": 3},
   "lingua": {"type": "open", "instructions": "In che lingua è scritto il messaggio?", "top_k": 3}}}
```

```json
"giorno": {"type": "open", "answer": "Venerdì",
  "candidates": [{"text": "Venerdì", "p": 0.779}, {"text": "domenica", "p": 0.061}, {"text": "Giovedì", "p": 0.034}],
  "confidence": 0.779},
"lingua": {"type": "open", "answer": "Italiano",
  "candidates": [{"text": "Italiano", "p": 0.913}, {"text": "Italieno", "p": 0.040}, {"text": "IT", "p": 0.018}],
  "confidence": 0.913}
```

**Come si legge.** `answer` è la parola più probabile; `candidates` le alternative con la loro probabilità (`top_k`, da 1 a 20).
- **Adatto a:** estrarre un nome, un giorno, una città, un colore, una lingua.
- **Non adatto a:** risposte di più parole o concetti astratti. Per quelli conviene `choice` con le opzioni.

### `embed`: vettore semantico dello stato

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base examples/primitive/embed.json
```

```json
{"state": "Buongiorno, sono tre giorni che i pagamenti ...", "embed": true}
```

```json
"answers": {},
"state": {"embedding": [0.0239, -0.0038, -0.0358, -0.0114, ...], "embedding_dim": 2048}
```

**Come si usa:**
- `questions` può mancare se si chiede solo `embed`;
- il vettore è normalizzato: la similarità fra due stati è il prodotto scalare;
- per un archivio di casi, conviene prima sottrarre la media dei vettori (anisotropia).

La memoria dei casi usa proprio questo vettore ([06-memoria.md](06-memoria.md)).

### Da Python

```python
import json
from egeria.calibration import load_temperatures
from egeria.scorer import DecisionScorer

scorer = DecisionScorer("Qwen/Qwen3.5-2B-Base")  # device="cpu", dtype="float32" per il CPU
temperatures, exit_temperatures = load_temperatures("runs/Qwen3.5-2B-Base/temperature.json")
body = json.load(open("examples/primitive_it.json"))
response = scorer.decide(body, permutations=2, temperatures=temperatures)
print(response["answers"]["priorita"]["ranking"], response["answers"]["giorni"]["value"])
```

Le temperature si applicano per readout: `rank` usa quella delle `choice`, `number` quella delle `score`. `open` non è calibrata.
