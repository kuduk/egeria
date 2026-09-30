**Italiano** · [English](en/05-reading-primitives.md)

# Primitive di lettura oltre noul, choice e score

> Stato: implementate e **verificate** `number` e `open` (§4), più il vettore semantico dello stato, che oggi si chiede solo con `POST /v1/embed` del server del modello.
>
> **Tolte:**
> - il 29/09/2026, per semplificare: `rank`, che è una `choice` con le opzioni ordinate per probabilità, ed `embed` dentro `/v1/systemone`;
> - prima, perché non hanno superato la prova: `multi` (bias verso il "sì") e `surprise` (AUROC 0.50). Il 29/09/2026 `multi` è stata abbandonata anche come obiettivo per il training: equivale a N domande Sì/No.
>
> Quelle con training (`span`, `locate`, `value`, `tags`, `why`) sono da fare. Esempi d'uso nella §5.

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
| `rank` (tolta il 29/09/2026) | ordinamento completo con probabilità | distribuzione della `choice`: basta ordinarla | riordinare documenti o azioni |
| `number` (a intervalli) | valore atteso + intervallo | `score` con intervalli numerici come livelli | quante persone nell'immagine |
| `open` | top-k su tutto il vocabolario | softmax sull'lm_head completo nell'ultima posizione | colore dell'auto, senza elencare le opzioni (risposte di un solo token) |
| vettore dello stato (`/v1/embed`) | vettore semantico dello stato | hidden finale (come Qwen3-Embedding) | ricerca di casi simili, deduplica, cambio di scena nel video |
| ricordi simili (`memory.recall`) | i ricordi più simili (con decisioni e note) | ricerca nella memoria con il vettore dello stato | "cosa ti ricorda?" ([06-memoria.md](06-memoria.md) §7). Come tipo di domanda `recall` è stata tolta il 29/09/2026 |
| `known` (non tenuta, abbandonata il 29/09/2026) | già visto / simile / nuovo | soglie sulla similarità con la memoria | "è qualcosa che hai in memoria?" ([06-memoria.md](06-memoria.md) §7) |
| `surprise` (tolta, vedi §4) | quanto lo stato è inatteso | log-probabilità del modello sui token dello stato | anomalie in email o fotogrammi |
| incertezza | entropia, margine, accordo fra permutazioni | già calcolata in parte (`status` rispetto a `min_confidence`) | difficoltà della decisione |

### Con una testa addestrata

`span`, `locate` e `why` sono in roadmap anche in una versione **zero-shot**, senza testa addestrata (§3 e [02-implicazioni-e-proposta.md](02-implicazioni-e-proposta.md) §6).

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
| Profondità dinamica | ✗ | ✗ | tolta il 29/09/2026 (guadagno reale zero-shot 0.5–4%) | – |
| Stato con immagini | ✗ | ✗ | ✓ | ✓ |
| `number`, `open` | ✗ (solo il valore atteso di `score`) | ✗ | ✓ zero-shot | ✓ |
| Vettore dello stato e memoria | ✗ | ✗ | ✓ (`/v1/embed`, `memory`) | ✓ |
| `span`, `locate`, `value`, `tags` | ✗ | ✗ | – | con training |
| Fine-tuning del cliente | ✗ | ✓ | ✓ | ✓ |
| Esecuzione locale | ✗ (cloud) | ✓ | ✓ | ✓ |

## 3. Priorità proposte

1. **Il vettore dello stato per il video in tempo reale.** Il vettore del fotogramma è gratuito nello stesso passaggio: si fanno le domande solo quando la scena cambia.
2. **`span`** (in roadmap, subito dopo F0.5): valori copiati alla lettera dallo stato, senza training. Il completamento di `open` accetta solo i token che continuano un pezzo presente nello stato, quindi la risposta non può essere inventata.
3. **`locate`** (in roadmap, con l'indagine su destra/sinistra): "sì, qui" è molto più utile di "sì". Zero-shot, dalle coordinate che i modelli visivi Qwen sanno produrre.
4. **`why`** (in roadmap, con l'interfaccia per gli operatori): le frasi dello stato che hanno deciso, togliendone una alla volta.
5. **`open`**: economica e zero-shot. (`multi` abbandonata il 29/09/2026: si usano N domande Sì/No.)

## 4. Implementazione e verifica (zero-shot)

| Tipo | Come è fatto | Risposta |
|---|---|---|
| `number` | readout ordinale come `score`, sugli intervalli; `null` = estremo aperto | `value` (valore atteso sui punti medi), `interval` (10–90%, uniforme a tratti), `range`, `probabilities`, `unit` |
| `open` | softmax sull'intero vocabolario (token speciali esclusi). Le candidate si completano fino al confine di parola: un prefill + ≤4 token in greedy, cache duplicata | `answer`, `candidates` con `p` (probabilità del primo token), `confidence` |
| vettore dello stato (`POST /v1/embed`) | media degli hidden finali sui token dello stato, normalizzata | `embedding`, `embedding_dim` |

**Verifica** ([scripts/verifica_primitive.py](../scripts/verifica_primitive.py), 2B-Base, soglie decise prima dei risultati). Si tiene una primitiva solo se supera la soglia.

| Primitiva | Prova | Soglia | Risultato | Esito |
|---|---|---|---|---|
| `rank` | 4 situazioni con azione prioritaria ovvia | ≥ 3/4 | **4/4** | tenuta, poi tolta il 29/09/2026: la stessa prova oggi si fa con `choice` |
| `number` | 5 quantità scritte nel testo (giorni, persone, euro, kg, minuti) | ≥ 4/5 nell'intervallo | **5/5** (15.4 persone, 1.303 €, 7.7 kg, 39 min) | tenuta |
| `open` | 6 risposte di una parola (giorno, lingua, città, colore, mese, cognome) | ≥ 5/6 | **6/6** | tenuta |
| `embed` | recupero di casi con le stesse decisioni (kNN su typed-decisions) | meglio della Prior | **0.564** contro 0.478, come Qwen3-Embedding-0.6B | tenuta (oggi con `/v1/embed` e la memoria) |
| `surprise` | 6 testi normali contro 6 anomali | AUROC ≥ 0.85 | **0.50** | **tolta** |
| `multi` | ticket e foto | – | 0.8B: "sì" a quasi tutto (bias verso il sì, non correggibile con la temperatura) | **tolta** |

**Perché `surprise` non funziona.** La perplessità misura quanto un testo è *prevedibile*, non quanto è *anomalo* per il caso d'uso:
- il *Lorem ipsum* ha perplessità 2.9, perché il modello lo conosce benissimo;
- una prompt injection in buon italiano ha 12.7, meno di molti ticket normali (18–70);
- solo il testo casuale e le parole mescolate risultano "sorprendenti".

**Limiti noti delle primitive tenute:**
- `open` risponde con una sola "parola" o un valore. La probabilità riguarda il primo token; il completamento prosegue fino al primo spazio (entro 16 token), così date, importi e codici escono interi (una data completa nel formato "gg.mm.aaaa", l'importo "14,21"). Funziona anche con le immagini ([04-stato-con-immagini.md](04-stato-con-immagini.md) §7).
- il vettore dello stato costa un passaggio in più sullo stato.

## 5. Esempi d'uso

Tutti gli esempi sono in [examples/primitive/](../examples/primitive/). Gli output qui sotto sono **reali** (Qwen3.5-2B-Base, 2 permutazioni, RTX 4070 Laptop). Una richiesta può contenere domande di tipi diversi insieme ([examples/primitive_it.json](../examples/primitive_it.json)).

> **`rank` non c'è più** (29/09/2026). Per ordinare delle opzioni si usa `choice`: restituisce già la probabilità di ognuna, e ordinarle è una riga di codice. La domanda "Quale azione va fatta per prima?" di [examples/primitive_it.json](../examples/primitive_it.json) ora è una `choice`.

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

### Vettore semantico dello stato: `POST /v1/embed`

Dal 29/09/2026 non si chiede più dentro `/v1/systemone` (`"embed": true`), ma con una chiamata a parte al server del modello:

```bash
curl -s localhost:8100/v1/embed -H 'Content-Type: application/json' \
  -d '{"state": "Buongiorno, sono tre giorni che i pagamenti ..."}'
```

```json
{"embedding": [0.0239, -0.0038, -0.0358, -0.0114, ...], "embedding_dim": 2048, "input_tokens": ..., "latency_ms": ...}
```

**Come si usa:**
- lo stato può contenere immagini (in base64), con `image_max_side` facoltativo;
- il vettore è normalizzato: la similarità fra due stati è il prodotto scalare;
- per un archivio di casi, conviene prima sottrarre la media dei vettori (anisotropia).

La memoria dei casi usa proprio questo vettore ([06-memoria.md](06-memoria.md)). Da Python lo stesso vettore si ottiene con `scorer.analyze_state(stato)`.

### Da Python

```python
import json
from egeria.calibration import load_temperatures
from egeria.scorer import DecisionScorer

scorer = DecisionScorer("Qwen/Qwen3.5-2B-Base")  # device="cpu", dtype="float32" per il CPU
temperatures = load_temperatures("runs/Qwen3.5-2B-Base/temperature.json")
body = json.load(open("examples/primitive_it.json"))
response = scorer.decide(body, permutations=2, temperatures=temperatures)
print(response["answers"]["priorita"]["choice"], response["answers"]["giorni"]["value"])
```

Le temperature si applicano per readout: `number` usa quella delle `score`. `open` non è calibrata.
