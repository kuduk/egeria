**Italiano** · [English](en/04-image-states.md)

# Stato con immagini

> Stato: **integrato nell'API** (`egeria decide` con lo stato come lista di parti testo/immagine, §6). **26/26** sui 2B nella suite di prova, anche su foto reali ([scripts/demo_immagini.py](../scripts/demo_immagini.py)).

## 1. Perché è possibile

I Qwen3.5, anche 0.8B e 2B, sono **multimodali nativi**:
- l'architettura è `Qwen3_5ForConditionalGeneration`;
- la torre visiva è un ViT a 24 layer con hidden 1024, già inclusa nei pesi;
- un processore converte le immagini in token visivi inseriti nella sequenza.

Il readout decisionale non cambia: l'immagine entra nello stato, e alla fine si leggono i logit delle lettere nell'ultima posizione.

Jev e Laya accettano solo testo. Fra le repliche aperte solo decider ha una variante "vision" (2B). Uno stato con immagini (documenti scansionati, fatture, foto di prodotti o di danni, screenshot, cartelli) è quindi un **elemento di differenziazione**.

## 2. Come funziona il prototipo

1. Si carica il modello completo e `AutoProcessor`. La memoria GPU del 2B in bf16 è 4.4 GB.
2. Il prompt è lo stesso formato testuale della F0. Nello stato c'è il segnaposto `<|vision_start|><|image_pad|><|vision_end|>`, e il testo si costruisce con il chat template **del tokenizer**: il processore dei modelli Base non ha un chat template.
3. `processor(text=[prompt], images=[img])` espande il segnaposto nei token visivi e prepara `pixel_values` e `image_grid_thw`.
4. `model.model(**inputs)` restituisce l'hidden finale. Il readout usa le righe dell'lm_head delle lettere, con 2 permutazioni.

## 3. Risultati

**Suite:** [examples/suite_immagini.jsonl](../examples/suite_immagini.jsonl), 26 domande su 9 immagini, con 2 permutazioni delle opzioni.
- 3 immagini sintetiche: forma colorata, cartello di pericolo, fattura.
- 6 foto e screenshot **reali** da Wikimedia Commons, con licenze libere; autori e licenze in [examples/immagini/FONTI.md](../examples/immagini/FONTI.md):
  - scontrino olandese fotografato e piegato;
  - email di phishing (domain slamming) in Gmail;
  - auto incidentata;
  - segnale di stop con cartello della via;
  - casa in fiamme;
  - gatto che dorme, usato come controllo "innocuo" con le **stesse opzioni d'azione** dell'incendio.

| Modello | Corrette | Errori |
|---|---|---|
| **Qwen3.5-2B-Base** | **26/26** | – |
| **Qwen3.5-2B** | **26/26** | – |
| Qwen3.5-0.8B-Base | 21/26 | fattura "corrisponde" (sbagliato), scontrino "supera 50 €" (sì), incidente "ribaltamento", gatto "pericolo sì" e "monitorare" |

**Esempi dal 2B-Base** (probabilità della risposta):
- **Estrazione da uno scontrino reale, piegato e fotografato:**
  - totale "14,21": 1.00;
  - pagamento con PIN: 1.00;
  - città "Rotterdam": 1.00;
  - "la spesa supera 50 €? no": 0.68. Il confronto numerico è meno sicuro dell'estrazione.
- **Email:**
  - oggetto: 1.00;
  - "chiede di cliccare un link per aggiornare dati? sì": 0.97.
- **Incidente:** "urto contro un ostacolo fisso": 0.90; colore "bianca": 1.00.
- **Stop:** segnale 1.00; nome della via "via XXV Aprile" letto dal cartello: 1.00.
- **Stessa domanda d'azione, due immagini:**
  - incendio → "chiamare subito i vigili del fuoco" (0.97), gravità "alta" (0.99);
  - gatto → "nessuna azione" (0.87), "c'è pericolo? no" (0.99).

**Cosa emerge:**
1. **Sulle immagini i 2B zero-shot sono molto forti**, al contrario del benchmark testuale typed-decisions, dove restavano al livello della Prior. Queste domande sono di **percezione ed estrazione** (leggere un valore, riconoscere un oggetto o una situazione evidente), in cui i modelli visione-linguaggio sono addestrati a fondo. typed-decisions chiede invece di allinearsi a una politica di decisione implicita su casi ambigui.
2. **Il 0.8B cede sui confronti e sui giudizi** (importo diverso, soglia di 50 €, pericolo), non sulla lettura: legge bene totale, fornitore, città e via.
3. **La latenza dipende dalla risoluzione.** A 800 px lo scontrino in verticale diventa ~1.400 token visivi: 549 ms sul 2B, 289 ms sul 0.8B. Le immagini piccole costano 60–100 ms. Ridurre la risoluzione massima del processore (`max_pixels`) e calcolare la torre visiva una volta per richiesta sono le due ottimizzazioni principali.
4. **Cautela:** 26 domande scelte per avere una risposta netta non sono un benchmark. Servono casi ambigui, documenti lunghi e immagini di scarsa qualità, oltre alla calibrazione sulle immagini.

## 4. Verso il tempo reale

**Scenario.** Un fotogramma alla volta, 3 domande per fotogramma nello stesso batch (incendio in corso? azione? gravità?), 1 permutazione, bf16, RTX 4070 Laptop. Il tempo comprende la preparazione dell'immagine sul CPU (ridimensionamento e normalizzazione) e il forward.

**Risoluzione** (prima dei kernel veloci):

| Lato massimo | Token visivi | 0.8B-Base | 2B-Base |
|---|---|---|---|
| 800 px | 475 | 274 ms (3.6 fps) | 476 ms (2.1 fps) |
| 448 px | 140 | 101 ms (9.9 fps) | 166 ms (6.0 fps) |
| 224 px | 70 | 80 ms (12.5 fps) | 122 ms (8.2 fps) |

Le risposte sono corrette a tutte le risoluzioni. Sotto i 448 px il tempo quasi non scende più: è un "pavimento" dovuto al numero di kernel, oltre 3.000 per forward, non alla quantità di calcolo.

**Da dove veniva il pavimento** (profiler, 0.8B, 224 px, forward di 62 ms):
- la torre visiva prende 15 ms, perché **la stessa immagine viene codificata 3 volte**, una per domanda;
- la parte linguistica prende 47 ms, dominata dai kernel del fallback PyTorch dei DeltaNet: risoluzioni triangolari `batch_trsm` e `sgemm` in fp32, convoluzione depthwise, centinaia di kernel elementwise.

**Effetto dei kernel veloci** (forward, 224 px):

| | Fallback PyTorch | + flash-linear-attention | + causal-conv1d | + `torch.compile` (reduce-overhead) |
|---|---|---|---|---|
| 0.8B-Base | 77 ms | 68 ms | **45 ms** | 49 ms |
| 2B-Base | 116 ms | 101 ms | **91 ms** | 94 ms |

**Cosa emerge:**
- **Con entrambi i kernel veloci:** circa 51 ms per fotogramma sul 0.8B (~20 fps) e circa 97 ms sul 2B (~10 fps), preparazione compresa.
- **`torch.compile` con CUDA graphs** aiutava senza kernel veloci (77 → 50 ms), ma con i kernel custom il grafo si spezza e non porta più guadagno. La compilazione costa 15–140 s una tantum.
- **La build di causal-conv1d** compila da sorgente in circa 6 minuti per torch 2.11 con `TORCH_CUDA_ARCH_LIST="8.9"`: non esiste una wheel precompilata.

**Immagine una volta contro immagine per ogni domanda** ([scripts/bench_prefisso_immagine.py](../scripts/bench_prefisso_immagine.py)). Nelle misure sopra, **ogni domanda riprocessava l'immagine**: batch di 3 prompt completi, ognuno con la sua copia. La variante condivisa:
1. calcola istruzioni + immagine **una volta**, con la cache;
2. duplica la cache ibrida per le domande: stato DeltaNet + conv + KV dei layer di attenzione;
3. passa in batch solo le code (domanda + opzioni), con posizioni M-RoPE = indice + `rope_deltas`.

Le risposte sono identiche e le probabilità coincidono entro il rumore bf16.

| Modello, lato immagine | 3 prompt completi (immagine ×3) | Immagine + prefisso una volta, 3 code | 1 sola domanda |
|---|---|---|---|
| 0.8B, 224 px | **51 ms** | 80 ms | 42 ms |
| 0.8B, 448 px | **64 ms** | 78 ms | 44 ms |
| 0.8B, 800 px | 181 ms | **97 ms** (−46%) | 61 ms |
| 2B, 224 px | 96 ms | **92 ms** | 54 ms |
| 2B, 448 px | 133 ms | **96 ms** (−28%) | 59 ms |
| 2B, 800 px | 392 ms | **166 ms** (−58%) | 130 ms |

**Cosa emerge:**
- **Con immagini piccole la ripetizione costa poco.** Sul 0.8B a 224 px le 2 domande in più, immagine compresa, costano 9 ms (42 → 51 ms). La variante condivisa fa **due passaggi in sequenza** (prefisso, poi code), quindi paga due volte il costo fisso per passaggio, circa 35–40 ms di piccoli kernel sul 0.8B.
- **Con immagini grandi o più domande la condivisione vince nettamente**, fino a −58%, perché ogni domanda in più costa solo la sua coda. Dal 29/09/2026 lo scorer sceglie da solo, in base ai token evitati: con le immagini della suite il tempo scende fino all'80% ([08-riuso-dello-stato.md](08-riuso-dello-stato.md)).
- **Il vero limite è il costo fisso per passaggio**: 42 ms per una domanda sul 0.8B, di cui solo 8 ms di torre visiva.

**Prossimi passi per il tempo reale:**
1. **CUDA graphs sul nostro ciclo dei layer.** `torch.compile` si spezza sui kernel custom. In streaming però le forme sono fisse, quindi si può catturare a mano il forward dello scorer, con maschere e rotary precalcolate, per il prefisso e per le code. È l'attacco diretto al costo fisso.
2. **Profondità dinamica** (abbandonata il 29/09/2026). In un flusso video la maggior parte dei fotogrammi è "non succede nulla", con risposte molto sicure (il gatto: "nessuna azione", "pericolo: no" a 0.99): sarebbero stati i casi ideali per uscire presto. Zero-shot però il guadagno reale era dello 0.5–4% ([03-baseline-f0.md](03-baseline-f0.md) §6), e il codice è stato tolto.
3. **Stato ricorrente fra fotogrammi.** I layer Gated DeltaNet hanno uno stato di dimensione fissa. In linea di principio si potrebbero processare solo i token del fotogramma nuovo, mantenendo lo stato: resterebbe da gestire la cache dei layer full-attention con una finestra. È da esplorare.
4. **Cascata System 1 → System 2.** Le decisioni per fotogramma girano a 10–20 fps sul modello piccolo, e solo gli eventi `uncertain` o rilevanti vanno a un modello grande.

## 5. Integrazione nell'API

- **Stato.** È una **lista di parti**: `{"type": "text", "text": ...}` e `{"type": "image", "path" | "url" | "base64": ...}`. Più immagini sono ammesse e il loro ordine viene rispettato nel prompt.
- **`image_max_side`** (opzionale, 64–4096): riduce le immagini, così i token visivi calano e la latenza scende (§4).
- **Caricamento del modello.** `egeria decide` riconosce le immagini nello stato e carica da solo il modello completo con la torre visiva (`--vision` lo forza). La memoria del 2B in bf16 è 4.4 GB.
- **Tipi di domanda supportati:** `noul`, `choice`, `score`, `estimate` e `short_answer`, con le permutazioni e `min_confidence` (sotto la soglia la risposta è `uncertain`).
- **La memoria e il vettore dello stato (`/v1/embed`) funzionano anche con le immagini:** il vettore include i token visivi. Verifica: 5/6 in [06-memoria.md](06-memoria.md) §6.
- **`short_answer` (una risposta breve: una parola o un valore) funziona anche con le immagini** e legge date, importi e codici (§7).
- **Efficienza.** Dal 29/09/2026 istruzioni e immagini si calcolano una volta sola per tutte le domande e permutazioni, quando conviene (modalità `auto`, [08-riuso-dello-stato.md](08-riuso-dello-stato.md)). Con immagini piccole e poche domande resta il prompt intero per ogni domanda, che lì costa meno.

## 6. Esempi d'uso

### Sinistro auto ([examples/immagini_sinistro.json](../examples/immagini_sinistro.json))

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --permutations 2 examples/immagini_sinistro.json
```

```json
{
  "state": [
    {"type": "text", "text": "Foto allegata dal cliente alla denuncia di sinistro n. 2026-0913:"},
    {"type": "image", "path": "examples/immagini/incidente_auto.jpg"}
  ],
  "image_max_side": 448,
  "questions": {
    "danneggiato": {"type": "noul", "instructions": "Il veicolo nella foto è danneggiato?"},
    "dinamica": {"type": "choice", "instructions": "Che tipo di incidente mostra la foto?",
                 "criteria": {"urto_ostacolo": "Urto contro un ostacolo fisso", "tamponamento": "Tamponamento fra veicoli",
                              "ribaltamento": "Ribaltamento del veicolo", "nessun_incidente": "Nessun incidente visibile"}},
    "gravita": {"type": "score", "instructions": "Quanto è grave il danno visibile?",
                "criteria": ["Nessun danno", "Lieve (graffi)", "Moderato (carrozzeria deformata)", "Grave (veicolo distrutto)"]}
  }
}
```

Risposta reale (2B-Base, 2.3 s compreso il caricamento dell'immagine):

```json
"danneggiato": {"type": "noul", "noul": 0.984, "confidence": 0.968},
"dinamica": {"type": "choice", "choice": "tamponamento",
  "probabilities": {"urto_ostacolo": 0.477, "tamponamento": 0.478, "ribaltamento": 0.015, "nessun_incidente": 0.030},
  "confidence": 0.305},
"gravita": {"type": "score", "score": 2.11,
  "probabilities": {"0": 0.014, "1": 0.043, "2": 0.760, "3": 0.182}, "confidence": 0.746}
```

**Come si legge:**
- `danneggiato`: sì, con alta confidenza, corretto.
- `gravita`: moderato, corretto (carrozzeria deformata).
- `dinamica`: pareggio fra "urto contro un ostacolo fisso" e "tamponamento", con confidenza bassa (0.30). L'ambiguità è reale: sullo sfondo, dietro le siepi, c'è un'altra auto. Il modello *segnala* di non saperlo distinguere. Con `"min_confidence": 0.5` quella domanda risulterebbe `uncertain`, cioè da mandare a un umano.

**Domande atomiche invece di una domanda composta.** È il pattern "harness" consigliato anche da TypeSafe: domande semplici al modello, logica nel codice. Sulla stessa foto:

| Domanda atomica (`noul`) | P(sì) | Confidenza |
|---|---|---|
| L'auto bianca in primo piano ha urtato con il muso un palo o un oggetto fisso? | 0.84 | 0.67 |
| Ha danni visibili nella parte posteriore? | 0.06 | 0.87 |
| Si vede un contatto fra l'auto bianca e un altro veicolo? | 0.48 | 0.03 |
| Nella foto è visibile anche un altro veicolo, sullo sfondo? | 0.86 | 0.73 |

**Cosa ne esce:**
- Il modello vede l'altra auto, ma riconosce l'urto frontale contro un oggetto fisso senza danni posteriori.
- Una regola nel codice ("urto frontale su oggetto fisso e nessun danno posteriore → urto contro ostacolo") dà la dinamica giusta.
- L'unico dubbio reale, il contatto fra i due veicoli, resta esplicito, con confidenza 0.03.

### Fotogramma di una telecamera ([examples/immagini_incendio.json](../examples/immagini_incendio.json))

```json
{
  "state": [{"type": "text", "text": "Fotogramma dalla telecamera 3:"},
            {"type": "image", "path": "examples/immagini/casa_incendio.jpg"}],
  "image_max_side": 336,
  "min_confidence": 0.6,
  "questions": {
    "incendio": {"type": "noul", "instructions": "C'è un incendio in corso?"},
    "azione": {"type": "choice", "instructions": "Quale azione è più appropriata?",
               "criteria": {"chiamare_vigili": "Chiamare subito i vigili del fuoco",
                            "monitorare": "Monitorare la situazione", "nessuna": "Nessuna azione"}}
  }
}
```

```json
"incendio": {"type": "noul", "noul": 0.969, "confidence": 0.938, "min_confidence": 0.6, "status": "decided"},
"azione": {"type": "choice", "choice": "chiamare_vigili",
  "probabilities": {"chiamare_vigili": 0.942, "monitorare": 0.028, "nessuna": 0.030},
  "confidence": 0.913, "min_confidence": 0.6, "status": "decided"}
```

**Immagini da URL o in base64** (al posto di `path`):

```json
{"type": "image", "url": "https://upload.wikimedia.org/.../foto.jpg"}
{"type": "image", "base64": "iVBORw0KGgoAAAANSUhEUgAA..."}
```

**Suite di prova con risposte attese.** Formato JSONL più semplice, una domanda per riga, per confrontare i modelli:

```bash
.venv/bin/python scripts/demo_immagini.py Qwen/Qwen3.5-2B-Base examples/suite_immagini.jsonl
```

```json
{"image": "examples/immagini/incidente_auto.jpg", "question": "Il veicolo è danneggiato?", "options": ["sì", "no"], "expected": "sì"}
```

## 7. Leggere contro ragionare

Una prova su una **carta d'identità** fotografata (i dati personali non sono riportati qui) e su uno scontrino. Modello: Qwen3.5-2B-Base, senza calibrazione.

| Domanda | Tipo | Risultato |
|---|---|---|
| "Vi è una persona nella foto?" | Sì/No | ✅ Sì, 93–94% |
| "È un documento di identità?" | Sì/No | ✅ Sì, 94–95% |
| Anno di nascita (scelta fra anni vicini) | scelta | ✅ corretto, 99% |
| Data di nascita | valore (`short_answer`) | ✅ letta per intero, 92% a 448 px, 97% a 896 px |
| Data di scadenza | valore (`short_answer`) | ✅ letta per intero, 98–99% |
| Totale dello scontrino | valore (`short_answer`) | ✅ "14,21", 98–99% |
| "La persona ha più di 21 anni?" | Sì/No | ❌ No al 31% (448 px); 50% a 896 px |
| Stessa domanda, con la data di oggi nel testo | Sì/No | ⚠️ 57–62% |
| "È nata prima del 2005?" | Sì/No | ❌ No al 39%, pur leggendo correttamente l'anno |

**Il modello legge molto bene ma non sa fare confronti e calcoli** in un solo passaggio. Per "più di 21 anni?" dovrebbe:
1. leggere la data di nascita;
2. conoscere la data di oggi, che non sa;
3. calcolare l'età;
4. confrontarla.

È lo stesso limite documentato per Jev (date, aritmetica).

**Regola pratica:** si chiede al modello il **valore** ("Qual è la data di nascita?", tipo *Una risposta breve*) e il confronto si fa nel codice o a mano. Per leggere testi piccoli conviene la qualità immagine *Massima* (896 px).

**Correzioni fatte dopo questa prova:**
- *Una risposta breve* ora funziona anche con le immagini.
- La lettura non si ferma più alla punteggiatura: il completamento prosegue fino al primo spazio, entro 16 token, perché il tokenizer di Qwen spezza le cifre una per una.
- La calibrazione non viene più applicata alle immagini, e il server di default non la usa. Con quella dei ticket di testo, "Vi è una persona nella foto?" scendeva al 59%.
