**Italiano** · [English](en/06-memory.md)

# Memoria: ricordi richiamati per somiglianza

> Stato: **implementata**, per testo e **immagini**. Il **recupero** funziona: il voto dei ricordi vale 0.56 contro 0.47 del modello zero-shot (testo), e con le foto il ricordo più simile è della categoria giusta in 5 casi su 6 (§6). Mettere i ricordi **nel prompt** zero-shot non migliora le decisioni, quindi è un'opzione spenta di default (`inject`).

## 1. Idea

La memoria è **un oggetto esterno al modello**: un archivio di casi passati con le decisioni prese (meglio se confermate da un esito o da una correzione), note e regole. Si aggiunge, si ispeziona e si cancella **senza riaddestrare nulla**.

Davanti a uno stato nuovo si recuperano i ricordi più simili tramite l'`embed` dello stato, gratuito nello stesso modello, e si restituiscono in output. Per ogni domanda che i ricordi coprono si restituisce anche il **voto dei ricordi**.

## 2. Il recupero funziona?

**Prova minima** (6 frasi): l'hidden dell'ultimo token è anisotropo, con coseno 0.91–0.97 per qualsiasi coppia. La **media sui token dello stato** separa molto meglio, quindi `embed` usa la media.

**Prova sulle decisioni** (typed-decisions): per ogni caso di test si recuperano i 10 casi di train più simili e si predice con le loro decisioni.

| Vettore / metodo | Accuratezza | NLL | ECE |
|---|---|---|---|
| **Qwen3.5-2B-Base (media sui token), voto con distribuzioni gold** | **0.564** | **0.923** | 0.057 |
| Qwen3.5-0.8B-Base (media sui token), idem | 0.564 | 0.930 | 0.052 |
| Qwen3-Embedding-0.6B (modello dedicato), idem | 0.552 | 0.934 | 0.033 |
| 2B-Base, voto con **decisioni secche**, 10 ricordi | **0.562** | 0.963 | 0.043 |
| 2B-Base, decisioni secche, 5 ricordi | 0.554–0.560 | 1.07 | ~0.10 |
| 2B-Base, decisioni secche, 3 ricordi | 0.538 | 1.22 | ~0.16 |
| Prior (nessun recupero) | 0.478 | 1.034 | 0.034 |
| *2B-Base zero-shot, senza ricordi* | *0.468* | *1.049* | *0.029* |

**Cosa emerge:**
- **Il vettore del nostro modello recupera casi utili** quanto un modello di embedding dedicato.
- **Con decisioni secche** (come nell'uso reale) l'accuratezza regge, ma per probabilità oneste servono **10 ricordi**: con 3 il voto è troppo sicuro.

## 3. Ricordi nel prompt: non aiuta, per ora

Esperimento: 2B-Base, 3 ricordi con le decisioni gold nel prompt prima dello stato, 2 permutazioni, confronto appaiato con lo stesso modello senza ricordi, sulle stesse 2.000 decisioni.

| | Δ accuratezza | Δ NLL |
|---|---|---|
| totale | +0.002 [−0.018, +0.022] | +0.009 |
| incidenti di sicurezza | **+0.074*** | **−0.041*** |
| choice | +0.050 [0.000, +0.101] | −0.011 |
| score | **−0.041*** | +0.024* |
| trace di agenti | **−0.058*** | +0.052* |

\* intervallo al 95% che esclude lo zero.

**Combinazione modello + voto dei ricordi**, con il peso fittato sul train: il peso ottimale del modello è **0**. Il voto da solo fa 0.562 contro 0.468, **+9.5 punti** [+6.0, +12.6], e NLL −0.119.

**Conclusione.** Su decisioni ripetitive come queste, i casi simili già decisi valgono più del modello zero-shot, e il 2B non sa ancora sfruttarli nel contesto. Per questo:
- di default la memoria **restituisce** i ricordi e il loro voto;
- l'iniezione nel prompt è opzionale (`"inject": true`) ed è da riprovare dopo il training (F1), insegnando al modello a usare i ricordi.

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

**Opzioni di `memory`:**
- `recall` (0–10): quanti ricordi restituire.
- `min_similarity`: soglia di similarità.
- `vote` (default `true`): voto per ogni domanda coperta. Usa i 10 ricordi più simili che hanno quella domanda, pesati con softmax(similarità / 0.05), più una smussatura del 10%.
- `inject` (default `false`): mette i ricordi anche nel prompt.

**Risposta reale** (2B-Base; archivio di prova con 3 ricordi, [examples/memoria_ticket.json](../examples/memoria_ticket.json)):

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

**Come si legge:**
- `memories` sono i casi più simili, con decisioni e note. Il primo (0.94) è il caso dei bonifici, e la sua nota ("blocco antifrode") è un'informazione che il modello da solo non ha.
- `answers[q].memory` è il voto dei ricordi (`support` = quanti ricordi coprono la domanda), da affiancare alla risposta del modello. Quando i due concordano la decisione è più solida; quando divergono, è un segnale per un umano.

## 5. Gestire l'archivio

```bash
M="--model Qwen/Qwen3.5-2B-Base --memory runs/mia-memoria"
# aggiungere un caso confermato (lo stato può essere un file JSON di richiesta o solo lo stato; - per stdin)
.venv/bin/egeria memory $M add --id caso-101 --state examples/ticket_it.json \
  --decisions '{"reparto": "pagamenti", "urgente": "true"}' --note "IBAN corretto: era un blocco antifrode della banca"
# cercare i più simili
.venv/bin/egeria memory $M search --state examples/memoria_ticket.json -k 3
# decidere con la memoria
.venv/bin/egeria decide $M --permutations 2 examples/memoria_ticket.json
# archivio da un dataset (typed-decisions train, decisioni = etichette gold)
.venv/bin/egeria memory --model Qwen/Qwen3.5-2B-Base --memory runs/memoria-Qwen3.5-2B-Base build
```

L'archivio è una cartella con `memories.jsonl` (ricordi leggibili) e `vectors.npy` (vettori). Il vettore dipende dal modello: se si cambia modello, l'archivio va ricostruito.

**Cosa salvare.** Solo decisioni **confermate** (esiti, correzioni umane). Salvare le risposte del modello senza verifica ne rinforzerebbe gli errori; per questo `decide` non scrive mai nella memoria.

**Valutazione riproducibile:** `scripts/eval_memoria.sh` (ricordi nel prompt, con `--inject`).

## 6. Memoria con le immagini

Gli stati possono contenere immagini ([04-stato-con-immagini.md](04-stato-con-immagini.md)), e la memoria funziona allo stesso modo. Il vettore è la media degli hidden sui token dello stato, **token visivi compresi**, calcolati dal modello multimodale. Così si archiviano foto (sinistri, documenti, fotogrammi) con le decisioni prese, e si recuperano le più simili a una foto nuova.

```bash
M="--model Qwen/Qwen3.5-2B-Base --memory runs/memoria-foto --image-max-side 448"
echo '{"state": [{"type": "text", "text": "Immagine ricevuta:"}, {"type": "image", "path": "examples/immagini/casa_incendio.jpg"}]}' \
  | .venv/bin/egeria memory $M add --id foto-incendio --state - --decisions '{"azione": "chiamare_vigili"}'
```

Nelle richieste si usa lo stesso `image_max_side` con cui si sono archiviate le foto: il vettore dipende dalla risoluzione.

**Verifica** ([scripts/verifica_rag_immagini.py](../scripts/verifica_rag_immagini.py), 2B-Base, soglia decisa prima: ≥ 5/6).
- **Archivio:** le 6 foto reali di `examples/immagini/`, ognuna con categoria e azione.
- **Interrogazioni:** 6 foto **nuove** da Wikimedia, delle stesse categorie ma diverse:

| Foto nuova | Ricordo più simile | Similarità | Voto sull'azione | Modello zero-shot (categoria) |
|---|---|---|---|---|
| scontrino di un altro supermercato, in inglese | ✓ scontrino | 0.83 | ✓ archiviare | ✓ |
| incidente con uno scooter a terra | ✓ incidente auto | 0.88 | ✓ aprire sinistro | ✓ |
| segnale di stop storico | ✓ stop | 0.85 | ✓ nessuna | ✓ |
| altro incendio | ✓ incendio | 0.86 | ✓ chiamare i vigili | ✓ |
| altro gatto | ✓ gatto | 0.91 | ✓ nessuna | ✓ |
| email di phishing in sloveno, da smartphone | ✗ scontrino (0.62; phishing 0.58) | – | ✗ archiviare | ✓ phishing |

**Esito: 5/6, tenuta.**

**Cosa emerge dall'errore:**
- **Il vettore di un'immagine cattura molto l'aspetto visivo.** Uno screenshot bianco pieno di testo "somiglia" a uno scontrino più che a un'altra email con grafica diversa.
- **Il modello, interrogato direttamente, riconosce correttamente tutte e 6 le foto.** Il disaccordo fra voto dei ricordi e risposta del modello è quindi il segnale per fermarsi e chiedere a un umano.
- **Con archivi più ricchi** (più esempi per categoria, da fonti diverse) l'effetto dell'aspetto visivo si diluisce.

**Limiti:**
- con le immagini `open` non è ancora supportata;
- `inject` mostra le immagini dei ricordi solo come riferimento testuale (`[image: percorso]`), non le rimette nel prompt.

## 7. Domande sulla memoria

### `recall`: "cosa ti ricorda?" (tenuta)

È una domanda come le altre: restituisce i ricordi più simili allo stato, con similarità, decisioni e note. Il recupero è quello verificato (§2 e §6). `instructions` è facoltativa, perché il recupero usa lo stato e non il testo della domanda; `k` va da 1 a 10.

```json
{
  "state": [{"type": "text", "text": "Immagine ricevuta:"}, {"type": "image", "path": "examples/immagini/nuove/incendio2.jpg"}],
  "image_max_side": 448,
  "questions": {
    "ricorda": {"type": "recall", "instructions": "Cosa ti ricorda questa immagine?", "k": 2},
    "azione": {"type": "choice", "instructions": "Quale azione è appropriata?",
               "criteria": {"chiamare_vigili": "Chiamare i vigili del fuoco", "aprire_sinistro": "Aprire una pratica di sinistro", "nessuna": "Nessuna azione"}}
  }
}
```

```bash
.venv/bin/egeria decide --model Qwen/Qwen3.5-2B-Base --memory runs/memoria-foto --permutations 2 examples/memoria_ricorda.json
```

Risposta reale: archivio di 4 foto con note ([examples/memoria_ricorda.json](../examples/memoria_ricorda.json)), interrogato con una foto **nuova** di un altro incendio.

```json
"ricorda": {"type": "recall", "memories": [
  {"id": "foto-incendio-0812", "similarity": 0.8591, "decisions": {"azione": "chiamare_vigili"},
   "note": "Incendio in via Verdi, 12 agosto: vigili arrivati in 9 minuti"},
  {"id": "foto-sinistro-2026-0913", "similarity": 0.7985, "decisions": {"azione": "aprire_sinistro"},
   "note": "Urto contro palo, pratica 2026-0913 liquidata"}]},
"azione": {"type": "choice", "choice": "chiamare_vigili",
  "probabilities": {"chiamare_vigili": 0.857, "aprire_sinistro": 0.050, "nessuna": 0.093}, "confidence": 0.786}
```

### `known`: "è qualcosa che hai in memoria?" (non tenuta)

L'idea era rispondere su tre livelli: **stesso** caso già visto, **simile** a qualcosa in memoria, oppure **nuovo**. Verifica ([scripts/verifica_known.py](../scripts/verifica_known.py), 2B-Base; soglia decisa prima: AUROC ≥ 0.85 su entrambe le separazioni, per ogni tipo di dato):

| Separazione | Immagini (archivio di 6 foto) | Testo (archivio di 1.200 casi) |
|---|---|---|
| stesso (copia ritagliata o con un dettaglio in più) contro simile | **1.00**: copie ≥ 0.956, foto diverse ≤ 0.908 | 0.79 |
| simile contro nuovo (categoria o tema assente) | 0.81 | **1.00**: stesso tipo ≥ 0.898, fuori tema ≤ 0.29 |

La misura relativa all'archivio (percentile rispetto ai vicini interni) non fa meglio. Ogni tipo di dato riesce in una sola delle due domande, quindi `known` **non è stata tenuta**:
- **testo, stesso contro simile:** i casi di typed-decisions sono generati da modelli di testo e si somigliano quasi quanto le copie modificate;
- **immagini, simile contro nuovo:** tornano gli errori dovuti all'aspetto visivo (lo screenshot di phishing somiglia a uno scontrino; la rastrelliera colorata a 0.83).

**Cosa resta utilizzabile, con cautela.** `recall` restituisce le similarità, e dalla verifica emergono due regole pratiche:
- **testo:** similarità del ricordo più vicino sotto ~0.6 → argomento assente dalla memoria (qui fra 0.29 e 0.90 non c'era nulla);
- **immagini:** similarità ≥ ~0.94 → con ogni probabilità è la stessa foto, anche ritagliata (qui fra 0.908 e 0.956).

Sono soglie misurate su pochi casi e su questi archivi. Dipendono dal modello, da `image_max_side` e, sopra i 20 ricordi, dalla centratura. Per i duplicati esatti di testo basta un confronto dell'hash del testo normalizzato.

### Perché `known` fallisce e come può rientrare

**La causa comune: `recall` ordina, `known` usa una soglia.** `recall` deve solo ordinare i ricordi, e l'ordine resta giusto anche con similarità "schiacciate". `known` richiede una soglia assoluta, ma la scala delle similarità dipende da modello, tipo di dato, risoluzione e dimensione dell'archivio (centratura oltre i 20 ricordi).

**Testo, stesso contro simile.** Il vettore è una media su tutto lo stato: rappresenta *che tipo di cosa* è lo stato, non *quale istanza*. Casi diversi dello stesso flusso (stessa struttura, pochi valori diversi) hanno similarità 0.90–0.996; le copie modificate 0.95–0.996.

**Immagini, simile contro nuovo.** I token visivi dominano la media, quindi il vettore cattura soprattutto l'aspetto (colori, composizione, "foglio con testo"). Il modello, interrogato direttamente, riconosce invece tutte e 6 le categorie: il significato c'è, ma non emerge nella media.

**Come può rientrare:**
1. **Senza training: recupero + verifica.** Il recupero trova il candidato; una `noul` con entrambi gli stati nel prompt chiede "è lo stesso caso?" / "è lo stesso tipo di situazione?". Da verificare con le stesse soglie.
2. **F1, loss propria su coppie** (l'equivalente supervisionato di RLCD). Rende calibrata la probabilità di quella `noul`. Coppie costruibili: copie modificate e ritagli (stesso), casi diversi dello stesso flusso o categoria (simile), temi e categorie assenti (nuovo).
3. **F1, embedding contrastivo con LoRA.** Avvicina i vettori dello stesso tipo e allontana gli altri, con esempi difficili (screenshot contro scontrino). Migliora anche `recall` e il voto dei ricordi.
4. **Stesso caso nel testo:** controllo sul contenuto (hash del testo normalizzato, MinHash/Jaccard, campi identificativi), più affidabile di qualsiasi vettore semantico.
