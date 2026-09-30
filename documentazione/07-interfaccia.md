**Italiano** · [English](en/07-web-interface.md)

# Interfaccia web

> Stato: **implementata e collaudata** in un browser headless ([scripts/collaudo_ui.py](../scripts/collaudo_ui.py)). Seconda versione: la prima (console, monitor e memoria separati, domande in JSON, modelli di domande specifici) era difficile da usare ed è stata sostituita.

## 1. Avvio

Il modello e l'interfaccia sono **due server separati** (§1bis). Si avviano in due terminali:

```bash
cd /home/kuduk/egeria
uv pip install --python .venv/bin/python -e ".[model,eval,vision,server]"   # solo la prima volta

# 1) il modello: carica Qwen sulla GPU, porta 8100
.venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base

# 2) l'interfaccia: leggera, senza torch, porta 8000
.venv/bin/egeria serve --model-url http://127.0.0.1:8100
```

Poi, nel browser (anche da Windows, se il server gira in WSL): **http://localhost:8000/**.

- **Ordine di avvio:** indifferente. Se il modello non risponde, l'interfaccia lo segnala in alto ("Modello non raggiungibile") e alla prima domanda dice come avviarlo. Storico e ricordi restano consultabili.
- **Cambiare modello senza chiudere l'interfaccia:** si ferma e si riavvia solo `model-server`, per esempio con `--model Qwen/Qwen3.5-0.8B-Base`. Storico e ricordi non cambiano: alla prima domanda con i ricordi, i loro vettori si ricalcolano per il nuovo modello ([06-memoria.md](06-memoria.md) §5).
- **Per fermarli:** `Ctrl+C` in ciascun terminale. Se sono stati avviati in background:
  - `pgrep -f '^/home/kuduk/egeria/.venv/bin/python .venv/bin/egeria model-server' | xargs kill`
  - `pgrep -f '^/home/kuduk/egeria/.venv/bin/python .venv/bin/egeria serve' | xargs kill`
- **Dove finiscono i dati** (solo nel server web):
  - lo storico delle domande e le immagini caricate in `runs/console/` (`--data`);
  - i ricordi in `runs/memoria-console/` (`--memory`).
- **Calibrazione:** è un'opzione del server del modello (`model-server --temperatures`). Di default non la usa, e mostra le probabilità del modello così come sono (con 2 permutazioni delle opzioni).
  - Le temperature disponibili (`runs/*/temperature.json`) sono fittate sui casi di typed-decisions. Su domande generiche e sulle immagini appiattiscono le risposte: "Il veicolo è danneggiato? Sì" passava da 97% a 62%.
  - `--temperatures` va usato solo con una calibrazione fatta sui propri dati. Anche in quel caso **non viene mai applicata agli stati con immagini**, né ai fotogrammi dal vivo.

> **Prima della separazione** il comando era `egeria serve --model ... --temperatures ...`: ora `--model`, `--temperatures`, `--permutations`, `--min-confidence`, `--text-only` vanno a `model-server`.

## 1bis. Architettura: modello e server web separati

```
browser ──HTTP──▶ egeria serve (porta 8000)            ──HTTP──▶ egeria model-server (porta 8100)
                  interfaccia, storico (SQLite),                 Qwen3.5 sulla GPU, lock GPU
                  immagini caricate, ricordi,                    nessuno stato: niente storico,
                  voto dei ricordi                               niente ricordi, niente file
```

| | `egeria model-server` | `egeria serve` |
|---|---|---|
| Cosa fa | solo inferenza: decisioni e vettori semantici | interfaccia, storico, immagini caricate, ricordi |
| Dipendenze | torch, transformers, GPU | fastapi, uvicorn, httpx, numpy: **niente torch** |
| Stato | nessuno | `runs/console/`, `runs/memoria-console/` |
| Porta di default | 8100 | 8000 |
| Si riavvia per | cambiare modello, calibrazione, permutazioni, riuso dello stato (`--share-state`, [08](08-riuso-dello-stato.md)) | aggiornare l'interfaccia |

**Come passa una domanda:**
1. Il server web valida la richiesta e, se lo stato contiene immagini indicate con un percorso, le legge e le trasforma in base64. Accetta solo file dentro il progetto o dentro la cartella dei dati.
2. Se servono i ricordi (`memory.recall` > 0) e l'archivio non è vuoto, chiede al modello il vettore dello stato (`POST /v1/embed`) e cerca i ricordi più simili.
3. Manda al modello (`POST /v1/systemone`) la richiesta **senza** `memory`: al modello arrivano solo stato e domande.
4. Alla risposta del modello aggiunge il voto dei ricordi (`answers[q].memory`) e i ricordi richiamati (`memories`).

### API del server del modello

| Metodo e percorso | Body | Risposta |
|---|---|---|
| `GET /v1/info` | | modello, dispositivo, GPU, parte visiva, permutazioni, soglia di default, calibrazione |
| `POST /v1/systemone` | body Jev (`state`, `questions`, `min_confidence`, `image_max_side`) più `permutations` (1–8) e `calibrated` (`false` = probabilità grezze) | risposta Jev con le estensioni Egeria |
| `POST /v1/embed` | `{"state": ..., "image_max_side": 448}` | `embedding` (normalizzato L2), `embedding_dim`, `model` (il modello che l'ha calcolato), `input_tokens`, `latency_ms` |

Esempio, direttamente sul server del modello:

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

Con un'immagine, dal disco al base64:

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d "{
  \"state\": [{\"type\": \"image\", \"base64\": \"$(base64 -w0 examples/immagini/gatto.jpg)\"}],
  \"questions\": {\"gatto\": {\"type\": \"noul\", \"instructions\": \"Nella foto c'è un gatto?\"}}
}"
```

**Sicurezza del server del modello:**
- ascolta su `127.0.0.1` di default;
- rifiuta le immagini indicate con un percorso locale, perché leggerebbe file arbitrari: le immagini arrivano in base64 o url. `--allow-paths` le riabilita, per le prove in locale;
- rifiuta `memory`, perché i ricordi stanno nel server web: il messaggio d'errore lo spiega;
- `--token SEGRETO` (oppure la variabile `EGERIA_TOKEN`) richiede `Authorization: Bearer SEGRETO` su tutte le chiamate. Il server web lo passa con `--model-token` (o la stessa variabile). Serve se il modello gira su un'altra macchina, per esempio una GPU a noleggio raggiunta con un tunnel SSH:

```bash
# sulla macchina con la GPU
EGERIA_TOKEN=... .venv/bin/egeria model-server --model Qwen/Qwen3.5-2B-Base
# in locale: tunnel e interfaccia
ssh -N -L 8100:127.0.0.1:8100 utente@macchina-gpu &
EGERIA_TOKEN=... .venv/bin/egeria serve --model-url http://127.0.0.1:8100
```

**Prestazioni:** la separazione aggiunge un passaggio HTTP in locale, qualche millisecondo. Le immagini viaggiano in base64 (circa +33% di byte). Per i ricordi il modello fa un passaggio in più sullo stato (il vettore), come prima quando stava tutto in un processo.

## 2. Idea

L'interfaccia fa una cosa sola: **guarda qualcosa e rispondi alle mie domande, dicendomi quanto sei sicuro.** Non si vede mai il formato tecnico. Le domande si costruiscono con campi e menu, e gli esempi sono solo nel menu *Prova un esempio*.

In alto ci sono tre voci:
- **Chiedi**, la pagina principale;
- **Storico**, con il numero di analisi da controllare;
- **Ricordi**, con il numero di ricordi.

Accanto: lo stato del server, le impostazioni (ingranaggio) e il tema chiaro/scuro.

## 3. Chiedi

**A sinistra, "Cosa guardare"**, con due modalità:
- **Testo e immagini.** Un riquadro per scrivere o incollare un testo. Le immagini si aggiungono trascinandole, incollandole (Ctrl+V) o con *scegli un file*, e ognuna si toglie con la ×.
- **Dal vivo.** Si sceglie *Telecamera*, *Schermo* o *Video da file*; la sorgente la gestisce il browser, quindi funziona anche con il server in WSL. Si imposta quante immagini al secondo guardare e da che sicurezza avvisare (cursore, default 80%). Gli **Avvisi** mostrano immagine, ora e risposta, con *Salva nei ricordi* e *Ignora*.

Sotto: *Prova un esempio* (messaggio di un cliente, foto di un incidente, foto di uno scontrino, controllo incendi dal vivo) e *Ricomincia da zero*.

**A destra, "Cosa chiedere"**: una scheda per domanda.
1. Si scrive la domanda in italiano.
2. Si sceglie il tipo di risposta:

| Tipo di risposta | Cosa si imposta | Cosa si ottiene |
|---|---|---|
| **Sì / No** | niente | Sì o No, e quanto è sicuro |
| **Una tra più opzioni** | le opzioni: si scrive e si preme Invio, la × toglie | l'opzione scelta, con le probabilità di tutte |
| **Una scala** | i livelli dal più basso al più alto (default Basso, Medio, Alto) | il livello, con le probabilità di tutti |
| **Un numero** | "di solito tra … e …" e l'unità | "circa N", la fascia più probabile e l'intervallo quasi certo |
| **Una parola o un valore** | niente (funziona anche con le immagini) | una parola o un valore letto (nome, data, importo, codice), con le alternative |

3. In modalità dal vivo, ogni scheda ha anche *Avvisami quando la risposta è…*.

*Aggiungi una domanda* aggiunge una scheda; il cestino la toglie. *Salva queste domande* le salva con un nome, e *Domande salvate…* le ricarica: sono le domande dell'utente, salvate nel browser.

**Chiedi** (oppure Ctrl+Invio) mostra la risposta dentro ogni scheda:
- **la risposta**, grande, per esempio "Sì · sicuro al 97%";
- **le barre** delle probabilità;
- **«Non sono sicuro»** se la sicurezza è sotto la soglia scelta nelle impostazioni;
- **i ricordi**, se ci sono casi simili: "Nei casi simili che ricordo (n) avevi deciso: … — uguale / diverso".

Poi:
- **Correggi** accanto a ogni risposta: si sceglie quella giusta.
- **Salva nei ricordi**, in fondo, con una nota facoltativa: il caso, con le correzioni, viene ricordato per le prossime domande simili.

Se si cambia il testo o una domanda dopo aver chiesto, le risposte si sbiadiscono e compare "premi «Chiedi» per aggiornare".

## 4. Storico

Tutte le analisi fatte con *Chiedi*. Il filtro *Da controllare* mostra quelle con risposte incerte o diverse dai ricordi, ancora da salvare. Un clic riapre l'analisi nella pagina *Chiedi*, con testo, immagini, domande e risposte, pronta per essere corretta e salvata nei ricordi.

## 5. Ricordi

I casi salvati, in italiano: testo o immagine, "domanda → risposta", nota, data.
- **Cerca:** si descrive qualcosa e si ottengono i ricordi più simili.
- **Cerca con un'immagine:** stessa cosa, partendo da una foto.
- **Dimentica:** cancella un ricordo, con conferma.

## 6. Impostazioni (ingranaggio)

- **Quando segnalare una risposta come incerta:** spesso, normale o raramente (soglia di confidenza 0.7, 0.5 o 0.3).
- **Usa i ricordi:** mostra cosa si era deciso nei casi simili.
- **Qualità delle immagini:** veloce, normale, dettagliata o massima (lato 336, 448, 672 o 896 px). *Massima* serve per leggere testi piccoli nei documenti.

Le impostazioni restano nel browser.

## 7. API del server web (per altri programmi)

| Metodo e percorso | Uso |
|---|---|
| `POST /v1/systemone` | API compatibile Jev, con le estensioni Egeria **e i ricordi** (`memory.recall`, `memory.min_similarity`, `memory.vote`) |
| `GET /api/info` | modello e stato del server del modello (`model_status`: `ok` o `non_raggiungibile`), numero di ricordi e di analisi, modello dei vettori dei ricordi (`memory_model`) e ricordi sospesi (`memories_suspended`) |
| `GET /api/cases?view=da_rivedere\|tutti` | storico |
| `POST /api/cases` | `{"request": <body /v1/systemone>}`: analizza e salva nello storico |
| `GET /api/cases/{id}`, `POST /api/cases/{id}/review` | dettaglio; salvataggio nei ricordi con `{"decisions": {...}, "note": ...}` |
| `POST /api/monitor/frame` | un fotogramma (data URL) e le domande, per la modalità dal vivo |
| `GET/POST /api/memory`, `POST /api/memory/search`, `DELETE /api/memory/{id}` | ricordi |
| `POST /api/media` | carica un'immagine (data URL) e restituisce `path` e `url` |

Errori: `422` per una richiesta non valida (anche quando è il modello a rifiutarla), `503` se il server del modello non è raggiungibile o rifiuta il token.

## 8. Collaudo

`scripts/collaudo_ui.py` percorre l'interfaccia in Chromium headless, su un'istanza separata per non toccare i dati veri, e salva gli screenshot:
1. scrive due domande a mano, con opzioni aggiunte da tastiera;
2. chiede, corregge una risposta e salva nei ricordi;
3. carica l'esempio con la foto e chiede;
4. passa alla modalità dal vivo;
5. apre storico e ricordi;
6. cambia tema.

Ultimo esito: nessun errore nella console del browser. Risposte sul caso scritto a mano: "Il cliente chiede un rimborso? Sì, 90%"; "Qual è il problema? Consegna in ritardo, 99%".

Dopo la separazione (settembre 2026) il collaudo è stato ripetuto con `model-server` (Qwen3.5-0.8B-Base, porta 8100) e un'istanza web di prova (porta 8001): nessun errore in console. In più è stato provato il modello spento: in alto compare "Modello non raggiungibile" e *Chiedi* mostra "server del modello non raggiungibile su http://127.0.0.1:8100 (avvialo con: egeria model-server)".

Prove a mano sul percorso completo:
- testo: "4471" letto come numero d'ordine al 93%;
- foto indicata con un percorso: arriva al modello in base64; il ricordo salvato viene richiamato con similarità 1.0 e vota la risposta;
- stessa foto mandata con il percorso direttamente al server del modello: rifiutata con `422`.

I test automatici ([tests/test_server.py](../tests/test_server.py)) fanno parlare il client HTTP vero con il server del modello in-process. Coprono token, percorsi rifiutati, calibrazione mai applicata alle immagini, ricordi e voto calcolati dal server web, immagini in base64, modello non raggiungibile (`503`).

**Bug trovati e corretti con il collaudo:**
- barre di probabilità vuote (elemento inline senza `display: block`);
- opzioni duplicate quando si premeva Invio (il blur del campo ridisegnato le riaggiungeva);
- risposte appiattite dalla calibrazione fatta su un altro tipo di dati.

## 9. Scelte di progetto

- **Nessuna build:** HTML, CSS e JS statici serviti da FastAPI. Il DOM si costruisce solo con `textContent` (niente XSS).
- **Video da file, in modalità dal vivo:** si accettano solo MP4, WebM, Ogg, MOV e MKV. Il file diventa un URL `blob:` con il tipo preso da una tabella fissa (non dal file), e l'URL precedente si libera quando si cambia video. Così si è risolto il 29/09/2026 l'avviso di CodeQL `js/xss-through-dom`, verificato in locale con CodeQL 2.27.1: zero avvisi con la suite di code scanning e con `security-extended`.
- **Stile:** tema scuro o chiaro, Fira Sans/Fira Code, icone SVG.
  - **Logo:** [img/logo.png](../src/egeria/web/img/logo.png). Nella barra in alto c'è l'emblema su un riquadro chiaro arrotondato, uguale nei due temi, accanto al nome in maiuscolo spaziato come nella scritta del logo. Il riquadro chiaro serve perché il volto è disegnato con il bianco del fondo: su uno sfondo scuro l'emblema diventerebbe un negativo.
  - **Immagini derivate:** emblema, favicon, icona per iPhone e le due anteprime social 1280×640 per GitHub (logo centrato, oppure emblema e scritta affiancati) si generano con `.venv/bin/python scripts/genera_icone.py`, da rilanciare se cambia il logo.
  - Colori di stato sempre affiancati da testo e icona: ambra = non sono sicuro, viola = diverso dai ricordi, rosso = avviso.
  - Focus visibile e `prefers-reduced-motion` rispettato.
- **Modello separato dall'interfaccia** (§1bis): un solo modello caricato nel `model-server`, con un lock sulla GPU; il server web non importa torch. Entrambi ascoltano solo su `127.0.0.1` di default, e `/files` serve solo immagini dentro il progetto.
- **«Non sono sicuro» anche sul testo:** ogni risposta con una soglia riporta `status`. Prima della separazione, sul testo senza calibrazione `status` mancava e l'avviso compariva solo sulle immagini; ora c'è sempre.
- **Identificativi delle domande** ricavati dal testo della domanda. La stessa domanda fatta di nuovo ha lo stesso identificativo, e così il voto dei ricordi la riconosce.

## 10. Consigli d'uso emersi dalle prove

- **Chiedere di leggere, non di ragionare.** Il modello legge bene valori e testi, ma sbaglia confronti e calcoli: "ha più di 21 anni?" fallisce, "qual è la data di nascita?" funziona. Si chiede il valore con *Una parola o un valore* e il confronto lo si fa a mano o nel codice ([04-stato-con-immagini.md](04-stato-con-immagini.md) §7).
- **Documenti e testi piccoli:** qualità *Massima* nelle impostazioni.
- **Domande semplici:** una cosa per domanda. Vedi l'esempio del sinistro in [04-stato-con-immagini.md](04-stato-con-immagini.md) §6.
