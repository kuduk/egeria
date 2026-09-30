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
- **Lettura di default** (dal 29/09/2026, [09](09-controlli-senza-etichette.md) §9):
  - le opzioni passano per tutte le posizioni (`--permutations auto`: tutte le rotazioni per Sì/No e scelta, ordine diretto e inverso per le scale);
  - le domande Sì/No hanno la **correzione della tendenza al Sì**: la domanda si pone anche sullo stato vuoto "N/A", una volta sola, e se lì il modello pende verso il Sì quello spostamento si toglie. Si spegne con `--no-yes-correction`.
- **Calibrazione con le temperature:** è un'opzione del server del modello (`model-server --temperatures`). Di default non la usa.
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
| Si riavvia per | cambiare modello, calibrazione, permutazioni, correzione del Sì, riuso dello stato (`--share-state`, [08](08-riuso-dello-stato.md)) | aggiornare l'interfaccia |

**Come passa una domanda:**
1. Il server web valida la richiesta e, se lo stato contiene immagini indicate con un percorso, le legge e le trasforma in base64. Accetta solo file dentro il progetto o dentro la cartella dei dati.
2. Se servono i ricordi (`memory.recall` > 0) e l'archivio non è vuoto, chiede al modello il vettore dello stato (`POST /v1/embed`) e cerca i ricordi più simili.
3. Manda al modello (`POST /v1/systemone`) la richiesta **senza** `memory`: al modello arrivano solo stato e domande.
4. Alle domande con `"batch_calibration": true` applica la calibrazione sullo storico (sotto).
5. Alla risposta del modello aggiunge il voto dei ricordi (`answers[q].memory`) e i ricordi richiamati (`memories`).

### Calibrazione sullo storico (facoltativa, per domanda)

Una domanda con `"batch_calibration": true` (tipi `noul`, `choice`, `score`) viene corretta con le risposte date alla **stessa domanda** (stesso tipo, testo e opzioni) e allo stesso modello nei casi precedenti dello storico:
- la distribuzione del modello si divide per la media delle distribuzioni degli ultimi 200 casi, poi si rinormalizza;
- si applica **da 50 casi**: prima la risposta resta com'è e riporta quanti casi mancano;
- ogni risposta riporta `batch_calibration`: `{"cases": n, "min_cases": 50, "applied": true|false}`, e da applicata anche `raw`, la distribuzione prima della correzione. Nello storico conta sempre `raw`, così le correzioni non si sommano;
- **non va usata per gli eventi rari** (per esempio "c'è un incendio?"): presume che le risposte vere siano varie, e su una domanda che risponde quasi sempre "No" spingerebbe verso falsi allarmi. **Non si applica mai dal vivo**, dove i fotogrammi si somigliano.

Sul 2B, su typed-decisions, è la correzione più forte: 0.529 di accuratezza con una sola permutazione, 0.545 con tutte le rotazioni ([09](09-controlli-senza-etichette.md) §8). Nella pagina Chiedi è la casella **Calibra sullo storico** di ogni domanda.

### API del server del modello

| Metodo e percorso | Body | Risposta |
|---|---|---|
| `GET /v1/info` | | modello, dispositivo, GPU, parte visiva, permutazioni, correzione del Sì, soglia di default, calibrazione |
| `POST /v1/systemone` | body Jev (`state`, `questions`, `min_confidence`, `image_max_side`) più `permutations` (`"auto"`, il default, oppure 1–26, [09](09-controlli-senza-etichette.md) §2), `yes_correction` (`false` = senza la correzione del Sì) e `calibrated` (`false` = senza temperature) | risposta Jev con le estensioni Egeria |
| `POST /v1/embed` | `{"state": ..., "image_max_side": 448}` | `embedding` (normalizzato L2), `embedding_dim`, `model` (il modello che l'ha calcolato), `input_tokens`, `latency_ms` |

Esempio, direttamente sul server del modello:

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Salve, il mio ordine 4471 non è ancora arrivato. Vorrei il rimborso.",
  "questions": {
    "rimborso": {"type": "noul", "instructions": "Il cliente chiede un rimborso?"},
    "ordine":   {"type": "short_answer", "instructions": "Qual è il numero d ordine?"}
  },
  "permutations": 1,
  "yes_correction": false
}'
```

Con `"permutations": 1` e `"yes_correction": false` è la lettura più veloce, senza correzioni.

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

Accanto: lo stato del server, le impostazioni (ingranaggio), il tema chiaro/scuro e la lingua: un pulsante che mostra «EN» quando l'interfaccia è in italiano e «IT» quando è in inglese (§6).

L'interfaccia è in **italiano e in inglese**, con la stessa cura per le due lingue.

## 3. Chiedi

**A sinistra, "Cosa guardare"**, con due modalità. La pagina si apre **dal vivo** (dal 29/09/2026); si passa all'altra con il selettore in alto nel riquadro:
- **Dal vivo.** Si sceglie *Telecamera*, *Schermo* o *Video da file*; la sorgente la gestisce il browser, quindi funziona anche con il server in WSL. Si imposta quante immagini al secondo guardare (default *Il più possibile*: il fotogramma successivo parte appena arriva la risposta al precedente, uno alla volta) e da che sicurezza avvisare (cursore, default 80%). Gli **Avvisi** mostrano immagine, ora e risposta, con *Salva nei ricordi* e *Ignora*.
- **Testo e immagini.** Un riquadro per scrivere o incollare un testo. Le immagini si aggiungono trascinandole, incollandole (Ctrl+V) o con *scegli un file*, e ognuna si toglie con la ×.

Sotto: *Prova un esempio* (messaggio di un cliente, foto di un incidente, foto di uno scontrino, controllo incendi dal vivo) e *Ricomincia da zero*. Con l'interfaccia in inglese gli esempi sono in inglese: stesse immagini, testi e domande tradotti.

**A destra, "Cosa chiedere"**: una scheda per domanda.
1. Si scrive la domanda, in italiano o in inglese.
2. Si sceglie il tipo di risposta:

| Tipo di risposta | Cosa si imposta | Cosa si ottiene |
|---|---|---|
| **Sì / No** | niente | Sì o No, e quanto è sicuro |
| **Una tra più opzioni** | le opzioni: si scrive e si preme Invio, la × toglie | l'opzione scelta, con le probabilità di tutte |
| **Una scala** | i livelli dal più basso al più alto (default Basso, Medio, Alto) | il livello, con le probabilità di tutti |
| **Una risposta breve (un nome, un numero, una data…)** | niente (funziona anche con le immagini) | una parola o un valore letto (nome, data, importo, codice), con le alternative |
| **Una stima (quanti, quanto…)** | "di solito tra … e …" e l'unità | "circa N", la fascia più probabile e l'intervallo quasi certo. Se la media cade fuori dalla fascia più probabile, in grande c'è la fascia e la media va nella riga sotto. Serve a stimare quantità non scritte: per leggere un numero scritto si usa *Una risposta breve* (lo dice il suggerimento sotto la domanda) |

Fino al 29/09/2026 questi due tipi si chiamavano *Una parola o un valore* e *Un numero*, e nell'API `open` e `number`: i vecchi nomi dell'API sono ancora accettati, e lo storico e le domande salvate nel browser si riaprono con i nomi nuovi.

3. Per Sì/No, scelta e scala, in modalità testo e immagini, la casella **Calibra sullo storico** corregge la risposta con quelle date alla stessa domanda nei casi precedenti (da 50 casi; §1bis). La spiegazione, con l'avvertenza di non usarla per gli eventi rari, compare quando la casella è spuntata.
4. In modalità dal vivo, ogni scheda ha anche *Avvisami quando la risposta è…*.

*Aggiungi una domanda* aggiunge una scheda; il cestino la toglie. *Salva queste domande* le salva con un nome, e *Domande salvate…* le ricarica: sono le domande dell'utente, salvate nel browser.

**Chiedi** (oppure Ctrl+Invio) mostra la risposta dentro ogni scheda:
- **la risposta**, grande, per esempio "Sì · sicuro al 97%";
- **le barre** delle probabilità;
- **«Non sono sicuro»** se la sicurezza è sotto la soglia scelta nelle impostazioni;
- **i ricordi**, se ci sono casi simili: "Nei casi simili che ricordo (n) avevi deciso: … — uguale / diverso";
- con la calibrazione sullo storico, una riga che dice se è stata applicata ("Calibrata sullo storico (n casi precedenti)") o quanti casi mancano ("Non ancora calibrata sullo storico: n casi su 50").

Poi:
- **Correggi** accanto a ogni risposta: si sceglie quella giusta.
- **Salva nei ricordi**, in fondo, con una nota facoltativa: il caso, con le correzioni, viene ricordato per le prossime domande simili.

Se si cambia il testo o una domanda dopo aver chiesto, le risposte si sbiadiscono e compare "premi «Chiedi» per aggiornare".

## 4. Storico

Tutte le analisi fatte con *Chiedi*. Il filtro *Da controllare* mostra quelle con risposte incerte o diverse dai ricordi, ancora da salvare. Un clic riapre l'analisi nella pagina *Chiedi*, con testo, immagini, domande e risposte, pronta per essere corretta e salvata nei ricordi.

## 5. Ricordi

I casi salvati, in forma leggibile: testo o immagine, "domanda → risposta", nota, data. Domande e opzioni restano come le ha scritte l'utente; le risposte Sì/No si vedono nella lingua dell'interfaccia (§6).
- **Cerca:** si descrive qualcosa e si ottengono i ricordi più simili.
- **Cerca con un'immagine:** stessa cosa, partendo da una foto.
- **Dimentica:** cancella un ricordo, con conferma.

## 6. Impostazioni (ingranaggio)

- **Lingua · Language:** italiano o inglese (è la prima voce). Il cambio è immediato, senza aspettare *Fatto*.
- **Quando segnalare una risposta come incerta:** spesso, normale o raramente (soglia di confidenza 0.7, 0.5 o 0.3).
- **Usa i ricordi:** mostra cosa si era deciso nei casi simili.
- **Qualità delle immagini:** veloce, normale, dettagliata o massima (lato 336, 448, 672 o 896 px). *Massima* serve per leggere testi piccoli nei documenti.

Le impostazioni restano nel browser.

### Lingua

- **Dove si cambia:** nelle impostazioni (*Lingua · Language*) oppure con il pulsante in alto accanto al tema, che mostra la sigla della lingua in cui si passa («EN» o «IT») e ha un'etichetta per i lettori di schermo ("Passa all'inglese (English)", "Switch to Italian (Italiano)").
- **Lingua di partenza:** quella scelta l'ultima volta; se non c'è, quella del browser (`navigator.languages`): italiano se comincia con `it`, inglese in tutti gli altri casi.
- **Dove resta:** nel browser, chiave `egeria-lang` di `localStorage` (`it` o `en`), come il tema. Se il browser non permette di salvarla, si riparte dalla lingua del browser. Anche `<html lang>` segue la lingua.
- **Cosa succede quando si cambia:** si ridisegna tutta l'interfaccia (testi, schede delle domande, risposte già mostrate, storico, ricordi, avvisi dal vivo, notifiche) senza perdere il lavoro in corso: testo, immagini, domande, risposte e correzioni restano. Il focus resta sul pulsante o sul menu usato.
- **Cosa cambia:** i tipi di risposta, i livelli di default di una scala (Basso, Medio, Alto → Low, Medium, High), Sì/No, "sicuro al 97%" → "97% sure", «Non sono sicuro» → "Not sure", gli esempi, i tempi ("5 minuti fa"), date e numeri (virgola o punto decimale).
- **Cosa non cambia:** i testi scritti dall'utente (domande, opzioni, livelli già visibili in una scheda, note, nomi delle domande salvate). I livelli di default (Basso, Medio, Alto) prendono la lingua attuale quando la scheda diventa di tipo *Una scala*; se sono già visibili restano come sono. Gli identificativi delle domande si ricavano sempre dal testo della domanda (§9), in qualunque lingua.
- **Ricordi:** il server salva le etichette leggibili in italiano ("Sì"/"No"). Per ogni risposta restituisce anche il valore grezzo (`readable[].value`, per esempio `"true"`), così l'interfaccia in inglese mostra Yes/No. Il testo "Immagine dal vivo:" nello stato dei ricordi salvati dal vivo resta in italiano, perché entra nel vettore del ricordo; nelle schede si vede tradotto.
- **Errori del server:** i dettagli che manda il server restano **in italiano** (per esempio quelli dei `422`). Con l'interfaccia in inglese si traducono solo quelli riconoscibili con certezza: server del modello non raggiungibile, token rifiutato, errore del server del modello, caso o ricordo non trovato, immagine non valida, decisioni mancanti, ricordo già esistente. Gli altri si vedono come arrivano.

## 7. API del server web (per altri programmi)

| Metodo e percorso | Uso |
|---|---|
| `POST /v1/systemone` | API compatibile Jev, con le estensioni Egeria, **i ricordi** (`memory.recall`, `memory.min_similarity`, `memory.vote`) e la calibrazione sullo storico per domanda (`"batch_calibration": true`, §1bis) |
| `GET /api/info` | modello e stato del server del modello (`model_status`: `ok` o `non_raggiungibile`), numero di ricordi e di analisi, modello dei vettori dei ricordi (`memory_model`) e ricordi sospesi (`memories_suspended`) |
| `GET /api/cases?view=da_rivedere\|tutti` | storico |
| `POST /api/cases` | `{"request": <body /v1/systemone>}`: analizza e salva nello storico |
| `GET /api/cases/{id}`, `POST /api/cases/{id}/review` | dettaglio; salvataggio nei ricordi con `{"decisions": {...}, "note": ...}` |
| `POST /api/monitor/frame` | un fotogramma (data URL) e le domande, per la modalità dal vivo (senza calibrazione sullo storico) |
| `GET/POST /api/memory`, `POST /api/memory/search`, `DELETE /api/memory/{id}` | ricordi; ognuno ha `readable`: domanda, risposta leggibile (`answer`: il testo dell'opzione, oppure Sì/No) e valore grezzo (`value`) |
| `POST /api/media` | carica un'immagine (data URL) e restituisce `path` e `url` |

Errori: `422` per una richiesta non valida (anche quando è il modello a rifiutarla), `503` se il server del modello non è raggiungibile o rifiuta il token.

## 8. Collaudo

`scripts/collaudo_ui.py` percorre l'interfaccia in Chromium headless, su un'istanza separata per non toccare i dati veri, e salva gli screenshot. Chromium headless ha il browser in inglese: il passaggio in italiano usa un browser con lingua `it-IT`.
1. controlla che la pagina si apra dal vivo, passa a *Testo e immagini*, scrive due domande a mano, con opzioni aggiunte da tastiera, e spunta *Calibra sullo storico* sulla seconda;
2. chiede (la seconda risposta dice quanti casi mancano alla calibrazione, la prima no), corregge una risposta e salva nei ricordi;
3. carica l'esempio con la foto e chiede;
4. passa alla modalità dal vivo;
5. apre storico e ricordi;
6. cambia tema;
7. con un browser `en-US` controlla che l'interfaccia parta in inglese;
8. passaggio in inglese: parte in italiano con testo e domanda già scritti, passa all'inglese con il pulsante in alto (il lavoro resta, il focus pure), controlla le voci principali, chiede, torna all'italiano e di nuovo all'inglese (le risposte restano, tradotte), corregge e salva, cambia lingua dalle impostazioni, apre il dialogo per salvare le domande, carica un esempio in inglese, passa al dal vivo, apre storico e ricordi (Sì/No dei ricordi mostrati come Yes/No), ricarica la pagina (la lingua resta).

In ogni pagina cerca le parole dell'interfaccia rimaste nell'altra lingua ("Chiedi", "Storico", "Ricordi", "Non sono sicuro", "Salva nei ricordi", "Impostazioni"… e, nel passaggio in italiano, "Ask", "History"…), esclusi i testi dell'utente, e i segnaposto `{nome}` non sostituiti. Controlla anche che niente esca dalla pagina a 1440 e a 390 px di larghezza. Esce con 1 se trova problemi o errori in console.

Esito della prima versione: nessun errore nella console del browser. Risposte sul caso scritto a mano: "Il cliente chiede un rimborso? Sì, 90%"; "Qual è il problema? Consegna in ritardo, 99%".

Dopo la separazione (settembre 2026) il collaudo è stato ripetuto con `model-server` (Qwen3.5-0.8B-Base, porta 8100) e un'istanza web di prova (porta 8001): nessun errore in console. In più è stato provato il modello spento (l'interfaccia era allora solo in italiano): in alto compare "Modello non raggiungibile" e *Chiedi* mostra "server del modello non raggiungibile su http://127.0.0.1:8100 (avvialo con: egeria model-server)".

Prove a mano sul percorso completo:
- testo: "4471" letto come numero d'ordine al 93%;
- foto indicata con un percorso: arriva al modello in base64; il ricordo salvato viene richiamato con similarità 1.0 e vota la risposta;
- stessa foto mandata con il percorso direttamente al server del modello: rifiutata con `422`.

I test automatici ([tests/test_server.py](../tests/test_server.py)) fanno parlare il client HTTP vero con il server del modello in-process. Coprono token, percorsi rifiutati, calibrazione mai applicata alle immagini, ricordi e voto calcolati dal server web, immagini in base64, modello non raggiungibile (`503`).

Collaudo delle due lingue (29/09/2026), con il modello Qwen3.5-0.8B-Base in CPU (porta 8110) e un'istanza web di prova (porta 8011): nessun errore in console, nessuna parola rimasta nell'altra lingua, niente fuori pagina a 1440 e a 390 px. Provati a mano anche: errore di una scheda e notifiche che si ritraducono, avvisi dal vivo con una telecamera finta (`--use-fake-device-for-media-stream`) salvati nei ricordi, modello spento ("model server unreachable at …" in inglese, il messaggio originale in italiano).

Collaudo della calibrazione sullo storico (29/09/2026), con Qwen3.5-2B-Base su GPU (porta 8101) e l'istanza web di prova (porta 8001): nessun errore in console, nessuna parola nell'altra lingua, niente fuori pagina. Corretti dopo aver guardato gli screenshot: "1 casi su 50" (ora singolare e plurale) e la spiegazione lunga su ogni scheda (ora compare solo con la casella spuntata).

**Bug trovati e corretti con il collaudo:**
- barre di probabilità vuote (elemento inline senza `display: block`);
- opzioni duplicate quando si premeva Invio (il blur del campo ridisegnato le riaggiungeva);
- risposte appiattite dalla calibrazione fatta su un altro tipo di dati;
- a 390 px di larghezza la pagina usciva di lato (larga 469 px): barra in alto troppo piena, menu con larghezza minima fissa. Ora sotto i 480 px nella barra resta solo l'emblema, i menu si stringono e nello storico i contrassegni vanno sotto il testo;
- dal vivo, ogni nuovo avviso cancellava i precedenti, e il pulsante *Salvato* di un avviso tornava cliccabile (trovati riscrivendo gli avvisi per le due lingue).

## 9. Scelte di progetto

- **Nessuna build:** HTML, CSS e JS statici serviti da FastAPI. Il DOM si costruisce solo con `textContent` (niente XSS).
- **Due lingue senza librerie:** [i18n.js](../src/egeria/web/i18n.js) contiene i due dizionari e `t(chiave, valori)`, con `{nome}` sostituito e il plurale scelto con `Intl.PluralRules` (`{one, other}`). In `index.html` i testi statici hanno gli attributi `data-i18n` (testo), `data-i18n-placeholder`, `data-i18n-aria-label` e `data-i18n-title`; i testi dinamici di `app.js` passano tutti da `t()`. Anche le traduzioni finiscono nel DOM solo con `textContent`. Per aggiungere un testo: una chiave in entrambi i dizionari (`it` è quello di riserva).
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

- **Affermazioni in forma positiva.** Per le domande Sì/No si scrive "il pacco è arrivato", non "il pacco non è arrivato": sulle negazioni il modello è inaffidabile ([09-controlli-senza-etichette.md](09-controlli-senza-etichette.md) §3).
- **Un numero scritto si legge con *Una risposta breve*,** non con *Una stima*, che stima a fasce. Sul totale di uno scontrino (14,21 €) *Una risposta breve* legge 14,21 al 97–99%; *Una stima* dà "circa 23–28" e a 448 px sbaglia anche la fascia ([05](05-primitive-di-lettura.md) §5).
- **Chiedere di leggere, non di ragionare.** Il modello legge bene valori e testi, ma sbaglia confronti e calcoli: "ha più di 21 anni?" fallisce, "qual è la data di nascita?" funziona. Si chiede il valore con *Una risposta breve* e il confronto lo si fa a mano o nel codice ([04-stato-con-immagini.md](04-stato-con-immagini.md) §7).
- **Documenti e testi piccoli:** qualità *Massima* nelle impostazioni.
- **Domande semplici:** una cosa per domanda. Vedi l'esempio del sinistro in [04-stato-con-immagini.md](04-stato-con-immagini.md) §6.
- **Pose e figure simili fra loro:** meglio *Una tra più opzioni* con le pose **descritte a parole** ("in piedi su una gamba, con il piede appoggiato all'interno dell'altra") che un Sì/No con il nome della posa. Con "La persona fa la posa dell'albero?" il 2B dice Sì anche a due guerrieri su tre; con le opzioni descritte le pose scambiate per l'albero scendono al 14%. Le pose su una gamba restano confuse fra loro ([09](09-controlli-senza-etichette.md) §10.3).
