**Italiano** · [English](en/10-left-right.md)

# Destra e sinistra nelle immagini

> **Stato:** indagine del 05/10/2026. Il 29/09 destra e sinistra erano state tolte dai controlli e dal primo LoRA perché "lì c'è qualcosa che non funziona proprio" ([02](02-implicazioni-e-proposta.md) §0). Script: [scripts/indagine_destra_sinistra.py](../scripts/indagine_destra_sinistra.py). Rapporti: `runs/destra_sinistra/<modello>.json`.

**In breve:**
- **Nessuno scambio destra/sinistra.** Su figure semplici il 2B risponde giusto al 100%.
- **Il difetto è sulle parti del corpo.** Il 2B-**Base** non guarda: alla domanda sulla gamba sollevata dà sempre la stessa risposta. Il 2B-**Instruct** la trova (88.5%).
- **La destra *della persona* non la sa nessuno dei due.** "La mano alzata è la destra?" riceve sempre No (il caso dell'utente, §3.1). Chiedere "in quale metà dell'immagine si trova la mano alzata?" funziona: 8/8 con tutti e due i modelli. Nemmeno sui primi piani il modello usa la chiralità della mano (§3.2).
- **Nel percorso dell'immagine c'era un bug** (orientamento EXIF), ora corretto. La webcam nel browser invece è pulita.

## 1. Il percorso dell'immagine

**Orientamento EXIF: bug, corretto.**
- **Il difetto:** un telefono spesso salva la foto con i pixel ruotati e un'etichetta EXIF che dice come girarla. `load_images` non la applicava. Una foto scattata col telefono girato arrivava al modello ruotata di 90° (il cerchio che l'utente vede a sinistra finiva in basso). Con le etichette 2, 4, 5 e 7, usate da alcune fotocamere frontali, arrivava **specchiata**, cioè con destra e sinistra scambiate.
- **Chi era colpito:** le immagini lette dal disco o mandate da un programma (riga di comando, API con percorso, url o base64).
- **Chi no:** le immagini caricate dall'interfaccia (il browser applica l'orientamento quando le ridimensiona) e i fotogrammi della webcam.
- **La correzione:** `ImageOps.exif_transpose` in [images.py](../src/egeria/images.py), con un test. Nessuna immagine di `examples/immagini` aveva un'etichetta EXIF diversa da "normale".

**Webcam: pulita.** Ho usato Chromium con una webcam finta che trasmette un video preparato apposta (`--use-file-for-fake-video-capture`): un cerchio rosso a sinistra e un quadrato blu a destra. Il fotogramma che l'interfaccia manda al server è stato intercettato.

| | Posizione del rosso (x) | Posizione del blu (x) |
|---|---|---|
| Video della webcam | 0.22 | 0.78 |
| Anteprima nella pagina | 0.22 | 0.78 |
| Fotogramma mandato al modello | 0.22 | 0.78 |

- **Coerenza:** l'anteprima non è specchiata (nessun `transform` nel CSS), e quello che si vede è quello che vede il modello.
- **Driver della webcam:** può mandare l'immagine già specchiata, cosa che da qui non si può verificare.
- **Come accorgersene:** davanti alla webcam si alza la mano destra. In un'immagine normale compare **a sinistra** nell'anteprima, come in una foto; se compare a destra, come in uno specchio, l'immagine è specchiata.

## 2. Il modello sulle figure semplici: perfetto

48 scene sintetiche, metà con due figure affiancate e metà con una figura sopra l'altra, ognuna anche specchiata. 2B-Base, lettura di default, in italiano e in inglese:

| Domanda | Accuratezza | Coerente con lo specchio |
|---|---|---|
| "Il cerchio rosso è a sinistra o a destra del quadrato blu?" | 1.000 | 100% |
| "In quale metà dell'immagine si trova il cerchio rosso?" | 1.000 | 100% |
| Sì/No "Il cerchio rosso è a sinistra del quadrato blu?" | 1.000 | 100% |
| Controllo: "sopra o sotto?" | 1.000 | 100% |

Lo 0.8B-Base è molto più debole: 0.67–0.75, coerente con lo specchio solo in parte.

## 3. Il modello sulle parti del corpo

**I dati.** 13 foto di pose con una gamba sollevata ([examples/immagini/pose](../examples/immagini/pose)), più le loro versioni specchiate (26 immagini):
- 8 pose dell'albero viste di fronte;
- 1 vista di schiena;
- 4 viste di lato.

**Le domande**, nelle due cornici:
- **dell'immagine:** "Nell'immagine, la gamba sollevata è sul lato sinistro o sul lato destro?";
- **della persona:** "La persona ha sollevato la sua gamba destra o la sua gamba sinistra?". Per chi è visto di fronte, la sua destra sta a sinistra nell'immagine.

| Modello | Cornice dell'immagine: accuratezza (coerenza con lo specchio) | Cornice della persona: accuratezza | Persona di fronte: risponde con il lato dell'immagine |
|---|---|---|---|
| 0.8B-Base | it 0.50 (0%), en 0.81 (61%) | 0.50 | 50% |
| **2B-Base** (default) | it **0.50 (0%: sempre "sinistra")**, en 0.65 (31%) | 0.50 (sempre "destra") | 50% |
| **2B-Instruct** | **it 0.885 (92%), en 0.885 (77%)** | it 0.44, en 0.33 | 56–62% |

**Varianti provate sul 2B-Base** (cornice dell'immagine):

| Variante | Accuratezza | Coerenza con lo specchio |
|---|---|---|
| Immagine a 800 px invece di 448 | 0.50 | 0% |
| "Il ginocchio della gamba piegata sporge verso il lato sinistro o destro?" | 0.50 | 0% |
| Due Sì/No: "Nella metà sinistra / destra c'è una gamba piegata?" | 0.50 | 0% |
| **"In quale metà dell'immagine si trova il piede sollevato da terra?"** | **0.77** | 54% |
| **Immagine tagliata in due**, "c'è una gamba piegata?" su ogni metà | **0.73** | **92%** |

**Cosa emerge:**
1. **"Qualcosa che non funziona proprio" è il 2B-Base sulle parti del corpo.** Non scambia destra e sinistra: non guarda, e dà una risposta fissa. Più risoluzione non aiuta.
2. **Chiedere dove sta un oggetto preciso funziona meglio** (il piede: 0.77) che chiedere da che parte è una parte del corpo. Tagliare l'immagine costringe il modello a guardare (coerenza 92%).
3. **Il 2B-Instruct localizza le parti del corpo** (0.885). Su questo è molto più forte del Base.
4. **La destra della persona è difficile per tutti:** circa a caso, e l'Instruct tende a rispondere con il lato dell'immagine. Serve il training, oppure va convertita nel codice.

### 3.1 Mani alzate: il caso dell'utente

Il caso segnalato dall'utente: davanti alla webcam (immagine non specchiata) chiede se la mano alzata è la destra, e il modello sbaglia sia con la destra sia con la sinistra. Riprodotto con 4 foto di persone viste di fronte che alzano una mano ([examples/immagini/mani](../examples/immagini/mani)), più le versioni specchiate (8 immagini):

| Domanda | 2B-Base | 2B-Instruct |
|---|---|---|
| Sì/No **«La mano alzata è la destra?»** (e in inglese) | 4/8: **sempre No** | 4/8: **sempre No** |
| "La persona ha alzato la sua mano destra o la sua mano sinistra?" | 4/8: sempre "sinistra" | 4/8: sempre "sinistra" |
| "Nell'immagine, la mano alzata è sul lato sinistro o sul lato destro?" | 7/8 | **8/8** |
| **"In quale metà dell'immagine si trova la mano alzata?"** | **8/8** | **8/8** |

- **"È la destra?" riceve sempre No**, qualunque mano sia alzata. Se la domanda era "è la destra?" con la mano destra e "è la sinistra?" con la sinistra, la risposta è stata No tutte e due le volte: sbagliata in entrambi i casi, come osservato.
- **Chiedere in quale metà dell'immagine sta la mano funziona con tutti e due i modelli.** Con la webcam di fronte e non specchiata, la **metà sinistra** dell'immagine è la **mano destra** della persona.

### 3.2 Chiralità: la mano da sola dovrebbe bastare

**L'obiezione dell'utente:** le mani sono chirali, quindi una mano da sola dice se è destra o sinistra, indipendentemente dalla cornice. Con le dita in su, un palmo destro ha il pollice a destra e un dorso destro lo ha a sinistra. **Lo specchio la mette alla prova:** specchiata, una mano destra diventa una sinistra. Se il modello usasse la chiralità, la risposta si invertirebbe.

**Dati.** 7 primi piani ([examples/immagini/mani/primi_piani](../examples/immagini/mani/primi_piani)), con palmi e dorsi, dita in su e in giù, più le versioni specchiate.

| Domanda (14 immagini) | 2B-Base | 2B-Instruct |
|---|---|---|
| **"Questa è una mano destra o una mano sinistra?"** | 0.50: sempre "destra" (in inglese quasi sempre "left") | 0.50: idem |
| …risposta invertita dallo specchio | **0 su 7** | **0 su 7** |
| "Della mano si vede il palmo o il dorso?" | 0.71 | 0.79 |
| "Il pollice è sul lato sinistro o destro della mano?" | 0.50 (sempre "sinistro") | 0.67 a 448 px, 0.83 a 800 px |
| "Le dita puntano verso l'alto o verso il basso?" | 0.83 | 0.83 |
| **Mano ricomposta nel codice** dai tre indizi | 0.58 | **0.75** a 448 px (0.58 a 800 px) |

**Cosa emerge:**
- **Il modello non usa la chiralità.** Alla domanda diretta dà una risposta fissa anche sui primi piani, a qualsiasi risoluzione, e lo specchio non la cambia mai.
- **Gli ingredienti l'Instruct li vede in parte.** Palmo/dorso e direzione delle dita vanno bene; il lato del pollice è fragile e cambia con la risoluzione.
- **La ricomposizione nel codice** ("palmo con le dita in su e pollice a destra = mano destra") arriva a 0.75: meglio del caso, non abbastanza per fidarsi.
- **È un obiettivo ideale per il training.** Ogni foto di una mano, specchiata, dà gratis l'esempio con la risposta opposta. Si può anche fare a meno delle etichette, chiedendo al modello che P(destra | foto) sia uguale a P(sinistra | foto specchiata).

## 4. 2B-Base o 2B-Instruct?

Con i default di oggi (tutte le rotazioni, correzione del Sì), sul test di typed-decisions:

| | Totale | Sì/No | Scelta | Scala |
|---|---|---|---|---|
| 2B-Base | **0.496** | 0.527 | **0.603** | 0.393 |
| 2B-Instruct | 0.475 | **0.647** | 0.432 | 0.380 |
| Differenza (Instruct − Base) | −2.1 [−3.9, −0.2] | **+12.0** [+7.2, +16.8] | **−17.2** [−20.8, −13.6] | −1.3 (n.s.) |

**Sulle immagini** (verifica della §9–10 di [09](09-controlli-senza-etichette.md)):
- eventi rari: l'Instruct è pari al Base (0.987 contro 1.000);
- pose descritte a parole: l'Instruct scambia per l'albero le altre pose su una gamba nel 50% dei casi (il Base nel 100%).

Nessuno dei due vince su tutto. Sulle immagini e sulle Sì/No l'Instruct è più forte, sul testo con scelte il Base. Due modelli da 2B in bf16 non stanno insieme negli 8 GB della GPU locale.

## 5. Cosa fare

1. **Per destra e sinistra di oggetti e figure:** funziona già, anche con il Base.
2. **Per le parti del corpo:**
   - chiedere **dove si trova** la parte ("in quale metà dell'immagine si trova la mano alzata?", 8/8; per il piede sollevato 0.77) invece di "da che parte è";
   - oppure usare il 2B-Instruct come server del modello per i compiti sulle immagini.
3. **Per la destra della persona:** chiederla nella cornice dell'immagine e convertirla nel codice. Con una webcam di fronte e non specchiata, la destra della persona è il lato sinistro dell'immagine.
4. **Per il training (F1):** lo specchio dà **etichette gratis**. Un'immagine specchiata ha la risposta invertita, e in cornice della persona cambia anche la gamba. Le pose specchiate sono un'aumentazione già pronta per insegnare entrambe le cornici.
5. **Da decidere:**
   - quale modello usare per le immagini (Base o Instruct);
   - se rimettere destra/sinistra fra i controlli senza etichette, con la coerenza con lo specchio come misura;
   - se il primo LoRA deve insegnare anche la cornice della persona.

```bash
# indagine completa (GPU, ~20 s dopo il caricamento)
.venv/bin/python scripts/indagine_destra_sinistra.py Qwen/Qwen3.5-2B-Base
.venv/bin/python scripts/indagine_destra_sinistra.py Qwen/Qwen3.5-2B
```
