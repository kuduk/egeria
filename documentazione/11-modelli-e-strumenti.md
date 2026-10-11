**Italiano** · [English](en/11-models-and-tools.md)

# Prima di addestrare: modelli, strumenti e letteratura

> **Stato:** ricerca sul web del 07/10/2026. Domanda di partenza: i difetti misurati in [10](10-destra-sinistra.md) li risolve già qualche modello o strumento esistente, senza addestrare niente?
> - Difetti misurati: la destra e la sinistra della persona, la chiralità delle mani, le parti del corpo con il 2B-Base, le pose simili fra loro.
> - Fonti complete in [fonti.md](fonti.md). Le etichette [vendor], [terzi] e [indipendente] dicono chi ha misurato.

## 1. In breve

1. **La destra e la sinistra della persona non le risolve nessun modello aperto piccolo, e nemmeno i grandi.** È un problema aperto della ricerca, non un difetto nostro (§2).
2. **Le nostre osservazioni coincidono con quelle della letteratura.** Risposte fisse quando il segnale visivo è debole; ingredienti visti ma non combinati (la chiralità è una XOR fra palmo/dorso e lato del pollice); nessun aiuto dalla risoluzione.
3. **Un modello con il nostro stesso metodo esiste già:** **Intern-Decision** (InternLM, 26/09/2026, Apache-2.0), fine-tuning di Qwen3.5 in 0.8B, 2B e 4B. Va confrontato subito (§3).
4. **I rilevatori di punti del corpo danno già destra e sinistra della persona:** MediaPipe Pose e Hand Landmarker, Apache-2.0, sulla CPU. C'è un tranello: le mani di MediaPipe presumono un'immagine specchiata (§4).
5. **Se servirà addestrare, si sa come e quanto:** un LoRA con coppie specchiate (immagine specchiata, risposta invertita), qualche migliaio di esempi per compito. AIST++ però non va bene: i video sono solo per la ricerca accademica e non ci sono pose di yoga (§5).

## 2. Cosa dice la letteratura

**La destra della persona è un problema aperto anche per i modelli grandi** [indipendente]:

| Benchmark | Cosa misura | Risultati |
|---|---|---|
| **MirrorQA** (ACL 2026) | La destra/sinistra del soggetto ("quale mano tiene…", "quale occhio è aperto"), 5.549 domande | Tutti i modelli aperti zero-shot 41.9–48.1% (a caso 42.5%). Il miglior LoRA 65.4%, gli umani 99.3%. Il 51–55% degli errori usa la cornice di chi guarda |
| **SPACE** (ACM MM 2026) | Braccio e mano destra/sinistra del soggetto | Diversi modelli aperti poco sopra il 50%. Qwen3-VL-8B 66.6–70.4, Gemini-3.1-Pro 81.1 |
| **FlipSet** (2026) | Cosa vede un'altra persona | Il 91% di 103 modelli (1–90B) è sotto il caso; il migliore 33.9% |
| **OmniSpatial** (ICLR 2026) | Cornice di un altro osservatore | Il migliore (o3) 48.4%, gli umani 94.3% |
| **3DSRBench** (ICCV 2025) | Orientamento, con il "FlipEval" a immagini specchiate | Il migliore circa 40%, gli umani 97.7% |

**Risposte fisse:** sono documentate.
- **Isle-Brick-V2:** i Qwen3-VL piccoli ripiegano su una direzione predefinita quando devono dire dove guarda una figura. Ingrandire non aiuta; gli autori parlano di "priori linguistici al posto dei fatti visivi".
- **ENACT:** trova una preferenza per la mano destra.

**Le cause proposte** che si adattano alle nostre misure:
1. **Cornice non dichiarata nei testi di addestramento.** Le didascalie dicono "a sinistra" quasi sempre nel senso di chi guarda. Per questo destra e sinistra degli oggetti funzionano, quelle della persona no.
2. **Composizione.** Il modello vede i pezzi ma non li combina:
   - la mano destra è "palmo XOR pollice a destra";
   - la destra della persona è "di fronte XOR metà dell'immagine".

   Una XOR non si legge in modo lineare dai due ingredienti separati.
3. **Ripiego su una risposta abituale quando il segnale visivo è debole.** Spiega le risposte fisse, lo specchio che non le cambia e la risoluzione che non aiuta.
4. **Base contro Instruct:** l'addestramento a istruzioni di Qwen3-VL contiene frasi di relazione ("la tazza a sinistra del portatile"). È la spiegazione probabile del perché solo l'Instruct trova la gamba sollevata.

**Lo specchiamento durante l'addestramento dell'encoder visivo probabilmente non è la causa.**
- CLIP, OpenCLIP e SigLIP 2 non lo usano nelle perdite principali.
- L'encoder dei Qwen3.5 piccoli ha le dimensioni di SigLIP2 (deduzione dalle configurazioni: Qwen non lo documenta).
- Lo usa invece Meta Perception Encoder (in Perception-LM e Step3-VL-10B): da evitare per la chiralità.

**Le correzioni con più prove**, in ordine di costo:
1. **Scomporre la domanda e combinare le risposte nel codice**, senza training: quello che già funziona da noi ("in quale metà dell'immagine…").
2. **Strumenti:** rilevatori di punti del corpo (§4).
3. **LoRA con coppie specchiate:**
   - **CounterCurate:** destra/sinistra da 61.8 a 95.7%.
   - **"Art of Interrogation" (2026):** LoRA r=32 su Qwen2.5-VL-3B con 2.4–7.7 mila esempi per compito; una versione **senza etichette**, che premia solo la coerenza con lo specchio, arriva a 74.1 contro 76.4 della supervisionata.
   - **Plateau:** intorno ai 4 mila esempi (AlloEgo-VLM).
4. **Modelli "Thinking" o più grandi:** guadagni modesti (+3.8 punti fra Instruct e Thinking su SPACE). Il ragionamento passo passo spesso **peggiora** le domande spaziali.

**Prima di un LoRA sulla chiralità**, una prova economica: una sonda lineare sulle caratteristiche dell'encoder visivo, con coppie specchiate tenute da parte. Se non separa la mano destra dalla sinistra, un LoRA sul solo modello linguistico non basterà.

## 3. Modelli da provare

Nessun modello piccolo ha numeri pubblicati sulla destra della persona, sulle mani o sulle pose. Quelli da misurare sui nostri controlli, in ordine di costo:

| Modello | Perché | Costo per provarlo |
|---|---|---|
| **Intern-Decision-2B / 4B** (Apache-2.0, su Qwen3.5) | Stesso metodo di Egeria: un solo passaggio, logit delle lettere, stato condiviso, fino a 8 immagini, calibrazione. Dichiara typed-decisions 79.35 (2B) e 80.55 (4B) [vendor], contro il nostro 0.496 zero-shot. Nessun numero sulle immagini | Nessuno: stessa architettura del nostro scorer. Il suo formato di prompt (scheletro JSON con un segnaposto per domanda) è diverso dal nostro, e va provato in tutti e due i formati |
| **Qwen3.5-4B e 9B**, Base e Instruct, in 4 bit | Dice se la taglia da sola risolve le parti del corpo e le pose. Il 9B ha un encoder visivo più grande (dimensioni di SigLIP2-So400m). Il 4B batte il 2B su BLINK e CV-Bench [indipendente e terzi] | Basso: stesso codice, serve `bitsandbytes` per il 4 bit |
| **Qwen3-VL-4B-Instruct** e **Spatial-SSRL-Qwen3VL-4B** | Qwen3-VL inietta nel modello linguistico le caratteristiche di più strati dell'encoder (DeepStack) e ha dati di relazioni spaziali. SpinBench 61.4 [indipendente]. Spatial-SSRL è addestrato anche a riconoscere pezzi d'immagine specchiati | Medio: architettura diversa, va adattato lo scorer |
| **InternVL3.5-2B / 4B** (anche `-Pretrained`) | Il migliore fra i piccoli su SpinBench (4B: 65.8) [indipendente]. Encoder non addestrato con lo specchio. La coppia pre-addestrato/finale ripete il nostro confronto Base/Instruct | Medio: architettura diversa |
| **HY-Embodied-0.5 (2B attivi)** o **VST-3B** | I migliori numeri sotto i 4B sulla prospettiva (ViewSpatial e 3DSRBench, 53–57) [vendor] | Medio-alto: versione di transformers fissata, licenza da verificare |

Esclusi:
- **per licenza:** FastVLM, Isaac, Perception-LM (non commerciali);
- **per lo specchiamento nell'encoder:** Perception-LM e Step3-VL;
- **per architettura:** Florence-2 (senza ragionamento).

**Altri benchmark utili come set di valutazione:**
- **MirrorQA:** dati su GitHub. Il paper indica CC BY 4.0, GitHub dice "Other": da verificare.
- SpinBench, ViewSpatial-Bench, 3DSRBench.
- Per la chiralità delle mani e per le pose zero-shot **non esiste un benchmark**: i nostri set restano necessari.

## 4. Strumenti: rilevatori di punti del corpo

| Strumento | Destra/sinistra | Licenza | Dove gira | Note |
|---|---|---|---|---|
| **MediaPipe Pose Landmarker** | 33 punti con nome sinistro/destro, coordinate 3D | Apache-2.0 | Python e browser, 25–50 ms su CPU | La documentazione non dice di chi è la sinistra: va verificato. Debole se la testa non si vede (persona di schiena) |
| **MediaPipe Hand Landmarker** | 21 punti per mano, etichetta "Left/Right" | Apache-2.0 | Python e browser, ~17 ms su CPU | **Presume un'immagine specchiata** (selfie): sui nostri fotogrammi, che non lo sono, l'etichetta va **invertita**. Accuratezza non pubblicata; errori noti vicino ai bordi e con una sola mano |
| **RTMW / RTMPose** (via `rtmlib`, ONNX) | 133 punti, mani comprese | Codice Apache-2.0; pesi da dataset con licenze miste | CPU, 11–45 ms | Seconda opinione per persone di schiena o più persone |
| Da evitare | | | | YOLO-pose (AGPL), OpenPose e Sapiens v1 (non commerciali), WiLoR (NC-ND) |

**Uno schema ibrido possibile:**
- **Dove:** MediaPipe Pose e Hand sul server, sulla CPU, così la GPU resta al modello.
- **Le mani:** si assegnano al polso più vicino dello scheletro; l'etichetta di MediaPipe, invertita, si usa solo per i primi piani senza corpo.
- **Due usi:**
  1. le domande fisse ("alza la mano destra?") si risolvono direttamente dai punti;
  2. per le altre, i fatti entrano nello stato come testo: "la mano destra della persona è sopra la testa", "la persona è di fronte".
- **Controllo con lo specchio:** sul fotogramma specchiato le etichette devono scambiarsi, altrimenti il fatto è incerto. È lo stesso principio dei nostri controlli senza etichette.
- **Risoluzione:** i fotogrammi oggi sono a 448 px e le mani vengono piccole. Al rilevatore va mandato il fotogramma più grande, o un ritaglio attorno ai polsi.
- **Rischi:**
  - il 2B può ignorare i fatti nello stato, va misurato;
  - persone di schiena e più persone;
  - punti che tremano da un fotogramma all'altro;
  - 25–45 ms di CPU per modello e per fotogramma.
- **Pose di yoga:** albero e ballerino si distinguono con regole sui punti 3D (piede sollevato contro la gamba d'appoggio, oppure dietro i fianchi). Non esiste un'accuratezza pubblicata delle regole: va misurata sulle nostre foto.

## 5. Dati, se servirà un LoRA

- **AIST++:** le annotazioni sono CC BY 4.0, ma **video e immagini sono solo per la ricerca accademica**. L'uso commerciale richiede il consenso scritto di AIST e la ridistribuzione è vietata. In più è street dance, senza yoga. **Non adatto** al primo LoRA.
- **Candidati migliori:**
  - il train di **MirrorQA** (~3.9 mila esempi, licenza da verificare);
  - le **nostre coppie specchiate**, che danno etichette gratis;
  - foto con licenza libera etichettate automaticamente da MediaPipe, verificate con lo specchio.
- **Per la nostra lettura a lettere,** l'equivalente economico della "coerenza con lo specchio" è una perdita KL simmetrica fra P(destra | foto) e P(sinistra | foto specchiata). Non richiede etichette (nostra proposta, non provata in letteratura).

## 6. Piano proposto, prima di addestrare

Aggiornato l'08/10/2026 con le idee del §7:
1. **Il segnale "il modello sta guardando?"** (§7.1). Serve anche al confronto, per vedere quale modello guarda davvero.
2. **Confronto dei modelli (in locale, ~1 giorno).**
   - Prima Intern-Decision-2B/4B, poi Qwen3.5-4B/9B (Base e Instruct, in 4 bit; serve `bitsandbytes`), sui nostri controlli: [indagine su destra e sinistra](../scripts/indagine_destra_sinistra.py), [verifica della calibrazione e pose](../scripts/verifica_calibrazione.py), typed-decisions e suite immagini.
   - Poi, con un piccolo adattamento dello scorer, Qwen3-VL-4B e InternVL3.5-4B.
3. **Domande derivate** (§7.2).
4. **Solo se 1–3 non bastano:** la sonda lineare sull'encoder, poi un LoRA con coppie specchiate (MirrorQA + nostre) e le coerenze senza etichette (§7.3).
5. **A priorità bassa:** gli strumenti di supporto (§4 e §7.4).

## 7. Idee riutilizzabili, oltre destra e sinistra (08/10/2026)

Decise con l'utente e messe nella roadmap ([02](02-implicazioni-e-proposta.md) §6).

1. **Il modello sta guardando?** (priorità alta) La lezione più utile della ricerca: quando non sa, il modello dà sempre la stessa risposta e non lo segnala. Tre prove senza etichette:
   - **stato vuoto:** risposta sullo stato vero quasi uguale a quella su "N/A" o su un'immagine grigia (lo stato vuoto lo calcoliamo già per la correzione del Sì);
   - **rotazioni:** le rotazioni delle opzioni non concordano (MirrorQA la chiama valutazione "circolare");
   - **specchio**, per le domande di lato: la risposta non cambia.

   Insieme danno un segnale nella risposta e lo stato `uncertain`.
2. **Domande derivate** (priorità alta). Il modello percepisce bene le cose semplici e sbaglia quando deve combinarle (la XOR del §2). Nell'API si dichiarano le domande semplici e una regola (non, e, o, XOR, soglie); la probabilità della domanda derivata si calcola da quelle semplici. Esempi:
   - "è la mano destra della persona" = "nella metà sinistra" XOR "di spalle";
   - "è una mano destra" = palmo XOR "pollice a destra" (con le dita in su);
   - "l'auto è contromano" = lato della strada XOR senso di marcia.
3. **Le simmetrie come dati gratis** (nel primo LoRA). I controlli senza etichette diventano perdite di addestramento senza etichette (KL simmetrica fra le due risposte):
   - **specchio:** P(destra | foto) = P(sinistra | foto specchiata);
   - **negazione:** P(Sì | A) = P(No | non A);
   - **lingua:** stessa risposta in italiano e in inglese;
   - **rotazioni:** stessa distribuzione con qualunque ordine delle opzioni.

   Permettono anche di adattare il modello ai propri dati (per esempio ai fotogrammi di una webcam) senza etichettarli.
4. **Strumenti che danno fatti al modello** (priorità bassa):
   - MediaPipe, anche per mandare dal vivo solo lo scheletro senza immagini, per i ricordi di pose (somiglianza dei punti, come il k-NN di MediaPipe) e per i gesti;
   - OCR per i testi minuti;
   - un rilevatore per contare;
   - codici a barre e QR.

**Cosa si può prendere da Intern-Decision** (da verificare nel confronto dei modelli):
- **Una possibile scorciatoia per il testo.** È Qwen3.5 già addestrato con il nostro metodo, con licenza Apache-2.0. Se il suo 79 dichiarato su typed-decisions regge sui nostri controlli, contro il nostro 50 zero-shot, i suoi pesi potrebbero diventare la base di Egeria per le domande di testo e risparmiare buona parte di F2.
  - Cautele: i numeri sono suoi; probabilmente ha visto il train di typed-decisions; sulle immagini non ha numeri.
- **Velocità.** Mette tutte le domande in un solo passaggio: uno scheletro JSON con un segnaposto per domanda, e i logit letti prima di ciascuno. Noi facciamo un passaggio per domanda e per rotazione, sia pure riusando lo stato. Va misurato se è più veloce e se le domande si disturbano.
- **Dettagli minori:** 62 simboli per le opzioni invece di 26 (A–Z, a–z, 0–9); la calibrazione distribuita insieme al checkpoint.

## 8. Documentati per dopo

Spunti della ricerca che per ora non entrano nella roadmap:
- **Il ragionamento passo passo peggiora le domande spaziali** (13 benchmark, https://arxiv.org/abs/2604.16060; FlipSet). Conferma la scelta "System 1": un argomento per il README.
- **Piccole teste addestrate sulle caratteristiche dell'encoder** come alternativa leggera al LoRA per i compiti di percezione. L'informazione spesso c'è nell'encoder ma non viene letta (VLMs are blind, Theodoridis et al. 2026); è la stessa idea delle teste di CLM ([01](01-stato-dell-arte.md) §5.5).
- **Modelli che indicano un punto** (Molmo2, Moondream): una via già pronta per la primitiva `locate`.
- **Set di valutazione esterni:** MirrorQA (licenza da verificare), SpinBench, ViewSpatial-Bench, 3DSRBench, con la valutazione "circolare" (giusta solo se lo è per ogni ordine delle opzioni).
