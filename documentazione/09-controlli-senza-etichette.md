**Italiano** · [English](en/09-label-free-checks.md)

# Controlli senza etichette

> Stato: **eseguiti** il 29/09/2026 (F0.5, punto 2 della roadmap in [02-implicazioni-e-proposta.md](02-implicazioni-e-proposta.md) §6), sul 2B e sullo 0.8B. Nessuna delle tre soglie è superata, ma i risultati cambiano due decisioni: il numero di permutazioni e il passo 3 della roadmap (§5). Le correzioni scelte (§8–9) sono **nel codice dal 29/09/2026**: tutte le rotazioni e la correzione del Sì di default, la calibrazione sullo storico facoltativa.

## 1. Cosa misurano

Tre difetti che si misurano senza etichette, confrontando il modello con sé stesso:

1. **Posizione delle opzioni.** Domande `choice` con opzioni tutte identiche ("Opzione", "Opzione", …): un modello senza preferenze di posizione dà probabilità uguali a tutte. Si misura la distanza di variazione totale (TV) media dalla distribuzione uniforme, da 2 a 5 opzioni, con 1, 2 e tutte le permutazioni (rotazioni cicliche, come nello scorer).
2. **Tendenza al "Sì".** Ogni affermazione Sì/No è posta anche negata, con le descrizioni di vero e falso scambiate. Se il modello è coerente, P(Sì | A) + P(Sì | non A) ≈ 1: l'**eccesso** sopra 1 è la tendenza al "Sì". Una **contraddizione** è la stessa risposta ad A e a non A. Variante: la stessa affermazione come `choice` con l'opzione "lo stato non permette di dirlo".
3. **Lingua.** La stessa domanda sullo stesso stato in italiano e in inglese: quota di risposte uguali (**accordo**) e TV media fra le due distribuzioni.

**Dati** ([examples/controlli/](../examples/controlli/)), tutti generici e senza dati personali:
- **negazioni** ([negazioni.json](../examples/controlli/negazioni.json)): 348 coppie. Sono 300 in inglese su 200 casi di typed-decisions (6 affermazioni), 20 sulla suite italiana (6 affermazioni, in italiano e in inglese) e 28 sulle immagini (14 affermazioni, di cui 6 false apposta, nelle due lingue);
- **traduzioni** ([lingue.json](../examples/controlli/lingue.json)): 1051 domande confrontate. Sono le 20 domande di typed-decisions in italiano (su 200 casi), le 15 della suite italiana in inglese e le 26 delle immagini in inglese;
- **posizione:** 81 stati (60 di typed-decisions, 12 della suite, 9 immagini).

**Soglie fissate prima dei risultati:**

| Controllo | Soglia |
|---|---|
| Posizione | TV media ≤ 0.05 per ogni numero di opzioni |
| "Sì" | eccesso ≤ 0.05 e contraddizioni ≤ 10% |
| Lingua | accordo ≥ 90% e TV media ≤ 0.10 |

Script: [scripts/controlli_senza_etichette.py](../scripts/controlli_senza_etichette.py). Il rapporto completo va in `runs/controlli/`.

## 2. Posizione: preferenza forte per la A, e 2 permutazioni non bastano

**Con opzioni identiche e una sola permutazione il modello sceglie la A:**

| | 2 opzioni | 3 | 4 | 5 |
|---|---|---|---|---|
| 2B, probabilità media della A | 0.76–0.82 | 0.81–0.87 | 0.82–0.86 | 0.84–0.88 |
| 0.8B, probabilità media della A | 0.93–0.96 | 0.88–0.92 | 0.86–0.91 | 0.86–0.91 |

(Gli intervalli vanno dalle istruzioni in italiano a quelle in inglese.)

**TV media dalla distribuzione uniforme** (la peggiore fra italiano e inglese):

| Permutazioni | 2B | 0.8B |
|---|---|---|
| 1 | 0.71 | 0.71 |
| 2 (il default di oggi) | **0.41** | **0.38** |
| tutte le rotazioni | **0.00** | **0.00** |

Con 2 opzioni le 2 permutazioni bastano. **Da 3 opzioni in su no:** con 2 rotazioni un'opzione non passa mai in posizione A. Con 3 opzioni identiche il 2B dà in media 0.47 / 0.10 / 0.42, e quella centrale è penalizzata senza motivo. Con tutte le rotazioni ogni opzione passa per ogni posizione, e la distorsione sparisce per costruzione.

**Verifica con le etichette** (test di typed-decisions, 400 casi, 2000 decisioni, 2B, senza calibrazione):

| Permutazioni | Accuratezza | `choice` | NLL | ECE | Tempo per caso |
|---|---|---|---|---|---|
| 1 | 0.458 | 0.478 | 1.385 | 0.243 | 150 ms |
| 2 | 0.469 | 0.525 | 1.193 | 0.224 | 239 ms |
| **tutte** | **0.493** | **0.603** | 1.197 | **0.204** | 331 ms |

Confronto appaiato, bootstrap per caso con intervallo al 95%:
- **tutte contro 2, sul 2B:** accuratezza +2.4 punti [+1.5, +3.2], sulle `choice` **+7.8 punti [+5.2, +10.6]**, Brier migliore, NLL invariata. Le `noul` e le `score` non cambiano: con 2 opzioni, o con l'ordine diretto e inverso delle scale, le 2 permutazioni sono già complete;
- **1 contro 2:** peggio in modo significativo (accuratezza −1.2, NLL +0.19);
- **0.8B, tutte contro 2:** nessuna differenza (−0.4 punti, intervallo che include lo zero), tempo da 145 a 188 ms per caso.

**Conclusioni:**
- **Una sola permutazione è da escludere.**
- **Con tutte le rotazioni il 2B zero-shot arriva a 0.493 e supera per la prima volta la prior (0.478).** La preferenza per la A copriva risposte che il modello "sa".
- **Il costo** col riuso dello stato è +38% di tempo sul 2B e +30% sullo 0.8B, non il doppio o il triplo.

## 3. "Sì": due difetti diversi, non uno

**Riepilogo** (eccesso, contraddizioni, e fra queste la quota "entrambe Sì" o "entrambe No"):

| Fonte | 2B: eccesso | 2B: contraddizioni | 0.8B: eccesso | 0.8B: contraddizioni |
|---|---|---|---|---|
| typed-decisions (en) | +0.36 | 73% (70% entrambe Sì) | +0.43 | 100% (tutte entrambe Sì) |
| suite italiana (it) | −0.15 | 20% (entrambe No) | +0.36 | 100% (entrambe Sì) |
| suite italiana (en) | −0.06 | 30% | +0.40 | 100% (entrambe Sì) |
| immagini (it) | −0.28 | 43% (entrambe No) | +0.22 | 79% (entrambe Sì) |
| immagini (en) | −0.23 | 36% (entrambe No) | +0.26 | 86% (entrambe Sì) |

**Cosa emerge:**
- **Lo 0.8B dice "Sì" a tutto,** anche alle negazioni: è la tendenza al "Sì" vera e propria.
- **Il 2B su stati astratti (JSON di typed-decisions) dice "Sì" sia all'affermazione sia alla negazione.** Per esempio "This trace requires human review" 0.90 e "does not require" 0.78, e così in 4 affermazioni su 6.
- **Il 2B sui casi concreti (immagini, suite) riconosce bene le affermazioni vere e le loro negazioni false:** "c'è un incendio in corso" 0.98, "non c'è" 0.01. **Fatica però a dire "Sì" a una negazione vera:** "la forma *non* è un quadrato" 0.35, "lo scontrino *non* è di una farmacia" 0.22. Qui le contraddizioni sono tutte "entrambe No".
- **L'opzione "lo stato non permette di dirlo" aiuta poco e non ovunque:** sul 2B le contraddizioni su typed-decisions scendono dal 73% al 56%, sulle immagini in italiano dal 43% al 29%, ma sulla suite aumentano. Non è una correzione.

**Conseguenza per il passo 3 della roadmap** (calibrazione con spostamento): **un unico termine di spostamento non basta.** La direzione cambia con il modello (0.8B verso il Sì) e con il dominio (2B verso il Sì sugli stati astratti, verso il No sulle negazioni vere). Uno spostamento per dominio richiede etichette di quel dominio.

**Consiglio pratico, da subito:** scrivere le affermazioni in forma positiva ("il pacco è arrivato", non "il pacco non è arrivato"). Sulle negazioni il modello è inaffidabile, sia il 2B sia lo 0.8B.

## 4. Lingua: il disaccordo viene dall'incertezza più che dalla lingua

| | 2B: accordo | 2B: TV | 0.8B: accordo | 0.8B: TV |
|---|---|---|---|---|
| tutte (1051 domande) | 0.76 | 0.17 | 0.74 | 0.14 |
| typed-decisions (1000) | 0.75 | 0.17 | 0.73 | 0.14 |
| suite italiana (25) | **0.96** | 0.08 | **0.96** | 0.08 |
| immagini (26) | **1.00** | 0.03 | **0.96** | 0.05 |
| **risposte sicure** (p ≥ 0.7 in almeno una lingua) | **0.91** (614) | 0.16 | **0.98** (311) | 0.11 |

**Cosa emerge:**
- **La soglia non è superata sull'insieme,** ma dove il modello è sicuro l'accordo supera il 90%, e sui casi chiari è del 96–100%.
- **Il disaccordo si concentra su typed-decisions,** dove il modello è quasi a caso anche in una lingua sola.
- **Più che la lingua pesa l'incertezza.** Le distribuzioni però restano diverse anche sulle risposte sicure (TV 0.11–0.16): la lingua sposta la confidenza più della risposta.

## 5. Decisioni e prossimi passi

1. **Permutazioni.** *(Dal 29/09/2026 il default è `auto`, tutte le rotazioni: §9.)* Il default era 2. Proposta: **tutte le rotazioni** per le `choice`, perché sul 2B danno +7.8 punti e una calibrazione migliore, al costo di +38% di tempo. Si ottiene già oggi con `--permutations 26` (26 è il numero massimo di opzioni; le `noul` e le `score` restano a 2). Dal 29/09/2026 una singola richiesta può chiedere fino a 26 permutazioni anche al server del modello (`"permutations"`).
2. **Passo 3 (calibrazione con spostamento):** da rivedere, perché uno spostamento globale non corregge difetti che cambiano direzione con modello e dominio. Le negazioni diventano dati per il training: coppie affermazione/negazione con etichette opposte, già previste fra le aumentazioni di F1.
3. **Questo script diventa una prova di regressione:** si rilancia dopo ogni LoRA (F1, F2) per vedere se posizione, "Sì" e lingua migliorano.
4. **Destra/sinistra** resta fuori, per l'indagine a parte.

```bash
# rilanciare i controlli (GPU; circa 2–4 minuti)
.venv/bin/python scripts/controlli_senza_etichette.py Qwen/Qwen3.5-2B-Base
# permutazioni su typed-decisions con le etichette
.venv/bin/egeria predict --model Qwen/Qwen3.5-2B-Base --split test --permutations 26 --out runs/perm-tutte/test.jsonl
.venv/bin/egeria paired --a runs/Qwen3.5-2B-Base/test.jsonl --b runs/perm-tutte/test.jsonl
```

## 6. Cosa fanno Laya e SemIf (codice letto il 29/09/2026)

| | Laya ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), commit 9d95567) | SemIf ([TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), commit 23cf1f3) | Egeria |
|---|---|---|---|
| **Posizione delle opzioni** | La **misura**: opzioni identiche per `score` (`research/eval/presentation_checks.py`) e ordine rimescolato con etichette rese opache (`metamorphic.py`). Il checkpoint inglese preferisce le prime posizioni, di più con più opzioni; quello multilingua evita la prima (#131). Cambi di risposta con l'ordine: 28% e 10.6% su due compiti. La correzione prevista è un **riaddestramento** bilanciato sulle posizioni. In inferenza un solo passaggio, nessuna media | La **misura** soltanto: opzioni invertite, 10 cambi di risposta su 36 casi, "non risolto" nel loro rapporto. Un'alternativa (reranker con un Sì/No indipendente per opzione) è invariante all'ordine per costruzione ma meno accurata (0.53 contro 0.72) | Media sulle permutazioni in inferenza; con **tutte le rotazioni** la distorsione sparisce (§2) |
| **Negazioni e "Sì"** | Difetto documentato e non corretto: nelle richieste negate la scelta segue la domanda, non lo stato (#377). `noul` può seguire le etichette `false`/`true` invece dello stato (#156): rimedio con etichette opache (`labels`: `A`/`B`) | Ogni decisione ha tre opzioni fisse: supportato, **insufficiente**, contraddetto. Coppie con il criterio rovesciato **e le etichette**: devono risultare giuste entrambe (soglia 20 coppie su 24). Stati senza l'informazione: deve scegliere "insufficiente" | Coppie affermazione/negazione senza etichette (§3); `noul` con le lettere A/B nei due ordini, quindi senza etichette "true/false" |
| **Lingua** | Instradamento per lingua a un checkpoint multilingua. Coerenza misurata fra cinese semplificato e tradizionale: 12.8% di risposte diverse | Non trattata | Stessa domanda in italiano e inglese (§4) |
| **Calibrazione** | Una temperatura per tipo e numero di opzioni | Una temperatura per carico di lavoro, fittata sui suoi dati | Una temperatura per tipo; lo spostamento è da rivedere (§3) |

**Cosa si può riprendere:**
- **Da SemIf, tre controlli che mancano:** contesto irrilevante aggiunto allo stato, criterio riformulato con lo stesso significato, e stati privi dell'informazione, dove la risposta giusta è "non si può dire". I primi due si misurano senza etichette come gli altri; il terzo ha l'etichetta per costruzione.
- **Da Laya, il test delle etichette opache:** sostituire le chiavi delle opzioni (per esempio `spam_truffa`) con lettere neutre, per separare l'effetto delle parole delle etichette da quello della posizione.
- **Per il training (F1):** dati bilanciati sulle posizioni, come prevede Laya, e coppie con il criterio rovesciato e le etichette, come SemIf.

**Dove Egeria è avanti:** nessuno dei due corregge la posizione in inferenza. Laya la lascia al riaddestramento, SemIf la misura soltanto. La media su tutte le rotazioni la elimina già oggi, e col riuso dello stato costa +38% di tempo invece di un riaddestramento.

## 7. Stato dell'arte e strumenti matematici (ricerca del 29/09/2026)

**Chi ha lavorato su questi problemi:**

| Problema | Lavori | Cosa propongono |
|---|---|---|
| Posizione delle opzioni ("selection bias") | Zheng et al., ICLR 2024 (**PriDe**) | La distorsione viene soprattutto dalla preferenza per i token delle lettere. Si stima la preferenza a priori per ogni lettera permutando le opzioni su pochi esempi, poi si divide via dalle previsioni degli altri: una sola passata, senza etichette |
| | Choi et al., ACL 2025 | **Potatura dei nodi di distorsione** (0.002% dei parametri della proiezione d'uscita) e un'opzione ausiliaria "non lo so". Nuova metrica, la Choice KL Divergence |
| | Guda et al., IJCNLP-AACL 2026 | Metrica di distorsione per permutazione senza etichette, LoRA non supervisionato, e **voto su tutte le permutazioni con la cache di domanda e contesto condivisa**: è lo stesso principio del nostro riuso dello stato |
| | Zheng et al., ACL 2026 (**PA-GRPO**) | Training con gruppi di permutazioni e una ricompensa per la coerenza fra ordini diversi |
| "Sì"/"No" e negazioni | Braun, EMNLP Findings 2025 | 37.975 varianti in inglese, tedesco e polacco: in inglese gli LLM tendono al **"No"**, il contrario dell'acquiescenza umana. È coerente con il nostro 2B sui casi concreti |
| | Huang, luglio 2026 | "Simmetrizzazione incrociata": si invertono in modo bilanciato ordine delle risposte, parole e verdetto. La tendenza viene dalla forma (ordine, la parola "no"), non dal giudizio. Modello P = σ((θ ± m)/s): m misura la sensibilità alla formulazione, s quanto il modello è deciso |
| | Burns et al., ICLR 2023 (**CCS**) | Coppie affermazione/negazione e il vincolo di coerenza P(A) ≈ 1 − P(non A), usati per trovare senza etichette una direzione "vero/falso" negli stati interni |
| Lingua | Qi et al., EMNLP 2023 (**RankC**); lavori del 2025–2026 sulla coerenza fra lingue | Metrica di coerenza sulle risposte classificate; insiemi di più lingue con voto; training con ricompensa sulla coerenza fra lingue |
| Calibrazione senza etichette | Zhao et al., ICML 2021 (Calibrate Before Use); Zhou et al., ICLR 2024 (**Batch Calibration**) | Si stima la preferenza "di contesto" con un input vuoto o con la media delle previsioni su un lotto dello stesso carico, e la si divide via |

**Strumenti matematici e come si applicano a noi:**

1. **Media su un gruppo di trasformazioni (quadrato latino).** Se il logit di un'opzione è `contenuto(opzione) + b(posizione)`, la media dei log-probabilità su tutte le rotazioni cicliche fa passare ogni opzione per ogni posizione una volta: il termine `b` diventa uguale per tutte e si cancella **esattamente**. Con 2 rotazioni e 3 o più opzioni questo non succede, ed è il risultato della §2. Costa n passaggi (le code, col riuso dello stato).
2. **Preferenza a priori stimata una volta (PriDe).** Si stima `b(lettera)` una volta, per numero di opzioni e lingua, dalle opzioni identiche della §2 (sono esattamente un input "senza contenuto"), poi si usa **una sola permutazione** togliendo `b` dai logit delle lettere. Nel nostro readout è una riga: i logit delle lettere sono `hidden @ slot_weight.T`, e basta sottrarre un vettore. Promette la qualità di tutte le rotazioni al costo di una.
3. **Simmetrizzazione della polarità** (dalla psicometria: item bilanciati contro l'acquiescenza; la stessa idea di CCS). Si pone la domanda e la sua negazione e si combinano: `logit P*(A) = (logit P(A) − logit P(non A)) / 2`. Se la tendenza è un termine additivo nei logit, si cancella esattamente, qualunque sia la sua direzione. È ciò che serve per un difetto che cambia verso con il dominio (§3). Costa il doppio; la negazione va scritta, oppure generata con un modello fisso ("Non è vero che…").
4. **Calibrazione di lotto o di contesto** (Batch Calibration). Si divide via la media delle previsioni sul carico di lavoro, senza etichette. Corregge una tendenza diversa per ogni dominio, se il lotto è abbastanza grande e bilanciato.
5. **Opzione ausiliaria "non lo so".** In letteratura aiuta su alcuni modelli; da noi ha dato risultati misti (§3).
6. **Nel training (F1):** coerenza fra permutazioni (PA-GRPO, LoRA guidato dalla metrica di permutazione), dati bilanciati sulle posizioni, coppie con la polarità invertita e le etichette, e la coerenza fra lingue come obiettivo.

**Prove possibili subito, con i dati che abbiamo** (etichette di typed-decisions):
- preferenza a priori (2) contro tutte le rotazioni (1): accuratezza e tempo;
- simmetrizzazione della polarità (3) sulle 600 domande Sì/No, con le negazioni scritte a mano e con una negazione automatica;
- calibrazione di lotto (4) sulla tendenza al "Sì";
- insieme italiano più inglese per la lingua.

## 8. Correzioni a basso costo: risultati (29/09/2026)

Script: [scripts/esperimento_debiasing.py](../scripts/esperimento_debiasing.py). Una sola passata del modello sul test di typed-decisions calcola i logit delle lettere per tutte le rotazioni, per lo stato vuoto ("N/A") e per le Sì/No negate; poi le varianti si simulano sugli stessi numeri. Valutazione su 380 casi (i 20 di taratura esclusi), senza temperatura. Tempi per caso dalle misure sul 2B della §2.

**2B:**

| Variante | Accuratezza | Sì/No | Scelta | Scala | NLL | ECE | Tempo per caso |
|---|---|---|---|---|---|---|---|
| 2 permutazioni (oggi) | 0.471 | 0.511 | 0.519 | 0.405 | 1.193 | 0.220 | ~240 ms |
| tutte le rotazioni | 0.494 | 0.511 | 0.596 | 0.405 | 1.197 | 0.201 | ~330 ms |
| 1 permutazione + priore fisso per lettera (PriDe) | 0.422 | 0.507 | 0.449 | 0.338 | 1.456 | 0.320 | ~150 ms |
| 1 permutazione + priore dallo stato vuoto | 0.485 | 0.535 | 0.498 | 0.437 | 1.287 | 0.202 | ~150 ms + 1 passata per domanda |
| 2 permutazioni + input vuoto | 0.473 | 0.533 | 0.511 | 0.399 | **1.065** | **0.153** | ~240 ms + 1 passata per domanda |
| **1 permutazione + lotto (stima su altri casi)** | **0.529** | 0.605 | 0.528 | 0.474 | 1.023 | **0.053** | **~150 ms** |
| 2 permutazioni + lotto (stima su altri casi) | 0.540 | 0.625 | 0.505 | 0.503 | 1.005 | 0.073 | ~240 ms |
| tutte le rotazioni + lotto (stima su altri casi) | **0.545** | 0.625 | 0.523 | 0.503 | **1.000** | 0.080 | ~330 ms |
| tutte le rotazioni + lotto, storico di soli 20 casi | 0.516 | 0.635 | 0.451 | 0.475 | 1.084 | 0.065 | ~330 ms |
| Sì/No simmetrizzate, negazione scritta a mano | – | 0.556 | – | – | – | – | Sì/No al doppio |
| Sì/No simmetrizzate, negazione automatica | – | 0.505 | – | – | – | – | Sì/No al doppio |

Accuratezza rispetto a oggi, bootstrap per caso (intervallo al 95%): tutte le rotazioni +2.3 [+1.5, +3.2]; priore dallo stato vuoto +1.4 [+0.1, +2.7]; **1 permutazione + lotto +5.8 [+3.0, +8.6]**; 2 permutazioni + lotto +6.9 [+4.3, +9.5]; tutte le rotazioni + lotto +7.4 [+4.7, +10.1].

**0.8B:**

| Variante | Accuratezza | Sì/No | NLL | ECE |
|---|---|---|---|---|
| 2 permutazioni (oggi) | 0.446 | 0.504 | 1.184 | 0.123 |
| **2 permutazioni + input vuoto** | **0.459** | 0.525 | 1.097 | **0.035** |
| 1 permutazione + lotto (stima su altri casi) | 0.426 | 0.568 | 1.112 | 0.028 |
| tutte le rotazioni + lotto (stima su altri casi) | 0.446 | 0.558 | 1.117 | 0.058 |
| Sì/No simmetrizzate, negazione scritta a mano | – | **0.602** | – | – |

Sullo 0.8B le rotazioni e il lotto non migliorano l'accuratezza (differenze entro ±3 punti, intervalli che includono lo zero). La calibrazione migliora molto: ECE da 0.12 a 0.03–0.06.

**Cosa emerge:**
1. **La preferenza fissa per lettera (PriDe) peggiora** su entrambi i modelli: la distorsione di posizione dei nostri modelli piccoli cambia da caso a caso, non è un vettore costante. Da scartare.
2. **La calibrazione di lotto è la correzione più forte e non costa nulla in inferenza.** Sul 2B con 1 sola permutazione fa +5.8 punti, ed è il 37% più veloce di oggi. Recupera quasi tutto il beneficio delle rotazioni, perché la distorsione è abbastanza stabile *dentro la stessa domanda*.
   - **Serve uno storico per domanda:** con ~50 casi funziona come sugli stessi dati, con ~5 le domande a scelta peggiorano.
   - **Non va usata per eventi rari.** Spinge la media delle previsioni verso l'equilibrio: su "c'è un incendio?", dove quasi sempre la risposta è "No", creerebbe falsi allarmi.
3. **La calibrazione a input vuoto** costa una passata per domanda, memorizzabile, e non richiede storico. Migliora la calibrazione su entrambi i modelli (ECE: 2B da 0.22 a 0.15, 0.8B da 0.12 a 0.035) con un piccolo guadagno di accuratezza. È la più sicura come default.
4. **La simmetrizzazione della polarità** funziona solo con negazioni scritte da una persona: +4.5 punti sul 2B e +10 sullo 0.8B nelle Sì/No. Con la negazione automatica non c'è guadagno. Costa il doppio sulle Sì/No.
5. **Per le scale, tutte le rotazioni cicliche peggiorano** (0.404 contro 0.405 sul 2B, NLL da 1.20 a 1.54): per le scale si restano l'ordine diretto e quello inverso.

**Per tipo di domanda** (stessi dati, accuratezza ed ECE; analisi del 29/09/2026). L'input vuoto non aiuta allo stesso modo tutti i tipi:

| Tipo | Variante | 2B | 0.8B |
|---|---|---|---|
| Sì/No | 2 permutazioni (oggi) | 0.511 · ECE 0.240 | 0.504 · ECE 0.216 |
| Sì/No | **+ input vuoto** | **0.533 · ECE 0.131** | **0.525 · ECE 0.040** |
| Scelta | 2 permutazioni (oggi) | 0.519 · ECE 0.170 | 0.481 · ECE 0.139 |
| Scelta | **tutte le rotazioni** | **0.596 · ECE 0.107** | 0.461 · ECE 0.118 |
| Scelta | tutte le rotazioni + input vuoto | 0.565 · ECE 0.074 | 0.472 · ECE 0.065 |
| Scala | ordine diretto e inverso (oggi) | 0.405 · NLL 1.50 · ECE 0.249 | 0.378 · NLL 1.38 · ECE 0.075 |
| Scala | **+ input vuoto** | 0.399 · **NLL 1.33 · ECE 0.217** | 0.388 · **NLL 1.29 · ECE 0.062** |

- **Sì/No:** l'input vuoto migliora accuratezza e calibrazione su entrambi i modelli, perché corregge proprio la tendenza al Sì. Con lo stato vuoto il modello dice Sì con probabilità mediana 0.64 (2B) e 0.72 (0.8B), quindi la correzione spinge verso il No. Le Sì/No diverse in typed-decisions però sono solo 6: il risultato si regge su poche domande.
- **Scelta:** sul 2B l'input vuoto toglie 3 punti alle rotazioni (0.596 → 0.565).
- **Scala:** l'accuratezza non cambia, ma NLL ed ECE migliorano.
- **Politica mista** (Sì/No e scala con input vuoto, scelta con tutte le rotazioni): 2B da 0.471 a 0.498, 0.8B da 0.446 a 0.451, senza storico.

**Proposta** (poi verificata e corretta in §9):
- **calibrazione a input vuoto di default per Sì/No e scale,** nel server del modello, con la previsione sullo stato vuoto memorizzata per domanda;
- **tutte le rotazioni di default per le domande a scelta,** senza input vuoto;
- **calibrazione di lotto facoltativa, spenta di default,** nel server web, che ha già lo storico delle risposte per domanda:
  - si attiva per domanda da ~50 casi;
  - mai per gli eventi rari, né per il flusso live, dove la stessa domanda riceve molti fotogrammi quasi uguali;
  - quando è attiva basta 1 permutazione e sostituisce l'input vuoto (le due insieme non sono state misurate);
- **prima del default, tre verifiche:**
  1. l'input vuoto su domande con immagini, e su una domanda da evento raro dove lo stato vuoto dice No: lì la correzione spingerebbe verso il Sì, quindi va valutato se limitarla;
  2. le temperature vanno rifatte sopra la nuova calibrazione;
  3. a parità di `min_confidence`, cambiano gli esiti `decided`/`uncertain`.

**Conseguenza per la roadmap:** il passo 3 di F0.5 (calibrazione con spostamento) diventa "calibrazione a input vuoto + calibrazione di lotto sullo storico", entrambe senza etichette.

## 9. Verifica e scelta finale (29/09/2026)

Prima di rendere default la calibrazione a input vuoto, l'abbiamo provata dove poteva fare danni: domande su immagini e su eventi rari. Script: [scripts/verifica_calibrazione.py](../scripts/verifica_calibrazione.py); dati: [examples/controlli/eventi_rari.json](../examples/controlli/eventi_rari.json). Modello 2B, 2 permutazioni.
- **Immagini:** le 21 immagini di `examples/immagini` × 6 domande Sì/No su eventi presenti in 2 immagini su 21 (incendio, incidente, gatto, email, segnale di stop, scontrino), in italiano e in inglese: 250 domande, 24 positive.
- **Testi:** 30 messaggi scritti a mano (15 per lingua) con domande su incendio ed emergenza medica, 3 positivi su 15 per lingua. Alcuni negativi sono vicini all'evento: un camino acceso, delle candele, un mal di testa.

| Correzione della Sì/No (310 domande) | Accuratezza | Falsi allarmi | NLL | ECE |
|---|---|---|---|---|
| nessuna | 1.000 | 0.000 | 0.066 | 0.060 |
| piena, stato vuoto "N/A" | 0.987 | 0.015 | 0.148 | 0.112 |
| piena, stato vuoto = immagine grigia (solo le 250 sulle immagini) | 0.912 | 0.097 | 0.381 | 0.184 |
| con un tetto di ±1 in logit | 0.990 | 0.011 | 0.112 | 0.087 |
| **solo verso il No** | **1.000** | **0.000** | **0.066** | **0.060** |

**Perché la correzione piena fa danni.**
- Su queste domande lo stato vuoto pende verso il **No**. P(Sì | "N/A") va da 0.10 a 0.29 sulle immagini e da 0.22 a 0.44 sui testi; con l'immagine grigia, da 0.03 a 0.10.
- Su typed-decisions pendeva invece verso il Sì (mediana 0.64 sul 2B).
- La correzione piena toglie anche la tendenza al No, che qui è giusta: senza informazioni, a "c'è un incendio?" la risposta è No. Toglierla spinge verso il Sì e crea falsi allarmi.

**La regola scelta: correggere solo verso il No.** Lo spostamento si toglie solo quando lo stato vuoto pende verso il Sì.
- Su typed-decisions dà lo stesso risultato della correzione piena: sul 2B Sì/No 0.533 con ECE 0.130; sullo 0.8B identico. Lì quasi tutte le domande pendono verso il Sì.
- Sugli eventi rari non cambia niente.
- Il tetto a ±1 riduce i danni, ma su typed-decisions perde parte del guadagno (2B 0.519).

**Scale: niente correzione.**
- Su typed-decisions lo stato vuoto pende verso il livello più alto. Le domande sono di urgenza e gravità, e sul 2B l'ultimo livello prende tra 0.58 e 0.89.
- Sull'unica scala della suite immagini ("Quanto è grave la situazione?") la correzione peggiora la NLL, da 0.28 a 0.56.
- Su typed-decisions migliora solo la calibrazione, non l'accuratezza.
- Per le scale non c'è una direzione "giusta" come il No per le Sì/No.

**Scelta:** tutte le rotazioni, senza input vuoto. Sulle immagini le 18 domande a scelta della suite sono giuste in tutte le varianti, e con tutte le rotazioni la NLL è la più bassa (0.036 contro 0.043).

**Controllo finale, con il codice.** `egeria predict` con i nuovi default sul test completo di typed-decisions (400 casi, 2.000 decisioni). Il confronto è appaiato con le predizioni precedenti a 2 permutazioni, con bootstrap per caso:

| | Accuratezza | NLL | ECE | ms/caso |
|---|---|---|---|---|
| 2B, prima (2 permutazioni) | 0.468 | 1.195 | 0.226 | |
| **2B, nuovi default** | **0.496** | **1.161** | **0.169** | 251 |
| 0.8B, prima (2 permutazioni) | 0.450 | 1.185 | 0.122 | |
| 0.8B, nuovi default | 0.441 | **1.156** | **0.086** | 163 |

- **2B:**
  - accuratezza +2.9 [+1.7, +4.1];
  - scelta +8.2 [+5.5, +11.0];
  - Sì/No +1.0 [−2.1, +4.1], con NLL −0.119 [−0.164, −0.074];
  - scale invariate.
- **0.8B:**
  - accuratezza −0.9 [−3.1, +1.3], non significativa;
  - NLL −0.029 e Brier −0.022, entrambe significative;
  - ECE delle Sì/No da 0.215 a 0.064.

**Cosa è stato implementato:**
- **Tutte le rotazioni di default.** `permutations="auto"` vale ovunque: server del modello, `decide`, `predict` e `suite`. Sono tutte le rotazioni per Sì/No e scelta, l'ordine diretto e inverso per scale e numeri. Un intero da 1 a 26 resta possibile.
- **Correzione della tendenza al Sì.** Vale per le domande Sì/No ed è attiva di default. Si spegne con `"yes_correction": false` nella richiesta o `--no-yes-correction` all'avvio.
  - Lo spostamento si calcola sullo stato "N/A" con le stesse permutazioni, una volta per domanda, e resta in cache (fino a 4.096 domande).
  - La applica anche `egeria predict`, così le temperature si fittano sopra la correzione.
- **Calibrazione sullo storico.** Sta nel server web ed è facoltativa per domanda (`"batch_calibration": true`, casella *Calibra sullo storico*). Si attiva da 50 casi e mai dal vivo ([07](07-interfaccia.md) §1bis).

**Cosa restava** (fatto nella §10): rifare le temperature, misurare come cambiano gli esiti `decided`/`uncertain`, verificare la tendenza al Sì sulle pose.

```bash
# verifica della correzione su immagini ed eventi rari (GPU, ~20 s dopo il caricamento)
.venv/bin/python scripts/verifica_calibrazione.py Qwen/Qwen3.5-2B-Base
# controllo finale su typed-decisions con i default nuovi
.venv/bin/egeria predict --model Qwen/Qwen3.5-2B-Base --split test --out runs/calibrazione/2B-default/test.jsonl
.venv/bin/egeria paired --a runs/Qwen3.5-2B-Base/test.jsonl --b runs/calibrazione/2B-default/test.jsonl
```

## 10. Temperature, soglia e pose (29/09/2026)

### 10.1 Temperature rifatte

Predizioni sul train di typed-decisions (1.200 casi) con i nuovi default, poi `egeria calibrate`, valutate sul test:

| | T Sì/No | T scelta | T scala | NLL test | ECE test |
|---|---|---|---|---|---|
| 2B, vecchie impostazioni | 7.08 | 2.34 | 3.91 | | |
| **2B, nuovi default** | **3.85** | 2.58 | 3.81 | 1.161 → **1.051** | 0.169 → **0.050** |
| 0.8B, vecchie impostazioni | 20 (il massimo) | 2.19 | 2.73 | | |
| 0.8B, nuovi default | 20 (il massimo) | 2.35 | 2.70 | 1.156 → 1.127 | 0.086 → 0.046 |

- **2B:** con la correzione del Sì la temperatura delle Sì/No si dimezza, da 7.08 a 3.85. Resta comunque alta: su typed-decisions il 2B sulle Sì/No è poco sopra il caso (0.527).
- **0.8B:** la temperatura delle Sì/No resta al massimo. Su typed-decisions le sue Sì/No non portano informazione, e la temperatura le appiattisce a 0.5.
- **Solo sui propri dati:** le temperature sono fittate su typed-decisions. Come prima ([07](07-interfaccia.md) §1), vanno usate solo su dati simili e mai sulle immagini. Stanno in `runs/calibrazione/<modello>-default/temperature.json`; le vecchie in `runs/*/temperature.json` sono superate.

### 10.2 Soglia: quante risposte diventano incerte

`egeria evaluate --min-confidence 0.5` (la soglia di default del server del modello) conta le risposte sotto soglia e l'accuratezza di quelle decise. 2B, test di typed-decisions:

| | Incerte | Accuratezza delle decise | Sì/No incerte | Sì/No: accuratezza delle decise | Scelta: accuratezza delle decise |
|---|---|---|---|---|---|
| prima (2 permutazioni) | 46% | 0.570 | 47% | 0.647 | 0.608 |
| **nuovi default** | 58% | **0.590** | 86% | **0.819** | **0.667** |
| nuovi default + temperature | 95% | **0.905** | 100% | – | 0.905 |

- **Senza temperature:** con i nuovi default le Sì/No di typed-decisions risultano incerte molto più spesso. Quelle che restano decise sono molto più affidabili: 0.819 contro 0.647.
- **Casi netti:** restano decisi. Sulle domande con immagini della §9 il modello dà P(Sì) 0.04 sulle negative e 0.92 sulle positive.
- **Con le temperature:** la soglia 0.5 lascia decise solo le scelte più nette, giuste al 90%. È il comportamento onesto su un compito dove il modello zero-shot sa poco.

### 10.3 La tendenza al Sì sulle pose

**Dati.** 24 foto di Wikimedia Commons in `examples/immagini/pose/` (fonti e licenze in `FONTI.md`), in italiano e in inglese:
- 9 pose dell'albero;
- 4 altre pose su una gamba (ballerino e simili);
- 1 posa con le braccia alzate e i piedi a terra;
- 7 altre pose (guerriero, sedia);
- 3 immagini senza persone: un albero vero, una tigre fra gli alberi, un uccello.

Modello 2B, lettura di default. Script: `scripts/verifica_calibrazione.py`, parte 4.

| Domanda | Albero (detto sì / albero) | Su una gamba | Braccia alzate | Altre pose | Senza persone |
|---|---|---|---|---|---|
| Sì/No "La persona sta facendo la posa dell'albero?" | 100% | 62% | 50% | 64% | 0% |
| Sì/No "La persona sta in equilibrio su una gamba sola?" | 100% | 100% | 0% | 14% | 0% |
| Sì/No "Il piede della gamba sollevata è appoggiato all'interno dell'altra gamba?" | 50% | 0% | 0% | 0% | 0% |
| Scelta generica: albero / un'altra posa / nessuna persona | 100% | 100% | 100% | 100% | 0% |
| **Scelta con le pose descritte** (in piedi su una gamba con il piede all'interno dell'altra; su una gamba con l'altra sollevata dietro; gambe divaricate e ginocchio piegato; ginocchia piegate come su una sedia; in piedi su due piedi; nessuna persona) | 100% | 100% | **0%** | **14%** | **0%** |

**Cosa emerge:**
1. **Non è la tendenza al Sì dello stato vuoto.** Per la domanda sulla posa dell'albero, lo stato vuoto pende verso il No (P(Sì) 0.31–0.37), quindi la correzione non interviene. Quella piena peggiorerebbe: falsi allarmi dal 50% al 67%.
2. **È una confusione di contenuto.** Il 2B riconosce "una persona che fa yoga", non quale posa: con Sì/No dice "albero" anche a due guerrieri su tre. I suoi Sì non sono una tendenza generica: a una persona in equilibrio su una gamba risponde bene (95.8%).
3. **Descrivere le opzioni a parole aiuta.** Le altre pose scambiate per l'albero scendono dal 64% al 14%, e le braccia alzate a 0.
4. **Resta la confusione fra l'albero e le altre pose su una gamba.** Il 2B non distingue il piede appoggiato all'interno della gamba dalla gamba sollevata dietro. La domanda sul solo piede è troppo difficile: nessun falso allarme, ma perde metà delle pose dell'albero.

**Consigli per chi lo usa (anche in [07](07-interfaccia.md) §10):**
- per le pose, meglio una **scelta con le pose descritte** di un Sì/No con il nome della posa;
- per distinguere le pose su una gamba fra loro serve il **training**. È un buon argomento per il primo LoRA sulle pose di danza (F1, [02](02-implicazioni-e-proposta.md) §7).

```bash
# temperature con i nuovi default
.venv/bin/egeria predict --model Qwen/Qwen3.5-2B-Base --split train --out runs/calibrazione/2B-default/train.jsonl
.venv/bin/egeria calibrate --predictions runs/calibrazione/2B-default/train.jsonl --out runs/calibrazione/2B-default/temperature.json
.venv/bin/egeria evaluate --predictions runs/calibrazione/2B-default/test.jsonl --temperatures runs/calibrazione/2B-default/temperature.json --min-confidence 0.5
```
