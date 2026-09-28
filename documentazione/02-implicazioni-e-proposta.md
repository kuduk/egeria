# Implicazioni per Egeria e proposta di partenza

> Questo documento traduce lo [stato dell'arte](01-stato-dell-arte.md) in una proposta concreta. Le decisioni ancora aperte sono in fondo (§7).

## 0. Decisioni prese

| Data | Decisione | Motivo |
|---|---|---|
| 26/09/2026 | **Backbone: famiglia Qwen3.5** (come SemIf), non Qwen3.8 | Qwen3.8 non ha taglie piccole né checkpoint Base. Qwen3.5 ha la stessa architettura. Qwen3.8-27B resta candidato come teacher |
| 26/09/2026 | **Si parte dalla baseline F0 zero-shot**: readout dei logit delle lettere, senza training | Stabilisce il riferimento da battere con LoRA e distillazione. Implementata, vedi [03-baseline-f0.md](03-baseline-f0.md) |
| 26/09/2026 | **Hardware di sviluppo: RTX 4070 Laptop (8 GB), WSL2** | In bf16 entrano 0.8B e 2B. 4B e 9B solo quantizzati a 4 bit, oppure su GPU cloud |
| 26/09/2026 | **Nuova direzione: profondità dinamica per asserzione** (§4bis) | Idea dell'utente. Si adatta bene a un modello a singolo forward pass |
| 26/09/2026 | **`min_confidence` per asserzione + campo globale** (§4bis) | Il chiamante conosce il costo dell'errore. Il locale sovrascrive il globale; di default si usa il modello completo |

## 1. Cosa cambia rispetto all'idea iniziale ("partire da Qwen 3.8")

| Assunzione iniziale | Realtà (settembre 2026) | Conseguenza |
|---|---|---|
| Qwen 3.8 esiste in varie taglie | Solo 27B denso, 2.4T MoE e Flash-Next (125B MoE). **Nessuna taglia piccola** | Lo *student* veloce deve partire da **Qwen3.5 Base** (0.8B–9B), che ha la stessa architettura del 3.8 |
| Si può partire da un checkpoint base | Qwen3.8 ha **solo versioni post-trained**, più sovraconfidenti | Qwen3.8-27B va usato come **teacher** o come student "grande", non come punto di partenza ideale per la calibrazione |
| Si può copiare Laya (encoder + `[MASK]` per opzione) | Qwen è **causale e ibrido** (DeltaNet), quindi non diventa un BERT | Marker **dopo** le opzioni e pointer head (stile Kev). Prefix-fork per più domande |
| RLCD è la chiave | Con etichette note, l'RL ≈ una loss propria supervisionata, ma più rumorosa | Training **supervisionato con soft CE + Brier + RPS**; RL solo per feedback bandit o d'ambiente |
| Nessuno l'ha ancora fatto su Qwen | Esistono già Kev-27B, Open-Jev-27B, decider e JevK5 | Serve una **differenziazione** chiara (§2) |

## 2. Possibili elementi di differenziazione

Su qualità generica inglese Jev, decider e Kev sono già a pari livello. Aree in cui il settore è debole:

1. **Italiano ed europeo multilingua.** Jev è "English-first". Laya-multilingual è debole (MASSIVE 0.366). Le repliche Qwen non misurano l'italiano. Un benchmark decisionale in italiano oggi **non esiste**.
2. **Astensione onesta.** Nessun sistema gestisce bene gli item inconoscibili e l'incertezza aleatoria (Jev bluffa nel 50% dei casi). Si può addestrare un'opzione esplicita "non determinabile", con dati dedicati e un risk–coverage dichiarato.
3. **Invarianza all'ordine e coerenza.** Flip rate del 13–49% nei sistemi attuali. Si possono usare permutation augmentation, readout simmetrizzato e una loss di consistenza fra domande equivalenti o negate.
4. **Un dominio verticale.** Per esempio PA, legale, sanità o customer care italiano, con dati propri: TypeSafe stessa dice che il vantaggio sono i dati.
5. **Compatibilità API con Jev** (`/v1/systemone`, stesse formule di `confidence`). Gli strumenti esistenti funzionerebbero subito: SDK, JevBench, harness.

## 3. Architettura proposta (v0)

```
                ┌───────────────────────────── prefisso condiviso (calcolato 1 volta) ──┐
input:  <state> …stato testo/JSON… </state>                                             │
                └───────────────────────────────────────────────────────────────────────┘
                     │ prefix-fork: copia di KV + stato conv/DeltaNet per ogni domanda
      ┌──────────────┴──────────────┐
ramo q1:  <q type=choice> istruzioni </q> <opt> A: … </opt> <opt> B: … </opt> … <decide>
ramo q2:  <q type=noul> … </q> <opt> false: … </opt> <opt> true: … </opt> <decide>
      │
      ▼
testa pointer:  logit_i = f(h(</opt>_i), h(<decide>))   →  softmax ristretta / T_tipo
```

- **Backbone.**
  - Student: **Qwen3.5-4B-Base** (compromesso) oppure 2B-Base (latenza), con LoRA r16–32 e i token speciali aggiunti.
  - Teacher: **Qwen3.8-27B** (in 4 bit localmente, oppure via API).
- **Readout.** Pointer head alla Kev: hidden del marker di fine opzione confrontato con quello del marker `<decide>`. Il readout è in fp32.
  - Baseline zero-shot: **logit delle lettere** alla SemIf, senza training.
- **Primitive.**
  - `choice`: softmax sulle opzioni.
  - `noul`: due slot false/true.
  - `score`: softmax sui livelli, con valore atteso e loss RPS.
- **Più domande per stato.** Prefix-fork dello stato ibrido (vedi qwen-rlcd e Kev). In training si può partire con una riga per domanda, più semplice, e ottimizzare dopo.
- **Esperimento opzionale.** Maschera bidirezionale sui soli layer full-attention (stile dQwen3.5), come ablation.

## 4. Training proposto

1. **Dati.**
   - Dataset pubblici di classificazione, NLI, safety e intent, convertiti nelle tre primitive.
   - Stati sintetici con etichette **soft** del teacher Qwen3.8-27B, con N campioni e temperatura calibrata.
   - Set italiano dedicato.
   - Augmentation: permutazione delle opzioni, parafrasi delle istruzioni, stato come testo o come JSON, domande distrattrici, item inconoscibili.
2. **Loss.** Soft CE (log-score) + λ·Brier + RPS per `score`. Mix di etichette hard (umane) e soft (teacher). Opzionale: consistenza fra permutazioni. È anche la ricetta che Laya usa davvero: la soft CE a peso pieno è il termine principale e il policy gradient rumoroso è un'aggiunta ([01](01-stato-dell-arte.md) §4.6).
3. **Calibrazione post-hoc.** Temperatura per tipo e per fascia di numero di opzioni (come Laya: 2, 3–5, 6–10, 11+), fittata su un pezzo tolto dal training **prima** di iniziare, **mai** sul train. La temperatura non corregge un'inclinazione sistematica (per esempio verso il "Sì"): quella si corregge con dati bilanciati, oppure con un termine di spostamento fittato su etichette.
4. **Progettazione dei dati contro le scorciatoie** (lezioni dal fine-tuning di Laya come agente web):
   - negativi difficili (per il "Sì": casi che somigliano al positivo ma non lo sono) e classi rare ripesate;
   - etichette "al contrario" (prima la risposta, poi un LLM scrive lo stato o la domanda che la richiede);
   - correzioni sulle proprie traiettorie (DAgger) per gli usi in cui il modello sceglie azioni.
5. **RL (fase successiva, solo se serve).** Reward bounded (Brier o sferica), baseline leave-one-out **senza normalizzazione per std**, su feedback reale di tipo bandit o su ambienti.

## 4bis. Profondità dinamica per asserzione

**Idea.** Ogni asserzione (domanda) usa solo la profondità che le serve. Dopo ogni blocco del modello si legge la distribuzione sulle opzioni. Se la confidenza *calibrata per quel layer* supera una soglia, si esce; altrimenti si prosegue. Le domande facili costano pochi layer, quelle difficili tutto il modello.

**Perché si adatta bene a questo progetto.**
- **Nessuna propagazione di KV-cache.** Nella generazione token per token l'early exit è complicato (CALM, LayerSkip) perché i token successivi hanno bisogno degli stati dei layer saltati. Qui c'è **un solo forward pass per domanda**, quindi uscire presto è un risparmio netto.
- **L'architettura ha già confini naturali.** Qwen3.5 è organizzato in blocchi `3 × Gated DeltaNet + 1 × Attention`: le uscite naturali sono dopo ogni layer full-attention (6 uscite nel 0.8B/2B, 8 nel 4B/9B).
- **In batch basta rimuovere le righe già decise** fra un blocco e il successivo: non serve padding dinamico.
- **Si combina con la cascata di modelli.** Profondità dinamica dentro il modello piccolo, poi escalation al modello grande o al "thinking" solo se anche l'ultimo layer resta incerto. Nella letteratura (NYU, arXiv 2609.24574) la cascata System 1 → LLM raggiunge la qualità frontier al 25–50% del costo.

**Rischi noti** (dalla letteratura sull'early exit):
- **I layer bassi sono sovraconfidenti o hanno logit di scala diversa.** Serve una temperatura **per layer** (ACT, "annealed confidence threshold", 2026). In alternativa, un criterio di "pazienza": la stessa risposta per k uscite consecutive (PABEE).
- **Il logit lens sui layer intermedi è debole senza training.** Nel training (F1) servono **teste di uscita addestrate** (tuned lens / LayerSkip), con una loss su ogni uscita pesata per profondità.
- **Serve una garanzia sulla qualità.** Si può scegliere la soglia con **conformal risk control** (CATs, Schuster et al. 2021), in modo che la decisione anticipata coincida con quella del modello completo con probabilità ≥ 1 − ε.

**Piano.**
1. **F0 (fatto).** Diagnostica zero-shot: readout a ogni confine di blocco, temperature per (tipo, layer), simulazione di soglie diverse → curva calcolo/accuratezza. Risultato: il risparmio potenziale (oracolo) è del 27–36%, quello ottenuto con la confidenza zero-shot solo dello 0.5–4%, sempre con accordo del 100% con il modello completo. Vedi [03-baseline-f0.md](03-baseline-f0.md) §6.2.
2. **API e uscita reale (fatto).** Parametro `min_confidence` per asserzione, con campo globale nella richiesta e default del server. Il valore locale sovrascrive il globale; senza soglia si usa il default prudente, cioè il modello completo. Esecuzione blocco per blocco con rimozione delle domande già decise; risposta con `depth` e `status` (`decided`/`uncertain`). La soglia è sulla confidenza corretta per il caso (0 = a caso, 1 = certezza). Misura della latenza: la richiesta aspetta la domanda più profonda. Su GPU il guadagno di latenza arriva solo se escono tutte le domande; su CPU è proporzionale. Vedi [03-baseline-f0.md](03-baseline-f0.md) §6.4.
3. **F1.** Teste di uscita addestrate su ogni confine di blocco, con loss propria su tutte le uscite. Soglia scelta con conformal risk control, in modo che `min_confidence` diventi una garanzia empirica di accuratezza.
4. **F3.** Cascata a più livelli: uscita anticipata → modello completo → modello più grande o thinking → escalation umana. `status: uncertain` è il segnale d'ingresso.

## 5. Valutazione proposta

- **Generalista (zero-shot):** typed-decisions (test, senza addestrare sul train split), JevBench public, Decision Index, BTZSC.
- **Robustezza:** CLINC150 con OOS (astensione), Banking77 (cardinalità), flip rate da permutazione, domande negate.
- **Controlli senza etichette** (stile `research/eval` di Laya, più due nostri):
  - opzioni identiche, per misurare la preferenza di posizione e decidere se servono 2 permutazioni o ne basta 1;
  - coerenza della negazione: P(Sì | domanda) + P(Sì | domanda negata) ≈ 1, che misura l'inclinazione verso il "Sì";
  - immagine specchiata, per destra e sinistra.
- **Confidenza:** AUROC contro la correttezza e accuratezza sulla metà più sicura, oltre all'ECE.
- **Multilingua:** XNLI-it, MASSIVE-it, più un **nuovo set decisionale in italiano** da costruire.
- **Metriche:** accuratezza, NLL, Brier, ECE (binning dichiarato), copertura al 5% d'errore, latenza p50/p95 del solo modello.
- **Baseline da battere:** SemIf zero-shot (stesso backbone), Kev-4B/9B, decider-4b, Laya-multilingual, Jev (via API, se c'è budget).

## 5bis. Limiti attuali di Egeria (settembre 2026)

- **Zero-shot su typed-decisions siamo al livello della prior:** 0.468 contro 0.478. Sopra Laya base (0.362), ma sotto una baseline banale. Ciò che batte la prior è il voto dei ricordi (0.562).
- **Costo lineare nel numero di domande, e doppio per le permutazioni.** Lo stato viene rielaborato per ogni domanda e per ogni permutazione, anche se il formato (stato prima della domanda) permetterebbe di calcolarlo una volta.
- **La calibrazione non si trasferisce.** Le temperature di typed-decisions peggiorano le domande generiche e le immagini, non c'è un termine di spostamento, e l'inclinazione verso il "Sì" non è misurata.
- **Il profilo di latenza è diverso da Laya.** Un LLM da 0.8–2B per token costa più di un encoder da 421M; la compensazione possibile è il riuso dello stato.
- **Profondità dinamica** con poco guadagno reale zero-shot; **al massimo 26 opzioni**; `inject`, `known` e `surprise` senza effetto zero-shot.

**Priorità per migliorare il modello, dalla meno costosa:**
1. **Riuso dello stato fra le domande** (prefix-fork, anche delle immagini).
2. **Controlli senza etichette** (§5), per decidere se basta una permutazione e per misurare l'inclinazione verso il "Sì".
3. **Calibrazione temperatura + spostamento per fascia di opzioni**, valutata fuori dominio.
4. **F1** con loss supervisionata propria, dati contro le scorciatoie, permutazioni con loss di coerenza, etichette umane soft.

## 6. Roadmap indicativa

| Fase | Obiettivo | Output |
|---|---|---|
| F0 | Setup e baseline zero-shot: logit delle lettere su Qwen3.5 (0.8B/2B in bf16, 4B/9B in 4 bit) e diagnostica della profondità dinamica | **Completata**: nessun modello piccolo batte la Prior zero-shot; profondità dinamica con 27–36% di risparmio potenziale (vedi [03-baseline-f0.md](03-baseline-f0.md) §6) |
| F1 | LoRA + pointer head su dati pubblici, teste di uscita per la profondità dinamica. Più le primitive rinviate: `multi` (loss propria per opzione) e `known` (loss propria su coppie stesso/simile/nuovo + embedding contrastivo con LoRA), vedi [06-memoria.md](06-memoria.md) §7 | Primo checkpoint e confronto con Kev/decider |
| F2 | Distillazione soft da Qwen3.8-27B e set italiano | Checkpoint v1 multilingua |
| F3 | Calibrazione, astensione, invarianza all'ordine | Report risk–coverage |
| F4 | Serving: prefix-fork, API compatibile Jev, quantizzazione, risposte progressive (streaming per domanda) e batching continuo a livello di blocco | Server di inferenza |
| F5 (opz.) | RL con feedback reale; loss ausiliaria LLM-JEPA | Ablation |

## 7. Decisioni aperte

1. **GPU per il training (F1+).** L'utente è disposto a noleggiare una GPU. Piano: sviluppo e debug in locale su 0.8B/2B; noleggio solo per i passaggi pesanti:
   - etichettatura con il teacher Qwen3.8-27B via vLLM: 80 GB in bf16, oppure 48 GB con la versione FP8;
   - LoRA su 4B/9B: 48–80 GB.

   Da decidere: provider e budget.
2. **Lingua target.** Italiano prioritario o inglese?
3. **Scopo.** Modello generalista (tipo Jev) o verticale su un dominio specifico?
4. **Compatibilità API Jev.** La F0 è già compatibile con il formato di richiesta e risposta. Resta da decidere se esporre anche un server HTTP `/v1/systemone`.
5. **Licenza e pubblicazione.** Rilascio open (Apache-2.0) o uso interno?
