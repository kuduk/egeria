**Italiano** · [English](en/01-state-of-the-art.md)

# Stato dell'arte: modelli decisionali semantici "System 1"

> Ricerca aggiornata al **26 settembre 2026**; CLM (§5.5) è stato aggiunto il 29 settembre. La categoria è nata a metà settembre 2026: quasi tutti i numeri sono dichiarati dai vendor o da leaderboard della community vecchie di pochi giorni.
>
> Etichette usate:
> - **[vendor]**: dichiarato da chi ha fatto il modello.
> - **[community]**: misura di terzi non peer-reviewed.
> - **[incerto]**: fonti in disaccordo o dato non verificato.
>
> Le fonti complete sono in [fonti.md](fonti.md).

## Indice

1. [Sintesi](#1-sintesi)
2. [Cos'è un modello decisionale "System 1"](#2-cosè-un-modello-decisionale-system-1)
3. [Jev (TypeSafe AI)](#3-jev-typesafe-ai)
4. [Laya (Convai Innovations)](#4-laya-convai-innovations)
5. [SemIf e le altre repliche aperte](#5-semif-e-le-altre-repliche-aperte)
6. [Benchmark e metriche](#6-benchmark-e-metriche)
7. [La famiglia Qwen 3.8](#7-la-famiglia-qwen-38)
8. [Tecniche: da decoder a classificatore in un solo forward pass](#8-tecniche-da-decoder-a-classificatore-in-un-solo-forward-pass)
9. [Calibrazione, scoring rule e RLCD](#9-calibrazione-scoring-rule-e-rlcd)
10. [Altri paradigmi "semantici" (JEPA, LCM, CALM…)](#10-altri-paradigmi-semantici-jepa-lcm-calm)
11. [Problemi aperti del settore](#11-problemi-aperti-del-settore)
12. [Punti incerti](#12-punti-incerti)

---

## 1. Sintesi

- **Cosa sono.** Jev, Laya e SemIf non sono LLM generativi. Sono **modelli decisionali "System 1"**: ricevono uno *stato* (testo o JSON) e delle *domande tipizzate* con opzioni dichiarate. Restituiscono, in un solo forward pass, una **distribuzione di probabilità calibrata** sulle opzioni. Non generano testo.
- **Jev** (TypeSafe AI, 15/09/2026) è chiuso e disponibile solo via API. È il riferimento di qualità. Le misure black-box indicano un **decoder grande** (conoscenza da MMLU-Pro ≈ 84%), non un piccolo encoder.
- **Laya** (Convai, 18/09/2026) è aperto (Apache-2.0) e usa un encoder ModernBERT da 421M.
  - È veloce: circa 33 ms su una T4.
  - Zero-shot è però **quasi casuale** (0.36 su typed-decisions) e va sempre fine-tunato.
  - Il suo "RLCD" è di fatto una versione rumorosa di un training supervisionato con scoring rule proprie.
- **Esistono già repliche aperte su Qwen**, e diverse eguagliano o superano Jev sui benchmark della community:
  - Kev (anche su **Qwen3.8-27B**);
  - Open-Jev-27B (su **Qwen3.8-27B**);
  - decider;
  - JevK5;
  - SemIf.

  La ricetta vincente oggi è: Qwen3.5 **Base** + LoRA, readout dei logit delle opzioni (lettera o pointer head), softmax ristretta, distillazione soft da un teacher Qwen più grande, e una temperatura fittata.
- **CLM** (Stanford + NVIDIA, 23/09/2026) è aperto e sceglie un'altra strada: un **dual encoder contrastivo**, cioè Qwen3-8B congelato più due piccole teste.
  - È molto veloce e non ha bias di posizione.
  - Nei suoi stessi test, però, concorda con la risposta giusta molto meno di Jev (§5.5).
- **Qwen 3.8 non ha modelli piccoli né checkpoint Base.** Esistono solo 27B denso, 2.4T MoE e Flash-Next (125B MoE). Il 27B è realistico come **teacher**, oppure come student solo con una GPU da 80 GB.
- **Architettura ibrida.** Qwen 3.5/3.6/3.8 usa un'architettura ibrida (75% Gated DeltaNet lineare). Queste parti **non si possono rendere bidirezionali** cambiando la maschera, e le **tree attention mask non funzionano**. Servono tecniche specifiche, come il "prefix-fork" dello stato.
- **Gli altri paradigmi "semantici"** (JEPA, Large Concept Models, CALM, ragionamento latente) sono interessanti ma **non servono** per un modello decisionale. Al massimo una loss ausiliaria LLM-JEPA è un esperimento a basso costo.

> **Nota sui nomi.** Nella richiesta iniziale "semif laya" è stato interpretato come **SemIf** + **Laya**, due progetti distinti descritti sotto.

---

## 2. Cos'è un modello decisionale "System 1"

Il nome richiama il "System 1" di Kahneman: decisioni rapide e intuitive, contrapposte al ragionamento lento del "System 2".

| | LLM generativo (System 2) | Modello decisionale (System 1) |
|---|---|---|
| Output | Testo libero (token per token) | Tipi chiusi: probabilità sulle opzioni dichiarate |
| Forward pass | Molti (uno per token) | Uno per richiesta (o per domanda) |
| Latenza tipica | Secondi | 5–500 ms |
| "Allucinazioni" | Testo inventato, JSON malformato | Impossibili *sintatticamente*, ma la risposta può essere sbagliata |
| Confidenza | Verbalizzata, poco affidabile | Distribuzione esplicita, ricalibrabile |
| Uso tipico | Chat, generazione, ragionamento | Routing, triage, guardrail, gating di agenti, moderazione, reranking |

### Le tre primitive (standard di fatto, introdotte da Jev)

| Primitiva | Semantica | Output |
|---|---|---|
| `noul` (da *Bernoulli*) | Domanda sì/no | `P(true)` |
| `choice` | Una fra N opzioni (Jev arriva a 255) | Distribuzione sulle opzioni, più l'argmax |
| `score` | Scala ordinale (2–10 livelli) | Distribuzione sui livelli e valore atteso |

Il pattern d'uso è un **harness in codice**: la logica di business (somme, date, regole) sta nel codice, mentre il modello risponde solo a domande atomiche. TypeSafe mostra che anche i modelli frontier migliorano quando sono usati così. Per esempio Opus 5 passa dal 64.8% in modalità prompt al 73.1% in modalità workflow.

---

## 3. Jev (TypeSafe AI)

### 3.1 Fatti

- **Azienda.** TypeSafe AI. Il fondatore è Diogo Almeida (ex Google Brain e OpenAI, coautore di InstructGPT e GPT-4); i cofondatori sono Erik Gafni e Sasha Sheng.
  - Seed round da $40M guidato da DCVC; valutazione di $200M secondo Forbes.
- **Lancio.** 15/09/2026, dopo circa 2 anni in stealth. Il modello corrente è `jev-1.13.0`.
- **Distribuzione.** Chiuso, solo API, cloud USA. Non si può fare fine-tuning: "gli stessi pesi servono ogni account". La zero data retention è riservata ai clienti enterprise.
- **Prezzo.** $0.042 per milione di token di input; l'output è gratuito.
- **Limiti di contesto.** 64k token per richiesta; 32k per lo stato più la domanda più lunga.
- **Lingue.** Principalmente inglese. CJK è gestito "ma non altrettanto bene".
- **Latenza.**
  - Dichiarata [vendor]: 70–500 ms end-to-end.
  - Misurata [community]: 236–276 ms p50 dal client.
  - Tempo lato server: circa 57–218 ms al crescere dello stato.

### 3.2 API (`POST /v1/systemone`)

```json
{"state": "Help! My payouts have been failing for 3 days.",
 "model": "jev-latest",
 "questions": {
   "is_urgent":  {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
   "department": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Payments, invoicing, refunds",
                               "technical": "Bugs, outages, integrations",
                               "sales": "Pricing, upgrades, new accounts"}},
   "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                  "criteria": ["Calm", "Frustrated", "Very angry"]}}}
```

La risposta contiene, per ogni domanda, l'argmax, le `probabilities` e una `confidence`.

**Formule di `confidence`.** Sono statistiche della distribuzione, non una stima appresa della correttezza.
- choice: `c = (n·p_max − 1)/(n − 1)`
- score: `c = max(0, 1 − Σ p_i·|i − mode| / D)`, dove `D = (1/n)·Σ|i − (n−1)/2|`

**Astensione.** Non esiste un campo di astensione: va fatta lato client con soglie, per esempio act, confirm o escalate. Il cookbook ufficiale manda in revisione umana i casi con p_max < 0.60, ottenendo il 99.2% di consistenza con il 74.2% dei casi automatizzati.

SDK ufficiali: Python `typesafe-sdk`, JavaScript `@typesafe-ai/sdk`. Esistono integrazioni con LangChain e Vercel AI Gateway.

### 3.3 Architettura: cosa si sa

**Dichiarazioni ufficiali:**
- "Nuova architettura + parallel sampler + RLCD". FAQ: "Jev non è né piccolo né un LLM".
- Nessuna uscita a stringa: tutti gli output sono calcolabili in parallelo.
- "Siamo soprattutto un laboratorio di ricerca sui dati." Il vantaggio competitivo dichiarato sono i dati, non l'architettura.

**Analisi black-box** (Archer Hume, circa 10.000 chiamate) [community]:
- **Stato condiviso, domande isolate.** Un segreto messo in una domanda sorella è invisibile alle altre; messo nello stato viene trovato. È compatibile con un prefix KV-cache condiviso e suffissi separati per domanda.
- **Opzioni listwise.** Aggiungere un'opzione irrilevante sposta i log-odds fra le altre, violando l'IIA (indipendenza dalle alternative irrilevanti). Conta anche l'ordine delle opzioni.
- **Tokenizer.** Non coincide con nessuno dei 192 tokenizer pubblici testati; il più vicino è **Qwen**.
- **Conoscenza.** MMLU-Pro ≈ 84.6%. Questo fa pensare a un decoder grande, probabilmente un MoE con circa 10B parametri attivi [incerto].

### 3.4 RLCD ("Reinforcement Learning for Calibrated Decisions")

Tecnica **non pubblicata**. L'unica descrizione ufficiale: ottimizza "risposte con probabilità epistemicamente oneste", in contrapposizione a RLHF (sycophancy) e RLVR ("intelligenza spikey").

Le interpretazioni di terzi convergono su un **obiettivo con scoring rule proprie** (log o Brier) applicato agli slot delle opzioni.

### 3.5 Prestazioni e debolezze

**TypeSafe Workflow Evals** [vendor]:
- Quattro workflow: sicurezza, trace di agenti, fatture, customer service.
- Le etichette di riferimento sono la media di GPT-6 Astra e Fable 5.1, quindi il benchmark misura **l'accordo con LLM frontier**, non la verità.

| Modello | Accuratezza media | $/caso | s/caso |
|---|---|---|---|
| Jev | 67.8% | 0.0004 | 0.4 |
| GPT-5.6 Sol | 74.1% | 0.0836 | 23.3 |
| Opus 5 | 73.1% | 0.1761 | 37.8 |
| Sonnet 5 | 67.8% | – | – |
| Haiku 4.5 | 53.6% | – | – |

**Debolezze dichiarate** (pagina "Jev 1.13 jaggedness"):
- lettura troppo letterale;
- aritmetica, conteggi e date;
- domande multi-hop;
- stati lunghi pieni di rumore;
- prompt injection nello stato;
- nessuna coerenza strutturale fra domande. Per esempio `noul` "rimborso" = 0.72 e `noul` "non rimborso" = 0.47, che sommano a 1.19.

**Misure di terzi** [community]:
- **Bluffing.** Ammette di non sapere solo nel 49.7% dei casi impossibili, contro il 97–100% degli LLM.
- **Sensibilità all'ordine.** Permutando le opzioni cambia risposta nel 13–14.6% dei casi.
- **Incertezza aleatoria.** Su una moneta equa dà P(testa) = 0.92.
- **typed-decisions.** Accuratezza 0.727, ma KL dalla distribuzione gold 1.442: è troppo sicuro.
- **NYU Abu Dhabi** (arXiv 2609.24574, 18 task di social science computazionale):
  - dietro il miglior LLM in 14 task su 15 (mediana −11.6 F1);
  - ECE mediana 0.157;
  - una **cascata** Jev → LLM sui casi incerti raggiunge la qualità frontier al 25–50% del costo.

---

## 4. Laya (Convai Innovations)

### 4.1 Fatti

- **Autore e licenza.** Nandakishor M (Convai Innovations), ricercatore indipendente. Rilasciato il 18/09/2026, **Apache-2.0**, `pip install laya`, pesi su HF `convaiinnovations/laya`.
- **Checkpoint:**

| Checkpoint | Encoder | Parametri | Contesto (default) |
|---|---|---|---|
| `laya` | ModernBERT-large (395M, 28 layer) | 421M | 512 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 (fino a 8192) |
| `laya-typed-decisions` | ModernBERT-large, fine-tuned | 421M | 1024 |

### 4.2 Architettura (dal codice)

**Packing dell'input**, una sequenza per domanda:
```
[CLS] "<tipo> question: <istruzioni>" [SEP] [MASK] opz0 [MASK] opz1 … [SEP] <stato> [SEP]
```

**Testa decisionale:**
1. All'output dell'encoder si somma un embedding del tipo di domanda.
2. Seguono 2 layer transformer aggiuntivi (circa 25M parametri).
3. Si raccolgono gli hidden state nelle posizioni `[MASK]`.
4. Un MLP produce un logit per opzione, seguito da temperatura e softmax.

**Primitive:**
- `noul` usa sempre due slot, `[false, true]`.
- `score` restituisce il valore atteso `Σ i·p_i`.

**Testa Act/Escalate.** È un MLP su CLS, p_top1, margine, entropia e k. Nella versione rilasciata **è rotta** (issue #185): restituisce sempre ≈1.0 ed è anti-correlata alla correttezza.

**Router multilingua.** Rileva script e lingua in meno di 0.5 ms e sceglie il checkpoint. Serve perché il checkpoint inglese sul khmer ha accuratezza 0.000 con confidenza 0.952.

### 4.3 Training

**Dati.** Dataset pubblici etichettati da umani: triage, NLI, safety, jailbreak, rubriche, conversazioni di vendita. La lista esatta non è pubblicata.

**"RLCD" alla Laya:**
1. Rumore gaussiano a media zero sui logit, con G = 8 campioni e σ che scende da 1.0 a 0.3.
2. Reward = log-score + 0.5·spherical − 1.0·RPS. L'RPS è usato solo per `score`.
3. Vantaggio normalizzato sul gruppo, poi REINFORCE. Per il multi-turn si usa TD(λ = 1).

**Fine-tuning pubblico** (notebook per 2×T4 su Kaggle):
- La loss è RL più **1.0 × soft cross-entropy** verso le distribuzioni gold, quindi non è RL "puro".
- La temperatura viene fittata per tipo di domanda con L-BFGS su dati held-out.

> **Analisi** (condivisa da più commentatori). Con una policy gaussiana sui logit, REINFORCE è una stima zeroth-order (stile evolution strategies) del gradiente della scoring rule. Per σ → 0 coincide con il gradiente supervisionato, calcolabile esattamente con la backprop. Il log-score equivale alla soft cross-entropy.
>
> **Quando le reward derivano dalle etichette, l'"RL" aggiunge solo varianza.** L'RL ha senso solo con feedback d'ambiente o bandit (vedi §9.3).

### 4.4 Risultati [vendor, salvo diversa indicazione]

| Metrica | Valore |
|---|---|
| typed-decisions, base zero-shot | **0.362** (casuale 0.318; maggioranza 0.461) |
| typed-decisions, dopo fine-tuning sul train split | **0.766** (Jev 0.727) |
| Banking77 (77 classi) | **0.425** (Jev 0.870); il budget di token per le opzioni si esaurisce |
| AG News / DAIR Emotion | 0.950 / 0.595 (Jev 0.910 / 0.480) |
| ECE prima e dopo il refit della temperatura | 0.466 → 0.081 |
| Latenza su T4, 1 domanda | 32.8–39.5 ms; 103–332 domande/s in batch |
| MASSIVE (51 lingue, 20 opzioni) | 0.227 (EN) / 0.366 (multilingua) |
| Flip rate permutando le opzioni | 0.15–0.23 |

### 4.5 Critiche

- **Il confronto 0.766 vs 0.727 non è alla pari.** Laya è uno *specialista* addestrato sul train split, Jev è *generalista* zero-shot. Superare il "tetto" del teacher (0.735) indica, secondo la scheda stessa del dataset, che il modello ha imparato le idiosincrasie del teacher.
- **Il Brier "2.4× migliore" è un disallineamento di metrica.** Con un harness indipendente Laya ha Brier 0.213 contro 0.148 di Jev.
- **La latenza non è confrontabile.** Quella di Laya è misurata in locale, quella di Jev include il round-trip di rete.
- **Stress test indipendente** (gazelle93):

  | | Jev | Laya |
  |---|---|---|
  | Accuratezza a 128 candidati | 60% | 39% |
  | Risposte cambiate solo riordinando le opzioni | 14.6% | 49.4% |

- **Le rivendicazioni di "prior art"** (paper arXiv 2025 dell'autore) sono state contestate: non riguardano decisioni tipizzate con schema.

### 4.6 Dal codice: cosa fa davvero Laya e cosa ci serve (verificato il 28/09/2026)

Letti il runtime TypeScript ([receptron/laya](https://github.com/receptron/laya)), il codice Python ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya): `laya/common.py`, `laya/shortlist.py`, `research/`), la guida al fine-tuning e i limiti dichiarati nel README.

**Architettura (inferenza).**
- La sequenza è `[CLS] <tipo> question: istruzioni [SEP] [MASK] opzione0 [MASK] opzione1 … [SEP] stato [SEP]`. Un logit per ogni `[MASK]`, poi softmax sulle opzioni della domanda.
- `noul` è una scelta fra due opzioni descritte: `false: no, the statement does not hold` e `true: yes, the statement holds`. Le descrizioni si possono sostituire con `criteria` (`{"false": "la recensione è negativa", "true": "…positiva"}`), e le etichette mostrate al modello con `labels` (per esempio `B`/`A`).
- **La calibrazione in inferenza è solo una temperatura per tipo e per numero di opzioni** (`temperature_by_options`):

  | Fascia | Temperatura |
  |---|---|
  | `noul:2` | 1.98 |
  | `choice:2` | 1.91 |
  | `choice:3-5` | 1.76 |
  | `choice:6-10` | 1.00 |
  | `choice:11+` | 0.10 |
  | `score:3-5` | 1.25 |

  Non c'è un termine di spostamento: un'inclinazione sistematica verso una risposta (per esempio il "Sì") non si corregge in inferenza, solo con il training.
- **La testa `act` ("agire o passare la mano") non funziona:** vale ~1.0 quasi sempre, e ha AUROC 0.30 contro la correttezza (issue #185). La confidenza invece arriva a un AUROC di 0.77. Per decidere quando fidarsi basta la confidenza, se una testa dedicata non viene addestrata con un obiettivo proprio.

**Training (RLCD in pratica).**
- **Obiettivi:** le distribuzioni del teacher (soft), non etichette secche.
- **Loss:** la soft cross-entropy a peso pieno è il termine principale. Si aggiunge un termine di policy gradient su logit rumorosi, stile GRPO: 4 campioni, rumore da 0.4 a 0.1, reward log-score + sferica (0.5–0.75) − RPS per `score`.
- **Calibrazione post-hoc:** su un pezzo di dati tolto dal training *prima* di iniziare (fino a 400 esempi o il 10%). Si fitta una temperatura per tipo con L-BFGS su log T, limitata a [0.1, 10]. La calibrazione va salvata con i pesi: le temperature vecchie per fascia avrebbero la precedenza e annullerebbero quella nuova.
- **Costo:** con 2×T4 servono minuti per 6.000 decisioni e circa 4–5 ore per 30.000 domande.
- **TD(λ) nelle conversazioni:** ogni prefisso della conversazione riceve come obiettivo l'esito finale. Con λ = 1 coincide con il Monte Carlo.

**Fine-tuning come testa decisionale di un agente web** (`docs/finetune_browser_agent.md`, una sola RTX 4070 Ti SUPER da 16 GB, nessuna API a pagamento). È il caso più vicino all'uso "modello = attuatore" (giochi, robot):

| | Zero-shot | Fine-tuned |
|---|---|---|
| Elemento giusto su pagine nuove (~45 candidati) | 0.10 | **0.66** (421M) |
| Operazione giusta (click / scrivi / seleziona / fine) | 0.54 | 0.88–0.89 |
| Compiti reali riusciti | 0% | 50–62% |
| Latenza per passo | | 17–50 ms |

Lezioni:
- **Il formato dell'input conta più dei dati.** Portare i candidati fuori dallo stato e dentro le opzioni ha dato +7 punti e 10 compiti su 16 invece di 6.
- **Etichette pulite "al contrario":** si sceglie prima la risposta, poi un LLM locale scrive l'obiettivo che la richiede. Nessun teacher deve risolvere il compito.
- **Scorciatoie da evitare.** Obiettivi scritti con un modello fisso fanno imparare la frase, non il compito. "C'è una cronologia" finisce per voler dire "ho finito". Servono negativi a metà compito e le azioni rare ripesate (×3–×4).
- **Correzioni sulle proprie traiettorie** (DAgger): si fa girare il modello e si corregge dove sbaglia.
- **Passare i casi incerti a un LLM da 8B o 27B ha peggiorato i risultati:** su questi compiti il 322M addestrato decide meglio, e in 21 ms invece di 4.7 s.

**Valutazioni senza etichette** (`research/eval/`):
- **Metamorfiche:** permutare le opzioni o sostituire le etichette con lettere opache, poi confrontare le distribuzioni (tasso di risposte cambiate).
- **Opzioni identiche:** tutte le opzioni con lo stesso testo. Un modello senza preferenza di posizione dà una distribuzione uniforme; sul checkpoint multilingua ha rivelato una preferenza contro il primo livello di `score` (#131).
- **Confidenza:** misurata con AUROC contro la correttezza e con l'accuratezza sulla metà più sicura, perché l'ECE da solo dipende dalla scala.

**Difetti noti dei checkpoint:**
- `noul` può seguire le etichette `false`/`true` invece dello stato, con "no" sicuri su casi chiaramente positivi (#156). Si aggira con una `choice` a due opzioni con chiavi neutre `A`/`B`.
- Nelle domande negate la scelta segue la domanda, non lo stato (#377).
- Nei `choice` le etichette `yes`/`no` e `true`/`false` vanno evitate.
- Oltre ~20 opzioni il budget di token si esaurisce. Il rimedio è `predict_shortlist`: un embedding sceglie le `k` opzioni più vicine, poi si decide fra quelle.

**Confronto con Egeria sullo stesso split** (test di typed-decisions, 400 casi, 2.000 decisioni):

| Modello | Accuratezza |
|---|---|
| Laya base, zero-shot | 0.362 (sotto la maggioranza, 0.461) |
| Egeria, Qwen3.5-2B-Base zero-shot | 0.468 |
| Egeria, voto dei ricordi | 0.562 |
| Laya fine-tuned sul train split | 0.766 |

La distanza da 0.766 è ciò che deve colmare la fase F1.

**Limiti che si vedono dal codice, e cosa si può fare meglio:**

| Limite di Laya | Meglio |
|---|---|
| **Lo stato viene rielaborato per ogni domanda.** La domanda sta *prima* dello stato e l'encoder è bidirezionale, quindi niente si riusa: 1 domanda 39.5 ms, 10 domande 159 ms, 50 domande 771 ms | In un modello causale con lo stato **prima** della domanda (il nostro formato), lo stato si calcola una volta e ogni domanda costa solo la sua coda. Con le immagini il guadagno è ancora maggiore |
| **Dipende dall'ordine delle opzioni:** 13–23% di risposte cambiate dal loro test, 49% nello stress test indipendente. In inferenza non fa nulla | Noi mediamo 2 permutazioni, al costo di un calcolo doppio. Meglio ancora: addestrare con permutazioni e una loss di coerenza, così ne basta una |
| **Il termine RL è ridondante.** Con obiettivi noti, la soft CE dà già il gradiente esatto; il policy gradient rumoroso ne è una stima con più varianza e 4 campioni in più. Se il vantaggio è normalizzato per la deviazione standard del gruppo, spinge verso la sovraconfidenza, e infatti il modello esce sovraconfidente (ECE 0.466 prima del refit) | Loss supervisionata propria pura. RL solo con feedback reale (ambienti, bandit), baseline senza normalizzazione per la deviazione standard |
| **Il tetto è il teacher.** Gli obiettivi sono distribuzioni di un LLM e typed-decisions misura l'accordo con quel teacher, non la verità. Superare il suo tetto (0.766 > 0.735) vuol dire aver imparato le sue idiosincrasie | Mescolare etichette umane soft (più annotatori: ChaosNLI, Galaxy Zoo) e valutare anche su set etichettati da persone |
| **Calibrazione fragile.** Solo temperatura, senza spostamento. Fittata nel dominio del training (400 esempi). La fascia `choice:11+` vale 0.10, cioè il limite inferiore del fit: un fit degenerato. Fuori dominio la confidenza non regge (khmer: accuratezza 0 con confidenza 0.95) | Temperatura più spostamento per fascia; scartare i fit che toccano i limiti; misurare la calibrazione **fuori** dominio; soglie con garanzia (conformal risk control) |
| **La testa "agisci o passa la mano" non è addestrata** | O la si addestra con un obiettivo (la risposta era giusta?), o si usa la confidenza con soglie garantite |
| **Il budget di token per le opzioni è fisso:** 48 token per opzione, ~20 opzioni al massimo, stato a 512 token | Nel nostro formato le opzioni seguono lo stato senza un budget fisso; il limite è quello delle 26 lettere, superabile con la selezione preliminare via `embed` |
| **Solo scegliere:** non legge valori (date, importi), non vede immagini | Già coperto da `short_answer`, `estimate` e dagli stati con immagini |

---

## 5. SemIf e le altre repliche aperte

### 5.1 SemIf (TheoLeeCJ, MIT)

- **Idea.** Nessun training: prende **Qwen3.5-4B** (o MiniCPM5-2B, Qwen3-0.6B, oppure un 27B quantizzato EXL3). Mette nel prompt lo stato, la domanda e le opzioni etichettate con lettere, e legge **i logit delle lettere nell'ultima posizione**. La softmax è ristretta alle opzioni dichiarate.
- **Calibrazione.** Temperatura per workload, aggiunta di recente [incerto: le versioni precedenti erano dichiarate "non calibrate"].
- **Numeri** [community]:
  - 21 criteri binari su RTX 3090: 1.02 s con logit diretti contro 5.33 s generando un JSON.
  - Balanced accuracy 0.845 su un sottoinsieme TypeSafe (Jev 0.883).
  - Con stato condiviso e suffissi in parallelo: 20 decisioni/s.
- **Esecuzione.** CUDA, Apple MLX, CPU (llama.cpp/GGUF) e WebGPU nel browser.
- **Perché conta.** È la dimostrazione più semplice che **un LLM Qwen è già un modello decisionale** con il solo readout dei logit.

### 5.2 Repliche su decoder (le più vicine al nostro progetto)

| Progetto | Base | Readout | Training | Risultati [community] |
|---|---|---|---|---|
| **Kev** (J. Palmer, Apache-2.0) | Qwen3.5-Base 0.8B/4B/9B e **Qwen3.8-27B** | **Pointer head**: hidden di `</opt>` confrontato con il token `<decide>` | LoRA r16 + CE; T = 2.30 | Kev-9B OOD 0.852 (Jev 0.857), ECE 0.042. Kev-27B dev/test 0.848/0.896. Kev-4B: 18 ms per 6 domande su H100 |
| **Open-Jev** (Zefan Cai, MIT) | **Qwen3.8-27B** | Testa scalare inizializzata da Yes − No | LoRA; soft CE + 0.1·Brier; 148k righe | JevBench public 85.3% (Jev +3 risposte) |
| **decider** (Mapika) | Qwen3.5-2B/4B/35B-A3B-Base | Logit di lettere in uno slot per domanda | CE su circa 95 dataset + etichette da un teacher Qwen3.5-27B (1.47M esempi); v10 aggiunge RL calibrato su giochi e browser | **#1 JevBench v1.4.2** (decider-4b 64.13 vs Jev 63.29). ECE migliore del Decision Index. 18 ms (2B) su B300 |
| **JevK5** | Qwen3.5-4B + LoRA r16 | Lettere, T = 1.532 | Distillato da Qwen3.6-27B thinking | #2 JevBench v1.4; ECE 0.066; ~13 ms su H100 |
| **open-alternative-jev** | Qualsiasi LLM | Multi-domanda impacchettata, 2 permutazioni | Solo temperature scaling | Qwen3.6-27B **zero-shot** su typed-decisions: 73.7% acc, ECE 0.020 (Jev 72.7% / 0.144) |
| **"Your LM is already a decision model"** | Qwen3.5-9B zero-shot | Lettere ruotate | Nessuno | JevBench 80.5% vs Jev 86.1%; batte Jev su PhishNChips e MetaTool |
| **qwen3-0.6b-rlcd** | Qwen3-0.6B-Base + LoRA r32 | Hidden state per opzione | CE su frequenze soft (507k domande, 2.456 task) | Held-out: acc 0.783, ECE 0.031 |
| **eve-rlcd** | Qwen3-0.6B-Base | 26 lettere | RL con reward `c − p(a)` (= ½ gradiente di Brier) | Vedi §9.3: il supervisionato è pari o migliore |
| **AnyJev** (Nokia) | Qualsiasi LLM | Readout senza training + testa closed-form fittata su 100–500 etichette | – | Flip d'ordine 0.230 → 0.073; ECE 0.240 → 0.095 |
| **qwen-rlcd** | Qwen3.5-0.8B | Prototipo di **prefix-fork** dello stato DeltaNet | – | – |
| **SCX Router** (Knowledgator, arXiv 2609.02292) | Qwen3-0.6B causale + scorer bidirezionale leggero + testa GLiClass | Etichette valutate sopra un KV-cache persistente | 150k task | Blueprint ibrido decoder + encoder |

### 5.3 Repliche su encoder e servizi

- **GLiNER2.5-Decide** (Fastino, 24–25/09/2026):
  - DeBERTa-v3-large da 340M, schema-conditioned.
  - "Fast Decisions" (17 domini × 300 esempi): 60.2%, contro JevK5 57.6, SemIf 56.4, Laya 46.6.
  - Latenza 38 ms su V100.
- **meraGPT Decider 1**: chiuso, API compatibile con Jev. typed-decisions 0.768 zero-shot, Brier 0.052.
- **Featherless "Simple Jev"**: readout dei logit su modelli aperti (per esempio Qwen3.6-35B-A3B) con prompt cache condivisa; trainer "RFDT".

### 5.4 Cosa ci dicono le repliche

1. **Un Qwen di 4–9B con LoRA e readout dei logit raggiunge Jev** sui benchmark della community.
2. **Il gap principale è la conoscenza del backbone**, non l'architettura della testa. decider ha 0.51 contro 0.69 di Jev sull'area "knowledge" del Decision Index.
3. **Su Qwen3.8-27B esistono già due baseline aperte**, Kev-27B e Open-Jev-27B. Per essere utile, un nuovo modello deve **differenziarsi**: per esempio su lingua, astensione, invarianza all'ordine o dominio.

### 5.5 CLM (Stanford + NVIDIA, 23/09/2026): un dual encoder contrastivo

Codice letto il 29/09/2026 (commit bb42c6c), insieme alle schede Hugging Face.

- **Cos'è.** CLM-8B (*Contrastive Language Model*): J. Kwok, A. Mirhoseini, C. Ré, M. Pavone e altri, Stanford con NVIDIA. È Apache-2.0 (pacchetto `contrastive-lm`). Usa lo stesso formato di Jev: `POST /v1/systemone` con `noul`, `choice` e `score` (verificato nel codice). In più ha un `rank` su candidati liberi.
- **Il modello.**
  - Encoder: **Qwen3-8B congelato**. Non è un Qwen3.5: niente DeltaNet, niente immagini. È servito da vLLM in modalità *pooling* e usa l'embedding dell'**ultimo token** (4096 dimensioni). Gli stati sono troncati a 2048 token.
  - Sopra ci sono **due teste MLP**, una per lo stato e una per l'azione: 4096 → 1536 → 1536 → 512, con LayerNorm. Insieme fanno circa 19M di parametri (75 MB in fp32) e sono l'unica parte allenata.
  - Punteggio: `exp(logit_scale) · cos(testa_stato(s), testa_azione(c))`, poi una softmax sui candidati della domanda.
- **Come legge le domande.**
  - Lo stato, seguito dalle `instructions`, va nella testa stato.
  - Ogni opzione è un testo a sé nella testa azione: la sua descrizione, oppure la chiave.
  - Per `noul` i due candidati sono `true: Yes. This is true: {domanda}` e `false: No. This is false: {domanda}`.
  - Non c'è calibrazione, solo una temperatura passata dal chiamante.
- **Training** [vendor], in tre fasi:
  1. 60M coppie domanda-risposta Nemotron, con InfoNCE bidirezionale.
  2. 30M hard negative sintetici generati con Gemini 2.5 Flash-Lite. La top-1 su 10 negativi passa da 52.1% a 69.2%. Usati fin dall'inizio, invece, gli hard negative si fermano a 62.4%.
  3. 1M traiettorie di agenti (ADP, Endless-Terminals, LiteCoder). Ogni passo è una coppia (contesto, azione presa). Il mix contiene il 40% di replay di Nemotron: senza replay, la top-1 su Nemotron scende da 69% a 56.2%.
- **Velocità.** Le azioni si codificano una volta sola e restano in cache. Con candidati ripetuti, una domanda costa un embedding dello stato più un prodotto scalare. Su T-Rex: 2.6 ms di modello e 16.5 ms p50 lato client. Jev, via API, ne impiega 150.

**Come gestisce i benchmark di agenti (dal codice e dai file pubblicati):**

| Benchmark | Cosa fa davvero | Nota critica |
|---|---|---|
| T-Rex (gioco del dinosauro di Chrome, 5 percorsi × 60 s) | Un pianificatore fisico calcola quali azioni sono sicure e lo scrive nelle opzioni: `Safe. … Best.`, `Safe. …`, `Unsafe. … Collision.`. Al modello resta da leggere l'etichetta. Uno "scudo" sostituisce le risposte non sicure | Sopravvivono entrambi 5/5, ma CLM concorda col pianificatore solo nel **65.8%** delle decisioni (Jev 98.7%). Lo scudo interviene **4.883 volte** per CLM e 28 per Jev: il pari lo fa lo scudo. Il test misura la latenza, non la decisione |
| DeepSWE (verificatore best-of-4) | Una testa **allenata apposta** su 59 task (22.576 coppie prese da traiettorie riuscite). Ogni passo riceve il coseno stato-azione e la traiettoria vale la media degli ultimi 12 passi. Si sceglie la traiettoria con il valore più alto | 38 task held-out, di cui solo **13** decidibili. A caso: 28/38 (73.7%). CLM: **31/38** (81.6%). Oracolo: 34/38. Sono **3 task** in più del caso. Jev invece è usato zero-shot. Il dataset di valutazione citato nel README non risulta pubblico (29/09/2026) |
| Terminal-Bench 2.1 (87.6%) | Stesso schema, con candidati generati da Fable 5 | Testa e dati non pubblicati |
| BFCL v4, WikiRacing, Super Mario | Solo nel grafico del README. BFCL 95.2% (Jev 99.2%), WikiRacing 26/30 (Jev 30/30) [vendor] | Nessun codice nel repository |

La guida al fine-tuning (`docs/FINETUNING.md`) fa girare un agente che modifica il trainer in ciclo e tiene una modifica se migliora il tasso best-of-N "held-out". Così il set di test diventa il criterio di selezione. Non è detto che i numeri pubblicati vengano da quel ciclo; se sì, sono ottimisti.

**Cosa ci serve:**
1. **Invarianza all'ordine per costruzione.** Ogni opzione è codificata da sola, quindi il bias di posizione misurato in [09](09-controlli-senza-etichette.md) non esiste. Il prezzo è che stato e opzioni non si incontrano mai dentro il modello. Le negazioni e le etichette quasi uguali (`Safe`/`Unsafe`) diventano difficili, come mostra il 65.8% su T-Rex.
2. **Molte opzioni a basso costo.** Il nostro readout con le lettere si ferma a 26 opzioni; CLM ne tiene migliaia in cache (WikiRacing). Se servirà scegliere tra molti candidati, una testa contrastiva sopra il nostro `/v1/embed` è la strada più economica.
3. **Un'alternativa alla LoRA per F1.** Le teste si allenano su embedding pre-calcolati da un backbone congelato: richiede minuti, non ore. Si possono provare come baseline del primo task sulle immagini, da confrontare con la LoRA.
4. **Una ricetta dati per F2.** Prima dati ampi, poi gli hard negative (+7 punti rispetto a usarli da subito), più un 40% di replay per non dimenticare. Vale anche per le nostre LoRA.
5. **Cose da non copiare:**
   - confronti tra teste allenate sul benchmark e Jev zero-shot;
   - benchmark con la risposta scritta nelle opzioni;
   - selezione fatta sul set di test.

**Hardware.** Qwen3-8B in bf16 occupa circa 16 GB. Sulla nostra GPU da 8 GB andrebbe quantizzato, ma la testa è stata allenata su embedding bf16 e la quantizzazione li sposta: va verificato. È annunciato un CLM-35B-A3B multimodale per inizio ottobre 2026 [vendor, da articoli].

---

## 6. Benchmark e metriche

### 6.1 Benchmark specifici per modelli decisionali (tutti di settembre 2026)

| Benchmark | Contenuto | Note |
|---|---|---|
| **typed-decisions** (HF `LocalLLaMA/typed-decisions`, Apache-2.0) | 4 workflow sintetici. 1.200 casi di train e 400 di test (2.000 decisioni) | Etichette gold da un teacher "classe 4B" (3 campioni). Il tetto di auto-accordo del teacher è 0.735. **Non confrontare specialisti e generalisti** |
| **TypeSafe Workflow Evals** | Gli stessi 4 workflow | Misura l'accordo con GPT-6 Astra + Fable 5.1 |
| **JevBench v1.4.2** | 534 decisioni (hard, standard, easy, judge) | Composito: media geometrica di accuratezza, calibrazione, velocità e costo |
| **Decision Index v0.1** | 132k richieste, 37 benchmark (BFCL, RouterBench, MMLU, GPQA, GSM8K, ForecastBench…) | Le astensioni valgono 0 |
| **Fast Decisions** (Fastino) | 17 domini × 300 esempi | – |
| **nibzard DMB**, **decision-models-under-pressure** | Stress test: cardinalità, distrattori, ordine | – |

### 6.2 Benchmark classici utili

- **BTZSC** (22 dataset di classificazione zero-shot): il migliore è Qwen3-Reranker-8B con macro-F1 0.72.
- **Banking77**: 77 classi, mette alla prova l'alta cardinalità.
- **CLINC150**: 150 intent più richieste **out-of-scope**, quindi è il test naturale per l'astensione.
- **AG News**, **XNLI** e **MASSIVE**, questi ultimi due per il multilingua.
- **MTEB**: attenzione, la sua classificazione usa la regressione logistica su embedding, quindi non è zero-shot.

### 6.3 Metriche da riportare sempre

- **Accuratezza.** Tenere separate la modalità generalista (zero-shot) e quella specialista.
- **Qualità probabilistica.** NLL, **Brier**, **ECE** (dichiarando il binning), KL dalla distribuzione gold se soft.
- **Risk–coverage.** Per esempio la copertura a un budget d'errore del 5%, e l'accuratezza a p ≥ 0.9.
- **Robustezza.** Flip rate permutando le opzioni, coerenza fra domande equivalenti o negate, confidenza sugli item "inconoscibili".
- **Latenza.** Separare il tempo del modello dal round-trip di rete.

---

## 7. La famiglia Qwen 3.8

### 7.1 Checkpoint aperti (verificati via API Hugging Face il 26/09/2026)

| Repo | Tipo | Parametri | Licenza | Note |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` (+FP8) | Denso, multimodale | 27.78B (compresa la torre visiva) | **Apache-2.0** | Solo post-trained ("thinking" disattivabile). **Nessun Base** |
| `Qwen/Qwen3.8-2.4T-A95B` (+FP8) | MoE | 2.446T, 95B attivi | Licenza custom "Qwen3.8-Max" | Serve hardware di classe B300/GB300 |
| `Qwen/Qwen3.8-Flash-Next` (+FP8) | MoE + n-gram embedding ("anteprima Qwen4") | 125B, 6B attivi | qwen-community-1.0 | Richiede codice nuovo `qwen4_exp` |

- **Non esistono Qwen3.8 da 0.6–9B.** I piccoli con variante **Base** esistono solo in **Qwen3.5**: 0.8B, 2B, 4B, 9B e 35B-A3B, tutti Apache-2.0, con **la stessa architettura** e la stessa famiglia di tokenizer del 3.8.
- **Qwen3.8-27B è architetturalmente identico a Qwen3.5-27B e 3.6-27B** (`model_type: qwen3_5`, stesso numero di parametri). Cambia solo il post-training.

### 7.2 Architettura ibrida e implicazioni

**Qwen3.8-27B:**
- 64 layer, organizzati come `16 × (3 × Gated DeltaNet → 1 × Gated Attention)`.
- hidden 5120, vocabolario 248.320.
- Contesto nativo di 262k token (circa 1M con YaRN), 1 layer MTP.

**Qwen3.5 piccoli**, stesso schema 3:1:

| Modello | Layer | Hidden |
|---|---|---|
| 0.8B | 24 | 1024 |
| 2B | 24 | 2048 |
| 4B | 32 | 2560 |
| 9B | 32 | 4096 |

**Conseguenze pratiche:**

1. **Niente bidirezionalità "alla BERT".** I layer Gated DeltaNet sono causali per costruzione: ricorrenza più conv1d causale, non una maschera. Si può rendere bidirezionale **solo il 25% di layer full-attention**. transformers lo permette con una maschera a dizionario `{"full_attention": …, "linear_attention": …}`.
   - Unico precedente pubblicato: **dQwen3.5** (arXiv 2609.20751), che fa esattamente questo.
   - Nessuno ha ancora pubblicato risultati di classificazione con questa configurazione.
2. **Il pattern `[MASK]` prima dell'opzione di Laya non funziona su un modello causale**, perché il marker non vede il testo dell'opzione. I marker vanno messi **dopo** le opzioni, idealmente in un blocco finale.
3. **Tree attention mask e packing non isolano i rami.** La ricorrenza DeltaNet non si può mascherare per ramo.
   - Per più domande sullo stesso stato serve un **prefix-fork**: calcolare lo stato una volta e copiare KV + stato conv + stato ricorrente per ogni domanda.
   - Fare la backprop attraverso lo stato forkato non è banale, perché lo stato DeltaNet è aggiornato in-place.
4. **Bug noti:**
   - In transformers 5.2–5.8.1 il packing faceva trapelare lo stato fra campioni (corretto in 5.9.0).
   - Senza i kernel `fla`/`causal_conv1d` si ricade silenziosamente su PyTorch lento.
   - vLLM: il prefix caching sugli ibridi è sperimentale (NaN su Qwen3.8-27B, issue aperta) e ci sono problemi con LoRA sulle proiezioni GDN.
5. **Il readout va fatto in fp32.** In bf16 sono state osservate discrepanze fino a 2.3e-2 fra forme di batch diverse.

### 7.3 Hardware (stime Unsloth per LoRA bf16 su Qwen3.5; il 27B vale anche per il 3.8)

| Modello | LoRA bf16 | Full FT (stima) | Inferenza |
|---|---|---|---|
| Qwen3.5-0.8B | 3 GB | ~15 GB | – |
| Qwen3.5-2B | 5 GB | ~36 GB | – |
| Qwen3.5-4B | 10 GB | ~75 GB | ~60–80 ms per 1k token su 4090 (stima) |
| Qwen3.5-9B | 22 GB | ~155 GB | – |
| Qwen3.8-27B | **56 GB** | ~450 GB | 4-bit: 16–19 GB (4090/5090); ~100–150 ms per 1k token su H100 (stima) |

- **Unsloth sconsiglia QLoRA a 4 bit** sui modelli Qwen3.5.
- Serve transformers v5.
- Il supporto al fine-tuning di Qwen3.8 in Unsloth non è documentato, ma l'architettura è identica.

### 7.4 Tooling

- **transformers 5.17.** Ha già `Qwen3_5ForSequenceClassification` e `Qwen3_5ForTokenClassification`, con pooling sull'ultimo token.
- **PEFT/LoRA.** Moduli target:
  - DeltaNet: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
  - attention: `q/k/v/o_proj`
  - MLP: `gate/up/down_proj`
- **vLLM 0.30.** Supporta il pooling di classificazione con `classifier_from_token`, come per Qwen3-Reranker. Non ha una classe nativa `SequenceClassification` per Qwen3.5 [incerto sulla conversione generica].
- **ms-swift.** Supporta Qwen3.8 con task di classificazione, reranker ed embedding.

### 7.5 Precedenti Qwen di "decoder usato come scorer"

- **Qwen3-Embedding** (0.6B/4B/8B): pooling sull'ultimo token, #1 su MMTEB a giugno 2025.
- **Qwen3-Reranker**: softmax sui logit "yes"/"no" nell'ultima posizione. È di fatto un `noul`.
- **Qwen3Guard-Stream**: **testa di classificazione sugli hidden state** di Qwen3. È un modello diretto per le nostre teste.
- Non esistono ancora versioni ufficiali 3.5/3.8 di questi modelli.

---

## 8. Tecniche: da decoder a classificatore in un solo forward pass

### 8.1 Encoder vs decoder (evidenza)

- **A parità di dimensione vince l'encoder.** Ettin (ICLR 2026) su MNLI: encoder 400M 91.3 contro decoder 400M 88.2. L'encoder da 400M batte anche il decoder da 1B.
  - Convertire un decoder in encoder (50B token MNTP) arriva solo a 87.6.
- **I decoder più grandi con fine-tuning vincono.**
  - Gemma Encoder, GLUE con maschera bidirezionale: 84.5 → 89.4 (2B) e 87.5 → 90.9 (9B).
  - Un LLM causale fino a 8B con **testa sull'ultimo token** e LoRA 4-bit eguaglia o supera BERT fine-tunato (arXiv 2512.12677).
- **La conversione bidirezionale richiede training.**
  - LLM2Vec: maschera bidirezionale + MNTP + SimCSE.
  - **BidirLM** (2026): Qwen3-0.6B/1.7B bidirezionalizzati. Conclusione: "abilitare la bidirezionalità con un obiettivo di masking è critico".
  - Ha un token `[MASK]`, quindi il design di Laya è riusabile su backbone Qwen3 (non 3.5).
- **Zero-shot contro fine-tuned.** Gli encoder fine-tunati battono gli LLM zero-shot su task chiusi, a costi 30–100× inferiori. Gli LLM invece vincono nettamente in zero-shot e sulla conoscenza.

### 8.2 Readout in un solo passaggio con un decoder causale

| Tecnica | Come funziona | Precedenti | Note |
|---|---|---|---|
| **Logit delle lettere** | Opzioni etichettate A/B/C…; softmax ristretta sui token-lettera nell'ultima posizione | SemIf, decider, JevK5 | Ha un bias di posizione: mitigarlo con PriDe o permutazioni. Le lettere devono essere token singoli |
| **Yes/No** | Softmax su {yes, no} | Qwen3-Reranker, Open-Jev | Naturale per `noul` |
| **Testa sull'ultimo token** | Linear o MLP sull'hidden finale | `*ForSequenceClassification`, Qwen3Guard-Stream | Richiede un numero di classi fisso: poco adatta a opzioni dinamiche |
| **Marker dopo ogni opzione + testa** | Un token speciale dopo ogni candidato; testa sugli hidden dei marker | **Kev** (pointer `</opt>` × `<decide>`), jina-reranker-v3, YOJO | È l'equivalente causale di Laya. YOJO: 3.9× meno latenza, +10 punti |
| **Prefix sharing / tree mask** | Stato condiviso, domande in rami isolati | Jev (ipotesi), Hydragen, DeFT | **Non funziona su GDN**: serve il prefix-fork dello stato |
| **FIRST** | Ranking dai logit del primo token generato | FIRST (2024) | Per il reranking |

---

## 9. Calibrazione, scoring rule e RLCD

### 9.1 Fondamenti

- **Scoring rule strettamente proprie.** Log (NLL), Brier (quadratica), sferica, RPS/CRPS (per ordinali). Sono massimizzate in attesa solo dalla distribuzione vera (Gneiting & Raftery 2007).
- **L'ECE è incompleta:** ignora il "grouping loss". Va sempre accompagnata da NLL e Brier.
- **Temperature scaling** (Guo et al. 2017). Tutti i modelli decisionali la usano:

  | Modello | Temperatura |
  |---|---|
  | Kev | T = 2.30 |
  | JevK5 | T = 1.53 |
  | Laya | una per tipo di domanda e numero di opzioni |

- **Base vs post-trained.** I modelli **Base** sono meglio calibrati dei post-trained (RLHF, thinking), che sono sovraconfidenti (BaseCal, ACL 2026). È un argomento forte per partire da un **Base**, che Qwen3.8 non ha.
- **Bias sulle opzioni.** PriDe (ICLR 2024) stima e rimuove la preferenza per certe lettere o posizioni. Batch Calibration (ICLR 2024) corregge il bias contestuale.
- **Guard e LLM generalisti come moderatori** sono sovraconfidenti: la temperatura ottimale va da 2.7 a 9.0 (arXiv 2609.19072).

### 9.2 Scoring rule come reward in RL

- **RLCR, "Beyond Binary Rewards"** (ICLR 2026). Reward = correttezza − Brier(confidenza).
  - Le rule *limitate* (Brier) incentivano insieme accuratezza e calibrazione.
  - Il log-score non limitato può premiare risposte sbagliate e sicure.
  - HotpotQA: ECE 0.37 → 0.03 a parità di accuratezza.
- **GRPO è sovraconfidente** su esiti stocastici. La causa è la **normalizzazione per la deviazione standard del gruppo**: rimuoverla risolve (Bereket & Leskovec 2025). Laya la usa.
- **Altri lavori:**
  - Rewarding Doubt (log-score + PPO);
  - ConfTuner (Brier tokenizzato);
  - C2GSPG e CAPO;
  - "Confidence reward hacking" (2026): la scelta della reward dipende dal dataset, quindi va trattata come iperparametro.

### 9.3 Serve davvero l'RL? Evidenza da eve-rlcd (Qwen3-0.6B)

| Training | Accuratezza | Brier | ECE | Note |
|---|---|---|---|---|
| RLCD (reward `c − p(a)`) | 0.823 | 0.267 | 0.023 | Su un probe ambiguo (verità 0.5): 0.593 |
| RLVR (reward = correttezza) | 0.778 | – | 0.213 | Collassa a confidenza 0.99 |
| **Supervisionato** | 0.817 | **0.176** | **0.014** | – |

**Conclusione condivisa.** Quando esistono etichette o distribuzioni teacher, conviene **minimizzare direttamente una loss propria** (soft CE, eventualmente + Brier, + RPS per `score`).

L'RL con scoring rule ha senso solo con:
- **feedback bandit**, quando si osserva l'esito della sola opzione scelta;
- **esiti ritardati o multi-turn**;
- **ambienti veri**, come i giochi e i browser task di decider v10.

### 9.4 Astensione, cascata, conformal

- **Selective prediction.** Accuratezza a una data copertura; soglie diverse per rischio.
- **Cascate System 1 → LLM.** Mandare all'LLM solo i casi incerti dà qualità frontier al 25–50% del costo (NYU, 2609.24574).
- **Conformal prediction.** Insiemi di opzioni con copertura garantita, costruibili sopra qualsiasi softmax calibrata. È utile quando l'errore costa.

### 9.5 Distillazione da LLM

- I classificatori addestrati su etichette di LLM rendono circa quanto quelli su etichette umane.
- **Calibrare il teacher prima di distillare**: la calibrazione del teacher è correlata all'accuratezza dello student.
- **Mix di etichette hard e soft** è meglio di ciascuna da sola (ICML 2026).
- Teacher usati nella pratica: Qwen3.5-27B (decider) e Qwen3.6-27B thinking (JevK5). **Qwen3.8-27B è il candidato naturale per noi.**

---

## 10. Altri paradigmi "semantici" (JEPA, LCM, CALM…)

Sono il significato più ampio di "modello semantico": modelli che ragionano o predicono in uno **spazio di rappresentazioni** invece che di token.

| Lavoro | Idea | Risultato chiave | Praticabile su Qwen? |
|---|---|---|---|
| **LLM-JEPA** (Huang, LeCun, Balestriero; ICLR 2026) | Loss LM + λ·distanza fra la predizione dell'embedding di una "vista" e l'embedding dell'altra vista | Llama-3.2-1B: GSM8K 32.4 → 36.4. Costo di training circa 2× | **Sì**, come loss ausiliaria in fine-tuning. Servono coppie di viste (per esempio parafrasi dello stato) |
| **Semantic Tube Prediction** (2026) | Regolarizzatore stile JEPA sulle traiettorie degli hidden state | Stessa accuratezza con 16× meno dati | Sì, come regolarizzatore |
| **VL-JEPA** (Meta, dic 2025) | Predice *embedding* del testo target; decodifica solo se serve | −50% parametri addestrabili; batte CLIP/SigLIP2 in classificazione video | Solo l'idea: predire un embedding e confrontarlo con quelli delle opzioni |
| **The JEPA Paradox in Language** (lug 2026) | Il testo mascherato ha molte continuazioni valide, senza un centro latente | T-JEPA puro collassa in modo sistematico | **Avvertenza** contro la sola predizione latente sul testo |
| **Large Concept Models** (Meta, dic 2024) | Autoregressione su embedding di frase SONAR | Ottimo zero-shot multilingua | **No**: spazio diverso, richiede SONAR e pretraining |
| **CALM** (WeChat AI, ott 2025) | Autoencoder che comprime 4 token in un vettore; testa energy-score | −44% FLOP di training a parità di qualità | **No**: pretraining da zero |
| **Beyond Tokens** (EACL 2026) | Target come *insiemi di concetti* (parafrasi) invece del token esatto | Migliora perplessità e 7 task su Llama-3-8B | Sì, come post-training economico. Si sposa con target soft |
| **Coconut** e il ragionamento latente | Il ragionamento avviene nell'hidden state invece che nel CoT | Buono su pianificazione, misto su matematica | Poco rilevante per il System 1 |

**Verdetto.** Nessuno di questi paradigmi è necessario per un modello decisionale.
- Sono esperimenti economici e opzionali: la loss ausiliaria **LLM-JEPA/STP** con parafrasi dello stato, e i **target soft o a concetti**.
- Richiedono pretraining e sono fuori portata: LCM, CALM e il ragionamento latente.

---

## 11. Problemi aperti del settore

1. **La calibrazione non è portabile.** Dipende dalla distribuzione dei dati: va sempre ricalibrata sul proprio traffico con qualche centinaio di etichette.
2. **Incertezza aleatoria e "non lo so".** Moneta, dado e item inconoscibili sono gestiti male da quasi tutti.
3. **Listwise o indipendente.** Confrontare le opzioni fra loro (listwise) introduce sensibilità all'ordine e violazioni dell'IIA. Punteggi indipendenti sono invarianti all'ordine ma non gestiscono "nessuna delle precedenti".
4. **Coerenza fra domande.** Nessun sistema garantisce invarianti come P(A) + P(¬A) = 1 fra domande diverse.
5. **Alta cardinalità.** Con più di 20–77 opzioni gli encoder crollano; tutti degradano al raddoppio delle opzioni.
6. **Tetto di conoscenza.** In un solo passaggio non si fanno matematica multi-step, date o multi-hop: vanno delegati al codice o al System 2.
7. **Stati lunghi e prompt injection.** Una sola riga iniettata può far crollare l'accuratezza (openjev: 0.833 → 0.467). I training usano stati corti (384–4k token).
8. **Igiene della valutazione.** Gold da teacher, specialisti mescolati con generalisti, test split usati in training, metriche definite in modo diverso, latenze non comparabili.
9. **I dati sono il vero vantaggio competitivo** (TypeSafe lo dichiara). Le repliche aperte ereditano i bias dei teacher.

---

## 12. Punti incerti

- Architettura e dimensione di Jev: non dichiarate, solo ipotesi black-box.
- Date di lancio di Jev: 15, 18 o 19 settembre a seconda delle fonti. Qui si usa il 15, dal blog ufficiale e da Forbes.
- ECE di Jev: varia da 0.107 a 0.246 a seconda di task, binning e fonte.
- Classifiche JevBench: diverse fra le versioni (v1.4 e v1.4.2) e con 36, 76 o 93 sistemi.
- Qualità di Qwen3.5/3.8 in **italiano**: Qwen dichiara 201 lingue, ma non ci sono numeri per lingua.
- Qualità di classificazione con solo il 25% dei layer bidirezionali su Qwen ibrido: mai pubblicata.
- Stime di latenza e memoria per il full fine-tuning: calcoli teorici, da misurare.
- I nomi di modelli frontier (GPT-6 Astra, Fable 5.1, GPT-5.6 Sol, Opus 5…) sono riportati come appaiono nelle fonti.
