**Italiano** · [English](en/04-dynamic-depth.md)

# Profondità dinamica per asserzione

> **Proposta.** Ogni asserzione (domanda) usa solo la profondità di calcolo che le serve. Le decisioni facili escono presto, quelle incerte proseguono.
>
> **Stato.** Misurazione di fattibilità in F0 con readout "logit lens" ai confini di blocco. Risultati in [03-baseline-f0.md](03-baseline-f0.md).
>
> **Abbandonata il 29/09/2026.** Il risparmio potenziale (oracolo) era del 27–36%, ma quello ottenuto zero-shot con la confidenza solo dello 0.5–4%, e il percorso di esecuzione a blocchi complicava lo scorer. Il codice (uscita anticipata, letture intermedie, temperature per layer, `depth.py`, `--exits`, `--depth`) è stato tolto. `min_confidence` è rimasto: oggi serve solo a marcare le risposte incerte (`status`). Il documento resta come traccia dell'idea e delle misure.

## 1. Perché si adatta bene a un modello System 1

- **Nessun KV-cache da propagare.** Una decisione è **un solo forward pass** con readout nell'ultima posizione. Nell'early exit per la generazione (CALM, LayerSkip), se un token esce presto i token successivi non trovano i KV dei layer saltati, e bisogna ricopiarli o ricalcolarli. Qui questo problema non c'è.
- **Il risparmio è reale per ogni domanda.** Uscire dopo il layer `l` costa `(l+1)/L` del forward completo.
  - In batch, le righe che escono vengono tolte e il batch si compatta prima del blocco successivo.
- **Il segnale è già calibrato.** La confidenza di un modello decisionale è per costruzione una probabilità calibrata, quindi è il segnale naturale per decidere se fermarsi.
- **È nuovo.** Nessuno dei modelli decisionali esistenti (Jev, Laya, Kev, decider, SemIf) usa una profondità adattiva.

## 2. Le "profondità" possibili

| Livello | Cosa si adatta | Costo di una decisione facile | Note |
|---|---|---|---|
| **A. Layer (intra-modello)** | Uscita a un confine di blocco di Qwen3.5 | Da 4/L del forward in su | Oggetto della misurazione F0 |
| **B. Permutazioni** | Se l'ordine delle opzioni cambia la risposta, si valutano altre permutazioni | 1 forward invece di P | Riduce il bias d'ordine solo dove serve |
| **C. Modello (cascata)** | 0.8B → 4B → Qwen3.8-27B | Solo il modello piccolo | Come la cascata Jev → LLM del paper NYU (qualità frontier al 25–50% del costo) |
| **D. System 2** | Se è incerto anche il modello grande, si passa al ragionamento ("thinking") o all'umano | – | Astensione onesta |

I livelli si compongono: per esempio uscita al layer 12 del 0.8B, oppure 0.8B completo, oppure 4B, oppure thinking.

## 3. Vincoli dell'architettura Qwen3.5

- **Si esce solo ai confini di blocco.** I layer sono organizzati in blocchi `3 × Gated DeltaNet + 1 × full attention`. Il layer full-attention è l'unico che mescola globalmente tutta la sequenza. Uscire a metà blocco significa rinunciare all'ultimo mixing globale, quindi le uscite sensate sono i layer 3, 7, 11, … (indici 0-based).

  | Modelli | Layer | Uscite possibili |
  |---|---|---|
  | 0.8B / 2B | 24 | 6 |
  | 4B / 9B | 32 | 8 |

- **I readout intermedi hanno scale diverse.** Applicare la norm finale e l'lm_head agli hidden state intermedi (logit lens) produce logit con magnitudini molto diverse fra layer: nel 0.8B vanno da ~1 al blocco 1 a ~29 all'ultimo. Servono **temperature per layer** (come l'Annealed Confidence Threshold) o **teste d'uscita addestrate**.
- **Stato condiviso fra domande.** Con il prefix-fork dello stato (più domande sullo stesso stato), anche il prefisso va calcolato solo fino alla profondità massima richiesta dalle domande ancora attive. Si può calcolare in modo "pigro", blocco per blocco.

## 4. Criteri di uscita

| Criterio | Regola | Pro e contro |
|---|---|---|
| **Soglia di confidenza** | Esci se `max p_T(l) ≥ τ`, con `T(l)` la temperatura del layer | Semplice. Richiede calibrazione per layer |
| **Margine** | Esci se `p1 − p2 ≥ δ` | Più robusto con molte opzioni |
| **Patience** (PABEE) | Esci se l'argmax è uguale per `k` uscite consecutive | Non richiede calibrazione, ma parte tardi |
| **Garanzia conforme** (CATs) | `τ` scelta su un set di calibrazione in modo che `P(pred_uscita ≠ pred_completa) ≤ ε` | Garanzia statistica di coerenza con il modello completo |

## 5. Letteratura di riferimento

- **DeeBERT** (Xin et al., 2020) e **PABEE** (Zhou et al., NeurIPS 2020): early exit per BERT. Rispettivamente con entropia e con patience.
- **CATs, "Consistent Accelerated Inference via Confident Adaptive Transformers"** (Schuster et al., EMNLP 2021): soglie con garanzia conforme di coerenza con il modello completo.
- **CALM, "Confident Adaptive Language Modeling"** (Schuster et al., NeurIPS 2022): early exit per token nella generazione, con misure di confidenza calibrate. Da non confondere con CALM "continuous autoregressive" del 2025.
- **LayerSkip** (Elhoushi et al., ACL 2024): layer dropout più una **early-exit loss** su tutti i layer in training, con testa condivisa. È il modello per addestrare le uscite.
- **SimLens** (arXiv 2507.17618): predizioni latenti accurate dai layer intermedi con un token in più.
- **Annealed Confidence Threshold**: temperatura più alta nei layer bassi per contrastarne la sovraconfidenza.
  - https://www.emergentmind.com/topics/early-exit-large-language-models
- **Early exit bidimensionale** (arXiv 2604.18592): layer per frase. Speedup di 1.4–2.3× sulla classificazione del sentiment.
  - https://arxiv.org/html/2604.18592
- **HELIOS** (MLSys 2026): scelta dinamica di modello ed early exit nel serving.
  - https://arxiv.org/abs/2504.10724
- **Cascata System 1 → LLM** (NYU Abu Dhabi, arXiv 2609.24574): qualità frontier al 25–50% del costo.

## 6. Piano

| Fase | Attività |
|---|---|
| **F0** (fatto) | Readout logit lens ai confini di blocco (`egeria predict --exits blocks`). Temperature per (tipo, layer) fittate sul train. Simulazione delle soglie (`egeria evaluate --depth`): calcolo medio contro accuratezza, NLL ed ECE |
| **F1** | Teste d'uscita addestrate: LoRA + early-exit loss (proper, soft CE + Brier) su ogni confine di blocco, con peso crescente con la profondità. Alternativa: self-distillation dall'uscita finale verso le intermedie |
| **F3** | Politica d'uscita calibrata: soglia per tipo con garanzia conforme (CATs). Livello B (permutazioni adattive) e livello C (cascata 0.8B → 4B → 27B) |
| **F4** | Implementazione reale nel serving: esecuzione a blocchi con compattazione del batch e calcolo pigro del prefisso condiviso. Misura della latenza reale, non solo della frazione di layer |

**Metriche da riportare:**
- calcolo medio (frazione di layer) contro accuratezza, NLL, Brier ed ECE;
- istogramma delle uscite;
- tasso di disaccordo con il modello completo;
- latenza reale p50/p95.
