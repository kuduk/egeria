[Italiano](../fonti.md) · **English**

# Sources

Sources consulted for [01-state-of-the-art.md](01-state-of-the-art.md), accessed between 25 and 26/09/2026.

Many pages were read through automatic summaries. Before quoting a number in a formal context, double-check it against the original source.

## Jev / TypeSafe AI

**Official documentation**
- Launch blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Docs: https://docs.typesafe.ai/ · API: https://docs.typesafe.ai/api.md · Models: https://docs.typesafe.ai/models.md
- Confidence: https://docs.typesafe.ai/confidence.md · Jaggedness: https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
- Primitives: https://docs.typesafe.ai/primitives/noul.md · https://docs.typesafe.ai/primitives/choice.md · https://docs.typesafe.ai/primitives/score.md
- Consistency cookbook: https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md
- Workflow Evals: https://evals.typesafe.ai/

**CEO's blog and podcast**
- https://www.completeskeptic.com/p/the-bitterest-lesson
- https://www.latent.space/p/jev

**Press and funding**
- https://www.forbes.com/sites/the-prompt/2026/09/15/this-200-million-startup-wants-to-fix-ais-overconfidence-problem/
- https://www.wsgr.com/en/insights/wilson-sonsini-advises-typesafe-ai-on-dollar40-million-seed-round-as-company-emerges-from-stealth.html
- https://www.tomshardware.com/tech-industry/artificial-intelligence/typesafe-ais-jev-offers-an-alternative-to-llms-that-claims-to-be-193x-faster-and-445x-cheaper-system-one-type-model-is-bespoke-for-probabilistic-decision-making

**Third-party analyses**
- Black-box analysis: https://archerhume.com/posts/jevs-architecture-unmasked
- Calibration: https://www.alexmolas.com/2026/09/23/jev-cant-be-calibrated.html
- https://di-zhang-llm.github.io/blog/what-is-rlcd-the-secret-behind-jev/
- Community benchmarks: https://github.com/AbdelStark/jev-benchmarks · https://github.com/nibzard/decision-model-benchmark · https://github.com/gazelle93/decision-models-under-pressure
- NYU Abu Dhabi paper: https://arxiv.org/html/2609.24574v1
- HN discussions: https://news.ycombinator.com/item?id=49717558 · https://news.ycombinator.com/item?id=49765348

**Jev vs Laya comparison**
- https://wilsonwu.me/en/blog/2026/jev-vs-laya/
- https://www.eesel.ai/blog/laya-ai

## Laya / Convai Innovations
- Code: https://github.com/NandhaKishorM/laya
  - Issues cited: #131, #156, #185, #186, #377
  - Read on 28/09/2026 (commit 9d95567): `laya/common.py` (`proper_reward`, `td_lambda_targets`), `laya/shortlist.py`, `docs/finetune.md`, `docs/finetune_browser_agent.md`, `research/eval/metamorphic.py`, `research/eval/presentation_checks.py`, `research/scripts/confidence_definitions.py`
  - Read on 29/09/2026 for the label-free checks: `research/eval/presentation_checks.py`, `research/eval/README.md`, `BENCHMARKS.md` (answer changes with option order and between Chinese variants)
- TypeScript/ONNX runtime: https://github.com/receptron/laya (commit 6478649, 21/09/2026)
- Fine-tuned web agent: https://huggingface.co/cklxx/laya-browser
- Models: https://huggingface.co/convaiinnovations/laya · https://huggingface.co/convaiinnovations/laya-typed-decisions · https://huggingface.co/convaiinnovations/laya-multilingual
- Author's article: https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me
- Author's 2025 papers: https://arxiv.org/abs/2503.23303 · https://arxiv.org/abs/2510.01237
- Critique of the "prior art" claim: https://xtxinversexty.com/layas-prior-art-claim-is-absurd/
- Dataset: https://huggingface.co/datasets/LocalLLaMA/typed-decisions

## SemIf and other replications
- SemIf: https://github.com/TheoLeeCJ/SemIf
  - Read on 29/09/2026 (commit 23cf1f3): `benchmarks/build_perturbations.py`, `benchmarks/evaluate.py` (reversed criterion, states without the information), `docs/RESULTS.md`, `docs/CALIBRATION.md`
- Kev: https://github.com/jaredpalmer/kev
- Open-Jev: https://github.com/Zefan-Cai/Open-Jev · https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1
- decider: https://github.com/Mapika/decider
- JevK5: https://github.com/ngrok-adhoc/jevk5
- open-alternative-jev: https://github.com/ikermoel/open-alternative-jev
- "Your language model is already a decision model": https://github.com/ntlm1686/Your-language-model-is-already-a-decision-model
- qwen3-0.6b-rlcd: https://huggingface.co/thefloydd/qwen3-0.6b-rlcd
- eve-rlcd: https://github.com/anthony-maio/eve-rlcd · https://anthonymaio.substack.com/p/honest-about-uncertainty-i-tried
- AnyJev: https://github.com/nokia-applied-research/AnyJev
- qwen-rlcd: https://github.com/shamazharikh/qwen-rlcd
- SCX Router: https://arxiv.org/abs/2609.02292
- GLiNER2.5-Decide: https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model · https://huggingface.co/fastino/GLiNER2.5-Decide
- Services: https://meragpt.com/models/state-decider-1 · https://simple-jev.featherless.ai
- Community index: https://github.com/Amal-David/awesome-jev

## CLM (Stanford + NVIDIA)
- Code: https://github.com/Contrastive-LM/CLM
  - Read on 29/09/2026 (commit bb42c6c): `src/clm/schema.py`, `engine.py`, `heads.py`, `embedder.py`, `train/finetune.py`, `train/adapters.py`, `evaluation/bon_eval.py`, `docs/FINETUNING.md`, `examples/t_rex/` (including `results/*_realtime.json`)
- Weights and cards: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B · https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k (`verification.json`, `split_summary.json`) · https://huggingface.co/Contrastive-LM
- Blog: https://contrastive-lm.notion.site (could not be read automatically on 29/09/2026)
- Articles: https://venturebeat.com/technology/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests · https://aimag.no/en/nyheter/new-open-model-from-stanford-and-nvidia-picks-agent-actions-by-matching-not-token-generation · https://www.neoteo.com/en/stanford-and-nvidia-release-clm-8b-to-rank-ai-agent-actions · https://pasqualepillitteri.it/en/news/19074/clm-8b-nvidia-stanford-jev-en · https://www.kucoin.com/news/flash/stanford-and-nvidia-open-source-clm-8b-9x-faster-than-jev-in-decision-making · https://pro.edgex.exchange/news/article/stanford-nvidia-clm-8b-9x-faster-ai-agents
- Community reports: https://github.com/bianzhilong2-ctrl/agents-radar/issues/1336 · https://github.com/kouweizhu/agents-radar/issues/240

## Benchmarks
- JevBench: https://github.com/fstandhartinger/jevbench
- Decision Index: https://github.com/sahibzada-allahyar/decision-index
- BTZSC: https://arxiv.org/html/2603.11991
- CLINC150: https://aclanthology.org/D19-1131/

## Qwen

**Qwen 3.8**
- Model list via the HF API: https://huggingface.co/api/models?author=Qwen&search=Qwen3.8
- Models: https://huggingface.co/Qwen/Qwen3.8-27B · https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B · https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Announcement: https://qwen.ai/blog?id=qwen3.8
- vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- Unsloth guide: https://unsloth.ai/docs/models/qwen3.8

**Qwen 3.5**
- Base model: https://huggingface.co/Qwen/Qwen3.5-4B-Base
- Unsloth fine-tuning: https://unsloth.ai/docs/models/qwen3.5/fine-tune

**Qwen as a scorer**
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- Qwen3Guard: https://arxiv.org/abs/2510.14276

**Tooling**
- transformers: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- ms-swift issue on packing: https://github.com/modelscope/ms-swift/issues/9618
- vLLM issue: https://github.com/vllm-project/vllm/issues/55766

**Bidirectional hybrids**
- dQwen3.5: https://arxiv.org/html/2609.20751

## From decoder to encoder / classifier
- LLM2Vec: https://arxiv.org/abs/2404.05961
- NV-Embed: https://arxiv.org/abs/2405.17428
- Gemma Encoder: https://arxiv.org/html/2503.02656
- Ettin: https://arxiv.org/html/2507.11412
- BidirLM: https://arxiv.org/html/2604.02045
- Decoder with a classification head: https://arxiv.org/abs/2512.12677
- ModernBERT-Large-Instruct: https://arxiv.org/abs/2502.03793
- GLiGuard: https://arxiv.org/abs/2605.07982
- Encoders: https://huggingface.co/jhu-clsp/mmBERT-base · https://huggingface.co/EuroBERT/EuroBERT-610m · https://huggingface.co/answerdotai/ModernBERT-large
- Multi-option readout:
  - jina-reranker-v3: https://arxiv.org/html/2509.25085v3
  - YOJO: https://arxiv.org/html/2604.10966v2
  - FIRST: https://arxiv.org/abs/2406.15657
  - PriDe: https://arxiv.org/abs/2309.03882

## Calibration and RL
- Gneiting & Raftery 2007: https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Temperature scaling: https://arxiv.org/abs/1706.04599
- BaseCal: https://arxiv.org/abs/2601.03042
- Calibration of moderators: https://arxiv.org/abs/2609.19072
- RLCR: https://arxiv.org/html/2507.16806v2
- Rewarding Doubt: https://arxiv.org/abs/2503.02623
- ConfTuner: https://arxiv.org/abs/2508.18847
- GRPO overconfidence: https://arxiv.org/abs/2508.11800
- Confidence reward hacking: https://arxiv.org/abs/2607.04332
- Abstention: https://arxiv.org/abs/2407.18418
- Conformal prediction: https://arxiv.org/pdf/2405.01976
- Distillation:
  - https://arxiv.org/abs/2605.11954
  - https://arxiv.org/abs/2508.20224
  - https://arxiv.org/abs/2605.26246

## Semantic paradigms / latent space
- LLM-JEPA: https://arxiv.org/html/2509.14252v2
- Semantic Tube Prediction: https://arxiv.org/abs/2602.22617
- VL-JEPA: https://arxiv.org/abs/2512.10942
- The JEPA Paradox in Language: https://arxiv.org/abs/2607.23531
- Large Concept Models: https://arxiv.org/abs/2412.08821
- CALM: https://arxiv.org/html/2510.27688
- Beyond Tokens: https://arxiv.org/html/2601.11791
- Coconut: https://arxiv.org/html/2412.06769v3

## Option position, "Yes" and language (research of 29/09/2026)
- PriDe, Zheng et al., ICLR 2024: https://arxiv.org/abs/2309.03882
- Bias node pruning and auxiliary options, Choi et al., ACL 2025: https://arxiv.org/abs/2409.18857 · https://aclanthology.org/2025.acl-long.259.pdf
- Permutation bias metric, LoRA and KV-cached majority voting, Guda et al., IJCNLP-AACL 2026: https://arxiv.org/abs/2511.21709
- PA-GRPO, Zheng et al., ACL 2026: https://arxiv.org/abs/2603.21016
- MCQA selection bias in vision-language models: https://arxiv.org/pdf/2509.16805
- Acquiescence bias in LLMs, Braun, EMNLP Findings 2025: https://aclanthology.org/2025.findings-emnlp.607.pdf · https://arxiv.org/pdf/2509.08480
- Yes-no bias, answer order and wording, Huang, 2026: https://arxiv.org/abs/2607.05552
- Contrast-Consistent Search, Burns et al., ICLR 2023: https://arxiv.org/abs/2212.03827
- Calibrate Before Use, Zhao et al., ICML 2021: https://arxiv.org/abs/2102.09690
- Batch Calibration, Zhou et al., ICLR 2024: https://arxiv.org/abs/2309.17249
- RankC, cross-lingual consistency, Qi et al., EMNLP 2023: https://arxiv.org/abs/2310.10378
- Cross-lingual self-consistency, 2026: https://arxiv.org/html/2606.01464
