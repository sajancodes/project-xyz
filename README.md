# PROJECT XYZ — Small English LLM: From Next-Token Prediction to Semantic Behavior

## START HERE — NEW TO PROJECT XYZ?

1. Read [Project Overview](#project-overview) — what we are building and why.
2. Read [Project Timeline](#project-timeline) — the whole story in chronological order.
3. Read [How the Model Works](#how-the-model-works) — the architecture in plain language.
4. Read [Training History](#training-history) — every major experiment and lesson.
5. Read [How We Test the Model](#how-we-test-the-model) — evaluation philosophy and tools.
6. Read [Current State](#current-state) — exactly where the project stands today.
7. Read [The SEMV2E Mission](#the-semv2e-mission) — the active controlled experiment.
8. Follow the [Reproduction Guide](#reproduction-guide) when you are ready to run things.

Detailed documentation lives in the `docs/` folder (links are included in each section below).

---

## PROJECT OVERVIEW

### What are we trying to build?

A very small English language model (**14,657,664 parameters ≈ 14.66M**) trained **from scratch** on a single laptop GPU, and then progressively taught useful semantic behavior — not just fluent text generation.

### The original problem

A normal decoder-only language model is trained to predict the next token:

```
Input:   "The sun rises in the"
Target:  "east"
```

Predicting the next token is **NOT automatically the same thing as understanding**. A model can continue a sentence fluently yet fail to answer a question written in a slightly different way.

### Our research goal

Can a 14.66M-parameter model demonstrate **measurable task-level semantic behavior**?

- answering questions (factual QA)
- remembering facts seen earlier in the prompt (context retention)
- tracking entities across sentences (entity tracking)
- understanding relationships and their direction (relations, reverse relations)
- handling paraphrases (generalizing to unseen wording)
- following instructions
- performing simple reasoning (transitive chains, multi-hop)
- handling negation
- distinguishing similar facts (binding: entity ↔ object ↔ property)

**IMPORTANT**: we do NOT claim the model has human-like understanding. We measure "demonstrable task-level semantic behavior" on held-out examples — nothing more.

### The sharp distinction we care about

| Next-token prediction | Semantic / task-level behavior |
|---|---|
| "The sun rises in the" → "east" | "Where does the sun set?" → "west" (on unseen wording) |
| Fluency only | Correct answers on questions never trained on |
| Loss reduction | Held-out benchmark scores + real generation tests |

---

## PROJECT TIMELINE

| Stage | What | Result | Lesson / Next |
|---|---|---|---|
| 0 | Random init | no language | prepare tokenizer + model |
| 1 | Tokenizer + architecture | 16k BPE, 14.66M decoder-only | ready to train |
| 2 | FineWeb pretraining 0–106k | grammar 55.6%, factual QA **0%** | next-token ≠ QA → semantic fine-tuning |
| 3 | Semantic fine-tuning 5k | factual QA 100%, **grammar 33.3%** | catastrophic forgetting → mixed data |
| 4 | Fact injection (10,000× repetition) | no generalization | repetition ≠ understanding |
| 5 | FineWeb continued to 110k | grammar 44.4%, QA 28.6% | more pretraining is not the fix |
| 6 | Mixed FineWeb+Semantic (1k–10k) | mixed-2k best balance; oscillation after 2k | mixture works; constant LR is the problem |
| 7 | SEMV2 (semantic v2, 1k–5k) | peak composite 0.662→0.697 | loss masking + procedural data + LR schedule |
| 8 | SEMV2B (3k/5k) | 3k = 0.732, 5k = 0.676 | 3k sweet spot |
| 9 | SEMV2C (2k/3k) | **3k = 0.780 (BEST)** | arithmetic 0%, sun-set weak → SEMV2D/E |
| 10 | SEMV2D (designed) | **NOT MEASURED** (script exists, no verified training run) | script ≠ result |
| 11 | Real generation test | 5/15 (33%) on live prompts | benchmark ≠ free-form; add deterministic eval |
| 12 | SEMV2E (current) | TRAINING COMPLETED (5000 steps, loss 0.2976); EVALUATION PENDING | next: benchmark all milestones vs baseline |

Full story: [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)

---

## HARDWARE / ENVIRONMENT

- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM**: 6 GB
- **CUDA**: 12.8 (PyTorch 2.11.0+cu128)
- **OS**: Windows 11
- **Training throughput** (FineWeb, batch 16, seq 512): ~35,000–45,000 tok/s
- SEMV2E full run (5000 steps): **~963 s (~16 min)** on GPU

---

## HOW THE MODEL WORKS

### Architecture (14.66M params)

| Parameter | Value |
|---|---|
| Vocabulary size | 16,000 |
| Context length (max_seq_len) | 512 |
| Embedding size (d_model) | 256 |
| Layers (n_layers) | 8 |
| Attention heads (n_heads) | 8 |
| FFN dimension (d_ff) | 1,024 |
| **Total parameters** | **14,657,664 (14.66M)** |

- **Type**: Decoder-only Transformer
- **Normalization**: Pre-LayerNorm (pre-LN)
- **Attention**: causal self-attention (masked so a token can only see past tokens)
- **Activation**: GELU
- **Positional embeddings**: learned (a vector per position 0..511, added to token embeddings)
- **Output head**: Linear → 16,000 vocab logits (not weight-tied with the token embedding)

Full architecture explanation for beginners: [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)

### What "understanding" means here (and what it does NOT mean)

The model does NOT store facts like a database. Training adjusts its parameters so certain token sequences become more probable. Evaluation on held-out examples is the only way to know whether it generalized — that is the heart of this project.

---

## TOKENIZER

| Property | Value |
|---|---|
| Type | BPE (Byte-Pair Encoding), HuggingFace `tokenizers` |
| Vocabulary | 16,000 tokens |
| Special tokens | `<pad>` (0), unknown (1), `<bos>` (2), `<eos>` (3) |
| Pre-tokenizer | ByteLevel with prefix space |
| Decoder | ByteLevel |
| Trained on | 50,000 FineWeb documents (CC-MAIN-2025-26) |
| File | `tokenizer.json` (tracked in git) |

Because the tokenizer is byte-level, any text (even unknown words) can be represented via byte pieces.

---

## CHECKPOINT LINEAGE

```
none (random init)
└─ checkpoints/pretrain/checkpoint-106k.pt   (FineWeb pretraining, 106,000 steps, loss 3.5570)
   ├─ checkpoints/semantic/checkpoint-semantic-5000.pt (fine-tune 5k, loss 0.0050)
   ├─ checkpoints/pretrain/checkpoint_fineweb.pt        (FineWeb continued, loss 0.0020)
   ├─ checkpoints/mixed/checkpoint-mixed-2k.pt          (mixed 90/10 best-of-era, composite 0.514)
   ├─ checkpoints/semantic_v2/* (SEMV2 exp1: checkpoint_exp1_*.pt)
   │   └─ SEMV2B (exp2: checkpoint_exp2_*.pt, composite 0.732)
   │       └─ SEMV2C → checkpoints/best/checkpoint_semv2c-3000.pt (composite 0.780) ★
   │            └─ checkpoints/best/checkpoint_best.pt   (copy of semv2c-3000; the OFFICIAL baseline)
   └─ checkpoints/semv2e/* (SEMV2E from checkpoint_best.pt — training complete, eval pending)
        └─ checkpoint_semv2e-{500,1000,2000,3000,5000}.pt + -latest.pt + smoke tests
```

### The current BEST model (official baseline)

- **File**: `checkpoints/best/checkpoint_best.pt`
- **Metadata**: step 3000, loss 3.5710, `training_type = semantic_v2_mixed`, base = `checkpoints/pretrain/checkpoint-106k.pt`
- **Held-out composite**: **0.780** (protocol: 0.55·catA_strict + 0.15·catA_loose + 0.15·catB_topic + 0.10·no-drift + 0.05·no-rep)
- **Canonical greedy composite** (deterministic): **0.7463**
- Backups: `checkpoint_best_original.pt`, `checkpoint_semv2b-3000.pt`, `checkpoint_semv2c-3000.pt` (all in `checkpoints/best/`)

⚠️ **WARNING**: `checkpoints/latest_checkpoint.pt` is the **step-20 smoke-test** checkpoint (loss 3.6832, type `smoke_test`). It is NOT the best model. Never use it as the baseline. The official best is `checkpoints/best/checkpoint_best.pt`.

Full checkpoint guide (every file, step, loss, lineage): [docs/CHECKPOINT_GUIDE.md](docs/CHECKPOINT_GUIDE.md)

---

## TRAINING HISTORY

### Experiment 1 — FineWeb Pretraining (0 → 106k)
- FineWeb streaming (CC-MAIN-2025-26), 868M tokens, AdamW lr 3e-4, batch 16, seq 512.
- Final loss 3.5570.
- **Result**: grammar 55.6%, entity relations 100%, **factual QA 0%**, paraphrase 33.3%, instruction 16.7%, reasoning 16.7%.
- **Lesson**: next-token pretraining learns language structure but zero question answering.

### Experiment 2 — Semantic Fine-tuning (5k)
- 41 synthetic examples repeated; fresh AdamW lr 5e-5; loss 0.0050.
- **Result**: factual QA 0→100%, reasoning →100%, paraphrase →66.7%, **grammar 55.6→33.3% (catastrophic forgetting)**, instruction unchanged 16.7%.
- **Lesson**: tiny synthetic tuning destroys general language ability while teaching specific facts.

### Experiment 3 — Fact Injection (DEPRECATED)
- Repeated "sun rises in east" 10,000×.
- **Result**: no generalization (factual QA 0% on unseen phrasings).
- **Lesson**: repetition ≠ understanding; need diverse paraphrases.

### Experiment 4 — FineWeb Continued (106k → 110k)
- 4,000 more FineWeb steps; loss collapsed to 0.0020.
- **Result**: factual QA 0→28.6%, grammar 55.6→44.4%.
- **Lesson**: more pretraining alone doesn't teach facts; loss collapse = memorization.

### Experiments 5–12 — Mixed FineWeb+Semantic (1k → 10k)
- 90% FineWeb + 10% synthetic semantic; lr 1e-4 constant; starting from 106k.
- **Peak: mixed-2k** — grammar 55.6%, factual QA 100%, reasoning 100%, paraphrase 100% (composite 0.514).
- Beyond 2k scores oscillate (constant LR); by 10k factual QA 28.6%, reasoning 33.3%.
- **Lessons**: (a) mixed data prevents catastrophic forgetting; (b) **constant LR + longer runs cause oscillation** — 2k was the peak; need LR scheduling.

### Experiment 13 — SEMV2 (semantic v2, exp1)
- Procedural + Alpaca-cleaned data, **assistant-only loss masking**, warmup+cosine LR, FineWeb held-out val monitor.
- Milestones 1k→5k: composite 0.662 → **0.697 (peak 3k)** → decline.
- Lesson: structured semantic training + loss masking + LR schedule beats old mixed runs; overtraining after 3k.

### Experiment 14 — SEMV2B (exp2, 3k continuation)
- 90/10 FineWeb, lr 5e-5, extra arithmetic + transitive data.
- **Result**: semv2b-3000 composite 0.732 (best), semv2b-5000 0.676. Reduced overfit (val 3.57 vs train 3.48).

### Experiment 15 — SEMV2C (2k/3k) ★ BEST
- Continuation targeting weak categories (sun, entity tracking, instruction).
- semv2c-2000: composite 0.751 (arithmetic 33% — only nonzero arithmetic ever recorded).
- **semv2c-3000: composite 0.780** — copied to `checkpoints/best/checkpoint_best.pt`.
- Per-category (strict): relations 100%, state 100%, transitive 100%, instruction 86%, why_context 80%, entity_tracking 80%, negation 75%, sun_facts 50%, **arithmetic 0%**.

### Experiment 16 — Real Generation Test
- 15 ad-hoc prompts, checkpoint_best.pt, chat wrapper ON, temp 0.3, top-k 40.
- **Score 5/15 (33%)** — see [Real Generation Test](#real-generation-test).
- **Lesson**: a high composite on a structured benchmark can coexist with poor free-form generation on harder prompts; both are real measurements.

### Experiment 17 — SEMV2D (designed, NOT MEASURED)
- `src/training/train_semantic_v2d.py` exists: 50% semantic / 20% arithmetic / 15% sun / 15% FineWeb.
- **There is NO verified SEMV2D training run or evaluated checkpoint on disk.** SEMV2D is therefore **BLOCKED / NOT MEASURED** — this corrects an earlier README claim of "Mission complete." A script is not a result.

### Experiment 18 — SEMV2E (current; training complete, evaluation pending)
- See [The SEMV2E Mission](#the-semv2e-mission) below.

Full experiment-by-experiment detail: [docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md)

---

## HOW WE TEST THE MODEL

### Why evaluation matters

Training loss does NOT measure understanding. A model can memorize 10,000 copies of "John has a red car" and still fail on "What color is John's car?" phrased differently. Every benchmark below uses **held-out examples the model has never trained on**.

### Data splits

- **TRAINING data** — the optimizer sees it.
- **VALIDATION data** — FineWeb held-out window, used ONLY to monitor regression (not to tune).
- **HELD-OUT TEST data** — benchmarks/diagnostics, never trained on.
- **AD-HOC REAL TESTS** — free-form hand-written prompts run interactively.

### Evaluation files

| File | Measures | Decoding |
|---|---|---|
| `src/evaluation/heldout_benchmark.py` | Cat A (10 categories, 51 prompts) + Cat B generation + plain QA | temp 0.3, top-k 40 |
| `src/evaluation/canonical_eval.py` | same core — **greedy (temp 0)**, full metadata, + sun/arithmetic/plain diagnostics | greedy (deterministic) |
| `src/evaluation/semv2e_benchmark.py` | 600-question binding benchmark, 10 categories (A–J) | greedy |
| `src/evaluation/arithmetic_diagnostic.py` | arithmetic (120 Q) | temp 0.3 |
| `src/evaluation/sun_diagnostic.py` | sunrise/sunset direction (26 Q) | temp 0.3 |
| `src/evaluation/plain_qa_diagnostic.py` | plain (no chat wrapper) QA | temp 0.3 |
| `src/evaluation/targeted_diagnostics.py` | why_context, negation, relation_reversal, transitive, arithmetic | temp 0.3 |
| `src/evaluation/evaluation_suite.py` | classic 8-category suite (grammar, factual QA, ...) | old protocol |
| `src/evaluation/verify_baseline.py` | re-verify checkpoint_best.pt metadata | — |

### Composite score (official protocol)

```
composite = 0.55 * catA_strict + 0.15 * catA_loose + 0.15 * catB_topic
          + 0.10 * (1 - catB_dialog_drift) + 0.05 * (1 - catB_repetition)
```

**A composite of 0.780 is NOT "78% understanding".** It is a weighted index of keyword-match scores on held-out examples. Free-form quality is measured separately.

### Cat A categories (heldout benchmark)

| Category | Tests |
|---|---|
| sun_facts | sunrise→east, sunset→west |
| relations | simple ownership / relationship |
| state | location state |
| why_context | why-questions over context facts |
| negation | simple negation |
| entity_tracking | track one entity across sentences |
| transitive | chain reasoning (every dog barks, Jenny is a dog) |
| arithmetic | simple computation |
| instruction | follow format/content instructions |
| context_retention | remember earlier fact |

### Cat B categories (generation)

Topic relevance, repetition ratio, dialog drift (spontaneous `User:`/`Assistant:` markers). These measure free-form fluency, not correctness.

### Benchmark leaderboard (chronological, held-out composite)

| Model | Composite | catA strict | Note |
|---|---|---|---|
| checkpoint-106k.pt | 0.380 | 24% | pretraining only |
| checkpoint-mixed-2k.pt | 0.514 | 39% | mixed best-of-era |
| semv2-1000 | 0.662 | 57% | |
| semv2-3000 | 0.697 | 65% | SEMV2 peak |
| semv2b-3000 | 0.732 | 71% | |
| semv2c-2000 | 0.751 | 76% | only nonzero arithmetic (33%) |
| **semv2c-3000 (checkpoint_best.pt)** | **0.780** | 75% | **BEST / OFFICIAL BASELINE** |
| semv2c-3000 (canonical greedy) | 0.746 | 73% | deterministic re-measure |
| SEMV2D | NOT MEASURED | — | no verified run |
| SEMV2E milestones | PENDING | — | training complete, eval next |

Full evaluation guide and per-category definitions: [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)

---

## REAL GENERATION TEST

Benchmark scores and free-form generation are **different measurements**. A model can score 0.78 on a structured benchmark yet fail 2/3 of hand-written prompts. We record this honestly.

**Configuration**: `checkpoints/best/checkpoint_best.pt` (step 3000), chat wrapper ON (`User: ...\nAssistant:`), temp 0.3, top-k 40, max_new_tokens 40, CPU.

**Score: 5/15 (33%)**

| # | Category | Expected | RAW output (verbatim) | Result |
|---|---|---|---|---|
| 1 | Simple fact QA | red | The book is a red book. | PASS |
| 2 | Entity property | green | The green bicycle is a green bicycle. | PASS |
| 3 | Subject→object | a red car / car | Sajan. | FAIL |
| 4 | Multi-fact reasoning | red | Sajan is a beautiful green car. | FAIL |
| 5 | Entity disambiguation | red | The blue bicycle is a blue bicycle. | FAIL |
| 6 | Reverse relation | Sajan | Sajan is a red car. | PASS |
| 7 | Paraphrased question | red | The color of his car is blue. | FAIL |
| 8 | Context retention | a red car / car | Sajan is a student. | FAIL |
| 9 | Simple world knowledge | east | The sun rises to the east. | PASS |
| 10 | Sunset | west | The sun sets the tone for the sun. | FAIL |
| 11 | Instruction following (one word) | green | Yellow. | FAIL |
| 12 | Negation | on the ground | The bird is not flying. | FAIL |
| 13 | Simple relation | yes / Jenny barks | Yes, Jenny is a dog barks. | PASS |
| 14 | Longer context | a green book | Sita is on the table. | FAIL |
| 15 | Previous failure (multi-fact) | red | Sajan is a streetwear brand. | FAIL |

**Can do**: single-fact object-property retrieval, sun-rises-east, simple forward relations.
**Cannot do**: multi-fact reasoning/disambiguation, paraphrase (color wrong), sunset/west, negation inference, longer-context retention, instruction content.

---

## CURRENT STATE

**As of 2026-08-13** (after SEMV2E training completed; evaluation pending):

### Model
- `checkpoints/best/checkpoint_best.pt` — 14.66M decoder-only Transformer, step 3000, composite 0.780, canonical greedy 0.7463.

### Verified capabilities (held-out, canonical greedy of baseline)
| Capability | Level |
|---|---|
| relations / state / transitive / context_retention | 100% |
| instruction following | 86% |
| why_context / entity_tracking | 80% |
| negation | 75% |
| sun_facts | 33% (rise_east ~62%, set_west ~12%) |
| arithmetic | 0% (every protocol, every scale) |
| real free-form chat (15 prompts) | 33% (5/15) |

### SEMV2E binding benchmark (baseline, greedy, 600 prompts)
- **Chat-wrapped: 43.2% (259/600)** · **Plain (no wrapper): 25.8% (155/600)**
- Strong: relation_reversal 75%, paraphrase 58%, single_fact 53%, multi_hop 52%.
- Weak (the binding gap): distractor 25%, pronoun 30%, entity_generalization 33%, distractor_binding 32%, multi_fact 37%.
- **The chat wrapper matters enormously** — D_relation_reversal is 75% wrapped vs 0% plain.

### Known weaknesses
1. **Arithmetic is 0%** across every protocol and scale tested.
2. **Sunset (west)** is weak (~12%) vs sunrise (east, ~62%) — phrase-template overfitting.
3. **Multi-fact binding** is fragile — entity confusion with distractors present.
4. **Format dependence** — scores drop sharply without the chat wrapper.
5. **Repetition / echo** in generation, especially after fine-tuning.

---

## THE SEMV2E MISSION

### Purpose
Teach robust **semantic binding**: entity → relationship → property → context → correct answer, using leakage-safe procedural data, then verify with a new 600-question benchmark that the baseline (43.2% chat / 25.8% plain) scores poorly on.

### Design (mission sections 9–10)
- **Training data** (`src/utils/semv2e_data.py`): procedural, leakage-safe — training names (Aaron, Bella, Caleb, ...) are **DISJOINT** from benchmark names (Maya, Ravi, Sita, ...); an overlap guard raises a hard error. 20,000 binding + 20,000 existing semantic + 8,000 Alpaca instruction + 1,000 arithmetic examples.
- **Mixture per batch (16 rows)**: 55% binding (9) + 15% instruction (2) + 15% existing semantic (2) + 10% FineWeb (2) + 5% arithmetic (1).
- **Loss**: assistant-answer-only masking preserved from semantic-v2.
- **Optimizer**: AdamW, peak LR 1e-4, warmup 150 + cosine to 10%.
- **Base**: `checkpoints/best/checkpoint_best.pt` (the 0.780 baseline).
- **Milestones**: 500 / 1000 / 2000 / 3000 / 5000 steps.

### Evaluation
- **Baseline measured** (before training): chat 43.2% / plain 25.8% on the 600-question benchmark; canonical greedy composite 0.7463.
- **Training completed**: 5000 steps in ~963 s, final train loss 0.2976, FineWeb held-out val 4.3915. Checkpoints saved to `checkpoints/semv2e/`.
- **EVALUATION OF MILESTONES IS THE CURRENT NEXT STEP** — each of the 5 milestones must be benchmarked and compared to the baseline before any promotion.

### Promotion rule (a milestone becomes the new best only if)
1. Benchmark improves **genuinely** (deterministic, reproducible) over 0.780/0.7463.
2. Meaningful improvement in semantic binding.
3. **No unacceptable regression** in strong areas (relations, state, transitive, context retention).
4. Held-out validity maintained (leakage rules still enforced).
5. Does not depend on decoding randomness.

Full design, data, and promotion discussion: [docs/SEMANTIC_RESEARCH.md](docs/SEMANTIC_RESEARCH.md)

---

## DATA STRATEGY

### Sources
1. **FineWeb** (HuggingFaceFW/fineweb, CC-MAIN-2025-26) — streaming pretraining + preservation share in mixed/SEMV2E runs.
2. **Synthetic semantic/instruction corpus** — grew from 41 examples (exp 1) → procedural + Alpaca (SEMV2) → leakage-safe binding data (SEMV2E).
3. **Held-out benchmarks** — fixed, evaluation-only, never trained on.

### Data safety rules (the project's guardrails)
- Never repeat a fixed string thousands of times — templates must vary names, objects, colors, relations, structures, wording, order, length.
- Generate underlying facts and **multiple surface realizations** (owns/has/possesses/keeps; "What does X own?"/"Who owns the X?"/...).
- **Hold out** entities/templates: SEMV2E training names are disjoint from benchmark names (runtime assertion).
- **TRAINING DATA MUST NEVER CONTAIN THE HELD-OUT TEST EXAMPLES.**

Full data guide (every data type, example, risk): [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md)

---

## REPRODUCTION GUIDE

### Environment
```powershell
cd "C:\Users\Sajan\Desktop\Project XYZ"
.venv\Scripts\activate
# Verify CUDA torch:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.11.0+cu128 True
```

### Training
```powershell
python src\training\train_fineweb.py            # FineWeb pretraining (resumable)
python src\training\train_semantic_v2.py        # SEMV2 (procedural+Alpaca)
python src\training\train_semv2e.py --smoke     # SEMV2E smoke test (10 steps, no network)
python src\training\train_semv2e.py --steps 5000 --name semv2e   # real SEMV2E run
```

### Evaluation
```powershell
python src\evaluation\heldout_benchmark.py checkpoints\best\checkpoint_best.pt --output experiments\bench_checkpoint_best.json
python src\evaluation\canonical_eval.py checkpoints\best\checkpoint_best.pt --output experiments\eval_checkpoint_best.json
python src\evaluation\semv2e_benchmark.py eval checkpoints\best\checkpoint_best.pt --output experiments\semv2e_checkpoint_best.json
python src\evaluation\sun_diagnostic.py checkpoints\best\checkpoint_best.pt --output experiments\sun_checkpoint_best.json
python src\evaluation\arithmetic_diagnostic.py checkpoints\best\checkpoint_best.pt --output experiments\arith_checkpoint_best.json
python src\evaluation\plain_qa_diagnostic.py checkpoints\best\checkpoint_best.pt --output experiments\plainqa_checkpoint_best.json
```

### Interactive generation
```powershell
python src\evaluation\test.py
```

### Experiment tracking
```powershell
python src\utils\experiment_tracker.py lineage
python src\utils\experiment_tracker.py record ...   # add an experiment
```
Auto-generated summary: `experiments/SUMMARY.md`. Raw logs: `experiments/results.jsonl`, `experiments/summary.json`.

---

## TROUBLESHOOTING (quick list)

| Symptom | Cause / Fix |
|---|---|
| FineWeb download hangs | First run downloads; use `--smoke` (preset buffer) |
| `python -c "..."` parse errors in PowerShell | PowerShell mangles quotes; write a .py file instead |
| torch is `2.13.0+cpu` (no CUDA) | Some scripts inject the uv CPU-torch path; use the venv torch (2.11.0+cu128) |
| Wrong results from `latest_checkpoint.pt` | That's the step-20 smoke test; use `checkpoints/best/checkpoint_best.pt` |
| Repetition/echo in output | High temp + prompt-continuation patterns; eval treats echo as FAIL |
| Same prompt, different answers (13/15 vs 5/15) | Stochastic sampling; use canonical greedy eval for benchmark decisions |
| cp1252 Unicode errors on Windows | Set `PYTHONIOENCODING=utf-8` or write output to files |

Full troubleshooting with verified fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## CHECKPOINT SAFETY

1. **NEVER overwrite** the best checkpoint (`checkpoints/best/checkpoint_best.pt` + backups).
2. Before any experiment: back up the best.
3. During experiments: write to a NEW experiment directory (e.g. `checkpoints/semv2e/`).
4. At milestones: save immutable checkpoints.
5. Promotion rule (see SEMV2E mission) — no promotion without verification.
6. Integrity check: `checkpoint_best_original.pt` and `checkpoint_best.pt` must have identical SHA-256.

Full checkpoint inventory and safety rules: [docs/CHECKPOINT_GUIDE.md](docs/CHECKPOINT_GUIDE.md)

---

## PROJECT PRINCIPLES

1. **Never declare success from memorization** — must generalize to unseen wording.
2. **Preserve best checkpoints** — never overwrite the only copy of a working model.
3. **Document everything** — failures as thoroughly as successes.
4. **Measure generalization** — paraphrase, OOD, held-out tests.
5. **Prevent catastrophic forgetting** — mixed data, replay, lower LR.
6. **Autonomous loop** — Train → Evaluate → Analyze → Improve → Train.
7. **README never lies** — UNKNOWN / NOT MEASURED for unknowns.

---

## FILES IN PROJECT

### Source (`src/`)
| Path | Purpose |
|---|---|
| `src/models/model.py` | SmallEnglishLLM architecture (14.66M params) |
| `src/config/model_config.py` | ModelConfig dataclass + param counter |
| `src/paths.py` | central project paths (import this!) |
| `src/training/train_fineweb.py` | FineWeb streaming training (resumable) |
| `src/training/train_fineweb_fast.py` | optimized FineWeb training |
| `src/training/train_semantic_v2.py` | SEMV2 training |
| `src/training/train_semantic_v2d.py` | SEMV2D (designed, NOT verified) |
| `src/training/train_semv2e.py` | SEMV2E binding training (CURRENT) |
| `src/training/train_mixed.py` | mixed FineWeb+semantic |
| `src/training/train_fact_injection.py` | DEPRECATED fact injection |
| `src/utils/semantic_data.py` | SemanticGenerator + Alpaca subset loader |
| `src/utils/semv2e_data.py` | SEMV2E leakage-safe procedural generator |
| `src/utils/experiment_tracker.py` | experiment logging & lineage |
| `src/evaluation/*` | heldout_benchmark, canonical_eval, semv2e_benchmark, diagnostics, ... |

### Data & results
| Path | Purpose |
|---|---|
| `tokenizer.json` | 16k BPE tokenizer (tracked) |
| `data/semv2e_benchmark.json` | 600-question held-out SEMV2E benchmark |
| `experiments/bench_*.json` | heldout benchmark results |
| `experiments/eval_*.json` | classic/other eval results |
| `experiments/semv2e_*.json` | SEMV2E benchmark + canonical baseline |
| `experiments/results.jsonl` / `summary.json` / `SUMMARY.md` | experiment tracker |
| `checkpoints/*` | model checkpoints (gitignored — keep backups) |

### Documentation
| File | Purpose |
|---|---|
| `README.md` | THIS FILE — master entry point |
| `PROJECT_SUMMARY.md` | compact state-of-the-project |
| `TRAINING_HISTORY.md` | chronological training log (kept as history) |
| `BENCHMARK_HISTORY.md` | all benchmark runs with scores |
| `docs/PROJECT_HISTORY.md` | full narrative history |
| `docs/MODEL_ARCHITECTURE.md` | architecture for beginners |
| `docs/TRAINING_GUIDE.md` | how training works + reproduce runs |
| `docs/EVALUATION_GUIDE.md` | evaluation system in depth |
| `docs/EXPERIMENT_HISTORY.md` | experiment-by-experiment detail |
| `docs/CHECKPOINT_GUIDE.md` | checkpoint inventory + safety |
| `docs/DATA_GUIDE.md` | every data type + safety rules |
| `docs/TROUBLESHOOTING.md` | problems, causes, verified fixes |
| `docs/SEMANTIC_RESEARCH.md` | the research thread + SEMV2E design |
| `docs/GLOSSARY.md` | plain-language definitions |

---

## GLOSSARY (quick)

| Term | Definition |
|---|---|
| Token | a piece of text (word/subword/byte) as an integer ID |
| BPE | Byte-Pair Encoding tokenizer |
| d_model | hidden/embedding size (256) |
| Pre-LN | LayerNorm before the attention/FFN sub-blocks |
| Loss masking | restrict loss to the Assistant answer span |
| Composite | weighted index of benchmark scores (not "understanding") |
| Cat A / Cat B | semantic categories / generation quality categories |
| catA_strict | fraction of Cat A answers with keyword in first 8 tokens |
| Held-out | examples never trained on |
| Catastrophic forgetting | new training destroys old ability (grammar 55.6→33.3%) |
| OOD | out-of-distribution / unseen |

Full glossary: [docs/GLOSSARY.md](docs/GLOSSARY.md)

---

*Last updated: 2026-08-13 (documentation mission: repository fully audited and documented; SEMV2E training complete, milestone evaluation pending)*
*Document created during active experimentation — updated continuously.*
