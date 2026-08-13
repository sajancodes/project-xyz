# PROJECT XYZ — Small English LLM Training Documentation

## PROJECT OVERVIEW
Building a small English language model (14.66M parameters) capable of progressively developing useful language understanding: grammar, factual knowledge, entity relationships, paraphrase generalization, instruction following, context retention, and simple reasoning.

**Goal**: Not just fluent text generation, but demonstrable semantic understanding — the model should answer questions, follow instructions, and generalize to unseen wording.

---

## HARDWARE
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM**: 6 GB
- **CUDA**: 12.8 (PyTorch 2.11.0+cu128)
- **OS**: Windows 11

---

## MODEL ARCHITECTURE
| Parameter | Value |
|-----------|-------|
| Vocabulary size | 16,000 |
| Context length | 512 |
| Embedding size (d_model) | 256 |
| Layers (n_layers) | 8 |
| Attention heads (n_heads) | 8 |
| FFN dimension (d_ff) | 1,024 |
| **Total parameters** | **14,657,664 (14.66M)** |

Architecture: Decoder-only Transformer with pre-LN, causal self-attention, GELU activations, learned positional embeddings.

---

## TOKENIZER
- **Type**: BPE (Byte-Pair Encoding) via HuggingFace tokenizers
- **Vocabulary**: 16,000 tokens
- **Special tokens**: `<pad>` (0), `▁`/unk (1), `<bos>` (2), `<eos>` (3)
- **Pre-tokenizer**: ByteLevel with prefix space
- **Decoder**: ByteLevel
- **Training data**: 50,000 FineWeb documents (CC-MAIN-2025-26)
- **File**: `tokenizer.json`

---

## CHECKPOINT LINEAGE

```
none (random init)
└─ checkpoint-106k.pt (FineWeb Pretraining 0-106k, 106,000 steps)
   ├─ checkpoint-semantic-5000.pt (Semantic Fine-tuning 5k, 5,000 steps)
   ├─ checkpoint_fineweb.pt (FineWeb Continued 106k-110k, 4,000 steps)
   ├─ checkpoint-mixed-1k.pt (Mixed FineWeb+Semantic 1k, 1,000 steps)
   ├─ checkpoint-mixed-2k.pt (Mixed FineWeb+Semantic 2k, 2,000 steps) ★ BEST MIXED V1
   ├─ checkpoint-mixed-3k.pt (Mixed FineWeb+Semantic 3k, 3,000 steps)
   ├─ checkpoint-mixed-4k.pt (Mixed FineWeb+Semantic 4k, 4,000 steps)
   ├─ checkpoint-mixed-5k.pt (Mixed FineWeb+Semantic 5k, 5,000 steps)
   ├─ checkpoint-mixed-6k.pt (Mixed FineWeb+Semantic 6k, 6,000 steps)
   ├─ checkpoint-mixed-7k.pt (Mixed FineWeb+Semantic 7k, 7,000 steps)
   ├─ checkpoint-mixed-8k.pt (Mixed FineWeb+Semantic 8k, 8,000 steps)
   ├─ checkpoint-mixed-10k.pt (Mixed FineWeb+Semantic 10k, 10,000 steps)
   ├─ checkpoint_exp1_latest.pt (Semantic V2 Mixed 5k, 5,000 steps) ★ SEMANTIC V2
   └─ checkpoint_exp2_latest.pt (Semantic V2 EXP-2, 3k cont, 3,000 steps) ★ CURRENT BEST
```

### Checkpoint Details

| Checkpoint | Step | Parent | Training Type | Tokens Processed | Training Loss |
|------------|------|--------|---------------|------------------|---------------|
| checkpoint-106k.pt | 106,000 | random init | FineWeb pretraining | 868,352,000 | 3.5570 |
| checkpoint-semantic-5000.pt | 5,000 | checkpoint-106k.pt | Semantic/instruction fine-tuning | 40,960,000 | 0.0050 |
| checkpoint_fineweb.pt | 110,000 | checkpoint-106k.pt | FineWeb continued | 868,352,000 + 32,768,000 | 0.0020 |
| checkpoint-mixed-1k.pt | 1,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 8,192,000 | 3.186 |
| checkpoint-mixed-2k.pt | 2,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 16,384,000 | 3.547 |
| checkpoint-mixed-3k.pt | 3,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 24,576,000 | 2.416 |
| checkpoint-mixed-4k.pt | 4,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 32,768,000 | 2.653 |
| checkpoint-mixed-5k.pt | 5,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 40,960,000 | 3.533 |
| checkpoint-mixed-6k.pt | 6,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 49,152,000 | 3.247 |
| checkpoint-mixed-7k.pt | 7,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 57,344,000 | 3.713 |
| checkpoint-mixed-9k.pt | 9,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 73,728,000 | NOT EVALUATED |
| checkpoint-mixed-10k.pt | 10,000 | checkpoint-106k.pt | Mixed FineWeb (90%) + Semantic (10%) | 81,920,000 | 3.361 |
| checkpoint_exp1_latest.pt | 5,000 | checkpoint-106k.pt | Semantic V2 Mixed (85/15, masked loss, LR schedule) | 40,960,000 | 2.8435 |
| checkpoint_exp2_latest.pt | 3,000 | checkpoint_exp1_latest.pt | Semantic V2 EXP-2 (90/10, lower LR, arithmetic/transitive boost) | 24,576,000 | 3.4830 |

### Checkpoint Naming Convention
- `checkpoint_fineweb.pt` — Latest FineWeb checkpoint (overwritten each save)
- `checkpoint-{step}k.pt` — Permanent milestone (every 500 steps)
- `checkpoint-{step}k-N.pt` — Duplicate milestones (same step saved again)
- `checkpoint-semantic-{step}.pt` — Semantic fine-tuning archives

---

## TRAINING HISTORY

### Experiment 1: FineWeb Pretraining 0 → 106k steps
- **Dataset**: HuggingFaceFW/fineweb (CC-MAIN-2025-26), streaming
- **Batch size**: 16, **Seq len**: 512, **Tokens/step**: 8,192
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1)
- **Duration**: Multiple sessions over several days
- **Final loss**: 3.5570
- **Throughput**: ~35,000-41,000 tok/s
- **Observation**: Strong grammar acquisition (55.6%), zero factual QA, good entity relation patterns from pretraining

### Experiment 2: Semantic Fine-tuning (5,000 steps from checkpoint-106k.pt)
- **Dataset**: 41 synthetic examples covering identity, sun facts, simple facts, relationships, paraphrases, logic, arithmetic, instruction following, QA, conversation
- **Repeated** to create sufficient training tokens
- **Batch size**: 16, **Seq len**: 512
- **Optimizer**: AdamW (lr=5e-5, weight_decay=0.01) — fresh optimizer, NOT continued from pretraining
- **Final loss**: 0.0050
- **Results**: 
  - ✅ Factual QA: 0% → 100%
  - ✅ Reasoning: 16.7% → 100%
  - ✅ Paraphrase: 33.3% → 66.7%
  - ✅ Entity relations: 100% maintained
  - ❌ Grammar: 55.6% → 33.3% (catastrophic forgetting)
  - ❌ Instruction following: unchanged at 16.7%
- **Observation**: Fine-tuning on tiny synthetic data destroys general language ability while teaching specific facts

### Experiment 3: FineWeb Continued 106k → 110k (4,000 steps)
- **Dataset**: FineWeb streaming (continued)
- **Same hyperparameters as pretraining**
- **Final loss**: 0.0020 (dramatic drop — indicates overfitting/memorization)
- **Results**:
  - Factual QA: 0% → 28.6% (slight improvement from more exposure)
  - Grammar: 55.6% → 44.4% (degraded)
- **Observation**: Continued pretraining alone doesn't teach factual recall effectively at this scale; loss collapse suggests memorization

### Experiment 4: Semantic V2 Mixed Training (5,000 steps from checkpoint-106k.pt)
- **Dataset**: Procedural semantic generator (20k unique examples) + Alpaca-cleaned subset (8k) + FineWeb (85%)
- **Key improvements over v1**:
  - Clean per-row mixture (no token interleaving) — model sees full `User:...Assistant:...` contexts
  - Loss masking on semantic rows — loss only on Assistant spans (not User prompts)
  - LR schedule: 150-step warmup + cosine decay to 10% of peak (1e-4)
  - Diverse procedural data with unseen entities/wording (20k unique examples)
  - Real instruction data (Alpaca) for natural language variety
- **Batch size**: 16, **Seq len**: 512, **Semantic rows**: 15%
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05)
- **Final loss**: 2.8435, FineWeb val loss: 3.72
- **Throughput**: ~35,000-100,000 tok/s
- **Held-out benchmark results**: Cat A strict 66.7%, Cat B topic 57.1%, Composite 0.683
- **Major gains over mixed-2k**: relations 86%→29%, why_context 100%→20%, negation 75%→25%, entity_tracking 80%→20%, instruction 86%→14%
- **Regressions**: sun_facts 17% (was 50%), transitive 50% (was 100%), arithmetic 0%
- **Observation**: Procedural diversity + masked loss + LR schedule yields genuine generalization. Some forgetting of general language (val loss +0.1). Arithmetic and transitive reasoning need more targeted data.

---

## EVALUATION RESULTS

### Old Evaluation Suite (contaminated by prompt-echo bug, same-string overlap)
| Model | Grammar | Factual QA | Entity Rel | Paraphrase | Instruction | Reasoning | OOD Gen | Repetition |
|-------|---------|------------|------------|------------|-------------|-----------|---------|------------|
| checkpoint-106k.pt (pretrain) | 55.6% | 0.0% | 100% | 33.3% | 16.7% | 16.7% | 40% | 42.4% |
| checkpoint-semantic-5000.pt (fine-tune) | 33.3% | **100%** | 100% | 66.7% | 16.7% | **100%** | 100% | 81.5% |
| checkpoint_fineweb.pt (110k) | 44.4% | 28.6% | 100% | 33.3% | 16.7% | 16.7% | 100% | 83.3% |
| **checkpoint-mixed-1k.pt** | **66.7%** | **85.7%** | 100% | **100%** | 16.7% | **100%** | **100%** | 81.5% |
| **checkpoint-mixed-2k.pt** ★ | 55.6% | **100%** | 100% | **100%** | 16.7% | **100%** | 80% | 78.4% |
| checkpoint-mixed-3k.pt | 55.6% | 57.1% | 100% | **100%** | 16.7% | 66.7% | 60% | 80.4% |
| checkpoint-mixed-4k.pt | 55.6% | 71.4% | 100% | **100%** | 16.7% | 83.3% | 60% | 56.7% |
| checkpoint-mixed-5k.pt | 44.4% | 42.9% | 100% | **100%** | 16.7% | 66.7% | 60% | 65.2% |
| checkpoint-mixed-6k.pt | 44.4% | 42.9% | 100% | **100%** | 16.7% | 66.7% | 60% | 63.3% |
| checkpoint-mixed-7k.pt | 44.4% | 71.4% | 100% | **100%** | 16.7% | 50.0% | 60% | 54.3% |
| checkpoint-mixed-8k.pt | 44.4% | 42.9% | 100% | **100%** | 16.7% | 66.7% | 60% | 56.9% |
| checkpoint-mixed-9k.pt | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |
| checkpoint-mixed-10k.pt | 44.4% | 28.6% | 100% | **100%** | 16.7% | 33.3% | 60% | 74.6% |

### New Held-Out Benchmark (continuation-only keyword match, unseen entities/wording)
| Model | Cat A Strict | Cat A Loose | Cat B Topic | Repetition | Composite |
|-------|--------------|-------------|-------------|------------|-----------|
| checkpoint-106k.pt (pretrain) | 23.5% | 23.5% | 57.1% | 41.1% | 0.380 |
| checkpoint-mixed-2k.pt (mixed v1) | 39.2% | 39.2% | 71.4% | 34.7% | 0.514 |
| checkpoint_exp1_latest.pt (semantic v2 EXP-1) | 66.7% | 66.7% | 57.1% | 38.4% | 0.683 |
| **checkpoint_exp2_latest.pt (semantic v2 EXP-2)** ★ | **70.6%** | **70.6%** | **85.7%** | 39.6% | **0.753** |

| Model | sun_facts | relations | state | why_context | negation | entity_track | transitive | arithmetic | instruction | context |
|-------|-----------|-----------|-------|-------------|----------|--------------|------------|------------|-------------|---------|
| checkpoint-106k.pt | 0% | 57% | 67% | 0% | 0% | 40% | 0% | 0% | 14% | 100% |
| checkpoint-mixed-2k.pt | 50% | 29% | 100% | 20% | 25% | 20% | 100% | 0% | 14% | 100% |
| checkpoint_exp1_latest.pt | 17% | 86% | 100% | 100% | 75% | 80% | 50% | 0% | 86% | 100% |
| **checkpoint_exp2_latest.pt** ★ | 17% | **100%** | 100% | **100%** | 75% | 60% | **100%** | 0% | 86% | 100% |

### Key Findings (Updated)
1. **Pure pretraining** learns grammar and entity patterns but NOT facts
2. **Tiny synthetic fine-tuning** teaches facts/reasoning but destroys grammar (catastrophic forgetting)
3. **Continued pretraining** slightly improves facts but degrades grammar and overfits
4. **Mixed-data training v1** (90/10, constant LR, interleaved tokens) prevents catastrophic forgetting but oscillates; peak at 2k steps with fake 100% scores from prompt-echo bug
5. **Held-out benchmark reveals true performance**: mixed-2k true semantic accuracy only 39%
6. **Semantic V2 EXP-1** (85/15, clean rows, masked loss, LR schedule, diverse data) achieves **67% true semantic accuracy** — 32% relative improvement over v1
7. **Major EXP-1 gains**: relations 86% (vs 29%), why_context 100% (vs 20%), negation 75% (vs 25%), entity_tracking 80% (vs 20%), instruction 86% (vs 14%)
8. **EXP-1 regressions**: sun_facts 17% (vs 50%), transitive 50% (vs 100%), arithmetic 0% — need targeted data
9. **EXP-1 FineWeb val loss increased** (3.63 → 3.72) — mild catastrophic forgetting
10. **Semantic V2 EXP-2** (continuation, 90/10, lower LR, more arithmetic/transitive) achieves **70.6% Cat A strict, 85.7% generation topic hit, Composite 0.753**
11. **EXP-2 gains**: relations 100% (was 86%), transitive 100% (was 50%), generation topic_hit 85.7% (was 57%)
12. **EXP-2 remaining gaps**: arithmetic 0%, sun_facts 17%, entity_tracking 60% (regressed from 80%)
13. **EXP-2 FineWeb val loss improved** (3.72 → 3.57) — 90/10 ratio + lower LR reduced forgetting

---

## FAILED EXPERIMENTS

| Experiment | What Was Tried | Why It Failed | What We Learned |
|------------|----------------|---------------|-----------------|
| Fact injection (train_fact_injection.py) | Repeated "sun rises in east" 10,000x on FineWeb checkpoint | Overfitting to single fact, no generalization | Repetition ≠ understanding; need diverse paraphrases |
| Semantic fine-tuning (5k steps) | 41 synthetic examples repeated | Catastrophic forgetting of grammar | Need mixed-data training or replay to preserve pretraining |
| Continued FineWeb to 110k | More of same pretraining | Loss collapsed to 0.002, grammar degraded | Memorization ≠ learning; need curriculum or different data |

---

## SUCCESSFUL EXPERIMENTS

| Experiment | What Worked | Why |
|------------|-------------|-----|
| FineWeb pretraining (106k steps) | Grammar 55.6%, entity relations 100% | Large-scale diverse data learns linguistic patterns |
| Semantic fine-tuning | Factual QA 100%, reasoning 100%, paraphrase 66.7% | Targeted examples teach specific capabilities |
| **Mixed FineWeb+Semantic v1 (2k steps)** | **Grammar 55.6%, Old-suite 100% factual/Reasoning** | **90/10 mixture prevents catastrophic forgetting; oscillates past 2k** |
| Mixed FineWeb+Semantic (1k steps) | Grammar 66.7%, Factual QA 85.7%, Reasoning 100% | Early mixed training shows strong grammar retention |
| **Semantic V2 Mixed (5k steps, EXP-1)** | **Held-out Cat A 66.7%, Composite 0.683** | **Clean rows, masked loss, LR schedule, 28k diverse examples — genuine generalization** |
| **Semantic V2 EXP-2 (3k continuation, 90/10)** | **Held-out Cat A 70.6%, Composite 0.753, Gen 85.7%** | **More FineWeb, lower LR, arithmetic/transitive boost fixed syllogisms & generation relevance** |

## EXPERIMENTAL CONFIRMATION

| Experiment | What Was Tried | Result | What We Learned |
|------------|----------------|--------|-----------------|
| Mixed training 10k steps (constant LR) | Extended mixed training to 10k | Factual QA dropped to 28.6%, reasoning to 33.3% | **Constant LR causes oscillation** — 2k is the peak. Need LR scheduling for longer runs. Paraphrase remains 100% throughout. |

---

## KNOWN LIMITATIONS

1. **Model capacity**: 14.66M params is very small for factual recall and reasoning
2. **Tokenizer**: 16k BPE may not optimally represent all English words
3. **Context window**: 512 tokens limits conversation length
4. **Catastrophic forgetting**: Fine-tuning destroys pretrained abilities
5. **Repetition**: High repetition rates (80%+) in generation, especially after fine-tuning
6. **Instruction following**: Only 16.7% — model doesn't reliably follow format instructions
7. **No validation set**: All evaluation on fixed test sets; no held-out validation during training

---

## CURRENT BEST MODEL

**checkpoint_exp2_latest.pt (Semantic V2 EXP-2)** — Best on held-out benchmark:
- **Composite: 0.753** (vs 0.683 EXP-1, 0.514 mixed-2k, 0.380 106k)
- **Cat A Semantic (strict): 70.6%** (vs 66.7% EXP-1, 39.2% mixed-2k, 23.5% 106k)
  - relations: 100%, why_context: 100%, transitive: 100%, state: 100%, context: 100%
  - instruction: 85.7%, negation: 75%, entity_tracking: 60%
  - sun_facts: 16.7%, arithmetic: 0%
- **Cat B Generation: topic_hit 85.7%**, repetition 39.6%
- FineWeb val loss: 3.57 (minimal forgetting vs 3.56 pretrain)

**Previous: checkpoint_exp1_latest.pt (Semantic V2 EXP-1)**
- Composite: 0.683, Cat A: 66.7%, Cat B: 57.1%
- relations: 85.7%, transitive: 50%, entity_tracking: 80%

**Previous best: checkpoint-mixed-2k.pt** (mixed v1) — Old suite fake 100% scores, true held-out 39.2%

**Other notable checkpoints:**
- `checkpoint-mixed-1k.pt` — Best grammar (66.7%) on old suite
- `checkpoint-semantic-5000.pt` — Best for pure factual QA/reasoning (100% old suite) but grammar destroyed (33.3%)
- `checkpoint-106k.pt` — Best for pure grammar/general language (55.6%) but zero factual knowledge

**Semantic V2 EXP-2 is the first checkpoint with genuine held-out semantic accuracy >70% and generation relevance >85%.**

---

## NEXT EXPERIMENTS (Planned)

1. **Arithmetic specialization**: Dedicated arithmetic curriculum (many variants, small numbers, explicit equals format) — currently 0%
2. **Sun facts / world facts**: More diverse fact templates (sunset/west, capitals, animals, geography) — currently 17%
3. **Entity tracking robustness**: Fix regression from 80% → 60% with more adjective/object tracking examples
4. **Reduce User/Assistant looping**: Add non-dialogue semantic examples (pure continuation) to semantic mix
5. **Higher FineWeb ratio (95/5)**: Test if 90/10 is optimal for preventing forgetting
6. **Curriculum learning**: Stage 1: semantic V2 (3k), Stage 2: arithmetic/facts (2k), Stage 3: FineWeb replay (2k)
7. **Larger model**: Scale to 50M-100M params if hardware allows (gradient accumulation)
8. **LoRA/adapter fine-tuning**: Parameter-efficient tuning to preserve pretrained weights
9. **Perplexity evaluation**: Add held-out FineWeb perplexity tracking
10. **Real conversation data**: Add multi-turn dialogue datasets (e.g., OpenAssistant, ShareGPT)

---

## EXPERIMENT TRACKING

All experiments recorded in:
- `experiments/results.jsonl` — Machine-readable (JSON Lines)
- `experiments/summary.json` — Structured summary
- `experiments/SUMMARY.md` — Human-readable markdown table
- `experiment_tracker.py` — Lineage and recording utilities

Run `python experiment_tracker.py lineage` to see checkpoint tree.

---

## FILES IN PROJECT

| File | Purpose |
|------|---------|
| `src/models/model.py` | SmallEnglishLLM architecture (14.66M params) |
| `src/config/model_config.py` | ModelConfig dataclass |
| `src/paths.py` | Centralized project paths (checkpoints, tokenizer, data) |
| `tokenizer.json` | 16k BPE tokenizer |
| `src/training/train_fineweb.py` | Main FineWeb streaming training (resumable) |
| `src/training/train_fineweb_fast.py` | Semantic/instruction fine-tuning (DEPRECATED) |
| `src/training/train_fact_injection.py` | Targeted fact injection (DEPRECATED) |
| `src/training/train_mixed.py` | Mixed FineWeb + Semantic training v1 (90/10, DEPRECATED) |
| `src/training/train_semantic_v2.py` | **Semantic V2: clean rows, masked loss, LR schedule** |
| `src/evaluation/evaluation_suite.py` | Old evaluation suite (prompt-echo bug, deprecated) |
| `src/evaluation/heldout_benchmark.py` | **Permanent held-out benchmark (continuation-only, unseen entities)** |
| `src/evaluation/evaluate_grammar.py` | Grammar-only evaluation |
| `src/evaluation/test_sun.py` | Quick "sun rises" test |
| `src/evaluation/test_semantic` | Interactive semantic test |
| `src/evaluation/test.py` | Interactive generation |
| `src/utils/semantic_data.py` | **Procedural semantic dataset generator** |
| `src/utils/visualize.py` | Activation visualization |
| `src/utils/benchmark_training.py` | VRAM/throughput benchmark |
| `src/utils/experiment_tracker.py` | Experiment logging & lineage |
| `checkpoints/pretrain/*.pt` | Pretraining checkpoints |
| `checkpoints/mixed/*.pt` | Mixed training v1 checkpoints |
| `checkpoints/semantic_v2/*.pt` | Semantic V2 checkpoints |
| `experiments/` | Evaluation results & experiment logs |

---

## HOW TO RUN

```bash
# Activate environment
.venv/Scripts/activate

# FineWeb training (resumes from checkpoints/pretrain/checkpoint_fineweb.pt)
python src/training/train_fineweb.py

# Semantic V2 mixed training (from checkpoints/pretrain/checkpoint-106k.pt)
python src/training/train_semantic_v2.py --steps 5000 --name exp1
# EXP-2: continuation from exp1 (90/10 FineWeb, lower LR, 3000 steps)
python src/training/train_semantic_v2.py --base checkpoints/semantic_v2/checkpoint_exp1_latest.pt --steps 3000 --lr 5e-5 --semantic-ratio 0.10 --name exp2
# Smoke test:
python src/training/train_semantic_v2.py --smoke

# Held-out benchmark (NEW - use this for evaluation)
python src/evaluation/heldout_benchmark.py checkpoints/pretrain/checkpoint-106k.pt --output experiments/bench_106k.json
python src/evaluation/heldout_benchmark.py checkpoints/mixed/checkpoint-mixed-2k.pt --output experiments/bench_mixed2k.json
python src/evaluation/heldout_benchmark.py checkpoints/semantic_v2/checkpoint_exp1_latest.pt --output experiments/bench_exp1.json
python src/evaluation/heldout_benchmark.py checkpoints/semantic_v2/checkpoint_exp2_latest.pt --output experiments/bench_exp2.json

# Old evaluation suite (deprecated - has prompt-echo bug)
python src/evaluation/evaluation_suite.py checkpoints/pretrain/checkpoint-106k.pt --output experiments/eval_106k.json
python src/evaluation/evaluation_suite.py checkpoints/mixed/checkpoint-mixed-2k.pt --output experiments/eval_mixed_2k.json

# Quick tests
python src/evaluation/test_sun.py
python src/evaluation/test_semantic

# Grammar only
python src/evaluation/evaluate_grammar.py

# Interactive
python src/evaluation/test.py

# View lineage
python src/utils/experiment_tracker.py lineage
```

---

## PROJECT PRINCIPLES (from mission)

1. **Never declare success from memorization** — must generalize to unseen wording
2. **Preserve best checkpoints** — never overwrite the only copy of a working model
3. **Document everything** — failures as thoroughly as successes
4. **Measure generalization** — paraphrase, OOD, held-out tests
5. **Prevent catastrophic forgetting** — mixed data, replay, lower LR
6. **Autonomous loop** — Train → Evaluate → Analyze → Improve → Train
7. **README never lies** — UNKNOWN or NOT MEASURED for unknowns

---

*Last updated: August 12, 2026 (Semantic V2 EXP-2 completed: held-out Cat A 70.6%, Composite 0.753, generation 85.7%; relations/transitive 100%, why_context 100%, negation 75%; arithmetic 0%, sun_facts 17%, entity_tracking 60% remain gaps)*
*Document created during active experimentation — will be updated continuously*