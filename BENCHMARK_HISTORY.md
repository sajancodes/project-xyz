# PROJECT XYZ — Benchmark History

All evaluation/benchmark results across checkpoints. Sources: `experiments/eval_*.json` (canonical eval suite), `experiments/bench_*.json` (heldout benchmark, 51 category-A prompts + 7 generation, composite weighting catA_strict 0.55 / catA_loose 0.15 / catB_topic 0.15 / catB_no_dialog_drift 0.10 / catB_no_repetition 0.05), `semv2e_*` (600-prompt binding).

## 1. Classic Evaluation Suite (Exp 0001–0012 era)

| Model | Grammar | Factual QA | Entity Rel | Paraphrase | Instr | Reasoning | OOD Gen | Repeats |
|-------|---------|------------|------------|------------|-------|-----------|---------|---------|
| 106k pretrain | 55.6% | 0.0% | 100% | 33.3% | 16.7% | 16.7% | 40% | 42.4% |
| semantic-5000 | 33.3% | 100% | 100% | 66.7% | 16.7% | 100% | 100% | 81.5% |
| fineweb 110k | 44.4% | 28.6% | 100% | 33.3% | 16.7% | 16.7% | 100% | 83.3% |
| mixed-1k | 66.7% | 85.7% | 100% | 100% | 16.7% | 100% | 100% | 81.5% |
| **mixed-2k** ★ | 55.6% | **100%** | 100% | 100% | 16.7% | 100% | 80% | 78.4% |
| mixed-3k | 55.6% | 57.1% | 100% | 100% | 16.7% | 66.7% | 60% | 80.4% |
| mixed-4k | 55.6% | 71.4% | 100% | 100% | 16.7% | 83.3% | 60% | 56.7% |
| mixed-5k | 44.4% | 42.9% | 100% | 100% | 16.7% | 66.7% | 60% | 65.2% |
| mixed-6k | 44.4% | 42.9% | 100% | 100% | 16.7% | 66.7% | 60% | 63.3% |
| mixed-7k | 44.4% | 71.4% | 100% | 100% | 16.7% | 50.0% | 60% | 54.3% |
| mixed-8k | 44.4% | 42.9% | 100% | 100% | 16.7% | 66.7% | 60% | 56.9% |
| mixed-10k | 44.4% | 28.6% | 100% | 100% | 16.7% | 33.3% | 60% | 74.6% |

Takeaways: 2k is the sweet spot; longer constant-LR mixed training oscillates downward. Paraphrase stays 100% from 1k onward.

## 2. Heldout Benchmark — Composite (all bench_*.json)

| Checkpoint | Step | Composite | catA strict | catA loose | N | catB topic | rep avg |
|------------|------|-----------|-------------|------------|---|------------|---------|
| 106k | 106000 | 0.3799 | 24% | 24% | 51 | 57% | 0.411 |
| mixed-2k | 2000 | 0.5143 | 39% | 39% | 51 | 71% | 0.347 |
| exp1 (semv2-5000) | 5000 | 0.6832 | 67% | 67% | 51 | 57% | 0.384 |
| semv2-1000 | 1000 | 0.6621 | 57% | 57% | 51 | 86% | 0.290 |
| semv2-2000 | 2000 | 0.6746 | 65% | 65% | 51 | 57% | 0.281 |
| semv2-3000 | 3000 | 0.6965 | 65% | 67% | 51 | 71% | 0.331 |
| semv2-4000 | 4000 | 0.6769 | 63% | 63% | 51 | 71% | 0.389 |
| semv2-5000 | 5000 | 0.6691 | 65% | 65% | 51 | 57% | 0.391 |
| semv2b-3000 | 3000 | 0.7322 | 71% | 71% | 51 | 71% | 0.382 |
| semv2b-5000 | 5000 | 0.6758 | 69% | 69% | 51 | 43% | 0.378 |
| exp2 (semv2b-3000) | 3000 | 0.7529 | 71% | 71% | 51 | 86% | 0.396 |
| semv2c-2000 | 2000 | 0.7505 | 76% | 76% | 51 | 57% | 0.410 |
| **semv2c-3000** | 3000 | **0.7800** | 75% | 75% | 51 | 86% | 0.404 |
| best (semv2c-3000, verify) | 3000 | 0.7799 | 75% | 75% | 51 | 86% | 0.404 |
| best (plain) | 3000 | 0.7529 | 71% | 71% | 51 | 86% | 0.396 |

## 3. Heldout Benchmark — Per Category (strict %)

| Checkpoint | sun | rela | state | why | neg | ent | trans | arith | instr | ctx |
|------------|-----|------|-------|-----|-----|-----|-------|-------|-------|-----|
| 106k | 0 | 57 | 67 | 0 | 0 | 40 | 0 | 0 | 14 | 100 |
| mixed-2k | 50 | 29 | 100 | 20 | 25 | 20 | 100 | 0 | 14 | 100 |
| exp1/semv2-5000 | 17 | 86 | 100 | 100 | 75 | 80 | 50 | 0 | 86 | 100 |
| semv2-1000 | 0 | 86 | 100 | 80 | 75 | 60 | 0 | 0 | 86 | 100 |
| semv2-2000 | 0 | 86 | 100 | 100 | 75 | 80 | 50 | 0 | 86 | 100 |
| semv2-3000 | 0 | 86 | 100 | 100 | 75 | 80 | 50 | 0 | 86 | 100 |
| semv2-4000 | 17 | 86 | 100 | 80 | 75 | 80 | 25 | 0 | 86 | 100 |
| semv2-5000 | 0 | 86 | 100 | 100 | 75 | 80 | 50 | 0 | 86 | 100 |
| semv2b-3000 | 17 | 100 | 100 | 80 | 75 | 100 | 75 | 0 | 86 | 100 |
| semv2b-5000 | 17 | 100 | 100 | 80 | 75 | 80 | 100 | 0 | 71 | 100 |
| semv2c-2000 | 33 | 100 | 100 | 80 | 75 | 100 | 100 | **33** | 86 | 0 |
| **semv2c-3000** | 50 | 100 | 100 | 80 | 75 | 80 | 100 | 0 | 86 | 100 |

Consistent pattern: relations/state/transitive/instruction/context high; **arithmetic 0%** everywhere; sun_facts weak (esp. west); entity_tracking 60–80%.

## 4. Diagnostic Results (checkpoint_best.pt, 2026-08-13)

### Sun Diagnostic (n=26)
- rise_east: 62.5% (5/8) · set_west: 12.5% (1/8) · rise_paraphrase: 40% (2/5) · set_paraphrase: 20% (1/5) → overall **34.6%**.
- Model knows "sun rises east" but not "sun sets west"; heavily overfits phrase templates.

### Arithmetic Diagnostic
- Addition (n=50): 0% · Subtraction (n=40): 0% · Multi-step word problems (n=30): 0%. No arithmetic capability at any scale tested.

### Plain-QA Diagnostic (8 prompts, no chat wrapper): 62.5% — drops notably without the chat template.

## 5. SEMV2E Binding Benchmark (600 prompts, greedy, checkpoint_best.pt)

**Chat-wrapped overall: 43.2% (259/600)** · **Plain: 25.8% (155/600)**

| Category | Chat | Plain |
|----------|------|-------|
| A_single_fact | 53% | 32% |
| B_multi_fact | 37% | 23% |
| C_distractor | 25% | 8% |
| D_relation_reversal | 75% | 0% |
| E_paraphrase | 58% | 45% |
| F_pronoun | 30% | 45% |
| G_multi_hop | 52% | 22% |
| H_sentence_order | 37% | 33% |
| I_entity_generalization | 33% | 17% |
| J_distractor_binding | 32% | 33% |

Takeaways: chat wrapper matters hugely (D_relation_reversal 75% vs 0% plain). Distractor/pronoun/binding categories are weakest — the binding generalization gap.

## 6. Real Test — Live Chat Generations (checkpoint_best.pt)

**Score: 5/15 (33%)** — 15 ad-hoc prompts, temp 0.3, top-k 40, chat wrapper on.
- PASS: single-fact property (red book, green bicycle), sun rises east, forward relations (reverse relation Sajan, Jenny-barks).
- FAIL: multi-fact reasoning/disambiguation, paraphrase (color wrong), sunset/west, negation inference, longer-context retention, instruction-content.
- Morale: benchmark composite (0.78) flatters the model; ad-hoc prompts expose weak entity_tracking/sun_facts/instruction phrasing sensitivity.

## Known Caveats
- Bench files include `_verify` duplicates with identical scores (deterministic greedy decoding); `bench_best_plain` is the same checkpoint WITHOUT chat wrapper (composite 0.7529).
- `eval_mixed2k_fixed.json` labels `checkpoints/best/checkpoint_best.pt` as "step 2000, mixed_fineweb_semantic" but the real best checkpoint metadata says step 3000 `semantic_v2_mixed` — naming at `checkpoints/best/` is not always the true lineage; trust `checkpoint_info` in each file.