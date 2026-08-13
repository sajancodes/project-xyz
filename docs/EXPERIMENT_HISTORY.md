# EXPERIMENT HISTORY

Complete record of every experiment, with measured numbers only. Source: `experiments/results.jsonl`, `experiments/summary.json`, `experiments/bench_*.json`, `experiments/SUMMARY.md`, checkpoint metadata, and train logs.

## Master table

| Exp | Name | Parent | Steps | Tokens | LR | Train Loss | Val Loss | Checkpoint |
|-----|------|--------|-------|--------|----|------------|----------|------------|
| 0001 | FineWeb Pretraining 0–106k | none (random init) | 106,000 | 868,352,000 | 3e-4 | 3.5570 | N/A | checkpoints/pretrain/checkpoint-106k.pt |
| 0003 | Semantic Fine-tuning 5k | checkpoint-106k.pt | 5,000 | 40,960,000 | 5e-5 | 0.0050 | N/A | checkpoints/semantic/checkpoint-semantic-5000.pt |
| 0004 | FineWeb Continued 106k→110k | checkpoint-106k.pt | 4,000 | 32,768,000 | 3e-4 | 0.0020 | N/A | checkpoints/pretrain/checkpoint_fineweb.pt |
| 0004b | Mixed 1k | checkpoint-106k.pt | 1,000 | 8,192,000 | 1e-4 | 3.1860 | N/A | checkpoints/mixed/checkpoint-mixed-1k.pt |
| 0005 | Mixed 2k ★ | checkpoint-106k.pt | 2,000 | 16,384,000 | 1e-4 | 3.5470 | N/A | checkpoints/mixed/checkpoint-mixed-2k.pt |
| 0006 | Mixed 3k | checkpoint-106k.pt | 3,000 | 24,576,000 | 1e-4 | 2.4160 | N/A | checkpoints/mixed/checkpoint-mixed-3k.pt |
| 0007 | Mixed 4k | checkpoint-106k.pt | 4,000 | 32,768,000 | 1e-4 | 2.6530 | N/A | checkpoints/mixed/checkpoint-mixed-4k.pt |
| 0008 | Mixed 5k | checkpoint-106k.pt | 5,000 | 40,960,000 | 1e-4 | 3.5330 | N/A | checkpoints/mixed/checkpoint-mixed-5k.pt |
| 0009 | Mixed 6k | checkpoint-106k.pt | 6,000 | 49,152,000 | 1e-4 | 3.2470 | N/A | checkpoints/mixed/checkpoint-mixed-6k.pt |
| 0010 | Mixed 7k | checkpoint-106k.pt | 7,000 | 57,344,000 | 1e-4 | 3.7130 | N/A | checkpoints/mixed/checkpoint-mixed-7k.pt |
| 0011 | Mixed 8k | checkpoint-106k.pt | 8,000 | 65,536,000 | 1e-4 | 3.3480 | N/A | checkpoints/mixed/checkpoint-mixed-8k.pt |
| 0012 | Mixed 10k | checkpoint-106k.pt | 10,000 | 81,920,000 | 1e-4 | 3.3610 | N/A | checkpoints/mixed/checkpoint-mixed-10k.pt |
| 0013 | Semantic V2 (exp1, semv2) | checkpoint-106k.pt (106000) | 5,000 | 40,960,000 | 1e-4 | 2.8435 | 3.7246 | checkpoints/semantic_v2/checkpoint_exp1_*.pt |
| 0014 | Semantic V2 EXP-2 (semv2b) | checkpoint_exp1_latest (5000) | 3,000 | 24,576,000 | 5e-5 | 3.4830 | 3.5742 | checkpoints/semantic_v2/checkpoint_exp2_*.pt |
| — | SEMV2C continuation | semv2b-3000 | 3,000 | — | — | — | — | checkpoints/best/checkpoint_semv2c-3000.pt |
| — | SEMV2D | (designed) | — | — | — | — | — | **NOT MEASURED** |
| — | SEMV2E | checkpoint_best.pt | 5,000 | 40,960,000 | 1e-4 | 0.2976 | 4.3915 | checkpoints/semv2e/checkpoint_semv2e-{500..5000}.pt |

## Per-experiment writeups

### Exp 0001 — FineWeb Pretraining
- Data: FineWeb (CC-MAIN-2025-26) streaming; ~868M tokens; AdamW lr 3e-4, batch 16, seq 512.
- Result: grammar 55.6%, factual QA 0%, entity rel 100%, paraphrase 33.3%, instruction 16.7%, reasoning 16.7%, OOD 40%, repetition 42.4%.
- Lesson: pretraining learns grammar/entity patterns but no facts.

### Exp 0003 — Semantic Fine-tuning 5k
- Data: 41 synthetic examples repeated; fresh AdamW lr 5e-5.
- Result: factual QA 0→100%, reasoning →100%, paraphrase →66.7%, relations held 100%, **grammar 55.6→33.3%**, instruction unchanged 16.7%, repetition 81.5%.
- Lesson: tiny synthetic tuning destroys general language ability (catastrophic forgetting).

### Exp 0004 — FineWeb Continued
- 4,000 more FineWeb steps, same hyperparameters.
- Result: factual QA 0→28.6%, grammar 55.6→44.4%, loss collapsed 0.0020 (memorization).
- Lesson: more pretraining alone doesn't teach facts.

### Exp 0004b–0012 — Mixed FineWeb+Semantic (constant LR)
- 90% FineWeb + 10% synthetic, lr 1e-4 constant.
- **mixed-2k**: grammar 55.6%, factual QA 100%, reasoning 100%, paraphrase 100%, composite 0.514 — best of era.
- Oscillation beyond 2k; by 10k factual QA 28.6%, reasoning 33.3%. Paraphrase stays 100% throughout.
- Lesson: mixture prevents forgetting; constant LR + longer runs oscillate.

### Exp 0013 — Semantic V2 (exp1)
- Procedural + Alpaca-cleaned data, assistant-loss masking, warmup+cosine LR, FineWeb val monitor.
- Result: composite 0.683 (catA strict 67%, catB topic 57%), relations 86%, why_context 100%, instruction 86%, sun_facts 17%, arithmetic 0%. Val 3.7246 vs train 2.8435 → mild overfit.
- Milestones 1k→5k: composite 0.662 / 0.663 / 0.697 (peak) / 0.677 / 0.669.

### Exp 0014 — Semantic V2 EXP-2 (semv2b)
- 3k continuation of exp1-latest; 90/10 FineWeb; lr 5e-5; extra arithmetic + transitive.
- Result: semv2b-3000 composite 0.732, catA strict 71%, catB topic 86%; relations 100%, transitive 100%, entity_tracking 100%; sun_facts 17%, arithmetic 0%. Val 3.5742 vs train 3.4830 → reduced overfit.
- semv2b-5000 regressed to 0.676 (indicates overtraining past 3k).

### SEMV2C
- Continuation targeting weak categories (sun, entity tracking, instruction).
- semv2c-2000: composite 0.751; catA strict 76%; **arithmetic 33% (the only nonzero arithmetic ever)**; but context_retention 0%.
- **semv2c-3000: composite 0.780** (best); catA strict 75%; relations 100%, state 100%, transitive 100%, instruction 86%, why 80%, entity 80%, negation 75%, sun 50%, **arithmetic 0%**. Copied to `checkpoints/best/checkpoint_best.pt`.

### SEMV2D (status: NOT MEASURED)
- `src/training/train_semantic_v2d.py` defines a 50/20/15/15 mixture (semantic/arithmetic/sun/FineWeb).
- **No verified training run or evaluated checkpoint exists.** Previously the README called SEMV2D "Mission complete" — that claim is corrected here: **a script is not a result**. SEMV2D is BLOCKED / NOT MEASURED.

### Real generation test (checkpoint_best.pt)
- 15 ad-hoc prompts, chat wrapper ON, temp 0.3, top-k 40, max 40 tokens, CPU.
- Score 5/15 (33%) — full raw outputs in README.

### Canonical greedy evaluation (checkpoint_best.pt)
- Composite **0.7463**; catA strict 72.5% (n=51); sun 33.3%, relations/state/transitive/context 100%, why 80%, negation 75%, entity_tracking 80%, instruction 86%, arithmetic 0%. CatB topic 71%, rep 0.374, drift 0%. Plain-QA 62.5%. Sun diag 34.6% (rise_east 62.5%, set_west 12.5%). Arith diag 0%. Saved: `experiments/semv2e_bench_baseline_greedy.json`.

### SEMV2E binding benchmark baseline (checkpoint_best.pt)
- 600 prompts, greedy: **chat 43.2% (259/600)**, plain 25.8% (155/600). Details in README and `experiments/semv2e_binding_baseline.json`.

### SEMV2E training (COMPLETED)
- Base checkpoint_best.pt; 5000 steps; mixture 55/15/15/10/5; final loss 0.2976; val 4.3915; 40.96M tokens; ~963 s GPU.
- Milestones saved: semv2e-500 (loss 2.2549), 1000 (1.3516), 2000 (0.8392), 3000 (0.5312), 5000 (0.2976), plus latest + smoke1/smoke2.
- **Evaluation of milestones: PENDING** (next step).

## How experiments are tracked

- `src/utils/experiment_tracker.py` — record/lineage utilities.
- `experiments/results.jsonl` — append-only JSON lines.
- `experiments/summary.json` — structured summary (14 experiments as of last update).
- `experiments/SUMMARY.md` — auto-generated table.

---

*Related: [docs/PROJECT_HISTORY.md](PROJECT_HISTORY.md), [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md).*