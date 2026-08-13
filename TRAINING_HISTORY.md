# PROJECT XYZ — Training History

Chronological log of every training experiment. Metrics are taken verbatim from `experiments/summary.json`, `experiments/results.jsonl`, and checkpoint metadata files.

| Exp | Name | Parent | Steps | Tokens | LR | Train Loss | Val Loss | Note |
|-----|------|--------|-------|--------|----|------------|----------|------|
| 0001 | FineWeb Pretraining 0-106k | random init | 106,000 | 868,352,000 | 3e-4 | 3.5570 | constant LR, AdamW |
| 0003 | Semantic Fine-tuning 5k | checkpoint-106k.pt | 5,000 | 40,960,000 | 5e-5 | 0.0050 | 41 synth examples repeated |
| 0004 | FineWeb Continued 106k→110k | checkpoint-106k.pt | 4,000 | 32,768,000 | 3e-4 | 0.0020 | loss collapse |
| 0004b | Mixed FineWeb+Semantic 1k | checkpoint-106k.pt | 1,000 | 8,192,000 | 1e-4 | 3.1860 | 90/10 |
| 0005 | Mixed FineWeb+Semantic 2k | checkpoint-106k.pt | 2,000 | 16,384,000 | 1e-4 | 3.5470 | 90/10 ★ best of era |
| 0006 | Mixed FineWeb+Semantic 3k | checkpoint-106k.pt | 3,000 | 24,576,000 | 1e-4 | 2.4160 | |
| 0007 | Mixed FineWeb+Semantic 4k | checkpoint-106k.pt | 4,000 | 32,768,000 | 1e-4 | 2.6530 | |
| 0008 | Mixed FineWeb+Semantic 5k | checkpoint-106k.pt | 5,000 | 40,960,000 | 1e-4 | 3.5330 | |
| 0009 | Mixed FineWeb+Semantic 6k | checkpoint-106k.pt | 6,000 | 49,152,000 | 1e-4 | 3.2470 | |
| 0010 | Mixed FineWeb+Semantic 7k | checkpoint-106k.pt | 7,000 | 57,344,000 | 1e-4 | 3.7130 | |
| 0011 | Mixed FineWeb+Semantic 8k | checkpoint-106k.pt | 8,000 | 65,536,000 | 1e-4 | 3.3480 | |
| 0012 | Mixed FineWeb+Semantic 10k | checkpoint-106k.pt | 10,000 | 81,920,000 | 1e-4 | 3.3610 | long-run oscillation |
| 0013 | Semantic V2 Mixed (exp1) | checkpoint-106k.pt (106000) | 5,000 | 40,960,000 | 1e-4 | 2.8435 | 3.7246 | procedural+Alpaca, masked loss, LR schedule |
| 0014 | Semantic V2 EXP-2 (semv2b) | checkpoint_exp1_latest (5000) | 3,000 | 24,576,000 | 5e-5 | 3.4830 | 3.5742 | 90/10, more FineWeb, arithmetic+transitive boost |
| —    | SEMV2C continuation | semv2b-3000 | 3,000 | — | — | — | — | best composite 0.780 |
| —    | SEMV2E | checkpoint_best.pt | — | — | — | — | — | binding-focused training; smoke tests in checkpoints/semv2e |

Per-session config: batch 16, seq len 512, AdamW, batch=16, 8,192 tokens/step.

## Experiment Notes

### Exp 0001 — FineWeb Pretraining (0→106k)
- FineWeb streaming, resumable, several sessions over days. Throughput ~35–41k tok/s.
- **Result**: grammar 55.6%, factual QA 0%, entity relations 100%, paraphrase 33.3%, instruction 16.7%, reasoning 16.7%, OOD 40%, repetition 42.4%.
- Lesson: pretraining learns grammar/entity patterns but no facts.

### Exp 0003 — Semantic Fine-tuning (5k)
- 41 synthetic examples (identity, sun facts, simple facts, relations, paraphrases, logic, arithmetic, instruction, QA, conversation), repeated; fresh AdamW at lr 5e-5.
- **Result**: factual QA 0→100%, reasoning 16.7→100%, paraphrase 33.3→66.7%, relations 100% held; **grammar 55.6→33.3% (forgetting)**, instruction unchanged 16.7%, repetition 81.5%.
- Lesson: tiny synthetic tuning destroys general language ability while teaching specific facts.

### Exp 0004 — FineWeb Continued (106k→110k)
- Continued pretraining, same hyperparameters.
- **Result**: factual QA 0→28.6%, grammar 55.6→44.4%, loss collapsed 0.002 → memorization.
- Lesson: continued pretraining alone doesn't teach factual recall at this scale.

### Exp 0004b–0012 — Mixed FineWeb+Semantic (90/10)
- Mixed batches (90% FineWeb + 10% synthetic) at lr 1e-4, starting from 106k. Permanent milestones every 1k steps.
- **Result**: mixed training prevents catastrophic forgetting. Peak at **2k**: factual QA 100%, reasoning 100%, paraphrase 100%, grammar 55.6%. Beyond 2k scores oscillate; by 10k factual QA 28.6%, reasoning 33.3%. Paraphrase stays 100% throughout.
- Lesson: **constant LR + longer runs cause oscillation**; 2k was the peak. Need LR scheduling or ratio adjustment.

### Exp 0013 — Semantic V2 (exp1, semv2 family)
- Procedural + Alpaca-style synthetic data, **masked loss**, LR schedule, from 106k.
- **Result**: composite 68.3; catA strict 66.7, catB topic 57.1; relations 85.7, why_context 100%, instruction 85.7; sun_facts 16.7, arithmetic 0%. Val loss 3.7246 (train 2.8435) → mild overfit.
- Milestones kept: semv2-1000…5000. Benchmark composite rises 0.662→0.697 (peak ~3000) then declines.

### Exp 0014 — Semantic V2 EXP-2 (semv2b)
- 3k continuation of exp1-latest, 90/10 FineWeb, lr 5e-5, extra arithmetic + transitive data.
- **Result**: composite 75.3; catA strict 70.6, catB topic 85.7; relations 100%, transitive 100%, entity_tracking 60%; sun_facts 16.7, arithmetic 0%. Val 3.5742 vs train 3.483 → reduced overfit.

### SEMV2C
- Continuation that produced `checkpoint_semv2c-2000` (composite 0.7505, arithmetic 33% — only nonzero arithmetic result) and `checkpoint_semv2c-3000` (composite **0.780**, the best; arithmetic 0%). `checkpoint_semv2c-3000` is copied as `checkpoints/best/checkpoint_best.pt`.

### SEMV2E
- Binding-focused protocol (600-problem benchmark, 10 categories). Baseline on `checkpoint_best.pt`: chat 43.2% / plain 25.8%. Smoke test checkpoints (`smoke1`, `smoke2`) live in `checkpoints/semv2e/`.

## Known Data Point — Step Anomaly
`checkpoints/pretrain/checkpoint-10k.pt` records step **10500** (not 10k) — saved during an interrupted/continued session. 20k/50k milestones are also from the pretraining lineage.
