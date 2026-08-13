# PROJECT XYZ — Project Summary

Last updated: 2026-08-13

## What This Is

Training a small decoder-only English LLM (**14,657,664 params / 14.66M**) from scratch on a laptop GPU (RTX 4050, 6 GB VRAM). The goal is not just fluent text, but **demonstrable semantic understanding** — answer questions, follow instructions, and generalize to unseen wording (paraphrase, OOD, held-out).

## Current State (as of 2026-08-13)

- **Best production checkpoint**: `checkpoints/best/checkpoint_best.pt` (step 3000, `semantic_v2_mixed`, train loss 3.571, composite benchmark ~0.75–0.78).
- **Best real-generation test**: **5/15 (33%)** on a 15-prompt live chat test — single-fact object-property retrieval and simple forward relations work; multi-fact reasoning, paraphrase, sunset/west, and negation inference fail.
- **Verified benchmark (semv2c-3000, canonical):** composite 0.780; catA strict 75%, catB topic 86%, repetition-avg 0.404. Per-category: relations 100%, state 100%, transitive 100%, instruction 86%, why_context 80%, entity_tracking 80%, negation 75%, sun_facts 50%, arithmetic 0%.
- **SEMV2E binding baseline** (`checkpoints/best/checkpoint_best.pt`, greedy, 600 prompts): chat-wrapped 43.2% overall; single-fact 53%, relation-reversal 75%, multi-hop 52%; distractor 25%, pronoun 30%, entity-generalization 33%, distractor-binding 32%. Plain (no wrapper) 25.8%.
- **Known weaknesses**: arithmetic is 0% across every protocol; sunset (west) ~12.5% vs rise (east) ~62.5%; entity multi-fact binding is fragile.

## Hardware / Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128, Windows 11.
- Training throughput on FineWeb: ~35,000–41,000 tok/s.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Vocabulary | 16,000 |
| Context length | 512 |
| d_model | 256 |
| Layers | 8 |
| Attention heads | 8 |
| d_ff | 1,024 |
| **Params** | **14,657,664** |

Decoder-only transformer, pre-LN, causal self-attention, GELU, learned positional embeddings.

## Tokenizer

16k BPE (HuggingFace `tokenizers`), ByteLevel pre-tokenizer + decoder, special tokens `<pad>`=0, unk=1, `<bos>`=2, `<eos>`=3. Trained on 50,000 FineWeb documents (CC-MAIN-2025-26). File: `tokenizer.json`.

## Training Data Sources

1. **FineWeb** (HuggingFaceFW/fineweb, CC-MAIN-2025-26) — streaming pretraining + the 90% share in mixed runs.
2. **Synthetic semantic/instruction corpus** — grew from 41 examples (exp 1) → procedural + Alpaca (Semantic V2 family) → SEMV2E (600-problem binding benchmark).
3. Data is **gitignored** (`data/`, `*.json`); only `tokenizer.json` is tracked.

## Checkpoint Lineage (files on disk)

```
none (random init)
└─ checkpoints/pretrain/checkpoint-106k.pt        (106k steps, loss 3.557)
   ├─ checkpoints/mixed/checkpoint-mixed-2k.pt     (mixed 90/10, best-of-era)
   ├─ checkpoints/semantic_v2/*                    (exp1/semv2/semv2b/semv2c family)
   │    └─ checkpoint_semv2c-3000.pt → best         (composite 0.780)
   ├─ checkpoints/best/checkpoint_best.pt          (copy of semv2c-3000; 5/15 real test)
   │    └─ checkpoint_semv2b-3000.pt, checkpoint_semv2c-3000.pt (also kept)
   └─ checkpoints/semv2e/*                         (semv2e + smoke test runs)
```

Note: `checkpoints/` is **gitignored** — weights are not version-controlled (only `*.pt` locally). Keep backups.

## Key Findings (see TRAINING_HISTORY.md for detail)

1. Pure pretraining learns grammar + entity patterns, not facts.
2. Tiny synthetic fine-tuning teaches facts/reasoning but destroys grammar (catastrophic forgetting).
3. Mixed 90/10 FineWeb+semantic prevents forgetting; **2k steps was the earlier peak**, longer constant-LR runs oscillate/decay.
4. Semantic V2 (procedural+Alpaca, masked loss, LR schedule) beat the old mixed runs on the composite benchmark (0.68→0.78).
5. **Arithmetic never learns** at this scale — every protocol scores 0%.
6. Instruction-following and single-fact relations are strong; multi-fact binding, paraphrase beyond training templates, and negation inference remain weak.

## Project Principles

1. Never declare success from memorization — must generalize to unseen wording.
2. Preserve best checkpoints — never overwrite the only copy of a working model.
3. Document everything — failures as thoroughly as successes.
4. Measure generalization — paraphrase, OOD, held-out tests.
5. Prevent catastrophic forgetting — mixed data, replay, lower LR.
6. Autonomous loop — Train → Evaluate → Analyze → Improve → Train.
7. README never lies — UNKNOWN / NOT MEASURED for unknowns.

## Documentation Map

| File | Purpose |
|------|---------|
| `README.md` | Master doc: overview, lineage, exp history, eval tables, real-test log |
| `PROJECT_SUMMARY.md` | This file: compact state of the project |
| `TRAINING_HISTORY.md` | Chronological log of every training experiment |
| `BENCHMARK_HISTORY.md` | All evaluation/benchmark runs with scores |
| `experiments/summary.json` | Machine-readable structured summary (14 exp) |
| `experiments/results.jsonl` | JSON-lines log (14 rows) |
| `experiments/SUMMARY.md` | Auto-generated markdown table |

## How to Run

```bash
.venv/Scripts/activate
python train_fineweb_fast.py        # FineWeb pretraining (resumable)
python train_semantic_v2.py         # Semantic V2 mixed (procedural+Alpaca)
python train_semv2e.py              # SEMV2E binding training
python heldout_benchmark.py <ckpt> --output experiments/bench_<name>.json
python canonical_eval.py <ckpt> --output experiments/eval_<name>.json
python semv2e_benchmark.py <ckpt> --output experiments/semv2e_<name>.json
python test.py                      # interactive
```
