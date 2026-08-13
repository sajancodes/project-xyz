# PROJECT XYZ — Project History (Full Narrative)

This is the chronological story of Project XYZ, written so that a reader with no background can follow the whole arc: from a random-initialized toy model to a 14.66M-parameter model that demonstrably binds simple semantic facts.

> **Honesty rule**: every number below is a recorded measurement from the repository (`experiments/*.json`, `checkpoints/*` metadata, logs). When a number is NOT measured, it is written as `NOT MEASURED`. No results are invented.

---

## Stage 0 — The idea (random init)

- **Starting point**: nothing. The model begins with random weights.
- **What "random init" means**: before any training, the transformer produces gibberish. Language ability must be *learned* from data.
- **Goal set at this stage**: a small model (10–20M params target) that demonstrates measurable semantic understanding — not just fluent text.

## Stage 1 — Tokenizer and architecture

- Built a **16,000-token BPE tokenizer** (byte-level, prefix space) on 50,000 FineWeb documents (CC-MAIN-2025-26).
- Defined the architecture: decoder-only Transformer, **14,657,664 params**, 8 layers, d_model 256, 8 heads, d_ff 1024, context 512, GELU, pre-LN, learned positional embeddings.
- Files: `src/config/model_config.py`, `src/models/model.py`, `tokenizer.json`.

## Stage 2 — FineWeb pretraining (0 → 106k steps)

- **Data**: FineWeb streaming (868,352,000 tokens), AdamW lr 3e-4, batch 16, seq 512.
- **Result**: loss 3.5570. Grammar **55.6%**, entity relations **100%**, but **factual QA 0%**, paraphrase 33.3%, instruction 16.7%, reasoning 16.7%.
- **Lesson**: next-token prediction teaches structure (grammar, common patterns) but not question answering. This is the "next-token ≠ understanding" problem that the whole project is about.

## Stage 3 — Semantic fine-tuning (5k steps) — the forgetting warning

- **Data**: 41 synthetic examples (identity, sun facts, relations, paraphrases, logic, arithmetic, instruction, QA, conversation), repeated; fresh AdamW lr 5e-5.
- **Result**: factual QA 0→**100%**, reasoning →**100%**, paraphrase →66.7%; but **grammar 55.6→33.3%**.
- **Lesson**: **catastrophic forgetting**. Tiny synthetic fine-tuning teaches specific facts while destroying general language ability. This is the single most important early lesson.

## Stage 4 — Fact injection (DEPRECATED)

- **What**: repeated "sun rises in east" 10,000×.
- **Result**: no generalization — factual QA 0% on unseen phrasings.
- **Lesson**: repetition ≠ understanding. A model must see *varied* wording of the same fact, not one string a thousand times.

## Stage 5 — FineWeb continued (106k → 110k)

- **What**: 4,000 more FineWeb steps.
- **Result**: loss collapsed to 0.0020 (memorization of recent stream); factual QA 0→28.6%, grammar 55.6→44.4%.
- **Lesson**: more of the same pretraining is not the fix. Need task supervision plus preservation of general language.

## Stage 6 — Mixed FineWeb + Semantic (1k → 10k)

- **Idea**: mix 90% FineWeb text with 10% synthetic semantic in every batch, so the model keeps learning language while also learning to answer.
- **Result**: mixed training prevents catastrophic forgetting. **mixed-2k** peaks: grammar 55.6% (pretrain level), factual QA 100%, reasoning 100%, paraphrase 100%. Composite 0.514.
- **Beyond 2k**: constant LR → oscillation. By 10k, factual QA 28.6%, reasoning 33.3%.
- **Lesson**: (a) mixture is the fix for forgetting; (b) **constant LR + long runs oscillate** → need LR scheduling.

## Stage 7 — SEMV2 (semantic v2, exp1)

- **What changed**: (a) clean per-row mixture (FineWeb OR packed semantic, not interleaved), (b) **assistant-only loss masking** (the model must learn to respond, not echo the question), (c) warmup + cosine LR, (d) procedural generation of unique examples (disjoint names), (e) Alpaca instruction subset, (f) FineWeb held-out validation-loss monitor.
- **Result**: composite 0.662 (1k) → 0.697 (3k) → decline. catA strict 67%.
- **Lesson**: loss masking + procedural data + LR schedule beat the old mixed runs. Overtraining after ~3k.

## Stage 8 — SEMV2B (exp2, 3k continuation)

- **What**: 90/10 FineWeb, lr 5e-5, extra arithmetic + transitive data.
- **Result**: semv2b-3000 composite 0.732 (best); semv2b-5000 0.676. Val loss 3.574 vs train 3.483 → reduced overfit.
- **Lesson**: 3k remains the sweet spot; further steps regress.

## Stage 9 — SEMV2C (2k/3k) — CURRENT BEST

- **What**: continuation targeting weak categories (sun, entity tracking, instruction).
- **Result**: semv2c-2000 = 0.751 (with the only nonzero arithmetic ever recorded: 33%), **semv2c-3000 = 0.780**.
- **Milestone**: semv2c-3000 was copied to `checkpoints/best/checkpoint_best.pt` (with backups `checkpoint_best_original.pt`, `checkpoint_semv2b-3000.pt`, `checkpoint_semv2c-3000.pt`).
- Per-category (strict): relations 100%, state 100%, transitive 100%, instruction 86%, why_context 80%, entity_tracking 80%, negation 75%, sun_facts 50%, arithmetic 0%.

## Stage 10 — Real generation test (the wake-up call)

- 15 ad-hoc free-form prompts on `checkpoint_best.pt` (chat wrapper ON, temp 0.3, top-k 40).
- **Score: 5/15 (33%)**.
- Single-fact retrieval works; multi-fact reasoning, paraphrase, sunset/west, negation inference fail.
- **Lesson**: the 0.78 composite and the 33% free-form score are BOTH real. The benchmark measures narrow keyword-match tasks; free-form measures open behavior. This motivated **deterministic (canonical) evaluation** so benchmark decisions don't depend on sampling luck.

## Stage 11 — Canonical deterministic evaluation

- Built `src/evaluation/canonical_eval.py`: same held-out core, but **greedy (temp 0)**, with full metadata.
- Baseline (checkpoint_best.pt) canonical composite: **0.7463** (catA strict 72.5%).
- Fixes the earlier stochastic instability (same model gave 13/15 then 5/15 on interactive runs).

## Stage 12 — SEMV2E (current active experiment)

- **Motivation**: baseline is excellent on single-fact tasks (relations 100%, single_fact 53%) but poor on *binding under distraction* (distractor 25%, distractor_binding 32%, pronoun 30%).
- **Design**: leakage-safe procedural binding data; 55% binding / 15% instruction / 15% existing semantic / 10% FineWeb / 5% arithmetic.
- **Baseline on the new benchmark**: chat 43.2% / plain 25.8%.
- **Training**: COMPLETED 5000 steps in ~963 s; final loss 0.2976; val 4.3915. Checkpoints at 500/1000/2000/3000/5000.
- **Status now**: evaluation of the 5 milestones is the next step (pending).

---

## Timeline of composite scores (held-out, chronological)

| Stage | Model | Composite |
|---|---|---|
| 2 | checkpoint-106k.pt | 0.380 |
| 6 | checkpoint-mixed-2k.pt | 0.514 |
| 7 | semv2-3000 | 0.697 |
| 8 | semv2b-3000 | 0.732 |
| 9 | semv2c-2000 | 0.751 |
| 9 | **semv2c-3000 (best)** | **0.780** |
| 11 | semv2c-3000 (canonical greedy) | 0.7463 |
| 12 | SEMV2E milestones | PENDING |

---

*This document is part of the Project XYZ documentation set. See README.md for the main entry point.*
