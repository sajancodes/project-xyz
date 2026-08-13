# TRAINING GUIDE

How training works in Project XYZ, and how to reproduce each run.

## 1. The training loop (all runs)

1. Build a batch of 16 sequences × 512 tokens.
2. Forward pass → logits.
3. Compute loss (masked for semantic rows, full for FineWeb rows).
4. Backward pass → gradients; clip; AdamW step.
5. LR scheduler step (warmup + cosine in semantic/SEMV2E runs; constant in old mixed runs).
6. Log every 10 steps; compute FineWeb held-out val loss periodically; save milestones.

Tokens per step: `16 × 512 = 8,192`.

## 2. Loss masking (semantic/SEMV2E)

For `User: ...\nAssistant: <answer>` rows, the loss mask is **1 on the assistant span and 0 on everything before it**. The model therefore learns to *produce the answer*, not to copy the question. This principle (introduced in SEMV2) strongly reduces echo/repetition.

## 3. Training types and their scripts

| Training | Script | Base | LR | Mixture |
|---|---|---|---|---|
| FineWeb pretraining | `src/training/train_fineweb.py` / `train_fineweb_fast.py` | random init / resumable | 3e-4 | 100% FineWeb |
| Semantic fine-tuning | `src/training/train_fineweb_fast.py` (5k) | checkpoint-106k.pt | 5e-5 | 100% synthetic (41 examples) |
| Fact injection (DEPRECATED) | `src/training/train_fact_injection.py` | checkpoint-106k.pt | — | 1 repeated fact |
| Mixed | `src/training/train_mixed.py` | checkpoint-106k.pt | 1e-4 const | 90% FineWeb + 10% semantic |
| SEMV2 | `src/training/train_semantic_v2.py` | checkpoint-106k.pt | 1e-4 warmup+cosine | procedural + Alpaca + FineWeb |
| SEMV2D (designed, NOT verified) | `src/training/train_semantic_v2d.py` | — | — | 50/20/15/15 |
| **SEMV2E (current)** | `src/training/train_semv2e.py` | **checkpoint_best.pt** | 1e-4 warmup+cosine | 55/15/15/10/5 |

## 4. SEMV2E hyperparameters (the run just completed)

| Setting | Value |
|---|---|
| Base checkpoint | `checkpoints/best/checkpoint_best.pt` (step 3000) |
| Steps | 5000 (milestones at 500/1000/2000/3000/5000) |
| Peak LR | 1e-4 |
| LR schedule | warmup 150 + cosine to 10% |
| Batch | 16 |
| Seq len | 512 |
| Weight decay | 0.05 |
| Grad clip | 1.0 |
| Optimizer | AdamW |
| Device | cuda (RTX 4050 6 GB) |
| Throughput | ~42,000 tok/s (GPU) |
| Run time | ~963 s (~16 min) |
| Final train loss | 0.2976 |
| FineWeb held-out val loss | 4.3915 |
| Tokens | 40,960,000 |
| Rows/batch | binding 9 + instruction 2 + existing 2 + FineWeb 2 + arithmetic 1 |

### Data pools (SEMV2E)
- 20,000 binding examples (`src/utils/semv2e_data.py`, seed 20260813, leakage-safe names disjoint from benchmark).
- 20,000 existing semantic/reasoning examples (`SemanticGenerator`).
- 8,000 Alpaca-cleaned instruction examples (yahma/alpaca-cleaned).
- 1,000 arithmetic examples.
- FineWeb buffer with a held-out val window reserved (60,290 train + 42,312 val tokens).

## 5. How to reproduce the SEMV2E run

```powershell
cd "C:\Users\Sajan\Desktop\Project XYZ"
.venv\Scripts\activate
# Smoke test first (10 steps, no network, preset buffer):
python src\training\train_semv2e.py --smoke --name smoke3
# Real run:
python src\training\train_semv2e.py --steps 5000 --name semv2e
```

Outputs land in `checkpoints/semv2e/checkpoint_<name>-<step>.pt` and `checkpoint_<name>-latest.pt`.

## 6. FineWeb pretraining (for completeness)

```powershell
python src\training\train_fineweb.py            # resumable, saves to checkpoints/pretrain/
python src\training\train_fineweb_fast.py       # optimized version
```

Milestones every 1k/5k steps; final model at 106k.

## 7. Critical training rules (learned the hard way)

1. **Never train on held-out test data.** SEMV2E training names are disjoint from benchmark names by design.
2. **Constant LR over ~2k steps oscillates** — always use warmup + cosine for semantic runs.
3. **Tiny repeated datasets cause catastrophic forgetting** — always mix in FineWeb for preservation.
4. **Write checkpoints to experiment-specific directories** — never into `checkpoints/best/`.
5. **Verify the base checkpoint is the real best** (step 3000, `semantic_v2_mixed`, loss 3.5710), not `latest_checkpoint.pt` (a step-20 smoke test).

---

*Related: [docs/EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md), [docs/DATA_GUIDE.md](DATA_GUIDE.md).*
