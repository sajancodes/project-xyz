# CHECKPOINT GUIDE

Everything about checkpoints: where they live, what they contain, and how to keep them safe.

## 1. The rules (never break these)

1. **NEVER overwrite** the best checkpoint.
2. Before any experiment: back up the best.
3. During experiments: write to a NEW experiment directory (e.g. `checkpoints/semv2e/`).
4. At milestones: save immutable checkpoints (named with step/type).
5. Promotion rule: a new checkpoint becomes *best* only if it (a) genuinely improves the benchmark in a deterministic, reproducible way, (b) improves meaningful semantic behavior, (c) does not unacceptably regress strong areas, (d) holds out leakage rules, (e) does not depend on sampling luck.
6. Integrity: the primary best and its backup must be identical (same SHA-256).

## 2. Where checkpoints live

```
checkpoints/
├── best/
│   ├── checkpoint_best.pt            ★ OFFICIAL BASELINE (semv2c-3000, composite 0.780)
│   ├── checkpoint_best_original.pt     immutable backup (identical content)
│   ├── checkpoint_semv2b-3000.pt      preserved old best (0.732)
│   └── checkpoint_semv2c-3000.pt      preserved semv2c-3000 (0.780)
├── pretrain/
│   ├── checkpoint-10k.pt / -20k / -50k / -106k.pt
│   └── checkpoint_fineweb.pt
├── mixed/
│   └── checkpoint-mixed-2k.pt
├── semantic_v2/
│   ├── checkpoint_exp1_latest.pt  (SEMV2 exp1)
│   ├── checkpoint_exp2_latest.pt  (SEMV2B exp2)
│   └── checkpoint_test-10.pt / -20.pt (smoke tests)
├── semantic/
│   └── checkpoint-semantic-5000.pt
└── semv2e/
    ├── checkpoint_semv2e-500.pt / -1000 / -2000 / -3000 / -5000.pt  (CURRENT experiment)
    ├── checkpoint_semv2e-latest.pt
    └── checkpoint_smoke1-10.pt / -latest.pt, checkpoint_smoke2-10.pt / -latest.pt
```

⚠️ **`checkpoints/latest_checkpoint.pt` (root) is the step-20 SMOKE-TEST checkpoint** (loss 3.6832, `training_type=smoke_test`). It is NOT the best model. The official baseline is `checkpoints/best/checkpoint_best.pt`.

## 3. What a checkpoint contains

A checkpoint is a dict with (at minimum):
- `model_state_dict` — the weights
- `step`, `loss`, `training_type`, `base_checkpoint` (parent info)
- SEMV2/SEMV2E family additionally stores: `optimizer_state_dict`, `scheduler_state_dict`, RNG states, config, mixture config, hyperparameters — fully resumable.

## 4. Verified metadata (audited 2026-08-13)

| File | step | loss | training_type | parent |
|---|---|---|---|---|
| best/checkpoint_best.pt | 3000 | 3.5710 | semantic_v2_mixed | checkpoints/pretrain/checkpoint-106k.pt |
| best/checkpoint_best_original.pt | 3000 | 3.5710 | semantic_v2_mixed | same |
| best/checkpoint_semv2b-3000.pt | 3000 | 3.5710 | semantic_v2_mixed | same (content copy of semv2b-3000) |
| best/checkpoint_semv2c-3000.pt | 3000 | 3.5710 | semantic_v2_mixed | same |
| pretrain/checkpoint-106k.pt | 106000 | 3.5570 | pretrain | random init |
| mixed/checkpoint-mixed-2k.pt | 2000 | 3.5470 | mixed | checkpoints/pretrain/checkpoint-106k.pt |
| semv2e/checkpoint_semv2e-500.pt | 500 | 2.2549 | semv2e | best/checkpoint_best.pt |
| semv2e/checkpoint_semv2e-1000.pt | 1000 | 1.3516 | semv2e | best/checkpoint_best.pt |
| semv2e/checkpoint_semv2e-2000.pt | 2000 | 0.8392 | semv2e | best/checkpoint_best.pt |
| semv2e/checkpoint_semv2e-3000.pt | 3000 | 0.5312 | semv2e | best/checkpoint_best.pt |
| semv2e/checkpoint_semv2e-5000.pt | 5000 | 0.2976 | semv2e | best/checkpoint_best.pt |
| semv2e/checkpoint_semv2e-latest.pt | 5000 | 0.2976 | semv2e | best/checkpoint_best.pt |
| semv2e/checkpoint_smoke*-10.pt | 10 | 2.5494 | semv2e | best/checkpoint_best.pt |
| latest_checkpoint.pt (root) ⚠️ | **20** | 3.6832 | **smoke_test** | — |

## 5. The lineage tree

```
none (random init)
└─ checkpoints/pretrain/checkpoint-106k.pt
   ├─ checkpoints/semantic/checkpoint-semantic-5000.pt
   ├─ checkpoints/pretrain/checkpoint_fineweb.pt
   ├─ checkpoints/mixed/checkpoint-mixed-2k.pt
   ├─ checkpoints/semantic_v2/checkpoint_exp1_*.pt  (SEMV2)
   │   └─ checkpoints/semantic_v2/checkpoint_exp2_*.pt  (SEMV2B)
   │       └─ checkpoints/best/checkpoint_semv2c-3000.pt  (SEMV2C) ★
   │           └─ checkpoints/best/checkpoint_best.pt (copy)
   └─ checkpoints/semv2e/checkpoint_semv2e-*.pt  (SEMV2E, from best)
```

## 6. How to verify the best checkpoint

```powershell
python -c "import torch; c=torch.load(r'checkpoints\best\checkpoint_best.pt', map_location='cpu'); print(c['step'], c['loss'], c['training_type'])"
```
Expected: `3000 3.5710 semantic_v2_mixed`.

Hash integrity:
```powershell
Get-FileHash checkpoints\best\checkpoint_best.pt, checkpoints\best\checkpoint_best_original.pt
```
They must match.

## 7. Known caveats / history

- `eval_mixed2k_fixed.json` labels `checkpoints/best/checkpoint_best.pt` as "step 2000, mixed_fineweb_semantic", but the real metadata says step 3000 `semantic_v2_mixed`. **Trust `checkpoint_info` inside each JSON, not the file name.**
- `checkpoint-10k.pt` records step **10500** (saved mid-session).
- Several root-level `*.pt` in older layouts were migrated into `checkpoints/…`; some references in very old docs point to flat names.

---

*Related: [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md), [docs/EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).*