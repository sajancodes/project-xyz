# Project XYZ × K2 Horizon training stack

This directory turns Project XYZ into a reproducible, K2-Horizon-style model-development pipeline while deliberately **not fixing the model architecture**. You keep control of the model; this stack supplies the surrounding methodology: data preparation, packed token streams, reproducible configs, mixed-precision pretraining, gradient accumulation/clipping, checkpoint/resume, evaluation, experiment manifests, and bridges to IFM's released tooling.

## What is mirrored from the K2 Horizon workflow

- staged data preparation and auditable manifests
- tokenizer-driven pretraining data
- packed/contiguous token training
- AdamW with configurable beta values (K2's published 0.9B recipe uses 0.9/0.95)
- cosine LR schedule with warmup and end-LR ratio
- BF16-first training, FP16 fallback
- gradient clipping
- deterministic seed capture and environment manifest
- frequent resumable checkpoints
- validation loss/perplexity during training
- a scale-up path to xLLM/FSDP2 for multi-GPU training
- official IFM post-training/data/evaluation projects bootstrapped separately rather than copied into this repo

The goal is methodological parity, **not a claim that a laptop run is identical to K2 Horizon's cluster run**. Their public xLLM recipes target Linux/CUDA clusters and much larger GPU allocations. The local backend here is intentionally simpler so the exact same experiment config can be developed on one GPU first.

## Layout

```text
k2_horizon_stack/
  config/project_xyz.json     experiment config; replace model paths/classes here
  prepare_data.py             HF/local streaming -> packed token binary + manifest
  model_adapter.py            loads YOUR model without hard-coding architecture
  checkpointing.py            atomic resumable checkpoints + run metadata
  train.py                    single GPU / torchrun DDP trainer
  evaluate.py                 loss + perplexity evaluation
  bootstrap_ifm.py            clone official IFM tools into _upstream/ifm
  xllm_custom_model.md        how to scale the custom model with xLLM/FSDP2
  K2_PARITY.md                exact parity map and deliberate differences
  requirements.txt
```

## 1. Install the local backend

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r k2_horizon_stack/requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for your driver separately. Do not install a CPU-only build if you intend to train on NVIDIA CUDA.

## 2. Point the stack at your model

Edit `k2_horizon_stack/config/project_xyz.json`.

By default it loads the existing Project XYZ classes:

- `src/models/model.py :: SmallEnglishLLM`
- `src/config/model_config.py :: ModelConfig`

Change those four fields to any model/config you build. `model_adapter.py` accepts the common Project XYZ `(input_ids, target_ids) -> (logits, loss)` interface, a Hugging-Face-like object with `.logits/.loss`, or logits-only output.

## 3. Prepare data

Example with a Hugging Face dataset:

```powershell
python k2_horizon_stack/prepare_data.py `
  --dataset HuggingFaceFW/fineweb `
  --dataset-config sample-10BT `
  --split train `
  --tokenizer tokenizer.json `
  --output data/k2style/train.bin `
  --text-field text `
  --max-docs 10000
```

For a local JSONL file:

```powershell
python k2_horizon_stack/prepare_data.py `
  --jsonl data/raw/train.jsonl `
  --tokenizer tokenizer.json `
  --output data/k2style/train.bin
```

The script writes `train.bin` plus `train.bin.meta.json` containing source, tokenizer hash, document/token counts, dtype, EOS policy, and creation parameters.

## 4. Train locally

```powershell
python k2_horizon_stack/train.py --config k2_horizon_stack/config/project_xyz.json
```

Resume is automatic from `runs/project_xyz_k2style/latest.pt` unless disabled in the config.

For multiple local GPUs / Linux nodes, the same trainer supports `torchrun` DDP:

```bash
torchrun --standalone --nproc_per_node=4 k2_horizon_stack/train.py --config k2_horizon_stack/config/project_xyz.json
```

## 5. Evaluate

```powershell
python k2_horizon_stack/evaluate.py `
  --config k2_horizon_stack/config/project_xyz.json `
  --checkpoint runs/project_xyz_k2style/latest.pt `
  --data data/k2style/valid.bin
```

## 6. Pull IFM's official stack

```powershell
python k2_horizon_stack/bootstrap_ifm.py
```

This clones the official public projects into `k2_horizon_stack/_upstream/ifm/` (ignored by git): xLLM, pretraining-data-toolkit, xBridges, RL360, process_entry, and LCQA. Keeping them as external upstreams avoids silently forking huge fast-moving codebases and preserves their licenses/history.

Use `xllm_custom_model.md` when you are ready to register your own architecture in xLLM and scale to FSDP2 / context-parallel training.

## K2-like experiment lifecycle

1. collect + license-check sources
2. annotate/filter/bucket quality
3. shuffle and construct source mixture
4. tokenize/pack
5. pretrain with checkpointing
6. evaluate checkpoints, not only the final model
7. export/serve a selected checkpoint
8. SFT / reasoning / tool-use post-training
9. run process-quality gates and task verification
10. publish data mixture, config, logs, checkpoints, and evals for reproducibility

Project XYZ now has hooks for all ten stages; only the actual model architecture and the data/compute you choose are intentionally left to you.
