# K2 Horizon parity map

This file separates what Project XYZ now reproduces from what requires IFM's larger infrastructure.

## Local parity implemented here

| K2-style capability | Project XYZ implementation |
|---|---|
| auditable dataset preparation | `prepare_data.py` + `.meta.json` source/tokenizer manifest |
| streaming source ingestion | Hugging Face streaming or local JSONL |
| tokenizer-based token stream | `tokenizers` using Project XYZ's tokenizer or any replacement |
| packed contiguous training | memory-mapped token blocks with next-token targets |
| AdamW | configurable LR, betas, eps, weight decay |
| warmup + cosine decay | `LambdaLR`, configurable warmup and minimum LR ratio |
| BF16/FP16 | BF16 preferred; automatic FP16 fallback when unsupported |
| accumulation + clipping | configurable accumulation and global grad-norm clipping |
| multi-GPU | `torchrun` + DDP local backend |
| checkpoint/resume | model, optimizer, scheduler, scaler, RNG, config |
| intermediate evaluation | validation loss/perplexity at configurable intervals |
| experiment provenance | run manifest + data manifest + config stored with checkpoints |
| model independence | dynamic model/config loader; architecture is owned by the user |

## Official IFM components intentionally kept upstream

Run `bootstrap_ifm.py` to obtain these directly from IFM:

- **xLLM** — K2-style scalable pretraining, FSDP1/FSDP2, model/context parallelism, best-fit packing, long-context kernels, checkpoint/eval/export path.
- **pretraining-data-toolkit** — model-based annotation, quality bucketing, shuffling and dataset inspection.
- **xBridges** — checkpoint conversion and serving/evaluation bridges.
- **RL360** — K2 Horizon reinforcement-learning/post-training stack for coding/tool/sandbox tasks.
- **process_entry** — quality gate for training trajectories.
- **LCQA** — long-context QA data-generation pipeline.

They remain separate because importing a snapshot of thousands of upstream files into Project XYZ would make updates, attribution and security review worse. The bootstrap records the exact commit checked out for each run.

## What is NOT identical to K2 Horizon

1. **Your model architecture.** This is deliberately unspecified. K2's architecture/config is not forced onto Project XYZ.
2. **Your training corpus.** You choose and license the data. Reproducing a training method does not automatically reproduce IFM's complete corpus or mixture.
3. **Compute scale.** IFM's published xLLM K2 recipes target large Linux NVIDIA clusters. A 6 GB laptop GPU cannot reproduce their wall-clock scale or large-model memory footprint.
4. **FSDP2/context parallelism on Windows.** Use the local DDP trainer for development. For serious xLLM scale-up, use Linux/WSL2 where supported or a Linux cluster.
5. **RL360 cluster requirements.** The released RL360 example expects Slurm, Docker/NVIDIA Container Toolkit, shared storage and many datacenter GPUs. It is integrated as an upstream post-training stage, not pretended to be a laptop feature.

## Recommended path

- develop data/model correctness with the local backend
- train tiny/small architecture variants and verify loss decreases
- freeze a reproducible data manifest and tokenizer
- evaluate every checkpoint, not just the final one
- once architecture is stable, register it in xLLM and scale with FSDP2
- use IFM data tooling before large pretraining runs
- export selected checkpoints through xBridges
- only then add SFT/RL/tool-use post-training through the relevant upstream stack

That keeps Project XYZ close to K2 Horizon's reproducible engineering workflow without falsely claiming identical training conditions.
