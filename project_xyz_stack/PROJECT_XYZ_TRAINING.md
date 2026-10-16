# Project XYZ Training Specification

This document defines the engineering contract for the Project XYZ training system.

The goal is not to lock Project XYZ to one neural architecture. The architecture, parameter count, context length, tokenizer, corpus, and final weights remain Project XYZ decisions. The training system is designed so those choices can evolve without rewriting the entire pipeline.

## Core capabilities

| Capability | Project XYZ implementation |
|---|---|
| auditable dataset preparation | `prepare_data.py` plus a `.meta.json` source/tokenizer manifest |
| streaming source ingestion | Hugging Face streaming or local JSONL |
| tokenizer-controlled token stream | `tokenizers` using the Project XYZ tokenizer or a replacement tokenizer |
| packed contiguous training | memory-mapped token blocks with next-token targets |
| AdamW optimization | configurable learning rate, betas, epsilon, and weight decay |
| warmup plus cosine decay | configurable warmup and minimum learning-rate ratio |
| BF16 / FP16 | BF16 preferred with FP16 fallback when unsupported |
| accumulation and clipping | configurable gradient accumulation and global gradient-norm clipping |
| multi-GPU local backend | `torchrun` plus PyTorch DDP |
| checkpoint and resume | model, optimizer, scheduler, scaler, RNG state, configuration |
| intermediate evaluation | validation loss and perplexity at configurable intervals |
| experiment provenance | run manifest plus data manifest plus config stored with checkpoints |
| model independence | dynamic model/config loader; architecture remains owned by Project XYZ |

## Optional scale-up components

Run `bootstrap_project_xyz.py` to clone optional upstream research backends used when Project XYZ outgrows the local trainer:

- **xLLM** — scalable pretraining with FSDP, model/context parallelism, efficient packing, long-context kernels, checkpointing, evaluation, and export paths.
- **pretraining-data-toolkit** — annotation, quality bucketing, shuffling, and dataset inspection for large corpora.
- **xBridges** — checkpoint conversion and serving/evaluation bridges.
- **RL360** — optional reinforcement-learning and tool/sandbox training infrastructure.
- **process_entry** — optional quality gates for generated training trajectories.
- **LCQA** — optional long-context QA generation tooling.

These remain upstream dependencies instead of being copied into this repository. That keeps Project XYZ smaller, makes security review easier, and allows exact upstream revisions to be pinned per experiment.

## Local development contract

The local backend is the reference path for:

1. architecture development;
2. tokenizer verification;
3. data-pipeline verification;
4. loss-curve sanity checks;
5. reproducibility tests;
6. checkpoint correctness;
7. evaluation development;
8. small-model experiments;
9. single-GPU training;
10. small multi-GPU training with DDP.

The local backend is intentionally understandable. It should remain possible to inspect every step from raw text to a gradient update without needing a cluster runtime.

## Scale-up contract

When Project XYZ moves to larger models or longer contexts, the intended transition is:

1. freeze a known-good tokenizer;
2. freeze or version the corpus manifest;
3. verify the architecture with the local backend;
4. register the same architecture in the selected distributed backend;
5. reproduce a short local loss curve on the distributed backend;
6. validate checkpoint conversion in both directions where possible;
7. run a small distributed smoke test;
8. run evaluation before any long training job;
9. only then scale token count and GPU count.

A scale-up run is not considered equivalent merely because it starts successfully. Loss, gradients, parameter count, tokenizer behavior, data ordering, optimizer settings, and checkpoint reload behavior must be compared.

## Model independence

Project XYZ deliberately does not force a particular model configuration.

The default configuration points at the current repository implementation:

```text
src/models/model.py
src/config/model_config.py
```

To train a different architecture, change only the model section of:

```text
project_xyz_stack/config/project_xyz.json
```

Required information:

```json
{
  "model_file": "path/to/model.py",
  "model_class": "YourModel",
  "config_file": "path/to/config.py",
  "config_class": "YourConfig",
  "config_kwargs": {}
}
```

The adapter accepts common model-return styles and normalizes them to logits plus a scalar loss.

## Data requirements

Every serious Project XYZ run should be traceable to:

- source dataset names or local source paths;
- source revisions when available;
- tokenizer path and tokenizer hash;
- preprocessing version;
- document count;
- token count;
- output dtype;
- EOS handling;
- train/validation split definition;
- filtering decisions;
- deduplication decisions;
- mixture weights;
- known licensing constraints.

The lightweight local preparer records the fields it can know automatically. Larger corpus pipelines should extend the manifest rather than discard it.

## Optimization defaults

The checked-in configuration is a starting point, not a universal optimum.

Current defaults:

```text
optimizer: AdamW
beta1: 0.9
beta2: 0.95
weight_decay: 0.1
grad_clip: 1.0
scheduler: warmup + cosine decay
precision: bf16 preferred
```

Learning rate, batch size, warmup, context length, token budget, and minimum learning-rate ratio must be tuned for the architecture and compute regime actually used.

## Checkpoint requirements

A Project XYZ training checkpoint must contain enough state to continue the same optimization trajectory as closely as the runtime permits.

Stored state includes:

- model weights;
- optimizer state;
- scheduler state;
- gradient scaler state when applicable;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- CUDA RNG state when available;
- global step;
- Project XYZ configuration.

The current checkpoint format identifier is:

```text
project-xyz-checkpoint-v2
```

## Evaluation requirements

Do not select a model from training loss alone.

Minimum evaluation should include:

- held-out language-model loss;
- perplexity;
- Project XYZ semantic benchmarks;
- deterministic generation tests;
- regression tests against the previous official baseline;
- task-specific tests for capabilities the experiment is supposed to improve;
- contamination checks when benchmark data could overlap training data.

Intermediate checkpoints should be evaluated because the final checkpoint may not be the best checkpoint.

## Reproducibility requirements

For an experiment to be reproducible, record:

- git commit;
- config file;
- random seed;
- tokenizer hash;
- dataset manifest;
- package versions;
- PyTorch version;
- CUDA runtime;
- GPU model;
- number of GPUs;
- effective global batch size;
- total tokens consumed;
- training steps;
- optimizer state at resume boundaries;
- evaluation outputs;
- checkpoint hashes where practical.

## Compute realism

Project XYZ supports both laptop-scale research and cluster-scale training, but those are different operating regimes.

A 6 GB laptop GPU is appropriate for:

- the current small model;
- tiny architecture prototypes;
- tokenizer/data experiments;
- short pretraining runs;
- checkpoint/evaluation development;
- activation and weight-map experiments;
- debugging distributed-compatible code at small scale.

Larger models require techniques such as:

- activation checkpointing;
- sharded optimizer state;
- FSDP;
- tensor/model parallelism;
- context parallelism;
- sequence packing;
- efficient attention kernels;
- distributed checkpointing;
- high-bandwidth interconnects;
- larger host memory and storage bandwidth.

The repository should never claim laptop parity with infrastructure it does not have.

## Recommended Project XYZ path

- prove the model can overfit a tiny batch;
- prove it can learn from a small real corpus;
- freeze and validate the tokenizer;
- build a deterministic validation split;
- train several small architecture variants;
- compare quality per parameter and quality per token;
- select an architecture using evidence rather than intuition;
- scale the corpus only after data quality is measured;
- checkpoint frequently enough to study capability emergence;
- evaluate every important checkpoint;
- preserve intermediate checkpoints for weight/activation research;
- register the stable architecture in the distributed backend;
- scale compute gradually;
- add supervised post-training only after pretraining is stable;
- add preference/RL stages only when their data and reward signals are validated;
- publish enough manifests and configuration to reproduce each reported result.

This specification is the Project XYZ engineering baseline. Any future training backend should satisfy the same observability, reproducibility, and evaluation requirements even if its implementation changes.
