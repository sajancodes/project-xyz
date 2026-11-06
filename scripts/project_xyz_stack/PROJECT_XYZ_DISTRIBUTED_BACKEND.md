# Project XYZ Distributed Backend Guide

The local Project XYZ trainer is the reference implementation for architecture development and reproducibility. When the model or context becomes too large for the local backend, move the same Project XYZ architecture to a distributed training framework rather than changing the model just to fit the framework.

## 1. Bootstrap optional upstream backends

```bash
python project_xyz_stack/bootstrap_project_xyz.py
cd project_xyz_stack/_upstream/project_xyz_backends/xllm
```

Follow each upstream project's installation instructions for its supported PyTorch, CUDA, compiler, and native-extension versions.

## 2. Keep the model identity Project XYZ

Do not select an upstream example model configuration as a shortcut unless that is intentionally the architecture you want to train.

Create a unique model identifier such as:

```text
project-xyz-v1
project-xyz-v2
project-xyz-dense-1b
project-xyz-moe-4b-a1b
```

The identifier should correspond to a versioned architecture specification in this repository.

## 3. Register the Project XYZ architecture

For frameworks that expose a configuration registry, map Project XYZ architecture parameters explicitly.

Conceptual example:

```python
ConfStore["project-xyz-v1"] = ModelConf(
    arch="transformer",
    num_layers=YOUR_LAYERS,
    model_dim=YOUR_D_MODEL,
    num_heads=YOUR_HEADS,
    num_kv_heads=YOUR_KV_HEADS,
    head_dim=YOUR_HEAD_DIM,
    rope_base=YOUR_ROPE_BASE,
    ffn_hidden_dim=YOUR_FFN_DIM,
    swiglu=YOUR_SWIGLU_SETTING,
    apply_rmsnorm=YOUR_RMSNORM_SETTING,
    layernorm_num_groups=1,
    rmsnorm_eps=YOUR_EPS,
)
```

If the selected framework cannot represent the Project XYZ architecture faithfully, implement the missing primitive or choose another backend. Do not silently alter the architecture merely to make a launch command work.

## 4. Verify parameter parity

Before distributed training, compare:

- total parameter count;
- trainable parameter count;
- embedding dimensions;
- layer count;
- attention head count;
- KV-head count if grouped-query attention is used;
- FFN dimensions;
- normalization type and epsilon;
- positional encoding implementation;
- vocabulary size;
- tied versus untied embeddings;
- initialization statistics;
- attention masking semantics.

A mismatch in any of these means you are no longer training the same Project XYZ model.

## 5. Verify forward parity

Use identical weights and identical input tokens where possible.

Compare:

```text
logit shape
selected logit values
loss
hidden-state statistics
attention-mask behavior
```

Floating-point kernels can introduce small numerical differences. Large differences require investigation before scaling.

## 6. Verify data parity

The distributed backend must receive the same logical token stream as the local backend.

Check:

- tokenizer revision;
- BOS/EOS behavior;
- document boundaries;
- packing strategy;
- truncation rules;
- sequence length;
- train/validation split;
- mixture weights;
- shuffling seed;
- duplicate filtering;
- quality filtering.

## 7. Example Project XYZ distributed launch

The exact flags depend on the selected backend and hardware. A representative launch can look like:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --model project-xyz-v1 \
  --model.ddp_backend fsdp2 \
  --model.causal_attn_backend flash \
  --data "/shared/data/project_xyz:1.0:text:text" \
  --tokenizer.type huggingface \
  --tokenizer.path /shared/tokenizer \
  --dataloader.packing_type bestfit \
  --dataloader.buffer_size 65536 \
  --seq_len YOUR_CONTEXT \
  --batch_size YOUR_PER_RANK_BATCH \
  --dtype bf16 \
  --steps YOUR_STEPS \
  --optim.lr YOUR_LR \
  --optim.scheduler cosine \
  --optim.beta1 0.9 \
  --optim.beta2 0.95 \
  --optim.weight_decay 0.1 \
  --optim.clip 1.0 \
  --optim.warmup YOUR_WARMUP \
  --log_freq 10 \
  --dump_freq 500 \
  --eval_freq 5000 \
  --dump_dir /shared/runs/project_xyz_v1
```

Treat this as a template, not a guaranteed optimum. Tune learning rate, global batch size, context length, warmup, total tokens, and checkpoint frequency for the actual Project XYZ architecture.

## 8. Effective batch size

Always calculate and record the effective global batch size.

Conceptually:

```text
global_batch = micro_batch_per_rank
             × data_parallel_ranks
             × gradient_accumulation_steps
```

When sequence length is fixed, tokens per optimizer step are approximately:

```text
tokens_per_step = global_batch × sequence_length
```

Record both values in every run manifest.

## 9. Learning-rate scaling

Do not blindly scale learning rate with GPU count.

Changing GPU count often changes the effective batch size, which may justify learning-rate changes, but the correct scaling law depends on optimizer, model size, data distribution, and training regime.

Run short controlled sweeps before committing large compute.

## 10. Mixed precision

Project XYZ prefers BF16 where hardware supports it because BF16 provides a larger exponent range than FP16.

Use FP16 only when required by the hardware/runtime and keep gradient scaling enabled where needed.

Track:

- overflow events;
- gradient norms;
- NaNs/Infs;
- loss spikes;
- scaler behavior.

## 11. Sharding

For larger Project XYZ models, use sharding to distribute model parameters, gradients, and optimizer state.

Possible strategies include:

- DDP for models that fit independently on each GPU;
- FSDP for sharded parameters and optimizer state;
- tensor/model parallelism for layers too large for one GPU;
- context/sequence parallelism for long-context workloads;
- expert parallelism for mixture-of-experts models.

Select the simplest strategy that fits the model and achieves acceptable utilization.

## 12. Activation checkpointing

Activation checkpointing trades compute for memory.

Use it when activation memory prevents the desired batch/context size, but measure the throughput cost. Do not enable every recomputation option automatically.

## 13. Attention kernels

Efficient attention kernels can materially change memory and throughput.

Before enabling an optimized kernel:

- verify supported dtype;
- verify supported head dimension;
- verify causal-mask semantics;
- verify dropout behavior if used;
- run forward/backward parity tests;
- benchmark actual Project XYZ sequence lengths.

## 14. Dataset workers and packing

Large-scale utilization often depends as much on data delivery as on model kernels.

Monitor:

- CPU utilization;
- storage bandwidth;
- tokenizer throughput;
- dataloader queue depth;
- host-to-device transfer time;
- sequence packing efficiency;
- percentage of padding tokens.

## 15. Checkpointing at scale

Distributed checkpoints can be large and slow.

Design around:

- atomicity;
- partial-write recovery;
- per-rank shards;
- metadata versioning;
- optimizer-state size;
- checkpoint upload/archive time;
- retention policy;
- validation that a saved checkpoint actually reloads.

A checkpoint that has never been reload-tested is not a trusted checkpoint.

## 16. Resume testing

Before a long run, perform this test:

1. train for a short number of steps;
2. save a checkpoint;
3. stop all workers;
4. restart from the checkpoint;
5. verify restored global step;
6. verify optimizer/scheduler state;
7. verify data cursor behavior;
8. compare the continued loss curve to an uninterrupted control run.

## 17. Evaluation workers

For large runs, evaluation may execute asynchronously or on separate resources.

Keep evaluation reproducible by recording:

- exact checkpoint ID;
- evaluator commit;
- benchmark revision;
- tokenizer revision;
- decoding settings;
- number of examples;
- random seed;
- metric implementation.

## 18. Export and inference

After selecting a checkpoint, convert it to the serving format required by your inference runtime.

Conversion must preserve:

- all weights;
- model config;
- tokenizer config;
- special token IDs;
- positional encoding settings;
- tensor shapes and dtypes.

Always compare logits or generation results before and after conversion.

## 19. Post-training handoff

Only begin post-training after the base checkpoint is stable and reproducibly evaluated.

A Project XYZ post-training pipeline can include:

1. supervised instruction tuning;
2. reasoning-data tuning;
3. tool-use trajectory tuning;
4. preference optimization;
5. reinforcement-learning stages;
6. safety and robustness evaluation;
7. domain adaptation.

Each stage should create a new lineage branch instead of overwriting the base checkpoint.

## 20. Final rule

The distributed backend is infrastructure, not the identity of the model.

The model remains **Project XYZ**.

Architecture decisions live in Project XYZ.
Data decisions live in Project XYZ.
Experiment manifests live in Project XYZ.
Evaluation decisions live in Project XYZ.
Checkpoint lineage lives in Project XYZ.

The backend can change without changing the research identity of the model.
