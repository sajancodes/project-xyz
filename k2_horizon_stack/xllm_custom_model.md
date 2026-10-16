# Scaling your own Project XYZ model with IFM xLLM

The local trainer is for architecture development. IFM's xLLM is the closer match to the public K2 Horizon pretraining path when you have Linux/CUDA cluster resources.

## Important

Do **not** select `k2-horizon-0.9B` unless you actually want K2's architecture. Your requirement is to keep the model itself yours, so create a new xLLM model configuration for your architecture.

## 1. Bootstrap upstreams

```bash
python k2_horizon_stack/bootstrap_ifm.py
cd k2_horizon_stack/_upstream/ifm/xllm
```

Follow xLLM's own installation instructions for its required PyTorch/CUDA versions and native extensions.

## 2. Register your architecture

xLLM's transformer configurations live in `xllm/models/transformer.py` in the current upstream release. Add a new `ConfStore` entry with a unique name such as `project-xyz-v1` and map your architecture parameters:

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

If your model is not representable by xLLM's existing dense/MoE/MoVA primitives, implement a new xLLM architecture instead of forcing incorrect parameters into `ModelConf`.

## 3. Start from K2's *training mechanics*, not K2's model name

A scale-up launch should preserve the knobs you want from the K2 recipe while using your model id:

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

Tune LR, context, batch size, warmup and total tokens for **your** parameter count; copying K2's numeric values blindly can make a smaller Project XYZ model train worse.

## 4. Data preparation

Use IFM's `pretraining-data-toolkit` for large-run annotation/quality bucketing/shuffling. The local `prepare_data.py` is intentionally lightweight and useful for smoke tests, but it does not replace the full large-scale curation pipeline.

## 5. Evaluation and checkpoint selection

Evaluate intermediate checkpoints using xLLM's evaluation path. Keep validation perplexity plus downstream tasks appropriate to your model goals. Record the exact checkpoint, tokenizer, data mixture and upstream commit hashes.

## 6. Post-training

Once pretraining is stable:

1. export/convert through xBridges as needed;
2. prepare SFT/reasoning/tool trajectories;
3. apply process-quality gates;
4. use RL360 for compatible code/tool/sandbox RL workloads when you have its required cluster environment.

This gives you K2-like infrastructure while the neural architecture, size and final learned weights remain Project XYZ's.
