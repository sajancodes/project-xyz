# Track B: Qwen3 4B Cybersecurity QLoRA Architecture & Implementation

## 1. Overview & Research Objective

Project XYZ incorporates two distinct, non-interfering research tracks:
- **Track A**: 14.66M parameter decoder-only language model trained completely from scratch on FineWeb, studying the transition from next-token prediction to emergent semantic behavior.
- **Track B**: Pretrained **`Qwen/Qwen3-4B-Instruct-2507`** (4-billion parameter model) specialized via **QLoRA** into a state-of-the-art cybersecurity defensive assistant while preventing catastrophic regression in general instruction following, coding, and reasoning.

The two tracks use completely separate tokenizers, checkpoints, weights, and configurations.

---

## 2. Model & Quantization Architecture

### Base Model
- **Identifier**: `Qwen/Qwen3-4B-Instruct-2507` (Hugging Face official weights).
- **Format**: Unquantized PyTorch/Safetensors original checkpoint. Prequantized GGUF, AWQ, or GPTQ files are strictly avoided to ensure clean weight gradients and accurate NF4 runtime quantization.

### Runtime NF4 4-bit Quantization
Quantization is performed dynamically at load time via `bitsandbytes`:
- **Quantization Type**: NormalFloat 4 (NF4), an information-theoretically optimal quantile distribution for normally distributed neural network weights.
- **Nested / Double Quantization**: Enabled (`bnb_4bit_use_double_quant=True`), quantizing quantization constants to save ~0.37 bits per parameter.
- **Compute Precision**: `torch.float16` (`bnb_4bit_compute_dtype="float16"`), matching the native Tensor Core capabilities of the NVIDIA T4 GPU.

### Memory Footprint on NVIDIA T4 (16 GB VRAM)
- Base 4B model in NF4 4-bit: **~2.8 GB VRAM**
- LoRA Adapters (r=32): **~0.15 GB VRAM**
- KV Cache & Activations (Gradient Checkpointing, Seq 1024, Batch 1): **~2.5 GB VRAM**
- Optimizer States (8-bit Paged AdamW): **~0.4 GB VRAM**
- Total Training VRAM Footprint: **~6.0 – 8.5 GB VRAM**, safely within the 16 GB physical limit of the Google Colab NVIDIA T4 runtime.

---

## 3. PEFT QLoRA Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` | Official modern Qwen3 instruct checkpoint |
| LoRA Rank ($r$) | 32 (fallback 16 on OOM) | Ample expressive capacity for domain specialization |
| LoRA Alpha ($\alpha$) | 64 | $2 \times r$ scaling ratio |
| LoRA Dropout | 0.05 | Prevents overfitting on specialized patterns |
| Bias | `none` | Keeps trainable footprint minimal |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Adapts both self-attention and MLP projection layers |
| Optimizer | `paged_adamw_8bit` | Offloads memory spikes to system RAM if needed |
| Micro-batch Size | 1 | Zero OOM risk on T4 |
| Gradient Accumulation | 16 | Effective batch size of 16 sequences |
| Learning Rate | $2.0 \times 10^{-4}$ | Standard stable LoRA learning rate |
| LR Scheduler | Cosine with 5% warmup | Smooth decay preventing early shock |
| Max Sequence Length | 1024 (configurable to 2048) | Covers complete multi-turn cybersecurity dialogues |

---

## 4. Assistant-Only Loss Masking

A critical requirement of chat supervised fine-tuning (SFT) is that the language model must **only** be trained to generate assistant responses.

### Masking Architecture
Using the official Qwen chat template delimiters:
```
<|im_start|>system\n{system_content}<|im_end|>\n
<|im_start|>user\n{user_content}<|im_end|>\n
<|im_start|>assistant\n{assistant_content}<|im_end|>\n
```

1. **System Tokens**: Masked with `-100` (`IGNORE_INDEX`).
2. **User Question Tokens**: Masked with `-100`.
3. **Assistant Prompt Header** (`<|im_start|>assistant\n`): Masked with `-100`.
4. **Assistant Content Tokens**: **Active Loss** (`label = input_id`).
5. **End-of-Turn Delimiter** (`<|im_end|>`): **Active Loss** (teaches termination).
6. **Batch Padding Tokens**: Masked with `-100`.

This ensures gradient updates exclusively optimize the model's technical responses rather than memorizing user questions or prompt framing.

---

## 5. T4 OOM Fallback Strategy

The pipeline includes automated pre-training smoke verification (`run_smoke_forward_backward_step`). If CUDA OOM occurs:
1. Micro-batch size is enforced at `1`.
2. Sequence length is reduced to `768` or `512`.
3. LoRA rank drops to $r = 16$, $\alpha = 32$.
4. Gradient accumulation steps are increased to preserve effective batch size.
5. Garbage collection and `torch.cuda.empty_cache()` are triggered.
Full training is **never** silently routed to CPU.
