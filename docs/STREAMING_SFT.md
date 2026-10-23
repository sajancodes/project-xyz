# Track B: Experiment V2 — 16K Streaming SFT Pipeline

## 1. Overview & Architectural Motivation

In **Experiment V1**, Qwen3-4B-Instruct-2507 was fine-tuned on a tiny seed dataset (~12 training examples). While the technical training loop and loss masking functioned properly, training on such a minimal dataset over-fit to specific prompts and caused measurable regression on broad benchmark categories.

**Experiment V2** replaces the tiny dataset path with a high-volume, progressive streaming Supervised Fine-Tuning (SFT) pipeline consuming **~16,000 unique examples** across 4 primary domains without downloading massive multi-gigabyte datasets to disk.

---

## 2. Upstream Streaming Datasets & Quotas

All upstream datasets are consumed on-the-fly using Hugging Face `datasets.load_dataset(..., streaming=True)` with buffer shuffling (`buffer_size=10000`) and deterministic quota capping.

| Source | Hugging Face Dataset ID | Subset / Config | Target Quota | Target % | Focus Area |
|---|---|---|---:|---:|---|
| **OpenCoder** | `OpenCoder-LLM/opc-sft-stage2` | `educational_instruct` | 7,200 | 45.0% | Educational coding instruction pairs with unit tests |
| **BigCode** | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | default | 4,000 | 25.0% | Execution-validated coding instruction/response pairs |
| **SmolTalk** | `HuggingFaceTB/smoltalk` | `smol-magpie-ultra` | 3,200 | 20.0% | General reasoning & instruction preservation |
| **Cybersecurity** | `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | default | 1,600 | 10.0% | Enterprise-grade defensive security dialogues |
| **TOTAL** | -- | -- | **16,000** | **100.0%** | Comprehensive Multi-Capability Mixture |

### Anti-Repetition & Exhaustion Guarantees
- No dataset is endlessly cycled.
- When an upstream source reaches its quota or exhausts its stream, no more samples are drawn from that source.
- Small datasets are never repeatedly looped. If a source is depleted, remaining quota is proportionally redistributed.

---

## 3. Streaming Hygiene & Data Pipeline

```
Upstream Hugging Face Stream (streaming=True)
                 ↓
Streaming Shuffle (buffer_size=10,000, seed=42)
                 ↓
Dataset-Specific Normalizer (common chat schema)
                 ↓
Quality Filter (reject empty, placeholders, syntax errors, loops)
                 ↓
Benchmark Decontamination (reject matches with 17 multi-domain + 60 coding tasks)
                 ↓
Exact Content Deduplication (SHA-256 prompt+response memory set)
                 ↓
Deterministic Content Hash Partitioning (90% train, 5% val, 5% test)
                 ↓
Assistant-Only Loss Masking (MAX_SEQ_LENGTH = 2048)
                 ↓
PyTorch IterableDataset (Interleaved weighted round-robin)
```

### Deterministic Content-Hash Partitioning
Because upstream streaming datasets expose only a single split, examples are partitioned deterministically using:
$$\text{bucket} = \frac{\text{int}(\text{MD5}(\text{salt} \mathbin{\Vert} \text{instruction} \mathbin{\Vert} \text{response}))}{2^{128}}$$
- $\text{bucket} < 0.90 \implies$ **Train**
- $0.90 \le \text{bucket} < 0.95 \implies$ **Validation**
- $\text{bucket} \ge 0.95 \implies$ **Test**
This guarantees that an example always falls into the exact same split across training runs.

---

## 4. Finite Evaluation Datasets

To ensure Trainer evaluation terminates deterministically with exact loss metrics, finite subsets are materialized into JSONL before training:
- **Validation Dataset**: 600 unique records (`data/qwen4b/processed/stream_val.jsonl`)
- **Test Dataset**: 400 unique records (`data/qwen4b/processed/stream_test.jsonl`)

The training stream remains fully dynamic and progressive via `StreamSFTDataset`.

---

## 5. Experiment V2 Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` | Official Hugging Face weights |
| Quantization | 4-bit NormalFloat (NF4) | Double quant enabled, FP16 compute |
| LoRA Rank ($r$) | 16 | Conservative rank preventing overfitting on 16K samples |
| LoRA Alpha ($\alpha$) | 32 | $2 \times r$ scaling |
| LoRA Dropout | 0.05 | Regularization |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full linear projection coverage |
| Max Sequence Length | **2048** | Complete code implementations & explanations |
| Micro-Batch Size | 1 | Zero OOM on NVIDIA T4 |
| Gradient Accumulation | 16 | Effective batch size = 16 |
| Learning Rate | **$5.0 \times 10^{-5}$** | Stable rate for high-volume SFT |
| Learning Rate Scheduler | Cosine with 5% warmup | Smooth decay |
| **Max Steps** | **1,000** | $1,000 \times 16 = 16,000$ example exposures (1 full pass) |
| Optimizer | `paged_adamw_8bit` | 8-bit optimizer with page-locked memory |
| Evaluation / Save Frequency | Every 100 steps | 10 evaluations across the training run |
| Early Stopping Patience | 3 evaluations | Stops if validation loss plateaus |

---

## 6. Live Training Visibility & Callbacks

During training, `LiveTrainingProgressCallback` outputs real-time metrics:
```
Step [======>-------------]  30.0% ( 300/1000) | Loss: 0.7421 | LR: 4.8e-5 | Time: 00:45:12 | ETA: 01:45:28 | VRAM: 7.4G/8.2G
```
Every 100 steps, a full source intake breakdown is printed:
```
--- Live Source Intake Distribution ---
  - opencoder         :  2160 /  7200 accepted
  - bigcode           :  1200 /  4000 accepted
  - smoltalk          :   960 /  3200 accepted
  - cybersecurity     :   480 /  1600 accepted
---------------------------------------
```
All checkpoints automatically stream to Google Drive at `MyDrive/ProjectXYZ/qwen3-4b-coding-stream-v2/checkpoints/`.
