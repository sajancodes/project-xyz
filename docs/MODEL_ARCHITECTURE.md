# MODEL ARCHITECTURE

This document explains how the model works, aimed at readers with little or no ML background.

## 1. The big picture

The model is a small **decoder-only Transformer** language model. It reads a sequence of token IDs and, for every position, predicts the probability of the *next* token. That single "predict the next token" task, repeated over enormous amounts of text, is how the model learns language.

**Architecture summary:**

| Parameter | Value |
|---|---|
| Vocabulary size | 16,000 |
| Context length (max_seq_len) | 512 |
| Embedding size (d_model) | 256 |
| Layers (n_layers) | 8 |
| Attention heads (n_heads) | 8 |
| Feed-forward dimension (d_ff) | 1,024 |
| **Total parameters** | **14,657,664 (14.66M)** |

## 2. What the numbers mean in practice

- **14.66M parameters**: this is TINY. Commercial models are hundreds of billions of parameters. Our whole model fits on a 6 GB laptop GPU. This is intentional — the project asks what semantic behavior is achievable at this scale.
- **Context 512 tokens**: at most 512 token pieces of text can influence a prediction. Anything older is invisible to the model.
- **Vocabulary 16,000**: every piece of text is reduced to IDs in 0..15,999 before entering the model.

## 3. From text to numbers (tokenizer + embeddings)

1. **Tokenizer** (`tokenizer.json`): text → list of token IDs. Byte-level BPE means even unseen words can be represented (via byte pieces). Special IDs: `<pad>`=0, unknown=1, `<bos>`=2, `<eos>`=3.
2. **Token embedding** (Linear/Embedding): each token ID → a 256-dimensional vector ("meaning").
3. **Positional embedding** (learned): each position 0..511 → a 256-dimensional vector, added to the token embedding so the model knows order.

Result: each position carries `[meaning of token] + [position]` in 256 dimensions.

## 4. The Transformer block (8 of them, in series)

Each block does (in this order, with **pre-LN** — LayerNorm *before* each sub-block):

1. **LayerNorm** — normalizes activations for stable training.
2. **Causal self-attention** (8 heads) — for each token, compute "how much should I attend to each earlier token" and gather context. **Causal** means a token can only see tokens at the same or earlier positions (masked). The implementation uses a combined QKV projection + `torch.nn.functional.scaled_dot_product_attention` with a lower-triangular mask.
3. **Residual add** — `x = x + attention(LayerNorm(x))`.
4. **LayerNorm**.
5. **Feed-forward network** — `Linear(d_model→d_ff)` → GELU → `Linear(d_ff→d_model)`. This is where per-token patterns are transformed/non-linearized.
6. **Residual add** — `x = x + ffn(LayerNorm(x))`.

The attention heads (8 × 32 dims each) let the model build connections like "Sita" ↔ "red book", which is exactly what semantic binding needs.

## 5. From hidden states to words (logits → softmax → sample)

After 8 blocks and a final LayerNorm, a **language-model head** (`Linear(d_model → 16,000)`) produces **logits** — raw scores for each of the 16,000 tokens. A **softmax** turns them into probabilities. Generation picks the next token by sampling (temperature / top-k) or by argmax (greedy).

## 6. Training mechanics (brief)

- **Loss**: cross-entropy between predicted and true next-token probabilities (per token; summed/meaned over the batch).
- **Loss masking**: in semantic/SEMV2E training, loss is computed ONLY on the `Assistant:` answer span (the model must learn to respond, not echo the question). FineWeb rows use full loss.
- **Optimizer**: AdamW. Pretraining lr 3e-4; semantic/SEMV2E runs use warmup + cosine decay (e.g. peak 1e-4, warmup 150, cosine to 10%).
- **Checkpoint**: a saved snapshot with weights + optimizer + scheduler + RNG + config + loss + parent checkpoint info. Fully resumable.

## 7. What "understanding" does and does NOT mean here

- The model does NOT store facts in a database. Training shifts probabilities.
- Evaluation must use **held-out examples** (never trained on) to know if the model *generalized* vs *memorized*.
- We measure **demonstrable task-level semantic behavior**, not consciousness or human-level comprehension.

## 8. Key source locations

| File | Contents |
|---|---|
| `src/config/model_config.py` | `ModelConfig` dataclass + parameter counter |
| `src/models/model.py` | `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `SmallEnglishLLM` |
| `src/paths.py` | central path helpers (import this!) |

---

*Related: [docs/GLOSSARY.md](GLOSSARY.md) for definitions, [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md) for how we train.*
