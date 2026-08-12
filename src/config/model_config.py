# Experiment 1 — Model Configuration
# Small English causal language model
# Target: 10–20 million parameters

from dataclasses import dataclass


@dataclass
class ModelConfig:

    # Vocabulary / sequence
    vocab_size: int = 16_000
    max_seq_len: int = 512

    # Transformer
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8

    # Feed-forward network
    d_ff: int = 1_024

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3


config = ModelConfig()


def count_parameters(config):
    """
    Calculate the number of trainable parameters
    for our planned decoder-only Transformer.
    """

    V = config.vocab_size
    D = config.d_model
    L = config.n_layers
    F = config.d_ff
    S = config.max_seq_len

    # Token embedding
    token_embedding = V * D

    # Positional embedding
    position_embedding = S * D

    # Per Transformer block:
    #
    # Attention:
    # Q, K, V projections = 3 * D * D
    # Output projection   = D * D
    #
    # FFN:
    # D -> F
    # F -> D
    #
    # Biases are included for simplicity.

    attention_weights = 4 * D * D
    attention_biases = 4 * D

    ffn_weights = 2 * D * F
    ffn_biases = F + D

    # Two LayerNorms per block
    layernorm = 4 * D

    block_parameters = (
        attention_weights
        + attention_biases
        + ffn_weights
        + ffn_biases
        + layernorm
    )

    transformer_blocks = L * block_parameters

    # Final LayerNorm
    final_layernorm = 2 * D

    # Output language-model head
    lm_head = V * D + V

    total = (
        token_embedding
        + position_embedding
        + transformer_blocks
        + final_layernorm
        + lm_head
    )

    return total


if __name__ == "__main__":

    total = count_parameters(config)

    print("=" * 50)
    print("EXPERIMENT 1 — MODEL CONFIGURATION")
    print("=" * 50)

    print(f"Vocabulary size : {config.vocab_size:,}")
    print(f"Context length  : {config.max_seq_len}")
    print(f"Embedding size  : {config.d_model}")
    print(f"Layers          : {config.n_layers}")
    print(f"Attention heads : {config.n_heads}")
    print(f"FFN dimension   : {config.d_ff}")

    print("-" * 50)

    print(f"Total parameters: {total:,}")
    print(f"Parameters (M)  : {total / 1_000_000:.2f}M")

    if 10_000_000 <= total <= 20_000_000:
        print("Status          : ✓ Within 10–20M target")
    else:
        print("Status          : ✗ Outside 10–20M target")

    print("=" * 50)