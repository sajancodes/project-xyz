import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    )
)

from model_config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Combined Q, K, V projection
        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=True
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=True
        )

        # Causal attention mask
        mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len,
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # [B, T, 3D]
        qkv = self.qkv(x)

        # [B, T, 3, H, head_dim]
        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim
        )

        # [B, H, T, head_dim]
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product causal attention
        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.causal_mask[
                :, :, :seq_len, :seq_len
            ],
            dropout_p=0.0,
            is_causal=False
        )

        # [B, T, D]
        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.view(
            batch_size,
            seq_len,
            d_model
        )

        return self.out_proj(attention_output)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=True
        )

        self.fc2 = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=True
        )

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x):

        # Pre-LN architecture
        x = x + self.attention(self.ln1(x))

        x = x + self.feed_forward(self.ln2(x))

        return x


class SmallEnglishLLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Learned positional embeddings
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language-model output head
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids, targets=None):

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum {self.config.max_seq_len}"
            )

        # Positions: [0, 1, 2, ..., T-1]
        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        # Token + positional embeddings
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Vocabulary logits
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            # [B, T, V] -> [B*T, V]
            logits_flat = logits.view(
                -1,
                self.config.vocab_size
            )

            targets_flat = targets.view(-1)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":

    config = ModelConfig()

    model = SmallEnglishLLM(config)

    total_parameters = count_parameters(model)

    print("=" * 60)
    print("EXPERIMENT 1 — MODEL VERIFICATION")
    print("=" * 60)

    print(f"Parameters: {total_parameters:,}")
    print(f"Parameters: {total_parameters / 1_000_000:.2f}M")

    print("-" * 60)

    # Test forward pass
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = model.to(device)

    batch_size = 2
    seq_len = config.max_seq_len

    test_input = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        device=device
    )

    test_target = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        device=device
    )

    logits, loss = model(
        test_input,
        test_target
    )

    print(f"Input shape:  {test_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}")

    print("=" * 60)