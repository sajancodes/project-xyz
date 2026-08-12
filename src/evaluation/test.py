import torch
from model import ModelConfig, SmallEnglishLLM
from tokenizers import Tokenizer


# ============================================================
# EXPERIMENT 1 — NATURAL LANGUAGE GENERATION TEST
# ============================================================

CHECKPOINT = "checkpoint-mixed-2k.pt"
TOKENIZER_FILE = "tokenizer.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40


print("=" * 70)
print("EXPERIMENT 1 — NATURAL LANGUAGE GENERATION TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

config = ModelConfig()

model = SmallEnglishLLM(config).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

# Handle checkpoints that store the model under different names
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

if "step" in checkpoint:
    print(f"Training step: {checkpoint['step']}")

if "loss" in checkpoint:
    print(f"Training loss: {checkpoint['loss']:.4f}")

print("=" * 70)
print()
print("Type a sentence or prompt.")
print("The model will continue it.")
print("Type 'exit' to quit.")
print()


# ============================================================
# TOKEN GENERATION
# ============================================================

@torch.no_grad()
def generate(text, max_new_tokens=100, temperature=0.8, top_k=40):

    encoded = tokenizer.encode(text)

    input_ids = encoded.ids

    if len(input_ids) == 0:
        return text

    # Keep only the model's context window
    input_ids = input_ids[-config.max_seq_len:]

    x = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        # Keep context within 512 tokens
        x_cond = x[:, -config.max_seq_len:]

        logits, _ = model(x_cond)

        # Get logits for final token
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k is not None:

            values, indices = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            filtered = torch.full_like(
                logits,
                float("-inf")
            )

            filtered.scatter_(
                1,
                indices,
                values
            )

            logits = filtered

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        x = torch.cat(
            [x, next_token],
            dim=1
        )

        # Stop at EOS
        if next_token.item() == 3:
            break

    generated_ids = x[0].tolist()

    return tokenizer.decode(generated_ids)


# ============================================================
# INTERACTIVE TEST
# ============================================================

while True:

    try:
        prompt = input("You: ")

    except KeyboardInterrupt:
        print("\nExiting.")
        break

    if prompt.lower().strip() == "exit":
        break

    if not prompt.strip():
        continue

    print()
    print("Model:")

    output = generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    # Remove the original prompt from display
    if output.startswith(prompt):
        continuation = output[len(prompt):]
    else:
        continuation = output

    print(continuation)
    print()
    print("-" * 70)