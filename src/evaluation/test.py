import os
import sys
import argparse
from datetime import datetime

# Ensure uv Python packages are findable (torch etc.)
uv_base = r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none"
if uv_base not in sys.path:
    sys.path.insert(0, uv_base)
if uv_base + r"\Lib\site-packages" not in sys.path:
    sys.path.insert(0, uv_base + r"\Lib\site-packages")

import torch
from tokenizers import Tokenizer

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import TOKENIZER_PATH, LATEST_CHECKPOINT

from model import ModelConfig, SmallEnglishLLM


# ============================================================
# CHAT TEST — wraps input in the chat format the model was trained on,
# and decodes ONLY the newly generated tokens (no echo of the prompt).
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = LATEST_CHECKPOINT
TOKENIZER_FILE = TOKENIZER_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=CHECKPOINT)
parser.add_argument("--plain", action="store_true",
                    help="do NOT wrap input in chat format (raw continuation)")
args = parser.parse_args()
CHECKPOINT = args.checkpoint

# Verify checkpoint exists
if not os.path.exists(CHECKPOINT):
    print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
    sys.exit(1)

DEVICE = "cpu"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 40


print("=" * 70)
print("CHAT TEST")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Chat wrapper: {'OFF (plain continuation)' if args.plain else 'ON (User:/Assistant:)'}")

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
config = ModelConfig()
model = SmallEnglishLLM(config).to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
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


@torch.no_grad()
def generate(text, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    encoded = tokenizer.encode(text)
    input_ids = encoded.ids
    prompt_len = len(input_ids)
    if prompt_len == 0:
        return ""
    input_ids = input_ids[-config.max_seq_len:]
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        x_cond = x[:, -config.max_seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
        if next_token.item() == 3:
            break

    generated_ids = x[0, prompt_len:].tolist()
    return tokenizer.decode(generated_ids)


print("Type a sentence or question. Type 'exit' to quit.\n")
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

    model_input = prompt if args.plain else f"User: {prompt}\nAssistant:"
    output = generate(model_input)

    print("Model:", output.strip())
    print()
    print("-" * 70)