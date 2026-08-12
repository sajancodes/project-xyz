import torch
from model import ModelConfig, SmallEnglishLLM
from tokenizers import Tokenizer

CHECKPOINT = "checkpoint_fineweb.pt"
TOKENIZER_FILE = "tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_K = 40

print("=" * 70)
print("TESTING: 'the sun rises'")
print("=" * 70)
print(f"Device: {DEVICE}")

# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

# Load model
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

@torch.no_grad()
def generate(text, max_new_tokens=50, temperature=0.7, top_k=40):
    encoded = tokenizer.encode(text)
    input_ids = encoded.ids
    if len(input_ids) == 0:
        return text
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
    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)

# Test the specific prompt
prompt = "the sun rises"
print(f"Prompt: {prompt}")
output = generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
if output.startswith(prompt):
    continuation = output[len(prompt):]
else:
    continuation = output
print(f"Model: {continuation}")
print("-" * 70)

# Check if response contains "east"
if "east" in continuation.lower():
    print("✓ PASS: Model mentions 'east'")
else:
    print("✗ FAIL: Model does NOT mention 'east'")