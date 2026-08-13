import os, sys, json, hashlib
sys.path.insert(0, r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none")
sys.path.insert(0, r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\site-packages")
sys.path.insert(0, r"C:\Users\Sajan\Desktop\Project XYZ\src")
import torch
from models.model import ModelConfig, SmallEnglishLLM

BASE = r"C:\Users\Sajan\Desktop\Project XYZ\checkpoints\best\checkpoint_best.pt"
SAFE_COPY = r"C:\Users\Sajan\Desktop\Project XYZ\checkpoints\best\checkpoint_best_original.pt"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

ckpt = torch.load(BASE, map_location="cpu")
print("=" * 60)
print("PHASE 2 - BASELINE FROZEN CHECKPOINT VERIFICATION")
print("=" * 60)
print(f"path           : {BASE}")
print(f"step           : {ckpt.get('step')}")
print(f"loss           : {ckpt.get('loss')}")
print(f"training_type  : {ckpt.get('training_type')}")
print(f"base_checkpoint: {ckpt.get('base_checkpoint')}")
print(f"semantic_ratio : {ckpt.get('semantic_ratio')}")
print(f"peak_lr        : {ckpt.get('peak_lr')}")
print(f"scheduler      : {ckpt.get('scheduler')}")
print(f"sha256(best)   : {sha256(BASE)}")
print(f"sha256(orig)   : {sha256(SAFE_COPY)}")
print(f"identical      : {sha256(BASE) == sha256(SAFE_COPY)}")

config = ModelConfig()
model = SmallEnglishLLM(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"params         : {params:,}")
print(f"loaded state   : OK (state_dict matches architecture)")
print(f"immutable      : checkpoint_best.pt + checkpoint_best_original.pt preserved")

# quick forward sanity check
ids = torch.tensor([[5, 12, 3, 17, 9]])
with torch.no_grad():
    logits, loss = model(ids, ids.clone())
print(f"forward+loss   : OK (logits {tuple(logits.shape)}, loss {loss.item():.4f})")
print("=" * 60)
