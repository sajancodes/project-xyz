from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_adapter import build_model, forward_loss
from train import TokenBlockDataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Evaluate a Project XYZ checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batches", type=int, default=100)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    ds = TokenBlockDataset(args.data, cfg["data"]["seq_len"], cfg["data"].get("dtype", "auto"))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["training"]["micro_batch_size"], shuffle=False)

    precision = cfg["training"].get("precision", "bf16").lower()
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    losses = []
    tokens = 0
    for i, (x, y) in enumerate(loader):
        if i >= args.batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda" and precision in {"bf16", "fp16"}):
            _, loss = forward_loss(model, x, y)
        losses.append(float(loss.item()))
        tokens += int(y.numel())

    mean_loss = sum(losses) / max(1, len(losses))
    result = {
        "project": "Project XYZ",
        "checkpoint": args.checkpoint,
        "step": int(payload.get("step", -1)),
        "batches": len(losses),
        "tokens": tokens,
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
