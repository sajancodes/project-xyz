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


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the Project XYZ K2-style model bridge")
    ap.add_argument("--config", default="k2_horizon_stack/config/project_xyz.json")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = build_model(cfg["model"])
    model.train()

    model_cfg = getattr(model, "config", None)
    vocab_size = getattr(model_cfg, "vocab_size", None)
    if vocab_size is None:
        # Project XYZ's current model stores vocab size in embeddings/config internals,
        # so fall back to configured tokenizer vocabulary when available.
        cfg_module_path = Path(cfg["model"]["config_file"])
        import importlib.util
        spec = importlib.util.spec_from_file_location("validate_dynamic_config", cfg_module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        config_obj = getattr(module, cfg["model"]["config_class"])(**cfg["model"].get("config_kwargs", {}))
        vocab_size = config_obj.vocab_size

    x = torch.randint(0, int(vocab_size), (args.batch_size, args.seq_len), dtype=torch.long)
    y = torch.randint(0, int(vocab_size), (args.batch_size, args.seq_len), dtype=torch.long)
    logits, loss = forward_loss(model, x, y)

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")
    if logits is None or logits.shape[:2] != x.shape:
        raise RuntimeError(f"Unexpected logits shape: {None if logits is None else tuple(logits.shape)}")

    params = sum(p.numel() for p in model.parameters())
    result = {
        "ok": True,
        "parameters": params,
        "input_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "loss_finite": math.isfinite(float(loss.item())),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
