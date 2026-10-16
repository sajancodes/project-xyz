from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _load_module(path: str, name: str):
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Python file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the user's model without hard-coding Project XYZ architecture."""
    cfg_module = _load_module(model_cfg["config_file"], "project_xyz_dynamic_config")
    model_module = _load_module(model_cfg["model_file"], "project_xyz_dynamic_model")

    config_cls = getattr(cfg_module, model_cfg["config_class"])
    model_cls = getattr(model_module, model_cfg["model_class"])
    config_obj = config_cls(**model_cfg.get("config_kwargs", {}))

    sig = inspect.signature(model_cls)
    if len(sig.parameters) == 0:
        model = model_cls()
    else:
        model = model_cls(config_obj)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{model_cfg['model_class']} did not produce torch.nn.Module")
    return model


def forward_loss(model: torch.nn.Module, input_ids: torch.Tensor, target_ids: torch.Tensor):
    """Normalize common model output styles to (logits, scalar loss)."""
    try:
        output = model(input_ids, target_ids)
    except TypeError:
        output = model(input_ids)

    logits = None
    loss = None

    if isinstance(output, tuple):
        if len(output) >= 1:
            logits = output[0]
        if len(output) >= 2:
            loss = output[1]
    elif hasattr(output, "logits"):
        logits = output.logits
        loss = getattr(output, "loss", None)
    elif torch.is_tensor(output):
        logits = output
    else:
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    if loss is None:
        if logits is None:
            raise RuntimeError("Model returned neither logits nor loss")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

    return logits, loss
