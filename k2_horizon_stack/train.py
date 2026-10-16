from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from checkpointing import load_checkpoint, prune_checkpoints, save_checkpoint, write_run_manifest
from model_adapter import build_model, forward_loss


class TokenBlockDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, seq_len: int, dtype: str = "auto"):
        self.path = Path(path)
        meta_path = Path(str(self.path) + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if dtype == "auto":
            dtype = meta.get("dtype", "uint16")
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.path, mode="r", dtype=self.dtype)
        self.seq_len = int(seq_len)
        self.n_blocks = max(0, (len(self.tokens) - 1) // self.seq_len)
        if self.n_blocks < 1:
            raise RuntimeError(f"Dataset {path} is too small for seq_len={seq_len}")

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = int(idx) * self.seq_len
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, local_rank, world, device


def cosine_lambda(step: int, warmup: int, total: int, min_ratio: float):
    if step < warmup:
        return max(1e-8, step / max(1, warmup))
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, precision: str, max_batches: int):
    model.eval()
    losses = []
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = forward_loss(model, x, y)
        losses.append(loss.detach())
    model.train()
    if not losses:
        return float("nan")
    value = torch.stack(losses).mean()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
    return float(value.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    run_dir = Path(cfg["output"]["run_dir"])

    distributed, local_rank, world, device = init_dist()
    rank = torch.distributed.get_rank() if distributed else 0
    is_main = rank == 0

    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this training stack. Install a CUDA-enabled PyTorch build.")

    seed = int(cfg.get("seed", 42)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_ds = TokenBlockDataset(data_cfg["train_bin"], data_cfg["seq_len"], data_cfg.get("dtype", "auto"))
    valid_ds = None
    if data_cfg.get("valid_bin") and Path(data_cfg["valid_bin"]).exists():
        valid_ds = TokenBlockDataset(data_cfg["valid_bin"], data_cfg["seq_len"], data_cfg.get("dtype", "auto"))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=data_cfg.get("shuffle_blocks", True)) if distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg["micro_batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None and data_cfg.get("shuffle_blocks", True),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_ds, shuffle=False) if distributed else None
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=train_cfg["micro_batch_size"], sampler=valid_sampler, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(cfg["model"]).to(device)
    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_lambda(s, train_cfg["warmup_steps"], train_cfg["steps"], train_cfg["min_lr_ratio"]),
    )

    precision = train_cfg.get("precision", "bf16").lower()
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        if is_main:
            print("BF16 not supported; falling back to FP16")
        precision = "fp16"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    if is_main:
        write_run_manifest(run_dir, cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"model parameters: {params:,}")
        print(f"world size: {world} | device: {device} | precision: {precision}")

    start_step = 0
    latest = run_dir / "latest.pt"
    if train_cfg.get("resume", True) and latest.exists():
        start_step = load_checkpoint(latest, model, optimizer, scheduler, scaler, map_location=device)
        if is_main:
            print(f"resumed from step {start_step}")

    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    loader_iter = iter(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    wall_start = time.time()

    for step in range(start_step + 1, int(train_cfg["steps"]) + 1):
        step_loss = 0.0
        step_tokens = 0
        for micro in range(grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step)
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync_context = model.no_sync() if distributed and micro < grad_accum - 1 else torch.enable_grad()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=precision in {"bf16", "fp16"}):
                    _, loss = forward_loss(model, x, y)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            step_loss += float(loss.detach().item())
            step_tokens += int(x.numel()) * world

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if is_main and (step == 1 or step % int(train_cfg["log_every"]) == 0):
            elapsed = max(1e-6, time.time() - wall_start)
            tps = (step_tokens * max(1, step - start_step)) / elapsed
            print(f"step={step} loss={step_loss:.5f} lr={scheduler.get_last_lr()[0]:.3e} tok/s={tps:,.0f} vram={torch.cuda.max_memory_allocated(device)/2**30:.2f}GiB")

        if valid_loader is not None and step % int(train_cfg["eval_every"]) == 0:
            val_loss = evaluate(model, valid_loader, device, precision, int(train_cfg["eval_batches"]))
            if is_main:
                ppl = math.exp(min(20.0, val_loss))
                print(f"eval step={step} loss={val_loss:.5f} ppl={ppl:.3f}")

        if step % int(train_cfg["checkpoint_every"]) == 0:
            if distributed:
                torch.distributed.barrier()
            if is_main:
                path = save_checkpoint(run_dir, step, model, optimizer, scheduler, cfg, scaler)
                prune_checkpoints(run_dir, int(train_cfg.get("keep_last_checkpoints", 3)))
                print(f"checkpoint: {path}")

    if is_main:
        save_checkpoint(run_dir, int(train_cfg["steps"]), model, optimizer, scheduler, cfg, scaler)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
