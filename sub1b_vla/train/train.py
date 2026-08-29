"""Single-GPU training entrypoint.

Native PyTorch (no Lightning/DeepSpeed): `strategy: auto` here means one device,
bf16 autocast, and gradient accumulation -- which is all a single RTX PRO 6000
needs, and keeps the failure surface small enough to debug.

    python -m sub1b_vla.train.train --config sub1b_vla/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import DrivingVLADataset, collate
from ..models.vla_agent import DualHeadDiffusionVLA
from ..utils import load_config, setup_scratch_dirs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(opt, cfg):
    t = cfg["train"]
    warmup, total = t["warmup_steps"], t["max_steps"]
    lr, min_lr = t["lr"], t["min_lr"]

    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return (min_lr + (lr - min_lr) * cos) / lr

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def resolve_amp(precision: str, device: torch.device):
    """bf16-mixed on capable CUDA; otherwise fall back and say so."""
    if precision == "bf16-mixed":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        print(f"[precision] bf16-mixed unavailable on {device.type}; running fp32.")
    return False, torch.float32


def _split_batch(batch: dict, parts: int) -> list[dict]:
    """Split a batch along dim 0 into `parts` micro-batches."""
    n = next(v.shape[0] for v in batch.values() if torch.is_tensor(v))
    size = max(1, (n + parts - 1) // parts)
    out = []
    for i in range(0, n, size):
        out.append({k: (v[i:i + size] if torch.is_tensor(v) else v[i:i + size])
                    for k, v in batch.items()})
    return out


def forward_backward(model, batch, step, accum, device, use_amp, amp_dtype,
                     oom_splits: int = 1):
    """Forward+backward with an automatic micro-batching fallback on CUDA OOM.

    A single-GPU run that OOMs on an occasional large batch should degrade to
    smaller chunks and keep the same effective batch size, not crash the job
    hours in. Returns (breakdown, splits_used) or (None, splits) if even a
    single-sample chunk will not fit.
    """
    max_splits = 8
    while oom_splits <= max_splits:
        try:
            if oom_splits == 1:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    breakdown, _ = model(batch, step=step)
                    loss = breakdown.total / accum
                if not torch.isfinite(loss):
                    return None, oom_splits
                loss.backward()
                return breakdown, oom_splits

            chunks = _split_batch(batch, oom_splits)
            first = None
            for ch in chunks:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    b, _ = model(ch, step=step)
                    loss = b.total / (accum * len(chunks))
                if not torch.isfinite(loss):
                    return None, oom_splits
                loss.backward()
                first = first or b
            return first, oom_splits
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            oom_splits *= 2
            print(f"[step {step}] CUDA OOM; retrying with {oom_splits} micro-batches.",
                  flush=True)
    raise RuntimeError(
        f"CUDA OOM persisted at {max_splits} micro-batches. Reduce train.batch_size, "
        "model.image_size, or model.num_semantic_tokens."
    )


def train(cfg, config_path: str):
    setup_scratch_dirs(cfg)
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    out_dir = Path(t["out_dir"]) / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DualHeadDiffusionVLA(cfg).to(device)
    report = model.assert_parameter_budget()
    print(report, flush=True)
    (out_dir / "param_report.txt").write_text(str(report))

    ds = DrivingVLADataset(cfg, "train", tokenizer=model.language.tokenizer)
    sampler = ds.make_sampler()
    loader = DataLoader(
        ds, batch_size=t["batch_size"], sampler=sampler, shuffle=sampler is None,
        num_workers=t["num_workers"], collate_fn=collate, drop_last=True,
        pin_memory=device.type == "cuda", persistent_workers=t["num_workers"] > 0,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"], betas=(0.9, 0.95))
    sched = build_scheduler(opt, cfg)
    use_amp, amp_dtype = resolve_amp(t.get("precision", "bf16-mixed"), device)

    accum = t.get("accumulate_grad_batches", 1)
    max_steps = t["max_steps"]
    log_path = out_dir / "train_log.jsonl"
    step = 0
    oom_splits = 1
    t0 = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            breakdown, oom_splits = forward_backward(
                model, batch, step, accum, device, use_amp, amp_dtype, oom_splits
            )
            if breakdown is None:
                # Divergence guard: drop the batch rather than poisoning the
                # optimizer state with NaN/Inf gradients.
                print(f"[step {step}] non-finite loss; skipping batch.", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue

            if (step + 1) % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
                if torch.isfinite(gn):
                    opt.step()
                else:
                    print(f"[step {step}] non-finite grad norm; step skipped.")
                opt.zero_grad(set_to_none=True)
            sched.step()

            if step % t["log_every"] == 0:
                rec = breakdown.as_log()
                rec.update({"step": step, "lr": sched.get_last_lr()[0],
                            "elapsed_s": round(time.time() - t0, 1)})
                print(json.dumps(rec), flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

            if step > 0 and step % t["ckpt_every"] == 0:
                save_checkpoint(model, cfg, out_dir / f"step_{step}.pt", step)
            step += 1

    save_checkpoint(model, cfg, out_dir / "final.pt", step)
    print(f"[done] {step} steps in {time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return out_dir


def save_checkpoint(model, cfg, path: Path, step: int):
    """Store only trainable tensors -- frozen backbones are reloaded from their
    source checkpoints, so a run's artefacts stay in the tens of MB."""
    state = {k: v for k, v in model.state_dict().items()
             if k in {n for n, p in model.named_parameters() if p.requires_grad}}
    torch.save({"model": state, "config": cfg, "step": step}, path)
    print(f"[ckpt] {path} ({sum(v.numel() for v in state.values()):,} tensors' elements)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key.path=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
