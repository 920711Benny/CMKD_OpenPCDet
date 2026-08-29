"""Validate a GPU box and size the run BEFORE committing hours to it.

Runs the real model through real forward/backward steps on the actual device,
finds the largest batch that fits, measures throughput, and projects wall clock.
A multi-hour run that dies at hour three on an OOM, or crawls because bf16
silently fell back, is the expensive failure this exists to prevent.

    python -m sub1b_vla.tools.gpu_preflight --config sub1b_vla/configs/default.yaml \\
        --dataset-frames 3000000

Exit code is non-zero if anything would break or badly slow a long run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from ..utils import load_config, setup_device


def _fake_batch(cfg, bs, device):
    m, d = cfg["model"], cfg["data"]
    s = m["image_size"]
    tl = d.get("max_text_len", 96)
    b = {
        "image": torch.randn(bs, 3, s, s, device=device),
        "waypoints": torch.randn(bs, m["pred_len"], 2, device=device) * 5,
        "speed": torch.rand(bs, device=device) * 30,
        "target_point": torch.randn(bs, 2, device=device),
        "command": torch.randint(0, 7, (bs,), device=device),
        "text_ids": torch.randint(3, 100, (bs, tl), device=device),
        "text_mask": torch.ones(bs, tl, dtype=torch.long, device=device),
        "intent_id": torch.randint(0, 8, (bs,), device=device),
        "has_waypoints": torch.ones(bs, device=device),
    }
    b["text_labels"] = b["text_ids"].clone()
    return b


def _step(model, opt, batch, device, use_amp, amp_dtype):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        breakdown, _ = model(batch, step=0)
    breakdown.total.backward()
    opt.step()


def find_max_batch(model, opt, cfg, device, use_amp, amp_dtype, start=1, cap=256):
    """Largest power-of-two batch that completes a full training step."""
    best, bs = 0, start
    while bs <= cap:
        try:
            _step(model, opt, _fake_batch(cfg, bs, device), device, use_amp, amp_dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            best = bs
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    return best


def measure(model, opt, cfg, device, bs, use_amp, amp_dtype, iters=8, warmup=3):
    batch = _fake_batch(cfg, bs, device)
    for _ in range(warmup):
        _step(model, opt, batch, device, use_amp, amp_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _step(model, opt, batch, device, use_amp, amp_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0
    return (iters * bs) / dt, peak


def fmt_hours(h):
    if h < 1:
        return f"{h * 60:.0f} min"
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} days"


def main():
    from ..models.vla_agent import DualHeadDiffusionVLA  # noqa: PLC0415
    from ..train.train import resolve_amp  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-frames", type=int, default=None)
    ap.add_argument("--max-batch-cap", type=int, default=256)
    ap.add_argument("--skip-batch-search", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device, hw = setup_device(cfg)
    t = cfg["train"]
    problems: list[str] = []
    warnings_: list[str] = []

    w = 76
    print("=" * w)
    print("GPU PREFLIGHT".center(w))
    print("=" * w)
    for k, v in hw.items():
        print(f"{k:<26}{v}")

    if device.type != "cuda":
        problems.append(
            "No CUDA device visible. This model is meant to be trained on a GPU; "
            "CPU training is orders of magnitude slower and is not a supported path.")

    use_amp, amp_dtype = resolve_amp(t.get("precision", "bf16-mixed"), device)
    print(f"{'precision requested':<26}{t.get('precision')}")
    print(f"{'autocast active':<26}{use_amp} ({amp_dtype})")
    if t.get("precision") == "bf16-mixed" and device.type == "cuda" and not use_amp:
        problems.append("bf16-mixed requested but unsupported here; it silently fell "
                        "back to fp32, which will roughly halve throughput.")

    model = DualHeadDiffusionVLA(cfg).to(device)
    rep = model.assert_parameter_budget()
    print(f"{'total params':<26}{rep.total:,} ({rep.total / 1e9:.3f} B)")
    print(f"{'trainable params':<26}{rep.trainable:,} ({rep.trainable / 1e6:.1f} M)")
    if model.encoder.spatial_spec.is_stub or model.language.is_stub:
        problems.append("Backbones fell back to random stubs -- the checkpoints could "
                        "not be loaded. Fix HF access before training; results would "
                        "be meaningless.")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.1))

    print("-" * w)
    configured = int(t["batch_size"])
    max_bs = configured
    if device.type == "cuda" and not args.skip_batch_search:
        max_bs = find_max_batch(model, opt, cfg, device, use_amp, amp_dtype,
                                cap=args.max_batch_cap)
        print(f"{'max batch that fits':<26}{max_bs}")
        if max_bs == 0:
            problems.append("Even batch size 1 did not fit. Reduce model.image_size "
                            "or enable train.gradient_checkpointing.")
        elif max_bs < configured:
            problems.append(f"Configured batch_size={configured} does NOT fit "
                            f"(max {max_bs}). Lower it and raise "
                            f"accumulate_grad_batches to keep the effective batch.")
        elif max_bs >= configured * 4:
            warnings_.append(f"batch_size={configured} leaves headroom (up to {max_bs}). "
                             "A larger micro-batch with less accumulation would be faster.")

    bs = min(configured, max_bs) if max_bs else configured
    sps, peak = measure(model, opt, cfg, device, bs, use_amp, amp_dtype)
    print(f"{'measured throughput':<26}{sps:.2f} samples/s at batch {bs}")
    if device.type == "cuda":
        print(f"{'peak memory':<26}{peak:.2f} GB of {hw.get('total_memory_gb')} GB")

    accum = int(t.get("accumulate_grad_batches", 1))
    print(f"{'effective batch':<26}{bs * accum}")

    workers = int(t.get("num_workers", 0))
    import os  # noqa: PLC0415
    cores = os.cpu_count() or 1
    if workers > cores:
        warnings_.append(f"num_workers={workers} exceeds {cores} cores; oversubscription "
                         "usually slows the input pipeline.")
    if workers == 0 and device.type == "cuda":
        warnings_.append("num_workers=0 will starve the GPU on image decoding.")

    scratch = Path(os.environ.get("SUB1B_SCRATCH", "./scratch"))
    try:
        free_gb = shutil.disk_usage(scratch if scratch.exists() else Path(".")).free / 1024 ** 3
        print(f"{'scratch free space':<26}{free_gb:.1f} GB")
        if free_gb < 50:
            warnings_.append(f"Only {free_gb:.0f} GB free where checkpoints will be "
                             "written; a long run may fill it.")
    except OSError:
        pass

    projection = None
    if args.dataset_frames:
        epochs = int(t.get("max_epochs", 0) or 0)
        if epochs:
            hours = (args.dataset_frames * epochs) / max(sps, 1e-9) / 3600.0
            projection = {"dataset_frames": args.dataset_frames, "epochs": epochs,
                          "hours": hours}
            print("-" * w)
            print(f"{'PROJECTED WALL CLOCK':<26}{fmt_hours(hours)} "
                  f"({args.dataset_frames:,} frames x {epochs} epochs)")

    print("-" * w)
    for msg in warnings_:
        print(f"WARNING  {msg}")
    for msg in problems:
        print(f"BLOCKER  {msg}")
    verdict = "READY" if not problems else "NOT READY"
    print(f"\nVERDICT: {verdict}")
    print("=" * w)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"hardware": hw, "throughput_samples_per_s": sps, "batch": bs,
             "max_batch": max_bs, "peak_memory_gb": peak, "projection": projection,
             "warnings": warnings_, "blockers": problems}, indent=2, default=str))
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
