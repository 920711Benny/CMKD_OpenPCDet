"""Project training wall-clock from MEASURED throughput.

Answers "how long will a real run actually take?" without guessing: it times a
real forward+backward at the configured batch size on whatever device it runs
on, then extrapolates to the dataset size and epoch count. A projection for
hardware this box does not have is only produced when you supply that machine's
throughput explicitly -- the tool will not invent a number for a GPU it has
never seen.

    # measure here, project for this device
    python -m sub1b_vla.tools.compute_budget --config sub1b_vla/configs/default.yaml \
        --dataset-frames 3000000

    # project for a machine you measured separately
    python -m sub1b_vla.tools.compute_budget --config ... --dataset-frames 3000000 \
        --samples-per-s 46.0 --device-label "RTX PRO 6000"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RECIPE = Path("baselines/simlingo_training_recipe.json")


def measure_throughput(cfg, device, iters=6, warmup=2) -> float:
    """Samples/second for a full training step at the configured batch size."""
    from ..models.vla_agent import DualHeadDiffusionVLA  # noqa: PLC0415

    t = cfg["train"]
    m = cfg["model"]
    bs = int(t["batch_size"])
    model = DualHeadDiffusionVLA(cfg).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    s = m["image_size"]
    batch = {
        "image": torch.randn(bs, 3, s, s, device=device),
        "waypoints": torch.randn(bs, m["pred_len"], 2, device=device) * 5,
        "speed": torch.rand(bs, device=device) * 30,
        "target_point": torch.randn(bs, 2, device=device),
        "command": torch.randint(0, 7, (bs,), device=device),
        "text_ids": torch.randint(3, 100, (bs, cfg["data"].get("max_text_len", 96)), device=device),
        "text_mask": torch.ones(bs, cfg["data"].get("max_text_len", 96), dtype=torch.long,
                                device=device),
        "intent_id": torch.randint(0, 8, (bs,), device=device),
        "has_waypoints": torch.ones(bs, device=device),
    }
    batch["text_labels"] = batch["text_ids"].clone()

    def one_step():
        opt.zero_grad(set_to_none=True)
        breakdown, _ = model(batch, step=0)
        breakdown.total.backward()
        opt.step()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (iters * bs) / (time.perf_counter() - t0)


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} days"


def project(cfg, frames: int, samples_per_s: float, label: str) -> dict:
    t = cfg["train"]
    epochs = int(t.get("max_epochs", 0) or 0)
    if not epochs:
        raise ValueError("train.max_epochs must be set to project an epoch-based budget.")
    total_samples = frames * epochs
    seconds = total_samples / max(samples_per_s, 1e-9)
    eff_batch = int(t["batch_size"]) * int(t.get("accumulate_grad_batches", 1))
    return {
        "device": label,
        "samples_per_s": samples_per_s,
        "dataset_frames": frames,
        "epochs": epochs,
        "total_samples": total_samples,
        "optimizer_steps": total_samples // max(1, eff_batch),
        "effective_batch": eff_batch,
        "wall_clock_hours": seconds / 3600.0,
    }


def main():
    from ..utils import load_config  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-frames", type=int, required=True,
                    help="number of training frames in YOUR extracted dataset copy")
    ap.add_argument("--samples-per-s", type=float, default=None,
                    help="skip measurement and project from this throughput")
    ap.add_argument("--device-label", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.samples_per_s is not None:
        sps = args.samples_per_s
        label = args.device_label or "supplied throughput"
        measured = False
    else:
        sps = measure_throughput(cfg, device)
        label = args.device_label or (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
        measured = True

    proj = project(cfg, args.dataset_frames, sps, label)
    proj["throughput_measured_here"] = measured

    w = 74
    print("=" * w)
    print("TRAINING COMPUTE BUDGET".center(w))
    print("=" * w)
    src = "measured on this machine" if measured else "supplied by caller (not measured here)"
    print(f"{'device':<28}{label}")
    print(f"{'throughput':<28}{sps:.2f} samples/s   ({src})")
    print(f"{'dataset frames':<28}{args.dataset_frames:,}")
    print(f"{'epochs':<28}{proj['epochs']}")
    print(f"{'effective batch':<28}{proj['effective_batch']}")
    print(f"{'optimizer steps':<28}{proj['optimizer_steps']:,}")
    print("-" * w)
    print(f"{'PROJECTED WALL CLOCK':<28}{fmt_hours(proj['wall_clock_hours'])}")
    print("-" * w)

    if RECIPE.exists():
        r = json.loads(RECIPE.read_text())
        hw, opt = r["hardware"], r["optimisation"]
        print("SimLingo reference (quoted from RenzKa/simlingo, see file for provenance):")
        print(f"  {'gpus':<26}{hw['gpus']}")
        print(f"  {'slurm time limit':<26}{hw['slurm_time_limit']}  "
              f"(<= {hw['gpu_hours_upper_bound']} GPU-hours)")
        print(f"  {'epochs / eff. batch':<26}{opt['max_epochs']} / {opt['effective_batch']}")
        print(f"  {'lr / wd':<26}{opt['lr']} / {opt['weight_decay']}")
        print(f"  {'strategy / precision':<26}{opt['strategy']} / {opt['precision']}")
        print(f"  note: {hw['note']}")
    print("=" * w)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(proj, indent=2))


if __name__ == "__main__":
    main()
