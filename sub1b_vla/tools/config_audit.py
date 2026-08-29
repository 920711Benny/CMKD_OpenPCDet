"""Audit a training config against this project's constraints.

Checks the things that are expensive to discover late: a parameter budget that
does not fit, a dataloader that will starve the GPUs, an effective batch that no
longer matches the learning rate it was tuned for, and features that are
configured but switched off.

When the parameter budget does not fit, it enumerates the backbone pairs that
DO fit rather than just reporting failure.

    python -m sub1b_vla.tools.config_audit --config sub1b_vla/configs/carlavla_v3.yaml
"""
from __future__ import annotations

import argparse
import itertools
import os

from ..utils import load_config
from .param_budget import (
    QWEN2_05B, VISION_SPECS, build_report, lora_param_count, vit_param_count,
)

# SimLingo's published reference, for comparison rather than enforcement.
SIMLINGO_EFFECTIVE_BATCH = 48
SIMLINGO_LR = 3.0e-5


class Finding:
    def __init__(self, level: str, title: str, detail: str, fix: str = ""):
        self.level, self.title, self.detail, self.fix = level, title, detail, fix


def fitting_backbone_pairs(cfg: dict, fixed_other: int) -> list[tuple[str, str, int]]:
    """Backbone pairs whose total leaves the model under the limit."""
    limit = int(cfg.get("param_limit", 1_000_000_000))
    spatial = [k for k in VISION_SPECS if "dino" in k and not k.startswith("pretrained/")]
    semantic = [k for k in VISION_SPECS if "siglip" in k and not k.startswith("pretrained/")]
    out = []
    for a, b in itertools.product(sorted(spatial), sorted(semantic)):
        total = vit_param_count(VISION_SPECS[a]) + vit_param_count(VISION_SPECS[b]) + fixed_other
        if total < limit:
            out.append((a, b, total))
    return sorted(out, key=lambda r: -r[2])


def audit(cfg: dict) -> list[Finding]:
    f: list[Finding] = []
    m, d, t = cfg["model"], cfg["data"], cfg["train"]

    # ---- parameter budget
    rep = build_report(cfg)
    limit = rep["limit"]
    if not rep["within_budget"]:
        over = rep["total"] - limit
        vision = sum(n for name, _, n in rep["rows"] if "backbone" in name)
        fixed_other = rep["total"] - vision
        options = fitting_backbone_pairs(cfg, fixed_other)
        lines = [f"{a} + {b} -> {n / 1e9:.4f} B" for a, b, n in options[:5]]
        f.append(Finding(
            "BLOCKER",
            f"Parameter budget exceeded: {rep['total'] / 1e9:.4f} B >= {limit / 1e9:.3f} B",
            f"Over by {over / 1e9:.4f} B ({over / limit:.0%}). The two vision backbones "
            f"account for {vision / 1e9:.4f} B of it.",
            "Backbone pairs that fit, largest first:\n      " + "\n      ".join(lines)))
    else:
        f.append(Finding("OK", f"Parameter budget: {rep['total'] / 1e9:.4f} B < "
                               f"{limit / 1e9:.3f} B", "", ""))

    # ---- dataloader starvation
    workers = int(t.get("num_workers", 0) or 0)
    devices = int(t.get("devices", 1) or 1)
    cores = os.cpu_count() or 1
    if workers == 0:
        f.append(Finding(
            "BLOCKER" if devices > 1 else "WARNING",
            f"num_workers=0 with devices={devices}",
            "The dataloader runs in the training process, so every JPEG decode and "
            "resize blocks the step. At 1024x512 with two resizes per frame this is "
            "usually the single largest throughput loss, and it scales with the "
            "number of GPUs being starved.",
            f"Set num_workers to about {min(max(4, cores // max(devices, 1)), 16)} "
            "and keep persistent_workers on."))
    elif workers > cores:
        f.append(Finding("WARNING", f"num_workers={workers} exceeds {cores} cores",
                         "Oversubscription usually slows the input pipeline.", ""))
    else:
        f.append(Finding("OK", f"num_workers={workers}", "", ""))

    # ---- effective batch vs learning rate
    eff = int(t["batch_size"]) * int(t.get("accumulate_grad_batches", 1)) * devices
    lr = float(t["lr"])
    if eff == SIMLINGO_EFFECTIVE_BATCH:
        f.append(Finding("OK", f"Effective batch {eff} matches SimLingo "
                               f"(lr {lr:g})", "", ""))
    else:
        # A smaller batch is fine PROVIDED the lr was scaled for it. Flagging
        # every batch != 48 regardless of lr would train people to ignore this.
        ratio = SIMLINGO_EFFECTIVE_BATCH / max(eff, 1)
        linear = SIMLINGO_LR / ratio
        sqrt_scaled = SIMLINGO_LR / (ratio ** 0.5)
        lo, hi = sorted((linear, sqrt_scaled))
        if lo * 0.8 <= lr <= hi * 1.25:
            rule = "sqrt" if abs(lr - sqrt_scaled) < abs(lr - linear) else "linear"
            f.append(Finding(
                "OK",
                f"Effective batch {eff} with lr {lr:g} compensated ({rule} rule)",
                f"Smaller than SimLingo's {SIMLINGO_EFFECTIVE_BATCH}, but the lr "
                f"was scaled to match, so this is a deliberate trade rather than "
                f"an uncompensated mismatch. State the batch alongside any "
                f"comparison against SimLingo.", ""))
        else:
            f.append(Finding(
                "WARNING",
                f"Effective batch {eff} != SimLingo's {SIMLINGO_EFFECTIVE_BATCH}, "
                f"and lr {lr:g} is not scaled for it",
                f"lr={lr:g} is SimLingo's value for batch "
                f"{SIMLINGO_EFFECTIVE_BATCH}. At batch {eff} the gradient noise is "
                f"about {ratio:.1f}x larger per step, so the same lr is a "
                f"different optimisation and a delta against SimLingo would be "
                f"confounded by batch size rather than architecture.",
                f"Either restore batch {SIMLINGO_EFFECTIVE_BATCH} via gradient "
                f"accumulation (needs a one-line change to train.py, which does "
                f"not currently forward it), or set lr between {lo:.2g} (linear) "
                f"and {hi:.2g} (sqrt) and say so when reporting."))

    # ---- precision
    prec = str(t.get("precision", ""))
    if prec == "16-mixed":
        f.append(Finding(
            "WARNING", "precision=16-mixed (fp16) on Blackwell",
            "fp16 needs a loss scaler and can overflow; bf16 has the same range as "
            "fp32 and needs no scaler. Blackwell supports bf16 natively, and "
            "FlashAttention-2 accepts both.",
            "Set precision: bf16-mixed and drop fp16_loss_scale."))
    else:
        f.append(Finding("OK", f"precision={prec}", "", ""))

    # ---- CoT share
    parts = d.get("train_partitions") or {}
    cot = float(parts.get("drivecot", d.get("drivecot_weight", 0.0)))
    if abs(cot - 0.60) > 1e-6:
        f.append(Finding(
            "WARNING", f"CoT share is {cot:.0%}, not the specified 60%",
            "train_partitions.drivecot controls what fraction of each batch is "
            "language-only supervision.",
            "Set train_partitions: {driving: 0.25, drivecot: 0.60, dreamer: 0.15}."))
    else:
        f.append(Finding("OK", "CoT share 60%", "", ""))

    # ---- configured-but-disabled features
    if d.get("img_shift_augmentation_prob") and not d.get("img_shift_augmentation"):
        f.append(Finding(
            "WARNING", "img_shift_augmentation is off but its probability is set",
            "The probability has no effect while the augmentation is disabled. "
            "Shift augmentation is what teaches recovery from lateral offset.",
            "Set img_shift_augmentation: true, or drop the stale probability."))
    if not parts.get("dreamer"):
        f.append(Finding(
            "WARNING", "No dreamer/instruction-following partition",
            "Instruction following and safety refusal are unsupervised without it.",
            "Add a dreamer share to train_partitions."))

    # ---- diffusion parameterisation
    if str(m.get("prediction_type", "epsilon")) == "epsilon" and \
            int(m.get("diffusion_infer_steps", 10)) <= 20:
        f.append(Finding(
            "WARNING",
            f"epsilon-prediction at {m['diffusion_infer_steps']} inference steps",
            "Measured on a controlled overfit: epsilon 2.607 m vs v-prediction "
            "0.690 m mean absolute waypoint error at 10 steps. epsilon is poorly "
            "conditioned at low SNR, which is where a short schedule spends most "
            "of its steps.",
            "Set model.prediction_type: v."))

    lora_n = lora_param_count(QWEN2_05B, int(m.get("lora_r", 16)))
    f.append(Finding("OK", f"LoRA r={m.get('lora_r')} -> {lora_n:,} adapter params",
                     "", ""))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on warnings as well as blockers")
    args = ap.parse_args()

    findings = audit(load_config(args.config))
    w = 78
    print("=" * w)
    print(f"CONFIG AUDIT  --  {args.config}".center(w))
    print("=" * w)
    order = {"BLOCKER": 0, "WARNING": 1, "OK": 2}
    for f in sorted(findings, key=lambda x: order[x.level]):
        print(f"[{f.level:<7}] {f.title}")
        if f.detail:
            for line in _wrap(f.detail, w - 12):
                print(f"            {line}")
        if f.fix:
            print(f"    fix ->  {f.fix}")
        if f.detail or f.fix:
            print()
    n_block = sum(f.level == "BLOCKER" for f in findings)
    n_warn = sum(f.level == "WARNING" for f in findings)
    print("-" * w)
    print(f"{n_block} blocker(s), {n_warn} warning(s)")
    print("=" * w)
    raise SystemExit(1 if n_block or (args.strict and n_warn) else 0)


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for part in text.split():
        if len(line) + len(part) + 1 > width:
            out.append(line)
            line = part
        else:
            line = f"{line} {part}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
