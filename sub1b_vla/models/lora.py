"""Minimal dependency-free LoRA injection.

`peft` is used when present; otherwise this fallback provides the same
low-rank adapter semantics so training never hard-depends on an extra package.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return out + delta * self.scaling


# Attention projections of Qwen2/LLaMA-style decoders. The offline stub LM uses
# the same names, so it exercises this exact code path.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

# torch's fused transformer modules read `.weight` straight off their child
# Linears in the eval-mode fast path, so an adapter wrapper breaks them.
_FUSED_PARENTS = (nn.MultiheadAttention, nn.TransformerEncoderLayer,
                  nn.TransformerDecoderLayer)


def inject_lora(
    model: nn.Module,
    target_substrings: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> int:
    """Replace matching nn.Linear modules in-place. Returns count injected."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, _FUSED_PARENTS):
            continue
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in target_substrings):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced += 1
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_("lora_a" in n or "lora_b" in n)
