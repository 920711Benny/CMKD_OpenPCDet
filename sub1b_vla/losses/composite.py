"""Composite objective.

    L_total = lambda_lm * L_LM
            + lambda_diff * L_diffusion
            + lambda_align * L_consistency

`lambda_align` is warmed up: coupling the two streams before either produces
anything meaningful just injects noise, so the alignment term ramps in linearly
over `align_warmup_steps` optimizer steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LossWeights:
    lm: float = 1.0
    diffusion: float = 1.0
    consistency: float = 0.2
    # Supervised cross-entropy on the intent read-out, folded into the LM term.
    # Without it the intent head has no grounding in the scene: the consistency
    # loss only enforces AGREEMENT with the trajectory, which a constant
    # prediction satisfies trivially.
    intent_ce: float = 0.5
    align_warmup_steps: int = 2000

    def align_weight(self, step: int) -> float:
        if self.align_warmup_steps <= 0:
            return self.consistency
        return self.consistency * min(1.0, step / float(self.align_warmup_steps))


@dataclass
class LossBreakdown:
    total: torch.Tensor
    lm: torch.Tensor
    diffusion: torch.Tensor
    consistency: torch.Tensor
    align_weight: float
    extras: dict = field(default_factory=dict)

    def as_log(self) -> dict:
        out = {
            "loss/total": float(self.total.detach()),
            "loss/lm": float(self.lm.detach()),
            "loss/diffusion": float(self.diffusion.detach()),
            "loss/consistency": float(self.consistency.detach()),
            "loss/align_weight": self.align_weight,
        }
        out.update({f"extra/{k}": v for k, v in self.extras.items()})
        return out


def combine(
    lm_loss: torch.Tensor | None,
    diff_loss: torch.Tensor,
    cons_loss: torch.Tensor,
    weights: LossWeights,
    step: int,
    extras: dict | None = None,
) -> LossBreakdown:
    zero = diff_loss.new_zeros(())
    lm = lm_loss if lm_loss is not None else zero
    aw = weights.align_weight(step)
    total = weights.lm * lm + weights.diffusion * diff_loss + aw * cons_loss
    return LossBreakdown(
        total=total, lm=lm, diffusion=diff_loss, consistency=cons_loss,
        align_weight=aw, extras=extras or {},
    )
