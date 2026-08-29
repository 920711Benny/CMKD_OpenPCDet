"""Sub-1B Dual-Head Diffusion VLA -- full model assembly.

Parameter budget (the reason this fits under 1B):

  InternVL2-1B ships a 0.30B InternViT tower + a 0.49B Qwen2-0.5B decoder.
  We keep ONLY the decoder and delete the native tower, because the dual-head
  asymmetric encoder replaces it. That buys back ~0.30B, which is what makes
  room for DINOv2 + SigLIP + the diffusion decoder while staying under budget.

      Qwen2-0.5B decoder (frozen + LoRA)   ~0.49 B
      DINOv2-small       (frozen)          ~0.02 B
      SigLIP-base tower  (frozen)          ~0.09 B
      projectors / compressor / fusion     ~0.02 B
      CoC diffusion decoder                ~0.01 B
      -------------------------------------------
      total                                ~0.63 B   < 1.0 B  (enforced below)

`assert_parameter_budget` is a hard gate, not a comment: constructing the model
with a configuration that exceeds the limit raises.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..losses.composite import LossBreakdown, LossWeights, combine
from ..losses.consistency import ConsistencyThresholds, causal_consistency_loss
from .coc_language import CoCLanguageModel
from .coc_prompt import INTENTS
from .diffusion_head import CoCDiffusionHead
from .dual_head_encoder import DualHeadAsymmetricEncoder


@dataclass
class ParamReport:
    total: int
    trainable: int
    by_component: dict

    def __str__(self) -> str:
        lines = [f"{'component':<28}{'params':>14}{'trainable':>14}"]
        lines.append("-" * 56)
        for k, (tot, tr) in self.by_component.items():
            lines.append(f"{k:<28}{tot:>14,}{tr:>14,}")
        lines.append("-" * 56)
        lines.append(f"{'TOTAL':<28}{self.total:>14,}{self.trainable:>14,}")
        lines.append(f"{'':<28}{self.total / 1e9:>13.3f}B{self.trainable / 1e6:>13.1f}M")
        return "\n".join(lines)


@dataclass
class VLAOutput:
    waypoints: torch.Tensor                  # (B, T, 2) sampled trajectory
    intent_logits: torch.Tensor
    spatial_attn: torch.Tensor
    semantic_attn: torch.Tensor
    text_ids: torch.Tensor | None = None


class DualHeadDiffusionVLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.encoder = DualHeadAsymmetricEncoder(
            spatial_model=m["spatial_model"],
            semantic_model=m["semantic_model"],
            embed_dim=m["embed_dim"],
            spatial_dim=m["spatial_dim"],
            num_semantic_tokens=m["num_semantic_tokens"],
            image_size=m["image_size"],
            freeze=m.get("freeze", True),
            allow_stub=m.get("allow_stub", True),
        )
        self.language = CoCLanguageModel(
            model_id=m["language_model"],
            embed_dim=m["embed_dim"],
            lora_r=m.get("lora_r", 16),
            lora_alpha=m.get("lora_alpha", 32),
            lora_dropout=m.get("lora_dropout", 0.05),
            lora_targets=m.get("lora_targets"),
            allow_stub=m.get("allow_stub", True),
            use_language_tower_only=m.get("use_language_tower_only", True),
        )
        self.diffusion = CoCDiffusionHead(
            spatial_dim=m["spatial_dim"],
            sem_dim=m["embed_dim"],
            dim=m.get("diffusion_dim", 256),
            depth=m.get("diffusion_depth", 4),
            heads=m.get("diffusion_heads", 8),
            pred_len=m["pred_len"],
            train_steps=m["diffusion_train_steps"],
            infer_steps=m["diffusion_infer_steps"],
        )
        self.dt = m.get("waypoint_dt", 0.2)
        self.x0_clamp = float(m.get("x0_clamp_m", 80.0))
        self.weights = LossWeights(**cfg.get("loss", {}))
        self.thresholds = ConsistencyThresholds(**cfg.get("consistency", {}))
        self.param_limit = int(cfg.get("param_limit", 1_000_000_000))
        self.assert_parameter_budget()

    # ---- budget ----------------------------------------------------------
    def parameter_report(self) -> ParamReport:
        def count(mod):
            t = sum(p.numel() for p in mod.parameters())
            tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            return t, tr

        comps = {
            "encoder.spatial (frozen)": count(self.encoder.spatial_backbone),
            "encoder.semantic (frozen)": count(self.encoder.semantic_backbone),
            "encoder.projectors": (
                count(self.encoder.spatial_proj)[0] + count(self.encoder.semantic_compress)[0]
                + count(self.encoder.geo_to_sem)[0] + count(self.encoder.geo_kv)[0],
                count(self.encoder.spatial_proj)[1] + count(self.encoder.semantic_compress)[1]
                + count(self.encoder.geo_to_sem)[1] + count(self.encoder.geo_kv)[1],
            ),
            "language (LoRA)": count(self.language),
            "diffusion head": count(self.diffusion),
        }
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return ParamReport(total=total, trainable=trainable, by_component=comps)

    def assert_parameter_budget(self) -> ParamReport:
        rep = self.parameter_report()
        if rep.total >= self.param_limit:
            raise ValueError(
                f"Parameter budget exceeded: {rep.total:,} >= limit {self.param_limit:,}.\n"
                f"{rep}\nReduce backbone sizes or keep use_language_tower_only=True."
            )
        return rep

    # ---- forward ---------------------------------------------------------
    def encode(self, pixel_values):
        return self.encoder(pixel_values)

    def forward(self, batch: dict, step: int = 0) -> tuple[LossBreakdown, dict]:
        """Training step. Returns (loss breakdown, aux tensors)."""
        enc = self.encoder(batch["image"])
        lang = self.language(
            enc.semantic_tokens,
            text_ids=batch.get("text_ids"),
            text_mask=batch.get("text_mask"),
            labels=batch.get("text_labels"),
        )
        diff_per_sample, x0, _, reliability = self.diffusion.loss(
            batch["waypoints"], enc.spatial_tokens, enc.semantic_tokens,
            batch["speed"], batch["target_point"], batch["command"],
            x0_clamp=self.x0_clamp,
        )
        # DriveCoT/QA samples carry no expert trajectory: they supervise language
        # only and must not pull the diffusion head towards their filler waypoints.
        wp_mask = batch.get("has_waypoints")
        if wp_mask is None:
            wp_mask = torch.ones_like(diff_per_sample)
        diff_loss = (diff_per_sample * wp_mask).sum() / wp_mask.sum().clamp(min=1.0)

        cons_per_sample, dyn, _ = causal_consistency_loss(
            lang.intent_logits, x0, dt=self.dt, thresholds=self.thresholds, reduction="none"
        )
        # Weight by x0 reliability -- see CoCDiffusionHead.loss.
        cw = reliability * wp_mask
        cons_loss = (cons_per_sample * cw).sum() / cw.sum().clamp(min=1e-6)

        rel = reliability > 0.5
        # L_LM = next-token CE + weighted intent CE (both are language supervision;
        # the head is a differentiable read-out of the rationale's action slot).
        lm_total = lang.lm_loss
        if "intent_id" in batch:
            intent_ce = torch.nn.functional.cross_entropy(
                lang.intent_logits.float(), batch["intent_id"].long()
            )
            lm_total = intent_ce * self.weights.intent_ce if lm_total is None else \
                lm_total + self.weights.intent_ce * intent_ce

        extras = {
            "mean_accel": float((dyn.accel[rel].mean() if rel.any() else dyn.accel.mean()).detach()),
            "mean_final_speed": float(
                (dyn.final_speed[rel].mean() if rel.any() else dyn.final_speed.mean()).detach()),
            "reliable_frac": float(rel.float().mean()),
        }
        if "intent_id" in batch:
            pred = lang.intent_logits.argmax(-1)
            extras["intent_acc"] = float((pred == batch["intent_id"]).float().mean())
        breakdown = combine(lm_total, diff_loss, cons_loss, self.weights, step, extras)
        return breakdown, {"x0": x0, "encoded": enc, "language": lang}

    # ---- inference -------------------------------------------------------
    @torch.no_grad()
    def predict_trajectory(self, pixel_values, speed, target_point, command,
                           encoded=None, steps: int | None = None) -> VLAOutput:
        """Fast path: geometry -> diffusion. Deliberately does NOT decode text,
        which is what keeps the control loop inside its latency budget."""
        enc = encoded if encoded is not None else self.encoder(pixel_values)
        lang = self.language(enc.semantic_tokens)
        wp = self.diffusion.sample(
            enc.spatial_tokens, enc.semantic_tokens, speed, target_point, command, steps=steps
        )
        return VLAOutput(
            waypoints=wp,
            intent_logits=lang.intent_logits,
            spatial_attn=enc.spatial_attn,
            semantic_attn=enc.semantic_attn,
        )

    @torch.no_grad()
    def explain(self, encoded, prompt_ids=None, max_new_tokens: int = 48) -> list[str]:
        """Slow path: verbose CoC rationale for the HUD. Runs off the control loop."""
        ids = self.language.generate(
            encoded.semantic_tokens, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens
        )
        return self.language.decode(ids)


def intent_name(idx: int) -> str:
    return INTENTS[int(idx)]
