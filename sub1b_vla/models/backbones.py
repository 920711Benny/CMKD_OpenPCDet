"""Backbone construction with graceful degradation.

Real runs load frozen HuggingFace checkpoints. When weights are unavailable
(offline CI, CPU smoke tests) we fall back to a *shape-compatible* random ViT
so that every downstream shape/gradient path stays exercisable. The fallback is
always announced loudly -- a stub must never be mistaken for a trained encoder.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    name: str
    hidden_size: int
    patch: int
    image_size: int
    is_stub: bool = False

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch) ** 2


class StubViT(nn.Module):
    """Tiny random ViT used only when real weights cannot be fetched."""

    def __init__(self, spec: BackboneSpec, depth: int = 2, heads: int = 4):
        super().__init__()
        self.spec = spec
        d = spec.hidden_size
        self.proj = nn.Conv2d(3, d, kernel_size=spec.patch, stride=spec.patch)
        self.pos = nn.Parameter(torch.zeros(1, spec.num_patches, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 2,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values).flatten(2).transpose(1, 2)
        if x.shape[1] != self.pos.shape[1]:  # tolerate non-square / resized input
            pos = _interp_pos(self.pos, x.shape[1])
        else:
            pos = self.pos
        return self.norm(self.blocks(x + pos))


def _interp_pos(pos: torch.Tensor, n: int) -> torch.Tensor:
    src = int(math.sqrt(pos.shape[1]))
    dst = int(math.sqrt(n))
    grid = pos.reshape(1, src, src, -1).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=(dst, dst), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, dst * dst, -1)


_KNOWN = {
    "facebook/dinov2-small": (384, 14, 224),
    "facebook/dinov2-base": (768, 14, 224),
    "google/siglip-base-patch16-224": (768, 16, 224),
    "google/siglip-so400m-patch14-224": (1152, 14, 224),
}


def build_vision_backbone(model_id: str, image_size: int, allow_stub: bool = True):
    """Return (module, BackboneSpec). Module output: (B, N_patches, hidden)."""
    hidden, patch, native = _KNOWN.get(model_id, (384, 14, 224))
    spec = BackboneSpec(model_id, hidden, patch, image_size)
    try:
        from transformers import AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(model_id)
        if hasattr(model, "vision_model"):  # SigLIP / CLIP style wrapper
            model = model.vision_model
        cfg = getattr(model, "config", None)
        if cfg is not None:
            spec.hidden_size = getattr(cfg, "hidden_size", hidden)
            spec.patch = getattr(cfg, "patch_size", patch)
        return _HFVisionWrapper(model), spec
    except Exception as exc:  # noqa: BLE001 - offline / missing weights is expected
        if not allow_stub:
            raise
        warnings.warn(
            f"[STUB BACKBONE] '{model_id}' unavailable ({type(exc).__name__}: {exc}). "
            "Using randomly-initialised StubViT. Results are NOT meaningful.",
            RuntimeWarning,
            stacklevel=2,
        )
        spec.is_stub = True
        return StubViT(spec), spec


class _HFVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        h = out.last_hidden_state
        # DINOv2 emits a CLS token at index 0; SigLIP does not.
        if h.shape[1] % 2 == 1:
            h = h[:, 1:]
        return h
