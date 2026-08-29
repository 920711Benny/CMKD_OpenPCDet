"""Dual-Head Asymmetric Visual Encoder.

Two complementary, deliberately *asymmetric* streams:

  Spatial-Geometric  (DINOv2)  -> dense patch grid, high spatial fidelity.
                                  Routed DIRECTLY to the diffusion waypoint head.
                                  Never compressed to a handful of tokens: the
                                  trajectory decoder needs metric/affordance
                                  detail that pooling destroys.

  Semantic-Reasoning (SigLIP)  -> a small set of global semantic tokens placed
                                  in the LLM embedding space for causal
                                  rationale generation. Heavily compressed: the
                                  language model needs *what* and *why*, not
                                  per-patch geometry.

Both backbones stay frozen. Only the projection / compression / alignment
parameters train, which is what keeps the trainable footprint small enough to
fit a single-GPU LoRA budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbones import build_vision_backbone


@dataclass
class DualHeadOutput:
    spatial_tokens: torch.Tensor   # (B, N_s, spatial_dim)  -> diffusion head
    semantic_tokens: torch.Tensor  # (B, N_q, embed_dim)    -> LLM input embeds
    spatial_attn: torch.Tensor     # (B, N_s) saliency, for the HUD overlay
    semantic_attn: torch.Tensor    # (B, N_sem) saliency, for the HUD overlay


class TokenCompressor(nn.Module):
    """Learned-query cross-attention pooling (Perceiver/Q-Former style).

    Compresses a variable-length patch sequence into `num_queries` tokens and
    also exposes the attention mass per source patch, which the HUD renders as
    the semantic branch's attention map.
    """

    def __init__(self, in_dim: int, out_dim: int, num_queries: int, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, out_dim) * 0.02)
        self.kv_proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(out_dim)
        self.norm_kv = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        kv = self.norm_kv(self.kv_proj(x))
        q = self.norm_q(self.queries.expand(b, -1, -1))
        out, attn_w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        tokens = out + q
        tokens = tokens + self.ffn(tokens)
        return tokens, attn_w.mean(dim=1)  # (B, N_src)


class SpatialProjector(nn.Module):
    """Keeps the patch grid intact; only re-dimensions and adds a light
    depth/affordance-oriented residual refinement."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.refine = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )
        self.saliency = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(self.proj(x))
        h = h + self.refine(h)
        sal = self.saliency(h).squeeze(-1).softmax(dim=-1)
        return h, sal


class DualHeadAsymmetricEncoder(nn.Module):
    def __init__(
        self,
        spatial_model: str = "facebook/dinov2-small",
        semantic_model: str = "google/siglip-base-patch16-224",
        embed_dim: int = 896,
        spatial_dim: int = 256,
        num_semantic_tokens: int = 32,
        image_size: int = 224,
        freeze: bool = True,
        allow_stub: bool = True,
    ):
        super().__init__()
        self.spatial_backbone, self.spatial_spec = build_vision_backbone(
            spatial_model, image_size, allow_stub
        )
        self.semantic_backbone, self.semantic_spec = build_vision_backbone(
            semantic_model, image_size, allow_stub
        )
        self.frozen = freeze
        if freeze:
            for p in self.spatial_backbone.parameters():
                p.requires_grad_(False)
            for p in self.semantic_backbone.parameters():
                p.requires_grad_(False)
            self.spatial_backbone.eval()
            self.semantic_backbone.eval()

        self.spatial_proj = SpatialProjector(self.spatial_spec.hidden_size, spatial_dim)
        self.semantic_compress = TokenCompressor(
            self.semantic_spec.hidden_size, embed_dim, num_semantic_tokens
        )
        # Asymmetric cross-talk: geometry informs semantics (a stopped car ahead
        # is a *geometric* fact that must reach the rationale), but we do not let
        # the coarse semantic stream blur the dense geometric one.
        self.geo_to_sem = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.geo_kv = nn.Linear(spatial_dim, embed_dim)
        self.sem_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim

    def train(self, mode: bool = True):  # keep frozen backbones in eval
        super().train(mode)
        if self.frozen:
            self.spatial_backbone.eval()
            self.semantic_backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> DualHeadOutput:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            spatial_raw = self.spatial_backbone(pixel_values)
            semantic_raw = self.semantic_backbone(pixel_values)
        if self.frozen:
            spatial_raw = spatial_raw.detach()
            semantic_raw = semantic_raw.detach()

        spatial_tokens, spatial_attn = self.spatial_proj(spatial_raw)
        semantic_tokens, semantic_attn = self.semantic_compress(semantic_raw)

        kv = self.geo_kv(spatial_tokens)
        cross, _ = self.geo_to_sem(semantic_tokens, kv, kv, need_weights=False)
        semantic_tokens = self.sem_norm(semantic_tokens + cross)

        return DualHeadOutput(
            spatial_tokens=spatial_tokens,
            semantic_tokens=semantic_tokens,
            spatial_attn=spatial_attn,
            semantic_attn=semantic_attn,
        )
