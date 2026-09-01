"""Drop-in replacement for nn.MultiheadAttention that reaches FlashAttention-2.

Why it is needed: `nn.MultiheadAttention` does not dispatch to FlashAttention.
Its fused path is a separate native kernel with its own preconditions, and it
falls back to unfused math whenever they are not met. In particular
`need_weights` defaults to **True**, so a call written as

    attn_out, _ = self.cross_attn(q, kv, kv)

materialises the full attention matrix, discards it, and never touches a fast
kernel. That is the call shape in `dual_vision_model.CrossAttentionFusion`.

Why it is state-dict compatible: this module deliberately keeps
nn.MultiheadAttention's exact parameter layout --

    in_proj_weight   (3E, E)
    in_proj_bias     (3E,)
    out_proj.weight  (E, E)
    out_proj.bias    (E,)

-- so an existing checkpoint loads into it unchanged. A cleaner implementation
with separate q/k/v Linears would have renamed every key and orphaned the
trained cross_fusion weights.

Requires a CUDA device and fp16/bf16 for the flash kernel. On CPU, or in fp32,
it computes the same maths on the fallback path; the result is identical, only
slower.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:  # pragma: no cover
    _HAS_SDPA_KERNEL = False

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH = True
except Exception:  # noqa: BLE001
    flash_attn_func = None
    _HAS_FLASH = False

FLASH_DTYPES = (torch.float16, torch.bfloat16)


class FlashMultiheadAttention(nn.Module):
    """batch_first multi-head attention, state-dict compatible with
    `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)`."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, batch_first: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.last_backend = "uninitialised"

    def _project(self, x, idx):
        w = self.in_proj_weight[idx * self.embed_dim:(idx + 1) * self.embed_dim]
        b = None if self.in_proj_bias is None else \
            self.in_proj_bias[idx * self.embed_dim:(idx + 1) * self.embed_dim]
        return F.linear(x, w, b)

    def forward(self, query, key, value, need_weights: bool = False,
                average_attn_weights: bool = True, attn_mask=None, key_padding_mask=None):
        if attn_mask is not None or key_padding_mask is not None:
            raise NotImplementedError(
                "FlashMultiheadAttention does not take masks; the dual-vision "
                "cross-attention does not use them.")
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        b, tq, _ = query.shape
        tk = key.shape[1]
        q = self._project(query, 0).view(b, tq, self.num_heads, self.head_dim)
        k = self._project(key, 1).view(b, tk, self.num_heads, self.head_dim)
        v = self._project(value, 2).view(b, tk, self.num_heads, self.head_dim)
        drop = self.dropout if self.training else 0.0

        if need_weights:
            # Explicit path. FlashAttention never materialises a weight matrix --
            # that is the point of it -- so asking for one leaves the fast kernel.
            qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
            scores = (qh @ kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
            w = scores.softmax(dim=-1)
            out = (w @ vh).transpose(1, 2).reshape(b, tq, self.embed_dim)
            self.last_backend = "explicit_weights"
            out = self.out_proj(out)
            if not self.batch_first:
                out = out.transpose(0, 1)
            return out, (w.mean(dim=1) if average_attn_weights else w)

        if q.is_cuda:
            orig = q.dtype
            if orig not in FLASH_DTYPES:
                q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
            if _HAS_FLASH:
                out = flash_attn_func(q, k, v, dropout_p=drop)
                self.last_backend = "flash_attn"
            elif _HAS_SDPA_KERNEL:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        dropout_p=drop).transpose(1, 2)
                self.last_backend = "torch_sdpa_flash_kernel"
            else:  # pragma: no cover
                raise RuntimeError("no FlashAttention implementation available")
            out = out.to(orig)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=drop).transpose(1, 2)
            self.last_backend = "cpu_fallback"

        out = self.out_proj(out.reshape(b, tq, self.embed_dim))
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None
