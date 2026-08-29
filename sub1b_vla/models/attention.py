"""Attention that can actually reach FlashAttention-2.

Why this module exists: `nn.MultiheadAttention` does NOT dispatch to
FlashAttention. Its fused path is a separate native kernel with its own
conditions, and it silently falls back to an unfused math implementation
whenever they are not met (need_weights=True being the usual culprit). A model
built from `nn.MultiheadAttention` therefore gets no benefit from an installed
flash-attn, however carefully it was compiled.

`F.scaled_dot_product_attention` is the dispatcher that does reach the flash
kernels, so all attention here goes through it, under an explicit
`sdpa_kernel` preference when one is requested.

Requirements FlashAttention-2 imposes, which the caller must satisfy:
  * CUDA device; it has no CPU kernel.
  * fp16 or bf16 inputs -- fp32 silently falls back to the math backend. Under
    the bf16 autocast this project trains with, that holds.
  * head_dim <= 256 and, for older builds, a multiple of 8.

`attention_backend_report()` reports what actually ran rather than what was
requested, because the failure mode here is silent.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch >= 2.2
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:  # pragma: no cover - older torch
    _HAS_SDPA_KERNEL = False
    SDPBackend = None  # type: ignore[assignment]

BACKEND_PREFERENCE = ("flash", "efficient", "math")


def _backend_enum(name: str):
    if not _HAS_SDPA_KERNEL:
        return None
    return {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", SDPBackend.MATH),
    }.get(name)


@contextlib.contextmanager
def prefer_backends(names=BACKEND_PREFERENCE):
    """Ask SDPA to use these backends, in order, when the shapes allow it."""
    if not _HAS_SDPA_KERNEL or not torch.cuda.is_available():
        yield
        return
    backends = [b for b in (_backend_enum(n) for n in names) if b is not None]
    if not backends:
        yield
        return
    try:
        with sdpa_kernel(backends):
            yield
    except RuntimeError:
        # A shape/dtype combination no listed backend accepts. Fall back rather
        # than crash: correctness first, speed second.
        yield


class SDPAAttention(nn.Module):
    """Multi-head attention over `F.scaled_dot_product_attention`.

    A drop-in replacement for `nn.MultiheadAttention(batch_first=True)` for the
    cross/self-attention used in this model. It deliberately does NOT support
    returning attention weights from the fused path: materialising the weight
    matrix is exactly what forces the math backend, so weights are computed on a
    separate, explicit slow path only when asked for.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 kdim: int | None = None, vdim: int | None = None, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim or embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(vdim or embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, x, b, t):
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, is_causal: bool = False,
                need_weights: bool = False):
        b, tq, _ = query.shape
        tk = key.shape[1]
        q = self._shape(self.q_proj(query), b, tq)
        k = self._shape(self.k_proj(key), b, tk)
        v = self._shape(self.v_proj(value), b, tk)

        if need_weights:
            # Explicit slow path. Never mixed into the fused one -- asking for
            # weights there is what silently disables flash.
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.triu(torch.ones(tq, tk, dtype=torch.bool, device=q.device), 1)
                attn = attn.masked_fill(mask, float("-inf"))
            w = attn.softmax(dim=-1)
            out = w @ v
            out = out.transpose(1, 2).reshape(b, tq, self.embed_dim)
            return self.out_proj(out), w.mean(dim=1)

        with prefer_backends():
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal)
        out = out.transpose(1, 2).reshape(b, tq, self.embed_dim)
        return self.out_proj(out), None


def probe_attention_backend(device: torch.device, dtype=torch.bfloat16,
                            batch=2, heads=8, seq=256, head_dim=64) -> dict:
    """Determine which SDPA backends actually run here, by running them.

    Reports what worked, not what is installed -- a flash-attn wheel that does
    not match the driver or the GPU's compute capability imports fine and then
    never gets used.
    """
    report: dict = {"device": device.type, "dtype": str(dtype), "backends": {}}
    if device.type != "cuda":
        report["note"] = "no CUDA device: SDPA runs the CPU math kernel only"
        return report
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
               for _ in range(3))
    for name in ("flash", "efficient", "math"):
        backend = _backend_enum(name)
        if backend is None:
            report["backends"][name] = "unavailable (torch too old)"
            continue
        try:
            with sdpa_kernel([backend]):
                F.scaled_dot_product_attention(q, k, v)
            report["backends"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["backends"][name] = f"unusable ({type(exc).__name__})"
    report["flash_available"] = report["backends"].get("flash") == "ok"
    return report


def flash_attn_package_report() -> dict:
    """Whether the standalone flash-attn package is importable, and its version.

    Separate from the SDPA probe on purpose: torch's built-in flash kernels and
    the flash-attn package are different things, and HuggingFace's
    `attn_implementation="flash_attention_2"` needs the package specifically.
    """
    try:
        import flash_attn  # noqa: PLC0415

        return {"installed": True, "version": getattr(flash_attn, "__version__", "unknown")}
    except ImportError as exc:
        return {"installed": False, "reason": str(exc)}


def attention_backend_report(device: torch.device | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "sdpa": probe_attention_backend(device, dtype=dtype),
        "flash_attn_package": flash_attn_package_report(),
    }
