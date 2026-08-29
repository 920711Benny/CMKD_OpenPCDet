"""FlashAttention-2 attention, strictly.

Policy: this project uses FlashAttention-2. It does not fall back to `sdpa` or
to `eager`. A fallback is worse than a failure here -- it produces a run that is
quietly two to three times slower for reasons nobody sees until the wall-clock
bill arrives -- so an unusable FlashAttention raises with an actionable message
instead.

Two implementations carry the name and both are used, in this order:
  1. `flash_attn.flash_attn_func` -- the standalone package. Preferred, and the
     one HuggingFace's `attn_implementation="flash_attention_2"` requires.
  2. PyTorch's SDPA **FLASH_ATTENTION backend**, pinned explicitly. This is the
     same algorithm compiled into torch. It is NOT the `sdpa` fallback: the
     kernel is forced, and if the flash kernel cannot serve the shape the call
     raises rather than sliding onto math.

Escape hatch, deliberately loud and never the default: set
`SUB1B_ATTENTION_ALLOW_FALLBACK=1` to permit the math kernel. It exists so the
test suite and CPU-only CI can run at all. Every model built under it is marked,
and `gpu_preflight` treats it as a blocker.

FlashAttention exists only in fp16/bf16. On CUDA the inputs are cast to bf16 for
the kernel and cast back afterwards; that is not hidden because it is a real
precision statement, and it matches the bf16-mixed autocast training already
uses.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch >= 2.2
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:  # pragma: no cover
    _HAS_SDPA_KERNEL = False
    SDPBackend = None  # type: ignore[assignment]

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_PACKAGE = True
    _FLASH_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - a bad build raises more than ImportError
    flash_attn_func = None  # type: ignore[assignment]
    _HAS_FLASH_PACKAGE = False
    _FLASH_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

FLASH_DTYPES = (torch.float16, torch.bfloat16)


def fallback_allowed() -> bool:
    return os.environ.get("SUB1B_ATTENTION_ALLOW_FALLBACK", "0") == "1"


class FlashUnavailableError(RuntimeError):
    """Raised when FlashAttention-2 cannot serve a call and fallback is barred."""


def _explain(device: torch.device) -> str:
    lines = [
        "FlashAttention-2 is required but unusable here.",
        f"  device                 : {device.type}",
        f"  flash_attn package     : "
        f"{'importable' if _HAS_FLASH_PACKAGE else 'NOT importable -- ' + _FLASH_IMPORT_ERROR}",
        f"  torch SDPA flash kernel: {'available' if _HAS_SDPA_KERNEL else 'torch too old'}",
        "",
        "FlashAttention-2 needs a CUDA device and fp16/bf16 inputs. On Blackwell",
        "(RTX 6000 Pro, sm_120) it also needs a CUDA 12.8+ toolchain:",
        "  pip install torch --index-url https://download.pytorch.org/whl/cu128",
        "  MAX_JOBS=4 pip install flash-attn --no-build-isolation",
        "",
        "Run `./run_pipeline.sh preflight` to see which kernels actually execute.",
        "To run without it anyway (CPU tests / CI only, and it will be slow):",
        "  export SUB1B_ATTENTION_ALLOW_FALLBACK=1",
    ]
    return "\n".join(lines)


def _flash_sdpa(q, k, v, dropout_p, is_causal):
    """SDPA with the FLASH backend pinned. Raises if flash cannot serve it."""
    if not _HAS_SDPA_KERNEL:
        raise FlashUnavailableError("torch.nn.attention.sdpa_kernel is unavailable")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                              is_causal=is_causal)


class FlashAttention(nn.Module):
    """Multi-head attention on FlashAttention-2.

    Parameter-count identical to `nn.MultiheadAttention` (4d^2 + 4d), so
    swapping it in cannot move the parameter budget.
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
        self.last_backend = "uninitialised"

    def _heads(self, x, b, t):
        return x.view(b, t, self.num_heads, self.head_dim)

    def _attend(self, q, k, v, is_causal):
        """q/k/v: (B, T, H, D). Returns (B, T, H, D)."""
        device = q.device
        drop = self.dropout if self.training else 0.0

        if device.type == "cuda":
            orig_dtype = q.dtype
            if orig_dtype not in FLASH_DTYPES:
                # FlashAttention exists only in 16-bit. Stated rather than hidden.
                q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
            if _HAS_FLASH_PACKAGE:
                out = flash_attn_func(q, k, v, dropout_p=drop, causal=is_causal)
                self.last_backend = "flash_attn"
            else:
                out = _flash_sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                  drop, is_causal).transpose(1, 2)
                self.last_backend = "torch_sdpa_flash_kernel"
            return out.to(orig_dtype)

        if not fallback_allowed():
            raise FlashUnavailableError(_explain(device))
        self.last_backend = "math_fallback"
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=drop, is_causal=is_causal)
        return out.transpose(1, 2)

    def forward(self, query, key, value, is_causal: bool = False,
                need_weights: bool = False):
        b, tq, _ = query.shape
        tk = key.shape[1]
        q = self._heads(self.q_proj(query), b, tq)
        k = self._heads(self.k_proj(key), b, tk)
        v = self._heads(self.v_proj(value), b, tk)

        if need_weights:
            # Explicit weights path for the HUD only. FlashAttention never
            # materialises a weight matrix -- that is the point of it -- so
            # asking for one necessarily leaves the fused kernel.
            qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
            attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.triu(torch.ones(tq, tk, dtype=torch.bool, device=q.device), 1)
                attn = attn.masked_fill(mask, float("-inf"))
            w = attn.softmax(dim=-1)
            out = (w @ vh).transpose(1, 2).reshape(b, tq, self.embed_dim)
            return self.out_proj(out), w.mean(dim=1)

        out = self._attend(q, k, v, is_causal).reshape(b, tq, self.embed_dim)
        return self.out_proj(out), None


# Kept so existing imports do not break; it is the same strict class.
SDPAAttention = FlashAttention


def flash_attn_package_report() -> dict:
    if _HAS_FLASH_PACKAGE:
        import flash_attn  # noqa: PLC0415

        return {"installed": True, "version": getattr(flash_attn, "__version__", "unknown")}
    return {"installed": False, "reason": _FLASH_IMPORT_ERROR or "not installed"}


def probe_flash(device: torch.device, dtype=torch.bfloat16,
                batch=2, heads=8, seq=256, head_dim=64) -> dict:
    """Run FlashAttention-2 here and report what actually executed.

    Deliberately runs the kernel rather than inspecting versions: a flash-attn
    wheel built for the wrong compute capability imports cleanly and then never
    serves a call.
    """
    report = {"device": device.type, "dtype": str(dtype),
              "fallback_allowed": fallback_allowed()}
    if device.type != "cuda":
        report["usable"] = False
        report["reason"] = "FlashAttention-2 has no CPU kernel"
        return report

    q, k, v = (torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
               for _ in range(3))
    if _HAS_FLASH_PACKAGE:
        try:
            flash_attn_func(q, k, v)
            report.update(usable=True, backend="flash_attn")
            return report
        except Exception as exc:  # noqa: BLE001
            report["flash_attn_error"] = f"{type(exc).__name__}: {exc}"
    try:
        _flash_sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 0.0, False)
        report.update(usable=True, backend="torch_sdpa_flash_kernel")
    except Exception as exc:  # noqa: BLE001
        report.update(usable=False, reason=f"{type(exc).__name__}: {exc}")
    return report


def attention_backend_report(device: torch.device | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"flash": probe_flash(device, dtype=dtype),
            "flash_attn_package": flash_attn_package_report(),
            "fallback_allowed": fallback_allowed()}
