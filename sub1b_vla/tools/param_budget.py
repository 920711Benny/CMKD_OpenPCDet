"""Verify the strict <1B parameter budget.

When the HuggingFace hub is reachable this loads the real checkpoints and counts
their tensors. When it is not (air-gapped box, blocked proxy) it instead builds
*architecture-exact replicas* from each model's published configuration and
counts those. Parameter count is fully determined by architecture, not by weight
values, so the replica count equals the checkpoint count -- but the report always
states which path produced the number.

    python -m sub1b_vla.tools.param_budget
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

# Published configurations (config.json of each model card).
DINOV2_SMALL = dict(hidden=384, layers=12, heads=6, mlp=1536, patch=14,
                    image=518, layerscale=True, qkv_bias=True, cls=True)
DINOV2_BASE = dict(hidden=768, layers=12, heads=12, mlp=3072, patch=14,
                   image=518, layerscale=True, qkv_bias=True, cls=True)
SIGLIP_BASE = dict(hidden=768, layers=12, heads=12, mlp=3072, patch=16,
                   image=224, layerscale=False, qkv_bias=True, cls=False,
                   pooling_head=True)
# Published configs of the larger pair used by the CarlaVLA config.
DINOV2_LARGE = dict(hidden=1024, layers=24, heads=16, mlp=4096, patch=14,
                    image=518, layerscale=True, qkv_bias=True, cls=True)
SIGLIP_LARGE_384 = dict(hidden=1024, layers=24, heads=16, mlp=4096, patch=16,
                        image=384, layerscale=False, qkv_bias=True, cls=False,
                        pooling_head=True)
SIGLIP_BASE_384 = dict(hidden=768, layers=12, heads=12, mlp=3072, patch=16,
                       image=384, layerscale=False, qkv_bias=True, cls=False,
                       pooling_head=True)
SIGLIP_SO400M = dict(hidden=1152, layers=27, heads=16, mlp=4304, patch=14,
                     image=384, layerscale=False, qkv_bias=True, cls=False,
                     pooling_head=True)
QWEN2_05B = dict(hidden=896, layers=24, heads=14, kv_heads=2, head_dim=64,
                 intermediate=4864, vocab=151936, tie_embeddings=True)

VISION_SPECS = {
    "facebook/dinov2-small": DINOV2_SMALL,
    "facebook/dinov2-base": DINOV2_BASE,
    "google/siglip-base-patch16-224": SIGLIP_BASE,
    "facebook/dinov2-large": DINOV2_LARGE,
    "pretrained/dinov2-large": DINOV2_LARGE,
    "google/siglip-so400m-patch14-384": SIGLIP_SO400M,
    "pretrained/siglip-so400m": SIGLIP_SO400M,
    "google/siglip-large-patch16-384": SIGLIP_LARGE_384,
    "google/siglip-base-patch16-384": SIGLIP_BASE_384,
}


def vit_param_count(c: dict) -> int:
    """Exact parameter count of a ViT built to configuration `c`."""
    h, L, m = c["hidden"], c["layers"], c["mlp"]
    n_patch = (c["image"] // c["patch"]) ** 2
    total = 3 * c["patch"] ** 2 * h + h                      # patch conv + bias
    total += (n_patch + (1 if c["cls"] else 0)) * h          # positional
    total += h if c["cls"] else 0                            # cls token
    per_layer = 0
    per_layer += 3 * (h * h + (h if c["qkv_bias"] else 0))   # q,k,v
    per_layer += h * h + h                                   # attn out
    per_layer += h * m + m + m * h + h                       # mlp
    per_layer += 2 * (2 * h)                                 # 2x LayerNorm
    if c["layerscale"]:
        per_layer += 2 * h
    total += L * per_layer
    total += 2 * h                                           # final LayerNorm
    if c.get("pooling_head"):
        total += h                                           # learned probe
        total += 3 * (h * h + h) + h * h + h                 # pooling attention
        total += 2 * h + h * m + m + m * h + h               # head LN + MLP
    return total


def qwen2_param_count(c: dict) -> int:
    h, L = c["hidden"], c["layers"]
    kv = c["kv_heads"] * c["head_dim"]
    total = c["vocab"] * h                                   # tied embed/lm_head
    if not c["tie_embeddings"]:
        total += c["vocab"] * h
    per_layer = 0
    per_layer += h * h + h                                   # q_proj (+bias)
    per_layer += 2 * (h * kv + kv)                           # k_proj, v_proj (+bias)
    per_layer += h * h                                       # o_proj (no bias)
    per_layer += 3 * (h * c["intermediate"])                 # gate, up, down
    per_layer += 2 * h                                       # 2x RMSNorm
    total += L * per_layer
    total += h                                               # final RMSNorm
    return total


def lora_param_count(c: dict, r: int) -> int:
    """LoRA on q/k/v/o of every decoder layer."""
    h, L = c["hidden"], c["layers"]
    kv = c["kv_heads"] * c["head_dim"]
    per_layer = (r * h + h * r)          # q_proj
    per_layer += 2 * (r * h + kv * r)    # k_proj, v_proj
    per_layer += (r * h + h * r)         # o_proj
    return L * per_layer


# --- architecture-exact replicas (constructed and counted, not just arithmetic) --
class ReplicaViTLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, m = c["hidden"], c["mlp"]
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.q = nn.Linear(h, h, bias=c["qkv_bias"])
        self.k = nn.Linear(h, h, bias=c["qkv_bias"])
        self.v = nn.Linear(h, h, bias=c["qkv_bias"])
        self.o = nn.Linear(h, h)
        self.fc1, self.fc2 = nn.Linear(h, m), nn.Linear(m, h)
        if c["layerscale"]:
            self.ls1 = nn.Parameter(torch.ones(h))
            self.ls2 = nn.Parameter(torch.ones(h))


class ReplicaViT(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden"]
        n_patch = (c["image"] // c["patch"]) ** 2
        self.patch = nn.Conv2d(3, h, c["patch"], c["patch"])
        self.pos = nn.Parameter(torch.zeros(1, n_patch + (1 if c["cls"] else 0), h))
        if c["cls"]:
            self.cls = nn.Parameter(torch.zeros(1, 1, h))
        self.layers = nn.ModuleList([ReplicaViTLayer(c) for _ in range(c["layers"])])
        self.norm = nn.LayerNorm(h)
        if c.get("pooling_head"):
            self.probe = nn.Parameter(torch.zeros(1, 1, h))
            self.pool_attn = nn.MultiheadAttention(h, c["heads"], batch_first=True)
            self.pool_norm = nn.LayerNorm(h)
            self.pool_fc1 = nn.Linear(h, c["mlp"])
            self.pool_fc2 = nn.Linear(c["mlp"], h)


class ReplicaQwen2Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, kv, i = c["hidden"], c["kv_heads"] * c["head_dim"], c["intermediate"]
        self.q_proj = nn.Linear(h, h, bias=True)
        self.k_proj = nn.Linear(h, kv, bias=True)
        self.v_proj = nn.Linear(h, kv, bias=True)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.n1 = nn.Parameter(torch.ones(h))
        self.n2 = nn.Parameter(torch.ones(h))


class ReplicaQwen2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed = nn.Embedding(c["vocab"], c["hidden"])
        self.layers = nn.ModuleList([ReplicaQwen2Layer(c) for _ in range(c["layers"])])
        self.norm = nn.Parameter(torch.ones(c["hidden"]))
        if not c["tie_embeddings"]:
            self.lm_head = nn.Linear(c["hidden"], c["vocab"], bias=False)


def _count(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def try_real_count(model_id: str) -> int | None:
    try:
        from transformers import AutoModel  # noqa: PLC0415

        m = AutoModel.from_pretrained(model_id)
        if hasattr(m, "vision_model"):
            m = m.vision_model
        return _count(m)
    except Exception:  # noqa: BLE001
        return None


def build_report(cfg: dict) -> dict:
    from ..models.vla_agent import DualHeadDiffusionVLA  # noqa: PLC0415

    m = cfg["model"]
    rows, source = [], {}

    def add(name, model_id, spec, builder, counter):
        real = try_real_count(model_id)
        if real is not None:
            rows.append((name, model_id, real))
            source[name] = "hub checkpoint"
            return
        replica = _count(builder(spec))
        analytic = counter(spec)
        if replica != analytic:
            raise AssertionError(
                f"{name}: replica count {replica:,} != analytic {analytic:,}. "
                "One of the two derivations is wrong -- fix before trusting the budget."
            )
        rows.append((name, model_id, replica))
        source[name] = "architecture-exact replica (weights random; count is exact)"

    add("spatial backbone", m["spatial_model"], VISION_SPECS[m["spatial_model"]],
        ReplicaViT, vit_param_count)
    add("semantic backbone", m["semantic_model"], VISION_SPECS[m["semantic_model"]],
        ReplicaViT, vit_param_count)
    add("language decoder", m["language_model"], QWEN2_05B, ReplicaQwen2, qwen2_param_count)

    lora = lora_param_count(QWEN2_05B, m.get("lora_r", 16))
    rows.append(("LoRA adapters", f"r={m.get('lora_r', 16)} on q,k,v,o", lora))
    source["LoRA adapters"] = "analytic"

    # Trainable glue + diffusion head: built for real at the configured sizes.
    trainable_cfg = dict(cfg)
    heads = _build_trainable_only(cfg)
    for name, n in heads.items():
        rows.append((name, "trained from scratch", n))
        source[name] = "constructed"

    total = sum(r[2] for r in rows)
    limit = int(cfg.get("param_limit", 1_000_000_000))
    trainable = lora + sum(heads.values())
    return {"rows": rows, "total": total, "limit": limit, "trainable": trainable,
            "source": source, "within_budget": total < limit}


def _build_trainable_only(cfg: dict) -> dict:
    """Count the modules we actually train, at their configured dimensions."""
    from ..models.coc_prompt import INTENTS  # noqa: PLC0415
    from ..models.diffusion_head import CoCDiffusionHead  # noqa: PLC0415
    from ..models.dual_head_encoder import SpatialProjector, TokenCompressor  # noqa: PLC0415

    m = cfg["model"]
    e, sd = m["embed_dim"], m["spatial_dim"]
    sp_hidden = VISION_SPECS[m["spatial_model"]]["hidden"]
    se_hidden = VISION_SPECS[m["semantic_model"]]["hidden"]

    spatial_proj = SpatialProjector(sp_hidden, sd)
    compressor = TokenCompressor(se_hidden, e, m["num_semantic_tokens"])
    geo_attn = nn.MultiheadAttention(e, 8, batch_first=True)
    geo_kv = nn.Linear(sd, e)
    sem_norm = nn.LayerNorm(e)
    intent_head = nn.Sequential(nn.LayerNorm(e), nn.Linear(e, e // 2), nn.GELU(),
                                nn.Linear(e // 2, len(INTENTS)))
    diff = CoCDiffusionHead(
        spatial_dim=sd, sem_dim=e, dim=m.get("diffusion_dim", 256),
        depth=m.get("diffusion_depth", 4), heads=m.get("diffusion_heads", 8),
        pred_len=m["pred_len"], train_steps=m["diffusion_train_steps"],
        infer_steps=m["diffusion_infer_steps"],
    )
    return {
        "dual-head projectors": (_count(spatial_proj) + _count(compressor)
                                 + _count(geo_attn) + _count(geo_kv) + _count(sem_norm)),
        "intent head": _count(intent_head),
        "CoC diffusion decoder": _count(diff) - diff.schedule.betas.numel()
                                 - diff.schedule.alphas_cumprod.numel(),
    }


def render(rep: dict) -> str:
    w = 78
    out = ["=" * w, "PARAMETER BUDGET  --  strict limit < 1.000 B".center(w), "=" * w,
           f"{'component':<26}{'source / id':<32}{'params':>18}", "-" * w]
    for name, ident, n in rep["rows"]:
        out.append(f"{name:<26}{str(ident)[:31]:<32}{n:>18,}")
    out.append("-" * w)
    out.append(f"{'TOTAL':<58}{rep['total']:>18,}")
    out.append(f"{'':<58}{rep['total'] / 1e9:>17.4f}B")
    out.append(f"{'limit':<58}{rep['limit'] / 1e9:>17.4f}B")
    out.append(f"{'headroom':<58}{(rep['limit'] - rep['total']) / 1e9:>17.4f}B")
    out.append(f"{'TRAINABLE':<58}{rep['trainable']:>18,}")
    out.append(f"{'':<58}{rep['trainable'] / 1e6:>17.2f}M")
    out.append("-" * w)
    out.append("VERDICT: " + ("WITHIN BUDGET" if rep["within_budget"] else "OVER BUDGET"))
    out.append("=" * w)
    srcs = sorted(set(rep["source"].values()))
    out.append("count provenance: " + "; ".join(srcs))
    return "\n".join(out)


def main():
    from ..utils import load_config  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sub1b_vla/configs/default.yaml")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = build_report(cfg)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}) if args.json else render(rep))
    raise SystemExit(0 if rep["within_budget"] else 1)


if __name__ == "__main__":
    main()
