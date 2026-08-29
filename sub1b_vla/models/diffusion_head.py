"""Chain-of-Causation conditional diffusion trajectory decoder.

Denoises a `pred_len x 2` waypoint sequence in the ego BEV frame, conditioned on
three heterogeneous streams that are kept *separate* (not concatenated into one
soup) so each can be attended to on its own terms:

  1. dense spatial-geometric tokens   (where the drivable space / obstacles are)
  2. semantic Chain-of-Causation tokens (why the ego should act)
  3. a scalar ego-state embedding      (speed, target point, discrete command)

Training uses `diffusion_train_steps` (default 100) with a cosine schedule;
inference uses a DDIM subsequence of `diffusion_infer_steps` (default 10) which
is what makes the >=10 Hz control budget reachable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    return torch.cos((t + s) / (1.0 + s) * math.pi / 2).pow(2)


class NoiseSchedule(nn.Module):
    """Cosine (Nichol & Dhariwal) schedule stored as buffers."""

    def __init__(self, num_steps: int = 100):
        super().__init__()
        self.num_steps = num_steps
        t = torch.linspace(0, 1, num_steps + 1)
        ab = cosine_alpha_bar(t)
        ab = ab / ab[0].clamp(min=1e-8)
        betas = (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        ab = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def to_x0(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        ab = self.alphas_cumprod[t].view(-1, *([1] * (xt.dim() - 1)))
        return (xt - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)


class SinusoidalTimestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        ang = t.float()[:, None] * freqs[None]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class EgoStateEncoder(nn.Module):
    """Ego speed (km/h), target point (m, ego frame), discrete nav command."""

    def __init__(self, dim: int, num_commands: int = 7):
        super().__init__()
        self.speed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.target = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.command = nn.Embedding(num_commands, dim)
        self.fuse = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, speed_kmh, target_point, command):
        s = self.speed(speed_kmh.view(-1, 1).float() / 30.0)  # ~unit scale at 30 km/h
        tp = self.target(target_point.float())
        c = self.command(command.long().clamp(min=0, max=self.command.num_embeddings - 1))
        return self.fuse(s + tp + c)


class DenoiserBlock(nn.Module):
    """Self-attn over waypoints -> cross-attn to geometry -> cross-attn to
    causation tokens -> FiLM-modulated FFN on (timestep + ego state)."""

    def __init__(self, dim: int, spatial_dim: int, sem_dim: int, heads: int = 8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.geo_kv = nn.Linear(spatial_dim, dim)
        self.geo_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.sem_kv = nn.Linear(sem_dim, dim)
        self.sem_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n4 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.film = nn.Linear(dim, dim * 2)

    def forward(self, x, geo, sem, cond):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        gk = self.geo_kv(geo)
        x = x + self.geo_attn(h, gk, gk, need_weights=False)[0]
        h = self.n3(x)
        sk = self.sem_kv(sem)
        x = x + self.sem_attn(h, sk, sk, need_weights=False)[0]
        scale, shift = self.film(cond).unsqueeze(1).chunk(2, dim=-1)
        x = x + self.ffn(self.n4(x) * (1 + scale) + shift)
        return x


class CoCDiffusionHead(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 256,
        sem_dim: int = 896,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        pred_len: int = 11,
        train_steps: int = 100,
        infer_steps: int = 10,
        num_commands: int = 7,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.infer_steps = infer_steps
        self.schedule = NoiseSchedule(train_steps)
        self.in_proj = nn.Linear(2, dim)
        self.wp_pos = nn.Parameter(torch.randn(1, pred_len, dim) * 0.02)
        self.t_embed = SinusoidalTimestep(dim)
        self.ego = EgoStateEncoder(dim, num_commands)
        self.blocks = nn.ModuleList(
            [DenoiserBlock(dim, spatial_dim, sem_dim, heads) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 2)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def denoise(self, xt, t, geo, sem, speed, target_point, command):
        cond = self.t_embed(t) + self.ego(speed, target_point, command)
        h = self.in_proj(xt) + self.wp_pos
        for blk in self.blocks:
            h = blk(h, geo, sem, cond)
        return self.out_proj(self.out_norm(h))  # predicted epsilon

    def loss(self, waypoints, geo, sem, speed, target_point, command,
             x0_clamp: float = 80.0):
        """Returns (per-sample eps-MSE, clamped x0 estimate, timesteps, x0 reliability).

        `reliability` is alpha_bar(t): the x0 recovered from a heavily-noised
        sample is near-meaningless (it divides by sqrt(alpha_bar) -> 0), so any
        downstream term that consumes x0 must down-weight those samples rather
        than treat them as predictions. The clamp bounds the same pathology --
        a 2.2 s horizon cannot physically exceed a few tens of metres.
        """
        b = waypoints.shape[0]
        t = torch.randint(0, self.schedule.num_steps, (b,), device=waypoints.device)
        noise = torch.randn_like(waypoints)
        xt = self.schedule.add_noise(waypoints, noise, t)
        eps = self.denoise(xt, t, geo, sem, speed, target_point, command)
        per_sample = F.mse_loss(eps, noise, reduction="none").mean(dim=(1, 2))
        x0 = self.schedule.to_x0(xt, eps, t).clamp(-x0_clamp, x0_clamp)
        reliability = self.schedule.alphas_cumprod[t]
        return per_sample, x0, t, reliability

    @torch.no_grad()
    def sample(self, geo, sem, speed, target_point, command, steps: int | None = None,
               generator: torch.Generator | None = None):
        """Deterministic DDIM (eta=0) over an evenly spaced step subsequence."""
        steps = steps or self.infer_steps
        b = geo.shape[0]
        dev = geo.device
        seq = torch.linspace(self.schedule.num_steps - 1, 0, steps).long().to(dev)
        x = torch.randn(b, self.pred_len, 2, device=dev, generator=generator)
        ac = self.schedule.alphas_cumprod
        for i, t_val in enumerate(seq):
            t = t_val.repeat(b)
            eps = self.denoise(x, t, geo, sem, speed, target_point, command)
            a_t = ac[t_val].clamp(min=1e-8)
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            if i + 1 < len(seq):
                a_prev = ac[seq[i + 1]].clamp(min=1e-8)
                x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            else:
                x = x0
        return x
