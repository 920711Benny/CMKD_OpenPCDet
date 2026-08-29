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

from .attention import SDPAAttention


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

    def velocity(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """v-parameterisation target (Salimans & Ho, 2022)."""
        ab = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * noise - (1 - ab).sqrt() * x0

    def from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """Recover (x0, eps) from a predicted velocity."""
        ab = self.alphas_cumprod[t].view(-1, *([1] * (xt.dim() - 1)))
        sqrt_ab, sqrt_1mab = ab.sqrt(), (1 - ab).sqrt()
        x0 = sqrt_ab * xt - sqrt_1mab * v
        eps = sqrt_1mab * xt + sqrt_ab * v
        return x0, eps


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
        self.self_attn = SDPAAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.geo_kv = nn.Linear(spatial_dim, dim)
        self.geo_attn = SDPAAttention(dim, heads)
        self.n3 = nn.LayerNorm(dim)
        self.sem_kv = nn.Linear(sem_dim, dim)
        self.sem_attn = SDPAAttention(dim, heads)
        self.n4 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.film = nn.Linear(dim, dim * 2)

    def forward(self, x, geo, sem, cond):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h)[0]
        h = self.n2(x)
        gk = self.geo_kv(geo)
        x = x + self.geo_attn(h, gk, gk)[0]
        h = self.n3(x)
        sk = self.sem_kv(sem)
        x = x + self.sem_attn(h, sk, sk)[0]
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
        wp_offset: tuple[float, float] = (20.0, 0.0),
        wp_scale: tuple[float, float] = (20.0, 15.0),
        clip_denoised: float = 1.0,
        prediction_type: str = "v",
    ):
        super().__init__()
        if prediction_type not in ("v", "epsilon"):
            raise ValueError(f"prediction_type must be 'v' or 'epsilon', got {prediction_type!r}")
        self.pred_len = pred_len
        self.infer_steps = infer_steps
        self.clip_denoised = clip_denoised
        # v-parameterisation by default. With eps-prediction the x0 estimate at
        # low SNR divides by sqrt(alpha_bar) -> 0, so a 10-step DDIM schedule --
        # the one this agent must hit to stay above 10 Hz -- is markedly worse
        # than a 100-step one. v-prediction is well conditioned across the whole
        # schedule, which is what makes the short schedule usable.
        self.prediction_type = prediction_type
        self.schedule = NoiseSchedule(train_steps)
        # Waypoints arrive in METRES. A diffusion schedule assumes data on a
        # roughly unit scale: on raw metres the signal dominates the noise at
        # nearly every timestep, so eps-prediction collapses to "return the
        # input" -- training loss goes low while sampling from pure noise
        # diverges. Targets are mapped affinely into [-1, 1]:
        #
        #     normalised = (waypoints - offset) / scale
        #
        # Min-max rather than std normalisation on purpose: it lets the sampler
        # clip its x0 estimate at +-1 without ever truncating a valid trajectory,
        # which std-based scaling cannot promise (a 30 m waypoint is 6 sigma).
        # Both are buffers, so they travel with the checkpoint -- training and
        # inference MUST agree on them.
        self.register_buffer("wp_offset", torch.tensor(wp_offset, dtype=torch.float32))
        self.register_buffer("wp_scale", torch.tensor(wp_scale, dtype=torch.float32))
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
        """Network output: velocity or epsilon, per `prediction_type`."""
        cond = self.t_embed(t) + self.ego(speed, target_point, command)
        h = self.in_proj(xt) + self.wp_pos
        for blk in self.blocks:
            h = blk(h, geo, sem, cond)
        return self.out_proj(self.out_norm(h))

    def _resolve(self, xt, pred, t):
        """(x0, eps) from whatever the network predicted."""
        if self.prediction_type == "v":
            return self.schedule.from_v(xt, pred, t)
        return self.schedule.to_x0(xt, pred, t), pred

    def loss(self, waypoints, geo, sem, speed, target_point, command,
             x0_clamp: float = 80.0):
        """Returns (per-sample MSE on the prediction target, x0 in metres,
        timesteps, x0 reliability).

        The MSE is against velocity or epsilon per `prediction_type`.

        `reliability` is alpha_bar(t). An x0 recovered from a heavily-noised
        sample carries little information about the true trajectory whichever
        parameterisation is used, so any downstream term consuming x0 -- the
        causal consistency loss -- must down-weight those samples rather than
        treat them as predictions. The clamp is a second bound: a 2.2 s horizon
        cannot physically exceed a few tens of metres.
        """
        b = waypoints.shape[0]
        target = self.normalize(waypoints)
        t = torch.randint(0, self.schedule.num_steps, (b,), device=waypoints.device)
        noise = torch.randn_like(target)
        xt = self.schedule.add_noise(target, noise, t)
        pred = self.denoise(xt, t, geo, sem, speed, target_point, command)
        goal = self.schedule.velocity(target, noise, t) if self.prediction_type == "v" else noise
        per_sample = F.mse_loss(pred, goal, reduction="none").mean(dim=(1, 2))
        # x0 is handed back in METRES so downstream kinematics stay physical.
        x0_norm, _ = self._resolve(xt, pred, t)
        x0 = self.denormalize(x0_norm).clamp(-x0_clamp, x0_clamp)
        reliability = self.schedule.alphas_cumprod[t]
        return per_sample, x0, t, reliability

    def normalize(self, waypoints: torch.Tensor) -> torch.Tensor:
        return (waypoints - self.wp_offset) / self.wp_scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.wp_scale + self.wp_offset

    @torch.no_grad()
    def sample(self, geo, sem, speed, target_point, command, steps: int | None = None,
               generator: torch.Generator | None = None):
        """Deterministic DDIM (eta=0) over an evenly spaced step subsequence.

        Returns waypoints in METRES, denormalised from the [-1, 1] training space.
        """
        steps = steps or self.infer_steps
        b = geo.shape[0]
        dev = geo.device
        seq = torch.linspace(self.schedule.num_steps - 1, 0, steps).long().to(dev)
        x = torch.randn(b, self.pred_len, 2, device=dev, generator=generator)
        ac = self.schedule.alphas_cumprod
        for i, t_val in enumerate(seq):
            t = t_val.repeat(b)
            pred = self.denoise(x, t, geo, sem, speed, target_point, command)
            x0, eps = self._resolve(x, pred, t)
            # Clip the x0 estimate to the data range at every step. Early steps
            # sit at near-zero alpha_bar, where a small prediction error moves x0
            # a long way; unclipped DDIM diverges on an imperfect model even when
            # the training loss is low. The min-max normalisation is what makes
            # this clip safe -- real waypoints are guaranteed inside [-1, 1].
            if self.clip_denoised:
                x0 = x0.clamp(-self.clip_denoised, self.clip_denoised)
                # Re-derive eps from the clipped x0 so the DDIM update stays
                # self-consistent; using the unclipped eps reintroduces the drift.
                a_t = ac[t_val].clamp(min=1e-8)
                eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp(min=1e-8)
            if i + 1 < len(seq):
                a_prev = ac[seq[i + 1]].clamp(min=1e-8)
                x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            else:
                x = x0
        return self.denormalize(x)
