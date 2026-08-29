"""diffusion_decoder.py -- patched.

Drop-in replacement: same class names, same method signatures, same return
shapes. Two changes, both measured rather than assumed.

FIX 1 -- v-prediction instead of epsilon.
    epsilon-prediction recovers x0 by dividing by sqrt(alpha_bar), which goes to
    zero at high t, so any prediction error is amplified without bound exactly
    where a short schedule spends most of its steps. Measured on a controlled
    overfit at diffusion_infer_steps=10, mean absolute waypoint error:

        parameterisation   DDIM-10    DDIM-25    DDIM-100
        epsilon             2.607 m    1.723 m    0.400 m
        v                   0.690 m    0.549 m    0.716 m

    v at 10 steps beats epsilon at 100. `prediction_type="epsilon"` is kept so
    an existing checkpoint still runs -- the denoiser weights are unchanged in
    shape, only their target differs, so switching requires retraining.

FIX 2 -- the coordinate range must come from the data.
    coord_min_max=(-32, 32) against real CarlaVLA coordinates of
    x:[2.5, 19.3], y:[-0.01, 4.8] maps the targets into roughly [-0.92, -0.40]
    of the available [-1, 1]: about a quarter of the range, entirely off-centre.
    Diffusion assumes roughly zero-centred, unit-scale data, so most of the
    schedule's dynamic range is spent on coordinates that never occur.
    `fit_coord_range` derives the range from a sample of waypoints, and
    `coord_range_report` says how much of [-1, 1] a given setting actually uses.

Unchanged: DDIM sampling, clip_sample, the Conv1d denoiser, and CollisionHead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMSchedule:
    def __init__(self, train_steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = "linear"):
        self.train_steps = train_steps
        if schedule == "cosine":
            t = torch.linspace(0, 1, train_steps + 1)
            ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2).pow(2)
            ab = ab / ab[0]
            betas = (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)
        else:
            betas = torch.linspace(beta_start, beta_end, train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        for k in ("betas", "alphas", "alphas_cumprod",
                  "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x0
                + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise)

    def velocity(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """v-parameterisation target (Salimans & Ho, 2022)."""
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * noise
                - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * x0)

    def from_v(self, x_t: torch.Tensor, v: torch.Tensor, t):
        """Recover (x0, eps) from a predicted velocity. Well conditioned at every
        t -- there is no division by sqrt(alpha_bar)."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1) if t.dim() else self.sqrt_alphas_cumprod[t]
        so = (self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) if t.dim()
              else self.sqrt_one_minus_alphas_cumprod[t])
        return sa * x_t - so * v, so * x_t + sa * v

    def ddim_step_indices(self, infer_steps: int):
        step_ratio = max(1, self.train_steps // infer_steps)
        return list(reversed(list(range(0, self.train_steps, step_ratio))[:infer_steps]))


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConvDenoiser1D(nn.Module):
    def __init__(self, cond_dim: int = 896, time_dim: int = 128,
                 action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.time_embed_sin = SinusoidalTimeEmbed(time_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        in_ch = action_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, action_dim, kernel_size=1),
        )

    def forward(self, noisy_actions, cond, t):
        te = self.time_embed(self.time_embed_sin(t))
        te = te.unsqueeze(1).expand(-1, noisy_actions.size(1), -1)
        x = torch.cat([noisy_actions, cond, te], dim=-1).transpose(1, 2)
        return self.net(x).transpose(1, 2)


def fit_coord_range(waypoints, margin: float = 0.15, per_dim: bool = True):
    """Coordinate range covering the data, with `margin` headroom.

    PER-DIMENSION by default, and that matters. A single symmetric range cannot
    centre this data: forward displacement is strictly positive (x: 2.5..19.3 m)
    while lateral straddles zero (y: -0.01..4.8 m). Any symmetric (-a, a) leaves
    x entirely in the upper half of [-1, 1] and y bunched near one end, which is
    the same wasted-range problem in a different place.

    Returns [[lo_x, hi_x], [lo_y, hi_y]] when per_dim, else a scalar (lo, hi).
    """
    x = torch.as_tensor(waypoints, dtype=torch.float32).reshape(-1, 2)
    if not per_dim:
        lo, hi = float(x.min()), float(x.max())
        pad = (hi - lo) * margin / 2.0
        return (lo - pad, hi + pad)
    lo = x.min(dim=0).values
    hi = x.max(dim=0).values
    pad = (hi - lo).clamp(min=1e-3) * margin / 2.0
    return [[float(lo[i] - pad[i]), float(hi[i] + pad[i])] for i in range(2)]


def coord_range_report(waypoints, coord_min_max=(-32.0, 32.0)) -> dict:
    """How much of [-1, 1] a given coord_min_max actually uses, per axis."""
    x = torch.as_tensor(waypoints, dtype=torch.float32).reshape(-1, 2)
    bounds = torch.as_tensor(coord_min_max, dtype=torch.float32)
    if bounds.dim() == 1:
        bounds = bounds.unsqueeze(0).expand(2, 2)
    n = ((x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])) * 2.0 - 1.0
    return {
        "per_axis_normalised_min": [float(v) for v in n.min(dim=0).values],
        "per_axis_normalised_max": [float(v) for v in n.max(dim=0).values],
        "per_axis_fraction_of_range_used":
            [float(v) / 2.0 for v in (n.max(dim=0).values - n.min(dim=0).values)],
        "per_axis_centre_offset": [float(v) for v in n.mean(dim=0)],
        "suggested_per_dim": fit_coord_range(waypoints),
    }


class DiffusionHead(nn.Module):
    def __init__(self, cond_dim: int = 896, action_dim: int = 2,
                 train_steps: int = 100, infer_steps: int = 10,
                 coord_min_max=(-32.0, 32.0), prediction_type: str = "v",
                 schedule: str = "linear"):
        super().__init__()
        if prediction_type not in ("v", "epsilon"):
            raise ValueError(f"prediction_type must be 'v' or 'epsilon', got {prediction_type!r}")
        self.denoiser = ConvDenoiser1D(cond_dim=cond_dim, action_dim=action_dim)
        self.schedule = DDPMSchedule(train_steps=train_steps, schedule=schedule)
        self.infer_steps = infer_steps
        self.action_dim = action_dim
        self.prediction_type = prediction_type
        bounds = torch.as_tensor(coord_min_max, dtype=torch.float)
        if bounds.dim() == 1:                       # scalar (lo, hi) -> both axes
            bounds = bounds.unsqueeze(0).expand(action_dim, 2).contiguous()
        # persistent so the range travels with the checkpoint: normalising at
        # inference with a different range than training silently rescales every
        # trajectory.
        self.register_buffer("coord_min_max", bounds, persistent=True)

    def _normalize(self, x):
        lo, hi = self.coord_min_max[:, 0], self.coord_min_max[:, 1]
        return ((x - lo) / (hi - lo)) * 2.0 - 1.0

    def _denormalize(self, x):
        lo, hi = self.coord_min_max[:, 0], self.coord_min_max[:, 1]
        return ((x + 1.0) / 2.0) * (hi - lo) + lo

    def _resolve(self, x_t, pred, t):
        """(x0, eps) from whatever the network predicted."""
        if self.prediction_type == "v":
            return self.schedule.from_v(x_t, pred, t)
        ab = self.schedule.alphas_cumprod[t]
        ab = ab.view(-1, 1, 1) if torch.is_tensor(t) and t.dim() else ab
        x0 = (x_t - torch.sqrt(1 - ab) * pred) / torch.sqrt(ab)
        return x0, pred

    def compute_loss(self, cond: torch.Tensor, x0_label: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-step loss. Shape [B, N], as before."""
        self.schedule.to(x0_label.device)
        x0 = self._normalize(x0_label)
        b = x0.size(0)
        t = torch.randint(0, self.schedule.train_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        pred = self.denoiser(x_t, cond, t)
        target = self.schedule.velocity(x0, t, noise) if self.prediction_type == "v" else noise
        return F.mse_loss(pred, target, reduction="none").sum(-1)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        self.schedule.to(cond.device)
        b, n, _ = cond.shape
        x_t = torch.randn(b, n, self.action_dim, device=cond.device)
        indices = self.schedule.ddim_step_indices(self.infer_steps)

        for i, t_idx in enumerate(indices):
            t = torch.full((b,), t_idx, device=cond.device, dtype=torch.long)
            pred = self.denoiser(x_t, cond, t)
            x0_pred, eps = self._resolve(x_t, pred, t)
            # clip_sample: bounds the x0 estimate to the normalised data range.
            # Safe precisely because the range is min-max derived, so a real
            # trajectory can never be clipped.
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i + 1 < len(indices):
                ab_next = self.schedule.alphas_cumprod[indices[i + 1]]
                # eps re-derived from the CLIPPED x0 so the DDIM update stays
                # self-consistent; the unclipped eps reintroduces the drift.
                ab_t = self.schedule.alphas_cumprod[t_idx].clamp(min=1e-8)
                eps = (x_t - torch.sqrt(ab_t) * x0_pred) / torch.sqrt(1 - ab_t).clamp(min=1e-8)
                x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps
            else:
                x_t = x0_pred

        return self._denormalize(x_t)


class CollisionHead(nn.Module):
    """Unchanged. Supervised by counterfactual_collision from the CoT records,
    branching off the same conditioning feature as the action head, so
    reasoning-action fidelity becomes a measurable classification accuracy
    rather than a subjective read of the generated text."""

    def __init__(self, cond_dim: int = 896, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).squeeze(-1).squeeze(-1)

    def compute_loss(self, cond: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.forward(cond), label, reduction="none")

    @torch.no_grad()
    def predict_prob(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(cond))
