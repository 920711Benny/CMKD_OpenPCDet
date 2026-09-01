"""Causal Consistency Loss.

Couples the language stream's stated intent to the dynamics of the trajectory
the diffusion head actually produced.

Formulation. For each canonical intent k we define a differentiable *violation*
V_k(tau) >= 0 that is zero exactly when trajectory tau exhibits the dynamics the
intent claims. The loss is the violation expected under the language model's
intent distribution:

    L_consistency = sum_k  p_k(language)  *  V_k(tau)

Gradients flow into BOTH streams, which is the point:
  * through p_k  -> the LM is pushed away from intents the trajectory contradicts;
  * through V_k  -> the trajectory is pushed to satisfy the believed intent.

Neither stream is treated as ground truth for the other, so the term is a genuine
mutual-agreement penalty rather than a one-way distillation.

Frame convention (ego / BEV, CARLA-aligned):
    waypoints[..., 0] = forward displacement (metres, +ahead)
    waypoints[..., 1] = lateral  displacement (metres, +left)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.coc_prompt import INTENTS


@dataclass
class TrajectoryDynamics:
    speed: torch.Tensor        # (B, T-1) instantaneous speed, m/s
    accel: torch.Tensor        # (B,)     mean longitudinal acceleration, m/s^2
    final_speed: torch.Tensor  # (B,)     robust end speed (mean of last third)
    terminal_speed: torch.Tensor  # (B,)  speed over the LAST segment only
    displacement: torch.Tensor # (B,)     total path length, m
    heading_change: torch.Tensor  # (B,)  net heading change, rad (+left)
    lateral_offset: torch.Tensor  # (B,)  lateral offset at horizon, m (+left)


def compute_dynamics(waypoints: torch.Tensor, dt: float = 0.2) -> TrajectoryDynamics:
    """Differentiable kinematics from a waypoint sequence.

    `waypoints` is (B, T, 2) starting at the ego origin at t=0. A leading origin
    is prepended so the first segment measures motion away from the ego.
    """
    b = waypoints.shape[0]
    origin = waypoints.new_zeros(b, 1, 2)
    pts = torch.cat([origin, waypoints], dim=1)
    delta = pts[:, 1:] - pts[:, :-1]                      # (B, T, 2)
    seg = delta.norm(dim=-1)                              # (B, T)
    speed = seg / dt
    # Robust endpoint speeds: average the first/last third to damp diffusion jitter.
    k = max(1, speed.shape[1] // 3)
    v0 = speed[:, :k].mean(dim=1)
    v1 = speed[:, -k:].mean(dim=1)
    horizon = max(dt, dt * (speed.shape[1] - k))
    accel = (v1 - v0) / horizon
    heading = torch.atan2(delta[..., 1], delta[..., 0].abs().clamp(min=1e-3))
    heading_change = heading[:, -1] - heading[:, 0]
    return TrajectoryDynamics(
        speed=speed,
        accel=accel,
        # `final_speed` averages the last third: robust, and right for judging
        # acceleration trends. It is WRONG for judging arrival at rest -- braking
        # from 45 km/h to a stop line still averages ~1 m/s over that window, so a
        # genuine stop would never register. `terminal_speed` is the last segment
        # alone, which is what "ended at rest" actually means.
        terminal_speed=speed[:, -1],
        final_speed=v1,
        displacement=seg.sum(dim=1),
        heading_change=heading_change,
        lateral_offset=waypoints[:, -1, 1],
    )


def _sq_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).pow(2)


@dataclass
class ConsistencyThresholds:
    stop_speed: float = 0.3        # m/s below which we call the ego at rest
    decel: float = -0.5            # m/s^2 required to count as decelerating
    accel: float = 0.5
    keep_accel_band: float = 1.0   # |a| tolerated while holding speed
    turn_heading: float = 0.15     # rad of net heading change for a turn
    lane_change_lateral: float = 1.5   # m of lateral offset for a lane change
    straight_lateral: float = 1.0      # m tolerated while going straight


def intent_violations(
    dyn: TrajectoryDynamics, th: ConsistencyThresholds | None = None
) -> torch.Tensor:
    """Return (B, num_intents) non-negative violation scores."""
    th = th or ConsistencyThresholds()
    v: dict[str, torch.Tensor] = {}

    # Longitudinal.
    # "stop" is satisfied by COMING TO REST within the horizon, not by standing
    # still: braking from 45 km/h to zero covers real ground, and penalising that
    # displacement would make every legitimate stop-line approach a violation.
    v["stop"] = _sq_relu(dyn.terminal_speed - th.stop_speed)
    v["decelerate"] = _sq_relu(dyn.accel - th.decel)
    v["accelerate"] = _sq_relu(th.accel - dyn.accel)
    v["keep_speed"] = (
        _sq_relu(dyn.accel.abs() - th.keep_accel_band)
        + _sq_relu(dyn.lateral_offset.abs() - th.straight_lateral)
    )

    # Lateral. A turn needs heading change; a lane change needs lateral offset
    # while heading returns to roughly parallel -- that distinction is what stops
    # the model from labelling every swerve a "turn".
    v["turn_left"] = _sq_relu(th.turn_heading - dyn.heading_change)
    v["turn_right"] = _sq_relu(dyn.heading_change + th.turn_heading)
    v["lane_change_left"] = (
        _sq_relu(th.lane_change_lateral - dyn.lateral_offset)
        + _sq_relu(dyn.heading_change.abs() - th.turn_heading * 2)
    )
    v["lane_change_right"] = (
        _sq_relu(dyn.lateral_offset + th.lane_change_lateral)
        + _sq_relu(dyn.heading_change.abs() - th.turn_heading * 2)
    )

    missing = [k for k in INTENTS if k not in v]
    if missing:
        raise KeyError(f"No violation defined for intents: {missing}")
    return torch.stack([v[k] for k in INTENTS], dim=1)


def causal_consistency_loss(
    intent_logits: torch.Tensor,
    waypoints: torch.Tensor,
    dt: float = 0.2,
    thresholds: ConsistencyThresholds | None = None,
    temperature: float = 1.0,
    reduction: str = "mean",
):
    """Expected dynamic violation under the language intent distribution."""
    dyn = compute_dynamics(waypoints, dt=dt)
    viol = intent_violations(dyn, thresholds)               # (B, K)
    p = F.softmax(intent_logits / temperature, dim=-1)      # (B, K)
    per_sample = (p * viol).sum(dim=-1)
    if reduction == "none":
        return per_sample, dyn, viol
    return per_sample.mean(), dyn, viol


@torch.no_grad()
def action_cot_alignment_score(
    intent_ids: torch.Tensor,
    waypoints: torch.Tensor,
    dt: float = 0.2,
    thresholds: ConsistencyThresholds | None = None,
    tol: float = 0.1,
) -> torch.Tensor:
    """Reported metric: fraction of samples whose executed trajectory satisfies
    the intent the model stated (violation within `tol`). This is the evaluation
    counterpart of the training loss and uses the *decoded* intent, not the soft
    distribution."""
    dyn = compute_dynamics(waypoints, dt=dt)
    viol = intent_violations(dyn, thresholds)
    chosen = viol.gather(1, intent_ids.view(-1, 1).long()).squeeze(1)
    return (chosen <= tol).float()
