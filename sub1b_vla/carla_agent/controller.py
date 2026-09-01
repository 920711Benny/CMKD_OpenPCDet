"""Low-level controller: PID longitudinal + pure-pursuit lateral.

Consumes the diffusion head's waypoints directly. Kept deliberately simple and
stateless-per-tick apart from the PID integrator, because the interesting
behaviour must live in the learned trajectory, not in hand-tuned control logic
that would confound the benchmark comparison.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ControlOutput:
    throttle: float
    steer: float
    brake: float
    target_speed: float
    lookahead: tuple[float, float]


class PIDLongitudinal:
    def __init__(self, kp=1.0, ki=0.05, kd=0.2, window=20, dt=0.05):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.errors: deque[float] = deque(maxlen=window)

    def step(self, target_speed: float, current_speed: float) -> float:
        err = target_speed - current_speed
        self.errors.append(err)
        d = (self.errors[-1] - self.errors[-2]) / self.dt if len(self.errors) > 1 else 0.0
        i = sum(self.errors) * self.dt
        return float(np.clip(self.kp * err + self.ki * i + self.kd * d, -1.0, 1.0))

    def reset(self):
        self.errors.clear()


class PurePursuitLateral:
    """Frame/sign convention, which is easy to get backwards and expensive to
    get wrong: waypoints use +y = LEFT, while CARLA's VehicleControl.steer uses
    +1 = full RIGHT. `steer_sign` = -1 performs that conversion, so a
    left-bearing trajectory yields a negative (left) steering command."""

    def __init__(self, wheelbase=2.87, k_gain=1.0, min_lookahead=2.5,
                 max_steer_rad=1.22, steer_sign: float = -1.0):
        self.wheelbase = wheelbase
        self.k_gain = k_gain
        self.min_lookahead = min_lookahead
        self.max_steer_rad = max_steer_rad
        self.steer_sign = steer_sign

    def pick_lookahead(self, waypoints: np.ndarray, speed_ms: float):
        ld = max(self.min_lookahead, self.k_gain * speed_ms)
        dists = np.linalg.norm(waypoints, axis=1)
        idx = int(np.argmin(np.abs(dists - ld)))
        return waypoints[idx], ld

    def step(self, waypoints: np.ndarray, speed_ms: float) -> tuple[float, tuple]:
        pt, ld = self.pick_lookahead(waypoints, speed_ms)
        x, y = float(pt[0]), float(pt[1])
        ld_eff = max(math.hypot(x, y), 1e-3)
        alpha = math.atan2(y, max(x, 1e-3))
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld_eff)
        steer = float(np.clip(self.steer_sign * delta / self.max_steer_rad, -1.0, 1.0))
        return steer, (x, y)


class TrajectoryController:
    """Waypoints (ego BEV, metres) -> CARLA VehicleControl values."""

    def __init__(self, dt: float = 0.2, control_dt: float = 0.05,
                 max_speed_kmh: float = 40.0, brake_speed_thresh: float = 0.35,
                 steer_sign: float = -1.0):
        self.dt = dt
        self.lon = PIDLongitudinal(dt=control_dt)
        self.lat = PurePursuitLateral(steer_sign=steer_sign)
        self.max_speed_ms = max_speed_kmh / 3.6
        self.brake_speed_thresh = brake_speed_thresh

    def target_speed_from(self, waypoints: np.ndarray) -> float:
        """Desired speed implied by the waypoint spacing over the first second."""
        n = min(len(waypoints), max(1, int(round(1.0 / self.dt))))
        pts = np.vstack([np.zeros((1, 2)), waypoints[:n]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(min(seg.mean() / self.dt, self.max_speed_ms))

    def step(self, waypoints: np.ndarray, current_speed_ms: float) -> ControlOutput:
        wp = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
        target = self.target_speed_from(wp)
        steer, look = self.lat.step(wp, current_speed_ms)
        accel = self.lon.step(target, current_speed_ms)

        throttle, brake = (max(0.0, accel), 0.0) if accel >= 0 else (0.0, min(1.0, -accel))
        # Hard stop: a trajectory that asks for near-zero speed must brake, not
        # coast. This is what enforces stop-line adherence at red lights.
        if target < self.brake_speed_thresh:
            throttle, brake = 0.0, 1.0
            self.lon.reset()
        return ControlOutput(throttle=float(throttle), steer=steer, brake=float(brake),
                             target_speed=target, lookahead=look)
