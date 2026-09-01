"""Sensor rig definition and camera geometry.

Front RGB rig (CARLA standard, matches the training data collection):
    x = 1.3 m, y = 0.0 m, z = 2.3 m, pitch = -5 deg, roll = 0, yaw = 0,
    FOV = 100 deg, 1024 x 512.

The projection helpers are what let the HUD draw a BEV-frame trajectory back
onto the camera image, so what the operator sees is the trajectory the
controller is actually tracking -- not a separately-computed illustration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CameraRig:
    x: float = 1.3
    y: float = 0.0
    z: float = 2.3
    pitch: float = -5.0
    roll: float = 0.0
    yaw: float = 0.0
    fov: float = 100.0
    width: int = 1024
    height: int = 512

    @classmethod
    def from_config(cls, cfg: dict) -> "CameraRig":
        return cls(**{k: v for k, v in cfg["sensors"]["rgb_front"].items()
                      if k in cls.__dataclass_fields__})

    def to_carla_sensor_spec(self) -> dict:
        """Dict in the CARLA leaderboard `sensors()` format."""
        return {
            "type": "sensor.camera.rgb", "id": "rgb_front",
            "x": self.x, "y": self.y, "z": self.z,
            "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw,
            "width": self.width, "height": self.height, "fov": self.fov,
        }

    @property
    def focal(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov) / 2.0))

    def intrinsics(self) -> np.ndarray:
        f = self.focal
        return np.array([[f, 0.0, self.width / 2.0],
                         [0.0, f, self.height / 2.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def project_ego_to_image(self, points_xy: np.ndarray, z_ground: float = 0.0):
        """Project ego-frame BEV points (forward, left) onto the image plane.

        Returns (pixels, valid_mask). Points behind the camera are marked invalid
        rather than silently wrapping to the opposite side of the frame.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        # Ego (x fwd, y left, z up) -> camera-mounted frame.
        xf = pts[:, 0] - self.x
        yl = pts[:, 1] - self.y
        zu = np.full_like(xf, z_ground - self.z)

        p = math.radians(self.pitch)
        # Rotate about the camera's lateral axis by -pitch.
        x_cam = xf * math.cos(p) + zu * math.sin(p)
        z_cam = -xf * math.sin(p) + zu * math.cos(p)
        y_cam = yl

        valid = x_cam > 0.1
        depth = np.where(valid, x_cam, 1.0)
        f = self.focal
        # +y is left in the ego frame but image u grows rightward, hence the sign.
        u = self.width / 2.0 - f * (y_cam / depth)
        # z_cam is negative for ground points (below the camera); image v grows downward.
        v = self.height / 2.0 - f * (z_cam / depth)
        px = np.stack([u, v], axis=-1)
        valid &= (px[:, 0] > -self.width) & (px[:, 0] < 2 * self.width)
        return px, valid


def build_sensor_list(cfg: dict) -> list[dict]:
    """Full sensor suite handed to the CARLA leaderboard."""
    rig = CameraRig.from_config(cfg)
    return [
        rig.to_carla_sensor_spec(),
        {"type": "sensor.other.imu", "id": "imu", "x": 0.0, "y": 0.0, "z": 0.0,
         "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05},
        {"type": "sensor.speedometer", "id": "speed", "reading_frequency": 20},
        {"type": "sensor.opendrive_map", "id": "opendrive", "reading_frequency": 1e-6},
    ]
