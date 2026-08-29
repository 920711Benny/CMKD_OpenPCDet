"""CARLA leaderboard agent.

Entry point used by `leaderboard_evaluator.py --agent .../agent.py`. Subclasses
`AutonomousAgent` when the leaderboard package is importable and degrades to a
standalone base otherwise, so the module stays importable (and testable) outside
a CARLA installation.

Environment:
    SUB1B_CONFIG      path to the YAML config (required)
    SUB1B_CHECKPOINT  path to the trained checkpoint (required for real runs)
    SUB1B_HUD         "1" to open the pygame HUD (default on)
    SUB1B_HUD_DUMP    directory for PNG frames when no display is available
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..data.augment import cut_bottom_quarter, resize_nn, to_chw_normalized
from ..utils import load_config, setup_scratch_dirs
from .async_pipeline import AsyncVLARuntime
from .controller import TrajectoryController
from .hud import HUDState, PygameHUD
from .sensors import CameraRig, build_sensor_list

try:  # pragma: no cover - only present inside a CARLA leaderboard install
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

    _HAS_LEADERBOARD = True
except ImportError:  # pragma: no cover
    _HAS_LEADERBOARD = False

    class AutonomousAgent:  # minimal stand-in
        def setup(self, path_to_conf_file):
            ...

        def sensors(self):
            return []

        def run_step(self, input_data, timestamp):
            raise NotImplementedError

        def destroy(self):
            ...

    class Track:  # noqa: D101
        SENSORS = "SENSORS"


# CARLA RoadOption -> the command ids used during training.
ROAD_OPTION_TO_COMMAND = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
DEFAULT_COMMAND = 3  # LANEFOLLOW


class Sub1BVLAAgent(AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        cfg_path = path_to_conf_file or os.environ.get("SUB1B_CONFIG")
        if not cfg_path:
            raise RuntimeError("Set SUB1B_CONFIG (or pass a config) for Sub1BVLAAgent.")
        self.cfg = load_config(cfg_path)
        setup_scratch_dirs(self.cfg)
        self.track = Track.SENSORS
        self.rig = CameraRig.from_config(self.cfg)
        self.image_size = self.cfg["model"]["image_size"]
        self.cut_bottom = self.cfg["data"].get("cut_bottom_quarter", True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_agent_model(self.cfg, os.environ.get("SUB1B_CHECKPOINT"), device)
        self.runtime = AsyncVLARuntime(self.model, device, self.cfg).start()
        self.controller = TrajectoryController(dt=self.cfg["model"].get("waypoint_dt", 0.2))

        self.hud = None
        if os.environ.get("SUB1B_HUD", "1") == "1":
            self.hud = PygameHUD(self.rig, dump_dir=os.environ.get("SUB1B_HUD_DUMP"))
        self.step_idx = 0
        self.last_waypoints = np.zeros((self.cfg["model"]["pred_len"], 2), dtype=np.float32)

    def sensors(self):
        return build_sensor_list(self.cfg)

    # ---- per-tick --------------------------------------------------------
    def run_step(self, input_data, timestamp):
        import carla  # noqa: PLC0415 - only available inside the simulator

        rgb = input_data["rgb_front"][1][:, :, :3][:, :, ::-1]  # BGRA -> RGB
        speed_kmh = float(input_data.get("speed", (0, {"speed": 0.0}))[1]["speed"]) * 3.6
        target_point, command = self._route_target()

        chw = self._preprocess(np.asarray(rgb, dtype=np.float32) / 255.0)
        p = self.runtime.perceive(chw, speed_kmh, target_point, command)
        self.last_waypoints = p.waypoints

        ctrl = self.controller.step(p.waypoints, speed_kmh / 3.6)
        self.step_idx += 1

        if self.hud is not None:
            if not self.hud.poll():
                raise KeyboardInterrupt("HUD closed by operator")
            rationale, r_frame = self.runtime.latest_rationale()
            self.hud.render(HUDState(
                frame=np.asarray(rgb, dtype=np.float32) / 255.0,
                waypoints=p.waypoints,
                spatial_attn=p.spatial_attn,
                semantic_attn=p.semantic_attn,
                rationale=rationale,
                intent=p.intent_name,
                speed_kmh=speed_kmh,
                control={"throttle": ctrl.throttle, "brake": ctrl.brake,
                         "steer": ctrl.steer, "lookahead": ctrl.lookahead},
                latency_ms=p.latency_ms,
                fps=1000.0 / max(p.latency_ms, 1e-3),
                rationale_age_frames=max(0, p.frame_id - int(r_frame)),
            ))

        return carla.VehicleControl(throttle=ctrl.throttle, steer=ctrl.steer, brake=ctrl.brake)

    def _preprocess(self, rgb01: np.ndarray) -> np.ndarray:
        img = cut_bottom_quarter(rgb01) if self.cut_bottom else rgb01
        return to_chw_normalized(resize_nn(img, self.image_size))

    def _route_target(self):
        """Next route waypoint in the ego frame + the discrete nav command
        (`route_as: target_point_command`)."""
        plan = getattr(self, "_global_plan_world_coord", None)
        if not plan:
            return np.zeros(2, dtype=np.float32), DEFAULT_COMMAND
        idx = min(self.step_idx // 10, len(plan) - 1)
        wp, road_option = plan[idx]
        cmd = ROAD_OPTION_TO_COMMAND.get(int(getattr(road_option, "value", road_option)),
                                         DEFAULT_COMMAND)
        return np.array([wp.location.x, wp.location.y], dtype=np.float32), cmd

    def destroy(self):
        if getattr(self, "runtime", None) is not None:
            report = self.runtime.latency_report()
            Path("runs").mkdir(exist_ok=True)
            (Path("runs") / "latency_report.json").write_text(str(report))
            self.runtime.stop()
        if getattr(self, "hud", None) is not None:
            self.hud.close()


def load_agent_model(cfg: dict, checkpoint: str | None, device):
    """Build the model and load trained weights.

    Refuses to silently drive on random weights: without a checkpoint the caller
    must opt in explicitly via SUB1B_ALLOW_UNTRAINED=1.
    """
    from ..models.vla_agent import DualHeadDiffusionVLA  # noqa: PLC0415

    model = DualHeadDiffusionVLA(cfg)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        trained = set(state["model"])
        skipped = [n for n in missing if n in trained]
        print(f"[agent] loaded {len(trained)} trained tensors from {checkpoint}; "
              f"{len(unexpected)} unexpected, {len(skipped)} expected-but-missing.")
    elif os.environ.get("SUB1B_ALLOW_UNTRAINED") != "1":
        raise RuntimeError(
            "No SUB1B_CHECKPOINT given. Driving on untrained weights produces "
            "meaningless benchmark numbers. Set SUB1B_ALLOW_UNTRAINED=1 to override."
        )
    return model.to(device).eval()


def get_entry_point():  # leaderboard hook
    return "Sub1BVLAAgent"
