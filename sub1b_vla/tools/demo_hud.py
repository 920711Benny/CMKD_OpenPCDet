"""Offline HUD + async-runtime demo.

Drives the full inference stack over synthetic frames without CARLA: dual-head
encode -> 10-step DDIM -> controller -> HUD composite, with the rationale worker
running concurrently. Useful for checking the Output Separation Protocol and for
producing figures.

    python -m sub1b_vla.tools.demo_hud --config ... --checkpoint ... --frames 40 \
        --out runs/hud_frames
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..carla_agent.async_pipeline import AsyncVLARuntime
from ..carla_agent.controller import TrajectoryController
from ..carla_agent.hud import HUDState, PygameHUD
from ..carla_agent.sensors import CameraRig
from ..data.augment import cut_bottom_quarter, resize_nn, to_chw_normalized
from ..data.carla_surrogate import generate_frame


def main():
    from ..carla_agent.agent import load_agent_model  # noqa: PLC0415
    from ..utils import load_config  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="runs/hud_frames")
    ap.add_argument("--no-cot", action="store_true", help="disable the rationale worker")
    ap.add_argument("--latency-out", default="runs/latency_report.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_agent_model(cfg, args.checkpoint, device)
    rig = CameraRig.from_config(cfg)
    ctrl = TrajectoryController(dt=cfg["model"].get("waypoint_dt", 0.2))
    hud = PygameHUD(rig, dump_dir=args.out)
    rng = np.random.default_rng(0)
    image_size = cfg["model"]["image_size"]
    cut = cfg["data"].get("cut_bottom_quarter", True)

    per_scenario: dict[str, list] = {}
    with AsyncVLARuntime(model, device, cfg, cot_enabled=not args.no_cot) as rt:
        for _ in range(args.frames):
            f = generate_frame(rng, cfg["model"]["pred_len"], cfg["model"].get("waypoint_dt", 0.2))
            img = cut_bottom_quarter(f.image) if cut else f.image
            chw = to_chw_normalized(resize_nn(img, image_size))
            p = rt.perceive(chw, f.speed_kmh, f.target_point, f.command)
            c = ctrl.step(p.waypoints, f.speed_kmh / 3.6)
            rationale, r_frame = rt.latest_rationale()
            hud.render(HUDState(
                frame=f.image, waypoints=p.waypoints,
                spatial_attn=p.spatial_attn, semantic_attn=p.semantic_attn,
                rationale=rationale, intent=p.intent_name, speed_kmh=f.speed_kmh,
                control={"throttle": c.throttle, "brake": c.brake, "steer": c.steer,
                         "lookahead": c.lookahead},
                latency_ms=p.latency_ms, fps=1000.0 / max(p.latency_ms, 1e-3),
                rationale_age_frames=max(0, p.frame_id - int(r_frame)),
            ))
            per_scenario.setdefault(f.scenario, []).append(
                {"intent": p.intent_name, "steer": c.steer, "target_speed": c.target_speed,
                 "brake": c.brake, "latency_ms": p.latency_ms})
        report = rt.latency_report()

    hud.close()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    summary = {"latency": report,
               "per_scenario": {k: {
                   "n": len(v),
                   "mean_steer": float(np.mean([r["steer"] for r in v])),
                   "mean_target_speed": float(np.mean([r["target_speed"] for r in v])),
                   "brake_rate": float(np.mean([r["brake"] > 0.5 for r in v])),
                   "intents": sorted({r["intent"] for r in v}),
               } for k, v in sorted(per_scenario.items())}}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    # Same schema the CARLA agent writes, so `bench.run_benchmark --runtime`
    # accepts either source.
    Path(args.latency_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latency_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nHUD frames -> {args.out}")


if __name__ == "__main__":
    main()
