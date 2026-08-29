"""Pre-simulation atomic verification gates.

These run before any CARLA time is spent. Each gate isolates ONE primitive skill
and fails loudly if the model has not learned it, so a bad checkpoint is caught
in seconds on the terminal rather than in hours of closed-loop evaluation.

    Gate 1  steering polarity + distribution on left/right turn queries
    Gate 2  zero-speed stop-line adherence on red-light frames
    Gate 3  10-step diffusion denoising latency and sample stability

    python -m sub1b_vla.verify.atomic_checks --config ... --checkpoint ...
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..carla_agent.controller import TrajectoryController
from ..data.augment import cut_bottom_quarter, resize_nn, to_chw_normalized
from ..data.synthetic import COMMAND_TO_ID, generate_frame
from ..losses.consistency import compute_dynamics
from ..models.coc_prompt import INTENTS


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


def _prep(frame, image_size, cut_bottom=True):
    img = cut_bottom_quarter(frame.image) if cut_bottom else frame.image
    return to_chw_normalized(resize_nn(img, image_size))


def _scenario_batch(scenarios, cfg, n_per, seed=0):
    """Deterministically draw frames of the requested scenario types."""
    m = cfg["model"]
    out = {s: [] for s in scenarios}
    rng = np.random.default_rng(seed)
    tries = 0
    while any(len(v) < n_per for v in out.values()) and tries < 40000:
        f = generate_frame(rng, m["pred_len"], m.get("waypoint_dt", 0.2), hard_ratio=0.5)
        if f.scenario in out and len(out[f.scenario]) < n_per:
            out[f.scenario].append(f)
        tries += 1
    missing = [s for s, v in out.items() if len(v) < n_per]
    if missing:
        raise RuntimeError(f"Could not draw enough frames for {missing} after {tries} tries.")
    return out


@torch.inference_mode()
def _predict(model, cfg, frames, device):
    m = cfg["model"]
    imgs = torch.from_numpy(
        np.stack([_prep(f, m["image_size"], cfg["data"].get("cut_bottom_quarter", True))
                  for f in frames])
    ).to(device)
    speed = torch.tensor([f.speed_kmh for f in frames], dtype=torch.float32, device=device)
    tp = torch.from_numpy(np.stack([f.target_point for f in frames])).to(device)
    cmd = torch.tensor([f.command for f in frames], dtype=torch.long, device=device)
    return model.predict_trajectory(imgs, speed, tp, cmd)


# ---------------------------------------------------------------------------
# Gate 1: steering distribution on turn queries
# ---------------------------------------------------------------------------
def gate_steering_distribution(model, cfg, device, n=64, min_separation=0.25,
                               min_polarity_rate=0.75) -> GateResult:
    batches = _scenario_batch(["left_turn", "right_turn"], cfg, n, seed=11)
    ctrl = TrajectoryController(dt=cfg["model"].get("waypoint_dt", 0.2))
    stats = {}
    for name, frames in batches.items():
        out = _predict(model, cfg, frames, device)
        wps = out.waypoints.float().cpu().numpy()
        steers = np.array([ctrl.step(w, f.speed_kmh / 3.6).steer
                           for w, f in zip(wps, frames)])
        stats[name] = {
            "mean_steer": float(steers.mean()),
            "std_steer": float(steers.std()),
            "p10": float(np.percentile(steers, 10)),
            "p90": float(np.percentile(steers, 90)),
            "mean_lateral_m": float(wps[:, -1, 1].mean()),
        }
        # CARLA sign: steer < 0 turns left.
        want_negative = name == "left_turn"
        correct = (steers < 0) if want_negative else (steers > 0)
        stats[name]["correct_polarity_rate"] = float(correct.mean())

    sep = stats["right_turn"]["mean_steer"] - stats["left_turn"]["mean_steer"]
    pol = min(stats["left_turn"]["correct_polarity_rate"],
              stats["right_turn"]["correct_polarity_rate"])
    passed = sep >= min_separation and pol >= min_polarity_rate
    return GateResult(
        name="steering distribution (left/right turn queries)",
        passed=passed,
        detail=(f"separation(right-left)={sep:+.3f} (need >={min_separation}); "
                f"worst polarity rate={pol:.2f} (need >={min_polarity_rate})"),
        metrics={"separation": sep, "worst_polarity_rate": pol, **stats},
    )


# ---------------------------------------------------------------------------
# Gate 2: zero-speed stop-line adherence at red lights
# ---------------------------------------------------------------------------
def gate_red_light_stop(model, cfg, device, n=64, max_final_speed=0.5,
                        max_throttle_rate=0.15) -> GateResult:
    """Stop-line adherence.

    Measured on the trajectory's FINAL speed, not on its mean speed over the
    next second. Braking from 45 km/h to a stop line is still moving at 9 m/s a
    second in -- judging the approach speed would fail every correct stop and
    only pass a vehicle that was already stationary. What must hold is that the
    trajectory ENDS at rest and that no throttle is commanded on the way.
    """
    batches = _scenario_batch(["red_light", "green_light"], cfg, n, seed=23)
    ctrl = TrajectoryController(dt=cfg["model"].get("waypoint_dt", 0.2))
    res = {}
    for name, frames in batches.items():
        out = _predict(model, cfg, frames, device)
        wps = out.waypoints.float().cpu().numpy()
        ctrls = [ctrl.step(w, f.speed_kmh / 3.6) for w, f in zip(wps, frames)]
        throttle = np.array([c.throttle for c in ctrls])
        dyn = compute_dynamics(torch.from_numpy(wps).float(),
                               cfg["model"].get("waypoint_dt", 0.2))
        res[name] = {
            "mean_final_speed_ms": float(dyn.final_speed.mean()),
            "stopped_rate": float((dyn.final_speed.numpy() <= max_final_speed).mean()),
            "throttle_rate": float((throttle > 0.1).mean()),
            "mean_target_speed_ms": float(np.mean([c.target_speed for c in ctrls])),
            "mean_displacement_m": float(dyn.displacement.mean()),
        }
    red, green = res["red_light"], res["green_light"]
    passed = (red["mean_final_speed_ms"] <= max_final_speed
              and red["throttle_rate"] <= max_throttle_rate
              and green["mean_final_speed_ms"] > red["mean_final_speed_ms"])
    return GateResult(
        name="zero-speed stop-line adherence (red light)",
        passed=passed,
        detail=(f"red: final_speed={red['mean_final_speed_ms']:.3f} m/s "
                f"(need <={max_final_speed}), stopped_rate={red['stopped_rate']:.2f}, "
                f"throttle_rate={red['throttle_rate']:.2f} (need <={max_throttle_rate}); "
                f"green control: final_speed={green['mean_final_speed_ms']:.3f} m/s "
                f"(must exceed red)"),
        metrics=res,
    )


# ---------------------------------------------------------------------------
# Gate 3: diffusion latency and stability
# ---------------------------------------------------------------------------
def gate_diffusion_latency(model, cfg, device, iters=30, warmup=5,
                           budget_ms=80.0, max_spread_m=1.0, repeats=8) -> GateResult:
    m = cfg["model"]
    frames = _scenario_batch(["free_flow"], cfg, 1, seed=7)["free_flow"]
    img = torch.from_numpy(
        _prep(frames[0], m["image_size"], cfg["data"].get("cut_bottom_quarter", True))
    )[None].to(device)
    speed = torch.tensor([frames[0].speed_kmh], device=device)
    tp = torch.from_numpy(frames[0].target_point)[None].to(device)
    cmd = torch.tensor([frames[0].command], device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model.predict_trajectory(img, speed, tp, cmd)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # End-to-end control-path latency (encode + 10-step DDIM).
        e2e = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model.predict_trajectory(img, speed, tp, cmd)
            if device.type == "cuda":
                torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t0) * 1000.0)

        # Denoiser-only latency, with the encoder cost factored out.
        enc = model.encoder(img)
        denoise = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model.diffusion.sample(enc.spatial_tokens, enc.semantic_tokens, speed, tp, cmd)
            if device.type == "cuda":
                torch.cuda.synchronize()
            denoise.append((time.perf_counter() - t0) * 1000.0)

        # Stability: independent noise draws on ONE frame must agree. A
        # trajectory that jumps between samples would make the controller chatter.
        samples = np.stack([
            model.diffusion.sample(enc.spatial_tokens, enc.semantic_tokens,
                                   speed, tp, cmd)[0].float().cpu().numpy()
            for _ in range(repeats)
        ])
        spread = float(np.linalg.norm(samples.std(axis=0), axis=-1).mean())

    e2e_a, dn_a = np.asarray(e2e), np.asarray(denoise)
    p95 = float(np.percentile(e2e_a, 95))
    passed = p95 <= budget_ms and spread <= max_spread_m
    return GateResult(
        name=f"{m['diffusion_infer_steps']}-step diffusion latency & stability",
        passed=passed,
        detail=(f"end-to-end p95={p95:.1f} ms (budget {budget_ms} ms, "
                f"{1000.0 / e2e_a.mean():.1f} Hz mean); sample spread={spread:.3f} m "
                f"(need <={max_spread_m})"),
        metrics={
            "e2e_mean_ms": float(e2e_a.mean()), "e2e_p95_ms": p95,
            "e2e_max_ms": float(e2e_a.max()), "mean_hz": float(1000.0 / e2e_a.mean()),
            "denoise_mean_ms": float(dn_a.mean()), "denoise_p95_ms": float(np.percentile(dn_a, 95)),
            "sample_spread_m": spread, "device": device.type,
            "infer_steps": m["diffusion_infer_steps"],
        },
    )


# ---------------------------------------------------------------------------
def run_all(model, cfg, device, budget_ms=None, strict=True) -> list[GateResult]:
    budget = budget_ms or cfg.get("inference", {}).get("latency_budget_ms", 80)
    return [
        gate_steering_distribution(model, cfg, device),
        gate_red_light_stop(model, cfg, device),
        gate_diffusion_latency(model, cfg, device, budget_ms=budget),
    ]


def render(results: list[GateResult], header: str = "") -> str:
    w = 96
    out = ["=" * w, "ATOMIC VERIFICATION GATES (pre-simulation)".center(w)]
    if header:
        out.append(header.center(w))
    out += ["=" * w, f"{'gate':<52}{'verdict':>10}", "-" * w]
    for r in results:
        out.append(f"{r.name:<52}{('PASS' if r.passed else 'FAIL'):>10}")
        out.append(f"    {r.detail}")
    out.append("-" * w)
    n_pass = sum(r.passed for r in results)
    out.append(f"{n_pass}/{len(results)} gates passed")
    out.append("=" * w)
    return "\n".join(out)


def main():
    from ..carla_agent.agent import load_agent_model  # noqa: PLC0415
    from ..utils import load_config  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--budget-ms", type=float, default=None)
    ap.add_argument("--no-strict", action="store_true",
                    help="report gate failures without a non-zero exit code")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_agent_model(cfg, args.checkpoint, device)
    results = run_all(model, cfg, device, budget_ms=args.budget_ms)
    tag = f"device={device.type}  checkpoint={args.checkpoint or 'NONE (untrained)'}"
    print(render(results, tag))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    raise SystemExit(0 if (args.no_strict or all(r.passed for r in results)) else 1)


if __name__ == "__main__":
    main()
