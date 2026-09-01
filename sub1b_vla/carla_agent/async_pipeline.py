"""Decoupled asynchronous inference runtime.

The problem this solves: autoregressive CoT decoding takes hundreds of
milliseconds. If actuation waits on it, the control loop collapses to ~2 Hz and
the vehicle drives badly *because of* its own explanations.

The split:

  FAST PATH (control thread, target >= 10 Hz / < 80 ms)
      frame -> dual-head encode -> 10-step DDIM -> waypoints -> PID/pure-pursuit
      Never touches the token decoder.

  SLOW PATH (rationale thread, ~1 Hz, best effort)
      cached encoding -> autoregressive CoC rationale -> HUD / telemetry
      Publishes into a latest-wins slot. If it is late, the HUD shows a slightly
      stale rationale; the vehicle is unaffected.

The two paths share the encoder output through `LatestSlot`, so the rationale
always describes a frame the controller actually saw. A cooperative priority
gate lets the rationale thread yield between token steps whenever the control
thread has work pending, which keeps GIL/GPU contention off the critical path.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch


class LatestSlot:
    """Single-slot latest-wins mailbox. Readers never block on writers."""

    def __init__(self, initial=None):
        self._lock = threading.Lock()
        self._value = initial
        self._stamp = 0.0
        self._seq = 0

    def put(self, value):
        with self._lock:
            self._value = value
            self._stamp = time.perf_counter()
            self._seq += 1

    def get(self):
        with self._lock:
            return self._value, self._stamp, self._seq


class PriorityGate:
    """Cooperative yield point. The low-priority worker calls `should_yield()`
    between units of work and steps aside while the control loop is running."""

    def __init__(self):
        self._pending = threading.Event()

    def control_begin(self):
        self._pending.set()

    def control_end(self):
        self._pending.clear()

    def should_yield(self) -> bool:
        return self._pending.is_set()


@dataclass
class LatencyTracker:
    name: str
    samples: list[float] = field(default_factory=list)
    cap: int = 4000

    def add(self, ms: float):
        self.samples.append(ms)
        if len(self.samples) > self.cap:
            del self.samples[: len(self.samples) - self.cap]

    def stats(self) -> dict:
        if not self.samples:
            return {"n": 0}
        a = np.asarray(self.samples)
        return {
            "n": int(a.size),
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()),
            "fps_mean": float(1000.0 / a.mean()),
        }


@dataclass
class Perception:
    waypoints: np.ndarray
    intent_id: int
    intent_name: str
    spatial_attn: np.ndarray
    semantic_attn: np.ndarray
    frame_id: int
    latency_ms: float


class AsyncVLARuntime:
    """Owns the model and runs the two decoupled paths."""

    def __init__(self, model, device, cfg, cot_enabled: bool = True):
        from ..models.coc_prompt import INTENTS  # noqa: PLC0415

        self.model = model.eval().to(device)
        self.device = device
        self.cfg = cfg
        self.intents = INTENTS
        inf = cfg.get("inference", {})
        self.cot_period = 1.0 / max(1e-3, float(inf.get("cot_hz", 1)))
        self.latency_budget_ms = float(inf.get("latency_budget_ms", 80))
        self.max_new_tokens = int(inf.get("cot_max_tokens", 48))

        self.perception_slot = LatestSlot()
        self.encoding_slot = LatestSlot()
        self.rationale_slot = LatestSlot(("(warming up)", -1))
        self.gate = PriorityGate()

        self.traj_latency = LatencyTracker("trajectory")
        self.encode_latency = LatencyTracker("encode")
        self.cot_latency = LatencyTracker("rationale")

        self._stop = threading.Event()
        self._cot_thread: threading.Thread | None = None
        self._frame_id = 0
        self.cot_enabled = cot_enabled

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        if self.cot_enabled and self._cot_thread is None:
            self._stop.clear()
            self._cot_thread = threading.Thread(
                target=self._cot_worker, name="coc-rationale", daemon=True
            )
            self._cot_thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._cot_thread is not None:
            self._cot_thread.join(timeout=2.0)
            self._cot_thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ---- fast path -------------------------------------------------------
    @torch.inference_mode()
    def perceive(self, image_chw: np.ndarray, speed_kmh: float,
                 target_point: np.ndarray, command: int) -> Perception:
        """Blocking fast path. This is the whole control-critical computation."""
        self.gate.control_begin()
        try:
            t0 = time.perf_counter()
            img = torch.as_tensor(image_chw, dtype=torch.float32, device=self.device)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            enc = self.model.encoder(img)
            t_enc = time.perf_counter()

            speed = torch.tensor([speed_kmh], dtype=torch.float32, device=self.device)
            tp = torch.as_tensor(target_point, dtype=torch.float32, device=self.device).view(1, 2)
            cmd = torch.tensor([int(command)], dtype=torch.long, device=self.device)

            out = self.model.predict_trajectory(None, speed, tp, cmd, encoded=enc)
            t1 = time.perf_counter()
        finally:
            self.gate.control_end()

        self._frame_id += 1
        self.encode_latency.add((t_enc - t0) * 1000.0)
        self.traj_latency.add((t1 - t0) * 1000.0)

        intent_id = int(out.intent_logits.argmax(-1)[0])
        p = Perception(
            waypoints=out.waypoints[0].float().cpu().numpy(),
            intent_id=intent_id,
            intent_name=self.intents[intent_id],
            spatial_attn=out.spatial_attn[0].float().cpu().numpy(),
            semantic_attn=out.semantic_attn[0].float().cpu().numpy(),
            frame_id=self._frame_id,
            latency_ms=(t1 - t0) * 1000.0,
        )
        self.perception_slot.put(p)
        self.encoding_slot.put((enc, self._frame_id))
        return p

    # ---- slow path -------------------------------------------------------
    def _cot_worker(self):
        while not self._stop.is_set():
            slot, stamp, _ = self.encoding_slot.get()
            if slot is None:
                self._stop.wait(0.05)
                continue
            enc, frame_id = slot
            _, _, last = self.rationale_slot.get()
            if isinstance(last, int) and last == frame_id:
                self._stop.wait(self.cot_period)
                continue
            # Yield the moment the control loop wants the device.
            while self.gate.should_yield() and not self._stop.is_set():
                time.sleep(0.002)
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    text = self.model.explain(enc, max_new_tokens=self.max_new_tokens)[0]
                self.cot_latency.add((time.perf_counter() - t0) * 1000.0)
                self.rationale_slot.put((text, frame_id))
            except Exception as exc:  # noqa: BLE001 - rationale must never kill driving
                self.rationale_slot.put((f"(rationale unavailable: {type(exc).__name__})", frame_id))
            self._stop.wait(self.cot_period)

    def latest_rationale(self) -> tuple[str, int]:
        value, _, _ = self.rationale_slot.get()
        return value if value is not None else ("(warming up)", -1)

    # ---- reporting -------------------------------------------------------
    def latency_report(self) -> dict:
        traj = self.traj_latency.stats()
        return {
            "trajectory": traj,
            "encode": self.encode_latency.stats(),
            "rationale": self.cot_latency.stats(),
            "budget_ms": self.latency_budget_ms,
            "within_budget": bool(traj.get("p95_ms", float("inf")) <= self.latency_budget_ms),
            # Recorded so a latency figure can never be read as coming from
            # hardware it was not measured on.
            "device": self.device.type,
            "device_name": (torch.cuda.get_device_name(self.device)
                            if self.device.type == "cuda" else "cpu"),
        }
