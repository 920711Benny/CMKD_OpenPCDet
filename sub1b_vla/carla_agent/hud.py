"""Simulator HUD.

Output Separation Protocol: everything verbose -- video, trajectory curve, dual
attention maps, live CoC rationale -- renders HERE, never to the terminal. The
terminal is reserved for the benchmark table.

The compositor is pure numpy so it can be unit-tested and dumped to PNG on a
headless box; `PygameHUD` is a thin display wrapper on top of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sensors import CameraRig

# Distinct hues so the two attention maps are never confused with each other.
SPATIAL_COLOR = np.array([0.15, 0.85, 1.00])   # cyan   -- geometry branch
SEMANTIC_COLOR = np.array([1.00, 0.45, 0.10])  # amber  -- semantic branch
TRAJ_COLOR = np.array([0.20, 1.00, 0.40])      # green  -- predicted trajectory
LOOKAHEAD_COLOR = np.array([1.00, 0.95, 0.20])


@dataclass
class HUDState:
    frame: np.ndarray                 # (H, W, 3) float [0,1]
    waypoints: np.ndarray             # (T, 2) ego BEV
    spatial_attn: np.ndarray          # (N,) patch saliency
    semantic_attn: np.ndarray         # (N,) patch saliency
    rationale: str
    intent: str
    speed_kmh: float
    control: dict
    latency_ms: float
    fps: float
    rationale_age_frames: int = 0


def _attn_to_grid(attn: np.ndarray) -> np.ndarray:
    n = int(attn.shape[0])
    side = int(round(math.sqrt(n)))
    if side * side != n:  # non-square token grid: pad to the next square
        side = int(math.ceil(math.sqrt(n)))
        attn = np.pad(attn, (0, side * side - n))
    g = attn.reshape(side, side).astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    return (g - lo) / (hi - lo) if hi > lo else np.zeros_like(g)


def _upsample(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    yi = np.clip((np.arange(h) * grid.shape[0] // max(1, h)), 0, grid.shape[0] - 1)
    xi = np.clip((np.arange(w) * grid.shape[1] // max(1, w)), 0, grid.shape[1] - 1)
    return grid[yi][:, xi]


def overlay_attention(frame: np.ndarray, attn: np.ndarray, color: np.ndarray,
                      alpha: float = 0.35) -> np.ndarray:
    h, w = frame.shape[:2]
    heat = _upsample(_attn_to_grid(attn), h, w)[..., None]
    return np.clip(frame * (1 - alpha * heat) + color * (alpha * heat), 0, 1)


def _draw_line(img, p0, p1, color, thickness=2):
    h, w = img.shape[:2]
    x0, y0 = p0
    x1, y1 = p1
    n = int(max(abs(x1 - x0), abs(y1 - y0), 1)) * 2
    for t in np.linspace(0, 1, n):
        x, y = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
        if 0 <= x < w and 0 <= y < h:
            r = thickness
            img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = color


def _draw_disc(img, center, radius, color):
    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img[y0:y1, x0:x1][mask] = color


def draw_trajectory(frame: np.ndarray, waypoints: np.ndarray, rig: CameraRig,
                    color=TRAJ_COLOR, lookahead: tuple | None = None) -> np.ndarray:
    """Project the diffusion trajectory onto the camera image."""
    out = frame.copy()
    sx = frame.shape[1] / rig.width
    sy = frame.shape[0] / rig.height
    px, valid = rig.project_ego_to_image(waypoints)
    px = px * np.array([sx, sy])
    prev = None
    for i, (p, ok) in enumerate(zip(px, valid)):
        if not ok:
            prev = None
            continue
        # Nearer waypoints render larger -- reads as depth at a glance.
        radius = max(2, int(6 * (1 - i / max(1, len(px)))) + 2)
        _draw_disc(out, p, radius, color)
        if prev is not None:
            _draw_line(out, prev, p, color, thickness=1)
        prev = p
    if lookahead is not None:
        lp, lv = rig.project_ego_to_image(np.asarray(lookahead).reshape(1, 2))
        if bool(lv[0]):
            _draw_disc(out, lp[0] * np.array([sx, sy]), 6, LOOKAHEAD_COLOR)
    return out


def compose(state: HUDState, rig: CameraRig, show_spatial=True, show_semantic=True) -> np.ndarray:
    """Full HUD frame: video + attention + trajectory. Text is drawn by the
    pygame layer (numpy has no font rasteriser)."""
    frame = np.clip(np.asarray(state.frame, dtype=np.float32), 0, 1)
    if frame.ndim == 3 and frame.shape[0] == 3:  # CHW -> HWC
        frame = frame.transpose(1, 2, 0)
    if show_spatial:
        frame = overlay_attention(frame, state.spatial_attn, SPATIAL_COLOR, alpha=0.30)
    if show_semantic:
        frame = overlay_attention(frame, state.semantic_attn, SEMANTIC_COLOR, alpha=0.20)
    frame = draw_trajectory(frame, state.waypoints, rig,
                            lookahead=state.control.get("lookahead"))
    return frame


def telemetry_lines(state: HUDState) -> list[str]:
    c = state.control
    age = f" (+{state.rationale_age_frames}f)" if state.rationale_age_frames else ""
    return [
        f"speed {state.speed_kmh:5.1f} km/h   intent: {state.intent}",
        f"throttle {c.get('throttle', 0):.2f}  brake {c.get('brake', 0):.2f}  "
        f"steer {c.get('steer', 0):+.2f}",
        f"traj latency {state.latency_ms:5.1f} ms   {state.fps:4.1f} Hz",
        f"CoC{age}: {state.rationale}",
    ]


class PygameHUD:
    """Windowed HUD. Falls back to writing PNG frames when pygame or a display
    is unavailable, so a headless benchmark run still produces a visual record."""

    def __init__(self, rig: CameraRig, width=1024, height=512, dump_dir=None, title="Sub-1B VLA"):
        self.rig = rig
        self.width, self.height = width, height
        self.dump_dir = dump_dir
        self.frame_idx = 0
        self.screen = None
        self.font = None
        self.pygame = None
        self.show_spatial = True
        self.show_semantic = True
        try:
            import pygame  # noqa: PLC0415

            pygame.init()
            pygame.font.init()
            self.pygame = pygame
            self.screen = pygame.display.set_mode((width, height + 110))
            pygame.display.set_caption(title)
            self.font = pygame.font.SysFont("dejavusansmono", 16)
        except Exception as exc:  # noqa: BLE001
            print(f"[hud] windowed display unavailable ({type(exc).__name__}); "
                  f"{'dumping frames to ' + str(dump_dir) if dump_dir else 'HUD disabled'}.")

    def poll(self) -> bool:
        """Pump events. Returns False when the operator closes the window.
        Keys: S toggles the spatial map, D the semantic map."""
        if self.pygame is None:
            return True
        for ev in self.pygame.event.get():
            if ev.type == self.pygame.QUIT:
                return False
            if ev.type == self.pygame.KEYDOWN:
                if ev.key == self.pygame.K_ESCAPE:
                    return False
                if ev.key == self.pygame.K_s:
                    self.show_spatial = not self.show_spatial
                if ev.key == self.pygame.K_d:
                    self.show_semantic = not self.show_semantic
        return True

    def render(self, state: HUDState):
        img = compose(state, self.rig, self.show_spatial, self.show_semantic)
        self.frame_idx += 1
        if self.pygame is None:
            self._dump(img)
            return
        arr = (img * 255).astype(np.uint8)
        surf = self.pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
        surf = self.pygame.transform.scale(surf, (self.width, self.height))
        self.screen.fill((12, 12, 16))
        self.screen.blit(surf, (0, 0))
        for i, line in enumerate(telemetry_lines(state)):
            self.screen.blit(self.font.render(line[:150], True, (235, 235, 235)),
                             (10, self.height + 6 + i * 24))
        self.pygame.display.flip()

    def _dump(self, img):
        if not self.dump_dir:
            return
        from pathlib import Path  # noqa: PLC0415

        d = Path(self.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        arr = (img * 255).astype(np.uint8)
        try:
            from PIL import Image  # noqa: PLC0415

            Image.fromarray(arr).save(d / f"frame_{self.frame_idx:06d}.png")
        except ImportError:
            np.save(d / f"frame_{self.frame_idx:06d}.npy", arr)

    def close(self):
        if self.pygame is not None:
            self.pygame.quit()
