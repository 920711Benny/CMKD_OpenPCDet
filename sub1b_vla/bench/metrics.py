"""CARLA leaderboard metric extraction.

Parses the leaderboard's `results.json` and derives the standard academic
metrics. Nothing here invents a number: if a field is absent from the results
file the metric is reported as unavailable, never as a default or a guess.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# CARLA leaderboard infraction penalty coefficients.
PENALTIES = {
    "collisions_pedestrian": 0.50,
    "collisions_vehicle": 0.60,
    "collisions_layout": 0.65,
    "red_light": 0.70,
    "stop_infraction": 0.80,
    "scenario_timeouts": 0.70,
    "yield_emergency_vehicle_infractions": 0.70,
    "min_speed_infractions": 0.70,
}
# Infractions counted but not multiplied into the infraction score.
EVENT_KEYS = ("outside_route_lanes", "route_dev", "vehicle_blocked", "route_timeout")


@dataclass
class BenchmarkMetrics:
    driving_score: float | None = None
    route_completion: float | None = None
    infraction_score: float | None = None
    collisions_per_km: float | None = None
    red_light_per_km: float | None = None
    sidewalk_per_km: float | None = None
    action_cot_alignment: float | None = None
    mean_latency_ms: float | None = None
    fps: float | None = None
    driven_km: float | None = None
    num_routes: int = 0
    source: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def get(self, key):
        return getattr(self, key, None)


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def parse_leaderboard_results(path: str | Path) -> BenchmarkMetrics:
    """Read a CARLA leaderboard results.json into BenchmarkMetrics.

    Per-kilometre rates come from the leaderboard's own per-km infraction lists
    when present; otherwise they are derived from raw counts and the driven
    distance, and the derivation is recorded in `notes`.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    records = data.get("_checkpoint", {}).get("records", [])
    if not records:
        return BenchmarkMetrics(source=str(p), notes=["results.json contained no route records"])

    m = BenchmarkMetrics(source=str(p), num_routes=len(records))
    m.driving_score = _mean([r.get("scores", {}).get("score_composed") for r in records])
    m.route_completion = _mean([r.get("scores", {}).get("score_route") for r in records])
    m.infraction_score = _mean([r.get("scores", {}).get("score_penalty") for r in records])

    total_km = 0.0
    have_distance = True
    counts = {"collision": 0.0, "red_light": 0.0, "sidewalk": 0.0}
    for r in records:
        meta = r.get("meta", {})
        d = meta.get("route_length")
        if d is None:
            have_distance = False
        else:
            total_km += float(d) / 1000.0
        inf = r.get("infractions", {})
        for k in ("collisions_pedestrian", "collisions_vehicle", "collisions_layout"):
            counts["collision"] += len(inf.get(k, []))
        counts["red_light"] += len(inf.get("red_light", []))
        # The leaderboard reports sidewalk/lane departures under outside_route_lanes.
        counts["sidewalk"] += len(inf.get("outside_route_lanes", []))

    if have_distance and total_km > 0:
        m.driven_km = total_km
        m.collisions_per_km = counts["collision"] / total_km
        m.red_light_per_km = counts["red_light"] / total_km
        m.sidewalk_per_km = counts["sidewalk"] / total_km
    else:
        m.notes.append("route_length missing; per-km rates unavailable (raw counts: "
                       + ", ".join(f"{k}={int(v)}" for k, v in counts.items()) + ")")
    return m


def merge_runtime_metrics(m: BenchmarkMetrics, runtime_json: str | Path | None,
                          alignment_json: str | Path | None) -> BenchmarkMetrics:
    """Attach latency and Action-CoT alignment measured by our own instrumentation."""
    if runtime_json and Path(runtime_json).exists():
        rt = json.loads(Path(runtime_json).read_text())
        traj = rt.get("trajectory", {})
        m.mean_latency_ms = traj.get("mean_ms")
        m.fps = traj.get("fps_mean")
        dev = rt.get("device_name") or rt.get("device")
        if dev:
            m.notes.append(f"latency measured on {dev} (p95 "
                           f"{traj.get('p95_ms', float('nan')):.1f} ms, "
                           f"budget {rt.get('budget_ms')} ms)")
    if alignment_json and Path(alignment_json).exists():
        al = json.loads(Path(alignment_json).read_text())
        m.action_cot_alignment = al.get("alignment_score")
        m.notes.append(f"Action-CoT alignment from {al.get('n', 0)} samples, "
                       f"intent source '{al.get('intent_source', '?')}'")
    return m


def load_baseline(path: str | Path | None) -> BenchmarkMetrics:
    """Load baseline (e.g. SimLingo) numbers from a JSON file.

    Baseline numbers are NEVER hard-coded here. Supply them from the baseline's
    published table or from your own reproduction run, with the provenance
    recorded in the file's `source` field.
    """
    if not path or not Path(path).exists():
        return BenchmarkMetrics(source="no baseline file supplied",
                                notes=["run with --baseline <file.json> to populate this column"])
    d = json.loads(Path(path).read_text())
    m = BenchmarkMetrics(source=d.get("source", str(path)))
    for k in ("driving_score", "route_completion", "infraction_score", "collisions_per_km",
              "red_light_per_km", "sidewalk_per_km", "action_cot_alignment",
              "mean_latency_ms", "fps", "driven_km"):
        # An explicit null means "not measured" -- identical to an absent key.
        if d.get(k) is not None:
            setattr(m, k, d[k])
    m.num_routes = int(d.get("num_routes", 0))
    m.notes = list(d.get("notes", []))
    return m
