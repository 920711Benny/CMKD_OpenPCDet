"""Terminal report renderer.

Output Separation Protocol: this module prints the academic benchmark table and
nothing else. Video, trajectory curves, attention maps and CoC rationale go to
the HUD; debug logging is suppressed for the duration of a benchmark run.

A metric that was not measured prints as `--`. It is never filled with a
default, an estimate, or a value carried over from another run.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import warnings

from .metrics import BenchmarkMetrics

# (attribute, label, format, higher_is_better)
ROWS = [
    ("driving_score", "Driving Score (DS)", "{:.2f}", True),
    ("route_completion", "Route Completion (RC)", "{:.2f}", True),
    ("infraction_score", "Infraction Score (IS)", "{:.3f}", True),
    ("collisions_per_km", "Collision Rate / km", "{:.3f}", False),
    ("red_light_per_km", "Red Light Violations / km", "{:.3f}", False),
    ("sidewalk_per_km", "Sidewalk Incursions / km", "{:.3f}", False),
    ("action_cot_alignment", "Action-CoT Alignment Score", "{:.3f}", True),
]

NA = "--"


def _fmt(value, spec):
    return NA if value is None else spec.format(value)


def _delta(ours, base, higher_is_better, spec):
    if ours is None or base is None:
        return NA
    diff = ours - base
    improved = diff > 0 if higher_is_better else diff < 0
    if abs(diff) < 1e-12:
        return "0.00 (=)"
    arrow = "+" if improved else "-"
    return f"{arrow}{spec.format(abs(diff)).lstrip('+-')} ({'better' if improved else 'worse'})"


def _latency_cell(m: BenchmarkMetrics):
    if m.mean_latency_ms is None:
        return NA
    fps = m.fps if m.fps is not None else (1000.0 / m.mean_latency_ms)
    return f"{m.mean_latency_ms:.1f} / {fps:.1f}"


def render_table(baseline: BenchmarkMetrics, ours: BenchmarkMetrics,
                 baseline_name="SimLingo Baseline",
                 ours_name="Our Novel Dual-Head Diffusion VLA",
                 title="CARLA CLOSED-LOOP BENCHMARK") -> str:
    c0, c1, c2, c3 = 31, 22, 36, 22
    sep = "|" + "-" * (c0 + 2) + "|" + "-" * (c1 + 2) + "|" + "-" * (c2 + 2) + "|" + "-" * (c3 + 2) + "|"
    width = len(sep)
    lines = [title.center(width), "=" * width]
    lines.append(f"| {'Metric':<{c0}} | {baseline_name:<{c1}} | {ours_name:<{c2}} | {'Delta (Improvement)':<{c3}} |")
    lines.append(sep)
    for attr, label, spec, hib in ROWS:
        b, o = baseline.get(attr), ours.get(attr)
        lines.append(f"| {label:<{c0}} | {_fmt(b, spec):<{c1}} | {_fmt(o, spec):<{c2}} | "
                     f"{_delta(o, b, hib, spec):<{c3}} |")
    # Latency row is a composite cell.
    d_lat = _delta(ours.mean_latency_ms, baseline.mean_latency_ms, False, "{:.1f}")
    lines.append(f"| {'Mean Control Latency (ms) / FPS':<{c0}} | {_latency_cell(baseline):<{c1}} | "
                 f"{_latency_cell(ours):<{c2}} | {d_lat:<{c3}} |")
    lines.append(sep)
    return "\n".join(lines)


def render_provenance(baseline: BenchmarkMetrics, ours: BenchmarkMetrics) -> str:
    out = ["", "sources:",
           f"  baseline : {baseline.source}" + (f"  ({baseline.num_routes} routes)"
                                                if baseline.num_routes else ""),
           f"  ours     : {ours.source}" + (f"  ({ours.num_routes} routes)"
                                            if ours.num_routes else "")]
    if ours.driven_km:
        out.append(f"  distance : {ours.driven_km:.2f} km driven")
    for m, tag in ((baseline, "baseline"), (ours, "ours")):
        for n in m.notes:
            out.append(f"  note [{tag}]: {n}")
    if any(m.get(a) is None for a, *_ in ROWS for m in (baseline, ours)):
        out.append(f"  '{NA}' = not measured in this run; no value is inferred or defaulted.")
    return "\n".join(out)


@contextlib.contextmanager
def quiet_terminal(enabled: bool = True):
    """Suppress library logging, warnings and stray stdout for a benchmark run,
    so the terminal carries only the table. stderr stays open: a crash must
    still be visible."""
    if not enabled:
        yield
        return
    prev_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            yield real_stdout
        finally:
            sys.stdout = real_stdout
            logging.disable(prev_level)
