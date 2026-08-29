"""Benchmark orchestrator.

Two modes:

  --results <leaderboard results.json>
        Parse a completed CARLA leaderboard run and print the table.

  --launch
        Print the exact leaderboard command for the configured towns/weather
        (Town05 Long / Town05 Hard + long-tail scenario sets), then parse the
        results it writes. CARLA itself is launched by the user's simulator
        install; this module never fabricates simulator output.

Terminal output is ONLY the table (plus provenance). Everything verbose belongs
on the HUD.

    python -m sub1b_vla.bench.run_benchmark --config ... --results ... \
        --baseline baselines/simlingo.json --runtime runs/latency_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import BenchmarkMetrics, load_baseline, merge_runtime_metrics, parse_leaderboard_results
from .report import render_provenance, render_table

BENCHMARK_SUITES = {
    "town05_long": {
        "routes": "leaderboard/data/routes_town05_long.xml",
        "scenarios": "leaderboard/data/all_towns_traffic_scenarios.json",
    },
    "town05_hard": {
        "routes": "leaderboard/data/routes_town05_hard.xml",
        "scenarios": "leaderboard/data/longtail_traffic_scenarios.json",
    },
}
WEATHER_PRESETS = ("ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset",
                   "WetCloudySunset", "SoftRainSunset", "MidRainyNight")
LONGTAIL_SCENARIOS = ("occluded_pedestrian", "cut_in_vehicle", "blind_intersection")


def launch_command(cfg_path: str, ckpt: str, suite: str, out: str, weather: str) -> str:
    s = BENCHMARK_SUITES[suite]
    return (
        f"SUB1B_CONFIG={cfg_path} SUB1B_CHECKPOINT={ckpt} SUB1B_HUD=1 SUB1B_QUIET=1 \\\n"
        f"  python3 leaderboard/leaderboard/leaderboard_evaluator.py \\\n"
        f"    --agent sub1b_vla/carla_agent/agent.py \\\n"
        f"    --routes {s['routes']} \\\n"
        f"    --scenarios {s['scenarios']} \\\n"
        f"    --checkpoint {out} \\\n"
        f"    --weather {weather} \\\n"
        f"    --track SENSORS --repetitions 1"
    )


def hard_constraint_report(m: BenchmarkMetrics) -> list[str]:
    """The two constraints declared non-negotiable: zero red-light infractions
    and zero sidewalk incursions."""
    out = []
    for label, val in (("red-light infractions / km", m.red_light_per_km),
                       ("sidewalk incursions / km", m.sidewalk_per_km)):
        if val is None:
            out.append(f"  {label:<28} NOT MEASURED")
        elif val == 0.0:
            out.append(f"  {label:<28} 0.000  SATISFIED")
        else:
            out.append(f"  {label:<28} {val:.3f}  VIOLATED")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--results", nargs="*", default=[],
                    help="one or more leaderboard results.json files")
    ap.add_argument("--baseline", default=None, help="baseline metrics JSON")
    ap.add_argument("--runtime", default=None, help="latency report JSON from the agent")
    ap.add_argument("--alignment", default=None, help="alignment eval JSON")
    ap.add_argument("--suite", choices=list(BENCHMARK_SUITES), default="town05_long")
    ap.add_argument("--weather", default="ClearNoon", choices=list(WEATHER_PRESETS))
    ap.add_argument("--print-launch", action="store_true",
                    help="print the leaderboard command for this suite and exit")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.print_launch:
        print(launch_command(args.config, args.checkpoint or "$SUB1B_CHECKPOINT",
                             args.suite, f"results_{args.suite}_{args.weather}.json",
                             args.weather))
        return

    if args.results:
        parsed = [parse_leaderboard_results(r) for r in args.results]
        ours = _merge_runs(parsed)
    else:
        ours = BenchmarkMetrics(
            source="no leaderboard results supplied",
            notes=["closed-loop metrics require a completed CARLA run; "
                   "pass --results <results.json>"],
        )
    ours = merge_runtime_metrics(ours, args.runtime, args.alignment)
    baseline = load_baseline(args.baseline)

    print(render_table(baseline, ours))
    print(render_provenance(baseline, ours))
    print("\nhard constraints:")
    print("\n".join(hard_constraint_report(ours)))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"baseline": baseline.__dict__, "ours": ours.__dict__}, indent=2, default=str))


def _merge_runs(runs: list[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Combine several suites/weathers into one row set, weighting per-km rates
    by distance rather than averaging rates (which would over-weight short routes)."""
    if len(runs) == 1:
        return runs[0]
    out = BenchmarkMetrics(source=" + ".join(r.source for r in runs),
                           num_routes=sum(r.num_routes for r in runs))
    for key in ("driving_score", "route_completion", "infraction_score"):
        vals = [(r.get(key), r.num_routes) for r in runs if r.get(key) is not None]
        if vals:
            n = sum(w for _, w in vals)
            setattr(out, key, sum(v * w for v, w in vals) / n if n else None)
    km = sum(r.driven_km or 0.0 for r in runs)
    if km > 0:
        out.driven_km = km
        for key in ("collisions_per_km", "red_light_per_km", "sidewalk_per_km"):
            counts = sum((r.get(key) or 0.0) * (r.driven_km or 0.0) for r in runs)
            setattr(out, key, counts / km)
    else:
        out.notes.append("no distance information; per-km rates unavailable")
    return out


if __name__ == "__main__":
    main()
