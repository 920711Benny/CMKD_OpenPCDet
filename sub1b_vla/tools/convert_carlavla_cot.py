"""Convert the CarlaVLA CoT dataset into this repo's manifest format.

Reads `cot_dataset_v3_calibrated.json` (the 634,293-record file, schema per
`pdmlite_cot_builder.build_cot_record`) and emits JSONL that
`DrivingVLADataset` can consume, preserving what makes that dataset worth
having:

  * the FACTORED action label (lon_primitive x lat_primitive), not collapsed;
  * the counterfactual collision label, which is the supervision that turns
    "does the rationale predict the action" into a measurable number;
  * the five-part TASK/CRITICAL/RULE/INTENT/COUNTERFACTUAL text.

The split is by episode_id (route), never by frame. Adjacent frames of one route
are near-duplicates, so a frame-level split leaks the validation set into
training and every metric computed on it is optimistic.

    python -m sub1b_vla.tools.convert_carlavla_cot \\
        --cot-json database/cot_dataset_v3_calibrated.json \\
        --data-root database --out database
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from ..models.carlavla_primitives import (
    LAT_COUNTS, LAT_PRIMITIVES, LON_COUNTS, LON_PRIMITIVES, normalise, to_single_intent,
)

# Scenario names that make a frame rare, from dataset_carlavla_cot.py.
RARE_SCENARIOS = {"OppositeVehicleRunningRedLight", "SignalizedJunctionLeftTurn",
                  "SignalizedJunctionRightTurn", "ControlLoss", "DynamicObjectCrossing"}
SCENARIO_RE = re.compile(r"routes_(?:training|validation)/([^/]+)/")
# Near-miss threshold: ego extent ~2.4 m plus a typical oncoming vehicle ~2.0 m
# gives a collision radius near 4.4 m, so 6.0 m is a scraped-through margin.
NEAR_MISS_GAP_M = 6.0


def scenario_of(episode_id: str) -> str:
    m = SCENARIO_RE.search(episode_id or "")
    return m.group(1) if m else "unknown"


def is_hard_or_rare(rec: dict) -> bool:
    if scenario_of(rec.get("episode_id", "")) in RARE_SCENARIOS:
        return True
    cf = rec.get("counterfactual_raw")
    if cf:
        if cf.get("counterfactual_collision") is True:
            return True
        if not cf.get("factual_collision") and cf.get("factual_min_gap", 999) < NEAR_MISS_GAP_M:
            return True
    return False


def collision_label(rec: dict):
    """Lateral counterfactual first, then longitudinal, else None.

    Lateral is preferred because it is the rarer and harder signal; None means
    the frame had no usable counterfactual (no cause object, or not visible) and
    must be masked out of the collision loss rather than treated as a negative.
    """
    for key in ("counterfactual_lat_raw", "counterfactual_raw"):
        cf = rec.get(key)
        if cf is not None:
            return 1.0 if cf.get("counterfactual_collision") else 0.0
    return None


def to_manifest_record(rec: dict, data_root: Path) -> dict | None:
    episode, frame = rec.get("episode_id"), rec.get("frame_id")
    if not episode or frame is None:
        return None
    img = Path(episode) / "rgb" / f"{frame}.jpg"
    if not (data_root / img).exists():
        return None

    meta = rec.get("meta", {}) or {}
    fields = rec.get("fields", {}) or {}
    lon = normalise(meta.get("lon_primitive", ""), "lon")
    lat = normalise(meta.get("lat_primitive", ""), "lat")
    hard = is_hard_or_rare(rec)

    return {
        "image": str(img),
        "waypoints": [],                      # language-only; read from disk if needed
        "speed_kmh": float(meta.get("speed", 0.0)) * 3.6,
        "target_point": [0.0, 0.0],
        "command": int(meta.get("command", 4)),
        "buckets": ["all"],
        "weight": 0.082,
        "long_tail": hard,
        "perception": fields.get("critical", "") or fields.get("task", ""),
        "causation": fields.get("counterfactual", "") or fields.get("rule", ""),
        "intent": to_single_intent(lon, lat),
        "lon_primitive": lon,
        "lat_primitive": lat,
        "cot_text": rec.get("text", ""),
        "collision_label": collision_label(rec),
        "safety_flag": bool(hard),
        "scenario": scenario_of(episode),
        "episode_id": episode,
        "sample_type": "drivecot",
        "generated": False,
    }


def load_records(path: Path) -> list[dict]:
    """Accepts the .json array or a .jsonl (one record per line).

    The shipped .json is ~830 MB and json.load holds the whole structure in
    memory -- expect several GB. Convert once to .jsonl if that is a problem.
    """
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-json", required=True)
    ap.add_argument("--data-root", required=True,
                    help="root that episode_id paths are relative to")
    ap.add_argument("--out", required=True, help="directory for train/val jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_records(Path(args.cot_json))
    if args.limit:
        raw = raw[: args.limit]
    print(f"read {len(raw):,} CoT records")

    # Route-level split, sorted for determinism. A frame-level split would leak:
    # adjacent frames of one route are near-duplicates.
    routes = sorted({r.get("episode_id", "") for r in raw})
    n_val = max(1, int(len(routes) * args.val_frac)) if len(routes) > 1 else 0
    rng = random.Random(args.seed)
    shuffled = routes[:]
    rng.shuffle(shuffled)
    val_routes = set(shuffled[:n_val])
    print(f"{len(routes)} routes -> {len(val_routes)} validation, "
          f"{len(routes) - len(val_routes)} train")

    kept: dict[str, list[dict]] = {"train": [], "val": []}
    dropped = Counter()
    for r in raw:
        rec = to_manifest_record(r, root)
        if rec is None:
            dropped["missing rgb frame"] += 1
            continue
        kept["val" if r.get("episode_id") in val_routes else "train"].append(rec)

    for split, recs in kept.items():
        path = out_dir / f"cot_{split}.jsonl"
        with path.open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(f"\n{path}: {len(recs):,} records")
        if not recs:
            continue
        lon = Counter(r["lon_primitive"] for r in recs)
        lat = Counter(r["lat_primitive"] for r in recs)
        print("  lon: " + ", ".join(f"{k}={v}" for k, v in lon.most_common()))
        print("  lat: " + ", ".join(f"{k}={v}" for k, v in lat.most_common()))
        with_cf = sum(r["collision_label"] is not None for r in recs)
        pos = sum(r["collision_label"] == 1.0 for r in recs)
        print(f"  counterfactual coverage: {with_cf / len(recs):.1%} "
              f"({pos} would-collide, {with_cf - pos} would-not)")
        print(f"  hard/rare: {sum(r['long_tail'] for r in recs) / len(recs):.1%}")

    if dropped:
        print("\ndropped:")
        for reason, n in dropped.most_common():
            print(f"  {n:>8}  {reason}")
    print(f"\nPoint data.extra_manifests at {out_dir / 'cot_train.jsonl'}")


if __name__ == "__main__":
    main()
