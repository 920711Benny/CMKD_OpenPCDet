#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.prompts import build_commentary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/raw")
    parser.add_argument("--output-path", default="data/processed/manifest.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_path = source_dir / "rows.jsonl"
    rows = []
    with rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["commentary"] = build_commentary(
                command=row.get("command", "follow_lane"),
                speed_kmh=float(row.get("speed_kmh", 0.0)),
                brake=float(row.get("brake", 0.0)),
                throttle=float(row.get("throttle", 0.0)),
            )
            rows.append(row)

    train_rows, val_rows = train_test_split(rows, test_size=args.val_ratio, random_state=42)
    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"

    with output_path.open("w", encoding="utf-8") as f:
        for row in train_rows + val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
