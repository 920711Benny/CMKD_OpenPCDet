#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="immanuelpeter/carla-autopilot-multimodal-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"

    ds = load_dataset(args.dataset_id, split=args.split, streaming=True)
    count = 0
    with rows_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(ds, total=args.max_samples)):
            if count >= args.max_samples:
                break
            image = row.get("image_front")
            if image is None:
                continue
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            elif isinstance(image, dict) and "bytes" in image:
                pil_image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
            else:
                continue
            image_path = image_dir / f"sample_{count:06d}.jpg"
            pil_image.save(image_path, quality=95)
            sample = {
                "image_path": str(image_path),
                "speed_kmh": float(row.get("speed", row.get("speed_kmh", 0.0))),
                "steer": float(row.get("steer", 0.0)),
                "throttle": float(row.get("throttle", 0.0)),
                "brake": float(row.get("brake", 0.0)),
                "command": str(row.get("command", "follow_lane")),
                "run_id": str(row.get("run_id", "unknown")),
                "frame": int(row.get("frame", idx)),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    print(f"Saved {count} samples to {rows_path}")


if __name__ == "__main__":
    main()
