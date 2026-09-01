"""Compute the waypoint normalisation scale for a dataset.

The diffusion head needs approximately unit-variance targets. Run this on your
prepared manifests and put the result in `model.waypoint_scale`; a scale that
does not match the data is the difference between a low training loss and a
sampler that actually produces drivable trajectories.

    python -m sub1b_vla.tools.waypoint_stats --config sub1b_vla/configs/default.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..data.dataset import DrivingVLADataset, collate
from ..utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = DrivingVLADataset(cfg, args.split)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=collate, num_workers=0)
    chunks = []
    for i, b in enumerate(loader):
        if i >= args.max_batches:
            break
        chunks.append(b["waypoints"].numpy())
    wp = np.concatenate(chunks, 0).reshape(-1, 2)

    std = wp.std(0)
    p99 = np.percentile(np.abs(wp), 99, axis=0)
    print(f"samples          : {wp.shape[0]:,} waypoints")
    print(f"mean   (fwd,left): {wp.mean(0)[0]:8.3f} {wp.mean(0)[1]:8.3f}  m")
    print(f"std              : {std[0]:8.3f} {std[1]:8.3f}  m")
    print(f"p99 |.|          : {p99[0]:8.3f} {p99[1]:8.3f}  m")
    print(f"max |.|          : {np.abs(wp).max(0)[0]:8.3f} {np.abs(wp).max(0)[1]:8.3f}  m")
    # Min-max with a 15% margin so the sampler's +-1 clip never truncates real
    # data, and so a slightly wider test distribution still fits.
    lo, hi = wp.min(0), wp.max(0)
    span = np.maximum(hi - lo, 1e-3) * 1.15
    offset = (hi + lo) / 2.0
    scale = span / 2.0
    norm = (wp - offset) / scale
    print(f"\nsuggested model.waypoint_offset: [{offset[0]:.1f}, {offset[1]:.1f}]")
    print(f"suggested model.waypoint_scale : [{scale[0]:.1f}, {scale[1]:.1f}]")
    print(f"normalised range               : "
          f"[{norm.min(0)[0]:.3f}, {norm.max(0)[0]:.3f}] x "
          f"[{norm.min(0)[1]:.3f}, {norm.max(0)[1]:.3f}]  (must lie inside [-1, 1])")
    print(f"normalised std                 : {norm.std(0)[0]:.3f} {norm.std(0)[1]:.3f}")


if __name__ == "__main__":
    main()
