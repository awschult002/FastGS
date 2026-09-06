#!/usr/bin/env python3
"""Evaluate COLMAP-free pose solve ATE vs GT (Nope-NeRF Tanks / CF-3DGS protocol)."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cf3dgs_bridge.opencv_solver import OpenCVProgressiveSolver, OpenCVSolveConfig, absolute_trajectory_error
from cf3dgs_bridge.tanks_loader import load_tanks_scene

PAPER_ATE = {
    "Church": 0.002,
    "Barn": 0.003,
    "Museum": 0.005,
    "Family": 0.002,
    "Horse": 0.003,
    "Ballroom": 0.002,
    "Francis": 0.006,
    "Ignatius": 0.002,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to a Tanks scene dir e.g. data/Tanks/Francis")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--resize_width", type=int, default=960)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    ds, gt = load_tanks_scene(args.scene)
    if args.max_frames and args.max_frames < len(ds):
        ds.image_paths = ds.image_paths[: args.max_frames]
        if gt is not None:
            gt = gt[: args.max_frames]

    name = os.path.basename(os.path.normpath(args.scene))
    print(f"Scene {name}: {len(ds)} frames, GT poses={'yes' if gt else 'no'}")

    solver = OpenCVProgressiveSolver(ds, OpenCVSolveConfig(resize_width=args.resize_width))
    poses = solver.solve()

    result = {"scene": name, "num_frames": len(ds), "backend": "opencv_orb_essential"}
    if gt is None:
        print("WARNING: no GT poses found; cannot compute ATE")
        result["error"] = "no_gt"
    else:
        metrics = absolute_trajectory_error(poses.T_world_cam, gt)
        result.update(metrics)
        paper = PAPER_ATE.get(name)
        result["paper_cf3dgs_ate"] = paper
        print(json.dumps(result, indent=2))
        if paper is not None:
            print(
                f"Compare: our ATE={metrics['ate_rmse']:.4f} vs CF-3DGS paper ATE={paper:.4f} "
                f"(lower is better). OpenCV baseline is not expected to match photometric CF-3DGS."
            )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
