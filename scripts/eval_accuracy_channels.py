#!/usr/bin/env python3
"""Ablation runner for complementary accuracy channels (edge filter / multi-kf).

Writes JSON under eval_out/ without deleting other workers' results.
Reports ATE + RPE (trans/rot).

Examples
--------
  .venv/bin/python scripts/eval_accuracy_channels.py \\
      --scene /workspace/repos/data/Tanks/Francis --max_frames 40 --channel filtered_skips

  .venv/bin/python scripts/eval_accuracy_channels.py \\
      --scene ... --channel multi_kf --kf_stride 5 --kf_window 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cf3dgs_bridge.depth_pnp_solver import DepthPnPProgressiveSolver, DepthPnPSolveConfig
from cf3dgs_bridge.edge_filter import EdgeQualityConfig
from cf3dgs_bridge.multi_keyframe import MultiKeyframeConfig, MultiKeyframeDepthPnPSolver
from cf3dgs_bridge.opencv_solver import absolute_trajectory_error, relative_pose_error
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


def _parse_ks(s: str) -> tuple:
    return tuple(int(p.strip()) for p in s.split(",") if p.strip())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--resize_width", type=int, default=960)
    p.add_argument(
        "--channel",
        required=True,
        choices=[
            "baseline",
            "filtered_skips",
            "multi_kf",
            "multi_kf_all",
            "multi_kf_plus_skips",
        ],
        help="Accuracy channel to evaluate",
    )
    p.add_argument("--skip_ks", type=str, default="2,5")
    p.add_argument("--kf_stride", type=int, default=5)
    p.add_argument("--kf_window", type=int, default=3)
    p.add_argument("--min_inliers", type=int, default=30)
    p.add_argument("--no_align_t", action="store_true",
                   help="Keep raw PnP translation on long-range edges")
    p.add_argument("--rpe_delta", type=int, default=1)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    ds, gt, depth_paths = load_tanks_scene(args.scene)
    if args.max_frames and args.max_frames < len(ds):
        ds.image_paths = ds.image_paths[: args.max_frames]
        depth_paths = depth_paths[: args.max_frames]
        if gt is not None:
            gt = gt[: args.max_frames]

    name = os.path.basename(os.path.normpath(args.scene))
    n = len(ds)
    depth_cfg = DepthPnPSolveConfig(
        resize_width=args.resize_width,
        feature_backend="sift",
        pose_graph=False,
        photo_refine=False,
    )
    eq = EdgeQualityConfig(min_inliers=args.min_inliers)

    if args.channel == "baseline":
        solver = DepthPnPProgressiveSolver(ds, depth_paths, depth_cfg)
        poses = solver.solve()
        backend = f"depth_pnp_{solver.backend_name}_baseline"
        channel_meta = {"channel": "baseline"}
    else:
        if args.channel == "filtered_skips":
            mkf = MultiKeyframeConfig(
                mode="filtered_skips",
                extra_skip_ks=_parse_ks(args.skip_ks),
                use_edge_filter=True,
                edge_quality=eq,
                align_t_to_chain=not args.no_align_t,
            )
        elif args.channel == "multi_kf":
            mkf = MultiKeyframeConfig(
                mode="multi_kf",
                kf_stride=args.kf_stride,
                kf_window=args.kf_window,
                fuse_mode="best",
                use_edge_filter=True,
                edge_quality=eq,
                extra_skip_ks=(),
                align_t_to_chain=not args.no_align_t,
            )
        elif args.channel == "multi_kf_all":
            mkf = MultiKeyframeConfig(
                mode="multi_kf",
                kf_stride=args.kf_stride,
                kf_window=args.kf_window,
                fuse_mode="all",
                use_edge_filter=True,
                edge_quality=eq,
                extra_skip_ks=(),
                align_t_to_chain=not args.no_align_t,
            )
        else:  # multi_kf_plus_skips
            mkf = MultiKeyframeConfig(
                mode="multi_kf",
                kf_stride=args.kf_stride,
                kf_window=args.kf_window,
                fuse_mode="best",
                use_edge_filter=True,
                edge_quality=eq,
                extra_skip_ks=_parse_ks(args.skip_ks),
                align_t_to_chain=not args.no_align_t,
            )
        solver = MultiKeyframeDepthPnPSolver(ds, depth_paths, depth_cfg, mkf)
        poses = solver.solve()
        backend = f"multi_kf_{solver.backend_name}_{args.channel}"
        channel_meta = {
            "channel": args.channel,
            "kf_stride": args.kf_stride,
            "kf_window": args.kf_window,
            "skip_ks": list(_parse_ks(args.skip_ks)),
            "min_inliers": args.min_inliers,
            "fuse_mode": mkf.fuse_mode,
            "mode": mkf.mode,
        }

    result = {
        "scene": name,
        "num_frames": n,
        "backend": backend,
        "feature_backend": getattr(solver, "backend_name", "sift"),
        **channel_meta,
    }
    if gt is None:
        result["error"] = "no_gt"
    else:
        ate = absolute_trajectory_error(poses.T_world_cam, gt)
        rpe = relative_pose_error(poses.T_world_cam, gt, delta=args.rpe_delta)
        result.update(ate)
        result.update(rpe)
        result["paper_cf3dgs_ate"] = PAPER_ATE.get(name)
        print(json.dumps(result, indent=2))

    out = args.out_json
    if not out:
        tag = f"francis_{args.channel}_{n}"
        out = os.path.join(ROOT, "eval_out", f"{tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
