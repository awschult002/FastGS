#!/usr/bin/env python3
"""Synthetic large-jump stress: subsample Francis (stride N) and compare ATE.

Simulates big camera jumps between kept frames. Adjacent-only DepthPnP must
bridge large motion; hybrid adds LightGlue long-range edges that should help
when sequential matching is weak.

Example::

    .venv/bin/python scripts/eval_large_jump_stress.py \\
      --scene /workspace/repos/data/Tanks/Francis \\
      --stride 5 --max_keep 24 --resize_width 640 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cf3dgs_bridge.depth_pnp_solver import DepthPnPProgressiveSolver, DepthPnPSolveConfig
from cf3dgs_bridge.hybrid_seeder import HybridPoseSeeder, HybridSeedConfig
from cf3dgs_bridge.opencv_solver import absolute_trajectory_error
from cf3dgs_bridge.tanks_loader import load_tanks_scene


def subsample_indices(n: int, stride: int, max_keep: int = 0) -> List[int]:
    """Keep every ``stride``-th frame (0-based), optionally capped."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    idx = list(range(0, n, stride))
    if max_keep and max_keep < len(idx):
        idx = idx[:max_keep]
    return idx


def apply_subsample(ds, depth_paths, gt, indices: Sequence[int]):
    ds.image_paths = [ds.image_paths[i] for i in indices]
    depth_paths = [depth_paths[i] for i in indices]
    gt_sub = None
    if gt is not None:
        gt_sub = [gt[i] for i in indices]
    return ds, depth_paths, gt_sub


def _ate(poses, gt) -> dict:
    return absolute_trajectory_error(poses.T_world_cam, gt)


def main() -> int:
    p = argparse.ArgumentParser(description="Large-jump stress: depth_pnp vs hybrid")
    p.add_argument("--scene", required=True)
    p.add_argument("--stride", type=int, default=5, help="Keep every N-th frame")
    p.add_argument("--max_keep", type=int, default=24, help="Cap kept frames (0=all)")
    p.add_argument("--resize_width", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--pair_radius", type=int, default=2)
    p.add_argument("--long_range_stride", type=int, default=4)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    ds, gt, depth_paths = load_tanks_scene(args.scene)
    if gt is None:
        raise SystemExit("Need GT poses for ATE comparison")
    if not any(depth_paths):
        raise SystemExit("Need dpt/ depth maps for depth_pnp / hybrid")

    n_full = len(ds)
    indices = subsample_indices(n_full, args.stride, args.max_keep)
    if len(indices) < 3:
        raise SystemExit(f"Need >=3 kept frames; got {len(indices)}")

    ds, depth_paths, gt_sub = apply_subsample(ds, depth_paths, gt, indices)
    name = os.path.basename(os.path.normpath(args.scene))
    print(
        f"[large_jump] {name}: full={n_full} → keep={len(indices)} "
        f"(stride={args.stride}, indices[:5]={indices[:5]}...)"
    )

    # --- sequential-only DepthPnP on sparse frames ---
    print("[large_jump] Running depth_pnp (sequential-only)…")
    dcfg = DepthPnPSolveConfig(
        resize_width=args.resize_width,
        feature_backend="sift",
        pose_graph=False,
        photo_refine=False,
    )
    dsolver = DepthPnPProgressiveSolver(ds, depth_paths, dcfg)
    dposes = dsolver.solve()
    d_metrics = _ate(dposes, gt_sub)

    # --- hybrid with LightGlue long-range ---
    print("[large_jump] Running hybrid (DepthPnP + LightGlue jumps)…")
    # Reload paths (ds already subsampled; hybrid mutates nothing essential)
    hcfg = HybridSeedConfig(
        sequential_radius=1,
        pair_radius=args.pair_radius,
        long_range_stride=args.long_range_stride,
        resize_width=args.resize_width if args.resize_width > 0 else None,
        feature_backend="superpoint",
        device=args.device,
        pose_graph=True,
        depth_pnp_weight=100.0,
        lightglue_weight=0.05,
    )
    hseeder = HybridPoseSeeder(ds, depth_paths, hcfg)
    hposes = hseeder.solve()
    h_metrics = _ate(hposes, gt_sub)

    winner = "hybrid" if h_metrics["ate_rmse"] <= d_metrics["ate_rmse"] else "depth_pnp"
    # "not collapse" = hybrid finite and not wildly worse than depth_pnp * 2
    hybrid_ok = (
        np.isfinite(h_metrics["ate_rmse"])
        and h_metrics["ate_rmse"] < max(1.0, 2.0 * d_metrics["ate_rmse"] + 0.05)
    )

    result = {
        "scene": name,
        "stride": args.stride,
        "max_keep": args.max_keep,
        "num_full": n_full,
        "num_kept": len(indices),
        "indices_head": indices[:8],
        "resize_width": args.resize_width,
        "device": args.device,
        "depth_pnp": {
            "ate_rmse": d_metrics["ate_rmse"],
            "ate_mean": d_metrics["ate_mean"],
            "scale": d_metrics["scale"],
        },
        "hybrid": {
            "ate_rmse": h_metrics["ate_rmse"],
            "ate_mean": h_metrics["ate_mean"],
            "scale": h_metrics["scale"],
            "backend": hseeder.backend_name,
            "used_lightglue": hseeder.used_lightglue,
            "jump_edges": len(hseeder.jump_edges),
            "sequential_edges": len(hseeder.sequential_edges),
            "scale_ratio": hseeder.scale_ratio,
        },
        "winner": winner,
        "hybrid_ok": hybrid_ok,
        "ate_delta_hybrid_minus_depth_pnp": h_metrics["ate_rmse"] - d_metrics["ate_rmse"],
    }
    print(json.dumps(result, indent=2))
    print(
        f"[large_jump] depth_pnp ATE={d_metrics['ate_rmse']:.4f}  "
        f"hybrid ATE={h_metrics['ate_rmse']:.4f}  winner={winner}"
    )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
