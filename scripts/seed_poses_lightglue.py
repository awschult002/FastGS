#!/usr/bin/env python3
"""CLI: LightGlue (SuperPoint) pose seed → optional COLMAP export for FastGS.

Pipeline: images → LightGlue view-graph seed → (optional progressive refine) → FastGS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cf3dgs_bridge.dataset import SequenceDataset
from cf3dgs_bridge.export_colmap import export_solved_scene_to_colmap
from cf3dgs_bridge.lightglue_seeder import LightGluePoseSeeder, LightGlueSeedConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Seed camera poses with SuperPoint+LightGlue")
    p.add_argument("--scene", required=True, help="Scene root with images/ (or Tanks layout)")
    p.add_argument("--out_colmap", type=str, default="", help="Export COLMAP sparse model here")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--pair_radius", type=int, default=1)
    p.add_argument("--long_range_stride", type=int, default=0)
    p.add_argument("--resize_width", type=int, default=960)
    p.add_argument("--max_keypoints", type=int, default=2048)
    p.add_argument(
        "--feature_backend",
        type=str,
        default="superpoint",
        choices=["superpoint", "aliked", "auto", "sift"],
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--pose_graph", action="store_true", help="Opt-in SE3 pose-graph refine")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--images_subdir", type=str, default="images")
    args = p.parse_args()

    # Prefer Tanks loader when poses_bounds / dpt layout present; else folder dataset
    scene = args.scene
    try:
        from cf3dgs_bridge.tanks_loader import load_tanks_scene

        ds, _gt, _depths = load_tanks_scene(scene)
    except Exception:
        ds = SequenceDataset.from_folder(scene, images_subdir=args.images_subdir)

    if args.max_frames and args.max_frames < len(ds):
        ds.image_paths = ds.image_paths[: args.max_frames]

    cfg = LightGlueSeedConfig(
        pair_radius=args.pair_radius,
        long_range_stride=args.long_range_stride,
        resize_width=args.resize_width if args.resize_width > 0 else None,
        max_keypoints=args.max_keypoints,
        feature_backend=args.feature_backend,
        device=args.device,
        pose_graph=bool(args.pose_graph),
    )
    seeder = LightGluePoseSeeder(ds, cfg)
    poses = seeder.solve()

    result = {
        "scene": os.path.basename(os.path.normpath(scene)),
        "num_frames": len(ds),
        "backend": seeder.backend_name,
        "device": seeder.device,
        "pair_radius": args.pair_radius,
        "long_range_stride": args.long_range_stride,
        "view_graph_edges": len(seeder.view_graph),
        "tree_edges": len(seeder.tree_edges),
        "pose_graph": bool(cfg.pose_graph),
    }
    print(json.dumps(result, indent=2))

    if args.out_colmap:
        export_solved_scene_to_colmap(ds, poses, args.out_colmap)
        result["out_colmap"] = args.out_colmap
        print(f"Exported COLMAP model → {args.out_colmap}")
        print(
            "Next: python train.py -s %s -m output/<run>  "
            "(or optional CF-3DGS progressive refine first)" % args.out_colmap
        )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
