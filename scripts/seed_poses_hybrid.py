#!/usr/bin/env python3
"""CLI: Hybrid DepthPnP + LightGlue pose seed → optional COLMAP export for FastGS.

Pipeline: adjacent metric DepthPnP + long-range LightGlue (scale-aligned) →
optional pose-graph → export_colmap → FastGS train.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cf3dgs_bridge.export_colmap import export_solved_scene_to_colmap
from cf3dgs_bridge.hybrid_seeder import HybridPoseSeeder, HybridSeedConfig
from cf3dgs_bridge.tanks_loader import load_tanks_scene


def main() -> int:
    p = argparse.ArgumentParser(
        description="Hybrid DepthPnP (sequential) + LightGlue (jumps) pose seed"
    )
    p.add_argument("--scene", required=True, help="Scene root (Tanks layout with dpt/)")
    p.add_argument("--out_colmap", type=str, default="", help="Export COLMAP sparse model here")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--sequential_radius", type=int, default=1)
    p.add_argument("--pair_radius", type=int, default=2)
    p.add_argument("--long_range_stride", type=int, default=10)
    p.add_argument("--resize_width", type=int, default=960)
    p.add_argument("--max_keypoints", type=int, default=2048)
    p.add_argument(
        "--feature_backend",
        type=str,
        default="superpoint",
        choices=["superpoint", "aliked", "auto", "sift"],
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_pose_graph", action="store_true", help="Disable SE3 pose-graph fuse")
    p.add_argument("--depth_pnp_weight", type=float, default=100.0)
    p.add_argument("--lightglue_weight", type=float, default=0.05)
    p.add_argument("--no_scale_align", action="store_true")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images into export images/ instead of symlinking",
    )
    p.add_argument(
        "--model_hint",
        type=str,
        default="output/<run>",
        help="Hint path written into export README / printed train command",
    )
    args = p.parse_args()

    ds, _gt, depth_paths = load_tanks_scene(args.scene)
    if args.max_frames and args.max_frames < len(ds):
        ds.image_paths = ds.image_paths[: args.max_frames]
        depth_paths = depth_paths[: args.max_frames]

    if not any(depth_paths):
        raise SystemExit(
            f"No DPT depth maps under {args.scene}/dpt; hybrid needs DepthPnP"
        )

    cfg = HybridSeedConfig(
        sequential_radius=args.sequential_radius,
        pair_radius=args.pair_radius,
        long_range_stride=args.long_range_stride,
        resize_width=args.resize_width if args.resize_width > 0 else None,
        max_keypoints=args.max_keypoints,
        feature_backend=args.feature_backend,
        device=args.device,
        pose_graph=not args.no_pose_graph,
        depth_pnp_weight=args.depth_pnp_weight,
        lightglue_weight=args.lightglue_weight,
        scale_align=not args.no_scale_align,
    )
    seeder = HybridPoseSeeder(ds, depth_paths, cfg)
    poses = seeder.solve()

    result = {
        "scene": os.path.basename(os.path.normpath(args.scene)),
        "num_frames": len(ds),
        "backend": seeder.backend_name,
        "used_lightglue": seeder.used_lightglue,
        "scale_ratio": seeder.scale_ratio,
        "sequential_edges": len(seeder.sequential_edges),
        "jump_edges": len(seeder.jump_edges),
        "sequential_radius": args.sequential_radius,
        "pair_radius": args.pair_radius,
        "long_range_stride": args.long_range_stride,
        "pose_graph": bool(cfg.pose_graph),
    }
    print(json.dumps(result, indent=2))

    if args.out_colmap:
        export_solved_scene_to_colmap(
            ds,
            poses,
            args.out_colmap,
            force_copy_images=bool(args.copy_images),
            write_train_readme=True,
            model_path_hint=args.model_hint,
        )
        result["out_colmap"] = args.out_colmap
        sparse0 = os.path.join(args.out_colmap, "sparse", "0")
        images_dir = os.path.join(args.out_colmap, "images")
        n_img = (
            len(
                [
                    f
                    for f in os.listdir(images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            if os.path.isdir(images_dir)
            else 0
        )
        result["layout"] = {
            "sparse0": sparse0,
            "images": images_dir,
            "num_images": n_img,
            "has_cameras_txt": os.path.isfile(os.path.join(sparse0, "cameras.txt")),
            "has_images_txt": os.path.isfile(os.path.join(sparse0, "images.txt")),
            "has_points3D_txt": os.path.isfile(os.path.join(sparse0, "points3D.txt")),
        }
        print(f"Exported COLMAP model → {args.out_colmap}")
        print(
            "Layout OK: sparse/0/{cameras,images,points3D}.txt + images/ "
            f"({n_img} files). Scene() expects -s pointing at this root."
        )
        print(
            "Next: python train.py -s %s -m %s"
            % (args.out_colmap, args.model_hint)
        )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
