#!/usr/bin/env python3
"""COLMAP-free FastGS entrypoint (CF-3DGS-inspired camera solve → FastGS train).

Phase A: progressive pose solve on an image sequence (no ``sparse/`` needed).
Phase B: export a COLMAP-compatible scene.
Phase C (optional): call existing ``train.py`` on the exported scene.

Examples
--------
Dry-run solve + export only::

    python train_colmap_free.py -s /data/my_video --export_dir /data/my_video_solved

Then classic FastGS::

    python train.py -s /data/my_video_solved -m output/my_run

Or chain both (still uses dry-run poses until photometric solve is implemented)::

    python train_colmap_free.py -s /data/my_video --export_dir /data/my_video_solved --run_fastgs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from cf3dgs_bridge.dataset import SequenceDataset
from cf3dgs_bridge.export_colmap import export_solved_scene_to_colmap
from cf3dgs_bridge.progressive import ProgressiveCameraSolver, ProgressiveSolveConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="COLMAP-free FastGS via CF-3DGS-style pose solve")
    p.add_argument("-s", "--source_path", required=True, help="Folder with images/ (or images at root)")
    p.add_argument("--images", default="images", help="Images subdir name")
    p.add_argument("--export_dir", required=True, help="Where to write COLMAP sparse scene")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--local_iters", type=int, default=1000)
    p.add_argument("--single_step", type=int, default=500)
    p.add_argument(
        "--no_dry_run",
        action="store_true",
        help="Attempt real photometric solve (not implemented yet — will raise)",
    )
    p.add_argument(
        "--run_fastgs",
        action="store_true",
        help="After export, invoke train.py on the solved scene",
    )
    p.add_argument("-m", "--model_path", default="", help="FastGS output model path (with --run_fastgs)")
    p.add_argument("--fastgs_extra", nargs=argparse.REMAINDER, help="Extra args after -- passed to train.py")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset = SequenceDataset.from_folder(
        args.source_path,
        images_subdir=args.images,
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
    )
    cfg = ProgressiveSolveConfig(
        local_iters=args.local_iters,
        single_step=args.single_step,
        dry_run=not args.no_dry_run,
    )
    solver = ProgressiveCameraSolver(dataset, cfg)
    poses = solver.solve()
    export_root = export_solved_scene_to_colmap(dataset, poses, args.export_dir)

    if args.run_fastgs:
        train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        cmd = [sys.executable, train_py, "-s", export_root]
        if args.model_path:
            cmd += ["-m", args.model_path]
        extra = args.fastgs_extra or []
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd += extra
        print("[cf3dgs_bridge] Launching FastGS:", " ".join(cmd))
        return subprocess.call(cmd)

    print("[cf3dgs_bridge] Done. Next: python train.py -s", export_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
