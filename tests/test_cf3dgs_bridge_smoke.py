"""Offline smoke tests for cf3dgs_bridge (no CUDA required)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cf3dgs_bridge.dataset import SequenceDataset
from cf3dgs_bridge.export_colmap import export_solved_scene_to_colmap
from cf3dgs_bridge.pose_solver import PoseSequence, eye4
from cf3dgs_bridge.progressive import ProgressiveCameraSolver, ProgressiveSolveConfig


class BridgeSmokeTest(unittest.TestCase):
    def test_progressive_dry_run_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = os.path.join(tmp, "images")
            os.makedirs(img_dir)
            for i in range(4):
                Image.fromarray(np.full((48, 64, 3), i * 40, dtype=np.uint8)).save(
                    os.path.join(img_dir, f"frame_{i:03d}.png")
                )

            ds = SequenceDataset.from_folder(tmp, width=64, height=48)
            self.assertEqual(len(ds), 4)

            poses = ProgressiveCameraSolver(ds, ProgressiveSolveConfig(dry_run=True)).solve()
            self.assertEqual(len(poses.T_world_cam), 4)
            np.testing.assert_allclose(poses.T_world_cam[0], eye4())

            out = os.path.join(tmp, "solved")
            export_solved_scene_to_colmap(ds, poses, out)
            self.assertTrue(os.path.isfile(os.path.join(out, "sparse", "0", "cameras.txt")))
            self.assertTrue(os.path.isfile(os.path.join(out, "sparse", "0", "images.txt")))
            self.assertTrue(os.path.isfile(os.path.join(out, "sparse", "0", "points3D.txt")))


if __name__ == "__main__":
    unittest.main()
