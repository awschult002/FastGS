"""Unit tests for depth+PnP helpers (mocked depth, no real scene required)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cf3dgs_bridge.depth_pnp_solver import (
    discover_depth_maps,
    frame_numeric_id,
    load_depth_npz,
    resize_depth,
    unproject_pixels,
)
from cf3dgs_bridge.opencv_solver import recover_pose_opencv5


class DepthPnPHelperTest(unittest.TestCase):
    def test_frame_numeric_id(self):
        self.assertEqual(frame_numeric_id("/x/000401.jpg"), "000401")
        self.assertEqual(frame_numeric_id("depth_000401.npz"), "000401")
        self.assertEqual(frame_numeric_id("frame_12.png"), "12")

    def test_discover_and_load_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            dpt = os.path.join(tmp, "dpt")
            os.makedirs(dpt)
            depth = np.random.rand(1, 32, 48).astype(np.float32) + 1.0
            np.savez(os.path.join(dpt, "depth_000007.npz"), pred=depth)
            images = [
                os.path.join(tmp, "images", "000007.jpg"),
                os.path.join(tmp, "images", "000008.jpg"),
            ]
            paths = discover_depth_maps(tmp, images, depth_subdir="dpt")
            self.assertEqual(len(paths), 2)
            self.assertIsNotNone(paths[0])
            self.assertIsNone(paths[1])
            loaded = load_depth_npz(paths[0], key="pred")
            self.assertEqual(loaded.shape, (32, 48))
            resized = resize_depth(loaded, 96, 64)
            self.assertEqual(resized.shape, (64, 96))

    def test_unproject_pixels(self):
        h, w = 40, 60
        depth = np.full((h, w), 2.0, dtype=np.float32)
        K = np.array([[50.0, 0, 30.0], [0, 50.0, 20.0], [0, 0, 1]], dtype=np.float64)
        pts = np.array([[30.0, 20.0], [40.0, 20.0]], dtype=np.float64)
        pts3d, valid = unproject_pixels(pts, depth, K, min_depth=0.1, max_depth=10.0)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(pts3d[0], [0.0, 0.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(pts3d[1, 2], 2.0, atol=1e-6)
        self.assertGreater(pts3d[1, 0], 0.0)

    def test_recover_pose_opencv5_smoke(self):
        # Synthetic pure translation with known E via findEssentialMat
        import cv2

        K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1]], dtype=np.float64)
        rng = np.random.default_rng(0)
        # Random 3D points in front of cam0
        X = rng.uniform(-1, 1, size=(80, 3))
        X[:, 2] = rng.uniform(2.0, 5.0, size=80)
        R_gt = np.eye(3)
        t_gt = np.array([0.2, 0.0, 0.05])
        X2 = (R_gt @ X.T).T + t_gt
        pts1 = (K @ X.T).T
        pts1 = pts1[:, :2] / pts1[:, 2:3]
        pts2 = (K @ X2.T).T
        pts2 = pts2[:, :2] / pts2[:, 2:3]
        E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        self.assertIsNotNone(E)
        n, R, t, mask_pose = recover_pose_opencv5(E, pts1, pts2, K, mask=mask)
        self.assertGreater(n, 20)
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(t.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
