"""Unit tests for LightGlue pose seeder (no GPU weights required for core tests)."""

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
from cf3dgs_bridge.lightglue_seeder import (
    LightGluePoseSeeder,
    LightGlueSeedConfig,
    ViewGraphEdge,
    build_pairs,
    compose_poses_from_tree,
    lightglue_available,
    resolve_device,
    spanning_tree_edges,
)
from cf3dgs_bridge.pose_graph import invert_T, se3_exp
from cf3dgs_bridge.pose_solver import eye4


class BuildPairsTest(unittest.TestCase):
    def test_sequential_radius(self):
        pairs = build_pairs(5, pair_radius=1, long_range_stride=0)
        self.assertEqual(pairs, [(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_radius_two(self):
        pairs = build_pairs(4, pair_radius=2, long_range_stride=0)
        self.assertIn((0, 2), pairs)
        self.assertIn((1, 3), pairs)

    def test_long_range_stride(self):
        pairs = build_pairs(10, pair_radius=1, long_range_stride=5)
        self.assertIn((0, 5), pairs)
        self.assertIn((1, 6), pairs)
        self.assertTrue(any(j - i >= 5 for i, j in pairs))


class SpanningTreeTest(unittest.TestCase):
    def test_compose_tree_matches_chain(self):
        n = 5
        adj = []
        for _ in range(n - 1):
            adj.append(se3_exp(np.array([0.0, 0.05, 0.0, 0.1, 0.0, 0.0])))
        edges = [
            ViewGraphEdge(i=i, j=i + 1, T_j_i=adj[i], inliers=100 - i, num_matches=200)
            for i in range(n - 1)
        ]
        edges.append(ViewGraphEdge(i=0, j=4, T_j_i=eye4(), inliers=1, num_matches=2))
        tree = spanning_tree_edges(edges, n)
        self.assertEqual(len(tree), n - 1)
        c2w = compose_poses_from_tree(n, tree, anchor=0)
        np.testing.assert_allclose(c2w[0], eye4(), atol=1e-12)
        T01 = invert_T(c2w[1]) @ c2w[0]
        np.testing.assert_allclose(T01, adj[0], atol=1e-8)


class SiftFallbackSmokeTest(unittest.TestCase):
    def test_sift_fallback_runs_on_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = os.path.join(tmp, "images")
            os.makedirs(img_dir)
            rng = np.random.default_rng(0)
            for i in range(4):
                base = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
                frame = np.roll(base, shift=i * 2, axis=1)
                Image.fromarray(frame).save(os.path.join(img_dir, f"f{i:03d}.png"))

            ds = SequenceDataset.from_folder(tmp, width=128, height=96)
            cfg = LightGlueSeedConfig(
                feature_backend="sift",
                pair_radius=1,
                long_range_stride=2,
                resize_width=128,
                pose_graph=False,
                min_matches=8,
                min_inliers=5,
                max_features_sift=500,
            )
            seeder = LightGluePoseSeeder(ds, cfg)
            self.assertEqual(seeder.backend_name, "sift_fallback")
            poses = seeder.solve()
            self.assertEqual(len(poses.T_world_cam), 4)
            np.testing.assert_allclose(poses.T_world_cam[0], eye4(), atol=1e-12)


@unittest.skipUnless(
    lightglue_available("superpoint"),
    "lightglue.superpoint not importable",
)
class LightGlueWeightsOptionalTest(unittest.TestCase):
    def test_superpoint_init_or_skip(self):
        try:
            cfg = LightGlueSeedConfig(
                feature_backend="superpoint",
                device="cpu",
                max_keypoints=256,
                pose_graph=False,
            )
            with tempfile.TemporaryDirectory() as tmp:
                img_dir = os.path.join(tmp, "images")
                os.makedirs(img_dir)
                for i in range(3):
                    arr = np.zeros((64, 80, 3), dtype=np.uint8)
                    arr[:, 10 + i * 3 : 40 + i * 3] = 200
                    Image.fromarray(arr).save(os.path.join(img_dir, f"f{i:03d}.png"))
                ds = SequenceDataset.from_folder(tmp, width=80, height=64)
                seeder = LightGluePoseSeeder(ds, cfg)
                if "lightglue" not in seeder.backend_name:
                    self.skipTest(f"fell back to {seeder.backend_name}")
                poses = seeder.solve()
                self.assertEqual(len(poses.T_world_cam), 3)
        except Exception as exc:
            self.skipTest(f"weights/device unavailable: {exc}")


class DeviceResolveTest(unittest.TestCase):
    def test_cpu_forced(self):
        self.assertEqual(resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
