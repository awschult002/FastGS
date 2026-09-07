"""Unit tests for hybrid DepthPnP + LightGlue seeder."""

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
from cf3dgs_bridge.hybrid_seeder import (
    HybridPoseSeeder,
    HybridSeedConfig,
    apply_frame_gap_translation_prior,
    compose_chain_relative,
    estimate_lightglue_scale_ratio,
    jump_pairs_only,
    scale_align_lightglue_edges,
    scale_edge_translations,
)
from cf3dgs_bridge.lightglue_seeder import ViewGraphEdge
from cf3dgs_bridge.pose_graph import se3_exp
from cf3dgs_bridge.pose_solver import eye4


class JumpPairsTest(unittest.TestCase):
    def test_excludes_sequential(self):
        pairs = jump_pairs_only(10, sequential_radius=1, pair_radius=2, long_range_stride=5)
        for i, j in pairs:
            self.assertGreater(j - i, 1)
        self.assertIn((0, 5), pairs)
        self.assertIn((0, 2), pairs)  # pair_radius=2 beyond sequential_radius=1


class ScaleAlignHelperTest(unittest.TestCase):
    def test_estimate_median_ratio(self):
        # Adjacent translations of length 0.2 each
        adj = []
        for _ in range(4):
            adj.append(se3_exp(np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])))
        # LightGlue unit t along same direction for (0, 4): composed t ≈ 0.8
        T_lg = eye4()
        T_lg[:3, 3] = np.array([1.0, 0.0, 0.0])  # unit
        edges = [ViewGraphEdge(i=0, j=4, T_j_i=T_lg, inliers=50, num_matches=100)]
        # Also (1, 3): composed ≈ 0.4
        T_lg2 = eye4()
        T_lg2[:3, 3] = np.array([1.0, 0.0, 0.0])
        edges.append(ViewGraphEdge(i=1, j=3, T_j_i=T_lg2, inliers=40, num_matches=80))

        ratio = estimate_lightglue_scale_ratio(adj, edges)
        # median of [0.8/1, 0.4/1] = 0.6
        self.assertAlmostEqual(ratio, 0.6, places=5)

        scaled = scale_edge_translations(edges, ratio)
        self.assertAlmostEqual(
            float(np.linalg.norm(scaled[0].T_j_i[:3, 3])), 0.6, places=5
        )

    def test_compose_chain_relative(self):
        adj = [
            se3_exp(np.array([0.0, 0.05, 0.0, 0.1, 0.0, 0.0])),
            se3_exp(np.array([0.0, 0.05, 0.0, 0.1, 0.0, 0.0])),
        ]
        T = compose_chain_relative(adj, 0, 2)
        expected = adj[1] @ adj[0]
        np.testing.assert_allclose(T, expected, atol=1e-10)


    def test_gap_prior_then_median(self):
        adj = [se3_exp(np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])) for _ in range(4)]
        # unit essential for gap=4 and gap=2
        edges = []
        for i, j in [(0, 4), (1, 3)]:
            T = eye4()
            T[:3, 3] = np.array([1.0, 0.0, 0.0])
            edges.append(ViewGraphEdge(i=i, j=j, T_j_i=T, inliers=50, num_matches=100))
        scaled, ratio = scale_align_lightglue_edges(adj, edges, use_gap_prior=True)
        # After gap prior: t=4 and t=2; seq norms ≈0.4 and 0.2 → ratios 0.1, 0.1
        self.assertAlmostEqual(ratio, 0.1, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(scaled[0].T_j_i[:3, 3])), 0.4, places=5)

    def test_empty_edges_ratio_one(self):
        adj = [eye4()]
        self.assertEqual(estimate_lightglue_scale_ratio(adj, []), 1.0)


class HybridSmokeTest(unittest.TestCase):
    def test_sequential_only_fallback_synthetic(self):
        """Without real depth, DepthPnP returns identity relatives; still runs."""
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = os.path.join(tmp, "images")
            dpt_dir = os.path.join(tmp, "dpt")
            os.makedirs(img_dir)
            os.makedirs(dpt_dir)
            rng = np.random.default_rng(1)
            for i in range(4):
                base = rng.integers(0, 255, size=(64, 80, 3), dtype=np.uint8)
                frame = np.roll(base, shift=i * 3, axis=1)
                Image.fromarray(frame).save(os.path.join(img_dir, f"{i:03d}.png"))
                depth = np.full((64, 80), 2.0 + 0.01 * i, dtype=np.float32)
                np.savez(os.path.join(dpt_dir, f"depth_{i:03d}.npz"), pred=depth)

            ds = SequenceDataset.from_folder(tmp, width=80, height=64)
            depth_paths = [
                os.path.join(dpt_dir, f"depth_{i:03d}.npz") for i in range(4)
            ]
            cfg = HybridSeedConfig(
                sequential_radius=1,
                pair_radius=1,
                long_range_stride=0,
                resize_width=80,
                pose_graph=False,
                feature_backend="sift",
                min_matches=8,
                min_inliers=5,
                max_features=500,
            )
            seeder = HybridPoseSeeder(ds, depth_paths, cfg)
            poses = seeder.solve()
            self.assertEqual(len(poses.T_world_cam), 4)
            np.testing.assert_allclose(poses.T_world_cam[0], eye4(), atol=1e-12)
            self.assertGreaterEqual(len(seeder.sequential_edges), 3)

class LargeJumpSubsampleTest(unittest.TestCase):
    def test_subsample_indices_stride(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "eval_large_jump_stress",
            os.path.join(ROOT, "scripts", "eval_large_jump_stress.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        idx = mod.subsample_indices(40, stride=5, max_keep=0)
        self.assertEqual(idx, list(range(0, 40, 5)))
        capped = mod.subsample_indices(150, stride=8, max_keep=10)
        self.assertEqual(len(capped), 10)
        self.assertEqual(capped[0], 0)
        self.assertEqual(capped[1], 8)



if __name__ == "__main__":
    unittest.main()
