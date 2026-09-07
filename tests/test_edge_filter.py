"""Unit tests for skip-edge quality filter."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cf3dgs_bridge.edge_filter import (
    EdgeQualityConfig,
    evaluate_skip_edge,
    median_depth_ratio,
)
from cf3dgs_bridge.pose_solver import eye4


class EdgeFilterTest(unittest.TestCase):
    def test_reject_low_inliers(self):
        T = eye4()
        T[:3, 3] = [0.1, 0, 0]
        q = evaluate_skip_edge(T, 5, median_depth_ratio_val=1.0, median_reproj_val=1.0)
        self.assertFalse(q.accepted)

    def test_reject_bad_depth_ratio(self):
        T = eye4()
        T[:3, 3] = [0.1, 0, 0]
        T_chain = T.copy()
        q = evaluate_skip_edge(
            T,
            40,
            median_depth_ratio_val=0.07,
            median_reproj_val=1.0,
            T_chain=T_chain,
        )
        self.assertFalse(q.accepted)
        self.assertIn("depth_ratio", q.reason)

    def test_accept_good_edge(self):
        T = eye4()
        T[:3, 3] = [0.1, 0, 0]
        T_chain = T.copy()
        q = evaluate_skip_edge(
            T,
            40,
            median_depth_ratio_val=1.05,
            median_reproj_val=1.2,
            T_chain=T_chain,
            cfg=EdgeQualityConfig(min_inliers=20),
        )
        self.assertTrue(q.accepted)
        self.assertGreater(q.weight, 0)

    def test_median_depth_ratio(self):
        h, w = 32, 32
        d1 = np.ones((h, w), dtype=np.float32) * 2.0
        d2 = np.ones((h, w), dtype=np.float32) * 4.0
        rng = np.random.default_rng(0)
        pts = rng.uniform(2, 30, size=(12, 2))
        r = median_depth_ratio(pts, pts, d1, d2)
        self.assertAlmostEqual(r, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
