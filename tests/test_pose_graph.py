"""Unit tests for SE(3) exp/log and pose-graph residual reduction."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cf3dgs_bridge.pose_graph import (
    PoseGraphEdge,
    compose_chain_c2w,
    invert_T,
    optimize_pose_graph,
    pose_graph_cost,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)
from cf3dgs_bridge.pose_solver import eye4


class SE3RoundtripTest(unittest.TestCase):
    def test_so3_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = rng.normal(size=3) * 0.5
            R = so3_exp(w)
            w2 = so3_log(R)
            R2 = so3_exp(w2)
            np.testing.assert_allclose(R2, R, atol=1e-8)

    def test_se3_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            xi = rng.normal(size=6)
            xi[:3] *= 0.4
            xi[3:] *= 0.2
            T = se3_exp(xi)
            xi2 = se3_log(T)
            T2 = se3_exp(xi2)
            np.testing.assert_allclose(T2, T, atol=1e-7)

    def test_se3_identity(self):
        T = se3_exp(np.zeros(6))
        np.testing.assert_allclose(T, eye4(), atol=1e-12)
        np.testing.assert_allclose(se3_log(eye4()), np.zeros(6), atol=1e-12)


class PoseGraphSyntheticTest(unittest.TestCase):
    def test_pose_graph_reduces_residual(self):
        """Noisy chain + skip edges → optimizer lowers cost toward GT relatives."""
        rng = np.random.default_rng(42)
        n = 12
        # Ground-truth adjacent relatives: small forward motion + tiny yaw
        gt_adj = []
        for i in range(n - 1):
            xi = np.array([0.0, 0.02, 0.0, 0.05, 0.0, 0.0], dtype=np.float64)
            gt_adj.append(se3_exp(xi))
        gt_c2w = compose_chain_c2w(gt_adj)

        def rel_from_poses(i, j):
            return invert_T(gt_c2w[j]) @ gt_c2w[i]

        edges = []
        # Adjacent with noise
        noisy_adj = []
        for i in range(n - 1):
            T = rel_from_poses(i, i + 1)
            noise = se3_exp(rng.normal(size=6) * np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
            T_n = noise @ T
            noisy_adj.append(T_n)
            edges.append(PoseGraphEdge(i=i, j=i + 1, T_j_i=T_n, weight=1.0))
        # Skip edges with less noise (simulate better long-range)
        for k in (2, 5):
            for i in range(0, n - k):
                T = rel_from_poses(i, i + k)
                noise = se3_exp(rng.normal(size=6) * 0.003)
                edges.append(PoseGraphEdge(i=i, j=i + k, T_j_i=noise @ T, weight=0.4))

        init = compose_chain_c2w(noisy_adj)
        cost0 = pose_graph_cost(init, edges)
        refined = optimize_pose_graph(init, edges, max_nfev=60)
        cost1 = pose_graph_cost(refined, edges)
        self.assertLess(cost1, cost0 * 0.5)

        # Trajectory should move closer to GT centers
        def centers(poses):
            return np.stack([T[:3, 3] for T in poses], axis=0)

        err0 = np.linalg.norm(centers(init) - centers(gt_c2w), axis=1).mean()
        err1 = np.linalg.norm(centers(refined) - centers(gt_c2w), axis=1).mean()
        self.assertLess(err1, err0)


if __name__ == "__main__":
    unittest.main()
