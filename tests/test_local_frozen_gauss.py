"""Synthetic two-view test: soft_splat + frozen pose optimize recovers small SE3."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cf3dgs_bridge.local_frozen_gauss import _PoseParams, _rotmat_to_quat
from cf3dgs_bridge.soft_splat import (
    photometric_loss,
    quat_normalize,
    se3_from_quat_trans,
    soft_splat,
    try_fastgs_rasterizer,
)


def _make_scene(n: int = 80, H: int = 64, W: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Random points in front of camera
    xy = rng.uniform(-0.4, 0.4, size=(n, 2)).astype(np.float32)
    z = rng.uniform(1.5, 3.5, size=(n, 1)).astype(np.float32)
    means = np.concatenate([xy * z, z], axis=1)
    colors = rng.uniform(0.2, 0.9, size=(n, 3)).astype(np.float32)
    scales = np.full((n, 3), 0.04, dtype=np.float32)
    opacities = np.full((n,), 0.85, dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    fx = fy = 80.0
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=np.float32)
    return means, colors, scales, opacities, quats, K, H, W


def _small_se3(angle_deg: float = 3.0, trans: float = 0.05) -> np.ndarray:
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([trans, 0.01, 0.02], dtype=np.float64)
    return T


class SoftSplatPoseTest(unittest.TestCase):
    def test_soft_splat_renders_finite(self):
        means, colors, scales, opacities, quats, K, H, W = _make_scene()
        device = torch.device("cpu")
        img = soft_splat(
            torch.tensor(means, device=device),
            torch.tensor(colors, device=device),
            torch.tensor(opacities, device=device),
            torch.tensor(scales, device=device),
            torch.tensor(quats, device=device),
            torch.tensor(K, device=device),
            torch.eye(4, device=device),
            H,
            W,
            isotropic=True,
            mode="soft",
        )
        self.assertEqual(tuple(img.shape), (3, H, W))
        self.assertTrue(torch.isfinite(img).all())
        self.assertGreater(float(img.mean()), 0.01)

    def test_frozen_pose_recovers_small_se3(self):
        """Render target with known T_gt; optimize pose from identity / noisy init."""
        means_np, colors_np, scales_np, opac_np, quats_np, K_np, H, W = _make_scene(
            n=100, H=48, W=64, seed=1
        )
        T_gt = _small_se3(angle_deg=4.0, trans=0.08)
        device = torch.device("cpu")
        means = torch.tensor(means_np, device=device)
        colors = torch.tensor(colors_np, device=device)
        scales = torch.tensor(scales_np, device=device)
        opac = torch.tensor(opac_np, device=device)
        quats = torch.tensor(quats_np, device=device)
        K = torch.tensor(K_np, device=device)
        T_gt_t = torch.tensor(T_gt, device=device, dtype=torch.float32)

        with torch.no_grad():
            target = try_fastgs_rasterizer(
                means, colors, opac, scales, quats, K, T_gt_t, H, W, isotropic=True
            )

        # Warm init: mildly perturbed GT (simulates depth_pnp warm start)
        T_init = T_gt.copy()
        T_init[:3, 3] = T_gt[:3, 3] * 0.85 + np.array([0.01, 0.0, 0.0])
        dR = _small_se3(angle_deg=1.5, trans=0.0)
        T_init = dR @ T_init

        pose = _PoseParams(T_init, device)
        opt = torch.optim.Adam(
            [{"params": [pose.quat], "lr": 0.02}, {"params": [pose.trans], "lr": 0.02}]
        )
        t_err0 = np.linalg.norm(T_init[:3, 3] - T_gt[:3, 3])
        for _ in range(150):
            opt.zero_grad()
            T = pose.matrix()
            render = try_fastgs_rasterizer(
                means, colors, opac, scales, quats, K, T, H, W, isotropic=True
            )
            loss = photometric_loss(render, target, lambda_dssim=0.2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pose.quat.copy_(quat_normalize(pose.quat))

        T_est = pose.numpy()
        t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        R_err = T_est[:3, :3] @ T_gt[:3, :3].T
        cos_th = np.clip(0.5 * (np.trace(R_err) - 1.0), -1.0, 1.0)
        ang_err = np.degrees(np.arccos(cos_th))
        self.assertLess(t_err, max(0.05, 0.5 * t_err0), msg=f"t_err={t_err} (init {t_err0})")
        self.assertLess(ang_err, 2.5, msg=f"ang_err={ang_err}")

    def test_try_fastgs_falls_back(self):
        means, colors, scales, opacities, quats, K, H, W = _make_scene(n=20, H=32, W=40)
        device = torch.device("cpu")
        img = try_fastgs_rasterizer(
            torch.tensor(means, device=device),
            torch.tensor(colors, device=device),
            torch.tensor(opacities, device=device),
            torch.tensor(scales, device=device),
            torch.tensor(quats, device=device),
            torch.tensor(K, device=device),
            torch.eye(4, device=device),
            H,
            W,
        )
        self.assertEqual(tuple(img.shape), (3, H, W))


if __name__ == "__main__":
    unittest.main()
