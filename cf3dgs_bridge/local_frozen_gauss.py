"""Local frozen-Gaussian photometric SE(3) solver (CF-3DGS paper algorithm).

For each adjacent pair (t, t+1):
  1. Load DPT depth for frame t; lift to a point cloud with K + identity pose.
  2. Downsample → initialize local isotropic Gaussians (means, RGB, scales,
     opacity, identity quat).
  3. Fit Gaussians to frame t with photometric L1+SSIM (pose fixed at I).
  4. Freeze all Gaussian attributes.
  5. Optimize only SE(3) T (quaternion + translation) so that transforming
     Gaussians by T and rendering matches frame t+1.
  6. Relative pose T is ``T_dst_src`` (maps points in t into t+1), same as
     ``PoseSequence``.

Re-implements the *technique* from Fu et al. (CF-3DGS); does **not** copy
NVlabs source. Rasterization uses ``soft_splat`` on CPU (or the FastGS CUDA
hook when available).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .dataset import SequenceDataset
from .depth_pnp_solver import load_depth_npz, resize_depth
from .pose_solver import PoseSequence, eye4
from .soft_splat import (
    photometric_loss,
    quat_normalize,
    se3_from_quat_trans,
    try_fastgs_rasterizer,
)


@dataclass
class LocalFrozenGaussConfig:
    fit_iters: int = 100
    pose_iters: int = 80
    lr_gauss: float = 2e-2
    lr_pose: float = 1e-2
    downsample: int = 6
    resize_width: int = 320
    device: str = "cpu"
    lambda_dssim: float = 0.2
    min_depth: float = 1e-3
    max_depth: float = 80.0
    depth_key: str = "pred"
    bg_color: float = 0.0
    warm_start_depth_pnp: bool = True
    isotropic: bool = True
    splat_mode: str = "soft"  # soft | sorted
    max_gaussians: int = 2500
    opacity_init: float = 0.7
    scale_mult: float = 0.6  # neighbor-spacing → Gaussian scale
    log_every: int = 50
    pose_max_trans_ratio: float = 2.5  # reject if ||t|| > ratio * ||t_warm||
    pose_max_rot_rad: float = 0.15  # ~8.6 deg from warm start
    keep_best_pose: bool = True


def _load_rgb(path: str, resize_width: Optional[int]) -> Tuple[np.ndarray, float]:
    """Load BGR→RGB float [0,1], optional width resize. Returns (H,W,3), scale."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = 1.0
    if resize_width is not None and rgb.shape[1] > resize_width:
        scale = resize_width / float(rgb.shape[1])
        nh = max(1, int(round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, (resize_width, nh), interpolation=cv2.INTER_AREA)
    return (rgb.astype(np.float32) / 255.0), scale


def _scaled_K(K: np.ndarray, scale: float) -> np.ndarray:
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] *= scale
    K2[1, 2] *= scale
    return K2


def depth_to_local_gaussians(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    *,
    stride: int,
    min_depth: float,
    max_depth: float,
    max_gaussians: int,
    opacity_init: float,
    scale_mult: float,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Lift depth → downsampled local Gaussians in the camera frame.

    Returns means, colors, log_scales (N,3), opacity_logit (N,), quats (N,4).
    """
    h, w = depth.shape
    assert rgb.shape[0] == h and rgb.shape[1] == w

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    xs_f = grid_x.ravel().astype(np.float64)
    ys_f = grid_y.ravel().astype(np.float64)
    xi = np.clip(np.rint(xs_f).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys_f).astype(np.int32), 0, h - 1)
    z = depth[yi, xi].astype(np.float64)
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    xs_f, ys_f, z, xi, yi = xs_f[valid], ys_f[valid], z[valid], xi[valid], yi[valid]
    if xs_f.size == 0:
        raise RuntimeError("No valid depth samples for local Gaussians")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs_f - cx) / fx * z
    Y = (ys_f - cy) / fy * z
    means_np = np.stack([X, Y, z], axis=1)
    colors_np = rgb[yi, xi].astype(np.float64)

    # Neighbor spacing → isotropic scale (median NN distance among samples).
    if means_np.shape[0] > 1:
        # Cheap estimate: stride * z / f in 3D
        pix_spacing = float(stride)
        scales_np = scale_mult * (pix_spacing * z / max(fx, 1e-6))
    else:
        scales_np = np.array([0.05], dtype=np.float64)

    # Subsample if too many
    n = means_np.shape[0]
    if n > max_gaussians:
        idx = np.linspace(0, n - 1, max_gaussians).astype(np.int64)
        means_np = means_np[idx]
        colors_np = colors_np[idx]
        scales_np = scales_np[idx]
        n = max_gaussians

    means = torch.tensor(means_np, device=device, dtype=torch.float32)
    colors = torch.tensor(colors_np, device=device, dtype=torch.float32)
    log_scales = torch.log(
        torch.tensor(scales_np, device=device, dtype=torch.float32).clamp_min(1e-6)
    )
    if log_scales.ndim == 1:
        log_scales = log_scales.unsqueeze(-1).expand(-1, 3).clone()
    # opacity via logit
    o = float(np.clip(opacity_init, 1e-3, 1 - 1e-3))
    opacity_logit = torch.full(
        (n,), float(np.log(o / (1 - o))), device=device, dtype=torch.float32
    )
    quats = torch.zeros(n, 4, device=device, dtype=torch.float32)
    quats[:, 0] = 1.0  # identity
    return means, colors, log_scales, opacity_logit, quats


class _LocalGaussParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        log_scales: torch.Tensor,
        opacity_logit: torch.Tensor,
        quats: torch.Tensor,
    ):
        super().__init__()
        self.means = nn.Parameter(means.clone())
        self.colors = nn.Parameter(colors.clone())
        self.log_scales = nn.Parameter(log_scales.clone())
        self.opacity_logit = nn.Parameter(opacity_logit.clone())
        self.quats = nn.Parameter(quats.clone())

    def packed(self):
        scales = torch.exp(self.log_scales).clamp_min(1e-6)
        opacities = torch.sigmoid(self.opacity_logit)
        colors = self.colors.clamp(0.0, 1.0)
        quats = quat_normalize(self.quats)
        return self.means, colors, opacities, scales, quats


class _PoseParams(nn.Module):
    """Learnable SE(3) as quaternion + translation (T_dst_src)."""

    def __init__(self, T_init: np.ndarray, device: torch.device):
        super().__init__()
        T = np.asarray(T_init, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        # matrix → quat (wxyz)
        q = _rotmat_to_quat(R)
        self.quat = nn.Parameter(torch.tensor(q, device=device, dtype=torch.float32))
        self.trans = nn.Parameter(torch.tensor(t, device=device, dtype=torch.float32))

    def matrix(self) -> torch.Tensor:
        return se3_from_quat_trans(self.quat, self.trans)

    def numpy(self) -> np.ndarray:
        with torch.no_grad():
            return self.matrix().detach().cpu().numpy().astype(np.float64)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


class LocalFrozenGaussSolver:
    """Progressive local-frozen-Gaussian photometric VO (paper method)."""

    def __init__(
        self,
        dataset: SequenceDataset,
        depth_paths: List[Optional[str]],
        cfg: Optional[LocalFrozenGaussConfig] = None,
    ):
        self.dataset = dataset
        self.depth_paths = depth_paths
        self.cfg = cfg or LocalFrozenGaussConfig()
        if len(depth_paths) != len(dataset):
            raise ValueError(
                f"depth_paths length {len(depth_paths)} != num frames {len(dataset)}"
            )
        self.poses = PoseSequence(num_frames=len(dataset))
        self.device = torch.device(self.cfg.device)
        self.backend_name = "local_frozen_softsplat"

    def solve(self) -> PoseSequence:
        n = len(self.dataset)
        if n < 2:
            raise ValueError("Need >= 2 frames")
        missing = sum(1 for p in self.depth_paths if p is None)
        if missing:
            print(f"[local_frozen] WARNING: {missing}/{n} frames missing depth")

        print(
            f"[local_frozen] Progressive photometric SE3 on {n} frames "
            f"(fit={self.cfg.fit_iters}, pose={self.cfg.pose_iters}, "
            f"width={self.cfg.resize_width}, stride={self.cfg.downsample}, "
            f"device={self.cfg.device}, warm_pnp={self.cfg.warm_start_depth_pnp})"
        )
        self.poses.set_identity_anchor(0)

        # Optional depth_pnp warm starts for all edges
        warm = self._warm_start_relatives() if self.cfg.warm_start_depth_pnp else {}

        for i in range(1, n):
            T_init = warm.get((i - 1, i), eye4())
            T_rel = self.estimate_relative(i - 1, i, T_init=T_init)
            self.poses.set_relative(i - 1, i, T_rel)
            tnorm = float(np.linalg.norm(T_rel[:3, 3]))
            print(f"[local_frozen] {i-1:03d}→{i:03d} t_norm={tnorm:.4f}")

        self.poses.compose_forward()
        return self.poses

    def estimate_relative(
        self,
        prev_idx: int,
        curr_idx: int,
        T_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """CF-3DGS local fit → freeze → SE(3) for one edge."""
        cfg = self.cfg
        device = self.device
        rgb0, scale = _load_rgb(self.dataset.image_paths[prev_idx], cfg.resize_width)
        rgb1, scale1 = _load_rgb(self.dataset.image_paths[curr_idx], cfg.resize_width)
        # Use same scale (both resized to same width)
        assert rgb0.shape == rgb1.shape, (rgb0.shape, rgb1.shape)
        h, w = rgb0.shape[:2]
        K_full = self.dataset.intrinsics.as_K()
        K = _scaled_K(K_full, scale)

        dpath = self.depth_paths[prev_idx]
        if dpath is None:
            print(f"[local_frozen] missing depth for frame {prev_idx}; identity")
            return eye4() if T_init is None else np.asarray(T_init, dtype=np.float64)

        depth = load_depth_npz(dpath, key=cfg.depth_key)
        depth = resize_depth(depth, w, h)

        means, colors, log_scales, opacity_logit, quats = depth_to_local_gaussians(
            depth,
            rgb0,
            K,
            stride=cfg.downsample,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            max_gaussians=cfg.max_gaussians,
            opacity_init=cfg.opacity_init,
            scale_mult=cfg.scale_mult,
            device=device,
        )
        gauss = _LocalGaussParams(means, colors, log_scales, opacity_logit, quats)

        target0 = torch.tensor(rgb0.transpose(2, 0, 1), device=device, dtype=torch.float32)
        target1 = torch.tensor(rgb1.transpose(2, 0, 1), device=device, dtype=torch.float32)
        K_t = torch.tensor(K, device=device, dtype=torch.float32)
        I = torch.eye(4, device=device, dtype=torch.float32)

        # --- Step 3: fit local Gaussians to frame t (pose fixed) ---
        opt_g = torch.optim.Adam(
            [
                {"params": [gauss.means], "lr": cfg.lr_gauss * 0.1},
                {"params": [gauss.colors], "lr": cfg.lr_gauss},
                {"params": [gauss.log_scales], "lr": cfg.lr_gauss * 0.5},
                {"params": [gauss.opacity_logit], "lr": cfg.lr_gauss},
                {"params": [gauss.quats], "lr": cfg.lr_gauss * 0.1},
            ]
        )
        for it in range(cfg.fit_iters):
            opt_g.zero_grad()
            m, c, o, s, q = gauss.packed()
            render = try_fastgs_rasterizer(
                m, c, o, s, q, K_t, I, h, w,
                bg_color=cfg.bg_color,
                isotropic=cfg.isotropic,
                mode=cfg.splat_mode,
            )
            loss = photometric_loss(render, target0, lambda_dssim=cfg.lambda_dssim)
            loss.backward()
            opt_g.step()
            if cfg.log_every and (it % cfg.log_every == 0 or it == cfg.fit_iters - 1):
                print(
                    f"  [fit {prev_idx}] iter {it:03d}/{cfg.fit_iters} "
                    f"loss={float(loss.detach()):.4f} N={m.shape[0]}"
                )

        # --- Step 4: freeze ---
        for p in gauss.parameters():
            p.requires_grad_(False)

        # --- Step 5: optimize SE(3) only ---
        T0 = eye4() if T_init is None else np.asarray(T_init, dtype=np.float64)
        pose = _PoseParams(T0, device)
        opt_p = torch.optim.Adam(
            [
                {"params": [pose.quat], "lr": cfg.lr_pose},
                {"params": [pose.trans], "lr": cfg.lr_pose},
            ]
        )
        with torch.no_grad():
            m, c, o, s, q = gauss.packed()
        m = m.detach()
        c = c.detach()
        o = o.detach()
        s = s.detach()
        q = q.detach()

        t_warm = float(np.linalg.norm(T0[:3, 3])) + 1e-9
        best_loss = float("inf")
        best_T = T0.copy()
        with torch.no_grad():
            T = pose.matrix()
            render0 = try_fastgs_rasterizer(
                m, c, o, s, q, K_t, T, h, w,
                bg_color=cfg.bg_color,
                isotropic=cfg.isotropic,
                mode=cfg.splat_mode,
            )
            best_loss = float(photometric_loss(render0, target1, lambda_dssim=cfg.lambda_dssim).detach())
            best_T = pose.numpy()

        for it in range(cfg.pose_iters):
            opt_p.zero_grad()
            T = pose.matrix()
            render = try_fastgs_rasterizer(
                m, c, o, s, q, K_t, T, h, w,
                bg_color=cfg.bg_color,
                isotropic=cfg.isotropic,
                mode=cfg.splat_mode,
            )
            loss = photometric_loss(render, target1, lambda_dssim=cfg.lambda_dssim)
            loss.backward()
            opt_p.step()
            with torch.no_grad():
                pose.quat.copy_(quat_normalize(pose.quat))
                # Clamp translation away from pathological blow-ups
                tnorm = float(pose.trans.norm().item())
                max_t = cfg.pose_max_trans_ratio * t_warm
                if tnorm > max_t and tnorm > 1e-6:
                    pose.trans.mul_(max_t / tnorm)
            loss_f = float(loss.detach())
            if cfg.keep_best_pose and loss_f < best_loss:
                # Also require not too far from warm start in rotation
                T_cur = pose.numpy()
                R_rel = T_cur[:3, :3] @ T0[:3, :3].T
                cos_th = float(np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0))
                ang = float(np.arccos(cos_th))
                if ang <= cfg.pose_max_rot_rad:
                    best_loss = loss_f
                    best_T = T_cur
            if cfg.log_every and (it % cfg.log_every == 0 or it == cfg.pose_iters - 1):
                print(
                    f"  [pose {prev_idx}→{curr_idx}] iter {it:03d}/{cfg.pose_iters} "
                    f"loss={loss_f:.4f} best={best_loss:.4f}"
                )

        if cfg.keep_best_pose:
            return best_T
        return pose.numpy()

    def _warm_start_relatives(self) -> dict:
        """Run depth_pnp on the same sequence for SE3 warm starts."""
        try:
            from .depth_pnp_solver import DepthPnPProgressiveSolver, DepthPnPSolveConfig

            cfg = DepthPnPSolveConfig(
                resize_width=min(960, max(self.cfg.resize_width, 640)),
                feature_backend="sift",
                pose_graph=False,
                photo_refine=False,
            )
            print("[local_frozen] computing depth_pnp warm-start relatives…")
            solver = DepthPnPProgressiveSolver(self.dataset, self.depth_paths, cfg)
            poses = solver.solve()
            warm = {}
            for (a, b), rel in poses.relatives.items():
                warm[(a, b)] = rel.T_dst_src.copy()
            return warm
        except Exception as exc:
            print(f"[local_frozen] warm-start depth_pnp failed ({exc}); using identity")
            return {}


__all__ = [
    "LocalFrozenGaussConfig",
    "LocalFrozenGaussSolver",
    "depth_to_local_gaussians",
]
