"""Pose-graph optimization over SE(3) relative edges (CPU, scipy).

Poses are camera-to-world 4x4 matrices matching ``PoseSequence.T_world_cam``
(compose_forward convention). Relative edge measurements use the same
``T_dst_src`` convention as ``RelativePoseSE3``: maps points in src camera
frame into dst camera frame (i.e. OpenCV solvePnP output for dst←src).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .pose_solver import eye4


def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
        dtype=np.float64,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: so(3) vector → SO(3)."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + skew(w)
    k = w / theta
    K = skew(k)
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) → so(3) rotation vector."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
    if abs(theta - np.pi) < 1e-6:
        xx = max(0.0, (R[0, 0] + 1.0) * 0.5)
        yy = max(0.0, (R[1, 1] + 1.0) * 0.5)
        zz = max(0.0, (R[2, 2] + 1.0) * 0.5)
        x = float(np.sqrt(xx))
        y = float(np.sqrt(yy)) * (1.0 if R[0, 1] >= 0 else -1.0)
        z = float(np.sqrt(zz)) * (1.0 if R[0, 2] >= 0 else -1.0)
        w = np.array([x, y, z], dtype=np.float64)
        n = float(np.linalg.norm(w))
        if n < 1e-10:
            return np.array([theta, 0.0, 0.0], dtype=np.float64)
        return w * (theta / n)
    w = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    return w * (theta / (2.0 * np.sin(theta)))


def _left_jacobian_SO3(w: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3) used in SE(3) exp/log translation coupling."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + 0.5 * skew(w)
    K = skew(w)
    theta2 = theta * theta
    theta3 = theta2 * theta
    a = (1.0 - np.cos(theta)) / theta2
    b = (theta - np.sin(theta)) / theta3
    return np.eye(3) + a * K + b * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) 6-vector [wx,wy,wz, tx,ty,tz] → SE(3) 4x4.

    ``xi[3:]`` is ρ such that t = V(ω) @ ρ (standard SE(3) exponential).
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w, rho = xi[:3], xi[3:]
    R = so3_exp(w)
    V = _left_jacobian_SO3(w)
    t = V @ rho
    T = eye4()
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) 4x4 → se(3) 6-vector [wx,wy,wz, tx,ty,tz]."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    V = _left_jacobian_SO3(w)
    try:
        rho = np.linalg.solve(V, t)
    except np.linalg.LinAlgError:
        rho = np.linalg.lstsq(V, t, rcond=None)[0]
    return np.concatenate([w, rho])


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = eye4()
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


@dataclass
class PoseGraphEdge:
    """Relative measurement between cameras i (src) and j (dst)."""

    i: int
    j: int
    T_j_i: np.ndarray  # T_dst_src: maps points in cam i → cam j
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.T_j_i = np.asarray(self.T_j_i, dtype=np.float64)
        assert self.T_j_i.shape == (4, 4)


def compose_chain_c2w(relatives_adj: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Chain adjacent T_{k+1←k} into c2w poses (frame 0 = I)."""
    n = len(relatives_adj) + 1
    poses = [eye4() for _ in range(n)]
    for i, T_ip1_i in enumerate(relatives_adj):
        poses[i + 1] = poses[i] @ invert_T(T_ip1_i)
    return poses


def _pack_poses(poses: Sequence[np.ndarray]) -> np.ndarray:
    """Pack frames 1..n-1 as se3 logs of c2w (frame 0 fixed at identity)."""
    xs = [se3_log(poses[i]) for i in range(1, len(poses))]
    return np.concatenate(xs) if xs else np.zeros(0, dtype=np.float64)


def _unpack_poses(x: np.ndarray, n: int) -> List[np.ndarray]:
    poses = [eye4()]
    for k in range(n - 1):
        poses.append(se3_exp(x[6 * k : 6 * (k + 1)]))
    return poses


def relative_residual_c2w(
    T_i: np.ndarray, T_j: np.ndarray, T_j_i_meas: np.ndarray
) -> np.ndarray:
    """6-vector residual for edge measuring T_j←i given c2w poses T_i, T_j.

    Predicted relative (points i → j):  T_pred = inv(T_j) @ T_i
    Error in SE(3):  T_err = inv(T_meas) @ T_pred
    """
    T_pred = invert_T(T_j) @ T_i
    T_err = invert_T(T_j_i_meas) @ T_pred
    return se3_log(T_err)


def pose_graph_cost(
    poses: Sequence[np.ndarray],
    edges: Sequence[PoseGraphEdge],
) -> float:
    """Sum of squared weighted residuals (for diagnostics)."""
    total = 0.0
    for e in edges:
        r = relative_residual_c2w(poses[e.i], poses[e.j], e.T_j_i)
        total += float(e.weight) * float(np.dot(r, r))
    return total


def optimize_pose_graph(
    init_c2w: Sequence[np.ndarray],
    edges: Sequence[PoseGraphEdge],
    *,
    max_nfev: int = 80,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    verbose: int = 0,
) -> List[np.ndarray]:
    """Optimize c2w poses with frame 0 fixed; return refined c2w list."""
    n = len(init_c2w)
    if n < 2 or not edges:
        return [np.asarray(T, dtype=np.float64).copy() for T in init_c2w]

    x0 = _pack_poses(init_c2w)
    edge_list = list(edges)
    sqrt_w = np.array(
        [np.sqrt(max(e.weight, 1e-12)) for e in edge_list], dtype=np.float64
    )

    def fun(x: np.ndarray) -> np.ndarray:
        poses = _unpack_poses(x, n)
        residuals = []
        for e, sw in zip(edge_list, sqrt_w):
            r = relative_residual_c2w(poses[e.i], poses[e.j], e.T_j_i)
            residuals.append(sw * r)
        return np.concatenate(residuals)

    # LM requires n residuals >= n vars; trf is safer for underdetermined cases
    n_res = 6 * len(edge_list)
    n_var = x0.size
    method = "lm" if n_res >= n_var else "trf"
    res = least_squares(
        fun,
        x0,
        method=method,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        verbose=verbose,
    )
    return _unpack_poses(res.x, n)


def build_edges_from_measurements(
    measurements: Iterable[Tuple[int, int, np.ndarray, float]],
) -> List[PoseGraphEdge]:
    """measurements: (i, j, T_j_i, weight)."""
    return [
        PoseGraphEdge(
            i=int(i), j=int(j), T_j_i=np.asarray(T, dtype=np.float64), weight=float(w)
        )
        for i, j, T, w in measurements
    ]


def compose_relatives(relatives: Sequence[np.ndarray]) -> np.ndarray:
    """Compose T_{i+1←i} ... into T_{j←i} (list length j-i)."""
    T = eye4()
    for Tk in relatives:
        T = np.asarray(Tk, dtype=np.float64) @ T
    return T


def optimize_edge_scales(
    adj_rels: Sequence[np.ndarray],
    skip_meas: Sequence[Tuple[int, int, np.ndarray]],
    *,
    max_nfev: int = 60,
) -> List[np.ndarray]:
    """Rescale adjacent-edge translations so composed chains match skip PnP.

    Both adjacent edge i→i+1 and skip i→j unproject with depth at the *source*
    of their own PnP; skips that share source frame i provide a metric check on
    the mixed-scale chain through intermediate frames. We optimize a positive
    scale multiplier per adjacent edge (log-parameterized).

    Parameters
    ----------
    adj_rels : list of T_{k+1←k}
    skip_meas : list of (i, j, T_j_i_meas) from PnP using depth at i
    """
    n_e = len(adj_rels)
    if n_e == 0 or not skip_meas:
        return [np.asarray(T, dtype=np.float64).copy() for T in adj_rels]

    # Only translation magnitudes / SE3 log vs skips
    skips = [(int(i), int(j), np.asarray(T, dtype=np.float64)) for i, j, T in skip_meas]

    def apply_scales(log_s: np.ndarray) -> List[np.ndarray]:
        s = np.exp(log_s)
        out = []
        for k, T0 in enumerate(adj_rels):
            T = np.asarray(T0, dtype=np.float64).copy()
            T[:3, 3] = T[:3, 3] * float(s[k])
            out.append(T)
        return out

    def fun(log_s: np.ndarray) -> np.ndarray:
        scaled = apply_scales(log_s)
        res = []
        for i, j, T_meas in skips:
            T_comp = compose_relatives(scaled[i:j])
            # SE3 residual (full 6) — primarily catches scale + mild rot drift
            T_err = invert_T(T_meas) @ T_comp
            r = se3_log(T_err)
            # Emphasize translation (last 3)
            r = np.concatenate([0.25 * r[:3], r[3:]])
            res.append(r)
        # mild prior scales ~ 1
        res.append(0.05 * log_s)
        return np.concatenate(res)

    x0 = np.zeros(n_e, dtype=np.float64)
    n_res = 6 * len(skips) + n_e
    method = "lm" if n_res >= n_e else "trf"
    try:
        result = least_squares(fun, x0, method=method, max_nfev=max_nfev, ftol=1e-8, xtol=1e-8)
        scaled = apply_scales(result.x)
        print(
            f"[pose_graph] edge-scale opt: cost {float(np.sum(fun(x0)**2)):.4f} → "
            f"{float(np.sum(fun(result.x)**2)):.4f}, "
            f"log_s std={float(np.std(result.x)):.4f}"
        )
        return scaled
    except Exception as exc:
        print(f"[pose_graph] edge-scale opt failed ({exc}); keeping original scales")
        return [np.asarray(T, dtype=np.float64).copy() for T in adj_rels]


def filter_edges_by_quality(
    edges: Sequence[PoseGraphEdge],
    *,
    min_weight: float = 1e-6,
) -> List[PoseGraphEdge]:
    """Drop near-zero-weight edges (after upstream quality gating).

    Prefer ``cf3dgs_bridge.edge_filter.evaluate_skip_edge`` / ``filter_skip_measurements``
    for inlier / depth-ratio / reproj gates *before* constructing edges. This helper
    only removes edges already marked with tiny weights.
    """
    return [e for e in edges if float(e.weight) >= min_weight]
