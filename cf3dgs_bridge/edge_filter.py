"""Robust skip / long-range edge quality filter for depth-PnP pose graphs.

Motivation
----------
Skip-k PnP (esp. k=10) often returns translation scales ~0.07× the adjacent
chain because DPT depth is per-frame and scale-inconsistent. Feeding those
edges into a pose graph *increases* ATE. This module centralizes acceptance
tests used before edges enter ``pose_graph.optimize_*``:

* minimum RANSAC inlier count
* median pairwise depth ratio (depth_j / depth_i on matches) in [lo, hi]
* median reprojection error of the PnP solution (px)
* optional ||t_meas|| / ||t_chain|| scale ratio in [lo, hi]
* optional rotation disagreement vs adjacent chain

Usage
-----
::

    from cf3dgs_bridge.edge_filter import EdgeQualityConfig, evaluate_skip_edge

    cfg = EdgeQualityConfig()
    q = evaluate_skip_edge(
        T_meas, n_inliers, median_depth_ratio, median_reproj,
        T_chain=T_chain, cfg=cfg,
    )
    if q.accepted:
        edges.append(PoseGraphEdge(i=i, j=j, T_j_i=T_meas, weight=q.weight))

Also see ``filter_skip_measurements`` for batch filtering of (i, j, T, meta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pose_graph import so3_log
from .pose_solver import eye4


@dataclass
class EdgeQualityConfig:
    min_inliers: int = 30
    depth_ratio_lo: float = 0.5
    depth_ratio_hi: float = 2.0
    max_median_reproj: float = 3.0
    # Translation length vs adjacent-chain composition
    t_scale_ratio_lo: float = 0.5
    t_scale_ratio_hi: float = 2.0
    max_rot_err_rad: float = 0.25
    # Soft weight: base * exp(-(rot_err/rot_sigma)^2) * inlier_factor
    base_weight: float = 0.4
    rot_sigma: float = 0.1
    inlier_ref: float = 50.0
    require_depth_ratio: bool = True
    require_t_scale_ratio: bool = True
    require_reproj: bool = True


@dataclass
class EdgeQuality:
    accepted: bool
    reason: str
    n_inliers: int = 0
    median_depth_ratio: float = float("nan")
    median_reproj: float = float("nan")
    t_scale_ratio: float = float("nan")
    rot_err: float = float("nan")
    weight: float = 0.0


def rot_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log(R_a.T @ R_b)))


def sample_depths_at_pixels(
    pts_xy: np.ndarray,
    depth: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample depth at pixel locations; returns (z, valid_mask)."""
    h, w = depth.shape
    xs = pts_xy[:, 0]
    ys = pts_xy[:, 1]
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    z = depth[yi, xi].astype(np.float64)
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    return z, valid


def median_depth_ratio(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    depth_i: np.ndarray,
    depth_j: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> float:
    """Median of depth_j / depth_i over mutually valid matched pixels.

    A ratio far from 1 indicates inconsistent mono-depth scales between the
    two frames — a strong signal that skip PnP translation will be wrong even
    if RANSAC inliers look fine.
    """
    if depth_i is None or depth_j is None or len(pts_i) == 0:
        return float("nan")
    zi, vi = sample_depths_at_pixels(pts_i, depth_i, min_depth, max_depth)
    zj, vj = sample_depths_at_pixels(pts_j, depth_j, min_depth, max_depth)
    m = vi & vj
    if int(m.sum()) < 5:
        return float("nan")
    ratios = zj[m] / np.maximum(zi[m], 1e-12)
    return float(np.median(ratios))


def median_reproj_error(
    T_j_i: np.ndarray,
    pts3d_i: np.ndarray,
    pts2d_j: np.ndarray,
    K: np.ndarray,
    inlier_idx: Optional[np.ndarray] = None,
) -> float:
    """Median reprojection error (px) of object points through T_j_i."""
    if pts3d_i is None or len(pts3d_i) == 0:
        return float("nan")
    R = T_j_i[:3, :3]
    t = T_j_i[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    obj = pts3d_i.astype(np.float64)
    img = pts2d_j.astype(np.float64)
    if inlier_idx is not None and len(inlier_idx) > 0:
        idx = np.asarray(inlier_idx).reshape(-1)
        obj = obj[idx]
        img = img[idx]
    if len(obj) == 0:
        return float("nan")
    proj, _ = cv2.projectPoints(obj, rvec, t, K.astype(np.float64), None)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img, axis=1)
    return float(np.median(err))


def t_scale_ratio(T_meas: np.ndarray, T_chain: np.ndarray) -> float:
    nm = float(np.linalg.norm(T_meas[:3, 3]))
    nc = float(np.linalg.norm(T_chain[:3, 3]))
    if nc < 1e-9:
        return 0.0 if nm < 1e-9 else float("inf")
    return nm / nc


def evaluate_skip_edge(
    T_meas: np.ndarray,
    n_inliers: int,
    *,
    median_depth_ratio_val: float = float("nan"),
    median_reproj_val: float = float("nan"),
    T_chain: Optional[np.ndarray] = None,
    cfg: Optional[EdgeQualityConfig] = None,
) -> EdgeQuality:
    """Accept/reject a skip edge and assign a soft weight."""
    cfg = cfg or EdgeQualityConfig()
    q = EdgeQuality(
        accepted=False,
        reason="ok",
        n_inliers=int(n_inliers),
        median_depth_ratio=float(median_depth_ratio_val),
        median_reproj=float(median_reproj_val),
    )

    if n_inliers < cfg.min_inliers:
        q.reason = f"inliers<{cfg.min_inliers}"
        return q

    if cfg.require_depth_ratio and np.isfinite(median_depth_ratio_val):
        if (
            median_depth_ratio_val < cfg.depth_ratio_lo
            or median_depth_ratio_val > cfg.depth_ratio_hi
        ):
            q.reason = (
                f"depth_ratio={median_depth_ratio_val:.3f} "
                f"not in [{cfg.depth_ratio_lo},{cfg.depth_ratio_hi}]"
            )
            return q
    elif cfg.require_depth_ratio and not np.isfinite(median_depth_ratio_val):
        # Missing depth-ratio is soft-fail only if we *require* it; treat as reject
        q.reason = "depth_ratio_unavailable"
        return q

    if cfg.require_reproj and np.isfinite(median_reproj_val):
        if median_reproj_val > cfg.max_median_reproj:
            q.reason = f"reproj={median_reproj_val:.2f}>{cfg.max_median_reproj}"
            return q

    if T_chain is not None:
        q.rot_err = rot_angle(T_meas[:3, :3], T_chain[:3, :3])
        if q.rot_err > cfg.max_rot_err_rad:
            q.reason = f"rot_err={q.rot_err:.3f}>{cfg.max_rot_err_rad}"
            return q
        q.t_scale_ratio = t_scale_ratio(T_meas, T_chain)
        if cfg.require_t_scale_ratio:
            if (
                q.t_scale_ratio < cfg.t_scale_ratio_lo
                or q.t_scale_ratio > cfg.t_scale_ratio_hi
            ):
                q.reason = (
                    f"t_scale={q.t_scale_ratio:.3f} "
                    f"not in [{cfg.t_scale_ratio_lo},{cfg.t_scale_ratio_hi}]"
                )
                return q

    rot_term = 1.0
    if np.isfinite(q.rot_err):
        rot_term = float(np.exp(-((q.rot_err / max(cfg.rot_sigma, 1e-6)) ** 2)))
    inl_term = min(1.0, max(n_inliers, 1) / max(cfg.inlier_ref, 1.0))
    # Prefer depth ratios near 1
    dr = median_depth_ratio_val if np.isfinite(median_depth_ratio_val) else 1.0
    dr_term = float(np.exp(-((np.log(max(dr, 1e-6))) ** 2) / (2 * 0.25**2)))
    q.weight = float(cfg.base_weight * rot_term * inl_term * dr_term)
    q.accepted = True
    q.reason = "accepted"
    return q


def filter_skip_measurements(
    measurements: Iterable[Tuple[int, int, np.ndarray, dict]],
    cfg: Optional[EdgeQualityConfig] = None,
) -> Tuple[List[Tuple[int, int, np.ndarray, float]], List[EdgeQuality]]:
    """Filter (i, j, T, meta) skip measurements.

    ``meta`` may contain keys: inliers, median_depth_ratio, median_reproj, T_chain.
    Returns (accepted list of (i,j,T,weight), all qualities).
    """
    cfg = cfg or EdgeQualityConfig()
    kept: List[Tuple[int, int, np.ndarray, float]] = []
    qualities: List[EdgeQuality] = []
    for i, j, T, meta in measurements:
        meta = meta or {}
        q = evaluate_skip_edge(
            T,
            int(meta.get("inliers", 0)),
            median_depth_ratio_val=float(meta.get("median_depth_ratio", float("nan"))),
            median_reproj_val=float(meta.get("median_reproj", float("nan"))),
            T_chain=meta.get("T_chain"),
            cfg=cfg,
        )
        qualities.append(q)
        if q.accepted:
            kept.append((int(i), int(j), np.asarray(T, dtype=np.float64), q.weight))
    return kept, qualities
