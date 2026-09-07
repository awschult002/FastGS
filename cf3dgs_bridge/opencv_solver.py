"""CPU progressive relative-pose solver via OpenCV (interim COLMAP-free path).

This is *not* CF-3DGS photometric Gaussian solving. It estimates adjacent-frame
SE(3) with ORB + essential matrix, composes a trajectory, and is meant as a
measurable baseline until a CUDA photometric path is available.

CF-3DGS paper accuracy (Fu et al., CVPR 2024) requires their progressive
Gaussian + mono-depth photometric optimization on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .dataset import SequenceDataset
from .pose_solver import PoseSequence, eye4


@dataclass
class OpenCVSolveConfig:
    max_features: int = 4000
    match_ratio: float = 0.75
    ransac_prob: float = 0.999
    ransac_threshold: float = 1.0
    min_matches: int = 30
    resize_width: Optional[int] = 960  # speed / stability


def _load_gray(path: str, resize_width: Optional[int]) -> Tuple[np.ndarray, float]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    scale = 1.0
    if resize_width is not None and img.shape[1] > resize_width:
        scale = resize_width / float(img.shape[1])
        img = cv2.resize(img, (resize_width, int(round(img.shape[0] * scale))))
    return img, scale


def _scaled_K(K: np.ndarray, scale: float) -> np.ndarray:
    K2 = K.copy()
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] *= scale
    K2[1, 2] *= scale
    return K2


class OpenCVProgressiveSolver:
    """Progressive local→global pose from pairwise essential matrices."""

    def __init__(self, dataset: SequenceDataset, cfg: Optional[OpenCVSolveConfig] = None):
        self.dataset = dataset
        self.cfg = cfg or OpenCVSolveConfig()
        self.poses = PoseSequence(num_frames=len(dataset))
        self.orb = cv2.ORB_create(nfeatures=self.cfg.max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def solve(self) -> PoseSequence:
        n = len(self.dataset)
        if n < 2:
            raise ValueError("Need >= 2 frames")
        print(f"[opencv_solver] Progressive OpenCV solve on {n} frames")
        self.poses.set_identity_anchor(0)

        prev_img, prev_scale = _load_gray(self.dataset.image_paths[0], self.cfg.resize_width)
        prev_kp, prev_des = self.orb.detectAndCompute(prev_img, None)
        K_full = self.dataset.intrinsics.as_K()

        for i in range(1, n):
            img, scale = _load_gray(self.dataset.image_paths[i], self.cfg.resize_width)
            kp, des = self.orb.detectAndCompute(img, None)
            K = _scaled_K(K_full, 0.5 * (prev_scale + scale))
            T_rel, n_inliers = self._relative_pose(prev_kp, prev_des, kp, des, K)
            self.poses.set_relative(i - 1, i, T_rel)
            print(
                f"[opencv_solver] {i-1:03d}→{i:03d} inliers={n_inliers} "
                f"t_norm={np.linalg.norm(T_rel[:3, 3]):.4f}"
            )
            prev_img, prev_scale, prev_kp, prev_des = img, scale, kp, des

        self.poses.compose_forward()
        return self.poses

    def _relative_pose(self, kp1, des1, kp2, des2, K) -> Tuple[np.ndarray, int]:
        if des1 is None or des2 is None:
            return eye4(), 0
        knn = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.match_ratio * n.distance:
                good.append(m)
        if len(good) < self.cfg.min_matches:
            return eye4(), 0

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=self.cfg.ransac_prob,
            threshold=self.cfg.ransac_threshold,
        )
        if E is None or mask is None:
            return eye4(), 0
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        inliers = int(mask_pose.sum()) if mask_pose is not None else 0
        T = eye4()
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return T, inliers


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Umeyama similarity aligning src (Nx3) onto dst (Nx3). Returns s, R, t."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (src_c ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def absolute_trajectory_error(pred_T: List[np.ndarray], gt_T: List[np.ndarray]) -> dict:
    """ATE RMSE after Umeyama alignment of camera centers."""
    n = min(len(pred_T), len(gt_T))
    pred_c = np.stack([T[:3, 3] for T in pred_T[:n]], axis=0)
    gt_c = np.stack([T[:3, 3] for T in gt_T[:n]], axis=0)
    s, R, t = umeyama_alignment(pred_c, gt_c, with_scale=True)
    aligned = (s * (R @ pred_c.T)).T + t
    err = np.linalg.norm(aligned - gt_c, axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(err ** 2))),
        "ate_mean": float(err.mean()),
        "ate_median": float(np.median(err)),
        "num_frames": n,
        "scale": s,
    }
