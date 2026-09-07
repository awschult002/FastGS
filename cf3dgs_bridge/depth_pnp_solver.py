"""Progressive visual odometry with mono-depth + solvePnPRansac.

For each adjacent pair (i-1 → i):
  1. SIFT match features
  2. Unproject matched pixels in frame i-1 with aligned DPT depth
  3. ``cv2.solvePnPRansac`` → T_curr←prev (maps points in prev cam into curr)
  4. Chain relatives with ``PoseSequence``

Depth maps are loaded from ``<scene>/dpt/depth_<id>.npz`` (``pred`` key) and
resized to the working image resolution. Metric scale of DPT is arbitrary;
ATE evaluation applies Umeyama similarity alignment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .dataset import SequenceDataset
from .opencv_solver import _create_feature_backend, _good_matches, _load_gray, _scaled_K, OpenCVSolveConfig
from .pose_solver import PoseSequence, eye4


@dataclass
class DepthPnPSolveConfig:
    max_features: int = 4000
    match_ratio: float = 0.75
    min_matches: int = 30
    reproj_err: float = 3.0
    ransac_iters: int = 200
    confidence: float = 0.999
    resize_width: Optional[int] = 960
    min_depth: float = 1e-3
    max_depth: float = 80.0
    depth_subdir: str = "dpt"
    depth_key: str = "pred"
    feature_backend: str = "sift"


_FRAME_ID_RE = re.compile(r"(\d+)")


def frame_numeric_id(path: str) -> Optional[str]:
    """Extract the last contiguous digit group from a filename stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    matches = _FRAME_ID_RE.findall(stem)
    return matches[-1] if matches else None


def discover_depth_maps(
    scene_dir: str,
    image_paths: List[str],
    depth_subdir: str = "dpt",
) -> List[Optional[str]]:
    """Align depth npz paths to ``image_paths`` by numeric frame id.

    Looks under ``scene_dir/depth_subdir`` for ``depth_<id>.npz`` or ``<id>.npz``.
    Returns a list of the same length as ``image_paths`` (None if missing).
    """
    ddir = os.path.join(scene_dir, depth_subdir)
    if not os.path.isdir(ddir):
        return [None] * len(image_paths)

    by_id: Dict[str, str] = {}
    for name in os.listdir(ddir):
        if not name.lower().endswith(".npz"):
            continue
        fid = frame_numeric_id(name)
        if fid is None:
            continue
        # Prefer depth_<id>.npz over bare <id>.npz if both exist
        path = os.path.join(ddir, name)
        if fid not in by_id or name.startswith("depth_"):
            by_id[fid] = path

    out: List[Optional[str]] = []
    for img in image_paths:
        fid = frame_numeric_id(img)
        out.append(by_id.get(fid) if fid is not None else None)
    return out


def scene_has_dpt(scene_dir: str, depth_subdir: str = "dpt") -> bool:
    ddir = os.path.join(scene_dir, depth_subdir)
    if not os.path.isdir(ddir):
        return False
    return any(n.lower().endswith(".npz") for n in os.listdir(ddir))


def load_depth_npz(path: str, key: str = "pred") -> np.ndarray:
    """Load a DPT-style depth map; returns HxW float32."""
    data = np.load(path, allow_pickle=True)
    if key in data:
        arr = data[key]
    else:
        arr = data[data.files[0]]
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected depth shape {arr.shape} in {path}")
    return arr


def resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape[0] == height and depth.shape[1] == width:
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


def unproject_pixels(
    pts_xy: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject 2D pixels with a depth map into camera-frame 3D.

    Returns (points Nx3, valid_mask length N).
    """
    h, w = depth.shape
    xs = pts_xy[:, 0]
    ys = pts_xy[:, 1]
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    z = depth[yi, xi].astype(np.float64)
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    pts3d = np.stack([X, Y, z], axis=1)
    return pts3d, valid


class DepthPnPProgressiveSolver:
    """Progressive VO: DPT depth unprojection + solvePnPRansac on SIFT matches."""

    def __init__(
        self,
        dataset: SequenceDataset,
        depth_paths: List[Optional[str]],
        cfg: Optional[DepthPnPSolveConfig] = None,
    ):
        self.dataset = dataset
        self.depth_paths = depth_paths
        self.cfg = cfg or DepthPnPSolveConfig()
        if len(depth_paths) != len(dataset):
            raise ValueError(
                f"depth_paths length {len(depth_paths)} != num frames {len(dataset)}"
            )
        self.poses = PoseSequence(num_frames=len(dataset))
        # Reuse OpenCV feature factory
        feat_cfg = OpenCVSolveConfig(
            max_features=self.cfg.max_features,
            feature_backend=self.cfg.feature_backend,
        )
        self.detector, self.bf, self.backend_name = _create_feature_backend(feat_cfg)

    def solve(self) -> PoseSequence:
        n = len(self.dataset)
        if n < 2:
            raise ValueError("Need >= 2 frames")
        missing = sum(1 for p in self.depth_paths if p is None)
        if missing:
            print(f"[depth_pnp] WARNING: {missing}/{n} frames missing depth maps")
        print(
            f"[depth_pnp] Progressive depth+PnP solve on {n} frames "
            f"(features={self.backend_name}, cv={cv2.__version__})"
        )
        self.poses.set_identity_anchor(0)

        K_full = self.dataset.intrinsics.as_K()
        prev_img, prev_scale = _load_gray(self.dataset.image_paths[0], self.cfg.resize_width)
        prev_kp, prev_des = self.detector.detectAndCompute(prev_img, None)
        prev_depth = self._load_depth_for(0, prev_img.shape[1], prev_img.shape[0])

        for i in range(1, n):
            img, scale = _load_gray(self.dataset.image_paths[i], self.cfg.resize_width)
            kp, des = self.detector.detectAndCompute(img, None)
            # Use prev frame scale for K / depth (object points live in prev cam)
            K = _scaled_K(K_full, prev_scale)

            T_rel, n_inliers = self._relative_pose_pnp(
                prev_kp, prev_des, kp, des, prev_depth, K
            )
            self.poses.set_relative(i - 1, i, T_rel)
            print(
                f"[depth_pnp] {i-1:03d}→{i:03d} inliers={n_inliers} "
                f"t_norm={np.linalg.norm(T_rel[:3, 3]):.4f}"
            )
            prev_img, prev_scale, prev_kp, prev_des = img, scale, kp, des
            prev_depth = self._load_depth_for(i, img.shape[1], img.shape[0])

        self.poses.compose_forward()
        return self.poses

    def _load_depth_for(self, idx: int, width: int, height: int) -> Optional[np.ndarray]:
        path = self.depth_paths[idx]
        if path is None:
            return None
        depth = load_depth_npz(path, key=self.cfg.depth_key)
        return resize_depth(depth, width, height)

    def _relative_pose_pnp(
        self,
        kp1,
        des1,
        kp2,
        des2,
        depth1: Optional[np.ndarray],
        K: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        if depth1 is None:
            return eye4(), 0
        good = _good_matches(self.bf, des1, des2, self.cfg.match_ratio)
        if len(good) < self.cfg.min_matches:
            return eye4(), 0

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
        pts3d, valid = unproject_pixels(
            pts1, depth1, K, self.cfg.min_depth, self.cfg.max_depth
        )
        if int(valid.sum()) < self.cfg.min_matches:
            return eye4(), 0

        obj = pts3d[valid]
        img_pts = pts2[valid]
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj.astype(np.float64),
                img_pts.astype(np.float64),
                K,
                None,
                iterationsCount=self.cfg.ransac_iters,
                reprojectionError=self.cfg.reproj_err,
                confidence=self.cfg.confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception as exc:
            print(f"[depth_pnp] solvePnPRansac failed: {exc}")
            return eye4(), 0

        if not ok or rvec is None or tvec is None:
            return eye4(), 0

        R, _ = cv2.Rodrigues(rvec)
        T = eye4()
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        n_inl = int(len(inliers)) if inliers is not None else 0
        return T, n_inl
