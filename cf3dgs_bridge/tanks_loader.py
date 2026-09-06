"""Load Nope-NeRF / CF-3DGS Tanks preprocessed scenes for pose eval."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import List, Optional, Tuple

import numpy as np

from .dataset import CameraIntrinsics, SequenceDataset


def _find_images(scene_dir: str) -> List[str]:
    for sub in ("images", "rgb", "image"):
        d = os.path.join(scene_dir, sub)
        if os.path.isdir(d):
            files = sorted(
                [
                    os.path.join(d, f)
                    for f in os.listdir(d)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            if files:
                return files
    files = sorted(glob(os.path.join(scene_dir, "*.png")) + glob(os.path.join(scene_dir, "*.jpg")))
    if files:
        return files
    raise FileNotFoundError(f"No images under {scene_dir}")


def _load_poses_any(scene_dir: str, n_hint: Optional[int] = None) -> Optional[List[np.ndarray]]:
    candidates = [
        os.path.join(scene_dir, "poses_bounds.npy"),
        os.path.join(scene_dir, "poses.npy"),
        os.path.join(scene_dir, "cameras.npz"),
        os.path.join(scene_dir, "gt_poses.npy"),
        os.path.join(scene_dir, "pose", "poses_bounds.npy"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        if path.endswith(".npz"):
            data = np.load(path)
            for key in ("poses", "pose", "c2w", "T"):
                if key in data:
                    arr = data[key]
                    break
            else:
                arr = data[data.files[0]]
        else:
            arr = np.load(path)

        if arr.ndim == 3 and arr.shape[-2:] in ((3, 4), (4, 4)):
            poses = []
            for P in arr:
                T = np.eye(4)
                if P.shape == (3, 4):
                    T[:3, :4] = P
                else:
                    T[:] = P
                poses.append(T)
            return poses
        if arr.ndim == 2 and arr.shape[1] >= 12:
            poses = []
            for row in arr:
                pose = row[:15].reshape(3, 5)
                T = np.eye(4)
                T[:3, :4] = pose[:, :4]
                poses.append(T)
            return poses
    for name in ("transforms.json", "transforms_train.json"):
        path = os.path.join(scene_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            meta = json.load(f)
        poses = []
        for fr in meta.get("frames", []):
            poses.append(np.array(fr["transform_matrix"], dtype=np.float64))
        return poses
    return None


def load_tanks_scene(scene_dir: str) -> Tuple[SequenceDataset, Optional[List[np.ndarray]]]:
    images = _find_images(scene_dir)
    import cv2

    im0 = cv2.imread(images[0])
    h, w = im0.shape[:2]
    K = CameraIntrinsics.heuristic_from_resolution(w, h, fov_deg=60.0)
    pb = os.path.join(scene_dir, "poses_bounds.npy")
    if os.path.isfile(pb):
        arr = np.load(pb)
        if arr.ndim == 2 and arr.shape[1] >= 15:
            pose = arr[0, :15].reshape(3, 5)
            hwf = pose[:, 4]
            if hwf[2] > 1:
                K = CameraIntrinsics(
                    width=w, height=h, fx=float(hwf[2]), fy=float(hwf[2]), cx=w / 2, cy=h / 2
                )

    ds = SequenceDataset(root=scene_dir, image_paths=images, intrinsics=K)
    gt = _load_poses_any(scene_dir, n_hint=len(images))
    return ds, gt
