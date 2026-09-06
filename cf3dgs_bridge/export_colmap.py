"""Export solved poses into a minimal COLMAP sparse model for FastGS Scene()."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .dataset import SequenceDataset
from .pose_solver import PoseSequence


def _write_cameras_txt(path: str, dataset: SequenceDataset) -> None:
    K = dataset.intrinsics
    # SIMPLE_PINHOLE: camera_id model width height f cx cy
    f = 0.5 * (K.fx + K.fy)
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
        f"1 SIMPLE_PINHOLE {K.width} {K.height} {f} {K.cx} {K.cy}\n",
    ]
    with open(path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))


def _rotation_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to COLMAP quaternion (w, x, y, z)."""
    # Robust-enough conversion for scaffolding; replace with scipy if available.
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s
        qvec = [0.0, 0.0, 0.0, 0.0]
        qvec[i + 1] = 0.25 * s
        qvec[j + 1] = (R[j, i] + R[i, j]) / s
        qvec[k + 1] = (R[k, i] + R[i, k]) / s
        q[0], q[1], q[2], q[3] = qvec[0], qvec[1], qvec[2], qvec[3]
        # Fix assignment properly:
        q = np.array([qvec[0], qvec[1], qvec[2], qvec[3]], dtype=np.float64)
        # Actually redo cleanly:
        q = np.zeros(4, dtype=np.float64)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return q


def _write_images_txt(path: str, dataset: SequenceDataset, poses: PoseSequence) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(dataset)}",
    ]
    for i, T in enumerate(poses.T_world_cam):
        # COLMAP stores world-to-camera: X_cam = R X_world + t
        R_w2c = T[:3, :3].T
        t_w2c = -R_w2c @ T[:3, 3]
        q = _rotation_to_qvec(R_w2c)
        name = dataset.frame_name(i)
        lines.append(
            f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {name}"
        )
        lines.append("")  # empty POINTS2D line
    with open(path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines) + "\n")


def _write_points3d_txt(path: str, num_dummy: int = 100) -> None:
    """Write a tiny random point cloud so FastGS create_from_pcd has something.

    Real CF-3DGS grows Gaussians during progressive solve; once the photometric
    path is live, replace this with exported Gaussian means / fused PCD.
    """
    rng = np.random.default_rng(0)
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {num_dummy}",
    ]
    pts = rng.normal(size=(num_dummy, 3)) * 0.5
    for i, p in enumerate(pts, start=1):
        lines.append(f"{i} {p[0]} {p[1]} {p[2]} 128 128 128 0")
    with open(path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines) + "\n")


def export_solved_scene_to_colmap(
    dataset: SequenceDataset,
    poses: PoseSequence,
    output_root: str,
    images_symlink_or_copy: bool = True,
) -> str:
    """Write ``output_root/sparse/0/{cameras,images,points3D}.txt`` (+ images/).

    FastGS ``Scene`` detects ``sparse/`` and loads via the Colmap callback.
    Returns the path suitable for ``train.py -s``.
    """
    sparse = os.path.join(output_root, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    images_out = os.path.join(output_root, "images")
    os.makedirs(images_out, exist_ok=True)

    _write_cameras_txt(os.path.join(sparse, "cameras.txt"), dataset)
    _write_images_txt(os.path.join(sparse, "images.txt"), dataset, poses)
    _write_points3d_txt(os.path.join(sparse, "points3D.txt"))

    if images_symlink_or_copy:
        for src in dataset.image_paths:
            dst = os.path.join(images_out, os.path.basename(src))
            if os.path.exists(dst):
                continue
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                import shutil

                shutil.copy2(src, dst)

    print(f"[cf3dgs_bridge] Exported COLMAP sparse model → {sparse}")
    return output_root
