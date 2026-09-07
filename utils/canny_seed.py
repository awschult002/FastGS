"""Augment an initial point cloud by unprojecting Canny edge pixels.

Depth for each edge sample is estimated by inverse-distance weighting (IDW)
from existing PCD points projected into the same training view.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from utils.canny_utils import canny_edges, sample_edge_pixels
from utils.graphics_utils import BasicPointCloud, fov2focal, getWorld2View2


def _pil_to_rgb_uint8(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        amax = float(arr.max()) if arr.size else 0.0
        if amax <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _project_points(points_w: np.ndarray, cam) -> tuple:
    """Project world points into camera pixel coords + depth.

    Returns (u, v, depth, valid_mask) each length N.
    Uses 3DGS CameraInfo convention: getWorld2View2(R, T).
    """
    W2C = getWorld2View2(cam.R, cam.T)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    pts_c = (R @ points_w.T).T + t  # Nx3 camera frame
    z = pts_c[:, 2]
    w = int(cam.width)
    h = int(cam.height)
    fx = fov2focal(float(cam.FovX), w)
    fy = fov2focal(float(cam.FovY), h)
    cx = w * 0.5
    cy = h * 0.5
    # Avoid div by zero
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = fx * (pts_c[:, 0] / z_safe) + cx
    v = fy * (pts_c[:, 1] / z_safe) + cy
    valid = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u.astype(np.float64), v.astype(np.float64), z.astype(np.float64), valid


def _unproject(u: np.ndarray, v: np.ndarray, depth: np.ndarray, cam) -> np.ndarray:
    """Unproject pixel + depth to world coordinates. Returns Nx3."""
    w = int(cam.width)
    h = int(cam.height)
    fx = fov2focal(float(cam.FovX), w)
    fy = fov2focal(float(cam.FovY), h)
    cx = w * 0.5
    cy = h * 0.5
    x_c = (u - cx) / fx * depth
    y_c = (v - cy) / fy * depth
    z_c = depth
    pts_c = np.stack([x_c, y_c, z_c], axis=1)  # Nx3
    W2C = getWorld2View2(cam.R, cam.T)
    C2W = np.linalg.inv(W2C)
    R = C2W[:3, :3]
    t = C2W[:3, 3]
    return (R @ pts_c.T).T + t


def _idw_depth(
    query_uv: np.ndarray,
    support_uv: np.ndarray,
    support_z: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    max_dist_px: float = 40.0,
) -> np.ndarray:
    """Inverse-distance weighted depth for query pixels from support projections.

    Returns depth array (Q,) with NaN where estimation fails.
    """
    Q = query_uv.shape[0]
    out = np.full(Q, np.nan, dtype=np.float64)
    if support_uv.shape[0] == 0 or Q == 0:
        return out
    # Brute-force IDW is fine for typical seed sizes
    # For each query, find k nearest support points
    # Chunk to limit memory
    chunk = 2048
    k = min(k, support_uv.shape[0])
    for start in range(0, Q, chunk):
        end = min(start + chunk, Q)
        q = query_uv[start:end]  # Cx2
        # CxS distances
        d = np.linalg.norm(q[:, None, :] - support_uv[None, :, :], axis=2)
        nn_idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        nn_d = np.take_along_axis(d, nn_idx, axis=1)
        nn_z = support_z[nn_idx]
        # Reject if nearest is too far
        too_far = nn_d[:, 0] > max_dist_px
        nn_d = np.maximum(nn_d, 1e-3)
        w = 1.0 / (nn_d ** power)
        w_sum = w.sum(axis=1)
        depth = (w * nn_z).sum(axis=1) / np.maximum(w_sum, 1e-12)
        depth[too_far] = np.nan
        out[start:end] = depth
    return out


def augment_pcd_with_canny(
    pcd: BasicPointCloud,
    cameras: Sequence,
    num_points: int = 20000,
    low: float = 50.0,
    high: float = 150.0,
    idw_k: int = 8,
    max_dist_px: float = 40.0,
    seed: int = 0,
) -> BasicPointCloud:
    """Sample Canny edge pixels across views, IDW-depth, unproject, merge into PCD.

    Args:
        pcd: existing BasicPointCloud (COLMAP / random init).
        cameras: list of CameraInfo (or duck-typed with R,T,FovX,FovY,image,width,height).
        num_points: total new edge points to add (spread across cameras).
        low, high: Canny thresholds.
        idw_k: neighbors for IDW depth.
        max_dist_px: max pixel distance to accept an IDW estimate.
        seed: RNG seed.

    Returns:
        New BasicPointCloud with original + edge-seeded points.
    """
    if pcd is None or pcd.points is None or len(pcd.points) == 0:
        print("[canny_seed] empty PCD; skipping edge augmentation")
        return pcd
    if not cameras:
        print("[canny_seed] no cameras; skipping edge augmentation")
        return pcd

    rng = np.random.default_rng(seed)
    base_pts = np.asarray(pcd.points, dtype=np.float64)
    base_cols = np.asarray(pcd.colors, dtype=np.float64)
    if base_cols.max() > 1.5:
        base_cols = base_cols / 255.0
    base_normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.normals is not None else np.zeros_like(base_pts)

    n_cams = len(cameras)
    per_cam = max(1, int(np.ceil(num_points / float(n_cams))))
    new_pts: List[np.ndarray] = []
    new_cols: List[np.ndarray] = []

    for cam in cameras:
        rgb = _pil_to_rgb_uint8(cam.image)
        edges = canny_edges(rgb, low=low, high=high)
        edge_mask = edges > 0
        samples = sample_edge_pixels(edge_mask, per_cam, rng=rng)
        if samples.shape[0] == 0:
            continue

        u_s, v_s, z_s, valid = _project_points(base_pts, cam)
        if not np.any(valid):
            continue
        support_uv = np.stack([u_s[valid], v_s[valid]], axis=1)
        support_z = z_s[valid]

        depth = _idw_depth(
            samples, support_uv, support_z, k=idw_k, max_dist_px=max_dist_px
        )
        ok = np.isfinite(depth) & (depth > 1e-4)
        if not np.any(ok):
            continue
        samples_ok = samples[ok]
        depth_ok = depth[ok]
        world = _unproject(samples_ok[:, 0], samples_ok[:, 1], depth_ok, cam)

        # Colors from image at sample pixels
        xs = np.clip(np.round(samples_ok[:, 0]).astype(int), 0, rgb.shape[1] - 1)
        ys = np.clip(np.round(samples_ok[:, 1]).astype(int), 0, rgb.shape[0] - 1)
        cols = rgb[ys, xs].astype(np.float64) / 255.0

        new_pts.append(world)
        new_cols.append(cols)

    if not new_pts:
        print("[canny_seed] no edge points could be depth-estimated; returning original PCD")
        return pcd

    add_pts = np.concatenate(new_pts, axis=0)
    add_cols = np.concatenate(new_cols, axis=0)
    # Cap to num_points
    if add_pts.shape[0] > num_points:
        pick = rng.choice(add_pts.shape[0], size=num_points, replace=False)
        add_pts = add_pts[pick]
        add_cols = add_cols[pick]
    add_normals = np.zeros_like(add_pts)

    merged_pts = np.concatenate([base_pts, add_pts], axis=0)
    merged_cols = np.concatenate([base_cols, add_cols], axis=0)
    merged_normals = np.concatenate([base_normals, add_normals], axis=0)
    print(
        f"[canny_seed] added {add_pts.shape[0]} edge points "
        f"(PCD {base_pts.shape[0]} → {merged_pts.shape[0]})"
    )
    return BasicPointCloud(points=merged_pts, colors=merged_cols, normals=merged_normals)
