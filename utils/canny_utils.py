"""Canny edge helpers for seeding and densification boosts."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def torch_image_to_numpy_rgb(image) -> np.ndarray:
    """Convert a torch CHW float image in [0, 1] (or similar) to HxWx3 uint8 RGB."""
    if torch is not None and isinstance(image, torch.Tensor):
        arr = image.detach().float().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # Heuristic: floats in ~[0,1] vs already [0,255]
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def canny_edges(
    image: Union[np.ndarray, "torch.Tensor"],
    low: float = 50.0,
    high: float = 150.0,
) -> np.ndarray:
    """Run OpenCV Canny; returns HxW uint8 edge map (0 or 255)."""
    if not isinstance(image, np.ndarray) or (
        isinstance(image, np.ndarray) and image.dtype != np.uint8
    ):
        # torch / float numpy → uint8 RGB then gray
        if torch is not None and isinstance(image, torch.Tensor):
            rgb = torch_image_to_numpy_rgb(image)
        elif isinstance(image, np.ndarray) and image.dtype != np.uint8:
            rgb = torch_image_to_numpy_rgb(image)
        else:
            rgb = image
    else:
        rgb = image

    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb
    return cv2.Canny(gray, float(low), float(high))


def edge_mask_bool(
    image: Union[np.ndarray, "torch.Tensor"],
    low: float = 50.0,
    high: float = 150.0,
    dilate: int = 0,
) -> np.ndarray:
    """Boolean HxW edge mask; optional morphological dilation (kernel radius)."""
    edges = canny_edges(image, low=low, high=high)
    if dilate and dilate > 0:
        k = 2 * int(dilate) + 1
        kernel = np.ones((k, k), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges > 0


def sample_edge_pixels(
    edge_mask: np.ndarray,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample up to ``num_samples`` (u, v) = (x, y) pixel coords from a boolean edge mask.

    Returns Nx2 float32 array of (x, y) with x=column, y=row. May return fewer
    points if the mask has fewer edge pixels.
    """
    mask = np.asarray(edge_mask).astype(bool)
    ys, xs = np.where(mask)
    n = len(xs)
    if n == 0 or num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()
    take = min(int(num_samples), n)
    idx = rng.choice(n, size=take, replace=False)
    return np.stack([xs[idx].astype(np.float32), ys[idx].astype(np.float32)], axis=1)
