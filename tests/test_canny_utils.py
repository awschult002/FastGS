"""CPU unit tests for Canny edge helpers (no CUDA)."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.canny_utils import canny_edges, edge_mask_bool, sample_edge_pixels, torch_image_to_numpy_rgb


def _white_square_on_black(h: int = 64, w: int = 64, margin: int = 16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[margin : h - margin, margin : w - margin] = 255
    return img


def test_canny_edges_finds_square():
    img = _white_square_on_black()
    edges = canny_edges(img, low=50.0, high=150.0)
    assert edges.shape == (64, 64)
    assert edges.dtype == np.uint8
    assert int((edges > 0).sum()) > 20, "expected Canny to detect square edges"


def test_edge_mask_bool_and_dilate():
    img = _white_square_on_black()
    mask = edge_mask_bool(img, low=50.0, high=150.0, dilate=0)
    mask_d = edge_mask_bool(img, low=50.0, high=150.0, dilate=2)
    assert mask.dtype == bool
    assert mask_d.sum() >= mask.sum()


def test_sample_edge_pixels():
    img = _white_square_on_black()
    mask = edge_mask_bool(img, low=50.0, high=150.0)
    rng = np.random.default_rng(0)
    samples = sample_edge_pixels(mask, num_samples=50, rng=rng)
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert 0 < samples.shape[0] <= 50
    # Samples should lie on edge pixels
    for x, y in samples:
        assert mask[int(y), int(x)]


def test_torch_image_to_numpy_rgb():
    try:
        import torch
    except ImportError:
        return
    t = torch.zeros(3, 8, 8)
    t[:, 2:6, 2:6] = 1.0
    rgb = torch_image_to_numpy_rgb(t)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    assert rgb[4, 4, 0] == 255
