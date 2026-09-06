"""COLMAP-free camera solving bridge inspired by CF-3DGS (Fu et al., CVPR 2024 / NVlabs).

This package sits *in front of* FastGS training:

1. Load a sequential image/video folder without ``sparse/`` COLMAP output.
2. Progressively estimate relative camera poses (local → global) while growing
   a Gaussian set, following CF-3DGS's train_from_progressive / add_view_v2 idea.
3. Export a COLMAP-compatible ``sparse/0`` (and optional point cloud) so the
   existing FastGS ``train.py`` path can run unchanged.

Reference
---------
- Paper: https://arxiv.org/abs/2312.07504
- Code:  https://github.com/NVlabs/CF-3DGS
- Project: https://oasisyang.github.io/colmap-free-3dgs/

Licensing note: CF-3DGS is NVIDIA proprietary / all-rights-reserved in upstream
LICENSE. Do **not** copy NVlabs source verbatim into this fork. Re-implement
the *techniques* described in the paper, or vendor only under an explicit
license grant. This scaffold is an original FastGS-side integration shell.
"""

from .dataset import SequenceDataset, discover_images
from .pose_solver import PoseSequence, RelativePoseSE3
from .progressive import ProgressiveCameraSolver
from .export_colmap import export_solved_scene_to_colmap

__all__ = [
    "SequenceDataset",
    "discover_images",
    "PoseSequence",
    "RelativePoseSE3",
    "ProgressiveCameraSolver",
    "export_solved_scene_to_colmap",
]
