"""Sequential image / video loaders for COLMAP-free FastGS."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def discover_images(root: str, images_subdir: str = "images") -> List[str]:
    """Return sorted frame paths under ``root/images`` or ``root`` itself."""
    candidates = [
        os.path.join(root, images_subdir),
        root,
    ]
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        files = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.splitext(name.lower())[1] in IMAGE_EXTS
        ]
        if files:
            return files
    raise FileNotFoundError(
        f"No images found under {root!r} (looked for {images_subdir}/ and root)."
    )


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics. CF-3DGS allows COLMAP or heuristic landscape defaults."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def heuristic_from_resolution(cls, width: int, height: int, fov_deg: float = 60.0) -> "CameraIntrinsics":
        # Simple landscape default used when COLMAP intrinsics are unavailable.
        f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
        return cls(width=width, height=height, fx=float(f), fy=float(f), cx=width * 0.5, cy=height * 0.5)

    def as_K(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float64)
        K[0, 0], K[1, 1] = self.fx, self.fy
        K[0, 2], K[1, 2] = self.cx, self.cy
        return K


@dataclass
class SequenceDataset:
    """Ordered frames for progressive pose solving (no COLMAP required)."""

    root: str
    image_paths: List[str]
    intrinsics: CameraIntrinsics

    @classmethod
    def from_folder(
        cls,
        root: str,
        images_subdir: str = "images",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fov_deg: float = 60.0,
    ) -> "SequenceDataset":
        paths = discover_images(root, images_subdir=images_subdir)
        # Lazy size probe: callers can override W/H; otherwise read first frame.
        if width is None or height is None:
            try:
                from PIL import Image

                with Image.open(paths[0]) as im:
                    w, h = im.size
            except Exception as exc:  # pragma: no cover - optional dep at scaffold time
                raise RuntimeError(
                    "Need Pillow to infer image size, or pass --width/--height."
                ) from exc
            width = width or w
            height = height or h
        assert width is not None and height is not None
        K = CameraIntrinsics.heuristic_from_resolution(width, height, fov_deg=fov_deg)
        return cls(root=root, image_paths=paths, intrinsics=K)

    def __len__(self) -> int:
        return len(self.image_paths)

    def frame_name(self, idx: int) -> str:
        return os.path.basename(self.image_paths[idx])
