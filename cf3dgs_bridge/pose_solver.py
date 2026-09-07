"""Pose graph primitives for CF-3DGS-style local→global camera solving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


@dataclass
class RelativePoseSE3:
    """SE(3) transform relating consecutive cameras."""

    src: int
    dst: int
    T_dst_src: np.ndarray  # 4x4; maps points in src camera frame into dst frame

    def __post_init__(self) -> None:
        self.T_dst_src = np.asarray(self.T_dst_src, dtype=np.float64)
        assert self.T_dst_src.shape == (4, 4)


@dataclass
class PoseSequence:
    """Global camera poses composed from adjacent relative transforms.

    CF-3DGS keeps per-frame RT on the Gaussian model (``init_RT_seq`` / ``get_RT``)
    and fits local models between consecutive views (``init_two_view``,
    ``add_view_v2``), then composes a trajectory. This class mirrors that
    bookkeeping without depending on NVlabs source.
    """

    num_frames: int
    T_world_cam: List[np.ndarray] = field(default_factory=list)
    relatives: Dict[Tuple[int, int], RelativePoseSE3] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.T_world_cam:
            self.T_world_cam = [eye4() for _ in range(self.num_frames)]

    def set_identity_anchor(self, idx: int = 0) -> None:
        self.T_world_cam[idx] = eye4()

    def set_relative(self, src: int, dst: int, T_dst_src: np.ndarray) -> None:
        self.relatives[(src, dst)] = RelativePoseSE3(src=src, dst=dst, T_dst_src=T_dst_src)

    def compose_forward(self) -> None:
        """Compose relatives (i → i+1) into global poses with frame 0 as anchor."""
        self.set_identity_anchor(0)
        for i in range(self.num_frames - 1):
            key = (i, i + 1)
            if key not in self.relatives:
                self.set_relative(i, i + 1, eye4())
            T_ip1_i = self.relatives[key].T_dst_src
            # T_w_c{i+1} = T_w_c{i} @ inv(T_{i+1←i})
            self.T_world_cam[i + 1] = self.T_world_cam[i] @ np.linalg.inv(T_ip1_i)

    def as_Rt_lists(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        Rs, ts = [], []
        for T in self.T_world_cam:
            Rs.append(T[:3, :3].copy())
            ts.append(T[:3, 3].copy())
        return Rs, ts
