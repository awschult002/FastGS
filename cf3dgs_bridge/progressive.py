"""Progressive local-to-global camera / Gaussian solver (CF-3DGS-inspired)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dataset import SequenceDataset
from .pose_solver import PoseSequence, _eye4


@dataclass
class ProgressiveSolveConfig:
    """Hyperparameters mirroring CF-3DGS progressive schedule (scaffolded)."""

    local_iters: int = 1000
    single_step: int = 500  # CF-3DGS: 300 faster / 500 better
    densify_local: bool = False
    use_monocular_depth: bool = True  # paper uses mono depth cues
    dry_run: bool = True  # until CUDA training is wired


class ProgressiveCameraSolver:
    """Scaffold for CF-3DGS ``train_from_progressive`` on top of FastGS.

    Upstream CF-3DGS flow (see NVlabs ``trainer/cf3dgs_trainer.py``):
      1. ``init_two_view(0, ...)`` — fit Gaussians on the first frame (pose fixed).
      2. For each next frame ``fidx``: ``add_view_v2(fidx, fidx-1)`` — fit a *local*
         Gaussian model on the previous view, then optimize the relative camera
         of the new view via photometric (+ depth) loss.
      3. Compose local relatives into a global RT sequence stored on the model.
      4. Optionally refine / evaluate novel views.

    This scaffold implements the control flow and bookkeeping. The photometric
    local solve that needs CUDA + FastGS rasterizer is intentionally stubbed
    behind ``dry_run`` so the pipeline can be developed and unit-tested offline.
    """

    def __init__(self, dataset: SequenceDataset, cfg: Optional[ProgressiveSolveConfig] = None):
        self.dataset = dataset
        self.cfg = cfg or ProgressiveSolveConfig()
        self.poses = PoseSequence(num_frames=len(dataset))

    def solve(self) -> PoseSequence:
        n = len(self.dataset)
        if n < 2:
            raise ValueError("Need at least 2 frames for progressive pose solving.")

        print(f"[cf3dgs_bridge] Progressive solve on {n} frames (dry_run={self.cfg.dry_run})")
        self._init_anchor_view(0)

        for fidx in range(1, n):
            T_rel = self._estimate_relative_pose(fidx - 1, fidx)
            self.poses.set_relative(fidx - 1, fidx, T_rel)
            print(f"[cf3dgs_bridge] frames {fidx - 1:03d} → {fidx:03d} relative pose recorded")

        self.poses.compose_forward()
        return self.poses

    def _init_anchor_view(self, idx: int) -> None:
        """CF-3DGS ``init_two_view`` analogue: lock frame 0 as world origin."""
        self.poses.set_identity_anchor(idx)
        if self.cfg.dry_run:
            return
        # TODO: create FastGS GaussianModel from mono-depth / random PCD of frame 0,
        # optimize appearance with update_cam=False (pose fixed), matching CF-3DGS.
        raise NotImplementedError("Wire FastGS rasterizer + GaussianModel for anchor fit.")

    def _estimate_relative_pose(self, prev_idx: int, curr_idx: int) -> np.ndarray:
        """CF-3DGS ``add_view_v2`` analogue: local model + optimize new camera."""
        if self.cfg.dry_run:
            # Identity keeps the export path / CLI smoke-testable without GPU.
            # Replace with SE3 from photometric alignment when implementing for real.
            return _eye4()

        # TODO outline (non-binding implementation plan):
        # 1. Build local Gaussians from prev frame (or crop of global model).
        # 2. Initialize curr camera near prev (small motion prior / identity).
        # 3. Optimize T_curr_prev with photometric loss (+ optional mono depth).
        # 4. Grow / merge Gaussians into the global set; store RT for curr_idx.
        raise NotImplementedError("Wire relative SE3 photometric solve against FastGS render.")
