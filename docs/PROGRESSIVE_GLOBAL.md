# Progressive global joint Gaussian + pose growth (CF-3DGS §3.3)

After each **local frozen** relative pose for a new frame, maintain a world-frame
Gaussian set `G_global` and jointly refine Gaussians + recent SE(3) poses
against multiple views. This is the paper’s progressive global stage that
reduces drift vs local-only chaining.

Re-implements the *technique* from Fu et al. (CF-3DGS); does **not** vendor
NVlabs source. Rasterization uses `soft_splat` (CPU/CUDA).

## Pipeline (frame i = 1 … N−1)

1. **Local frozen** edge `i−1 → i` (`LocalFrozenGaussSolver.estimate_relative`)
   → relative `T_{i−1→i}` (optionally warm-started from depth_pnp).
2. Compose into global `PoseSequence` (`T_world_cam` = camera-to-world).
3. **Densify** `G_global`: lift new-view DPT points into world with current
   `c2w[i]`, append, opacity-prune / subsample to `max_global_gaussians`.
4. **Joint optimize** for `global_iters`: Gaussian attrs + last `pose_window`
   poses (frame 0 frozen) via photometric L1+SSIM through soft_splat on the
   optimizable window plus a small number of older frozen support views.
5. Write refined poses back into `PoseSequence` and re-sync adjacent relatives.

Frame 0 seeds `G_global` (lift + brief `init_fit_iters`).

## Config (`ProgressiveGlobalConfig`)

| Knob | Default | Role |
| --- | --- | --- |
| `fit_iters` / `pose_iters` | 80 / 60 | Local edge (CPU-friendly) |
| `global_iters` | 40 | Joint world stage per new frame |
| `pose_window` | 3 | How many recent poses stay optimizable |
| `max_global_gaussians` | 6000 | Hard cap on `G_global` |
| `densify_every` | 1 | Densify every N frames |
| `densify_stride` / `densify_max_new` | 8 / 1000 | New-view lift density |
| `resize_width` | 320 | Soft-splat resolution |

## CLI

```bash
cd /workspace/repos/FastGS
.venv/bin/python scripts/eval_pose_ate.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --backend progressive \
  --max_frames 20 \
  --resize_width 320 \
  --fit_iters 80 --pose_iters 60 --global_iters 40 \
  --out_json eval_out/francis_progressive_20.json
```

Also reachable as `ProgressiveCameraSolver(dry_run=False, backend="progressive")`
(alias: `backend="local_frozen_global"`).

## Local-only vs progressive

| | Local frozen | Progressive global |
| --- | --- | --- |
| Relative pose | Per-edge fit→freeze→SE3 | Same local step first |
| World model | None (discard after edge) | Growing `G_global` |
| Pose refine | Adjacent only | Joint over recent window |
| Drift | Accumulates with length | Partially corrected by multi-view photo |

Francis ATE (this fork, CPU soft_splat, Umeyama-aligned):

| Frames | Local frozen | Progressive global | Paper |
| --- | --- | --- | --- |
| 8 | 0.0089 | **0.0078** | 0.006 |
| 20 | 0.0234 | **0.0107** | 0.006 |
| 40 | 0.0507 | **0.0318** | 0.006 |

Progressive uses `fit=80/pose=60/global=40/pose_window=3/maxG=5000`. Local-only
baselines used `fit=100/pose=80`. Long-horizon drift is reduced by multi-view
world Gaussian fitting + regularized pose window updates.

## Modules

* `cf3dgs_bridge/progressive_global.py` — solver
* `cf3dgs_bridge/progressive.py` — `ProgressiveCameraSolver` dispatch
* `cf3dgs_bridge/local_frozen_gauss.py` — reused local edge
* `cf3dgs_bridge/soft_splat.py` — renderer
