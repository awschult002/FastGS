# Local frozen-Gaussian photometric SE(3)

Re-implementation of the CF-3DGS (Fu et al., CVPR 2024) **local → freeze →
relative SE(3)** camera step. Does **not** vendor NVlabs source.

## Algorithm (adjacent frames t, t+1)

1. Load DPT depth for frame t (`<scene>/dpt/depth_XXXXXX.npz`, key `pred`).
2. Lift depth → camera-frame point cloud with intrinsics `K` and identity pose;
   downsample (stride 4–8) → init local Gaussians (means, RGB, scales from
   neighbor spacing, opacity, identity quaternion).
3. **Fit** local Gaussians to frame t with photometric L1+SSIM
   (`fit_iters`, default 100 on CPU / 200–500 on GPU). Pose fixed at identity.
4. **Freeze** all Gaussian attributes.
5. Optimize only SE(3) `T` (quaternion + translation) so that transforming
   Gaussians by `T` and rendering matches frame t+1 (`pose_iters`).
6. Relative pose is `T_dst_src` (maps points in frame t into frame t+1), same
   convention as `PoseSequence`.

Optional: warm-start `T` from the classical `depth_pnp` edge (recommended).

## Renderer

| Path | Module | When |
| --- | --- | --- |
| Soft splat (default) | `cf3dgs_bridge/soft_splat.py` | Always; CPU + CUDA PyTorch |
| FastGS hook | `try_fastgs_rasterizer` | CUDA + `diff_gaussian_rasterization_fastgs` importable |

On this box torch is **CPU-only**, so soft splat is required. Keep
`resize_width≈320` and modest iters for measurable ATE in minutes–hours.

## Solver

`cf3dgs_bridge/local_frozen_gauss.py` → `LocalFrozenGaussSolver` /
`LocalFrozenGaussConfig`.

Wired from:

* `ProgressiveCameraSolver` when `dry_run=False`, `backend="local_frozen"`
* `scripts/eval_pose_ate.py --backend local_frozen`

## Quick Francis eval (CPU)

```bash
cd /workspace/repos/FastGS
.venv/bin/python scripts/eval_pose_ate.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --backend local_frozen \
  --max_frames 8 \
  --fit_iters 100 --pose_iters 80 --downsample 6 \
  --out_json eval_out/francis_local_frozen_8.json
```

Paper target (Table 2): Francis **ATE = 0.006**. Classical depth_pnp baseline
is ~**0.036 @ 40 frames**.

## NVIDIA GPU

```bash
# Build FastGS rasterizer submodule first, then:
.venv/bin/python scripts/eval_pose_ate.py \
  --scene /path/to/Francis \
  --backend local_frozen \
  --device cuda \
  --resize_width 480 \
  --fit_iters 300 --pose_iters 200
```

With CUDA, `try_fastgs_rasterizer` will detect the extension; until the
GaussianModel/Camera adapter is fully wired it still uses soft splat on GPU
(much faster than CPU). Point `device=cuda` regardless.
