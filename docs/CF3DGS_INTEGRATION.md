# CF-3DGS camera solve × FastGS (integration design)

## Motivation

- **COLMAP** is often the slowest step before 3DGS and the quality of that solve
  heavily gates reconstruction.
- **FastGS** (this repo) accelerates *training* assuming poses already exist.
- **CF-3DGS** (Fu et al., CVPR 2024; NVIDIA / UCSD / Berkeley) drops COLMAP by
  progressively growing Gaussians on a video sequence and jointly estimating
  cameras from temporal continuity + photometric loss.

Goal of this fork work: keep FastGS’s multi-view densification/pruning speed,
but *front-load* a CF-3DGS-inspired COLMAP-free camera solve so unposed video
can go end-to-end.

## References

| Piece | Link |
| --- | --- |
| CF-3DGS paper | https://arxiv.org/abs/2312.07504 |
| CF-3DGS project | https://oasisyang.github.io/colmap-free-3dgs/ |
| CF-3DGS code (do not copy verbatim) | https://github.com/NVlabs/CF-3DGS |
| FastGS paper | https://arxiv.org/abs/2511.04283 |

## Architecture

```
 images/ (ordered frames)
        │
        ▼
 cf3dgs_bridge.SequenceDataset
        │
        ▼
 ProgressiveCameraSolver          ← CF-3DGS train_from_progressive / add_view_v2 ideas
   • init anchor view (pose fixed)
   • for each next frame: local Gaussian fit + relative SE3
   • compose global T_world_cam
        │
        ▼
 export_solved_scene_to_colmap    → sparse/0/{cameras,images,points3D}.txt + images/
        │
        ▼
 FastGS train.py (unchanged)      ← VCD / VCP / compact-box path
```

Classic COLMAP FastGS remains the default: if ``sparse/`` already exists,
``train.py`` behaves as upstream.

## What’s implemented in this scaffold

| Module | Status |
| --- | --- |
| `cf3dgs_bridge/dataset.py` | Image discovery + heuristic intrinsics |
| `cf3dgs_bridge/pose_solver.py` | Relative / global SE3 bookkeeping |
| `cf3dgs_bridge/progressive.py` | Control flow; **dry_run** identity relatives |
| `cf3dgs_bridge/export_colmap.py` | Text COLMAP model export |
| `train_colmap_free.py` | CLI for solve → export → optional `train.py` |
| Photometric local SE3 solve | **TODO** (needs CUDA + FastGS rasterizer) |
| Mono-depth prior | **TODO** |
| Real progressive Gaussian growth | **TODO** |

## License caution

Upstream CF-3DGS ships an NVIDIA all-rights-reserved LICENSE. This bridge
**re-implements techniques described in the paper** and cites the work; it does
**not** vendor NVlabs source. Before shipping a binary redistribution, re-check
both FastGS / 3DGS licenses and CF-3DGS terms.

## How to run (scaffold / dry-run)

```bash
# 1) Solve + export (identity relatives until photometric path lands)
python train_colmap_free.py \
  -s /path/to/seq \
  --export_dir /path/to/seq_solved

# 2) Train FastGS as usual on the exported COLMAP layout
python train.py -s /path/to/seq_solved -m output/seq_run
```

Or via helper:

```bash
bash scripts/train_colmap_free.sh /path/to/seq /path/to/seq_solved
```

## Next implementation steps

1. Replace `ProgressiveCameraSolver._estimate_relative_pose` dry-run with a
   FastGS `render_fastgs` photometric loop (local Gaussians, optimize SE3).
2. Seed frame-0 Gaussians from mono-depth back-projection (as CF-3DGS does)
   instead of the placeholder `points3D.txt`.
3. After a full trajectory exists, optionally *jointly refine* poses for a few
   epochs, then freeze cameras and run stock FastGS densification.
4. Add a small synthetic / TUM-style smoke dataset under `datasets/` for CI.
5. Benchmark: COLMAP+FastGS vs COLMAP-free bridge on the same video (ATE / PSNR).

## Mapping to CF-3DGS symbols

| CF-3DGS | This fork |
| --- | --- |
| `CFGaussianTrainer.train_from_progressive` | `ProgressiveCameraSolver.solve` |
| `init_two_view` | `_init_anchor_view` |
| `add_view_v2` | `_estimate_relative_pose` |
| `gaussians.init_RT_seq` / `get_RT` | `PoseSequence` |
| export for FastGS | `export_solved_scene_to_colmap` |
