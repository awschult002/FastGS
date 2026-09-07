# LightGlue pose seeder (COLMAP-free large-jump seed)

## Product goal

Fast high-quality Gaussian splats **without expensive COLMAP**. Feed-forward
COLMAP-free models often fail when cameras jump a lot (common in handheld /
sparse capture). This seeder closes that gap:

1. **Robust learned matching seed** — SuperPoint + LightGlue (vid2scene / hloc
   style); ALIKED optional when deps allow.
2. **Optional progressive photometric refine** — CF-3DGS-style local/global
   where overlap is good (not the only path; no sequential-only assumption).
3. **FastGS train** — export COLMAP `sparse/0` and run stock `train.py`.

```
 images/
    │
    ▼
 SuperPoint / ALIKED  →  LightGlue matches
    │                      (sequential ±k + long-range every-Nth)
    ▼
 Essential / recoverPose  →  view graph (relative SE3 + inliers)
    │
    ▼
 Max-inlier spanning tree  →  PoseSequence  (+ optional pose-graph)
    │
    ├── optional: progressive / local_frozen photometric refine
    ▼
 export_solved_scene_to_colmap  →  FastGS train.py
```

Essential-matrix translations are **unit-scale** per edge; by default we
multiply ``t`` by ``|j-i|`` (constant-velocity prior) so long-range edges
do not collapse the path. Global metric scale is recovered later by
Umeyama (eval), depth-PnP, or photometric refine.

## Install

```bash
# From the FastGS venv
pip install git+https://github.com/cvg/LightGlue.git
# Needs: torch, kornia; torchvision is pulled in but SuperPoint works via
# lightglue.superpoint even if `import lightglue` fails on torchvision mismatches.
```

Device defaults to **CPU** when CUDA is absent; pass `--device cuda` when available.

If LightGlue / weights fail to load, the seeder **falls back to OpenCV SIFT**.

## CLI seed + COLMAP export

```bash
.venv/bin/python scripts/seed_poses_lightglue.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --max_frames 40 \
  --pair_radius 2 \
  --long_range_stride 10 \
  --out_colmap /tmp/francis_lg_seed \
  --out_json eval_out/francis_lg_seed.json
```

Then FastGS:

```bash
python train.py -s /tmp/francis_lg_seed -m output/francis_lg_fastgs
```

## ATE eval

```bash
.venv/bin/python scripts/eval_pose_ate.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --backend lightglue \
  --max_frames 40 \
  --pair_radius 2 \
  --long_range_stride 10 \
  --resize_width 640 \
  --device cpu \
  --out_json eval_out/francis_lightglue_40.json
```

Alias: `--backend seed_lightglue`.

## Module API

* `cf3dgs_bridge/lightglue_seeder.py`
  * `LightGlueSeedConfig` / `LightGluePoseSeeder`
  * `build_pairs`, `spanning_tree_edges`, `compose_poses_from_tree`
* Reuses `recover_pose_opencv5` (OpenCV 5-safe) and `optimize_pose_graph`.

## Knobs

| Flag | Role |
| --- | --- |
| `--pair_radius` | Sequential pairs `(i, i+d)` for `d=1..k` |
| `--long_range_stride` | Every-Nth keyframe + stride edges for large jumps |
| `--max_keypoints` | SuperPoint/ALIKED cap (default 2048) |
| `--feature_backend` | `superpoint` (default), `aliked`, `sift` |
| `--pose_graph` / `--pose_graph_seed` | Opt-in SE3 pose-graph refine (off by default; unit-scale edges) |

## Relation to other backends

| Backend | Role |
| --- | --- |
| `lightglue` | Learned match seed under large jumps |
| `depth_pnp` | Metric scale via DPT + PnP (good overlap) |
| `local_frozen` / `progressive` | Photometric CF-3DGS-style refine |
| `opencv` | Classical SIFT/ORB essential baseline |
| `hybrid` | DepthPnP sequential + scaled LightGlue jumps (see `HYBRID_SEEDER.md`) |

When DPT depth is available, prefer **hybrid** (metric adjacent + LightGlue jumps).
Otherwise: **LightGlue seed → (optional progressive) → FastGS**.
