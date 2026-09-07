# Hybrid pose seeder (DepthPnP + LightGlue)

## Product goal

Combine **metric sequential VO** with **robust long-range matching**:

1. **DepthPnP** (`depth_pnp_solver`) — adjacent (and optional ±k) relatives with
   metric scale from DPT mono-depth.
2. **LightGlue** (`lightglue_seeder`) — SuperPoint+LightGlue on jump pairs
   (pair_radius / long_range_stride beyond sequential).
3. **Scale-align** — median ratio of `||t_seq|| / ||t_lg||` vs the composed
   DepthPnP path between the same endpoints; apply to all LightGlue translations.
4. **Fuse** — prefer DepthPnP for adjacent edges; use scaled LightGlue for jumps;
   optional SE(3) pose-graph with higher weight on DepthPnP edges.

```
 images/ + dpt/
    │
    ├── DepthPnP  →  adjacent metric T_{i+1←i}  (+ optional skips)
    │
    └── LightGlue →  long-range unit-scale edges
           │
           ▼
    median scale ratio vs composed DepthPnP path
           │
           ▼
    pose-graph (depth_w ≫ lg_w)  →  PoseSequence
           │
           ▼
    export_colmap  →  FastGS train.py
```

If LightGlue cannot load, the seeder **falls back to sequential-only DepthPnP**.

**Recommended seed for large camera jumps → FastGS** when DPT depth exists.

## On your NVIDIA GPU

End-to-end on a CUDA box:

```bash
# 0) env (once)
cd FastGS
conda activate fastgs   # or: source .venv/bin/activate
pip install git+https://github.com/cvg/LightGlue.git   # SuperPoint+LightGlue

# 1) one-shot: hybrid seed + COLMAP export (+ auto-train if CUDA)
bash scripts/train_from_hybrid_seed.sh \
  /path/to/Tanks/Francis \
  ./eval_out/francis_hybrid_colmap \
  ./output/francis_hybrid_fastgs \
  -- --max_frames 40 --device cuda --pair_radius 2 --long_range_stride 10

# Or step-by-step:
.venv/bin/python scripts/seed_poses_hybrid.py \
  --scene /path/to/Tanks/Francis \
  --max_frames 40 --device cuda \
  --pair_radius 2 --long_range_stride 10 \
  --out_colmap ./eval_out/francis_hybrid_colmap \
  --model_hint ./output/francis_hybrid_fastgs

python train.py -s ./eval_out/francis_hybrid_colmap -m ./output/francis_hybrid_fastgs
```

Export layout (what FastGS `Scene` expects):

```
<out_colmap>/
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.txt
  images/                 # symlinks by default; --copy_images to copy
  README_FASTGS.txt       # one-liner train command
```

`cameras.json` is written by `Scene` into `-m` at train time (not into `-s`).

CPU-only boxes still run seed+export; `train_from_hybrid_seed.sh` skips `train.py`
with a clear message and prints the exact GPU command.

## CLI seed + COLMAP export

```bash
.venv/bin/python scripts/seed_poses_hybrid.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --max_frames 40 \
  --sequential_radius 1 \
  --pair_radius 2 \
  --long_range_stride 10 \
  --out_colmap eval_out/francis_hybrid_colmap \
  --out_json eval_out/francis_hybrid_seed.json
```

Then FastGS:

```bash
python train.py -s eval_out/francis_hybrid_colmap -m output/francis_hybrid_fastgs
```

## ATE eval

```bash
.venv/bin/python scripts/eval_pose_ate.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --backend hybrid \
  --max_frames 40 \
  --pair_radius 2 \
  --long_range_stride 10 \
  --resize_width 640 \
  --device cpu \
  --out_json eval_out/francis_hybrid_40.json
```

Large-jump stress (stride-subsampled frames; hybrid vs adjacent-only DepthPnP):

```bash
.venv/bin/python scripts/eval_large_jump_stress.py \
  --scene /workspace/repos/data/Tanks/Francis \
  --stride 5 --max_keep 24 --resize_width 640 --device cpu \
  --out_json eval_out/francis_large_jump_stride5.json
```

## Module API

* `cf3dgs_bridge/hybrid_seeder.py`
  * `HybridSeedConfig` / `HybridPoseSeeder`
  * `estimate_lightglue_scale_ratio`, `scale_edge_translations`, `jump_pairs_only`
* `scripts/train_from_hybrid_seed.sh` — seed → export → optional `train.py`
* `scripts/eval_large_jump_stress.py` — stride-subsample ATE comparison

## Knobs

| Flag | Role |
| --- | --- |
| `--sequential_radius` | DepthPnP pairs `(i, i+d)` for `d=1..r` |
| `--pair_radius` / `--long_range_stride` | LightGlue jump pairs (gaps > sequential_radius) |
| `--depth_pnp_weight` | Pose-graph weight multiplier for metric edges (default 100) |
| `--lightglue_weight` | Pose-graph weight multiplier for jump edges (default 0.05) |
| `--no_pose_graph` / `--no_hybrid_pose_graph` | Skip SE3 fusion |
| `--copy_images` | Copy into `images/` instead of symlinking |
| `--device cuda` | LightGlue on GPU (default `auto`) |

## Relation to other backends

| Backend | Role |
| --- | --- |
| `hybrid` | Metric sequential + scaled LightGlue jumps (this doc) — **preferred for large jumps** |
| `lightglue` | Learned match seed only (unit-scale) |
| `depth_pnp` | Metric adjacent VO only |
| `progressive` | Photometric CF-3DGS-style refine |

Recommended product path when DPT is available:
**hybrid seed → (optional progressive) → FastGS**.


## NVIDIA benchmark report (feedback loop)

For a **timed, machine-readable** end-to-end run on Alex’s NVIDIA box (env →
hybrid seed → ATE → jump stress → optional FastGS train), use:

```bash
bash scripts/run_on_nvidia.sh /path/to/Tanks/Francis --max_frames 40 --device cuda
```

This writes `eval_out/nvidia_report_<scene>_<timestamp>/report.json` (+
`REPORT.md`, `run.log`, `colmap_export/`). Bring `report.json` (or a zip of the
folder) back to Chief of Staff for knob tuning.

Full docs: [NVIDIA_REPORT.md](./NVIDIA_REPORT.md).

Default `--train_iters 7000` is a FastGS smoke; use `--train_iters 30000` for
full quality (same as stock `OptimizationParams.iterations`).
