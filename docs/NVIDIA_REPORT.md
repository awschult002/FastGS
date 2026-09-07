# NVIDIA report (hybrid seed → FastGS feedback loop)

Alex runs this on an **NVIDIA GPU box**. It produces a machine-readable
`report.json` that Chief of Staff / the agent can parse to tune hybrid-seed
knobs and decide next FastGS train settings.

## Install deps (once)

```bash
cd FastGS
conda activate fastgs   # or: source .venv/bin/activate

# LightGlue (SuperPoint + matcher) — needed for jump edges
pip install git+https://github.com/cvg/LightGlue.git

# FastGS CUDA rasterizer / fused ops (same as normal FastGS train setup)
# Follow README / environment.yml; train.py will fail clearly if missing.
```

CPU-only boxes can still run **seed + ATE + jump stress** with `--skip_train`.

## How to run

```bash
# Recommended wrapper (timestamped eval_out/nvidia_report_<scene>_<ts>/)
bash scripts/run_on_nvidia.sh /path/to/Tanks/Francis \
  --max_frames 40 --device cuda

# Full-quality FastGS train after a good seed (default smoke iters=7000)
bash scripts/run_on_nvidia.sh /path/to/Tanks/Francis \
  --max_frames 0 --device cuda --train_iters 30000

# Seed + metrics only
bash scripts/run_on_nvidia.sh /path/to/Tanks/Francis \
  --max_frames 40 --skip_train --device cuda
```

Or call the Python entry directly:

```bash
python scripts/run_nvidia_report.py \
  --scene /path/to/Tanks/Francis \
  --out_dir ./eval_out/nvidia_report_Francis_manual \
  --max_frames 40 --device cuda \
  --pair_radius 2 --long_range_stride 10
```

### Useful flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--max_frames` | 40 | `0` = all |
| `--train_iters` | 7000 | Smoke; document full **30000** |
| `--skip_train` | off | Seed+eval only |
| `--skip_jump_stress` | off | Skip stride-5 depth_pnp vs hybrid |
| `--pair_radius` / `--long_range_stride` | 2 / 10 | LightGlue jump graph |
| `--feature_backend` | superpoint | `superpoint` \| `aliked` \| `auto` \| `sift` |
| `--copy_images` | off | Copy into COLMAP `images/` (else symlink) |
| `--device` | auto | `auto` \| `cuda` \| `cpu` |

## Outputs (`out_dir/`)

| File | Role |
| --- | --- |
| **`report.json`** | Nested schema v1 — **bring this back to the agent** |
| `REPORT.md` | Human tables + “How to send this back” |
| `run.log` | Full stdout/stderr tee |
| `colmap_export/` | FastGS-ready sparse (`-s` for `train.py`) |
| `seed_stats.json` / `ate.json` / `jump_stress.json` | Phase intermediates |
| `fastgs_model/` | Written when train runs |

### `report.json` schema (summary)

```json
{
  "schema_version": 1,
  "run_id": "...",
  "timestamp_utc": "...",
  "git": {"commit": "...", "branch": "...", "dirty": false},
  "env": {"hostname": "...", "cuda_available": true, "gpu_name": "...", ...},
  "config": {"max_frames": 40, "train_iters": 7000, ...},
  "phases": {
    "hybrid_seed": {"ok": true, "seconds": 12.3, "stats": {...}},
    "ate_eval": {"ok": true, "metrics": {"ate_rmse": 0.03, ...}},
    "jump_stress": {"ok": true, "depth_pnp_ate": ..., "hybrid_ate": ...},
    "fastgs_train": {"ok": false, "skipped": true, "reason": "..."}
  },
  "artifacts": {"colmap": "...", "report_md": "...", "logs": "..."},
  "errors": [],
  "warnings": [],
  "notes_for_agent": "Free-text heuristics for next knobs"
}
```

## Bring results back to Chief of Staff

1. Minimum: paste or attach **`report.json`**.
2. Better: zip the whole `out_dir` (includes `run.log` + COLMAP export):

```bash
cd eval_out
zip -r nvidia_report_Francis.zip nvidia_report_Francis_<timestamp>
```

3. Ask the agent to read `notes_for_agent` and adjust
   `--pair_radius`, `--long_range_stride`, `--lightglue_weight`, frame count,
   or train iters.

See also [HYBRID_SEEDER.md](./HYBRID_SEEDER.md).
