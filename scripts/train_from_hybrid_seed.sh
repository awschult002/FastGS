#!/usr/bin/env bash
# Hybrid DepthPnP+LightGlue seed → COLMAP export → optional FastGS train.py
#
# Usage:
#   bash scripts/train_from_hybrid_seed.sh <scene> <export_dir> [model_path] [-- seed args...]
#
# Examples:
#   bash scripts/train_from_hybrid_seed.sh /data/Tanks/Francis ./eval_out/francis_hybrid_colmap
#   bash scripts/train_from_hybrid_seed.sh /data/Tanks/Francis ./out/francis_h \
#       ./output/francis_hybrid_fastgs -- --max_frames 40 --device cuda
#
# On CPU-only boxes, seed+export still run; train.py is skipped with a clear message.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <scene> <export_dir> [model_path] [-- extra seed_poses_hybrid.py args...]"
  echo "Example: $0 /data/Tanks/Francis ./eval_out/francis_hybrid_colmap ./output/francis_hybrid"
  exit 1
fi

SCENE="$1"
EXPORT="$2"
shift 2

MODEL=""
if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  MODEL="$1"
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

if [[ -z "$MODEL" ]]; then
  base="$(basename "$(dirname "$EXPORT")")_$(basename "$EXPORT")"
  MODEL="output/hybrid_$(basename "${SCENE}")"
fi

echo "[train_from_hybrid_seed] scene=$SCENE"
echo "[train_from_hybrid_seed] export=$EXPORT"
echo "[train_from_hybrid_seed] model=$MODEL"

"$PY" scripts/seed_poses_hybrid.py \
  --scene "$SCENE" \
  --out_colmap "$EXPORT" \
  --model_hint "$MODEL" \
  "$@"

echo
echo "========== FastGS train command =========="
echo "python train.py -s ${EXPORT} -m ${MODEL}"
echo "=========================================="
echo

# Only invoke train.py when CUDA is available (FastGS needs a GPU).
HAS_CUDA="$("$PY" - <<'PY'
try:
    import torch
    print("1" if torch.cuda.is_available() else "0")
except Exception:
    print("0")
PY
)"

if [[ "$HAS_CUDA" != "1" ]]; then
  echo "[train_from_hybrid_seed] CUDA not available — skipping train.py."
  echo "  Seed + COLMAP export are ready at: $EXPORT"
  echo "  On your NVIDIA box, run:"
  echo "    python train.py -s ${EXPORT} -m ${MODEL}"
  exit 0
fi

echo "[train_from_hybrid_seed] CUDA detected — launching train.py"
exec "$PY" train.py -s "$EXPORT" -m "$MODEL"
