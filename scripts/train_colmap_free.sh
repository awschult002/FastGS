#!/usr/bin/env bash
# COLMAP-free FastGS helper (CF-3DGS-inspired pose solve → optional train.py)
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <source_images_dir> <export_dir> [extra train_colmap_free.py args...]"
  echo "Example: $0 ./datasets/custom_seq ./datasets/custom_seq_solved --run_fastgs -m ./output/custom"
  exit 1
fi

SRC="$1"
EXPORT="$2"
shift 2

python train_colmap_free.py -s "$SRC" --export_dir "$EXPORT" "$@"
