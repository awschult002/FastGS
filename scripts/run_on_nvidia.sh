#!/usr/bin/env bash
# Thin wrapper: timestamped out_dir + run_nvidia_report.py
#
# Usage:
#   bash scripts/run_on_nvidia.sh /path/to/Tanks/Francis [extra args...]
#
# Example:
#   bash scripts/run_on_nvidia.sh /data/Tanks/Francis --max_frames 40 --device cuda
#   bash scripts/run_on_nvidia.sh /data/Tanks/Francis --max_frames 8 --skip_train --skip_jump_stress
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scene_path> [extra run_nvidia_report.py args...]"
  echo "Example: $0 /path/to/Tanks/Francis --max_frames 40 --device cuda"
  exit 1
fi

SCENE="$1"
shift

SCENE_NAME="$(basename "$SCENE")"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/eval_out/nvidia_report_${SCENE_NAME}_${TS}"
mkdir -p "$OUT_DIR"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

echo "[run_on_nvidia] scene=$SCENE"
echo "[run_on_nvidia] out_dir=$OUT_DIR"
echo "[run_on_nvidia] python=$PY"

# --out_dir last so wrapper timestamped path wins over any user --out_dir
"$PY" scripts/run_nvidia_report.py \
  --scene "$SCENE" \
  "$@" \
  --out_dir "$OUT_DIR"

REPORT_JSON="${OUT_DIR}/report.json"
echo
echo "========== report.json =========="
echo "$REPORT_JSON"
echo "================================="
if [[ -f "$REPORT_JSON" ]]; then
  echo "Bring this file (or zip of out_dir) back to Chief of Staff for tuning."
else
  echo "WARNING: report.json missing — check run.log under $OUT_DIR"
  exit 1
fi
