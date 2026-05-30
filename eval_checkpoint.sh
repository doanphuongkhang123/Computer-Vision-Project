#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT="${1:-results/bracs_full/checkpoints/best.pth}"
CONFIG="${2:-configs/bracs_server.yaml}"

source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
python train_bracs.py --config "$CONFIG" --eval-only --checkpoint "$CHECKPOINT"
