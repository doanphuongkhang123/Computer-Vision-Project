#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

FEATS_DIR="${1:-path/to/feats}"
CONFIG="${2:-configs/config.yaml}"

if [ ! -e "$FEATS_DIR" ]; then
  echo "Feature path not found: $FEATS_DIR" >&2
  echo "Usage: ./run.sh [path/to/feats] [configs/config.yaml]" >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [ ! -d ".venv" ]; then
  ./setup_env.sh
fi
source .venv/bin/activate

python prepare_splits.py \
  --feats-dir "$FEATS_DIR" \
  --metadata metadata/bracs_ftp_metadata.csv \
  --output-dir data \
  --strict

python train_baseline.py --config "$CONFIG"
