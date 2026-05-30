#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

FEATS_DIR="${1:-}"
CONFIG="${2:-configs/bracs_server.yaml}"

if [ -z "$FEATS_DIR" ]; then
  if [ -d "./feats" ]; then
    FEATS_DIR="./feats"
  elif [ -d "../feats" ]; then
    FEATS_DIR="../feats"
  else
    echo "Usage: ./run_all.sh /path/to/feats [configs/bracs_server.yaml]" >&2
    echo "Could not auto-find ./feats or ../feats." >&2
    exit 2
  fi
fi

if [ ! -e "$FEATS_DIR" ]; then
  echo "Feature path not found: $FEATS_DIR" >&2
  exit 2
fi

if [ ! -d ".venv" ]; then
  ./setup_env.sh
fi

source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# python prepare_splits.py \
#   --feats-dir "$FEATS_DIR" \
#   --metadata metadata/bracs_ftp_metadata.csv \
#   --output-dir data \
#   --strict

python prepare_splits.py \
  --feats-dir "$FEATS_DIR" \
  --metadata metadata/bracs_ftp_metadata.csv \
  --output-dir data \
  --strict

python train_bracs.py --config "$CONFIG"
