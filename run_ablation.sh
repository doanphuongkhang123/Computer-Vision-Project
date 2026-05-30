#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

FEATS_DIR="${1:-}"
ABLATION_CONFIG="${2:-configs/bracs_ablation.yaml}"
EXTRA_ARGS=("${@:3}")

if [ -z "$FEATS_DIR" ]; then
  if [ -d "./feats" ]; then
    FEATS_DIR="./feats"
  elif [ -d "../feats" ]; then
    FEATS_DIR="../feats"
  else
    echo "Usage: ./run_ablation.sh /path/to/feats [configs/bracs_ablation.yaml] [extra tune_bracs.py args]" >&2
    echo "Example: ./run_ablation.sh /data/bracs/feats configs/bracs_ablation.yaml --max-trials 2" >&2
    exit 2
  fi
fi

if [ ! -d ".venv" ]; then
  ./setup_env.sh
fi

source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

python prepare_splits.py \
  --feats-dir "$FEATS_DIR" \
  --metadata metadata/bracs_ftp_metadata.csv \
  --output-dir data \
  --strict

python tune_bracs.py --tuning-config "$ABLATION_CONFIG" "${EXTRA_ARGS[@]}"
