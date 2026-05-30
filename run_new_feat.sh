#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

FEATS_DIR="${1:-/mnt/disk4/khangdp/new_feats}"
ENCODER="${2:-natural}"
MODE="${3:-full}"

case "$ENCODER" in
  natural|natural_supervised)
    H5_FILE="$FEATS_DIR/bracs_patch_feats_pretrain_natural_supervised_x20.h5"
    FULL_CONFIG="configs/bracs_new_feat_natural.yaml"
    SMOKE_CONFIG="configs/bracs_new_feat_natural_smoke.yaml"
    ;;
  medical|medical_ssl)
    H5_FILE="$FEATS_DIR/bracs_patch_feats_pretrain_medical_ssl_x20.h5"
    FULL_CONFIG="configs/bracs_new_feat_medical_ssl.yaml"
    SMOKE_CONFIG=""
    ;;
  path_clip|path-clip)
    H5_FILE="$FEATS_DIR/patch_feats_pretrain_path-clip-L-336.h5"
    FULL_CONFIG="configs/bracs_new_feat_path_clip.yaml"
    SMOKE_CONFIG=""
    ;;
  *)
    echo "Unknown encoder: $ENCODER" >&2
    echo "Use one of: natural, medical_ssl, path_clip" >&2
    exit 2
    ;;
esac

if [ ! -f "$H5_FILE" ]; then
  echo "HDF5 feature file not found: $H5_FILE" >&2
  exit 2
fi

if [ "$MODE" = "smoke" ]; then
  if [ -z "$SMOKE_CONFIG" ]; then
    echo "Smoke config is currently defined for natural only. Use MODE=full or add a matching smoke config." >&2
    exit 2
  fi
  ./run_smoke.sh "$H5_FILE" "$SMOKE_CONFIG"
else
  ./run_all.sh "$H5_FILE" "$FULL_CONFIG"
fi
