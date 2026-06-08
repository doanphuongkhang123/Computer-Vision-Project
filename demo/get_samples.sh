#!/usr/bin/env bash
# Download class-representative BRACS slides on the server (direct FTP, fast).
# Run from the repo root: bash demo/get_samples.sh
set -e
FTP="ftp://histoimage.na.icar.cnr.it/BRACS_WSI"

dl() { [ -f "$2" ] || wget -O "$2" "$FTP/$1"; }

# class : path-on-ftp : local-name (names match SAMPLES in demo/app.py)
dl "train/Group_BT/Type_N/BRACS_1003718.svs"   "BRACS_1003718.svs"            # Benign  (~68MB)
dl "test/Group_AT/Type_ADH/BRACS_1003694.svs"  "BRACS_1003694_Atypical.svs"   # Atypical (~620MB)
dl "train/Group_MT/Type_IC/BRACS_1003677.svs"  "BRACS_1003677_Malignant.svs"  # Malignant(~181MB)

echo "Done. Slides ready in $(pwd)"
