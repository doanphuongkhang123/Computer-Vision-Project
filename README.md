# BRACS WSI Classification

Training code for BRACS whole-slide image classification from pre-extracted
patch embeddings. The main model uses learnable prototype retrieval,
cross-attention aggregation, and a transformer encoder. Four MIL baselines are
included under `baselines/`.

## Repository Layout

```text
configs/                 Main model, tuning, and ablation configs
models/                  Main prototype cross-attention transformer
utils/                   BRACS feature-bag dataset loader
baselines/               Self-contained baseline implementations
metadata/                BRACS metadata and official split mapping
prepare_splits.py        Build train/val/test CSVs from feature tensors
train_bracs.py           Main model training and evaluation
tune_bracs.py            Sequential tuning/ablation runner
```

Baselines:

```text
baselines/abmil
baselines/acmil
baselines/attrimil
baselines/clam
```

## Data Format

The code expects one feature tensor per WSI:

```text
path/to/feats/
  BRACS_1003678.pt          # [num_patches, 1536]
  BRACS_1003678_coords.json # optional
```

HDF5 feature stores are also supported by the dataset loader. CSV splits are
generated automatically from `metadata/bracs_ftp_metadata.csv`.

## Setup

```bash
cd bracs_server_package
./setup_env.sh
source .venv/bin/activate
```

The setup script uses `uv` and does not require sudo.

## Train Main Model

```bash
./run_all.sh /path/to/feats
```

Equivalent explicit commands:

```bash
python prepare_splits.py \
  --feats-dir /path/to/feats \
  --metadata metadata/bracs_ftp_metadata.csv \
  --output-dir data \
  --strict

python train_bracs.py --config configs/bracs_server.yaml
```

Use the low-memory config if needed:

```bash
./run_all.sh /path/to/feats configs/bracs_safe.yaml
```

## Smoke Test

```bash
./run_smoke.sh /path/to/feats
```

## Evaluate Checkpoint

```bash
./eval_checkpoint.sh results/bracs_full_stable/checkpoints/best.pth configs/bracs_server.yaml
```

Evaluation writes predictions to:

```text
<results_dir>/predictions/test_predictions.csv
```

## Tuning

```bash
./run_tuning.sh /path/to/feats configs/bracs_tuning.yaml
```

Useful options:

```bash
./run_tuning.sh /path/to/feats configs/bracs_tuning.yaml --max-trials 3
./run_tuning.sh /path/to/feats configs/bracs_tuning.yaml --trial <trial_name>
```

The summary is written under the tuning `output_dir`, for example:

```text
results/tuning_stage1/tuning_summary.csv
```

## Ablation

Core ablations for `max_retrieved`, `ratio`, `cls_ratio`, and
`normalize_features`:

```bash
./run_ablation.sh /path/to/feats configs/bracs_ablation.yaml
```

Summary:

```text
results/ablation_core/ablation_summary.csv
```

## Run Baselines

Each baseline is self-contained and must be run from inside its own folder.
The default feature path is `path/to/feats`.

```bash
cd baselines/abmil
./run.sh /path/to/feats
```

Other baselines:

```bash
cd baselines/acmil && ./run.sh /path/to/feats
cd baselines/attrimil && ./run.sh /path/to/feats
cd baselines/clam && ./run.sh /path/to/feats
```

Each baseline writes local outputs to:

```text
baselines/<name>/results/
```

## Outputs

Main training writes:

```text
results/<run_name>/metrics.csv
results/<run_name>/checkpoints/best.pth
results/<run_name>/checkpoints/best_f1.pth
results/<run_name>/checkpoints/best_acc.pth
results/<run_name>/checkpoints/best_loss.pth
results/<run_name>/predictions/test_predictions.csv
```

## Configuration

Primary config:

```text
configs/bracs_server.yaml
```

Important fields:

- `max_patches`, `eval_max_patches`: cap WSI bag size, or `null` for full WSI.
- `ratio`: prototype retrieval ratio.
- `max_retrieved`: maximum retrieved prototype tokens.
- `cls_ratio`: CLS/mean fusion weight.
- `normalize_features`: L2-normalize patch embeddings.
- `selection_metric`: checkpoint selection metric.

## Weights & Biases

Set in config:

```yaml
wandb: true
wandb_project: bracs_wsi_classification
wandb_run_name: bracs_full_stable
```

Authenticate with either:

```bash
export WANDB_API_KEY="..."
```

or:

```bash
echo "..." > wandb_api_key.txt
chmod 600 wandb_api_key.txt
```
