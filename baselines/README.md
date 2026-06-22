# BRACS Baselines

Each baseline is self-contained and can be run from inside its own folder.

```bash
cd baselines/abmil
./run.sh path/to/feats
```

The default feature path is `path/to/feats`, so this also works after replacing
that placeholder with a real directory:

```bash
./run.sh
```

Available baselines:

- `abmil`
- `acmil`
- `attrimil`
- `clam`

## Pretrained checkpoints

Each baseline includes checkpoints selected by validation metric under
`baselines/<baseline>/checkpoints/`:

- `best_f1.pth`: highest validation macro F1 (recommended)
- `best_acc.pth`: highest validation accuracy
- `best_loss.pth`: lowest validation loss
- `best.pth`: checkpoint selected by the configured primary metric

The checkpoints expect 1536-dimensional precomputed patch features and predict
the three BRACS classes: Benign, Atypical, and Malignant. To evaluate one:

```bash
cd baselines/abmil
python prepare_splits.py \
  --feats-dir /path/to/feats \
  --metadata metadata/bracs_ftp_metadata.csv \
  --output-dir data \
  --strict
python train_baseline.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_f1.pth \
  --eval-only
```
