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
