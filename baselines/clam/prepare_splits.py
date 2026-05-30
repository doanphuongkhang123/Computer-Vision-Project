import argparse
import json
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "slide_id",
    "file_name",
    "feat_path",
    "path",
    "coords_path",
    "label",
    "label_idx",
    "split",
    "patient_id",
    "source_name",
    "n_patches",
    "patch_size",
    "mag_level",
    "url",
    "group",
    "type",
]


def read_n_patches(coords_path):
    if not coords_path.exists():
        return ""
    try:
        with open(coords_path) as f:
            return json.load(f).get("n_patches", "")
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Create BRACS train/val/test CSVs from a feats directory.")
    parser.add_argument("--feats-dir", required=True, help="Directory containing BRACS_<id>.pt feature tensors.")
    parser.add_argument("--metadata", default="metadata/bracs_ftp_metadata.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--strict", action="store_true", help="Fail if any .pt feature lacks metadata.")
    args = parser.parse_args()

    feats_dir = Path(args.feats_dir).expanduser().resolve()
    if not feats_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feats_dir}")

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    for feat_path in sorted(feats_dir.rglob("*.pt")):
        slide_id = feat_path.stem
        coords_path = feats_dir / f"{slide_id}_coords.json"
        feature_rows.append(
            {
                "slide_id": slide_id,
                "feat_path": str(feat_path),
                "path": str(feat_path),
                "coords_path": str(coords_path) if coords_path.exists() else "",
                "n_patches": read_n_patches(coords_path),
            }
        )

    if not feature_rows:
        raise ValueError(f"No .pt feature tensors found in {feats_dir}")

    feats = pd.DataFrame(feature_rows)
    meta = pd.read_csv(metadata_path)
    df = feats.merge(meta, on="slide_id", how="left", validate="one_to_one")

    missing = df[df["label"].isna()]["slide_id"].tolist()
    if missing:
        message = f"{len(missing)} feature files have no BRACS metadata: {missing[:20]}"
        if args.strict:
            raise ValueError(message)
        print(f"[WARN] {message}. Dropping them.")
        df = df[~df["label"].isna()].copy()

    if df.empty:
        raise ValueError("No labeled BRACS features remain after metadata join.")

    df["label_idx"] = df["label_idx"].astype(int)
    df["patient_id"] = ""
    df["source_name"] = "BRACS"
    df["patch_size"] = 224
    df["mag_level"] = 0
    df = df[OUTPUT_COLUMNS]

    all_path = out_dir / "all.csv"
    df.to_csv(all_path, index=False)
    print(f"Wrote {len(df)} rows: {all_path}")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()
        split_path = out_dir / f"{split}.csv"
        split_df.to_csv(split_path, index=False)
        counts = split_df.groupby(["label", "label_idx"]).size().to_dict() if not split_df.empty else {}
        print(f"Wrote {len(split_df)} rows: {split_path} counts={counts}")


if __name__ == "__main__":
    main()
