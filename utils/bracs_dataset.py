import os
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


FEATURE_PATH_COLUMNS = ("feat_path", "feature_path", "features_path", "path", "wsi_feature_path")
H5_FEATURE_KEYS = ("features", "feats", "feat", "embeddings", "x")


def _first_present(row, columns):
    for col in columns:
        if col in row and row[col] not in ("", None):
            value = row[col]
            if isinstance(value, float) and value != value:
                continue
            return str(value)
    return ""


def _load_tensor(path):
    if "::" in str(path):
        return _load_h5_tensor(path)

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        for key in ("features", "feats", "embeddings", "x"):
            if key in data:
                data = data[key]
                break
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    if data.dim() != 2:
        raise ValueError(f"Expected a [num_patches, feature_dim] tensor in {path}, got shape {tuple(data.shape)}")
    return data.float()


def _dataset_to_tensor(data, path):
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    if data.dim() != 2:
        raise ValueError(f"Expected a [num_patches, feature_dim] tensor in {path}, got shape {tuple(data.shape)}")
    return data.float()


def _load_h5_tensor(path):
    h5_path, key = str(path).split("::", 1)
    key = key.strip("/")
    with h5py.File(h5_path, "r") as h5:
        if key not in h5:
            raise KeyError(f"HDF5 key '{key}' not found in {h5_path}")
        obj = h5[key]
        if isinstance(obj, h5py.Group):
            for feature_key in H5_FEATURE_KEYS:
                if feature_key in obj:
                    obj = obj[feature_key]
                    break
            else:
                datasets = [name for name, item in obj.items() if isinstance(item, h5py.Dataset)]
                if len(datasets) == 1:
                    obj = obj[datasets[0]]
                else:
                    raise KeyError(
                        f"HDF5 group '{key}' in {h5_path} does not contain one of {H5_FEATURE_KEYS}"
                    )
        data = obj[()]
    return _dataset_to_tensor(data, path)


def _optional_positive_int(value):
    if value in (None, "", "null", "None"):
        return None
    value = int(value)
    return value if value > 0 else None


class BracsWSIFeatureDataset(Dataset):
    """BRACS WSI feature-bag dataset.

    Each row must contain at least slide_id and label_idx. Feature paths can be
    provided by feat_path/path, or resolved as <feats_dir>/<slide_id>.pt.
    """

    def __init__(
        self,
        dataframe,
        feats_dir="",
        max_patches=None,
        sampling="random",
        training=True,
        normalize=False,
        feature_dim=1536,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.feats_dir = Path(feats_dir).expanduser() if feats_dir else None
        self.max_patches = _optional_positive_int(max_patches)
        self.sampling = sampling
        self.training = training
        self.normalize = normalize
        self.feature_dim = int(feature_dim)

        required = {"slide_id", "label_idx"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"BRACS dataframe is missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def _resolve_feature_path(self, row):
        slide_id = str(row["slide_id"])
        candidate = _first_present(row, FEATURE_PATH_COLUMNS)

        if "::" in candidate:
            h5_path, _ = candidate.split("::", 1)
            if Path(h5_path).expanduser().exists():
                return candidate

        if candidate and candidate.endswith(".pt") and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())

        if self.feats_dir is not None:
            direct = self.feats_dir / f"{slide_id}.pt"
            if direct.exists():
                return str(direct)

            matches = list(self.feats_dir.rglob(f"{slide_id}.pt"))
            if matches:
                return str(matches[0])

        if candidate and candidate.endswith(".pt"):
            return candidate
        return str(self.feats_dir / f"{slide_id}.pt") if self.feats_dir is not None else f"{slide_id}.pt"

    def _select_indices(self, n_patches):
        if self.max_patches is None or n_patches <= self.max_patches:
            return None

        if self.sampling == "first":
            return torch.arange(self.max_patches)
        if self.sampling == "uniform" or not self.training:
            return torch.linspace(0, n_patches - 1, self.max_patches).round().long()
        if self.sampling == "random":
            return torch.randperm(n_patches)[: self.max_patches].sort().values
        raise ValueError(f"Unknown WSI patch sampling mode: {self.sampling}")

    def __getitem__(self, index):
        row = self.df.iloc[index].to_dict()
        label = int(row["label_idx"])
        feat_path = self._resolve_feature_path(row)

        exists_path = feat_path.split("::", 1)[0] if "::" in feat_path else feat_path
        if not os.path.exists(exists_path):
            raise FileNotFoundError(f"Feature tensor not found for slide {row['slide_id']}: {feat_path}")

        features = _load_tensor(feat_path)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch for {feat_path}: expected {self.feature_dim}, got {features.shape[1]}"
            )

        idx = self._select_indices(features.shape[0])
        if idx is not None:
            features = features[idx]
        if self.normalize:
            features = F.normalize(features, dim=-1)

        length = features.shape[0]
        return features, label, length


def collate_wsi_features(batch):
    feats, labels, lengths = zip(*batch)
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    max_len = feats.shape[1]
    key_padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    return feats, labels, key_padding_mask
