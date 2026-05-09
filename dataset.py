
from pathlib import Path
from typing import Tuple, Dict, List
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CFG


def load_area_csvs(state: str, data_dir: str) -> pd.DataFrame:
    base_dir = Path(data_dir)
    frames = []
    for fname in CFG["AREAS"][state]:
        fpath = base_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Cannot find {fname}. Put CSVs in {base_dir}"
            )
        df = pd.read_csv(fpath)
        df["_source"] = fname
        frames.append(df)
    df_out = pd.concat(frames, ignore_index=True)
    return df_out


BAND_ORDER = ["Blue", "Green", "Red", "RE1", "RE2", "RE3",
              "NIR", "RE4", "SWIR1", "SWIR2"]


def parse_features(
    df: pd.DataFrame,
    state: str,
    cfg: dict = CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    label_col = None
    for candidate in ["label", "crop_type", "class", "cdl", "CDL", "Label", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError("No label column found in CSV.")
    
    raw_labels = df[label_col].fillna(0).astype(int)
    band_name_pattern = re.compile(
        r'^(Blue|Green|Red|RE1|RE2|RE3|RE4|NIR|SWIR1|SWIR2)_(\d{1,2})$'
    )
    mask_pattern = re.compile(r'^mask_(\d{1,2})$')

    spectral_map = {}
    mask_map = {}
    for col in df.columns:
        if col in {label_col, '.geo', 'system:index', 'area', 'year', '_source'}:
            continue
        m1 = band_name_pattern.match(col)
        if m1:
            band, t = m1.group(1), int(m1.group(2))
            spectral_map.setdefault(band, {})[t] = col
            continue
        m2 = mask_pattern.match(col)
        if m2:
            mask_map[int(m2.group(1))] = col
            continue

    if not spectral_map:
        raise ValueError("No spectral columns found in CSV.")

    times = sorted({t for band in spectral_map for t in spectral_map[band]})
    nt = len(times)

    found_bands = []
    band_cols = []
    for band in BAND_ORDER:
        if band not in spectral_map:
            continue
        found_bands.append(band)
        for t in times:
            col = spectral_map[band].get(t)
            if col is not None:
                band_cols.append(col)

    n_bands_found = len(found_bands)
    n_samples = len(df)
    
    raw = df[band_cols].values.astype(np.float32)
    X = raw.reshape(n_samples, n_bands_found, nt)
    X = X.transpose(0, 2, 1)  # (N, T, C)

   
    if mask_map:
        mask_cols = [mask_map.get(t) for t in times]
        if any(c is None for c in mask_cols):
            loaded_mask = np.zeros((n_samples, nt), dtype=np.float32)
            for i, col in enumerate(mask_cols):
                if col is not None:
                    loaded_mask[:, i] = df[col].astype(np.float32).values
        else:
            loaded_mask = df[mask_cols].astype(np.float32).values

        if loaded_mask.mean() > 0.5:
            mask = 1.0 - loaded_mask
        else:
            mask = loaded_mask
    else:
        mask = (X == 0).all(axis=2).astype(np.float32)
    cov_features, n_cov = get_covariate_features(cfg)
    covariates = np.zeros((n_samples, n_cov), dtype=np.float32)
    
    for i, feat in enumerate(cov_features):
        if feat in df.columns:
            covariates[:, i] = df[feat].astype(np.float32).values
        else:
            print(f"      WARNING: Covariate '{feat}' not found in data, using zeros.")


    if state == "Arkansas":
        remap = {1: 0, 2: 1, 3: 2, 4: 3, 99: 4}
        n_classes = 5
    else:
        remap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 99: 5}
        n_classes = 6

    labels = np.array(
        [remap.get(int(code), n_classes - 1) for code in raw_labels],
        dtype=np.int64,
    )

    valid_mask = raw_labels != 0
    X = X[valid_mask]
    mask = mask[valid_mask]
    covariates = covariates[valid_mask]
    labels = labels[valid_mask]

    return X, mask, covariates, labels, cov_features


def get_covariate_features(cfg, subset=None):
    from config import get_covariate_features as _get_cov
    return _get_cov(cfg, subset)


def normalise(X: np.ndarray) -> np.ndarray:
    X_norm = X.copy()
    for b in range(X.shape[2]):
        band = X[:, :, b]
        valid = band[band != 0]
        if len(valid) == 0:
            continue
        mn, mx = valid.min(), valid.max()
        if mx > mn:
            X_norm[:, :, b] = np.where(
                band == 0, 0.0,
                (band - mn) / (mx - mn + 1e-8)
            )
    return X_norm


def normalise_covariates(covariates: np.ndarray) -> np.ndarray:
    cov_norm = covariates.copy()
    for i in range(covariates.shape[1]):
        col = covariates[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            mean = valid.mean()
            std = valid.std() + 1e-8
            cov_norm[:, i] = (col - mean) / std
    return cov_norm


def stratified_split(
    X: np.ndarray,
    mask: np.ndarray,
    covariates: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 300,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    n_train = int(n_per_class * train_ratio)
    n_val = n_per_class - n_train

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)

        if len(idx) > n_per_class:
            tv = idx[:n_per_class]
            rest = idx[n_per_class:]
            train_idx.extend(tv[:n_train])
            val_idx.extend(tv[n_train:])
            test_idx.extend(rest)
        else:
            n_tv = min(len(idx), n_per_class)
            n_tr = int(n_tv * train_ratio)
            train_idx.extend(idx[:n_tr])
            val_idx.extend(idx[n_tr:n_tv])
            test_idx.extend(idx[n_tv:])

    def subset(indices):
        i = np.array(indices, dtype=int)
        return X[i], mask[i], covariates[i], labels[i]

    return subset(train_idx), subset(val_idx), subset(test_idx)


class CropDataset(Dataset):
    def __init__(self, X, mask, covariates, labels, use_covariates=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.covariates = torch.tensor(covariates, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.use_covariates = use_covariates

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_covariates:
            return self.X[idx], self.mask[idx], self.covariates[idx], self.labels[idx]
        else:
            return self.X[idx], self.mask[idx], self.labels[idx]