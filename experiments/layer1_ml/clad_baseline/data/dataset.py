"""
Data loading for CIC-IDS-2018 pre-split CSVs.

Split convention (same as the rest of this repo):
  train_isolation_benign.csv  — benign only  → (kept for reference / one-class baselines)
  train_xgboost_binary.csv    — benign+attack → train CLAD encoder (REQUIRED by CLAD)
  val_binary.csv              — benign+attack → tune threshold
  test_binary.csv             — benign+attack → final evaluation

IMPORTANT (paper fidelity): CLAD learns the density ratio P(x|y=0)/P(x|y≠0)
(Sec. III-B). It MUST be trained on benign + known-attack samples — the attacks
form the negative pairs that normalise the benign distribution. Training on
benign-only collapses CLAD into plain compactness (Deep SVDD). So the CLAD
training set is `train_xgboost_binary.csv`, NOT `train_isolation_benign.csv`.

Z-normalisation is still fit on benign-only rows of the training set so the
attack distribution never leaks into the scaler.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 78 CICFlowMeter features (canonical list, same as svdd_xgboost_ids.py)
FEATURES = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
    'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]


class ZNormaliser:
    """Fit on training data, transform any split."""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=1)
        # avoid div-by-zero for constant columns
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NIDSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_splits(data_dir: str, train_file: str = 'train_xgboost_binary.csv'):
    """
    Read the pre-split CSVs and return normalised numpy arrays.

    The CLAD training set is benign + known attack (`train_xgboost_binary.csv`),
    as required by the CLAD density-ratio loss. The scaler is fit on benign rows
    of the training set only, so the attack distribution never leaks.

    Returns
    -------
    X_train, y_train : training set (0=benign, 1=attack)
    X_val,   y_val   : val set (0=benign, 1=attack)
    X_test,  y_test  : test set (0=benign, 1=attack)
    normaliser       : fitted ZNormaliser (save if you need to reload)
    """
    dtype = {f: 'float32' for f in FEATURES}
    dtype['Label'] = 'int8'

    train_df = pd.read_csv(
        f"{data_dir}/{train_file}",
        usecols=FEATURES + ['Label'], dtype=dtype
    )
    val_df = pd.read_csv(
        f"{data_dir}/val_binary.csv",
        usecols=FEATURES + ['Label'], dtype=dtype
    )
    test_df = pd.read_csv(
        f"{data_dir}/test_binary.csv",
        usecols=FEATURES + ['Label'], dtype=dtype
    )

    X_train = train_df[FEATURES].values
    y_train = train_df['Label'].values.astype(int)
    X_val   = val_df[FEATURES].values
    y_val   = val_df['Label'].values.astype(int)
    X_test  = test_df[FEATURES].values
    y_test  = test_df['Label'].values.astype(int)

    # Replace inf / -inf with NaN then fill with benign-train column mean
    for X in (X_train, X_val, X_test):
        X[~np.isfinite(X)] = np.nan
    benign_train = X_train[y_train == 0]
    col_means = np.nanmean(benign_train, axis=0)
    for X in (X_train, X_val, X_test):
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Fit scaler on benign rows of the training set only (no attack leakage)
    normaliser = ZNormaliser().fit(X_train[y_train == 0])
    X_train = normaliser.transform(X_train)
    X_val   = normaliser.transform(X_val)
    X_test  = normaliser.transform(X_test)

    print(f"Train (0={(y_train==0).sum():,}, 1={(y_train==1).sum():,})")
    print(f"Val   (0={( y_val==0).sum():,}, 1={(y_val==1).sum():,})")
    print(f"Test  (0={(y_test==0).sum():,}, 1={(y_test==1).sum():,})")

    return X_train, y_train, X_val, y_val, X_test, y_test, normaliser


def make_train_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 4096):
    """Class-balanced DataLoader — oversamples so each batch sees ~equal benign:attack.

    This is essential for CLAD: every batch needs both benign anchors AND attack
    negatives, otherwise the density-ratio (repulsion) term is never active.
    Mirrors the reference repo's `balanced=True` sampling.
    """
    dataset = NIDSDataset(X, y)
    counts  = np.bincount(y)
    weights = 1.0 / counts[y]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      drop_last=True, num_workers=0, pin_memory=True)


def make_eval_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 4096):
    dataset = NIDSDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)
