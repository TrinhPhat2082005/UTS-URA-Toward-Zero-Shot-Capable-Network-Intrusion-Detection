"""Du lieu Protocol A: 560K benign train (chung cho 14 fold), val va test zero-shot cua tung fold."""
from functools import lru_cache

import pandas as pd

from . import paths


def folds():
    """14 fold leave-one-attack-out, ten fold = loai tan cong bi giu lai."""
    return sorted(p.name for p in paths.SPLITS.iterdir() if p.is_dir())


def read_split(path):
    """(X float32 [n, 78], y, ten 78 feature) cua mot file CSV split."""
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in ("Label", "Label_Text")]
    return df[feats].to_numpy("float32"), df["Label"].to_numpy(), feats


@lru_cache(maxsize=1)
def benign_train():
    """560K flow benign dung de dat nguong tau (giong nhau o moi fold)."""
    X, y, feats = read_split(paths.SPLITS_ROOT / "train_benign_only.csv")
    assert (y == 0).all() and len(feats) == 78
    return X


@lru_cache(maxsize=2)
def fold_data(fold):
    """(X_val, y_val, X_test) cua mot fold: val = benign + 13 loai da biet, test = loai bi giu lai."""
    Xv, yv, _ = read_split(paths.SPLITS / fold / "val_binary.csv")
    Xt, yt, _ = read_split(paths.SPLITS / fold / "test_zeroshot.csv")
    assert (yt == 1).all()
    return Xv, yv, Xt
