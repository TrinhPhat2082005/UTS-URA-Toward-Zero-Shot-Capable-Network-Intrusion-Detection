"""Step 1 - score every baseline checkpoint on the Protocol A data.

For each model and each of the 14 folds, store the attack scores of
    tb  the 560K benign training flows (train_benign_only.csv)   -> threshold calibration
    vb  benign flows of val_binary.csv                           -> realized FPR
    va  attack flows of val_binary.csv (13 known types)          -> known-attack recall
    zs  test_zeroshot.csv (the withheld type only)               -> zero-shot accuracy
in OUT_BASE/scores_<model>.npz, keys "<model>|<fold>|<part>" (same layout as
rq1_rq2_layer1/sweep_all_scores.npz). Models whose checkpoints are missing are skipped.

Usage:  python 01_score_baselines.py [model ...]      (default: every available model)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from baselines import REGISTRY, available, folds  # noqa: E402

OUT = paths.ensure(paths.OUT_BASE)


def read(path):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in ("Label", "Label_Text")]
    return df[feats].to_numpy("float32"), df["Label"].to_numpy(), feats


def main(names):
    X_tb, y_tb, feats = read(paths.SPLITS_ROOT / "train_benign_only.csv")
    assert (y_tb == 0).all() and len(feats) == 78
    data = {}
    for name in names:
        spec = REGISTRY[name]
        if not available(name):
            print(f"[skip] {name}: checkpoints not found under {paths.BASE_ART / name}", flush=True)
            continue
        t0, S, shared_tb = time.time(), {}, None
        try:
            for fold in folds():
                if fold not in data:
                    Xv, yv, fv = read(paths.SPLITS / fold / "val_binary.csv")
                    Xt, yt, ft = read(paths.SPLITS / fold / "test_zeroshot.csv")
                    assert fv == feats and ft == feats and (yt == 1).all()
                    data[fold] = (Xv, yv, Xt)
                Xv, yv, Xt = data[fold]
                score = spec.loader(fold)
                if spec.per_fold or shared_tb is None:
                    tb = np.asarray(score(X_tb), dtype=np.float32)
                    shared_tb = None if spec.per_fold else tb
                else:
                    tb = shared_tb
                sv = np.asarray(score(Xv), dtype=np.float32)
                S[f"{name}|{fold}|tb"] = tb
                S[f"{name}|{fold}|vb"] = sv[yv == 0]
                S[f"{name}|{fold}|va"] = sv[yv == 1]
                S[f"{name}|{fold}|zs"] = np.asarray(score(Xt), dtype=np.float32)
                print(f"  {name:14s} {fold:26s} {time.time() - t0:6.0f}s", flush=True)
        except NotImplementedError as e:
            print(f"[skip] {e}", flush=True)
            continue
        np.savez_compressed(OUT / f"scores_{name}.npz", **S)
        print(f"-> {OUT / f'scores_{name}.npz'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(REGISTRY))
