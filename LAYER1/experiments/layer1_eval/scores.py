"""Diem cua tung model tren du lieu Protocol A, co cache.

Moi model co 4 phan diem cho moi fold, luu voi khoa "<key>|<fold>|<phan>":
    tb  560K benign train        -> dat nguong tau
    vb  benign cua val_binary    -> FPR thuc te
    va  attack cua val_binary    -> recall / F1 loai da biet
    zs  test_zeroshot            -> ZS-Acc
Cache: model cua paper           -> OUT_L1/sweep_all_scores.npz (mot file chung, RQ3b cua LAYER2 cung doc file nay)
       model cua baseline study  -> OUT_BASE/scores_<key>.npz
Diem cua baseline study duoc ep float32, diem cua paper giu kieu goc, dung nhu luc tao cache: doi kieu co the
lam lech cac flow nam dung tren nguong, va bang se khong con khop tung chu so.
"""
import hashlib
import time

import numpy as np

from . import data, paths, scorers

PAPER, BASE = "paper", "baseline-study"
PAPER_FILE = paths.OUT_L1 / "sweep_all_scores.npz"


def cache_file(key, source):
    return PAPER_FILE if source == PAPER else paths.OUT_BASE / f"scores_{key}.npz"


def load(key, source):
    """Diem da cache cua model, hoac None neu chua co."""
    f = cache_file(key, source)
    if not f.exists():
        return None
    with np.load(f) as z:
        S = {k: z[k] for k in z.files if k.split("|")[0] == key}
    return S or None


def save(key, source, S):
    f = cache_file(key, source)
    paths.ensure(f.parent)
    if source == PAPER and f.exists():                  # file chung: giu diem cua cac model khac
        with np.load(f) as z:
            S = {**{k: z[k] for k in z.files if k.split("|")[0] != key}, **S}
    np.savez_compressed(f, **S)


def compute(key, source):
    """Cham lai tu checkpoint. Model theo fold phai cham 560K benign o moi fold, nen mat vai phut den vai chuc phut."""
    if key == "optiforest":
        return _compute_optiforest(source)
    spec = scorers.MODELS[key]
    cast = (lambda a: np.asarray(a, dtype=np.float32)) if source == BASE else np.asarray
    X_tb = data.benign_train()
    S, shared_tb, t0 = {}, None, time.time()
    for fold in data.folds():
        Xv, yv, Xt = data.fold_data(fold)
        score = scorers.loader(key, source, fold)
        if spec.per_fold or shared_tb is None:
            tb = cast(score(X_tb))
            shared_tb = None if spec.per_fold else tb
        else:
            tb = shared_tb
        sv = cast(score(Xv))
        S[f"{key}|{fold}|tb"] = tb
        S[f"{key}|{fold}|vb"], S[f"{key}|{fold}|va"] = sv[yv == 0], sv[yv == 1]
        S[f"{key}|{fold}|zs"] = cast(score(Xt))
        print(f"  {key:18s} {fold:26s} {time.time() - t0:6.0f}s", flush=True)
    return S


def _compute_optiforest(source):
    """OptIForest cham rat cham (thuan Python, ~100 flow/s). Diem benign train / val lay tu file nguoi train da
    luu (cham lai 8,000 flow trong do khop toi 1e-16); chi cham phan attack, moi flow khac nhau cham mot lan."""
    ck = scorers.ROOT[source] / "optiforest" / "checkpoint"
    tb = np.load(ck / "train_benign_scores.npy").astype(np.float32)
    vb = np.load(ck / "val_benign_scores.npy").astype(np.float32)
    parts, ref = {}, None
    for fold in data.folds():
        Xv, yv, Xt = data.fold_data(fold)
        h = hashlib.md5(np.ascontiguousarray(Xv[yv == 0]).tobytes()).hexdigest()
        assert ref in (None, h) and (yv == 0).sum() == len(vb), "benign val phai giong nhau o moi fold"
        ref, parts[fold] = h, (Xv[yv == 1], Xt)
    X_all = np.concatenate([x for fold in parts for x in parts[fold]])
    uniq, inv = np.unique(X_all, axis=0, return_inverse=True)
    print(f"  optiforest: {len(X_all):,} attack flows, {len(uniq):,} distinct -> scoring", flush=True)
    s = np.asarray(scorers.loader("optiforest", source, None)(uniq), dtype=np.float32)[inv.ravel()]
    S, i = {}, 0
    for fold, (Xva, Xt) in parts.items():
        S[f"optiforest|{fold}|tb"], S[f"optiforest|{fold}|vb"] = tb, vb
        S[f"optiforest|{fold}|va"], i = s[i:i + len(Xva)], i + len(Xva)
        S[f"optiforest|{fold}|zs"], i = s[i:i + len(Xt)], i + len(Xt)
    return S


def get(key, source, recompute=False):
    """Diem cua model: doc cache neu co, neu khong (hoac recompute=True) thi cham lai va luu cache."""
    S = None if recompute else load(key, source)
    if S is None:
        S = compute(key, source)
        save(key, source, S)
    return S
