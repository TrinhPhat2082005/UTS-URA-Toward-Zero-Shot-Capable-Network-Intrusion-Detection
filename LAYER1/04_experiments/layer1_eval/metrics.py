"""Nguong theo ngan sach FPR va cac chi so cua bang RQ1/RQ2.

Nguong cua mot model o fold f:  q_f = Q_{1-alpha}(diem tren 560K benign train), np.quantile(method="higher").
  rule "perfold":   fold f dung q_f cua chinh no.
  rule "meanfold":  moi fold dung chung tau = trung binh 14 gia tri q_f (quy tac cua bang tren slide).
Flow la alert neu diem >= nguong. Val va test khong bao gio tham gia dat nguong.
S la dict diem cua scores.get(): S["<key>|<fold>|tb|vb|va|zs"].
"""
import numpy as np


def folds_of(S):
    return sorted({k.split("|")[1] for k in S})


def fold_quantile(tb, alpha):
    return float(np.quantile(tb, 1 - alpha, method="higher"))


def thresholds(S, key, alpha, rule):
    """{fold: (nguong dung o fold, phan vi rieng cua fold)}."""
    q = {f: fold_quantile(S[f"{key}|{f}|tb"], alpha) for f in folds_of(S)}
    if rule == "perfold":
        return {f: (q[f], q[f]) for f in q}
    t = float(np.mean(list(q.values())))
    return {f: (t, q[f]) for f in q}


def fold_metrics(vb, va, zs, t):
    """Mot fold o nguong t: known-attack tren val_binary (benign + 13 loai da biet), ZS-Acc tren test_zeroshot."""
    tp, fp = int((va >= t).sum()), int((vb >= t).sum())
    fn, tn = len(va) - tp, len(vb) - fp
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(va)
    return {"acc": (tp + tn) / (tp + tn + fp + fn), "precision": pre, "recall": rec,
            "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0.0, "fpr": fp / len(vb),
            "zs_acc": float((zs >= t).mean())}


def evaluate(S, key, alpha, rule):
    """(trung binh 14 fold cua moi chi so, chi so tung fold, nguong tung fold)."""
    thr = thresholds(S, key, alpha, rule)
    per_fold = {f: fold_metrics(S[f"{key}|{f}|vb"], S[f"{key}|{f}|va"], S[f"{key}|{f}|zs"], t)
                for f, (t, _) in thr.items()}
    first = next(iter(per_fold.values()))
    macro = {k: float(np.mean([m[k] for m in per_fold.values()])) for k in first}
    return macro, per_fold, thr


def pauc(S, key, fmax=0.03, n=301):
    """Dien tich duoi duong ZS-Acc theo FPR thuc te tren benign val, FPR trong [0, fmax], chia cho fmax.
    Khong phu thuoc nguong; muc ngau nhien = fmax / 2 = 0.015."""
    fs, curve, folds = np.linspace(0, fmax, n), np.zeros(n), folds_of(S)
    for f in folds:
        vb, zs = np.sort(S[f"{key}|{f}|vb"]), np.sort(S[f"{key}|{f}|zs"])
        ts = np.quantile(vb, 1 - fs, method="higher")
        realized = (len(vb) - np.searchsorted(vb, ts, side="left")) / len(vb)
        zacc = (len(zs) - np.searchsorted(zs, ts, side="left")) / len(zs)
        order = np.argsort(realized)
        curve += np.interp(fs, realized[order], zacc[order])
    return float(np.trapezoid(curve / len(folds), fs) / fmax)
