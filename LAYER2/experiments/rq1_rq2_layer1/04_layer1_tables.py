"""Sinh bang ket qua Layer 1 cho paper (6 model, cung 1 quy trinh nguong).
Nguong: tau_r = Q_{1-alpha}(diem tren BENIGN CUA TRAIN) rieng tung round; val/test khong tham gia.
pAUC: duong ZS-Acc theo FPR-val (moi round nguong rieng tai tung muc FPR), tich phan tren [0, 3%],
chuan hoa ve [0,1] - la tom tat khong phu thuoc nguong, khong dung de chon diem van hanh."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  (experiments/paths.py)
import json
from pathlib import Path
import numpy as np

HERE = paths.ensure(paths.OUT_L1)
ART = paths.L1_ART
S = np.load(HERE / "sweep_all_scores.npz")
VARS = ["xgboost_only", "ae_only", "svdd_only", "clad", "ae_xgboost_mono", "svdd_xgboost_mono"]
NAME = {"xgboost_only": "XGBoost-only", "ae_only": "AE-only", "svdd_only": "SVDD-only", "clad": "CLAD",
        "ae_xgboost_mono": "AE-XGBoost (ours)", "svdd_xgboost_mono": "SVDD-XGBoost (ours)"}
LABELS = sorted({k.split("|")[1] for k in S.files})
ALPHAS = [0.001, 0.002, 0.004, 0.008, 0.010, 0.015, 0.020, 0.030]
out = []
P = out.append


def thr(v, l, a):
    return np.quantile(S[f"{v}|{l}|tb"], 1 - a, method="higher")


def fold(v, l, t):
    vb, va, zs = S[f"{v}|{l}|vb"], S[f"{v}|{l}|va"], S[f"{v}|{l}|zs"]
    tp, fp = (va >= t).sum(), (vb >= t).sum()
    fn, tn = len(va) - tp, len(vb) - fp
    rec = tp / len(va); pre = tp / (tp + fp) if tp + fp else 0.0
    return dict(zs=(zs >= t).mean(), acc=(tp + tn) / (tp + tn + fp + fn), pre=pre, rec=rec,
                f1=2 * pre * rec / (pre + rec) if pre + rec else 0.0, fpr=fp / len(vb))


def macro(v, a):
    rows = [fold(v, l, thr(v, l, a)) for l in LABELS]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def pauc(v, fmax=0.03, n=301):
    fs = np.linspace(0, fmax, n)
    curve = np.zeros(n)
    for l in LABELS:
        vb, zs = np.sort(S[f"{v}|{l}|vb"]), np.sort(S[f"{v}|{l}|zs"])
        ts = np.quantile(vb, 1 - fs, method="higher")
        realized = (len(vb) - np.searchsorted(vb, ts, side="left")) / len(vb)
        zacc = (len(zs) - np.searchsorted(zs, ts, side="left")) / len(zs)
        order = np.argsort(realized)
        curve += np.interp(fs, realized[order], zacc[order])
    curve /= len(LABELS)
    return float(np.trapezoid(curve, fs) / fmax)


P("## T-A. Zero-shot accuracy (macro over 14 rounds) vs. false-positive budget alpha\n")
P("| alpha | " + " | ".join(NAME[v] for v in VARS) + " |")
P("|---|" + "---|" * len(VARS))
for a in ALPHAS:
    cells = []
    for v in VARS:
        m = macro(v, a)
        cells.append(f"{m['zs']:.4f} ({100*m['fpr']:.2f}%)")
    P(f"| {100*a:.1f}% | " + " | ".join(cells) + " |")
P("| pAUC[0,3%] | " + " | ".join(f"{pauc(v):.4f}" for v in VARS) + " |")
P("\nCell = ZS-Acc (realized FPR on val_binary).\n")

for a in (0.004, 0.015):
    P(f"## T-B. Known-attack detection on val_binary at alpha = {100*a:.1f}% (macro over 14 rounds)\n")
    P("| Model | Acc | Precision | Recall | F1 | FPR | ZS-Acc |")
    P("|---|---|---|---|---|---|---|")
    for v in VARS:
        m = macro(v, a)
        P(f"| {NAME[v]} | {m['acc']:.4f} | {m['pre']:.4f} | {m['rec']:.4f} | {m['f1']:.4f} | {m['fpr']:.4f} | {m['zs']:.4f} |")
    P("")

P("## T-C. Reference: F1-max threshold tuned on train (supervised models only)\n")
P("| Model | tau | Acc | Precision | Recall | F1 | FPR | ZS-Acc |")
P("|---|---|---|---|---|---|---|---|")
for v in ["xgboost_only", "clad", "ae_xgboost_mono", "svdd_xgboost_mono"]:
    if v == "clad":
        t = float(np.mean([json.loads((ART / "clad" / l / "checkpoint" / "meta.json").read_text())["threshold"] for l in LABELS]))
    else:
        t = json.loads((ART / v / "metrics.json").read_text())["threshold_mean_used_for_eval"]
    rows = [fold(v, l, t) for l in LABELS]
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    P(f"| {NAME[v]} | {t:.4f} | {m['acc']:.4f} | {m['pre']:.4f} | {m['rec']:.4f} | {m['f1']:.4f} | {m['fpr']:.4f} | {m['zs']:.4f} |")
P("")

for a in (0.004, 0.015):
    P(f"## T-D. Per-type zero-shot accuracy at alpha = {100*a:.1f}%\n")
    P("| Withheld type | n test | " + " | ".join(NAME[v] for v in VARS) + " |")
    P("|---|---|" + "---|" * len(VARS))
    for l in LABELS:
        n = len(S[f"xgboost_only|{l}|zs"])
        P(f"| {l} | {n} | " + " | ".join(f"{(S[f'{v}|{l}|zs'] >= thr(v, l, a)).mean():.4f}" for v in VARS) + " |")
    P("")

txt = "\n".join(out)
(HERE / "layer1_tables.md").write_text(txt, encoding="utf-8")
print(txt)
