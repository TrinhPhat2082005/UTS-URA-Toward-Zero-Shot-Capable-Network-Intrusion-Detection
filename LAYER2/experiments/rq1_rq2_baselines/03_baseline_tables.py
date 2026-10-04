"""Step 3 - every Layer 1 model at the same false-positive budget.

Threshold rule, identical for all models and to rq1_rq2_layer1/04_layer1_tables.py:
    tau = Q_{1-alpha}(scores on the 560K benign training flows), per fold,
    np.quantile(..., method="higher"), predict attack if score >= tau.
No validation or test data enters the threshold. alpha = 0.5% (quantile 0.995) is the
budget of the baseline study; alpha = 0.4% is the paper's operating point.

Rows: the paper's six models (rq1_rq2_layer1/sweep_all_scores.npz) and every scored
baseline of step 1. A baseline that re-trains one of the paper's six models is listed
under "re-training variance", not as a separate method.

Outputs: OUT_BASE/baseline_tables.md, OUT_BASE/baseline_table.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from baselines import REGISTRY  # noqa: E402

OUT = paths.ensure(paths.OUT_BASE)
ALPHAS_MAIN = (0.005, 0.004)
ALPHAS_SWEEP = (0.001, 0.002, 0.004, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030)
PAPER = {"xgboost_only": ("XGBoost-only", "Supervised"), "ae_only": ("AE-only", "One-class"),
         "svdd_only": ("Mixture Deep SVDD", "One-class"), "clad": ("CLAD", "Contrastive"),
         "ae_xgboost_mono": ("AE-XGBoost (ours)", "Hybrid"),
         "svdd_xgboost_mono": ("SVDD-XGBoost (ours)", "Hybrid")}
RETRAIN_OF = {"xgboost_only": "xgboost_only", "ae_only": "ae_only", "svdd_only": "svdd_only",
              "clad": "clad", "ae_xgboost": "ae_xgboost_mono", "svdd_xgboost": "svdd_xgboost_mono"}

S, rows = {}, []
for k, v in np.load(paths.OUT_L1 / "sweep_all_scores.npz").items():
    S["paper:" + k] = v
for name, (lab, fam) in PAPER.items():
    rows.append(("paper", name, lab, fam))
for name in REGISTRY:
    f = OUT / f"scores_{name}.npz"
    if f.exists():
        for k, v in np.load(f).items():
            S["base:" + k] = v
        rows.append(("base", name, REGISTRY[name].label, REGISTRY[name].family))
FOLDS = sorted({k.split("|")[1] for k in S})
NEW = [r for r in rows if r[0] == "paper" or r[1] not in RETRAIN_OF]


def same_scores(name):
    """True if a baseline-study checkpoint scores exactly like the paper's model (same file)."""
    return all(np.allclose(S[f"base:{name}|{f}|{p}"], S[f"paper:{RETRAIN_OF[name]}|{f}|{p}"], rtol=0, atol=1e-6)
               for f in FOLDS for p in ("tb", "vb", "va", "zs"))


IDENTICAL = [r for r in rows if r[0] == "base" and r[1] in RETRAIN_OF and same_scores(r[1])]
RETRAIN = [r for r in rows if r[0] == "base" and r[1] in RETRAIN_OF and r not in IDENTICAL]


def get(src, name, fold, part):
    return S[f"{src}:{name}|{fold}|{part}"]


def thr(src, name, fold, a):
    return np.quantile(get(src, name, fold, "tb"), 1 - a, method="higher")


def fold_metrics(src, name, fold, t):
    vb, va, zs = (get(src, name, fold, p) for p in ("vb", "va", "zs"))
    tp, fp = int((va >= t).sum()), int((vb >= t).sum())
    fn, tn = len(va) - tp, len(vb) - fp
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(va)
    return {"acc": (tp + tn) / (tp + tn + fp + fn), "precision": pre, "recall": rec,
            "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0.0, "fpr": fp / len(vb),
            "zs_acc": float((zs >= t).mean())}


def macro(src, name, a):
    rs = [fold_metrics(src, name, f, thr(src, name, f, a)) for f in FOLDS]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def pauc(src, name, fmax=0.03, n=301):
    """Threshold-free summary: area under macro ZS-Acc vs realized val FPR on [0, fmax]."""
    fs, curve = np.linspace(0, fmax, n), np.zeros(n)
    for f in FOLDS:
        vb, zs = np.sort(get(src, name, f, "vb")), np.sort(get(src, name, f, "zs"))
        ts = np.quantile(vb, 1 - fs, method="higher")
        realized = (len(vb) - np.searchsorted(vb, ts, side="left")) / len(vb)
        zacc = (len(zs) - np.searchsorted(zs, ts, side="left")) / len(zs)
        order = np.argsort(realized)
        curve += np.interp(fs, realized[order], zacc[order])
    return float(np.trapezoid(curve / len(FOLDS), fs) / fmax)


def tag(src, name):
    return "" if src == "paper" or name not in RETRAIN_OF else " (retrain)"


def main():
    md, csv_rows = [], []
    for a in ALPHAS_MAIN:
        md.append(f"## Known-attack detection and zero-shot accuracy at alpha = {100 * a:.1f}% "
                  f"(quantile {1 - a:.3f} of benign-train scores, per fold; macro over {len(FOLDS)} folds)\n")
        for title, group in (("Methods", NEW),
                             ("Re-training variance (baseline-study run of the paper's models)", RETRAIN)):
            if not group:
                continue
            md.append(f"### {title}\n")
            md.append("| Model | Family | Acc | Precision | Recall | F1 | FPR | ZS-Acc | pAUC[0,3%] |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for src, name, lab, fam in group:
                r, p = macro(src, name, a), pauc(src, name)
                md.append(f"| {lab} | {fam} | {r['acc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                          f"{r['f1']:.4f} | {100 * r['fpr']:.2f}% | {r['zs_acc']:.4f} | {p:.4f} |")
                csv_rows.append({"alpha": a, "key": name, "model": lab,
                                 "source": "paper" if src == "paper" else "baseline-study",
                                 "retrain_of": RETRAIN_OF.get(name, "") if src == "base" else "",
                                 "family": fam, **{k: round(v, 6) for k, v in r.items()},
                                 "pauc": round(p, 6)})
            md.append("")
        if IDENTICAL:
            md.append("Identical checkpoints (scores equal to the paper's model on every fold, so not a "
                      "re-training): " + ", ".join(lab for _, _, lab, _ in IDENTICAL) + ".\n")
    md.append("## ZS-Acc vs alpha (realized val FPR in brackets)\n")
    md.append("| Model | " + " | ".join(f"{100 * a:.1f}%" for a in ALPHAS_SWEEP) + " |")
    md.append("|---|" + "---|" * len(ALPHAS_SWEEP))
    for src, name, lab, _ in NEW + RETRAIN:
        cells = [f"{r['zs_acc']:.3f} ({100 * r['fpr']:.2f}%)" for r in (macro(src, name, a) for a in ALPHAS_SWEEP)]
        md.append(f"| {lab}{tag(src, name)} | " + " | ".join(cells) + " |")
    md.append("\n## Per-type zero-shot accuracy at alpha = 0.5%\n")
    group = NEW + RETRAIN
    md.append("| Withheld type | " + " | ".join(lab + tag(s, n) for s, n, lab, _ in group) + " |")
    md.append("|---|" + "---|" * len(group))
    for f in FOLDS:
        md.append(f"| {f} | " + " | ".join(f"{(get(s, n, f, 'zs') >= thr(s, n, f, 0.005)).mean():.3f}"
                                           for s, n, _, _ in group) + " |")
    (OUT / "baseline_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT / "baseline_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
