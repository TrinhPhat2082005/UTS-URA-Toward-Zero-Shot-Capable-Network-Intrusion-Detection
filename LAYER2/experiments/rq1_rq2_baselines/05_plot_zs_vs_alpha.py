"""Step 5 - figure: zero-shot accuracy against the false-positive budget alpha.

Reads the long-format table written by
    python 03_baseline_tables.py --alphas 0.001 0.002 0.004 0.008 0.01 0.015 0.02 0.03
and draws one line per model (macro ZS-Acc over the 14 folds), with each model's pAUC[0, 3%]
in the legend. Re-trained copies of the paper's models are not drawn.

Usage:  python 05_plot_zs_vs_alpha.py [--models key ...] [--csv table_<tag>.csv]
Output: OUT_BASE/zs_vs_alpha.pdf and .png
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

OUT = paths.ensure(paths.OUT_BASE)
GRID_TAG = "alpha0.1_alpha0.2_alpha0.4_alpha0.8_alpha1_alpha1.5_alpha2_alpha3"
DEFAULT_MODELS = ["xgboost_only", "clad", "vae_xgboost", "ae_xgboost_mono", "svdd_xgboost_mono"]
STYLE = {   # colour, line width, marker; the paper's two variants drawn heavier
    "xgboost_only": ("#9E9E9E", 1.5, "o"), "clad": ("#E08E2B", 1.5, "s"), "vae_xgboost": ("#7B5EA7", 1.5, "D"),
    "ae_xgboost_mono": ("#1F497D", 2.4, "o"), "svdd_xgboost_mono": ("#2E75B6", 2.4, "^"),
}


def main(models, csv_path):
    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["key"] in models and not r["retrain_of"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for key in models:
        pts = sorted((float(r["alpha"]), float(r["zs_acc"]), float(r["pauc"]), r["model"])
                     for r in rows if r["key"] == key)
        if not pts:
            print(f"[skip] {key}: not in {csv_path.name}")
            continue
        col, lw, mk = STYLE.get(key, ("#555555", 1.5, "o"))
        ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], color=col, lw=lw, marker=mk, ms=5,
                label=f"{pts[0][3]} (pAUC {pts[0][2]:.3f})")
    alphas = sorted({100 * float(r["alpha"]) for r in rows})
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}%" for a in alphas])
    ax.minorticks_off()
    ax.set_xlabel("False-positive budget α (benign-train quantile, per fold)")
    ax.set_ylabel("Zero-shot accuracy (macro, 14 folds)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"zs_vs_alpha.{ext}", dpi=300)
    print(f"-> {OUT / 'zs_vs_alpha.pdf'}, .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--csv", type=Path, default=OUT / f"table_{GRID_TAG}.csv")
    a = ap.parse_args()
    main(a.models, a.csv)
