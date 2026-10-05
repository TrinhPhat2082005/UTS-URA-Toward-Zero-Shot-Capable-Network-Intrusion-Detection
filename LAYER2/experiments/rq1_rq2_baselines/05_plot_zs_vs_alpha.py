"""Step 5 - figure: zero-shot accuracy against the false-positive budget alpha.

Reads the long-format table written by
    python 03_baseline_tables.py --rule meanfold --alphas 0.001 0.002 0.004 0.008 0.01 0.015 0.02 0.03
and draws one line per model (macro ZS-Acc over the 14 folds), with each model's pAUC[0, 3%]
in the legend. --models-from picks the checkpoints: the baseline study's (default) or the paper's.

Usage:  python 05_plot_zs_vs_alpha.py [--models key ...] [--rule meanfold|perfold] [--grid <tag>] [--csv ...]
        e.g. --grid alpha0.1_alpha0.2_alpha0.3_alpha0.4_alpha0.5 (after 03 --alphas 0.001 ... 0.005)
Output: OUT_BASE/zs_vs_alpha[_meanfold].pdf and .png
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
DEFAULT_MODELS = ["xgboost_only", "clad", "vae_xgboost", "ae_xgboost_mono", "svdd_xgboost_mono"]   # roles
_B, _P = "baseline-study", "paper"
PICK = {   # role -> (key, source) in the table; --models-from chooses whose checkpoints play the roles
    "baseline": {"xgboost_only": ("xgboost_only", _P), "clad": ("clad", _B), "vae_xgboost": ("vae_xgboost", _B),
                 "ae_xgboost_mono": ("ae_xgboost", _B), "svdd_xgboost_mono": ("svdd_xgboost", _B)},
    "paper": {"xgboost_only": ("xgboost_only", _P), "clad": ("clad", _P), "vae_xgboost": ("vae_xgboost", _B),
              "ae_xgboost_mono": ("ae_xgboost_mono", _P), "svdd_xgboost_mono": ("svdd_xgboost_mono", _P)},
}
NAMES = {"ae_xgboost_mono": "AE-XGBoost (ours)", "svdd_xgboost_mono": "SVDD-XGBoost (ours)"}
STYLE = {   # colour, line width, marker; the paper's two variants drawn heavier
    "xgboost_only": ("#9E9E9E", 1.5, "o"), "clad": ("#E08E2B", 1.5, "s"), "vae_xgboost": ("#7B5EA7", 1.5, "D"),
    "ae_xgboost_mono": ("#1F497D", 2.4, "o"), "svdd_xgboost_mono": ("#2E75B6", 2.4, "^"),
}


def main(models, csv_path, sfx="", models_from="baseline"):
    with open(csv_path, encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    rows = []
    for role in models:
        key, src = PICK[models_from].get(role, (role, None))
        rows += [dict(r, key=role, model=NAMES.get(role, r["model"])) for r in table
                 if r["key"] == key and (r["source"] == src if src else not r["retrain_of"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for key in models:
        pts = sorted((float(r["alpha"]), float(r["zs_acc"]), float(r["pauc"]), r["model"])
                     for r in rows if r["key"] == key)
        if not pts:
            print(f"[skip] {key}: not in {csv_path.name}")
            continue
        col, lw, mk = STYLE.get(key, ("#555555", 1.5, "o"))
        ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], color=col, lw=lw, marker=mk, ms=5,
                label=f"{pts[0][3]} (pAUC$_{{[0,3\%]}}$ {pts[0][2]:.3f})")
    alphas = sorted({100 * float(r["alpha"]) for r in rows})
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}%" for a in alphas])
    ax.minorticks_off()
    how = "mean of the 14 fold thresholds" if sfx.startswith("_meanfold") else "per fold"
    ax.set_xlabel(f"False-positive budget α (benign-train quantile, {how})")
    ax.set_ylabel("Zero-shot accuracy (macro, 14 folds)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"zs_vs_alpha{sfx}.{ext}", dpi=300)
    print(f"-> {OUT / f'zs_vs_alpha{sfx}.pdf'}, .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--models-from", choices=sorted(PICK), default="baseline",
                    help="baseline: the baseline study's checkpoints; paper: the paper's own (default baseline)")
    ap.add_argument("--rule", choices=["meanfold", "perfold"], default="meanfold",
                    help="threshold rule of the table to plot (03_baseline_tables.py --rule)")
    ap.add_argument("--grid", default=GRID_TAG, help=f"alpha grid tag of the table (default {GRID_TAG})")
    ap.add_argument("--csv", type=Path, help="table to plot (default: table_<grid>[_<rule>].csv)")
    a = ap.parse_args()
    sfx = '' if a.rule == 'perfold' else '_' + a.rule
    out_sfx = sfx + ('' if a.grid == GRID_TAG else '_' + a.grid.replace('alpha', '').replace('_', '-'))
    out_sfx += "" if a.models_from == "baseline" else "_paper-models"
    main(a.models, a.csv or OUT / f"table_{a.grid}{sfx}.csv", out_sfx, a.models_from)
